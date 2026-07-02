import gc
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from density_sampling import (
    analyze_sampling_effectiveness,
    generate_collocation_points,
    report_collocation_coverage_gaps,
)


class SineFirst(nn.Module):
    """First layer of a SIREN network: linear transform then sine with scale omega_0."""

    def __init__(self, in_features, out_features, omega_0=30):
        super(SineFirst, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class Sine(nn.Module):
    """Sine activation for subsequent SIREN layers (multiplier = 1 by default)."""

    def __init__(self, omega_0=1.0):
        super(Sine, self).__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for fluid flow simulation.
    
    The PINN takes in features (x, z, t) and outputs (u, v, p), where:
    - u, v: Velocity components in the x and z directions
    - p: Pressure
    
    It also defines a trainable log-viscosity parameter.
    """
    
    def __init__(self, input_dim=3, output_dim=3, hidden_layers=5, hidden_size=64,
                 activation='sine', omega_0=30):
        """
        Initialize the PINN model.

        Parameters:
            input_dim: Number of input features (x, z, t)
            output_dim: Number of output features (u, v, p)
            hidden_layers: Number of hidden layers
            hidden_size: Number of neurons per hidden layer
            activation: Activation function ('sine' or 'tanh')
            omega_0: Scaling factor for SIREN activations
        """
        super(PINN, self).__init__()

        self.activation = activation
        self.omega_0 = omega_0

        effective_input_dim = input_dim

        # Build network layers
        layers = []
        
        if activation == 'sine':
            # SIREN-style network
            layers.append(SineFirst(effective_input_dim, hidden_size, omega_0))
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(Sine(omega_0=1.0))
            layers.append(nn.Linear(hidden_size, output_dim))
        else:
            # Tanh activation network
            layers.append(nn.Linear(effective_input_dim, hidden_size))
            layers.append(nn.Tanh())
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_size, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Trainable log-viscosity parameter
        self.log_mu = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        
        # Trainable scaling parameters
        self.output_scale = nn.Parameter(torch.tensor([1.0]))
        self.velocity_scale = nn.Parameter(torch.tensor([1.0]))
        
        # DataNormalizer attached later during training
        self.normalizer = None

        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights based on activation function."""
        if self.activation == 'sine':
            # SIREN initialization
            first_linear_found = False
            for m in self.network.modules():
                if isinstance(m, nn.Linear):
                    if not first_linear_found:
                        # first layer, narrow initialization
                        nn.init.uniform_(m.weight, -1.0 / m.in_features, 1.0 / m.in_features)
                        first_linear_found = True
                    else:
                        # subsequent layers, wider initialization
                        bound = np.sqrt(6 / m.in_features) / self.omega_0
                        nn.init.uniform_(m.weight, -bound, bound)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            # Xavier initialization for tanh networks
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
            x: Input tensor of shape (N, 3) corresponding to (x, z, t)
            
        Returns:
            u: Velocity component along x
            v: Velocity component along z
            p: Pressure
        """
        out = self.network(x)
        
        # network predicts values in normalised star space
        u_star = out[:, 0:1]
        v_star = out[:, 1:2]
        p_star = out[:, 2:3]
        
        # During training we keep everything in star space
        if self.training or not hasattr(self, "normalizer") or self.normalizer is None:
            return u_star, v_star, p_star
        
        # Inference phase - convert back to physical units
        u_phys = self.normalizer.unstar(u_star, 'u')
        v_phys = self.normalizer.unstar(v_star, 'v')
        p_phys = self.normalizer.unstar(p_star, 'p')
        return u_phys, v_phys, p_phys
    
    def get_mu(self):
        """Get the dynamic viscosity from the trainable parameter."""
        return torch.exp(self.log_mu)


class DataNormalizer:
    """Utility to compute and apply z-score normalisation for PINN inputs/outputs."""

    def __init__(self, stats: dict):
        # stats maps key -> (mean, std)
        self.stats = stats

    # Factory helpers
    @classmethod
    def from_dataset(cls, x_data, vel_data, default_p_stats=(0.0, 1.0)):
        """Compute per-component mean/std from the provided raw training data.

        Parameters
        ----------
        x_data : array like, shape (N,3)
            Raw coordinates (x,z,t) in physical units (torch or numpy).
        vel_data : array like, shape (N,2)
            Raw measured velocities (u,v) in physical units (torch or numpy).
        default_p_stats : tuple
            (mean,std) pair for pressure when no pressure data is provided.
        """
        # to numpy for stats, torch or ndarray
        def _to_numpy(arr):
            if torch.is_tensor(arr):
                return arr.detach().cpu().numpy()
            return np.asarray(arr)

        x_np = _to_numpy(x_data)
        v_np = _to_numpy(vel_data)

        mu_x,  sigma_x = x_np[:, 0].mean(), x_np[:, 0].std() + 1e-12
        mu_z,  sigma_z = x_np[:, 1].mean(), x_np[:, 1].std() + 1e-12
        mu_t,  sigma_t = x_np[:, 2].mean(), x_np[:, 2].std() + 1e-12
        mu_u,  sigma_u = v_np[:, 0].mean(), v_np[:, 0].std() + 1e-12
        mu_v,  sigma_v = v_np[:, 1].mean(), v_np[:, 1].std() + 1e-12
        mu_p,  sigma_p = default_p_stats

        stats = {
            'x': (mu_x, sigma_x),
            'z': (mu_z, sigma_z),
            't': (mu_t, sigma_t),
            'u': (mu_u, sigma_u),
            'v': (mu_v, sigma_v),
            'p': (mu_p, sigma_p),
        }
        return cls(stats)

    # Star / unstar helpers
    def _get_torch_pair(self, arr, key):
        mu, sd = self.stats[key]
        mu_t = torch.tensor(mu, dtype=arr.dtype, device=arr.device)
        sd_t = torch.tensor(sd, dtype=arr.dtype, device=arr.device)
        return mu_t, sd_t

    def star(self, arr, key):
        """Convert raw physical quantity to normalised (star) quantity."""
        if torch.is_tensor(arr):
            mu_t, sd_t = self._get_torch_pair(arr, key)
            return (arr - mu_t) / sd_t
        else:
            mu, sd = self.stats[key]
            return (arr - mu) / sd

    def unstar(self, arr_s, key):
        """Convert normalised (star) quantity back to physical units."""
        if torch.is_tensor(arr_s):
            mu_t, sd_t = self._get_torch_pair(arr_s, key)
            return arr_s * sd_t + mu_t
        else:
            mu, sd = self.stats[key]
            return arr_s * sd + mu


def force(x, z):
    """
    Define the forcing function f(x,z).
    
    Parameters:
        x, z: Tensors of shape (N,1) corresponding to spatial coordinates
        
    Returns:
        f_x, f_z: Force components (currently zero)
    """
    f_x = torch.zeros_like(x)
    f_z = torch.zeros_like(z)
    return f_x, f_z


def physics_loss_divergence_only(model, x_colloc_star, normalizer):
    """
    Lightweight physics loss enforcing only incompressibility: div(u) = 0.

    This avoids the momentum (Laplacian) term that over-smooths the velocity
    field, while still constraining mass conservation.  Only first-order
    spatial derivatives are needed, so the computational graph is smaller and
    gradients are more stable.
    """
    x_colloc_star = x_colloc_star.detach().requires_grad_(True)

    u_s, v_s, _p_s = model(x_colloc_star)

    sigma_x = normalizer.stats['x'][1]
    sigma_z = normalizer.stats['z'][1]

    grads_u = torch.autograd.grad(
        u_s, x_colloc_star,
        grad_outputs=torch.ones_like(u_s),
        retain_graph=True, create_graph=True,
    )[0]
    grads_v = torch.autograd.grad(
        v_s, x_colloc_star,
        grad_outputs=torch.ones_like(v_s),
        retain_graph=True, create_graph=True,
    )[0]

    u_x = grads_u[:, 0:1] / sigma_x
    v_z = grads_v[:, 1:2] / sigma_z

    e_div = u_x + v_z
    return torch.mean(e_div ** 2)


def physics_loss(
    model,
    x_colloc_star,
    normalizer,
    use_optimized=True,
):
    """
    Steady Stokes physics loss evaluated in normalised (star) coordinates with
    proper chain-rule scaling back to physical space.

    Enforces the steady (Stokes-like) momentum balance and incompressibility:
        -∇p + μ∇²u + f = 0
        ∇·u = 0

    Parameters
    ----------
    model : PINN
        The neural network (expects star inputs).
    x_colloc_star : torch.Tensor, shape (N_eq,3)
        Collocation points already normalised.
    normalizer : DataNormalizer
        Object holding the statistics for unstar → star conversion.
    use_optimized : bool
        Whether to use the optimised gradient computation as before.
    """
    x_colloc_star.requires_grad_(True)

    # forward pass in star space
    u_s, v_s, p_s = model(x_colloc_star)
    mu = model.get_mu()

    # Helper scalings
    sigma_x = normalizer.stats['x'][1]
    sigma_z = normalizer.stats['z'][1]
    inv_sigma_x = 1.0 / sigma_x
    inv_sigma_z = 1.0 / sigma_z
    inv_sigma_x2 = inv_sigma_x ** 2
    inv_sigma_z2 = inv_sigma_z ** 2

    if use_optimized:
        grads_all = []
        for output in [u_s, v_s, p_s]:
            grad = torch.autograd.grad(
                output, x_colloc_star,
                grad_outputs=torch.ones_like(output),
                retain_graph=True, create_graph=True
            )[0]
            grads_all.append(grad)

        # First derivatives in star space
        u_xs, u_zs = grads_all[0][:, 0:1], grads_all[0][:, 1:2]
        v_xs, v_zs = grads_all[1][:, 0:1], grads_all[1][:, 1:2]
        p_xs, p_zs = grads_all[2][:, 0:1], grads_all[2][:, 1:2]

        # Map to physical derivatives using chain rule
        u_x = u_xs * inv_sigma_x
        u_z = u_zs * inv_sigma_z
        v_x = v_xs * inv_sigma_x
        v_z = v_zs * inv_sigma_z
        p_x = p_xs * inv_sigma_x
        p_z = p_zs * inv_sigma_z

        # second derivatives in star space
        second_derivs = []
        first_derivs_list = [u_xs, u_zs, v_xs, v_zs]
        coord_indices = [0, 1, 0, 1]
        for first_deriv, coord_idx in zip(first_derivs_list, coord_indices):
            second_deriv = torch.autograd.grad(
                first_deriv, x_colloc_star,
                grad_outputs=torch.ones_like(first_deriv),
                retain_graph=True, create_graph=True
            )[0][:, coord_idx:coord_idx + 1]
            second_derivs.append(second_deriv)
        u_xxs, u_zzs, v_xxs, v_zzs = second_derivs

        # Map Laplacians to physical space
        u_xx = u_xxs * inv_sigma_x2
        u_zz = u_zzs * inv_sigma_z2
        v_xx = v_xxs * inv_sigma_x2
        v_zz = v_zzs * inv_sigma_z2
    else:
        raise RuntimeError("Sequential physics_loss not updated for normalisation - use use_optimized=True")

    laplacian_u = u_xx + u_zz
    laplacian_v = v_xx + v_zz

    # Forcing remains zero in physical space
    f_x, f_z = force(x_colloc_star[:, 0:1], x_colloc_star[:, 1:2])  # already zero

    # Steady momentum and continuity residuals in physical space
    #   -∇p + μ∇²u + f = 0
    #   ∇·u = 0
    e_mom_x = -p_x + mu * laplacian_u + f_x
    e_mom_z = -p_z + mu * laplacian_v + f_z
    e_div   = u_x + v_z

    loss_mom = torch.mean(e_mom_x ** 2 + e_mom_z ** 2)
    loss_div = torch.mean(e_div ** 2)
    loss = loss_mom + loss_div

    return loss


def data_loss(model, x_data, measured_vel, batch_size=100000,
              loss_fn="mse", huber_delta=1.0):
    """
    Compute the data loss measuring how well the model matches measured velocities.
    
    Parameters:
        model: The PINN model
        x_data: Data point coordinates
        measured_vel: Measured velocity values
        batch_size: Batch size for processing
        loss_fn: Loss function type. Options:
            - "mse": Mean Squared Error (default, original behavior)
            - "huber": Huber loss (smooth L1), more robust to outliers.
              Quadratic for |error| < delta, linear for |error| >= delta.
              Recommended for in vivo data where velocity targets are noisy.
        huber_delta: Threshold for Huber loss transition from quadratic to
            linear regime. Smaller delta = more robust to outliers but slower
            convergence on clean data. Default 1.0 (in z-scored velocity units).
            Typical values to try: 0.5, 1.0, 2.0.
        
    Returns:
        loss_data: Data loss scalar with gradients
    """
    model.train()  # keep train mode for gradients
    total_loss = 0.0
    total_valid_points = 0

    # Build the per-element loss function
    if loss_fn == "huber":
        _huber = torch.nn.HuberLoss(reduction='mean', delta=huber_delta)
    
    # Process in batches
    for i in range(0, len(x_data), batch_size):
        end_idx = min(i + batch_size, len(x_data))
        batch_x = x_data[i:end_idx]
        batch_vel = measured_vel[i:end_idx]
        
        # Ensure batch_x has correct shape [N, 3] (not [N, 3, 1] or similar)
        if batch_x.dim() > 2:
            batch_x = batch_x.squeeze(-1) if batch_x.shape[-1] == 1 else batch_x.view(batch_x.shape[0], -1)
        # Ensure batch_x is [N, 3]
        if batch_x.shape[1] != 3:
            # Try to reshape if it's [N, 3, 1]
            if batch_x.dim() == 3 and batch_x.shape[2] == 1:
                batch_x = batch_x.squeeze(2)
            else:
                # Reshape to [N, 3] by flattening extra dimensions
                batch_x = batch_x.view(batch_x.shape[0], -1)
                if batch_x.shape[1] != 3:
                    # If still wrong, take first 3 features
                    batch_x = batch_x[:, :3]
        
        # Check for valid measurements
        valid_mask = torch.isfinite(batch_vel).all(dim=1)
        # Ensure valid_mask is 1D
        if valid_mask.dim() > 1:
            valid_mask = valid_mask.squeeze()
        if not valid_mask.any():
            continue
            
        batch_x_valid = batch_x[valid_mask]
        batch_vel_valid = batch_vel[valid_mask]
        
        # Forward pass with gradient computation (removed torch.no_grad())
        u_pred, v_pred, _ = model(batch_x_valid)
        pred_vel = torch.cat((u_pred, v_pred), dim=1)
        
        # Check for valid predictions
        valid_pred_mask = torch.isfinite(pred_vel).all(dim=1)
        if not valid_pred_mask.any():
            continue
            
        final_pred = pred_vel[valid_pred_mask]
        final_meas = batch_vel_valid[valid_pred_mask]
        
        # Compute loss based on selected function
        if loss_fn == "huber":
            batch_loss = _huber(final_pred, final_meas)
        else:
            # default MSE
            diff = final_pred - final_meas
            batch_loss = torch.mean(diff**2)
        
        if torch.isfinite(batch_loss):
            total_loss += batch_loss * len(final_pred)  # Weight by number of points
            total_valid_points += len(final_pred)
    
    # Return average loss with proper gradients
    if total_valid_points == 0:
        print("Warning: No valid data points found for data loss computation")
        return torch.tensor(1.0, device=x_data.device, requires_grad=True)
    
    avg_loss = total_loss / total_valid_points
    if not torch.isfinite(avg_loss):
        print(f"Warning: Non-finite data loss detected, using fallback value")
        return torch.tensor(1.0, device=x_data.device, requires_grad=True)
        
    return avg_loss  # Return tensor with gradients intact


def total_loss(*args, **kwargs):
    """Deprecated - total loss computation moved inside train_pinn to handle
    normalisation correctly. Call train_pinn instead."""
    raise RuntimeError("total_loss() is outdated after normalisation refactor - use train_pinn or re-implement if needed.")


def train_pinn(model, optimizer, x_data, measured_vel, domain,
               n_colloc=1000, beta=1.0, epochs=2000, print_every=250,
               scheduler=None,
               data_only_epochs=0,
               data_loss_weight=1.0,
               compute_data_loss=True,
               stop_after_epochs=0,
               physics_mode="full",
               data_loss_floor=False,
               data_loss_floor_tolerance=0.05,
               data_loss_floor_decay=0.5,
               physics_ramp_start=1.0,
               physics_ramp_end=1.0,
               physics_ramp_epochs=0,
               physics_ramp_auto_start=False,
               physics_ramp_auto_target_ratio=1.0,
               physics_ramp_auto_clip=(1e-12, 1.0),
               pressure_gauge_weight=0.0,
               use_density_guided=True, grid_resolution=50, bandwidth=0.05,
               min_density_threshold=0.001, fallback_ratio=0.1, plot_density=False,
               use_fast_density=True, density_method="auto",
               data_batch_size=10000, use_optimized_physics=True,
               orig_bounds=None,
               colloc_coverage_reserve_frac=0.25,
               colloc_seed_with_epoch=False,
               colloc_strict_occupied_only=True,
               colloc_coverage_schedule=True,
               colloc_coverage_schedule_seed=0,
               data_loss_fn="mse",
               huber_delta=1.0,
               epoch_callback=None):
    """
    Train the PINN model.
    
    Parameters:
        model: The PINN model to train
        optimizer: Optimizer for training
        x_data: Training data coordinates
        measured_vel: Measured velocity values
        domain: Domain specification
        n_colloc: Number of collocation points
        beta: Physics loss weight
        epochs: Number of training epochs
        print_every: Print frequency
        scheduler: Learning rate scheduler
        data_only_epochs: Number of initial epochs to train with ONLY data loss.
            During this warmup, physics loss is disabled. After warmup, normal training resumes.
        data_loss_weight: Additional multiplicative weight applied to the data loss term
            (after any other weighting). Set to 0.0 for physics-only optimization.
        compute_data_loss: If False, skip computing data loss entirely (saves time).
        stop_after_epochs: If > 0, stop training after this many epochs (useful for smoke tests).
        physics_mode: Which physics loss to use during the physics phase.
            - "full": Full Stokes momentum + continuity (default, legacy behavior).
            - "divergence_only": Only incompressibility constraint (div u = 0).
              Avoids the Laplacian smoothing from momentum equations.
            - "none": Skip physics loss entirely (equivalent to beta=0).
        data_loss_floor: If True, monitor data loss during the physics phase.
            If data loss rises above (1 + tolerance) * warmup_end_data_loss,
            multiplicatively reduce the effective physics weight by decay factor.
        data_loss_floor_tolerance: Fractional tolerance above warmup end data loss
            before the floor mechanism kicks in.
        data_loss_floor_decay: Multiplicative factor applied to the physics weight
            each epoch the data loss exceeds the floor (e.g. 0.5 halves it).
        physics_ramp_start: Multiplicative scale applied to the physics loss at the FIRST
            physics epoch (i.e. epoch == data_only_epochs + 1). Typically small (e.g. 0.01).
        physics_ramp_end: Maximum (end) multiplicative scale applied to the physics loss at the END
            of the ramp. Default 1.0 preserves prior behavior. Use <1.0 to keep physics as a
            regularizer and avoid overpowering data fit.
        physics_ramp_epochs: Number of physics-phase epochs to linearly ramp the physics
            loss scale from `physics_ramp_start` to `physics_ramp_end` (inclusive). Set <=1 to disable.
        physics_ramp_auto_start: If True, automatically choose an appropriate effective
            physics_ramp_start at the first physics epoch based on observed loss magnitudes.
            This helps avoid a huge physics-loss shock when physics turns on.
        physics_ramp_auto_target_ratio: Target ratio for scaled physics loss vs data loss at
            the first physics epoch. A value of 1.0 aims for physics_loss_scaled ≈ data_loss.
        physics_ramp_auto_clip: (min,max) clip for the auto chosen start scale.
        pressure_gauge_weight: Weight for a pressure gauge constraint that removes the constant
            pressure nullspace. Implemented as (mean(p_star))^2 on collocation points during the
            physics phase. This stabilizes training but does not by itself prevent trivial constant
            velocity solutions.
        use_density_guided: Whether to use density guided sampling
        grid_resolution: Grid resolution for density estimation
        bandwidth: Bandwidth for density estimation
        min_density_threshold: Minimum density threshold
        fallback_ratio: Fallback ratio for uniform sampling
        plot_density: Whether to plot density visualization
        use_fast_density: Whether to use fast density estimation
        density_method: Method for density estimation
        data_batch_size: Batch size for data loss
        use_optimized_physics: Whether to use optimized physics computation
        orig_bounds: Optional dict with original coordinate bounds from the data loader
            (keys: X_min/X_max/Z_min/Z_max).
        colloc_coverage_reserve_frac: If > 0, reserve this fraction of collocation points for
            uniform sampling over every histogram cell that contains at least one data point
            (fills low density track regions over training). Remainder uses density-weighted cells.
        colloc_seed_with_epoch: If True, pass the training epoch as RNG seed for histogram
            subsampling (large datasets) and for the coverage-uniform draws (reproducible sweeps).
        colloc_strict_occupied_only: If True, disallow density collocation placement in histogram
            cells that contain zero raw data points (prevents smoothed-density leakage into empty space).
        colloc_coverage_schedule: If True and coverage reserve is enabled, walk occupied bins using a
            deterministic per epoch schedule so low density occupied regions are revisited systematically.
        colloc_coverage_schedule_seed: Seed used to build the fixed occupied bin permutation for the
            deterministic reserve schedule.
        
    Returns:
        data_loss_history: History of data loss values (unweighted)
        physics_loss_history: History of physics loss values (weighted)
        total_loss_history: History of total loss values
    """
    model.train()
    print(f"Using pinn_model module: {__file__}")

    data_loss_history = []
    physics_loss_history = []
    total_loss_history = []
    
    print("Training with ADAM")
    print(f"Dataset size: {len(x_data):,} points")
    print(f"Data batch size: {data_batch_size:,} points")
    print(f"Number of data batches: {len(x_data) // data_batch_size + 1}")

    # Compute normalisation statistics and star datasets
    normalizer = DataNormalizer.from_dataset(x_data, measured_vel)
    model.normalizer = normalizer  # attach for inference convenience

    # Pre-compute starred versions of training data
    # Use cat instead of stack to avoid creating extra dimension
    x_data_s = torch.cat(
        (
            normalizer.star(x_data[:, 0:1], 'x'),
            normalizer.star(x_data[:, 1:2], 'z'),
            normalizer.star(x_data[:, 2:3], 't'),
        ), dim=1)
    measured_vel_s = torch.cat(
        (
            normalizer.star(measured_vel[:, 0:1], 'u'),
            normalizer.star(measured_vel[:, 1:2], 'v'),
        ), dim=1)
    
    device = x_data.device if torch.is_tensor(x_data) else torch.device('cpu')

    # special epochs for detailed logging, global early debug
    special_epochs = {1, 2, 3, 4, 5, 10}

    # detailed logging right after warmup ends, physics-phase-relative epochs
    post_warmup_log_epochs = set(range(1, 11))  # 1-10
    post_warmup_log_epochs.update(range(20, 101, 10))  # 20,30,...100
    post_warmup_log_epochs.update({200, 300, 400})
    
    x_data_np = x_data.cpu().numpy() if torch.is_tensor(x_data) else x_data

    # verify x_data_np is in [0,1] normalized space for consistency checks
    if len(x_data_np) > 0:
        x_range = (x_data_np[:, 0].min(), x_data_np[:, 0].max())
        z_range = (x_data_np[:, 1].min(), x_data_np[:, 1].max())
        if x_range[0] < -0.01 or x_range[1] > 1.01 or z_range[0] < -0.01 or z_range[1] > 1.01:
            print(f"Warning: x_data_np appears to be outside [0,1] range - may cause issues")
    
    colloc_coverage_reserve_frac = float(np.clip(float(colloc_coverage_reserve_frac or 0.0), 0.0, 0.95))
    colloc_seed_with_epoch = bool(colloc_seed_with_epoch)
    colloc_strict_occupied_only = bool(colloc_strict_occupied_only)
    colloc_coverage_schedule = bool(colloc_coverage_schedule)
    colloc_coverage_schedule_seed = int(colloc_coverage_schedule_seed or 0)
    if colloc_coverage_reserve_frac > 0.0:
        print(
            f"Collocation coverage reserve: {colloc_coverage_reserve_frac:.3f} of each physics batch "
            f"sampled uniformly over data-occupied histogram cells (see density_sampling.py)."
        )
    if colloc_seed_with_epoch:
        print("Collocation RNG: epoch based seed for histogram subsample + coverage draws.")
    if colloc_strict_occupied_only:
        print("Collocation occupancy filter: STRICT (no sampling in zero occupancy histogram cells).")
    if colloc_coverage_schedule and colloc_coverage_reserve_frac > 0.0:
        print(
            f"Collocation coverage schedule: ON (deterministic occupied bin cycling, "
            f"seed={colloc_coverage_schedule_seed})."
        )

    colloc_occ_mask = None
    colloc_occ_edges = None
    if colloc_strict_occupied_only and len(x_data_np) > 0:
        cx = np.linspace(domain["x"][0], domain["x"][1], grid_resolution)
        cz = np.linspace(domain["z"][0], domain["z"][1], grid_resolution)
        ct = np.linspace(domain["t"][0], domain["t"][1], grid_resolution)
        colloc_occ_mask = np.histogramdd(
            [x_data_np[:, 0], x_data_np[:, 1], x_data_np[:, 2]],
            bins=[cx, cz, ct],
        )[0] > 0
        colloc_occ_edges = (cx, cz, ct)

    def _filter_colloc_to_occupied(points_np):
        if (not colloc_strict_occupied_only) or colloc_occ_mask is None:
            return points_np
        if points_np is None or len(points_np) == 0:
            return np.empty((0, 3), dtype=np.float32)
        pts = np.asarray(points_np, dtype=np.float32)
        cx, cz, ct = colloc_occ_edges
        x = np.clip(pts[:, 0], cx[0], cx[-1] - 1e-12)
        z = np.clip(pts[:, 1], cz[0], cz[-1] - 1e-12)
        t = np.clip(pts[:, 2], ct[0], ct[-1] - 1e-12)
        ix = np.searchsorted(cx, x, side="right") - 1
        iz = np.searchsorted(cz, z, side="right") - 1
        it = np.searchsorted(ct, t, side="right") - 1
        ix = np.clip(ix, 0, colloc_occ_mask.shape[0] - 1)
        iz = np.clip(iz, 0, colloc_occ_mask.shape[1] - 1)
        it = np.clip(it, 0, colloc_occ_mask.shape[2] - 1)
        keep = colloc_occ_mask[ix, iz, it]
        return pts[keep]

    print("Starting training loop...")
    t_data_values = x_data_np[:, 2] if len(x_data_np) > 0 else np.array([])
    physics_mode = str(physics_mode or "full").lower()
    if physics_mode not in ("full", "divergence_only", "none"):
        raise ValueError(f"physics_mode must be 'full', 'divergence_only', or 'none', got: {physics_mode}")
    print(f"Physics mode: {physics_mode}")

    data_loss_floor = bool(data_loss_floor)
    data_loss_floor_tolerance = float(data_loss_floor_tolerance)
    data_loss_floor_decay = float(data_loss_floor_decay)
    warmup_end_data_loss = None
    floor_phys_scale = 1.0
    if data_loss_floor:
        print(f"Data loss floor enabled: tolerance={data_loss_floor_tolerance}, decay={data_loss_floor_decay}")

    data_only_epochs = max(0, int(data_only_epochs or 0))
    if data_only_epochs > 0:
        print(f"Data-only warmup enabled: first {data_only_epochs} epochs use DATA loss only (physics disabled).")
    if physics_ramp_auto_start:
        print(
            f"Physics ramp auto start: ON (target_ratio={physics_ramp_auto_target_ratio}, "
            f"clip=[{physics_ramp_auto_clip[0]}, {physics_ramp_auto_clip[1]}])"
        )
    data_loss_weight = float(data_loss_weight)
    compute_data_loss = bool(compute_data_loss)
    if (not compute_data_loss) or data_loss_weight == 0.0:
        print(f"Data loss disabled for optimization (compute_data_loss={compute_data_loss}, data_loss_weight={data_loss_weight}).")
        if physics_ramp_auto_start:
            print("Warning: physics_ramp_auto_start requires data loss magnitude; disabling auto start.")
            physics_ramp_auto_start = False

    # effective start scale used by the physics ramp, may be auto-updated at physics_phase_epoch==1
    physics_ramp_start_effective = float(np.clip(float(physics_ramp_start), 0.0, 1.0))
    physics_ramp_end_effective = float(np.clip(float(physics_ramp_end), 0.0, 1.0))

    # optional early stop for smoke tests
    stop_after_epochs = int(stop_after_epochs or 0)
    if stop_after_epochs > 0:
        print(f"Early-stop enabled: will stop after {stop_after_epochs} epochs (smoke test).")

    # Training loop
    for ep in range(1, epochs + 1):
        if ep == 1:
            print(f"Starting epoch {ep}/{epochs}")
        
        optimizer.zero_grad()
        
        # Clear GPU cache periodically
        if ep % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        is_data_only_phase = (ep <= data_only_epochs)
        physics_phase_epoch = (ep - data_only_epochs) if (not is_data_only_phase) else 0

        # compute data loss, optional
        if compute_data_loss:
            loss_d = data_loss(model, x_data_s, measured_vel_s, batch_size=data_batch_size,
                               loss_fn=data_loss_fn, huber_delta=huber_delta)
        else:
            loss_d = torch.tensor(0.0, device=device, requires_grad=True)

        # Capture warmup end data loss for data_loss_floor mechanism
        if data_loss_floor and is_data_only_phase and ep == data_only_epochs and compute_data_loss:
            warmup_end_data_loss = float(loss_d.item())
            print(f"[Floor] Recorded warmup end data loss: {warmup_end_data_loss:.4e}")

        # Physics loss is skipped during data-only warmup
        x_colloc = None
        if is_data_only_phase:
            loss_p = torch.tensor(0.0, device=device, requires_grad=True)
            phys_scale = 0.0
            loss_p_raw = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # Physics loss global ramp (starts when physics turns on, after data_only_epochs)
            # Inclusive ramp: at physics_phase_epoch==1 => physics_ramp_start
            # at physics_phase_epoch==physics_ramp_epochs => physics_ramp_end
            p_start = float(min(physics_ramp_start_effective, physics_ramp_end_effective))
            p_end = float(physics_ramp_end_effective)
            p_len = int(physics_ramp_epochs) if physics_ramp_epochs is not None else 0
            if p_len <= 1:
                phys_scale = p_end
            else:
                progress = (physics_phase_epoch - 1) / float(p_len - 1)
                progress = float(np.clip(progress, 0.0, 1.0))
                phys_scale = p_start + (p_end - p_start) * progress

            # generate collocation points, physics phase only
            if ep == data_only_epochs + 1:
                print(f"Entering full PINN training at epoch {ep}: enabling physics losses.")
                print(f"Generating collocation points (use_density_guided={use_density_guided})...")
                if p_len > 1 or p_start != p_end:
                    print(f"Physics-loss ramp enabled: start={p_start:.4g} → {p_end:.4g} over {p_len} physics epochs")

            # regenerate colloc every 1000 physics epochs with epoch-based seeding so
            # the coverage schedule cycles through different occupied bins over time;
            # balances efficiency (no per-epoch regen) with spatial coverage
            _colloc_regen_interval = 1000  # regenerate every N physics epochs
            _need_new_colloc = (
                (ep == data_only_epochs + 1)  # first physics epoch
                or colloc_seed_with_epoch     # user requested per epoch variation
                or (physics_phase_epoch % _colloc_regen_interval == 0)  # periodic refresh
            )
            if _need_new_colloc or not hasattr(train_pinn, '_cached_colloc_np'):
                _epoch_seed = int(ep) if (colloc_seed_with_epoch or physics_phase_epoch % _colloc_regen_interval == 0) else None
                colloc_np = generate_collocation_points(
                    n_colloc, domain, x_data_np, use_density_guided,
                    grid_resolution, bandwidth, min_density_threshold,
                    fallback_ratio, use_fast_density, density_method,
                    coverage_reserve_frac=colloc_coverage_reserve_frac,
                    histogram_subsample_seed=_epoch_seed,
                    coverage_sample_seed=_epoch_seed,
                    strict_occupied_only=colloc_strict_occupied_only,
                    coverage_schedule=colloc_coverage_schedule,
                    coverage_schedule_seed=colloc_coverage_schedule_seed,
                    coverage_epoch=ep,
                )

                # analyze sampling effectiveness BEFORE converting to star space, do once
                if ep == data_only_epochs + 1 and use_density_guided:
                    print(f"\nAnalyzing density guided sampling effectiveness...")
                    analyze_sampling_effectiveness(x_data_np, colloc_np.copy(), domain, "density_sampling_analysis.png")

                # Cache for reuse until next regeneration
                train_pinn._cached_colloc_np = colloc_np.copy()
            else:
                colloc_np = train_pinn._cached_colloc_np.copy()

            if ep == data_only_epochs + 1:
                try:
                    cg = report_collocation_coverage_gaps(
                        x_data_np, np.clip(colloc_np, 0.0, 1.0), domain, grid_resolution
                    )
                    print(
                        f"[Colloc coverage] grid={grid_resolution} occupied_bins={cg['n_bins_occupied']} "
                        f"hit={cg['n_bins_hit']} gap={cg['n_bins_gap']} "
                        f"({100.0 * cg['frac_gap_among_occupied']:.1f}% of occupied bins have zero colloc) "
                        f"| reserve_frac={colloc_coverage_reserve_frac} seed_epoch={colloc_seed_with_epoch}"
                    )
                except Exception as _e:
                    print(f"[Colloc coverage] report failed: {_e}")

            # Convert to star space AFTER analysis
            colloc_np[:, 0] = normalizer.star(colloc_np[:, 0], 'x')
            colloc_np[:, 1] = normalizer.star(colloc_np[:, 1], 'z')
            colloc_np[:, 2] = normalizer.star(colloc_np[:, 2], 't')

            x_colloc = torch.tensor(colloc_np, dtype=torch.float32, device=device)

            if physics_mode == "divergence_only":
                loss_p_raw = physics_loss_divergence_only(model, x_colloc, normalizer)
            elif physics_mode == "none":
                loss_p_raw = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss_p_raw = physics_loss(
                    model, x_colloc, normalizer,
                    use_optimized=use_optimized_physics,
                )

            # auto-choose ramp start at first physics epoch from observed magnitudes;
            # avoids a huge physics shock when loss_p_raw >> loss_d
            if physics_ramp_auto_start and physics_phase_epoch == 1:
                eps = 1e-30
                auto_min = float(physics_ramp_auto_clip[0])
                auto_max = float(physics_ramp_auto_clip[1])
                target_ratio = float(physics_ramp_auto_target_ratio)
                # Choose start so that (start * loss_p_raw) ≈ target_ratio * loss_d
                auto_start = target_ratio * float(loss_d.detach().item()) / (float(loss_p_raw.detach().item()) + eps)
                auto_start = float(np.clip(auto_start, auto_min, auto_max))
                physics_ramp_start_effective = min(auto_start, physics_ramp_end_effective)
                # Recompute phys_scale for epoch 1 using the auto chosen start
                if p_len <= 1:
                    phys_scale = float(physics_ramp_end_effective)
                else:
                    phys_scale = physics_ramp_start_effective
                print(
                    f"Physics ramp auto start selected: {physics_ramp_start_effective:.3e} "
                    f"(loss_d={loss_d.item():.3e}, loss_p_raw={loss_p_raw.item():.3e})"
                )

            # Apply scaling (manual/auto start + ramp)
            loss_p = float(phys_scale) * loss_p_raw

            # Data loss floor: reduce physics weight when data loss degrades
            if data_loss_floor and warmup_end_data_loss is not None and compute_data_loss:
                current_data_loss = float(loss_d.item())
                threshold = warmup_end_data_loss * (1.0 + data_loss_floor_tolerance)
                if current_data_loss > threshold:
                    floor_phys_scale *= data_loss_floor_decay
                    floor_phys_scale = max(floor_phys_scale, 1e-6)
                else:
                    # Slowly recover when data loss is back under control
                    floor_phys_scale = min(floor_phys_scale * 1.02, 1.0)
                loss_p = loss_p * floor_phys_scale
                if physics_phase_epoch <= 5 or physics_phase_epoch % 500 == 0:
                    print(
                        f"[Floor] ep={ep}: data_loss={current_data_loss:.4e}, "
                        f"threshold={threshold:.4e}, floor_scale={floor_phys_scale:.4g}"
                    )

        # pressure gauge, physics phase only, remove constant-pressure nullspace by enforcing mean(p_star)=0
        loss_pressure_gauge = torch.tensor(0.0, device=device, requires_grad=True)
        pressure_gauge_weight = float(pressure_gauge_weight)
        if (not is_data_only_phase) and pressure_gauge_weight > 0.0 and x_colloc is not None and x_colloc.shape[0] > 0:
            # Model returns STAR outputs during training
            _, _, p_star = model(x_colloc)
            # mean pressure gauge, keeps p well-conditioned since pressure is defined up to a constant
            loss_pressure_gauge = torch.mean(p_star) ** 2

        # fixed-curriculum weighting, data weight = data_loss_weight, physics
        # weight = beta (ramp already applied to loss_p above via phys_scale)
        current_beta = beta
        current_alpha = float(data_loss_weight)

        # Combine losses
        weighted_phys_loss = current_beta * loss_p
        loss_total = (data_loss_weight * loss_d) + weighted_phys_loss

        # add pressure gauge loss, not ramped by phys_scale, just a tiny stabilization term
        weighted_pressure_gauge = pressure_gauge_weight * loss_pressure_gauge
        loss_total = loss_total + weighted_pressure_gauge

        # For logging/plots: define "physics loss" as *all non-data terms* (PDE + gauge)
        physics_loss_total = weighted_phys_loss + weighted_pressure_gauge

        # Backward pass
        loss_total.backward()

        # gradient clipping, must happen AFTER backward
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimization step
        optimizer.step()
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(loss_total)
        
        # Store loss values
        data_loss_history.append(loss_d.item())
        physics_loss_history.append(physics_loss_total.item())
        total_loss_history.append(loss_total.item())
        
        # Logging
        should_log_post_warmup = (not is_data_only_phase) and (physics_phase_epoch in post_warmup_log_epochs)
        if should_log_post_warmup or ep in special_epochs or ep % print_every == 0:
            log_str = (f"Epoch {ep:05d}: Total Loss = {loss_total.item():.4e}, "
                       f"Data Loss = {loss_d.item():.4e}, Physics Loss = {physics_loss_total.item():.4e} "
                       f"(w_d={current_alpha:.3f}, w_p={current_beta:.3f})")
            if not is_data_only_phase:
                log_str += f", phys_scale={phys_scale:.4g}"
                # Raw (unscaled) physics is useful for interpreting ramped runs
                try:
                    log_str += f", phys_raw={loss_p_raw.item():.4e}"
                except Exception:
                    pass
                # Also show the weighted PDE-only term explicitly to avoid ambiguity
                log_str += f", PDE(w)={weighted_phys_loss.item():.4e}"
                if pressure_gauge_weight > 0.0:
                    try:
                        log_str += f", P_gauge = {loss_pressure_gauge.item():.4e} (w={pressure_gauge_weight:.3g})"
                    except Exception:
                        pass
            print(log_str)
            
            # GPU memory status
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            # optional additive held-out guard hook, e.g. corr/R^2 collapse check
            # receives (epoch, model, normalizer); must not mutate training state
            if epoch_callback is not None:
                try:
                    epoch_callback(ep, model, normalizer)
                except Exception as _cb_e:
                    print(f"  [epoch_callback error] {_cb_e}")
                finally:
                    model.train()

        # optional early stop for smoke tests
        if stop_after_epochs > 0 and ep >= stop_after_epochs:
            print(f"Early-stop: reached epoch {ep} (limit {stop_after_epochs}). Stopping training loop.")
            break

    return data_loss_history, physics_loss_history, total_loss_history
