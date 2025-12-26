"""
PINN model + training loop
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from density_sampling import generate_collocation_points


class PINN(nn.Module):
    """Simple tanh MLP PINN"""

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 3,
        hidden_layers: int = 5,
        hidden_size: int = 64,
        activation: str = "tanh",
    ):
        super().__init__()
        if activation != "tanh":
            raise ValueError("Publishable PINN only supports activation='tanh'")

        layers: List[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, output_dim))
        self.network = nn.Sequential(*layers)

        self.log_mu = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.normalizer: Optional[DataNormalizer] = None

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_mu(self) -> torch.Tensor:
        return torch.exp(self.log_mu)

    def forward(self, x_star: torch.Tensor):
        out = self.network(x_star)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


class DataNormalizer:
    def __init__(self, stats: Dict[str, Tuple[float, float]]):
        self.stats = stats

    @staticmethod
    def _to_numpy(arr):
        if torch.is_tensor(arr):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    @classmethod
    def from_dataset(
        cls,
        x_data: torch.Tensor,
        vel_data: torch.Tensor,
        default_p_stats: Tuple[float, float] = (0.0, 1.0),
    ) -> "DataNormalizer":
        x_np = cls._to_numpy(x_data)
        v_np = cls._to_numpy(vel_data)
        return cls(
            {
                "x": (float(x_np[:, 0].mean()), float(x_np[:, 0].std() + 1e-12)),
                "z": (float(x_np[:, 1].mean()), float(x_np[:, 1].std() + 1e-12)),
                "t": (float(x_np[:, 2].mean()), float(x_np[:, 2].std() + 1e-12)),
                "u": (float(v_np[:, 0].mean()), float(v_np[:, 0].std() + 1e-12)),
                "v": (float(v_np[:, 1].mean()), float(v_np[:, 1].std() + 1e-12)),
                "p": (float(default_p_stats[0]), float(default_p_stats[1])),
            }
        )

    def _get_torch_pair(self, arr: torch.Tensor, key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, sd = self.stats[key]
        return (
            torch.tensor(mu, dtype=arr.dtype, device=arr.device),
            torch.tensor(sd, dtype=arr.dtype, device=arr.device),
        )

    def star(self, arr: torch.Tensor, key: str) -> torch.Tensor:
        mu_t, sd_t = self._get_torch_pair(arr, key)
        return (arr - mu_t) / sd_t


def _physics_loss(
    model: PINN,
    x_colloc_s: torch.Tensor,
    normalizer: DataNormalizer,
    include_unsteady: bool,
    rho_eff: float,
    use_optimized: bool = True,
) -> torch.Tensor:
    if not use_optimized:
        raise ValueError("use_optimized=False not supported")

    x_colloc_s = x_colloc_s.requires_grad_(True)
    u_s, v_s, p_s = model(x_colloc_s)
    mu = model.get_mu()

    sigma_x = normalizer.stats["x"][1]
    sigma_z = normalizer.stats["z"][1]
    sigma_t = normalizer.stats["t"][1]

    inv_sigma_x = 1.0 / sigma_x
    inv_sigma_z = 1.0 / sigma_z
    inv_sigma_t = 1.0 / sigma_t
    inv_sigma_x2 = inv_sigma_x**2
    inv_sigma_z2 = inv_sigma_z**2

    grads_u = torch.autograd.grad(u_s, x_colloc_s, torch.ones_like(u_s), retain_graph=True, create_graph=True)[0]
    grads_v = torch.autograd.grad(v_s, x_colloc_s, torch.ones_like(v_s), retain_graph=True, create_graph=True)[0]
    grads_p = torch.autograd.grad(p_s, x_colloc_s, torch.ones_like(p_s), retain_graph=True, create_graph=True)[0]

    u_xs, u_zs, u_ts = grads_u[:, 0:1], grads_u[:, 1:2], grads_u[:, 2:3]
    v_xs, v_zs, v_ts = grads_v[:, 0:1], grads_v[:, 1:2], grads_v[:, 2:3]
    p_xs, p_zs = grads_p[:, 0:1], grads_p[:, 1:2]

    u_x = u_xs * inv_sigma_x
    u_z = u_zs * inv_sigma_z
    v_x = v_xs * inv_sigma_x
    v_z = v_zs * inv_sigma_z
    p_x = p_xs * inv_sigma_x
    p_z = p_zs * inv_sigma_z
    u_t = u_ts * inv_sigma_t
    v_t = v_ts * inv_sigma_t

    u_xxs = torch.autograd.grad(u_xs, x_colloc_s, torch.ones_like(u_xs), retain_graph=True, create_graph=True)[0][:, 0:1]
    u_zzs = torch.autograd.grad(u_zs, x_colloc_s, torch.ones_like(u_zs), retain_graph=True, create_graph=True)[0][:, 1:2]
    v_xxs = torch.autograd.grad(v_xs, x_colloc_s, torch.ones_like(v_xs), retain_graph=True, create_graph=True)[0][:, 0:1]
    v_zzs = torch.autograd.grad(v_zs, x_colloc_s, torch.ones_like(v_zs), retain_graph=True, create_graph=True)[0][:, 1:2]

    u_xx = u_xxs * inv_sigma_x2
    u_zz = u_zzs * inv_sigma_z2
    v_xx = v_xxs * inv_sigma_x2
    v_zz = v_zzs * inv_sigma_z2

    lap_u = u_xx + u_zz
    lap_v = v_xx + v_zz

    base_mom_x = -p_x + mu * lap_u
    base_mom_z = -p_z + mu * lap_v

    if include_unsteady:
        rho_t = torch.tensor(float(rho_eff), dtype=u_s.dtype, device=u_s.device)
        mom_x = base_mom_x - rho_t * u_t
        mom_z = base_mom_z - rho_t * v_t
    else:
        mom_x = base_mom_x
        mom_z = base_mom_z

    div = u_x + v_z
    return torch.mean(mom_x**2 + mom_z**2) + torch.mean(div**2)


def _data_loss(model: PINN, x_data_s: torch.Tensor, vel_s: torch.Tensor, batch_size: int) -> torch.Tensor:
    total = torch.tensor(0.0, device=x_data_s.device)
    n = int(x_data_s.shape[0])
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        x_b = x_data_s[i:j]
        v_b = vel_s[i:j]
        u_s, v_s, _ = model(x_b)
        pred = torch.cat([u_s, v_s], dim=1)
        total = total + torch.mean((pred - v_b) ** 2) * (j - i)
    return total / float(n)


def train_pinn(
    model: PINN,
    optimizer: torch.optim.Optimizer,
    x_data: torch.Tensor,
    measured_vel: torch.Tensor,
    domain: Dict[str, Tuple[float, float]],
    n_colloc: int = 1000,
    beta: float = 1.0,
    epochs: int = 2000,
    print_every: int = 250,
    scheduler=None,
    data_only_epochs: int = 0,
    use_density_guided: bool = True,
    grid_resolution: int = 50,
    bandwidth: float = 0.05,
    min_density_threshold: float = 0.001,
    fallback_ratio: float = 0.1,
    plot_density: bool = False,
    use_fast_density: bool = True,
    density_method: str = "auto",
    data_batch_size: int = 10000,
    use_optimized_physics: bool = True,
    include_unsteady: bool = False,
    rho: float = 1000.0,
    unsteady_ramp_epochs: int = 0,
    **_ignored,
):
    device = x_data.device
    model.train()

    normalizer = DataNormalizer.from_dataset(x_data, measured_vel)
    model.normalizer = normalizer

    x_data_s = torch.cat(
        (
            normalizer.star(x_data[:, 0:1], "x"),
            normalizer.star(x_data[:, 1:2], "z"),
            normalizer.star(x_data[:, 2:3], "t"),
        ),
        dim=1,
    )
    vel_s = torch.cat(
        (
            normalizer.star(measured_vel[:, 0:1], "u"),
            normalizer.star(measured_vel[:, 1:2], "v"),
        ),
        dim=1,
    )

    x_data_np = x_data.detach().cpu().numpy()

    data_hist: List[float] = []
    phys_hist: List[float] = []
    total_hist: List[float] = []

    for ep in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        is_data_only = ep <= int(data_only_epochs)

        loss_d = _data_loss(model, x_data_s, vel_s, batch_size=int(max(1, data_batch_size)))

        if is_data_only:
            loss_p = torch.tensor(0.0, device=device)
        else:
            colloc_np = generate_collocation_points(
                int(n_colloc),
                domain,
                x_data=x_data_np,
                use_density_guided=bool(use_density_guided),
                grid_resolution=int(grid_resolution),
                bandwidth=float(bandwidth),
                min_density_threshold=float(min_density_threshold),
                fallback_ratio=float(fallback_ratio),
                use_fast_density=bool(use_fast_density),
                method=str(density_method),
            )
            x_colloc = torch.tensor(colloc_np, dtype=x_data.dtype, device=device)
            x_colloc_s = torch.cat(
                (
                    normalizer.star(x_colloc[:, 0:1], "x"),
                    normalizer.star(x_colloc[:, 1:2], "z"),
                    normalizer.star(x_colloc[:, 2:3], "t"),
                ),
                dim=1,
            )

            rho_eff = float(rho)
            if include_unsteady and int(unsteady_ramp_epochs) > 0:
                physics_epoch = max(1, ep - int(data_only_epochs))
                ramp = min(1.0, float(physics_epoch) / float(int(unsteady_ramp_epochs)))
                rho_eff = rho_eff * ramp

            loss_p = _physics_loss(
                model=model,
                x_colloc_s=x_colloc_s,
                normalizer=normalizer,
                include_unsteady=bool(include_unsteady),
                rho_eff=rho_eff,
                use_optimized=bool(use_optimized_physics),
            )

        loss_total = loss_d + float(beta) * loss_p
        loss_total.backward()
        optimizer.step()

        if scheduler is not None:
            try:
                scheduler.step(loss_total.detach())
            except TypeError:
                scheduler.step()

        data_hist.append(float(loss_d.detach().cpu().item()))
        phys_hist.append(float(loss_p.detach().cpu().item()))
        total_hist.append(float(loss_total.detach().cpu().item()))

        if print_every and (ep == 1 or ep % int(print_every) == 0 or ep == int(epochs)):
            if is_data_only:
                print(f"[{ep}/{epochs}] loss={total_hist[-1]:.4e} (data_only)")
            else:
                print(
                    f"[{ep}/{epochs}] loss={total_hist[-1]:.4e} "
                    f"(data={data_hist[-1]:.3e}, phys={phys_hist[-1]:.3e})"
                )

    return data_hist, phys_hist, total_hist
