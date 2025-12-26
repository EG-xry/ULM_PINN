"""
Vessel Mask Utilities for PINN Boundary Regulation

This module provides functions to:
1. Load vessel masks created by Ver1_vessel_mask.py
2. Identify wall pixels (boundary pixels)
3. Check if points are inside/outside vessels
4. Sample points from different regions (inside, outside, walls)
"""

import numpy as np
try:
    import torch  
except Exception:  
    torch = None
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_erosion


def load_vessel_mask(mask_path, normalize_to_domain=True, domain=None, orig_bounds=None):
    """
    Load vessel mask from .npz file and prepare for use in PINN.
    
    Parameters:
        mask_path: Path to the .npz file containing vessel_mask, x_grid, z_grid
        normalize_to_domain: If True, assumes coordinates are in [0,1] already.
            If the grid is NOT actually in [0,1], this function will attempt to
            normalize it (see orig_bounds fallback behavior below).
        domain: Dictionary with 'x' and 'z' bounds if normalization needed (legacy).
        orig_bounds: Optional dict with keys {'X_min','X_max','Z_min','Z_max'} from
            `data_loading.py`. If provided and the vessel mask grid is not in [0,1],
            the grid will be normalized using these bounds so that it matches the
            same [0,1] normalization applied to your coordinate data.
    
    Returns:
        vessel_mask: Binary 2D array (True = vessel region)
        x_grid: Grid x-coordinates (normalized to [0,1])
        z_grid: Grid z-coordinates (normalized to [0,1])
    """
    print(f"Loading vessel mask from: {mask_path}")
    data = np.load(mask_path)
    
    vessel_mask = data['vessel_mask'].astype(bool)
    x_grid = data['x_grid'].astype(np.float32)
    z_grid = data['z_grid'].astype(np.float32)
    
    print(f"Vessel mask shape: {vessel_mask.shape}")
    print(f"X grid range: [{x_grid.min():.4f}, {x_grid.max():.4f}]")
    print(f"Z grid range: [{z_grid.min():.4f}, {z_grid.max():.4f}]")
    print(f"Vessel coverage: {100*np.sum(vessel_mask)/vessel_mask.size:.2f}%")
    
    # Normalize grids to [0,1] if needed
    if not normalize_to_domain:
        # Legacy behavior: normalize using provided domain bounds
        if domain is None:
            raise ValueError("domain must be provided if normalize_to_domain=False")
        x_grid = (x_grid - domain['x'][0]) / (domain['x'][1] - domain['x'][0] + 1e-6)
        z_grid = (z_grid - domain['z'][0]) / (domain['z'][1] - domain['z'][0] + 1e-6)
    else:
        # Assume already normalized, but verify. If not, normalize intelligently.
        grid_not_normalized = (
            (x_grid.min() < -0.1) or (x_grid.max() > 1.1) or
            (z_grid.min() < -0.1) or (z_grid.max() > 1.1)
        )
        if grid_not_normalized:
            print(
                f"Warning: Vessel mask grid appears not normalized "
                f"(X {x_grid.min():.2f}..{x_grid.max():.2f}, Z {z_grid.min():.2f}..{z_grid.max():.2f})."
            )
            if isinstance(orig_bounds, dict) and all(k in orig_bounds for k in ("X_min", "X_max", "Z_min", "Z_max")):
                # Normalize using the SAME bounds used to normalize tracking coordinates -> [0,1]
                X_min, X_max = float(orig_bounds["X_min"]), float(orig_bounds["X_max"])
                Z_min, Z_max = float(orig_bounds["Z_min"]), float(orig_bounds["Z_max"])
                x_grid = (x_grid - X_min) / (X_max - X_min + 1e-6)
                z_grid = (z_grid - Z_min) / (Z_max - Z_min + 1e-6)
                print(
                    "  Normalized vessel mask grid using orig_bounds "
                    f"(X_min={X_min:.3f}, X_max={X_max:.3f}, Z_min={Z_min:.3f}, Z_max={Z_max:.3f})"
                )
            else:
                # Fallback: normalize by mask grid min/max (good for visualization; may be slightly off vs data normalization)
                x_min, x_max = float(x_grid.min()), float(x_grid.max())
                z_min, z_max = float(z_grid.min()), float(z_grid.max())
                x_grid = (x_grid - x_min) / (x_max - x_min + 1e-6)
                z_grid = (z_grid - z_min) / (z_max - z_min + 1e-6)
                print("  Normalized vessel mask grid using its own min/max (fallback).")
            print(f"  Normalized X grid range: [{x_grid.min():.4f}, {x_grid.max():.4f}]")
            print(f"  Normalized Z grid range: [{z_grid.min():.4f}, {z_grid.max():.4f}]")
        else:
            # Already in a reasonable normalized range
            pass
    
    return vessel_mask, x_grid, z_grid


def identify_wall_pixels(vessel_mask):
    """
    Identify wall pixels: pixels where mask=1 but at least one neighbor=0.
    
    Parameters:
        vessel_mask: Binary 2D array (True = vessel region)
    
    Returns:
        wall_mask: Binary 2D array (True = wall pixel)
        wall_coords: Array of shape (N_wall, 2) with (x_idx, z_idx) indices
    """
    print("Identifying wall pixels...")
    
    # Wall pixels are vessel pixels that have at least one non-vessel neighbor
    # Use erosion to find interior pixels, then subtract from mask
    eroded_mask = binary_erosion(vessel_mask, iterations=1)
    wall_mask = vessel_mask & ~eroded_mask
    
    # Get coordinates of wall pixels
    wall_indices = np.where(wall_mask)
    wall_coords = np.column_stack([wall_indices[0], wall_indices[1]])
    
    print(f"Found {len(wall_coords):,} wall pixels ({100*len(wall_coords)/vessel_mask.size:.2f}% of grid)")
    
    return wall_mask, wall_coords


def create_vessel_interpolator(vessel_mask, x_grid, z_grid):
    """
    Create an interpolator to check if arbitrary (x,z) points are inside vessels.
    
    Parameters:
        vessel_mask: Binary 2D array (True = vessel region)
        x_grid: Grid x-coordinates (normalized)
        z_grid: Grid z-coordinates (normalized)
    
    Returns:
        interpolator: scipy RegularGridInterpolator function
    """
    # Create interpolator for vessel mask (values: 1.0 inside, 0.0 outside)
    mask_values = vessel_mask.astype(np.float32)
    interpolator = RegularGridInterpolator(
        (x_grid, z_grid), 
        mask_values.T,  # Transpose because RegularGridInterpolator expects (z, x) indexing
        method='nearest',
        bounds_error=False,
        fill_value=0.0  # Points outside grid are considered outside vessel
    )
    return interpolator


def check_points_in_vessel(x_points, z_points, vessel_interpolator):
    """
    Check if points are inside the vessel.
    
    Parameters:
        x_points: Array of x-coordinates (can be numpy or torch)
        z_points: Array of z-coordinates (can be numpy or torch)
        vessel_interpolator: Interpolator function from create_vessel_interpolator
    
    Returns:
        in_vessel: Boolean array (True = inside vessel)
    """
    # Convert to numpy if needed
    if torch.is_tensor(x_points):
        x_np = x_points.detach().cpu().numpy()
    else:
        x_np = np.asarray(x_points)
    
    if torch.is_tensor(z_points):
        z_np = z_points.detach().cpu().numpy()
    else:
        z_np = np.asarray(z_points)
    
    # Flatten if needed
    x_flat = x_np.flatten()
    z_flat = z_np.flatten()
    
    # Check points
    points = np.column_stack([x_flat, z_flat])
    mask_values = vessel_interpolator(points)
    in_vessel = mask_values > 0.5  # Threshold at 0.5
    
    return in_vessel.reshape(x_np.shape) if x_np.ndim > 0 else in_vessel[0]


def sample_wall_points(wall_coords, x_grid, z_grid, n_samples, t_values=None):
    """
    Sample random points from wall coordinates.
    
    Parameters:
        wall_coords: Array of shape (N_wall, 2) with (x_idx, z_idx) indices
        x_grid: Grid x-coordinates
        z_grid: Grid z-coordinates
        n_samples: Number of points to sample
        t_values: Optional array of t-values to pair with wall points (if None, samples uniformly)
    
    Returns:
        wall_points: Array of shape (n_samples, 3) with (x, z, t) coordinates
    """
    if len(wall_coords) == 0:
        print("Warning: No wall coordinates available")
        return np.array([]).reshape(0, 3)
    
    # Sample random wall indices
    n_samples = min(n_samples, len(wall_coords))
    sampled_indices = np.random.choice(len(wall_coords), size=n_samples, replace=True)
    
    # Get grid indices
    x_indices = wall_coords[sampled_indices, 0]
    z_indices = wall_coords[sampled_indices, 1]
    
    # Convert to coordinates
    x_coords = x_grid[x_indices]
    z_coords = z_grid[z_indices]
    
    # Add small random perturbation to avoid exact grid alignment
    dx = (x_grid[1] - x_grid[0]) if len(x_grid) > 1 else 0.01
    dz = (z_grid[1] - z_grid[0]) if len(z_grid) > 1 else 0.01
    
    x_coords += np.random.uniform(-0.25 * dx, 0.25 * dx, len(x_coords))
    z_coords += np.random.uniform(-0.25 * dz, 0.25 * dz, len(z_coords))
    
    # Clip to valid range
    x_coords = np.clip(x_coords, x_grid.min(), x_grid.max())
    z_coords = np.clip(z_coords, z_grid.min(), z_grid.max())
    
    # Sample or use provided t-values
    if t_values is None:
        t_coords = np.random.uniform(0.0, 1.0, len(x_coords))
    else:
        t_coords = np.random.choice(t_values, size=len(x_coords), replace=True)
    
    wall_points = np.column_stack([x_coords, z_coords, t_coords])
    
    return wall_points


def sample_outside_vessel_points(vessel_interpolator, domain, n_samples, t_values=None):
    """
    Sample random points outside the vessel region.
    
    Parameters:
        vessel_interpolator: Interpolator function
        domain: Dictionary with 'x', 'z', 't' bounds
        n_samples: Number of points to sample
        t_values: Optional array of t-values (if None, samples uniformly)
    
    Returns:
        outside_points: Array of shape (n_samples, 3) with (x, z, t) coordinates
    """
    max_attempts = n_samples * 10  # Limit sampling attempts
    outside_points = []
    attempts = 0
    
    while len(outside_points) < n_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample random points
        x_sample = np.random.uniform(domain['x'][0], domain['x'][1])
        z_sample = np.random.uniform(domain['z'][0], domain['z'][1])
        
        # Check if outside vessel
        point_value = vessel_interpolator(np.array([[x_sample, z_sample]]))[0]
        if point_value < 0.5:  # Outside vessel
            if t_values is None:
                t_sample = np.random.uniform(domain['t'][0], domain['t'][1])
            else:
                t_sample = np.random.choice(t_values)
            outside_points.append([x_sample, z_sample, t_sample])
    
    if len(outside_points) < n_samples:
        print(f"Warning: Only sampled {len(outside_points)}/{n_samples} outside-vessel points")
    
    return np.array(outside_points)


def sample_inside_vessel_points(vessel_mask, x_grid, z_grid, n_samples, t_values=None, jitter_factor=0.25):
    """
    Sample random interior vessel points to emphasize intravascular dynamics.
    
    Parameters:
        vessel_mask: Binary 2D array marking vessel interior
        x_grid, z_grid: 1D coordinate arrays aligned with vessel_mask
        n_samples: Number of interior points to sample
        t_values: Optional array of time indices to sample from (if None, uniform [0,1])
        jitter_factor: Fraction of grid spacing used for random jitter to avoid aliasing
    
    Returns:
        inside_points: Array of shape (n_samples, 3) with (x, z, t) coordinates
    """
    vessel_indices = np.column_stack(np.where(vessel_mask))
    if len(vessel_indices) == 0 or n_samples <= 0:
        return np.empty((0, 3))
    
    # Sample indices with replacement to allow more points than grid cells
    sampled_indices = np.random.choice(len(vessel_indices), size=n_samples, replace=True)
    x_idx = vessel_indices[sampled_indices, 0]
    z_idx = vessel_indices[sampled_indices, 1]
    
    x_coords = x_grid[x_idx].astype(np.float32).copy()
    z_coords = z_grid[z_idx].astype(np.float32).copy()
    
    # Apply small jitter so physics points are not locked to grid centers
    dx = (x_grid[1] - x_grid[0]) if len(x_grid) > 1 else 0.01
    dz = (z_grid[1] - z_grid[0]) if len(z_grid) > 1 else 0.01
    jitter_x = jitter_factor * dx
    jitter_z = jitter_factor * dz
    if jitter_x > 0:
        x_coords += np.random.uniform(-jitter_x, jitter_x, size=len(x_coords))
    if jitter_z > 0:
        z_coords += np.random.uniform(-jitter_z, jitter_z, size=len(z_coords))
    
    x_coords = np.clip(x_coords, x_grid.min(), x_grid.max())
    z_coords = np.clip(z_coords, z_grid.min(), z_grid.max())
    
    if t_values is None or len(t_values) == 0:
        t_coords = np.random.uniform(0.0, 1.0, len(x_coords))
    else:
        t_vals = np.asarray(t_values).flatten()
        t_coords = np.random.choice(t_vals, size=len(x_coords), replace=True)
    
    return np.column_stack([x_coords, z_coords, t_coords])

