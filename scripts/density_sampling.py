import gc
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
sklearn_available = True
try:
    from sklearn.cluster import DBSCAN
except ImportError:
    sklearn_available = False

def compute_density_map_fast(x_data, domain, grid_resolution=50, histogram_subsample_seed=None):
    """
    Compute a density map via a fast histogram approach (faster than KDE, less smooth)
    Parameters:
        x_data: Array of shape (N, 3) containing (x, z, t) coordinates of microbubbles
        domain: Dictionary specifying the domain limits
        grid_resolution: Number of grid points per dimension for density estimation
        histogram_subsample_seed: If set, ``np.random.default_rng(seed)`` is used when
            subsampling >100k points (reproducible). If None, unseeded RNG (legacy)
    
    Returns:
        density_map: 3D array of density values
        x_grid, z_grid, t_grid: grid coordinates
    """
    try:
        x_coords = x_data[:, 0]
        z_coords = x_data[:, 1]
        t_coords = x_data[:, 2]

        # Subsample very large datasets for the histogram
        max_points_for_hist = 100000
        if len(x_data) > max_points_for_hist:
            rng = (
                np.random.default_rng(histogram_subsample_seed)
                if histogram_subsample_seed is not None
                else np.random.default_rng()
            )
            indices = rng.choice(len(x_data), size=max_points_for_hist, replace=False)
            x_coords = x_coords[indices]
            z_coords = z_coords[indices]
            t_coords = t_coords[indices]

        x_grid = np.linspace(domain['x'][0], domain['x'][1], grid_resolution)
        z_grid = np.linspace(domain['z'][0], domain['z'][1], grid_resolution)
        t_grid = np.linspace(domain['t'][0], domain['t'][1], grid_resolution)

        density_map = np.histogramdd(
            [x_coords, z_coords, t_coords],
            bins=[x_grid, z_grid, t_grid],
            density=True,
        )[0]

        # smooth
        density_map = gaussian_filter(density_map, sigma=0.5)

        return density_map, x_grid, z_grid, t_grid

    except Exception as e:
        print(f"Error in compute_density_map_fast: {e}")
        print("Falling back to uniform density map.")
        x_grid = np.linspace(domain['x'][0], domain['x'][1], grid_resolution)
        z_grid = np.linspace(domain['z'][0], domain['z'][1], grid_resolution)
        t_grid = np.linspace(domain['t'][0], domain['t'][1], grid_resolution)
        X, Z, T = np.meshgrid(x_grid, z_grid, t_grid, indexing='ij')
        density_map = np.ones_like(X)
        density_map = density_map / np.sum(density_map)
        return density_map, x_grid, z_grid, t_grid


def compute_density_map(x_data, domain, grid_resolution=50, bandwidth=0.05, use_fast_density=True, method="auto",
                          histogram_subsample_seed=None):
    """
    Compute a density map from microbubble data using various methods
    
    Parameters:
        x_data: Array of shape (N, 3) containing (x, z, t) coordinates of microbubbles
        domain: Dictionary specifying the domain limits
        grid_resolution: Number of grid points per dimension for density estimation
        bandwidth: Bandwidth parameter for kernel density estimation
        use_fast_density: Whether to use fast histogram method for large datasets
        method: Method for density estimation ("auto", "kde", "histogram", "clustering")
    
    Returns:
        density_map: 3D array of density values
        x_grid, z_grid, t_grid: Grid coordinates
    """
    # Fast histogram method for large datasets
    if method == "histogram" or (use_fast_density and len(x_data) > 50000):
        return compute_density_map_fast(
            x_data, domain, grid_resolution, histogram_subsample_seed=histogram_subsample_seed
        )

    # KDE method for smaller datasets
    try:
        x_coords = x_data[:, 0]
        z_coords = x_data[:, 1]
        t_coords = x_data[:, 2]

        # Reduce grid resolution for large datasets
        if len(x_data) > 10000:
            grid_resolution = min(grid_resolution, 30)

        # Subsample for KDE if too large
        max_points_for_kde = 10000
        if len(x_data) > max_points_for_kde:
            indices = np.random.choice(len(x_data), size=max_points_for_kde, replace=False)
            x_coords_kde = x_coords[indices]
            z_coords_kde = z_coords[indices]
            t_coords_kde = t_coords[indices]
        else:
            x_coords_kde = x_coords
            z_coords_kde = z_coords
            t_coords_kde = t_coords

        x_grid = np.linspace(domain['x'][0], domain['x'][1], grid_resolution)
        z_grid = np.linspace(domain['z'][0], domain['z'][1], grid_resolution)
        t_grid = np.linspace(domain['t'][0], domain['t'][1], grid_resolution)

        X, Z, T = np.meshgrid(x_grid, z_grid, t_grid, indexing='ij')

        coords = np.vstack([x_coords_kde, z_coords_kde, t_coords_kde])

        try:
            kde = gaussian_kde(coords, bw_method=bandwidth)
            grid_points = np.vstack([X.ravel(), Z.ravel(), T.ravel()])

            # Process in batches for large grids
            batch_size = 100000
            if len(grid_points[0]) > batch_size:
                density_values = []
                for i in range(0, len(grid_points[0]), batch_size):
                    end_idx = min(i + batch_size, len(grid_points[0]))
                    batch_points = grid_points[:, i:end_idx]
                    batch_density = kde(batch_points)
                    density_values.extend(batch_density)
                density_values = np.array(density_values)
            else:
                density_values = kde(grid_points)
            
            density_map = density_values.reshape(X.shape)
            
        except Exception as e:
            print(f"Warning: KDE failed, using uniform density. Error: {e}")
            density_map = np.ones_like(X)

        density_map = density_map / (np.sum(density_map) + 1e-10)

        return density_map, x_grid, z_grid, t_grid
        
    except Exception as e:
        print(f"Error in compute_density_map: {e}")
        print("Falling back to uniform density map")
        x_grid = np.linspace(domain['x'][0], domain['x'][1], grid_resolution)
        z_grid = np.linspace(domain['z'][0], domain['z'][1], grid_resolution)
        t_grid = np.linspace(domain['t'][0], domain['t'][1], grid_resolution)
        X, Z, T = np.meshgrid(x_grid, z_grid, t_grid, indexing='ij')
        density_map = np.ones_like(X)
        density_map = density_map / np.sum(density_map)
        return density_map, x_grid, z_grid, t_grid


def data_histogram_counts(x_data, domain, grid_resolution):
    """Raw occupancy counts per histogram cell (same edges as density map)"""
    xg = np.linspace(domain["x"][0], domain["x"][1], grid_resolution)
    zg = np.linspace(domain["z"][0], domain["z"][1], grid_resolution)
    tg = np.linspace(domain["t"][0], domain["t"][1], grid_resolution)
    counts = np.histogramdd(
        [x_data[:, 0], x_data[:, 1], x_data[:, 2]],
        bins=[xg, zg, tg],
    )[0]
    return counts.astype(np.int64), (xg, zg, tg)


def histogram_counts_with_edges(x_data, xg, zg, tg):
    """Raw occupancy counts on caller-provided histogram edges"""
    counts = np.histogramdd(
        [x_data[:, 0], x_data[:, 1], x_data[:, 2]],
        bins=[xg, zg, tg],
    )[0]
    return counts.astype(np.int64)


def report_collocation_coverage_gaps(x_data, colloc_points, domain, grid_resolution=50):
    """
    Compare data occupancy vs. which histogram cells receive at least one collocation point

    Returns
    -------
    dict with keys:
        n_bins_occupied: cells with >=1 data sample
        n_bins_hit: occupied cells that received >=1 collocation sample
        n_bins_gap: occupied cells with zero collocation (physics never evaluated there)
        frac_gap_among_occupied: n_bins_gap / n_bins_occupied
        total_bins: number of histogram cells (``np.prod(counts.shape)``; edges from ``linspace`` give ``grid_resolution - 1`` bins per axis)
    """
    counts, (xg, zg, tg) = data_histogram_counts(x_data, domain, grid_resolution)
    occupied = counts > 0
    n_occ = int(occupied.sum())

    cp = np.clip(colloc_points, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    c_hist = np.histogramdd(
        [cp[:, 0], cp[:, 1], cp[:, 2]],
        bins=[xg, zg, tg],
    )[0]
    hit = occupied & (c_hist > 0)
    n_hit = int(hit.sum())
    gap = occupied & ~(c_hist > 0)
    n_gap = int(gap.sum())
    nb = int(np.prod(counts.shape))
    return {
        "n_bins_occupied": n_occ,
        "n_bins_hit": n_hit,
        "n_bins_gap": n_gap,
        "frac_gap_among_occupied": float(n_gap / max(n_occ, 1)),
        "total_bins": nb,
        "frac_occupied_of_domain": float(n_occ / max(nb, 1)),
    }


def sample_coverage_collocation(
    n_cover,
    counts,
    xg,
    zg,
    tg,
    domain,
    rng,
    schedule=False,
    coverage_epoch=None,
    schedule_seed=0,
):
    """
    Uniform over occupied histogram cells (any cell with count>0), bin centers in [0,1]^3
    """
    n_cover = int(max(0, n_cover))
    if n_cover <= 0:
        return np.empty((0, 3), dtype=np.float32)

    occ_idx = np.argwhere(counts > 0)
    if occ_idx.size == 0:
        return generate_collocation_points_uniform(n_cover, domain)
    xc = 0.5 * (xg[:-1] + xg[1:])
    zc = 0.5 * (zg[:-1] + zg[1:])
    tc = 0.5 * (tg[:-1] + tg[1:])
    n_occ = int(occ_idx.shape[0])

    if not schedule or coverage_epoch is None:
        pick = rng.integers(0, n_occ, size=n_cover, endpoint=False)
    else:
        # deterministic per-epoch schedule, walk a fixed permutation in
        # n_cover chunks so occupied cells are revisited, not sampled with
        # replacement noise
        perm = np.random.default_rng(int(schedule_seed)).permutation(n_occ)
        ep = max(0, int(coverage_epoch) - 1)
        start = (ep * n_cover) % n_occ
        if n_cover >= n_occ:
            full_cycles = n_cover // n_occ
            rem = n_cover % n_occ
            chunks = []
            if full_cycles > 0:
                chunks.append(np.tile(perm, full_cycles))
            if rem > 0:
                end = start + rem
                if end <= n_occ:
                    chunks.append(perm[start:end])
                else:
                    chunks.append(np.concatenate([perm[start:], perm[: end - n_occ]]))
            pick = np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.int64)
        else:
            end = start + n_cover
            if end <= n_occ:
                pick = perm[start:end]
            else:
                pick = np.concatenate([perm[start:], perm[: end - n_occ]])

    rows = occ_idx[pick]
    return np.column_stack([xc[rows[:, 0]], zc[rows[:, 1]], tc[rows[:, 2]]]).astype(np.float32)


def generate_collocation_points_density(n_points, domain, x_data, grid_resolution=50, bandwidth=0.05, 
                                       min_density_threshold=0.001, fallback_ratio=0.1, use_fast_density=True, method="auto",
                                       histogram_subsample_seed=None,
                                       strict_occupied_only=True):
    """
    Generate collocation points using density guided sampling
    
    Parameters:
        n_points: Number of collocation points to generate
        domain: Dictionary specifying the domain limits
        x_data: Array of shape (N, 3) containing (x, z, t) coordinates of microbubbles
        grid_resolution: Number of grid points per dimension
        bandwidth: Bandwidth parameter for kernel density estimation
        min_density_threshold: Minimum density threshold for sampling
        fallback_ratio: Fraction of points to sample uniformly if density sampling fails (deprecated, not used)
        use_fast_density: Whether to use fast histogram method
        method: Method for density estimation
        strict_occupied_only: If True, never sample from histogram cells with zero raw data occupancy
    
    Returns:
        colloc_points: Array of shape (n_points, 3) containing collocation points
    """
    if len(x_data) == 0:
        raise ValueError("No microbubble data available for density guided sampling")

    density_map, x_grid, z_grid, t_grid = compute_density_map(
        x_data, domain, grid_resolution, bandwidth, use_fast_density, method,
        histogram_subsample_seed=histogram_subsample_seed,
    )

    strict_occupied_only = bool(strict_occupied_only)
    raw_counts = histogram_counts_with_edges(x_data, x_grid, z_grid, t_grid)
    occupied_mask = raw_counts > 0

    # density_map is (N,N,N) grid centers, raw_counts is (N-1,N-1,N-1) bins;
    # crop density_map to match so mask ops don't fail
    if density_map.shape != raw_counts.shape:
        slices = tuple(slice(0, s) for s in raw_counts.shape)
        density_map = density_map[slices]

    max_density = np.max(density_map)
    threshold = max_density * min_density_threshold
    high_density_mask = density_map > threshold
    if strict_occupied_only:
        sample_mask = high_density_mask & occupied_mask
    else:
        sample_mask = high_density_mask

    # fallback, threshold too aggressive -> use all occupied cells
    if np.sum(sample_mask) == 0 and np.sum(occupied_mask) > 0:
        sample_mask = occupied_mask.copy()
    if np.sum(sample_mask) == 0:
        raise ValueError("No eligible density regions found. Cannot generate collocation points.")
    
    high_density_indices = np.where(sample_mask)
    density_values = density_map[high_density_indices]
    if np.sum(density_values) <= 0.0:
        # safety, density mass collapsed -> use occupancy counts
        density_values = raw_counts[high_density_indices].astype(float)
    probabilities = density_values / np.sum(density_values)
    
    # deterministic allocation, no jitter, split n_points across cells
    # proportional to density and return grid centers, so collocation follows
    # the density map strictly without per epoch pixel jitter
    probs = probabilities.astype(float)
    probs = probs / (np.sum(probs) + 1e-30)
    expected = probs * float(n_points)
    counts = np.floor(expected).astype(int)
    remaining = int(n_points - int(np.sum(counts)))
    if remaining > 0:
        # remainder -> largest fractional parts, stable, deterministic
        frac = expected - counts
        order = np.argsort(-frac)
        counts[order[:remaining]] += 1
    elif remaining < 0:
        # rare corner, remove from smallest fractional parts
        frac = expected - counts
        order = np.argsort(frac)
        take = int(min(-remaining, len(order)))
        counts[order[:take]] = np.maximum(0, counts[order[:take]] - 1)

    # repeated indices; duplicates are intended for high density cells
    reps = np.repeat(np.arange(len(counts)), counts)
    if reps.size != n_points:
        # safeguard, pad/truncate deterministically
        if reps.size < n_points:
            pad = np.arange(len(counts))
            pad = pad[: (n_points - reps.size)]
            reps = np.concatenate([reps, pad], axis=0)
        else:
            reps = reps[:n_points]

    i_indices = high_density_indices[0][reps]
    j_indices = high_density_indices[1][reps]
    k_indices = high_density_indices[2][reps]

    x_coords = x_grid[i_indices]
    z_coords = z_grid[j_indices]
    t_coords = t_grid[k_indices]

    colloc_points = np.column_stack([x_coords, z_coords, t_coords])
    return colloc_points


def generate_collocation_points_uniform(n_points, domain):
    """
    Generate uniform random collocation points within the given domain
    
    Parameters:
        n_points: Number of collocation points
        domain: Dictionary specifying the domain limits
    
    Returns:
        colloc_points: Array of shape (n_points, 3) containing (x, z, t) coordinates
    """
    x = np.random.uniform(domain['x'][0], domain['x'][1], n_points)
    z = np.random.uniform(domain['z'][0], domain['z'][1], n_points)
    t = np.random.uniform(domain['t'][0], domain['t'][1], n_points)
    return np.stack([x, z, t], axis=-1)


def generate_collocation_points(n_points, domain, x_data=None, use_density_guided=True, 
                               grid_resolution=50, bandwidth=0.05, min_density_threshold=0.001, 
                               fallback_ratio=0.1, use_fast_density=True, method="auto",
                               coverage_reserve_frac=0.0, histogram_subsample_seed=None,
                               coverage_sample_seed=None,
                               strict_occupied_only=True,
                               coverage_schedule=True,
                               coverage_schedule_seed=0,
                               coverage_epoch=None):
    """
    Generate collocation points using density guided sampling.
    
    Parameters:
        n_points: Number of collocation points
        domain: Dictionary specifying the domain limits
        x_data: Array of microbubble coordinates (required for density guided)
        use_density_guided: Whether to use density guided sampling (must be True)
        grid_resolution: Number of grid points per dimension
        bandwidth: Bandwidth parameter for kernel density estimation
        min_density_threshold: Minimum density threshold for sampling
        fallback_ratio: Fraction of points to sample uniformly (deprecated, not used)
        use_fast_density: Whether to use fast histogram method
        method: Method for density estimation
        coverage_reserve_frac: If > 0, reserve this fraction of points for uniform
            sampling over every histogram cell that contains at least one data point
        histogram_subsample_seed: Optional seed for >100k-point histogram subsampling
        coverage_sample_seed: Optional seed for occupancy/coverage draws (e.g. epoch)
        strict_occupied_only: If True, disallow collocation placement in zero occupancy histogram cells
        coverage_schedule: If True, use deterministic reserve scheduling across occupied bins
        coverage_schedule_seed: Seed used to create the fixed occupied bin permutation for scheduling
        coverage_epoch: Physics epoch index used by deterministic reserve scheduling
    
    Returns:
        colloc_points: Array of shape (n_points, 3) containing collocation points
    """
    if not use_density_guided:
        raise ValueError("Random uniform collocation points have been removed. Please use density guided sampling (use_density_guided=True).")
    
    if x_data is None or len(x_data) == 0:
        raise ValueError("x_data is required for density guided collocation point generation. No random uniform sampling is available.")

    cov = float(np.clip(coverage_reserve_frac, 0.0, 0.95))
    if cov <= 0.0:
        return generate_collocation_points_density(
            n_points, domain, x_data, grid_resolution, bandwidth,
            min_density_threshold, fallback_ratio, use_fast_density, method,
            histogram_subsample_seed=histogram_subsample_seed,
            strict_occupied_only=strict_occupied_only,
        )

    n_cov = int(round(n_points * cov))
    n_den = max(0, n_points - n_cov)
    if n_cov <= 0:
        return generate_collocation_points_density(
            n_points, domain, x_data, grid_resolution, bandwidth,
            min_density_threshold, fallback_ratio, use_fast_density, method,
            histogram_subsample_seed=histogram_subsample_seed,
            strict_occupied_only=strict_occupied_only,
        )

    if n_den > 0:
        dense_pts = generate_collocation_points_density(
            n_den, domain, x_data, grid_resolution, bandwidth,
            min_density_threshold, fallback_ratio, use_fast_density, method,
            histogram_subsample_seed=histogram_subsample_seed,
            strict_occupied_only=strict_occupied_only,
        )
    else:
        dense_pts = np.empty((0, 3), dtype=np.float32)

    counts, (xg, zg, tg) = data_histogram_counts(x_data, domain, grid_resolution)
    rng = np.random.default_rng(coverage_sample_seed)
    if coverage_epoch is None and coverage_sample_seed is not None:
        try:
            coverage_epoch = int(coverage_sample_seed)
        except Exception:
            coverage_epoch = None
    cov_pts = sample_coverage_collocation(
        n_cov,
        counts,
        xg,
        zg,
        tg,
        domain,
        rng,
        schedule=bool(coverage_schedule),
        coverage_epoch=coverage_epoch,
        schedule_seed=int(coverage_schedule_seed),
    )

    out = np.vstack([dense_pts, cov_pts])
    if out.shape[0] > n_points:
        out = out[:n_points]
    elif out.shape[0] < n_points:
        n_pad = n_points - out.shape[0]
        if strict_occupied_only:
            pad = generate_collocation_points_density(
                n_pad, domain, x_data, grid_resolution, bandwidth,
                min_density_threshold, fallback_ratio, use_fast_density, method,
                histogram_subsample_seed=histogram_subsample_seed,
                strict_occupied_only=True,
            )
        else:
            pad = generate_collocation_points_uniform(n_pad, domain)
        out = np.vstack([out, pad])
    return out


def analyze_sampling_effectiveness(x_data, colloc_points, domain, output_filename="sampling_analysis.png"):
    """
    Analyze and visualize the effectiveness of density guided sampling.
    
    Parameters:
        x_data: Original microbubble data
        colloc_points: Generated collocation points
        domain: Domain specification
        output_filename: Output filename for the analysis plot
    """
    try:
        print("\n" + "="*60)
        print("DENSITY-GUIDED SAMPLING ANALYSIS")
        print("="*60)
        
        x_data_x, x_data_z = x_data[:, 0], x_data[:, 1]
        colloc_x, colloc_z = colloc_points[:, 0], colloc_points[:, 1]

        data_x_range = np.ptp(x_data_x)
        data_z_range = np.ptp(x_data_z)
        colloc_x_range = np.ptp(colloc_x)
        colloc_z_range = np.ptp(colloc_z)
        
        print(f"Original data spatial coverage: X={data_x_range:.3f}, Z={data_z_range:.3f}")
        print(f"Collocation points spatial coverage: X={colloc_x_range:.3f}, Z={colloc_z_range:.3f}")
        
        hist_data, x_edges, z_edges = np.histogram2d(x_data_x, x_data_z, bins=20, range=[[0, 1], [0, 1]])
        hist_colloc, _, _ = np.histogram2d(colloc_x, colloc_z, bins=20, range=[[0, 1], [0, 1]])

        correlation = np.corrcoef(hist_data.flatten(), hist_colloc.flatten())[0, 1]
        print(f"Correlation between data density and collocation density: {correlation:.3f}")

        high_density_threshold = np.percentile(hist_data, 75)
        high_density_mask = hist_data > high_density_threshold

        high_density_colloc = np.sum(hist_colloc[high_density_mask])
        total_colloc = np.sum(hist_colloc)
        concentration_ratio = high_density_colloc / total_colloc if total_colloc > 0 else 0

        print(f"Collocation points in high density regions: {high_density_colloc:.0f}/{total_colloc:.0f} ({concentration_ratio:.1%})")

        high_density_area_ratio = np.sum(high_density_mask) / high_density_mask.size
        uniform_expectation = high_density_area_ratio
        efficiency_ratio = concentration_ratio / uniform_expectation if uniform_expectation > 0 else 1

        print(f"Sampling efficiency (vs uniform): {efficiency_ratio:.1f}x")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        im1 = axes[0, 0].imshow(hist_data.T, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
        axes[0, 0].set_title('Original Data Density')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Z')
        plt.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(hist_colloc.T, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
        axes[0, 1].set_title('Collocation Points Density')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Z')
        plt.colorbar(im2, ax=axes[0, 1])

        axes[1, 0].scatter(x_data_x[::100], x_data_z[::100], c='blue', alpha=0.6, s=1, label='Original Data (sampled)')
        axes[1, 0].scatter(colloc_x, colloc_z, c='red', alpha=0.8, s=10, label='Collocation Points')
        axes[1, 0].set_title('Spatial Distribution Comparison')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Z')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].axis('off')
        stats_text = f"""Sampling Analysis Results:

Data Points: {len(x_data):,}
Collocation Points: {len(colloc_points):,}

Spatial Coverage:
  Data: X={data_x_range:.3f}, Z={data_z_range:.3f}
  Colloc: X={colloc_x_range:.3f}, Z={colloc_z_range:.3f}

Density Correlation: {correlation:.3f}
High-Density Concentration: {concentration_ratio:.1%}
Sampling Efficiency: {efficiency_ratio:.1f}x

Interpretation:
• Correlation > 0.5: Good density matching
• Efficiency > 2.0x: Effective concentration
• Coverage > 0.8: Good spatial coverage"""
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nAnalysis plot saved as: {output_filename}")

        print("\nASSESSMENT:")
        if correlation > 0.5:
            print("✓ Good density correlation, collocation points follow data distribution")
        else:
            print("⚠ Low density correlation, collocation points may not match data well")
            
        if efficiency_ratio > 2.0:
            print("✓ High sampling efficiency, effectively concentrates on high density regions")
        elif efficiency_ratio > 1.5:
            print("✓ Moderate sampling efficiency, some concentration on high density regions")
        else:
            print("⚠ Low sampling efficiency, similar to uniform sampling")
            
        if colloc_x_range > 0.8 and colloc_z_range > 0.8:
            print("✓ Good spatial coverage, collocation points span the domain well")
        else:
            print("⚠ Limited spatial coverage, collocation points may miss some regions")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error in sampling analysis: {e}")
        print("Skipping analysis visualization.") 
