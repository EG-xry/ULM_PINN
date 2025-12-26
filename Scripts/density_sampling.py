"""
Density-guided collocation sampling utilities 
"""

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

def compute_density_map_fast(x_data, domain, grid_resolution=50):
    """
    Compute a density map using fast histogram-based approach for large datasets
    TFaster than KDE but less smooth

    Parameters:
        x_data: Array of shape (N, 3) containing (x, z, t) coordinates of microbubbles
        domain: Dictionary specifying the domain limits
        grid_resolution: Number of grid points per dimension for density estimation
    
    Returns:
        density_map: 3D array of density values
        x_grid, z_grid, t_grid: Grid coordinates
    """
    try:
        x_coords = x_data[:, 0]
        z_coords = x_data[:, 1]
        t_coords = x_data[:, 2]
        
        # For very large datasets, subsample for histogram computation
        max_points_for_hist = 100000
        if len(x_data) > max_points_for_hist:
            indices = np.random.choice(len(x_data), size=max_points_for_hist, replace=False)
            x_coords = x_coords[indices]
            z_coords = z_coords[indices]
            t_coords = t_coords[indices]
        
        # Create grid for density estimation
        x_grid = np.linspace(domain['x'][0], domain['x'][1], grid_resolution)
        z_grid = np.linspace(domain['z'][0], domain['z'][1], grid_resolution)
        t_grid = np.linspace(domain['t'][0], domain['t'][1], grid_resolution)
        
        # Create 3D histogram
        density_map, _ = np.histogramdd(
            [x_coords, z_coords, t_coords],
            bins=[x_grid, z_grid, t_grid],
            density=True
        )
        
        density_map = gaussian_filter(density_map, sigma=0.5)
        
        return density_map, x_grid, z_grid, t_grid
        
    except Exception as e:
        print(f"Error in compute_density_map_fast: {e}")
        print("Falling back to uniform density map.")
        # Create uniform density map as fallback
        x_grid = np.linspace(domain['x'][0], domain['x'][1], grid_resolution)
        z_grid = np.linspace(domain['z'][0], domain['z'][1], grid_resolution)
        t_grid = np.linspace(domain['t'][0], domain['t'][1], grid_resolution)
        X, Z, T = np.meshgrid(x_grid, z_grid, t_grid, indexing='ij')
        density_map = np.ones_like(X)
        density_map = density_map / np.sum(density_map)
        return density_map, x_grid, z_grid, t_grid


def compute_vessel_regions_fast(x_data, domain, grid_resolution=30, min_cluster_size=100):
    """
    Compute vessel regions using spatial clustering for large datasets
    
    Parameters:
        x_data: Array of shape (N, 3) containing (x, z, t) coordinates of microbubbles
        domain: Dictionary specifying the domain limits
        grid_resolution: Number of grid points per dimension
        min_cluster_size: Minimum size for a cluster to be considered a vessel
    
    Returns:
        vessel_mask: 3D boolean array indicating vessel regions
        x_grid, z_grid, t_grid: Grid coordinates
    """
    try:
        # Check if sklearn is available
        if not sklearn_available:
            print("Warning: scikit-learn not available. Falling back to uniform vessel mask.")
            x_grid = np.linspace(domain['x'][0], domain['x'][1], grid_resolution)
            z_grid = np.linspace(domain['z'][0], domain['z'][1], grid_resolution)
            t_grid = np.linspace(domain['t'][0], domain['t'][1], grid_resolution)
            vessel_mask = np.ones((grid_resolution, grid_resolution, grid_resolution), dtype=bool)
            return vessel_mask, x_grid, z_grid, t_grid
        
        print("Computing vessel regions using spatial clustering...")
        
        # Subsample data for clustering if too large
        max_points_for_clustering = 50000
        if len(x_data) > max_points_for_clustering:
            indices = np.random.choice(len(x_data), size=max_points_for_clustering, replace=False)
            x_data_cluster = x_data[indices]
        else:
            x_data_cluster = x_data
        
        x_grid = np.linspace(domain['x'][0], domain['x'][1], grid_resolution)
        z_grid = np.linspace(domain['z'][0], domain['z'][1], grid_resolution)
        t_grid = np.linspace(domain['t'][0], domain['t'][1], grid_resolution)
        
        # Scale 
        x_scaled = (x_data_cluster[:, 0] - domain['x'][0]) / (domain['x'][1] - domain['x'][0])
        z_scaled = (x_data_cluster[:, 1] - domain['z'][0]) / (domain['z'][1] - domain['z'][0])
        t_scaled = (x_data_cluster[:, 2] - domain['t'][0]) / (domain['t'][1] - domain['t'][0])
        
        coords_scaled = np.column_stack([x_scaled, z_scaled, t_scaled * 0.5])
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=0.05, min_samples=min_cluster_size).fit(coords_scaled)
        
        # Create vessel mask
        vessel_mask = np.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=bool)
        unique_labels = set(clustering.labels_)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        print(f"Found {len(unique_labels)} vessel clusters")
        
        # Create KD-tree for efficient queries
        tree = cKDTree(x_data_cluster)
        
        # Mark grid points near clustered points
        X, Z, T = np.meshgrid(x_grid, z_grid, t_grid, indexing='ij')
        grid_points = np.column_stack([X.ravel(), Z.ravel(), T.ravel()])
        
        for i in range(0, len(grid_points), 10000):
            end_idx = min(i + 10000, len(grid_points))
            batch_points = grid_points[i:end_idx]
            distances, _ = tree.query(batch_points, k=1)
            vessel_mask.ravel()[i:end_idx] = distances < 0.05
        
        # Apply morphological operations to smooth
        from scipy.ndimage import binary_dilation, binary_erosion
        vessel_mask = binary_dilation(vessel_mask, iterations=1)
        vessel_mask = binary_erosion(vessel_mask, iterations=1)
        
        return vessel_mask, x_grid, z_grid, t_grid
        
    except Exception as e:
        print(f"Error in compute_vessel_regions_fast: {e}")
        print("Falling back to uniform vessel mask.")
        x_grid = np.linspace(domain['x'][0], domain['x'][1], grid_resolution)
        z_grid = np.linspace(domain['z'][0], domain['z'][1], grid_resolution)
        t_grid = np.linspace(domain['t'][0], domain['t'][1], grid_resolution)
        vessel_mask = np.ones((grid_resolution, grid_resolution, grid_resolution), dtype=bool)
        return vessel_mask, x_grid, z_grid, t_grid


def compute_density_map(x_data, domain, grid_resolution=50, bandwidth=0.05, use_fast_density=True, method="auto"):
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
    # Use fast histogram method for large datasets
    if method == "histogram" or (use_fast_density and len(x_data) > 50000):
        return compute_density_map_fast(x_data, domain, grid_resolution)
    
    # Use KDE method for smaller datasets
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
        
        # KDE computation
        coords = np.vstack([x_coords_kde, z_coords_kde, t_coords_kde])
        
        try:
            kde = gaussian_kde(coords, bw_method=bandwidth)
            grid_points = np.vstack([X.ravel(), Z.ravel(), T.ravel()])
            
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
        
        # Normalize
        density_map = density_map / (np.sum(density_map) + 1e-10)
        
        return density_map, x_grid, z_grid, t_grid
        
    except Exception as e:
        print(f"Error in compute_density_map: {e}")
        print("Falling back to uniform density map.")
        x_grid = np.linspace(domain['x'][0], domain['x'][1], grid_resolution)
        z_grid = np.linspace(domain['z'][0], domain['z'][1], grid_resolution)
        t_grid = np.linspace(domain['t'][0], domain['t'][1], grid_resolution)
        X, Z, T = np.meshgrid(x_grid, z_grid, t_grid, indexing='ij')
        density_map = np.ones_like(X)
        density_map = density_map / np.sum(density_map)
        return density_map, x_grid, z_grid, t_grid


def generate_collocation_points_density(n_points, domain, x_data, grid_resolution=50, bandwidth=0.05, min_density_threshold=0.001, fallback_ratio=0.1, use_fast_density=True, method="auto"):
    """
    Generate collocation points using density-guided sampling
    
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
    
    Returns:
        colloc_points: Array of shape (n_points, 3) containing collocation points
    """
    if len(x_data) == 0:
        raise ValueError("No microbubble data available for density-guided sampling. Please provide x_data.")
    
    # Compute density map
    density_map, x_grid, z_grid, t_grid = compute_density_map(x_data, domain, grid_resolution, bandwidth, use_fast_density, method)
    
    # Find high-density regions
    max_density = np.max(density_map)
    threshold = max_density * min_density_threshold
    high_density_mask = density_map > threshold
    
    if np.sum(high_density_mask) == 0:
        raise ValueError("No high-density regions found. Cannot generate density-guided collocation points.")
    
    # Sample all points from high-density regions (no fallback uniform points)
    high_density_indices = np.where(high_density_mask)
    density_values = density_map[high_density_indices]
    probabilities = density_values / np.sum(density_values)
    
    # Deterministic density allocation:
    # Allocate exactly n_points to grid cells proportional to density, then return the corresponding grid centers
    probs = probabilities.astype(float)
    probs = probs / (np.sum(probs) + 1e-30)
    expected = probs * float(n_points)
    counts = np.floor(expected).astype(int)
    remaining = int(n_points - int(np.sum(counts)))
    if remaining > 0:
        # Distribute remainder to largest fractional parts (stable + deterministic)
        frac = expected - counts
        order = np.argsort(-frac)
        counts[order[:remaining]] += 1
    elif remaining < 0:
        # Rare numerical corner: remove from smallest fractional parts
        frac = expected - counts
        order = np.argsort(frac)
        take = int(min(-remaining, len(order)))
        counts[order[:take]] = np.maximum(0, counts[order[:take]] - 1)

    # Build repeated indices (may include duplicates, which is intended when a cell has high density)
    reps = np.repeat(np.arange(len(counts)), counts)
    if reps.size != n_points:
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


def generate_collocation_points(n_points, domain, x_data=None, use_density_guided=True, grid_resolution=50, bandwidth=0.05, min_density_threshold=0.001, fallback_ratio=0.1, use_fast_density=True, method="auto"):
    """
    Generate collocation points using density-guided sampling.
    
    Parameters:
        n_points: Number of collocation points
        domain: Dictionary specifying the domain limits
        x_data: Array of microbubble coordinates (required for density-guided)
        use_density_guided: Whether to use density-guided sampling (must be True)
        grid_resolution: Number of grid points per dimension
        bandwidth: Bandwidth parameter for kernel density estimation
        min_density_threshold: Minimum density threshold for sampling
        fallback_ratio: Fraction of points to sample uniformly (deprecated, not used)
        use_fast_density: Whether to use fast histogram method
        method: Method for density estimation
    
    Returns:
        colloc_points: Array of shape (n_points, 3) containing collocation points
    """
    if not use_density_guided:
        raise ValueError("Random uniform collocation points have been removed. Please use density-guided sampling (use_density_guided=True).")
    
    if x_data is None or len(x_data) == 0:
        raise ValueError("x_data is required for density-guided collocation point generation. No random uniform sampling is available.")
    
    return generate_collocation_points_density(n_points, domain, x_data, 
                                             grid_resolution, bandwidth, 
                                             min_density_threshold, fallback_ratio, 
                                             use_fast_density, method)


def analyze_sampling_effectiveness(x_data, colloc_points, domain, output_filename="sampling_analysis.png"):
    """
    Analyze and visualize the effectiveness of density-guided sampling.
    
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
        
        # Spatial coverage analysis
        x_data_x, x_data_z = x_data[:, 0], x_data[:, 1]
        colloc_x, colloc_z = colloc_points[:, 0], colloc_points[:, 1]
        
        # Calculate spatial distribution statistics
        data_x_range = np.ptp(x_data_x)
        data_z_range = np.ptp(x_data_z)
        colloc_x_range = np.ptp(colloc_x)
        colloc_z_range = np.ptp(colloc_z)
        
        print(f"Original data spatial coverage: X={data_x_range:.3f}, Z={data_z_range:.3f}")
        print(f"Collocation points spatial coverage: X={colloc_x_range:.3f}, Z={colloc_z_range:.3f}")
        
        # Density-based analysis
        hist_data, x_edges, z_edges = np.histogram2d(x_data_x, x_data_z, bins=20, range=[[0, 1], [0, 1]])
        hist_colloc, _, _ = np.histogram2d(colloc_x, colloc_z, bins=20, range=[[0, 1], [0, 1]])
        
        # Calculate correlation
        correlation = np.corrcoef(hist_data.flatten(), hist_colloc.flatten())[0, 1]
        print(f"Correlation between data density and collocation density: {correlation:.3f}")
        
        # Concentration analysis
        high_density_threshold = np.percentile(hist_data, 75)
        high_density_mask = hist_data > high_density_threshold
        
        high_density_colloc = np.sum(hist_colloc[high_density_mask])
        total_colloc = np.sum(hist_colloc)
        concentration_ratio = high_density_colloc / total_colloc if total_colloc > 0 else 0
        
        print(f"Collocation points in high-density regions: {high_density_colloc:.0f}/{total_colloc:.0f} ({concentration_ratio:.1%})")
        
        # Efficiency metric
        high_density_area_ratio = np.sum(high_density_mask) / high_density_mask.size
        uniform_expectation = high_density_area_ratio
        efficiency_ratio = concentration_ratio / uniform_expectation if uniform_expectation > 0 else 1
        
        print(f"Sampling efficiency (vs uniform): {efficiency_ratio:.1f}x")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original data density
        im1 = axes[0, 0].imshow(hist_data.T, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
        axes[0, 0].set_title('Original Data Density')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Z')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Collocation points density
        im2 = axes[0, 1].imshow(hist_colloc.T, extent=[0, 1, 0, 1], origin='lower', cmap='hot')
        axes[0, 1].set_title('Collocation Points Density')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Z')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Scatter plot comparison
        axes[1, 0].scatter(x_data_x[::100], x_data_z[::100], c='blue', alpha=0.6, s=1, label='Original Data (sampled)')
        axes[1, 0].scatter(colloc_x, colloc_z, c='red', alpha=0.8, s=10, label='Collocation Points')
        axes[1, 0].set_title('Spatial Distribution Comparison')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Z')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistics summary
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
        
        # Assessment
        print("\nASSESSMENT:")
        if correlation > 0.5:
            print("Good density correlation - collocation points follow data distribution")
        else:
            print("Low density correlation - collocation points may not match data well")
            
        if efficiency_ratio > 2.0:
            print("High sampling efficiency - effectively concentrates on high-density regions")
        elif efficiency_ratio > 1.5:
            print("Moderate sampling efficiency - some concentration on high-density regions")
        else:
            print("Low sampling efficiency - similar to uniform sampling")
            
        if colloc_x_range > 0.8 and colloc_z_range > 0.8:
            print("Good spatial coverage - collocation points span the domain well")
        else:
            print("Limited spatial coverage - collocation points may miss some regions")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error in sampling analysis: {e}")
        print("Skipping analysis visualization.") 