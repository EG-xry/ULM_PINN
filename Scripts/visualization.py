"""
Visualization utilities 
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(data_loss, phys_loss, total_loss, output_filename="loss_curve.png"):
    """
    Plot loss histories on a symmetric log scale and save as PNG.
    
    Parameters:
        data_loss: History of data loss values
        phys_loss: History of physics loss values
        total_loss: History of total loss values
        output_filename: Output filename for the plot
    """
    epochs = range(1, len(total_loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data_loss, label="Data Loss", color="blue")
    plt.plot(epochs, phys_loss, label="Physics Loss", color="red")
    plt.plot(epochs, total_loss, label="Total Loss", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History Over Epochs")
    
    # Use symmetric log scale
    plt.yscale('symlog', linthresh=1e-4)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(output_filename)
    plt.close()
    print(f"Loss curve saved as {output_filename}")


def plot_density_sampling(x_data, colloc_points, density_map, x_grid, z_grid, t_grid, output_filename="density_sampling.png", time_slice=None):
    """
    Visualize density-guided sampling by plotting microbubble data, density map, and sampled collocation points.
    
    Parameters:
        x_data: Array of shape (N, 3) containing microbubble coordinates
        colloc_points: Array of shape (M, 3) containing sampled collocation points
        density_map: 3D array of density values
        x_grid, z_grid, t_grid: Grid coordinates
        output_filename: Output filename for the plot
        time_slice: Time slice to plot (if None, will plot middle slice)
    """
    try:
        if time_slice is None:
            time_slice = len(t_grid) // 2
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Microbubble data
        axes[0].scatter(x_data[:, 0], x_data[:, 1], c=x_data[:, 2], cmap='viridis', 
                       alpha=0.6, s=1, label='Microbubbles')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Z')
        axes[0].set_title('Microbubble Data (colored by time)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Density map at time slice
        density_slice = density_map[:, :, time_slice]
        im = axes[1].imshow(density_slice.T, extent=[x_grid[0], x_grid[-1], z_grid[0], z_grid[-1]], 
                           origin='lower', cmap='hot', aspect='auto')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')
        axes[1].set_title(f'Density Map (t={t_grid[time_slice]:.2f})')
        plt.colorbar(im, ax=axes[1], label='Density')
        
        # Plot 3: Sampled collocation points
        axes[2].scatter(colloc_points[:, 0], colloc_points[:, 1], c=colloc_points[:, 2], 
                       cmap='viridis', alpha=0.6, s=2, label='Collocation Points')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Z')
        axes[2].set_title('Sampled Collocation Points (colored by time)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Density sampling visualization saved as {output_filename}")
        
        # Print statistics
        print(f"\nDensity Sampling Statistics:")
        print(f"  Microbubble data points: {len(x_data)}")
        print(f"  Sampled collocation points: {len(colloc_points)}")
        print(f"  Density map shape: {density_map.shape}")
        print(f"  Grid resolution: {len(x_grid)} x {len(z_grid)} x {len(t_grid)}")
        print(f"  Max density: {np.max(density_map):.6f}")
        print(f"  Min density: {np.min(density_map):.6f}")
        print(f"  Mean density: {np.mean(density_map):.6f}")
        
    except Exception as e:
        print(f"Error in plot_density_sampling: {e}")
        print("Skipping density visualization plot.")
        # Try to close any open figures
        try:
            plt.close('all')
        except:
            pass


def calculate_track_statistics(tracks, stage_name="tracks"):
    """
    Calculate and print track length statistics.
    
    Parameters:
        tracks: List of tracks (each track is a list of tuples (x, z, t))
        stage_name: String identifier for the stage (e.g., "raw", "interpolated")
    
    Returns:
        stats: Dictionary containing track length statistics
    """
    if not tracks:
        print(f"{stage_name.capitalize()} track statistics: No tracks found")
        return None
    
    # Calculate track lengths
    track_lengths = [len(track) for track in tracks]
    
    # Calculate statistics
    stats = {
        'count': len(tracks),
        'min_length': min(track_lengths),
        'max_length': max(track_lengths),
        'mean_length': np.mean(track_lengths),
        'median_length': np.median(track_lengths),
        'std_length': np.std(track_lengths),
        'total_points': sum(track_lengths)
    }
    
    # Print statistics
    print(f"\n{stage_name.capitalize()} Track Statistics:")
    print(f"  Number of tracks: {stats['count']}")
    print(f"  Total points: {stats['total_points']}")
    print(f"  Track lengths - Min: {stats['min_length']}, Max: {stats['max_length']}")
    print(f"  Track lengths - Mean: {stats['mean_length']:.2f}, Median: {stats['median_length']:.2f}")
    print(f"  Track lengths - Std: {stats['std_length']:.2f}")
    
    return stats


def calculate_velocity_statistics(tracks, stage_name="tracks", orig_bounds=None):
    """
    Calculate and print velocity statistics from tracks.
    
    Parameters:
        tracks: List of tracks (each track is a list of tuples (x, z, t))
        stage_name: String identifier for the stage (e.g., "raw", "interpolated")
        orig_bounds: Original bounds for coordinate scaling (optional)
    
    Returns:
        velocity_stats: Dictionary containing velocity statistics
    """
    if not tracks:
        print(f"{stage_name.capitalize()} velocity statistics: No tracks found")
        return None
    
    print(f"\n{stage_name.capitalize()} Velocity Statistics:")
    
    # Collect all velocities from all tracks
    all_velocities_x = []
    all_velocities_z = []
    all_speeds = []
    
    for track in tracks:
        if len(track) < 2:
            continue  # Need at least 2 points to compute velocity
            
        track_array = np.array(track)
        x_coords = track_array[:, 0]
        z_coords = track_array[:, 1]
        t_coords = track_array[:, 2]
        
        # Vectorized velocity computation
        if len(track) >= 2:
            # Check for duplicate or very close time values
            dt_min = np.min(np.diff(t_coords)) if len(track) > 1 else 1.0
            if dt_min < 1e-8 or np.any(np.diff(t_coords) == 0):
                # Fall back to manual differences
                vx = np.zeros(len(track))
                vz = np.zeros(len(track))
                if len(track) > 2:
                    dt = t_coords[2:] - t_coords[:-2]
                    valid_dt = np.abs(dt) > 1e-8
                    vx[1:-1][valid_dt] = ((x_coords[2:] - x_coords[:-2]) / dt)[valid_dt]
                    vz[1:-1][valid_dt] = ((z_coords[2:] - z_coords[:-2]) / dt)[valid_dt]
                if len(track) > 1:
                    if np.abs(t_coords[1] - t_coords[0]) > 1e-8:
                        vx[0] = (x_coords[1] - x_coords[0]) / (t_coords[1] - t_coords[0])
                        vz[0] = (z_coords[1] - z_coords[0]) / (t_coords[1] - t_coords[0])
                    if np.abs(t_coords[-1] - t_coords[-2]) > 1e-8:
                        vx[-1] = (x_coords[-1] - x_coords[-2]) / (t_coords[-1] - t_coords[-2])
                        vz[-1] = (z_coords[-1] - z_coords[-2]) / (t_coords[-1] - t_coords[-2])
            else:
                # Use numpy.gradient for efficient computation
                vx = np.gradient(x_coords, t_coords, edge_order=2)
                vz = np.gradient(z_coords, t_coords, edge_order=2)
            
            # Handle any potential infinities or NaNs
            vx = np.nan_to_num(vx, nan=0.0, posinf=0.0, neginf=0.0)
            vz = np.nan_to_num(vz, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Single point track - zero velocity
            vx = np.zeros(len(track))
            vz = np.zeros(len(track))
        
        # Add to collections
        all_velocities_x.extend(vx)
        all_velocities_z.extend(vz)
        all_speeds.extend(np.sqrt(vx**2 + vz**2))
    
    # Convert to arrays for statistics
    all_velocities_x = np.array(all_velocities_x)
    all_velocities_z = np.array(all_velocities_z)
    all_speeds = np.array(all_speeds)
    
    # Remove any infinite or NaN values
    valid_mask = np.isfinite(all_velocities_x) & np.isfinite(all_velocities_z) & np.isfinite(all_speeds)
    all_velocities_x = all_velocities_x[valid_mask]
    all_velocities_z = all_velocities_z[valid_mask]
    all_speeds = all_speeds[valid_mask]
    
    if len(all_velocities_x) == 0:
        print("  No valid velocities found")
        return None
    
    # Calculate statistics
    velocity_stats = {
        'count': len(all_velocities_x),
        'vx_min': np.min(all_velocities_x),
        'vx_max': np.max(all_velocities_x),
        'vx_mean': np.mean(all_velocities_x),
        'vx_std': np.std(all_velocities_x),
        'vz_min': np.min(all_velocities_z),
        'vz_max': np.max(all_velocities_z),
        'vz_mean': np.mean(all_velocities_z),
        'vz_std': np.std(all_velocities_z),
        'speed_min': np.min(all_speeds),
        'speed_max': np.max(all_speeds),
        'speed_mean': np.mean(all_speeds),
        'speed_std': np.std(all_speeds),
        'speed_p50': np.percentile(all_speeds, 50),
        'speed_p95': np.percentile(all_speeds, 95),
        'speed_p99': np.percentile(all_speeds, 99)
    }
    
    # Print velocity statistics
    print(f"  Valid velocity points: {velocity_stats['count']:,}")
    print(f"  Velocity X (horizontal):")
    print(f"    Range: [{velocity_stats['vx_min']:.6f}, {velocity_stats['vx_max']:.6f}]")
    print(f"    Mean ± Std: {velocity_stats['vx_mean']:.6f} ± {velocity_stats['vx_std']:.6f}")
    print(f"  Velocity Z (vertical):")
    print(f"    Range: [{velocity_stats['vz_min']:.6f}, {velocity_stats['vz_max']:.6f}]")
    print(f"    Mean ± Std: {velocity_stats['vz_mean']:.6f} ± {velocity_stats['vz_std']:.6f}")
    print(f"  Speed (magnitude):")
    print(f"    Range: [{velocity_stats['speed_min']:.6f}, {velocity_stats['speed_max']:.6f}]")
    print(f"    Mean ± Std: {velocity_stats['speed_mean']:.6f} ± {velocity_stats['speed_std']:.6f}")
    print(f"    Percentiles: P50={velocity_stats['speed_p50']:.6f}, P95={velocity_stats['speed_p95']:.6f}, P99={velocity_stats['speed_p99']:.6f}")
    
    # If we have original bounds, convert to physical units
    if orig_bounds:
        print(f"\n  Physical Units (scaled by coordinate bounds):")
        x_scale = orig_bounds['X_max'] - orig_bounds['X_min']
        z_scale = orig_bounds['Z_max'] - orig_bounds['Z_min']
        t_scale = orig_bounds['T_max'] - orig_bounds['T_min']
        
        # Convert velocities to physical units
        vx_phys_mean = velocity_stats['vx_mean'] * x_scale / t_scale
        vz_phys_mean = velocity_stats['vz_mean'] * z_scale / t_scale
        speed_phys_mean = velocity_stats['speed_mean'] * np.sqrt(x_scale**2 + z_scale**2) / t_scale
        speed_phys_max = velocity_stats['speed_max'] * np.sqrt(x_scale**2 + z_scale**2) / t_scale
        
        print(f"    Mean velocity X: {vx_phys_mean:.6f} units/frame")
        print(f"    Mean velocity Z: {vz_phys_mean:.6f} units/frame")
        print(f"    Mean speed: {speed_phys_mean:.6f} units/frame")
        print(f"    Max speed: {speed_phys_max:.6f} units/frame")
        print(f"    Coordinate scaling: X={x_scale:.1f}, Z={z_scale:.1f}, T={t_scale:.1f}")
    
    return velocity_stats


