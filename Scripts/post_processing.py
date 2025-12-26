"""
Post-processing utilities 
"""

import gc
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import savemat
from scipy.signal import savgol_filter
from tqdm import tqdm

def interpolate_tracks(tracks, interp_factor=None, smooth_factor=20, res=10.0, max_linking_distance=0.05, savgol_polyorder=2, dtype=np.float32, batch_size=1000, gc_frequency=50, use_fast_mode=True):
    """
    Ultra-optimized track interpolation with configurable performance options.
    
    Parameters:
        tracks: List of tracks to interpolate
        interp_factor: Step size for interpolation in track index space. If None,
                       the value defaults to 1 / (max_linking_distance * res) * 0.8.
                       Values <= 0 disable interpolation and return the original tracks.
        smooth_factor: Smoothing window size
        res: Resolution factor
        max_linking_distance: Maximum linking distance
        savgol_polyorder: Polynomial order for Savitzky-Golay filter
        dtype: Data type for intermediate arrays
        batch_size: Number of tracks to process before garbage collection
        gc_frequency: Frequency of garbage collection
        use_fast_mode: Skip some optimizations for speed
        
    Returns:
        interp_tracks: List of interpolated tracks
    """
    # Calculate interpolation factor
    if interp_factor is None:
        interp_factor = 1.0 / (max_linking_distance * res) * 0.8
        print(f"Using default interpolation factor computed from res/max_linking_distance: {interp_factor:.6f}")
    elif interp_factor <= 0:
        print("Interpolation disabled (interp_factor <= 0). Returning original tracks without modification.")
        return tracks

    interp_tracks = []
    
    print(f"Interpolating {len(tracks)} tracks with batch_size={batch_size}, gc_frequency={gc_frequency}, fast_mode={use_fast_mode}")
    print(f"Interpolation factor: {interp_factor:.6f}")
    
    # Prefilter tracks to avoid processing very short ones
    valid_tracks = []
    for tr in tracks:
        if len(tr) >= 2:
            valid_tracks.append(tr)
    
    print(f"Processing {len(valid_tracks)} valid tracks (filtered from {len(tracks)})")
    
    # Process tracks in larger batches for better performance
    for batch_start in tqdm(range(0, len(valid_tracks), batch_size), desc='Interpolating tracks (batch)'):
        batch_end = min(batch_start + batch_size, len(valid_tracks))
        batch_tracks = valid_tracks[batch_start:batch_end]
        
        # Pre-allocate arrays for batch processing
        batch_interp_tracks = []
        
        for i, tr in enumerate(batch_tracks):
            if len(tr) < 3:
                batch_interp_tracks.append(tr)
                continue

            arr = np.asarray(tr, dtype=dtype)
            x, z, t = arr[:, 0], arr[:, 1], arr[:, 2]

            # Calculate interpolation indices
            new_idx = np.arange(1, len(x) + 1, interp_factor, dtype=dtype)
            base_idx = np.arange(1, len(x) + 1, dtype=dtype)

            # Apply smoothing
            if use_fast_mode or len(x) < 10:
                x_s, z_s = x, z
            else:
                win = min(smooth_factor, len(x) - (len(x) + 1) % 2)
                if win >= 3:
                    if win % 2 == 0:
                        win -= 1
                    x_s = savgol_filter(x, win, savgol_polyorder).astype(dtype)
                    z_s = savgol_filter(z, win, savgol_polyorder).astype(dtype)
                else:
                    x_s, z_s = x, z

            # Interpolate using vectorized operations
            try:
                interp_x = interp1d(base_idx, x_s, kind='linear',
                                    bounds_error=False, fill_value='extrapolate',
                                    assume_sorted=True)(new_idx)
                interp_z = interp1d(base_idx, z_s, kind='linear',
                                    bounds_error=False, fill_value='extrapolate',
                                    assume_sorted=True)(new_idx)
                interp_t = interp1d(base_idx, t, kind='linear',
                                    bounds_error=False, fill_value='extrapolate',
                                    assume_sorted=True)(new_idx)

                # Convert to track format 
                if use_fast_mode:
                    track = [(float(a), float(b), float(c)) 
                            for a, b, c in zip(interp_x, interp_z, interp_t)]
                else:
                    # More robust conversion
                    track = []
                    for a, b, c in zip(interp_x, interp_z, interp_t):
                        if np.isfinite(a) and np.isfinite(b) and np.isfinite(c):
                            track.append((float(a), float(b), float(c)))
                
                batch_interp_tracks.append(track)
                
            except Exception as e:
                # Fallback: use original track
                print(f"Warning: Interpolation failed for track {i}, using original: {e}")
                batch_interp_tracks.append(tr)

            # Cleanup intermediate variables
            if not use_fast_mode and i % gc_frequency == 0:
                del arr, x, z, t, x_s, z_s, interp_x, interp_z, interp_t
        
        # Add batch results
        interp_tracks.extend(batch_interp_tracks)
        
        if batch_start % (batch_size * gc_frequency) == 0:
            gc.collect()
    
    print(f"Interpolation complete: {len(interp_tracks)} tracks processed")
    return interp_tracks


def unnormalize_tracks(tracks, orig_bounds, batch_size=5000, gc_frequency=20, use_vectorized=True):
    """
    Ultra-optimized track unnormalization with vectorized operations.
    
    Parameters:
        tracks: List of tracks to unnormalize
        orig_bounds: Original coordinate bounds
        batch_size: Number of tracks to process before garbage collection
        gc_frequency: Frequency of garbage collection
        use_vectorized: Use vectorized operations for speed
        
    Returns:
        unnorm_tracks: List of unnormalized tracks
    """
    print(f"Unnormalizing {len(tracks)} tracks with batch_size={batch_size}, vectorized={use_vectorized}")
    
    # Pre-calculate scaling factors
    x_scale = orig_bounds['X_max'] - orig_bounds['X_min']
    z_scale = orig_bounds['Z_max'] - orig_bounds['Z_min']
    t_scale = orig_bounds['T_max'] - orig_bounds['T_min']
    
    # Vectorized transformation function
    def inv(pt):
        return (
            pt[0] * x_scale + orig_bounds['X_min'],
            pt[1] * z_scale + orig_bounds['Z_min'],
            pt[2] * t_scale + orig_bounds['T_min']
        )
    
    unnorm_tracks = []
    
    # Process tracks in batches for memory efficiency
    for batch_start in tqdm(range(0, len(tracks), batch_size), desc='Unnormalizing tracks (batch)'):
        batch_end = min(batch_start + batch_size, len(tracks))
        batch_tracks = tracks[batch_start:batch_end]
        
        # Pre-allocate batch results
        batch_unnorm_tracks = []
        
        for track in batch_tracks:
            if use_vectorized and len(track) > 10:
                # Vectorized processing for longer tracks
                try:
                    # Convert to numpy array for vectorized operations
                    track_array = np.array(track, dtype=np.float32)
                    
                    # Apply scaling vectorized
                    unnorm_x = track_array[:, 0] * x_scale + orig_bounds['X_min']
                    unnorm_z = track_array[:, 1] * z_scale + orig_bounds['Z_min']
                    unnorm_t = track_array[:, 2] * t_scale + orig_bounds['T_min']
                    
                    # Convert back to track format
                    unnorm_track = [(float(x), float(z), float(t)) 
                                   for x, z, t in zip(unnorm_x, unnorm_z, unnorm_t)]
                    batch_unnorm_tracks.append(unnorm_track)
                    
                except Exception as e:
                    # Fallback to element wise processing
                    print(f"Warning: Vectorized unnormalization failed, using fallback: {e}")
                    unnorm_track = [inv(pt) for pt in track]
                    batch_unnorm_tracks.append(unnorm_track)
            else:
                # Element-wise processing for shorter tracks or when vectorized fails
                unnorm_track = [inv(pt) for pt in track]
                batch_unnorm_tracks.append(unnorm_track)
        
        # Add batch results
        unnorm_tracks.extend(batch_unnorm_tracks)
        
        # Garbage collection after each batch
        if batch_start % (batch_size * gc_frequency) == 0:
            gc.collect()
    
    print(f"Unnormalization complete: {len(unnorm_tracks)} tracks processed")
    return unnorm_tracks


def save_tracks_mat(unnorm_tracks, output_matfile, min_length=15):
    """
    Save tracks to .mat file compatible with MATLAB ULM_tracking2D.
    
    Parameters:
        unnorm_tracks: List of unnormalized tracks
        output_matfile: Path to output .mat file
        min_length: Minimum track length (kept for compatibility)
        
    This function converts tracks to MATLAB format with columns:
    [x, z, vx, vz, timeline] to match MATLAB ULM_tracking2D velocityinterp mode.
    """
    #Tracks are already filtered at tracking stage
    filtered = unnorm_tracks
    
    if not filtered:
        print("save_tracks_mat: no tracks to save.")
        return

    mat_tracks = []
    for trk in filtered:
        # Convert to array
        arr = np.array(trk, dtype=np.float32)
        x_arr = arr[:, 0]
        z_arr = arr[:, 1]
        t_arr = arr[:, 2]
        N = len(arr)
        
        # Initialize velocity components
        vz = np.zeros(N, dtype=np.float32)
        vx = np.zeros(N, dtype=np.float32)
        timeline = np.zeros(N, dtype=np.float32)

        if N > 1:
            # Vectorized velocity computation
            dt_min = np.min(np.diff(t_arr)) if N > 1 else 1.0
            if dt_min < 1e-8 or np.any(np.diff(t_arr) == 0):
                # Fall back to manual differences
                vx = np.zeros(N, dtype=np.float32)
                vz = np.zeros(N, dtype=np.float32)
                if N > 2:
                    dt = t_arr[2:] - t_arr[:-2]
                    valid_dt = np.abs(dt) > 1e-8
                    vx[1:-1][valid_dt] = ((x_arr[2:] - x_arr[:-2]) / dt)[valid_dt]
                    vz[1:-1][valid_dt] = ((z_arr[2:] - z_arr[:-2]) / dt)[valid_dt]
                if N > 1:
                    if np.abs(t_arr[1] - t_arr[0]) > 1e-8:
                        vx[0] = (x_arr[1] - x_arr[0]) / (t_arr[1] - t_arr[0])
                        vz[0] = (z_arr[1] - z_arr[0]) / (t_arr[1] - t_arr[0])
                    if np.abs(t_arr[-1] - t_arr[-2]) > 1e-8:
                        vx[-1] = (x_arr[-1] - x_arr[-2]) / (t_arr[-1] - t_arr[-2])
                        vz[-1] = (z_arr[-1] - z_arr[-2]) / (t_arr[-1] - t_arr[-2])
            else:
                # Use numpy.gradient
                vx = np.gradient(x_arr, t_arr, edge_order=2).astype(np.float32)
                vz = np.gradient(z_arr, t_arr, edge_order=2).astype(np.float32)
            
            # Handle infinities or NaNs
            vx = np.nan_to_num(vx, nan=0.0, posinf=0.0, neginf=0.0)
            vz = np.nan_to_num(vz, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Timeline
            timeline = t_arr.astype(np.float32)
        else:
            timeline = t_arr

        # Column order: [x, z, vx, vz, timeline]
        mat_arr = np.column_stack([x_arr, z_arr, vx, vz, timeline])
        mat_tracks.append(mat_arr)

    # Split into three roughly equal parts
    total = len(mat_tracks)
    if total == 0:
        print("save_tracks_mat: no tracks to save.")
        return

    cuts = np.round(np.linspace(0, total, 4)).astype(int)
    Track_tot_1 = mat_tracks[cuts[0]:cuts[1]]
    Track_tot_2 = mat_tracks[cuts[1]:cuts[2]]
    Track_tot_3 = mat_tracks[cuts[2]:cuts[3]]

    # Save to .mat file
    savemat(output_matfile, {
        'Track_tot_1': Track_tot_1,
        'Track_tot_2': Track_tot_2,
        'Track_tot_3': Track_tot_3
    }, do_compression=False)

    print(f"Saved {len(Track_tot_1)} + {len(Track_tot_2)} + {len(Track_tot_3)} tracks to {output_matfile}")
    print(f"Track format: [x, z, vx, vz, timeline] - matches MATLAB ULM_tracking2D velocityinterp mode")


def save_tracks_mat_complete(unnorm_tracks, output_matfile, min_length=15, processing_time=None, ulm_params=None):
    """
    Save tracks with complete metadata to match original MATLAB ULM_tracking2D output format.
    
    Parameters:
        unnorm_tracks: List of unnormalized tracks
        output_matfile: Path to output .mat file
        min_length: Minimum track length (kept for compatibility)
        processing_time: Processing time in seconds
        ulm_params: ULM parameters structure
    """
    filtered = unnorm_tracks
    if not filtered:
        print("save_tracks_mat_complete: no tracks to save.")
        return

    mat_tracks = []
    for trk in filtered:
        arr = np.array(trk, dtype=np.float32)
        x_arr = arr[:, 0]
        z_arr = arr[:, 1]
        t_arr = arr[:, 2]
        N = len(arr)
        
        # Initialize velocity components
        vz = np.zeros(N, dtype=np.float32)
        vx = np.zeros(N, dtype=np.float32)
        timeline = np.zeros(N, dtype=np.float32)

        if N > 1:
            # Vectorized velocity computation
            dt_min = np.min(np.diff(t_arr)) if N > 1 else 1.0
            if dt_min < 1e-8 or np.any(np.diff(t_arr) == 0):
                # Fall back to manual differences
                vx = np.zeros(N, dtype=np.float32)
                vz = np.zeros(N, dtype=np.float32)
                if N > 2:
                    dt = t_arr[2:] - t_arr[:-2]
                    valid_dt = np.abs(dt) > 1e-8
                    vx[1:-1][valid_dt] = ((x_arr[2:] - x_arr[:-2]) / dt)[valid_dt]
                    vz[1:-1][valid_dt] = ((z_arr[2:] - z_arr[:-2]) / dt)[valid_dt]
                if N > 1:
                    if np.abs(t_arr[1] - t_arr[0]) > 1e-8:
                        vx[0] = (x_arr[1] - x_arr[0]) / (t_arr[1] - t_arr[0])
                        vz[0] = (z_arr[1] - z_arr[0]) / (t_arr[1] - t_arr[0])
                    if np.abs(t_arr[-1] - t_arr[-2]) > 1e-8:
                        vx[-1] = (x_arr[-1] - x_arr[-2]) / (t_arr[-1] - t_arr[-2])
                        vz[-1] = (z_arr[-1] - z_arr[-2]) / (t_arr[-1] - t_arr[-2])
            else:
                # Use numpy.gradient
                vx = np.gradient(x_arr, t_arr, edge_order=2).astype(np.float32)
                vz = np.gradient(z_arr, t_arr, edge_order=2).astype(np.float32)
            
            # Handle infinities or NaNs
            vx = np.nan_to_num(vx, nan=0.0, posinf=0.0, neginf=0.0)
            vz = np.nan_to_num(vz, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Timeline
            timeline = t_arr.astype(np.float32)
        else:
            timeline = t_arr

        # Column order: [x, z, vx, vz, timeline]
        mat_arr = np.column_stack([x_arr, z_arr, vx, vz, timeline])
        mat_tracks.append(mat_arr)

    # Split into three roughly equal parts
    total = len(mat_tracks)
    if total == 0:
        print("save_tracks_mat_complete: no tracks to save.")
        return

    cuts = np.round(np.linspace(0, total, 4)).astype(int)
    Track_tot_1 = mat_tracks[cuts[0]:cuts[1]]
    Track_tot_2 = mat_tracks[cuts[1]:cuts[2]]
    Track_tot_3 = mat_tracks[cuts[2]:cuts[3]]
    
    # Convert to vertical cell arrays
    Track_tot_1 = np.array(Track_tot_1, dtype=object).reshape(-1, 1)
    Track_tot_2 = np.array(Track_tot_2, dtype=object).reshape(-1, 1)
    Track_tot_3 = np.array(Track_tot_3, dtype=object).reshape(-1, 1)

    # Prepare save dictionary
    save_dict = {
        'Track_tot_1': Track_tot_1,
        'Track_tot_2': Track_tot_2,
        'Track_tot_3': Track_tot_3
    }
    
    # Add processing time if provided
    if processing_time is not None:
        save_dict['Tend'] = np.array([processing_time], dtype=np.float64)
    
    # Add ULM parameters if provided
    if ulm_params is not None:
        save_dict['ULM'] = ulm_params

    # Write .mat file
    savemat(output_matfile, save_dict, do_compression=False)

    print(f"Saved {len(Track_tot_1)} + {len(Track_tot_2)} + {len(Track_tot_3)} tracks to {output_matfile}")
    print(f"Track format: [x, z, vx, vz, timeline] - matches MATLAB ULM_tracking2D velocityinterp mode")
    if processing_time is not None:
        print(f"Processing time: {processing_time:.2f} seconds")
    if ulm_params is not None:
        print("ULM parameters included in output") 