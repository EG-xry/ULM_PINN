"""
Data loading utilities
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

def load_matlab_coordinates(
    mat_file,
    velocity_clip=0.1,
    savgol_polyorder=2,
    save_analysis_plot=False,
    analysis_plot_stride=1000,
    analysis_plot_path="matlab_coordinates_analysis.png",
):
    """
    Load localization coordinates directly from MATLAB-generated .mat file
    This function loads the Coord_all variable from PALA_InVivoRatBrain_Coordinates.mat
    which contains [Intensity, X, Z, ImageIndex] data from the original MATLAB pipeline.
    
    Parameters:
        mat_file: Path to the .mat file containing Coord_all
        velocity_clip: Clipping value for normalized velocities
        savgol_polyorder: Polynomial order for Savitzky-Golay filter
    
    Returns:
        X_norm: Normalized x-coordinates (np.array, shape (N,))
        Z_norm: Normalized z-coordinates (np.array, shape (N,))
        T_norm: Normalized time (np.array, shape (N,)) in the range [0,1]
        measured_vel: Placeholder (None); velocities are computed per-track later
        orig_bounds: Dictionary containing original coordinate bounds for unnormalization
    """
    print(f"Loading MATLAB coordinates: {mat_file}")
    
    try:
        mat_data = loadmat(mat_file)
        if 'Coord_all' not in mat_data:
            raise KeyError("Coord_all variable not found in the .mat file")
        
        coord_data = mat_data['Coord_all']
        
        # Verify the expected format: [Intensity, X, Z, ImageIndex]
        if coord_data.shape[1] != 4:
            raise ValueError(f"Expected 4 columns, got {coord_data.shape[1]}")
        
        # Extract columns according to MATLAB pipeline format
        intensity = coord_data[:, 0]  # Column 1: Intensity
        X = coord_data[:, 1].astype(np.float32)  # Column 2: X coordinates
        Z = coord_data[:, 2].astype(np.float32)  # Column 3: Z coordinates  
        T = coord_data[:, 3].astype(np.int32)    # Column 4: ImageIndex (frame number)
        
        # Keep logging concise (publishable default)
        print(f"  points={len(X):,}, frames=[{int(T.min())}, {int(T.max())}]")
        
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        raise
    
    # Sort the data by time (frame) 
    sort_idx = np.argsort(T)
    X = X[sort_idx]
    Z = Z[sort_idx]
    T = T[sort_idx]
    intensity = intensity[sort_idx]
    
    X_min, X_max = X.min(), X.max()
    Z_min, Z_max = Z.min(), Z.max()
    X_norm = (X - X_min) / (X_max - X_min + 1e-6)
    Z_norm = (Z - Z_min) / (Z_max - Z_min + 1e-6)
    
    # Store original bounds for unnormalization 
    orig_bounds = {
        'X_min': X_min, 'X_max': X_max,
        'Z_min': Z_min, 'Z_max': Z_max
    }
    
    # Convert ImageIndex to positive integers starting at 1
    T_int = T.astype(np.int32)
    T_frame = T_int - T_int.min() + 1  
    T_norm = (T_frame - T_frame.min()) / (T_frame.max() - T_frame.min() + 1e-6)
    
    # Update orig_bounds with time limits for later unnormalization
    orig_bounds['T_min'] = T_frame.min()
    orig_bounds['T_max'] = T_frame.max()
    
    measured_vel = None
    
    # Optional diagnostic plot (off by default)
    if save_analysis_plot:
        stride = int(max(1, analysis_plot_stride))
        plt.figure(figsize=(5, 4))
        plt.scatter(
            X_norm[::stride],
            Z_norm[::stride],
            c=T_norm[::stride],
            cmap="viridis",
            alpha=0.6,
            s=1,
        )
        plt.colorbar(label="Normalized Time")
        plt.title("Spatial Distribution (sampled)")
        plt.xlabel("Normalized X")
        plt.ylabel("Normalized Z")
        plt.tight_layout()
        plt.savefig(str(analysis_plot_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved analysis plot: {analysis_plot_path}")
    
    return X_norm, Z_norm, T_norm, measured_vel, orig_bounds