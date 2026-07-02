import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_matlab_coordinates(mat_file, velocity_clip=0.1, savgol_polyorder=2):
    """
    Load localization coordinates from a MATLAB-generated .mat file

    Loads the Coord_all variable (from PALA_InVivoRatBrain_Coordinates.mat),
    which contains [Intensity, X, Z, ImageIndex] data from the MATLAB pipeline

    Parameters:
        mat_file: Path to the .mat file containing Coord_all
        velocity_clip: Clipping value for normalized velocities
        savgol_polyorder: Polynomial order for Savitzky Golay filter

    Returns:
        X_norm: Normalized x-coordinates (np.array, shape (N,))
        Z_norm: Normalized z-coordinates (np.array, shape (N,))
        T_norm: Normalized time (np.array, shape (N,)) in the range [0,1]
        measured_vel: Placeholder (None); velocities are computed per track later
        orig_bounds: Dictionary containing original coordinate bounds for unnormalization
    """
    print(f"Loading MATLAB coordinates from: {mat_file}")

    try:
        mat_data = loadmat(mat_file)
        if 'Coord_all' not in mat_data:
            raise KeyError("Coord_all variable not found in the .mat file")

        coord_data = mat_data['Coord_all']
        print(f"Loaded Coord_all with shape: {coord_data.shape}")

        # Expected format: [Intensity, X, Z, ImageIndex]
        if coord_data.shape[1] != 4:
            raise ValueError(f"Expected 4 columns, got {coord_data.shape[1]}")

        intensity = coord_data[:, 0]
        X = coord_data[:, 1].astype(np.float32)
        Z = coord_data[:, 2].astype(np.float32)
        T = coord_data[:, 3].astype(np.int32)    # ImageIndex, frame number

        print(f"Data ranges:")
        print(f"  Intensity: [{intensity.min():.2f}, {intensity.max():.2f}]")
        print(f"  X: [{X.min():.2f}, {X.max():.2f}]")
        print(f"  Z: [{Z.min():.2f}, {Z.max():.2f}]")
        print(f"  ImageIndex: [{T.min()}, {T.max()}]")
        print(f"  Total points: {len(X):,}")
        
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        raise

    # Sort by time 
    sort_idx = np.argsort(T)
    X = X[sort_idx]
    Z = Z[sort_idx]
    T = T[sort_idx]
    intensity = intensity[sort_idx]

    # Normalize spatial coordinates to [0,1]
    X_min, X_max = X.min(), X.max()
    Z_min, Z_max = Z.min(), Z.max()
    X_norm = (X - X_min) / (X_max - X_min + 1e-6)
    Z_norm = (Z - Z_min) / (Z_max - Z_min + 1e-6)

    # bounds for later unnormalization
    orig_bounds = {
        'X_min': X_min, 'X_max': X_max,
        'Z_min': Z_min, 'Z_max': Z_max
    }

    # ImageIndex -> positive integers starting at 1
    T_int = T.astype(np.int32)
    T_frame = T_int - T_int.min() + 1

    # Normalize time to [0,1]
    T_norm = (T_frame - T_frame.min()) / (T_frame.max() - T_frame.min() + 1e-6)
    orig_bounds['T_min'] = T_frame.min()
    orig_bounds['T_max'] = T_frame.max()

    # velocities computed post-tracking, per track
    measured_vel = None

    # spatial coverage diagnostic
    plt.figure(figsize=(5, 4))
    plt.scatter(X_norm[::1000], Z_norm[::1000], c=T_norm[::1000], cmap='viridis', alpha=0.6, s=1)
    plt.colorbar(label='Normalized Time')
    plt.title("Spatial Distribution (sampled)")
    plt.xlabel('Normalized X')
    plt.ylabel('Normalized Z')
    plt.tight_layout()
    plt.savefig('matlab_coordinates_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis plot saved as: matlab_coordinates_analysis.png")
    
    return X_norm, Z_norm, T_norm, measured_vel, orig_bounds


def load_tracking_data(csv_file, velocity_clip=0.1, savgol_polyorder=2):
    """
    Load the microbubble tracking CSV file.
    Expected columns: Intensity, X, Z, ImageIndex.
    Coordinates (X, Z) are normalized to [0,1]; ImageIndex becomes the frame
    number (positive integer starting at 1).

    Returns:
        X_norm: Normalized x-coordinates (np.array, shape (N,))
        Z_norm: Normalized z-coordinates (np.array, shape (N,))
        T_norm: Normalized time (np.array, shape (N,)) in the range [0,1]
        measured_vel: Placeholder (None); velocities are computed per track later
        orig_bounds: Dictionary containing original coordinate bounds for unnormalization
    """
    df = pd.read_csv(csv_file)

    X = df['X'].values.astype(np.float32)
    Z = df['Z'].values.astype(np.float32)
    T = df['ImageIndex'].values

    # Sort by time 
    sort_idx = np.argsort(T)
    X = X[sort_idx]
    Z = Z[sort_idx]
    T = T[sort_idx]

    # Normalize spatial coordinates to [0,1]
    X_min, X_max = X.min(), X.max()
    Z_min, Z_max = Z.min(), Z.max()
    X_norm = (X - X_min) / (X_max - X_min + 1e-6)
    Z_norm = (Z - Z_min) / (Z_max - Z_min + 1e-6)

    # bounds for later unnormalization
    orig_bounds = {
        'X_min': X_min, 'X_max': X_max,
        'Z_min': Z_min, 'Z_max': Z_max
    }

    # ImageIndex -> positive integers starting at 1
    T_int = T.astype(np.int32)
    T_frame = T_int - T_int.min() + 1

    # Normalize time to [0,1]
    T_norm = (T_frame - T_frame.min()) / (T_frame.max() - T_frame.min() + 1e-6)
    orig_bounds['T_min'] = T_frame.min()
    orig_bounds['T_max'] = T_frame.max()

    measured_vel = None
    
    return X_norm, Z_norm, T_norm, measured_vel, orig_bounds