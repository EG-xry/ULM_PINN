"""
Generate a 2D vessel mask (x-z) from localized microbubble coordinates in a CSV.

Output is a compressed `.npz` with:
- vessel_mask: boolean array of shape (Nx, Nz) (True = vessel)
- x_grid, z_grid: 1D arrays of bin-centers used by the mask
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter


def load_coordinates(csv_path: str, x_col: str, z_col: str, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load X/Z columns from a (potentially large) CSV using chunked reads."""
    x_list: list[np.ndarray] = []
    z_list: list[np.ndarray] = []

    for chunk in pd.read_csv(csv_path, usecols=[x_col, z_col], chunksize=chunk_size):
        x_list.append(chunk[x_col].to_numpy(dtype=np.float32, copy=False))
        z_list.append(chunk[z_col].to_numpy(dtype=np.float32, copy=False))

    if not x_list:
        raise ValueError(f"No data read from CSV: {csv_path}")

    x = np.concatenate(x_list)
    z = np.concatenate(z_list)
    return x, z


def build_density_map(
    x: np.ndarray,
    z: np.ndarray,
    grid_resolution: int,
    sigma: float,
    pad_frac: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin points into a 2D histogram and optionally smooth with a Gaussian."""
    if x.size == 0:
        raise ValueError("Empty coordinate arrays.")

    x_min, x_max = float(np.min(x)), float(np.max(x))
    z_min, z_max = float(np.min(z)), float(np.max(z))
    x_pad = (x_max - x_min) * float(pad_frac)
    z_pad = (z_max - z_min) * float(pad_frac)
    x_min -= x_pad
    x_max += x_pad
    z_min -= z_pad
    z_max += z_pad

    # Histogram: density[x_bin, z_bin]
    density, x_edges, z_edges = np.histogram2d(
        x, z, bins=int(grid_resolution), range=[[x_min, x_max], [z_min, z_max]]
    )
    density = density.astype(np.float32, copy=False)

    if float(sigma) > 0:
        density = gaussian_filter(density, sigma=float(sigma))

    # Bin centers (monotonic increasing, good for interpolators)
    x_grid = 0.5 * (x_edges[:-1] + x_edges[1:]).astype(np.float32)
    z_grid = 0.5 * (z_edges[:-1] + z_edges[1:]).astype(np.float32)
    return density, x_grid, z_grid


def mask_from_density(density: np.ndarray, threshold_percentile: float, margin_pixels: int) -> np.ndarray:
    """Threshold a density map and apply light morphology for smoothing + margin."""
    positive = density[density > 0]
    if positive.size == 0:
        raise ValueError("Density map is empty (no points landed in any bin).")

    thr = float(np.percentile(positive, float(threshold_percentile)))
    mask = density > thr

    # Close small holes/gaps.
    mask = binary_dilation(mask, iterations=1)
    mask = binary_erosion(mask, iterations=1)

    if int(margin_pixels) > 0:
        mask = binary_dilation(mask, iterations=int(margin_pixels))

    return mask.astype(bool, copy=False)


def save_mask_npz(mask: np.ndarray, x_grid: np.ndarray, z_grid: np.ndarray, output_mask: str) -> None:
    np.savez_compressed(output_mask, vessel_mask=mask, x_grid=x_grid, z_grid=z_grid)


def maybe_save_viz(
    output_viz: str | None,
    density: np.ndarray,
    mask: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
) -> None:
    if not output_viz:
        return

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    extent = [float(x_grid[0]), float(x_grid[-1]), float(z_grid[0]), float(z_grid[-1])]

    im1 = ax1.imshow(density.T, origin="lower", aspect="auto", extent=extent, cmap="hot")
    ax1.set_title("Density")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(mask.T, origin="lower", aspect="auto", extent=extent, cmap="gray")
    ax2.set_title("Vessel mask")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    fig.savefig(output_viz, dpi=200)
    plt.close(fig)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create a 2D vessel mask (x-z) from a coordinate CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to the CSV file containing coordinates"
    )
    parser.add_argument(
        "--x-col",
        type=str,
        default="X",
        help="CSV column name for x"
    )
    parser.add_argument(
        "--z-col",
        type=str,
        default="Z",
        help="CSV column name for z"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="CSV rows per chunk"
    )
    parser.add_argument(
        "--grid-resolution", 
        type=int, 
        default=2000,
        help="Resolution of the mask grid (grid_resolution x grid_resolution)"
    )
    parser.add_argument(
        "--margin-pixels", 
        type=int, 
        default=1,
        help="Number of pixels to expand mask outward (~1 pixel margin)"
    )
    parser.add_argument(
        "--threshold-percentile", 
        type=float, 
        default=75.0,
        help="Percentile for density threshold (higher = less generous boundaries)"
    )
    parser.add_argument(
        "--sigma", 
        type=float, 
        default=1.0,
        help="Gaussian smoothing parameter for density map"
    )
    parser.add_argument(
        "--output-mask", 
        type=str, 
        default="vessel_mask.npz",
        help="Output path for the mask file"
    )
    parser.add_argument(
        "--output-viz", 
        type=str, 
        default=None,
        help="Optional output path for a quick visualization PNG"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    x, z = load_coordinates(args.csv_path, x_col=args.x_col, z_col=args.z_col, chunk_size=int(args.chunk_size))
    density, x_grid, z_grid = build_density_map(
        x,
        z,
        grid_resolution=int(args.grid_resolution),
        sigma=float(args.sigma),
    )
    mask = mask_from_density(density, threshold_percentile=float(args.threshold_percentile), margin_pixels=int(args.margin_pixels))

    save_mask_npz(mask, x_grid, z_grid, output_mask=args.output_mask)
    maybe_save_viz(args.output_viz, density, mask, x_grid, z_grid)

    print(f"Saved vessel mask: {args.output_mask}")
    if args.output_viz:
        print(f"Saved visualization: {args.output_viz}")


if __name__ == "__main__":
    main()

