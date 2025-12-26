#!/usr/bin/env python3
"""
ULM-PINN Framework - MAIN File
This script implements a Physics-Informed Neural Network (PINN) framework 
for ultrasound localization microscopy (ULM) microbubble tracking

The pipeline includes:
1. Data loading from MATLAB coordinate file
2. PINN training with physics-informed loss
3. Advanced tracking with LAPJV + PINN guidance
4. Post-processing and output generation

Author: Eric Gao
Date: 2025-12-25
Version: 1.0.0
SPDX-License-Identifier: MIT

Copyright (c) 2025 Eric Gao
See the repository-level LICENSE file for full license text.
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import time
import gc
import inspect
import json
import datetime
import sys
import shutil
import subprocess
from pathlib import Path
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory monitoring will be disabled.")

# Import our modular components
from data_loading import load_matlab_coordinates
from pinn_model import PINN, train_pinn
from tracking import (
    extract_tracks_hungarian_initial,
    compute_track_velocities,
    extract_tracks_lapjv_pinn
)
from post_processing import (
    interpolate_tracks,
    unnormalize_tracks,
    save_tracks_mat,
    save_tracks_mat_complete
)
from visualization import (
    plot_loss,
    calculate_track_statistics,
    calculate_velocity_statistics,
    plot_density_sampling,
)
from tracking import (
    calculate_track_statistics_memory_efficient,
    calculate_velocity_statistics_memory_efficient,
    verify_track_uniqueness
)

class Tee:
    """Class to write to both console and file simultaneously."""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

def setup_output_directory(base_dir="runs"):
    """
    Create a timestamped output directory for this run.
    
    Returns:
        output_dir: Path to the created output directory
        log_file: File handle for the log file
    """
    # Create base runs directory if it doesn't exist
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_path / f"run_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Created output directory: {output_dir}")

    log_file_path = output_dir / "run.log"
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    tee = Tee(sys.stdout, log_file)
    
    original_stdout = sys.stdout
    sys.stdout = tee
    
    return output_dir, log_file, original_stdout

def save_parameters(args, output_dir, additional_params=None):
    """
    Save all key parameters to a JSON file
    
    Parameters:
        args: Parsed command line arguments
        output_dir: Path to output directory
        additional_params: Dictionary of additional parameters to save
    """
    params = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "training_parameters": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "n_colloc": args.n_colloc,
            "beta": args.beta,
            "data_only_epochs": args.data_only_epochs,
            "include_unsteady": args.include_unsteady,
            "rho": args.rho,
            "unsteady_ramp_epochs": args.unsteady_ramp_epochs,
            "hidden_layers": args.hidden_layers,
            "hidden_size": args.hidden_size,
            "activation": args.activation,
        },
        "tracking_parameters": {
            "hungarian_w_geo": args.hungarian_w_geo,
            "hungarian_w_phys": args.hungarian_w_phys,
            "distance_threshold": args.distance_threshold,
            "hungarian_cost_threshold": args.hungarian_cost_threshold,
            "min_length": args.min_length,
            "max_tracks_per_frame": args.max_tracks_per_frame,
            "batch_size": args.batch_size,
            "initial_max_geo_radius": getattr(args, "initial_max_geo_radius", None),
            "initial_min_length": getattr(args, "initial_min_length", None),
            "initial_cost_threshold": getattr(args, "initial_cost_threshold", 50.0),
        },
        "interpolation_parameters": {
            "interp_factor": args.interp_factor,
            "interpolation_enabled": args.interp_factor > 0,
            "smooth_factor": args.smooth_factor,
            "res": args.res,
            "max_linking_distance": args.max_linking_distance,
            "savgol_polyorder": args.savgol_polyorder,
        },
        "data_parameters": {
            "mat_file": args.mat_file,
            "velocity_clip": args.velocity_clip,
        },
        "optimization_parameters": {
            "use_optimized_physics": args.use_optimized_physics,
            "use_vectorized_velocity": args.use_vectorized_velocity,
        },
        "density_sampling_parameters": {
            "use_density_guided": args.use_density_guided,
            "grid_resolution": args.grid_resolution,
            "bandwidth": args.bandwidth,
            "min_density_threshold": args.min_density_threshold,
            "fallback_ratio": args.fallback_ratio,
            "use_fast_density": args.use_fast_density,
            "density_method": args.density_method,
        },
        "geometry_parameters": {
            "colloc_sampling_mode": args.colloc_sampling_mode,
            "colloc_vessel_fraction": args.colloc_vessel_fraction,
            "colloc_wall_fraction": args.colloc_wall_fraction,
        },
        "vessel_mask_parameters": {
            "vessel_mask_path": args.vessel_mask_path,
            "wall_loss_weight": args.wall_loss_weight,
            "outside_vessel_loss_weight": args.outside_vessel_loss_weight,
            "n_wall_samples": args.n_wall_samples,
            "n_outside_samples": args.n_outside_samples,
        },
        "performance_parameters": {
            "interp_batch_size": args.interp_batch_size,
            "unnorm_batch_size": args.unnorm_batch_size,
            "interp_gc_frequency": args.interp_gc_frequency,
            "unnorm_gc_frequency": args.unnorm_gc_frequency,
            "gpu_memory_fraction": args.gpu_memory_fraction,
        },
    }
    
    # Add any additional parameters
    if additional_params:
        params["additional_parameters"] = additional_params
    
    # Save to JSON file
    params_file = output_dir / "parameters.json"
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    
    params_txt_file = output_dir / "parameters.txt"
    with open(params_txt_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RUN PARAMETERS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Run Timestamp: {params['run_timestamp']}\n\n")
        
        for section_name, section_params in params.items():
            if section_name == "run_timestamp":
                continue
            f.write(f"{section_name.upper().replace('_', ' ')}:\n")
            f.write("-" * 80 + "\n")
            for key, value in section_params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"Parameters saved to: {params_file}")
    print(f"Parameters (text) saved to: {params_txt_file}")

def monitor_memory_usage():
    """Monitor current memory usage and return usage statistics."""
    if not PSUTIL_AVAILABLE:
        return {
            'rss_mb': 0,
            'vms_mb': 0,
            'percent': 0
        }
    
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # MB
        'percent': memory_percent
    }

def get_adaptive_batch_sizes(dataset_size, available_memory_mb=None):
    """
    Determine optimal batch sizes based on dataset size and available memory
    
    Parameters:
        dataset_size: Number of data points
        available_memory_mb: Available memory in MB (if None, will estimate)
    
    Returns:
        dict: Optimal batch sizes for different operations
    """
    if available_memory_mb is None:
        # Estimate available memory 
        if PSUTIL_AVAILABLE:
            memory_info = psutil.virtual_memory()
            available_memory_mb = memory_info.available / 1024 / 1024 * 0.8  # Use 80% of available
        else:
            # Fallback to conservative estimate when psutil is not available
            available_memory_mb = 4000  # Assume 4GB available
    
    # Determine dataset category
    if dataset_size < 100000:
        category = "small"
    elif dataset_size < 1000000:
        category = "medium"
    elif dataset_size < 10000000:
        category = "large"
    else:
        category = "very_large"
    
    # Base batch sizes by category
    base_sizes = {
        "small": {
            "interp_batch_size": 2000,
            "unnorm_batch_size": 10000,
            "track_stats_batch_size": 5000,
            "velocity_stats_batch_size": 5000,
            "gc_frequency": 100
        },
        "medium": {
            "interp_batch_size": 1000,
            "unnorm_batch_size": 5000,
            "track_stats_batch_size": 2000,
            "velocity_stats_batch_size": 2000,
            "gc_frequency": 50
        },
        "large": {
            "interp_batch_size": 500,
            "unnorm_batch_size": 2000,
            "track_stats_batch_size": 1000,
            "velocity_stats_batch_size": 1000,
            "gc_frequency": 20
        },
        "very_large": {
            "interp_batch_size": 200,
            "unnorm_batch_size": 1000,
            "track_stats_batch_size": 500,
            "velocity_stats_batch_size": 500,
            "gc_frequency": 10
        }
    }
    
    # Adjust based on available memory
    memory_factor = min(available_memory_mb / 8000, 2.0)  # Normalize to 8GB baseline
    
    batch_sizes = base_sizes[category].copy()
    for key in batch_sizes:
        if key != "gc_frequency":
            batch_sizes[key] = int(batch_sizes[key] * memory_factor)
    
    print(f"Dataset category: {category} ({dataset_size:,} points)")
    print(f"Available memory: {available_memory_mb:.0f} MB")
    print(f"Memory factor: {memory_factor:.2f}")
    print(f"Adaptive batch sizes: {batch_sizes}")
    
    return batch_sizes

# Global variable for density sampling visualization
try:
    from density_sampling import plot_density_sampling
    DENSITY_PLOTTING_AVAILABLE = True
except ImportError:
    DENSITY_PLOTTING_AVAILABLE = False

def main(args):
    """
    Main pipeline function that orchestrates the entire ULM PINN workflow
    
    Parameters:
        args: Parsed command line arguments
    """
    # Set up output directory and logging
    output_dir, log_file, original_stdout = setup_output_directory()
    
    try:
        # Save parameters at the start
        save_parameters(args, output_dir)
        
        # Print parameter verification
        print("="*80)
        print("PARAMETER VERIFICATION")
        print("="*80)
        print("Key Tracking Parameters:")
        print(f"  distance_threshold: {args.distance_threshold} (max geometric linking distance)")
        print(f"  hungarian_w_geo: {args.hungarian_w_geo} (geometric weight in cost matrix)")
        print(f"  hungarian_w_phys: {args.hungarian_w_phys} (physics weight in cost matrix)")
        print(f"  hungarian_cost_threshold: {args.hungarian_cost_threshold} (max assignment cost)")
        print(f"  min_length: {args.min_length} (minimum track length)")
        print(f"  initial_max_geo_radius: {getattr(args, 'initial_max_geo_radius', None)} (None => uses distance_threshold for initial tracking)")
        print(f"  initial_min_length: {getattr(args, 'initial_min_length', None)} (None => uses min_length for initial tracking + velocity computation)")
        print(f"  initial_cost_threshold: {getattr(args, 'initial_cost_threshold', 50.0)} (assignment cost threshold for initial tracking)")
        print(f"  max_tracks_per_frame: {args.max_tracks_per_frame} (tracks per frame limit)")
        print("")
        print("Training Parameters:")
        print(f"  n_colloc: {args.n_colloc}")
        print(f"  epochs: {args.epochs}")
        print(f"  lr: {args.lr}")
        print(f"  beta: {args.beta}")
        print(f"  data_only_epochs: {args.data_only_epochs}")
        print(f"  include_unsteady: {args.include_unsteady}")
        if args.include_unsteady:
            print(f"  rho: {args.rho}")
            print(f"  unsteady_ramp_epochs: {args.unsteady_ramp_epochs}")
        print("")
        print("Optimization Parameters:")
        print(f"  use_optimized_physics: {args.use_optimized_physics}")
        print(f"  use_vectorized_velocity: {args.use_vectorized_velocity}")
        print("")
        print("Vessel Mask Parameters:")
        print(f"  vessel_mask_path: {args.vessel_mask_path} (path to vessel mask file)")
        print(f"  wall_loss_weight: {args.wall_loss_weight} (weight for wall boundary loss)")
        print(f"  outside_vessel_loss_weight: {args.outside_vessel_loss_weight} (weight for outside-vessel penalty)")
        print(f"  n_wall_samples: {args.n_wall_samples} (wall points per epoch)")
        print(f"  n_outside_samples: {args.n_outside_samples} (outside-vessel points per epoch)")
        print("="*80)
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device:", device)
        
        # Set memory management for PyTorch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
            
            # Print memory status
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {total_memory:.1f}GB total, {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            print(f"Using {args.gpu_memory_fraction*100:.0f}% of available GPU memory")
        
        # Start timing
        start_time = time.time()
        # ===================== DATA LOADING =====================
        if not os.path.exists(args.mat_file):
            raise FileNotFoundError(f"MATLAB file not found: {args.mat_file}")
        print("=== LOADING DATA (.mat) ===")
        X, Z, T_norm, measured_vel, orig_bounds = load_matlab_coordinates(
            args.mat_file,
            velocity_clip=args.velocity_clip,
            savgol_polyorder=args.savgol_polyorder,
        )
        x_norm = np.stack([X, Z, T_norm], axis=-1)
        print(f"Loaded {len(X):,} points from: {args.mat_file}")
        
        # Check for large datasets and get adaptive batch sizes
        if len(X) > 500000:
            print(f"Warning: Large dataset detected ({len(X)} points). This may cause memory issues.")
            if args.auto_disable_density and args.use_density_guided:
                print("Large dataset detected - using fast histogram-based density estimation for efficiency.")
                args.use_fast_density = True
        
        # Get adaptive batch sizes based on dataset size and available memory
        adaptive_batch_sizes = get_adaptive_batch_sizes(len(X))
        
        # Override user-specified batch sizes with adaptive ones for better performance
        args.interp_batch_size = adaptive_batch_sizes["interp_batch_size"]
        args.unnorm_batch_size = adaptive_batch_sizes["unnorm_batch_size"]
        args.interp_gc_frequency = adaptive_batch_sizes["gc_frequency"]
        args.unnorm_gc_frequency = adaptive_batch_sizes["gc_frequency"]
        
        print(f"Using adaptive batch sizes for optimal performance:")
        print(f"  Interpolation batch size: {args.interp_batch_size}")
        print(f"  Unnormalization batch size: {args.unnorm_batch_size}")
        print(f"  GC frequency: {args.interp_gc_frequency}")
        
        # Monitor initial memory usage
        initial_memory = monitor_memory_usage()
        print(f"Initial memory usage: {initial_memory['rss_mb']:.0f} MB ({initial_memory['percent']:.1f}%)")
        
        # ===================== INITIAL TRACKING =====================
        # Hungarian initial tracking to preserve maximum detection points
        initial_max_geo_radius = getattr(args, "initial_max_geo_radius", None)
        if initial_max_geo_radius is None:
            initial_max_geo_radius = args.distance_threshold
        initial_min_length = getattr(args, "initial_min_length", None)
        if initial_min_length is None:
            initial_min_length = args.min_length
        initial_cost_threshold = getattr(args, "initial_cost_threshold", 50.0)

        raw_tracks = extract_tracks_hungarian_initial(
            x_norm, 
            time_precision=0.001,
            max_geo_radius=float(initial_max_geo_radius),
            min_length=int(initial_min_length),
            max_tracks_per_frame=500000,
            cost_threshold=float(initial_cost_threshold),
            max_matrix_size=500000000,
        )
        print(f"Stage1: {len(raw_tracks)} raw tracks (Hungarian initial tracking)")
        
        # Calculate raw track statistics
        raw_stats = calculate_track_statistics(raw_tracks, "raw")

        # ===================== VELOCITY COMPUTATION =====================
        # Compute velocities from tracks
        pos2, vel2 = compute_track_velocities(
            raw_tracks, 
            savgol_polyorder=args.savgol_polyorder, 
            use_vectorized=args.use_vectorized_velocity,
            min_track_length=int(initial_min_length),
        )
        x_data = torch.tensor(pos2, dtype=torch.float32, device=device)
        measured_vel = torch.tensor(vel2, dtype=torch.float32, device=device)
        
        # Prepare data for PINN
        x_data_np = np.stack([X, Z, T_norm], axis=-1)
        domain = {'x': (0.0, 1.0), 'z': (0.0, 1.0), 't': (0.0, 1.0)}
        
        # ===================== MODEL INITIALIZATION =====================
        model = PINN(
            input_dim=3, 
            output_dim=3, 
            hidden_layers=args.hidden_layers, 
            hidden_size=args.hidden_size,
            activation=args.activation,
        ).to(device)

        # Important: optimize only trainable params
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=args.lr)
        
        # Add learning rate scheduler
        scheduler = None
        if args.use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=500, verbose=True
            )

        # ===================== TRAINING =====================
        print("Training PINN...")
        # Build training kwargs and filter by train_pinn signature for portability
        train_kwargs = dict(
            domain={'x': (0, 1), 'z': (0, 1), 't': (0, 1)},
            n_colloc=args.n_colloc,
            beta=args.beta,
            debug_output_dir=str(output_dir),
            epochs=args.epochs,
            print_every=args.print_every,
            scheduler=scheduler,
            data_only_epochs=args.data_only_epochs,
            use_density_guided=args.use_density_guided,
            grid_resolution=args.grid_resolution,
            bandwidth=args.bandwidth,
            min_density_threshold=args.min_density_threshold,
            fallback_ratio=args.fallback_ratio,
            plot_density=args.plot_density,
            use_fast_density=args.use_fast_density,
            density_method=args.density_method,
            data_batch_size=args.data_batch_size,
            use_optimized_physics=args.use_optimized_physics,
            include_unsteady=args.include_unsteady,
            rho=args.rho,
            unsteady_ramp_epochs=args.unsteady_ramp_epochs,
            orig_bounds=orig_bounds,
            vessel_mask_path=args.vessel_mask_path,
            wall_loss_weight=args.wall_loss_weight,
            outside_vessel_loss_weight=args.outside_vessel_loss_weight,
            n_wall_samples=args.n_wall_samples,
            n_outside_samples=args.n_outside_samples,
            colloc_sampling_mode=args.colloc_sampling_mode,
            colloc_vessel_fraction=args.colloc_vessel_fraction,
            colloc_wall_fraction=args.colloc_wall_fraction,
        )
        sig_params = set(inspect.signature(train_pinn).parameters.keys())
        filtered_kwargs = {k: v for k, v in train_kwargs.items() if k in sig_params}

        data_loss_history, physics_loss_history, total_loss_history = train_pinn(
            model, optimizer, x_data, measured_vel, **filtered_kwargs
        )
        
        print("Training complete!")

        print(f"Trainable viscosity mu: {model.get_mu().item():.4e}")
        
        # Plot loss curves (saved to output directory)
        loss_curve_path = output_dir / "loss_curve.png"
        plot_loss(data_loss_history, physics_loss_history, total_loss_history, str(loss_curve_path))
        for N in [10000, 5000, 1000, 500, 300, 200]:
            if len(total_loss_history) >= N:
                loss_curve_n_path = output_dir / f"loss_curve_{N}.png"
                plot_loss(data_loss_history[:N], physics_loss_history[:N], total_loss_history[:N], str(loss_curve_n_path))

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ===================== LAPJV + PINN RE-TRACKING =====================
        print("Starting LAPJV + PINN re-tracking...")
        print("Using all detection points for comprehensive tracking...")
        
        xtot = x_data_np
        print(f"Processing ALL {len(xtot)} detection points")
        
        # LAPJV + PINN tracking 
        tracks = extract_tracks_lapjv_pinn(
            xtot, model,
            w_geo=args.hungarian_w_geo, 
            w_phys=args.hungarian_w_phys,
            max_geo_radius=args.distance_threshold,
            dt=1.0,
            batch_size=args.batch_size,
            max_tracks_per_frame=args.max_tracks_per_frame,
            min_length=args.min_length,
            cost_threshold=args.hungarian_cost_threshold,
            use_multi_gpu=args.hungarian_use_multi_gpu,
            memory_efficient=True,
            max_matrix_size=args.hungarian_max_matrix_size,
            max_candidates_per_track=args.hungarian_max_candidates_per_track,
            max_total_candidates=args.hungarian_max_total_candidates,
            early_termination=True,
            adaptive_batch_sizing=True,
        )
        print(f"LAPJV + PINN tracking: {len(tracks)} high-quality tracks")
        
        # Calculate LAPJV track statistics
        lapjv_stats = calculate_track_statistics_memory_efficient(tracks, "LAPJV + PINN", batch_size=5000)
        lapjv_velocity_stats = calculate_velocity_statistics_memory_efficient(tracks, "LAPJV + PINN (normalized)", orig_bounds, batch_size=5000)
        
        # Memory management 
        from tracking import manage_memory_for_large_datasets
        is_large_dataset, memory_recommendations = manage_memory_for_large_datasets(tracks)
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ===================== POST-PROCESSING =====================
        # Auto-adjust performance parameters for large datasets
        total_tracks = len(tracks)
        if total_tracks > 100000:
            if args.interp_batch_size < 2000:
                args.interp_batch_size = 2000
                print(f"Large dataset detected - auto-increasing interp_batch_size to {args.interp_batch_size}")
            if args.unnorm_batch_size < 10000:
                args.unnorm_batch_size = 10000
                print(f"Large dataset detected - auto-increasing unnorm_batch_size to {args.unnorm_batch_size}")
        
        # Monitor memory before interpolation
        pre_interp_memory = monitor_memory_usage()
        print(f"Memory before interpolation: {pre_interp_memory['rss_mb']:.0f} MB ({pre_interp_memory['percent']:.1f}%)")
        
        # Interpolation
        print("\n Starting interpolation...")
        interp_start_time = time.time()
        interp_tracks = interpolate_tracks(
            tracks, 
            interp_factor=args.interp_factor, 
            smooth_factor=args.smooth_factor, 
            res=args.res, 
            max_linking_distance=args.max_linking_distance, 
            savgol_polyorder=args.savgol_polyorder,
            batch_size=args.interp_batch_size,
            gc_frequency=args.interp_gc_frequency,
            use_fast_mode=args.interp_fast_mode
        )
        interp_time = time.time() - interp_start_time
        print(f"Interpolation completed in {interp_time/60:.1f} minutes")
        
        # Monitor memory after interpolation
        post_interp_memory = monitor_memory_usage()
        print(f"Memory after interpolation: {post_interp_memory['rss_mb']:.0f} MB ({post_interp_memory['percent']:.1f}%)")
        print(f"Memory increase: {post_interp_memory['rss_mb'] - pre_interp_memory['rss_mb']:.0f} MB")
        
        # Memory management for interpolated tracks
        is_large_interp, _ = manage_memory_for_large_datasets(interp_tracks)
        
        # Calculate interpolated track statistics
        interp_stats = calculate_track_statistics_memory_efficient(
            interp_tracks, "interpolated", 
            batch_size=adaptive_batch_sizes["track_stats_batch_size"]
        )
        
        # Calculate interpolated velocity statistics 
        if is_large_interp:
            print("Large interpolated dataset detected - using conservative batch sizes")
            try:
                interp_velocity_stats = calculate_velocity_statistics_memory_efficient(
                    interp_tracks, "interpolated (normalized)", orig_bounds, 
                    batch_size=adaptive_batch_sizes["velocity_stats_batch_size"]
                )
            except MemoryError:
                print("Memory error during interpolated velocity statistics calculation. Skipping this step.")
                interp_velocity_stats = None
        else:
            try:
                interp_velocity_stats = calculate_velocity_statistics_memory_efficient(
                    interp_tracks, "interpolated (normalized)", orig_bounds, 
                    batch_size=adaptive_batch_sizes["velocity_stats_batch_size"]
                )
            except MemoryError:
                print("Memory error during interpolated velocity statistics calculation. Skipping this step.")
                interp_velocity_stats = None
        
        # Monitor memory before unnormalization
        pre_unnorm_memory = monitor_memory_usage()
        print(f"Memory before unnormalization: {pre_unnorm_memory['rss_mb']:.0f} MB ({pre_unnorm_memory['percent']:.1f}%)")
        
        # Unnormalization
        print("Starting unnormalization...")
        unnorm_start_time = time.time()
        unnorm_tracks = unnormalize_tracks(
            interp_tracks, orig_bounds,
            batch_size=args.unnorm_batch_size,
            gc_frequency=args.unnorm_gc_frequency,
            use_vectorized=args.unnorm_vectorized
        )
        unnorm_time = time.time() - unnorm_start_time
        print(f"Unnormalization completed in {unnorm_time/60:.1f} minutes")
        
        # Monitor memory after unnormalization
        post_unnorm_memory = monitor_memory_usage()
        print(f"Memory after unnormalization: {post_unnorm_memory['rss_mb']:.0f} MB ({post_unnorm_memory['percent']:.1f}%)")
        print(f"Memory increase: {post_unnorm_memory['rss_mb'] - pre_unnorm_memory['rss_mb']:.0f} MB")
        
        # Final statistics 
        unnorm_stats = calculate_track_statistics_memory_efficient(
            unnorm_tracks, "unnormalized", 
            batch_size=adaptive_batch_sizes["track_stats_batch_size"]
        )
        
        # Calculate final velocity statistics in physical units 
        try:
            final_velocity_stats = calculate_velocity_statistics_memory_efficient(
                unnorm_tracks, "final (physical units)", 
                batch_size=adaptive_batch_sizes["velocity_stats_batch_size"]
            )
        except MemoryError:
            print("Memory error during final velocity statistics calculation. Skipping this step.")
            final_velocity_stats = None
        
        # ===================== SAVE RESULTS =====================
        total_time = time.time() - start_time
        
        # Create ULM parameters
        ulm_params = {
            'numberOfParticles': 90,
            'size': [len(X), len(Z), len(np.unique(T_norm))],
            'scale': [1.0, 1.0, 1.0],
            'res': args.res,
            'SVD_cutoff': [5, len(np.unique(T_norm))],
            'max_linking_distance': args.max_linking_distance,
            'min_length': args.min_length,
            'fwhm': [3, 3],
            'max_gap_closing': 0,
            'interp_factor': 1.0 / (args.max_linking_distance * args.res) * 0.8,
            'LocMethod': 'PINN_Enhanced',
            'ButterCuttofFreq': [50, 249],
            'parameters': {'NLocalMax': 3},
            'lambda': 1540 / (15e6),
            'velocity_tol': 1.0
        }
        
        # Generate output filename (save to output directory)
        base_name = os.path.splitext(os.path.basename(args.mat_file))[0]
        matfile_name = str(output_dir / f"{base_name}_PINN_tracks.mat")
        
        # Save complete results
        save_tracks_mat_complete(
            unnorm_tracks, matfile_name, 
            processing_time=total_time, 
            ulm_params=ulm_params
        )
        
        # Update parameters file with final results
        final_results = {
            "total_processing_time_minutes": total_time / 60,
            "interpolation_time_minutes": interp_time / 60 if 'interp_time' in locals() else None,
            "unnormalization_time_minutes": unnorm_time / 60 if 'unnorm_time' in locals() else None,
            "input_detection_points": len(X),
            "final_tracks": len(unnorm_tracks),
            "output_file": matfile_name,
            "trainable_viscosity_mu": model.get_mu().item() if hasattr(model, 'get_mu') else None,
        }
        save_parameters(args, output_dir, additional_params=final_results)
        
        # ===================== FINAL SUMMARY =====================
        print("Processing completed successfully!")
        print(f"Performance Summary:")
        print(f"  Total processing time: {total_time/60:.1f} minutes")
        if 'interp_time' in locals():
            print(f"  Interpolation time: {interp_time/60:.1f} minutes")
        if 'unnorm_time' in locals():
            print(f"  Unnormalization time: {unnorm_time/60:.1f} minutes")
        
        print(f"Final Results:")
        print(f"  Input detection points: {len(X):,}")
        print(f"  Final tracks: {len(unnorm_tracks):,}")
        print(f"  Track format: MATLAB ULM_tracking2D compatible")
        print(f"  Output file: {matfile_name}")
        print(f"\nAll outputs saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to save any intermediate results
        try:
            if 'tracks' in locals():
                print("Attempting to save intermediate results...")
                matfile_name = str(output_dir / "tracks_partial.mat")
                save_tracks_mat(tracks, matfile_name)
                print(f"Partial results saved to {matfile_name}")
        except:
            print("Could not save intermediate results")
        
        raise
    finally:
        # Restore original stdout and close log file
        sys.stdout = original_stdout
        log_file.close()
        print(f"\nLog file saved to: {output_dir / 'run.log'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ULM Pipeline PINN Framework")
    
    # ===================== INPUT DATA =====================
    parser.add_argument("--mat_file", type=str, default="INSERT",
                        help="Path to MATLAB .mat file containing `Coord_all` (Intensity, X, Z, ImageIndex).")
    
    # ===================== MODEL =====================
    parser.add_argument("--hidden_layers", type=int, default=5, help="Number of hidden layers")
    parser.add_argument("--hidden_size", type=int, default=64, help="Neurons per hidden layer")
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh"], help="Activation function (fixed)")
    
    # ===================== TRAINING =====================
    parser.add_argument("--lr", type=float, default=8e-3, help="Learning rate")
    parser.add_argument("--n_colloc", type=int, default=5000, help="Number of collocation points")
    parser.add_argument("--beta", type=float, default=1.0, help="Physics loss weight")
    parser.add_argument("--epochs", type=int, default=20000, help="Training epochs")
    parser.add_argument("--print_every", type=int, default=200, help="Print frequency (epochs)")
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--data_only_epochs", type=int, default=0, help="Optional warmup epochs using data loss only (0 disables).")
    parser.add_argument("--include_unsteady", action="store_true", default=True, help="Include local acceleration term (rho * du/dt) in momentum residual.")
    parser.add_argument("--rho", type=float, default=1000.0, help="Density used for unsteady term")
    parser.add_argument("--unsteady_ramp_epochs", type=int, default=0, help="Linearly ramp the unsteady term from 0→full over this many epochs (0 disables).")
    
    # ===================== TRACKING ARGUMENTS =====================
    parser.add_argument("--min_length", type=int, default=15, help="Minimum track length")
    parser.add_argument("--distance_threshold", type=float, default=0.01, help="Default: 0.01 ****** Maximum linking distance")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for tracking")
    parser.add_argument("--max_tracks_per_frame", type=int, default=100000, help="Maximum tracks per frame")
    parser.add_argument("--initial_max_geo_radius", type=float, default=0.02, help="Max geometric linking radius for INITIAL Hungarian tracking stage. If None, uses --distance_threshold.")
    parser.add_argument("--initial_min_length", type=int, default=15, help="Minimum track length for INITIAL Hungarian tracking stage and velocity computation. If None, uses --min_length.")
    parser.add_argument("--initial_cost_threshold", type=float, default=10.0, help="Assignment cost threshold for INITIAL Hungarian tracking stage (controls permissiveness).")
    
    # ===================== LAPJV + PINN TRACKING ARGUMENTS =====================
    parser.add_argument("--hungarian_w_geo", type=float, default=0.5, help="Geometric weight in LAPJV")
    parser.add_argument("--hungarian_w_phys", type=float, default=0.5, help="Physics weight in LAPJV")
    parser.add_argument("--hungarian_cost_threshold", type=float, default=2.0, help="LAPJV cost threshold")
    parser.add_argument("--hungarian_use_multi_gpu", action="store_true", default=False, help="Use multi-GPU")
    parser.add_argument("--hungarian_max_matrix_size", type=int, default=500000000, help="Max matrix size")
    parser.add_argument("--hungarian_max_candidates_per_track", type=int, default=500, help="Max candidates per track")
    parser.add_argument("--hungarian_max_total_candidates", type=int, default=25000, help="Max total candidates")
    
    # ===================== DENSITY SAMPLING ARGUMENTS =====================
    parser.add_argument("--use_density_guided", action="store_true", default=True, help="Use density-guided sampling")
    parser.add_argument("--grid_resolution", type=int, default=50, help="Grid resolution for density")
    parser.add_argument("--bandwidth", type=float, default=0.05, help="Bandwidth for density estimation")
    parser.add_argument("--min_density_threshold", type=float, default=0.001, help="Minimum density threshold")
    parser.add_argument("--fallback_ratio", type=float, default=0.1, help="Fallback ratio for uniform sampling")
    parser.add_argument("--plot_density", action="store_true", default=False, help="Generate density plots")
    parser.add_argument("--auto_disable_density", action="store_true", default=False, help="Auto-disable for large datasets")
    parser.add_argument("--use_fast_density", action="store_true", default=True, help="Use fast density estimation")
    parser.add_argument("--density_method", type=str, default="auto", choices=["auto", "kde", "histogram", "clustering"], help="Density estimation method")
    parser.add_argument("--data_batch_size", type=int, default=50000, help="Data loss batch size")
    
    # ===================== GEOMETRY-AWARE TRAINING ARGUMENTS =====================
    parser.add_argument("--colloc_sampling_mode", type=str, choices=["global", "vessel", "hybrid"], default="hybrid", help="Focus physics collocation points on vessel interiors/walls instead of the entire domain")
    parser.add_argument("--colloc_vessel_fraction", type=float, default=0.85, help="Fraction of collocation points forced inside vessels when using vessel-aware sampling")
    parser.add_argument("--colloc_wall_fraction", type=float, default=0.15, help="Fraction of collocation points snapped near vessel walls for sharp boundary behaviour")
    
    # ===================== POST-PROCESSING ARGUMENTS =====================
    parser.add_argument("--velocity_clip", type=float, default=0.1, help="Velocity clipping value")
    parser.add_argument("--interp_factor", type=float, default=-1, help="Interpolation factor (<=0 skips interpolation, default auto-calculated)")
    parser.add_argument("--smooth_factor", type=int, default=20, help="Smoothing window size")
    parser.add_argument("--res", type=float, default=10.0, help="Resolution factor")
    parser.add_argument("--max_linking_distance", type=float, default=2.0, help="Max linking distance (pixels)")
    parser.add_argument("--savgol_polyorder", type=int, default=2, help="Savitzky-Golay polynomial order")
    
    # ===================== PERFORMANCE ARGUMENTS =====================
    parser.add_argument("--interp_batch_size", type=int, default=1000, help="Interpolation batch size")
    parser.add_argument("--interp_gc_frequency", type=int, default=50, help="Interpolation GC frequency")
    parser.add_argument("--interp_fast_mode", action="store_true", default=False, help="Fast interpolation mode")
    parser.add_argument("--unnorm_batch_size", type=int, default=5000, help="Unnormalization batch size")
    parser.add_argument("--unnorm_gc_frequency", type=int, default=20, help="Unnormalization GC frequency")
    parser.add_argument("--unnorm_vectorized", action="store_true", default=True, help="Vectorized unnormalization")
    parser.add_argument("--gpu_memory_fraction", type=float, default=0.95, help="GPU memory fraction")
    
    # ===================== OPTIMIZATION ARGUMENTS =====================
    parser.add_argument("--use_optimized_physics", action="store_true", default=True, help="Optimized physics loss")
    parser.add_argument("--use_vectorized_velocity", action="store_true", default=True, help="Vectorized velocity computation")
    
    # ===================== VESSEL MASK ARGUMENTS =====================
    parser.add_argument("--vessel_mask_path", type=str, default="INSERT", help="Path to vessel mask .npz file (omit / set to None to disable)")
    parser.add_argument("--wall_loss_weight", type=float, default=0.3, help="Weight for wall boundary loss (penalizes velocities at vessel walls)")
    parser.add_argument("--outside_vessel_loss_weight", type=float, default=0.3, help="Weight for outside-vessel velocity penalty loss (penalizes ||u||^2 outside vessels)")
    parser.add_argument("--n_wall_samples", type=int, default=500, help="Number of wall points to sample per epoch")
    parser.add_argument("--n_outside_samples", type=int, default=500, help="Number of outside-vessel points to sample per epoch")

    args = parser.parse_args()
    main(args) 