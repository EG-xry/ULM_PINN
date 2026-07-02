#!/usr/bin/env python3
"""
main.py - end-to-end ULM-PINN pipeline for a single dataset.

Given a set of microbubble localizations (x, z, t), this:
  1. loads and normalizes the detections to [0, 1]^3,
  2. seeds a velocity prior with a Kalman / Hungarian / greedy tracker,
  3. builds a (voxel mean) velocity target,
  4. trains a physics informed network with a data-only warmup followed by a
     Stokes flow physics ramp (Huber data loss),
  5. re-tracks the detections with the learned velocity field (predictive cost),
  6. smooths, interpolates, unnormalizes, and exports the tracks.

Input is a MATLAB .mat or CSV file of localizations. All tunable parameters are
exposed as command-line options (see --help).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from data_loading import load_matlab_coordinates, load_tracking_data
from tracking import compute_track_velocities, extract_tracks_hungarian_initial
from greedy_tracking import greedy_nn_track
from kalman_tracking import kalman_track
from velocity_target import build_voxel_mean_velocity_field, train_val_split, evaluate_fit
from pinn_model import PINN, train_pinn
from predictive_cost import extract_tracks_predictive_pinn
from post_processing import interpolate_tracks, unnormalize_tracks, save_tracks_mat


def pick_device(choice):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if choice in ("cpu", "cuda", "mps"):
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_detections(path):
    """Return (x_norm [N,3], orig_bounds) from a .mat or .csv localization file."""
    ext = os.path.splitext(path)[1].lower()
    loader = load_matlab_coordinates if ext == ".mat" else load_tracking_data
    X, Z, T, _, orig_bounds = loader(path)
    x_norm = np.column_stack([X, Z, T]).astype(np.float32)
    return x_norm, orig_bounds


def seed_tracks(x_norm, method, gate, min_length, kalman_q):
    if method == "kalman":
        return kalman_track(x_norm, max_distance=gate, min_length=min_length,
                            max_miss=0, process_noise_q=kalman_q, measurement_noise_r=1e-4)
    if method == "hungarian":
        return extract_tracks_hungarian_initial(
            x_norm, time_precision=0.001, max_geo_radius=gate, min_length=min_length,
            max_tracks_per_frame=500000, cost_threshold=10.0, max_matrix_size=500_000_000)
    return greedy_nn_track(x_norm, max_distance=gate, min_length=min_length, max_gap=0, k_neighbors=10)


def tracks_to_rows(tracks):
    """Flatten a list of [x,z,t] tracks into a [track_id, x, z, t] array."""
    rows = []
    for tid, tr in enumerate(tracks):
        for p in tr:
            rows.append([tid, p[0], p[1], p[2]])
    return np.asarray(rows, dtype=np.float64)


# ============================================================================
# TUNABLE PARAMETERS
# ----------------------------------------------------------------------------
# Every hyperparameter of the pipeline is defined below as a command-line
# option, grouped by stage (Data / Velocity seed / Velocity target / Network /
# Training / Re-tracking / Post-processing). Change a default here, or override
# it on the command line. This function is the single place to tune the model
# ============================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="End-to-end ULM-PINN tracking pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("Data")
    g.add_argument("--input", required=True, help="Localization file (.mat or .csv) with x, z, t")
    g.add_argument("--out_dir", default="ulm_pinn_out", help="Output directory")
    g.add_argument("--frame_skip", type=int, default=1, help="Keep every k-th frame before seeding")
    g.add_argument("--max_points", type=int, default=4_000_000, help="Cap on training velocity points")
    g.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])

    g = p.add_argument_group("Velocity seed")
    g.add_argument("--seed", default="kalman", choices=["kalman", "hungarian", "greedy"],
                   help="Tracker used to seed the velocity prior")
    g.add_argument("--seed_kalman_q", type=float, default=5000.0, help="Kalman process noise q for the seed")
    g.add_argument("--seed_min_length", type=int, default=10, help="Minimum seed-track length")
    g.add_argument("--seed_gate", type=float, default=0.055,
                   help="Seed linking gate in normalized units (scaled by frame_skip)")

    g = p.add_argument_group("Velocity target")
    g.add_argument("--bin", type=int, default=1, choices=[0, 1],
                   help="1=voxel mean binned target (denoised); 0=raw per detection")
    g.add_argument("--nx", type=int, default=96, help="Voxel bins along x")
    g.add_argument("--nz", type=int, default=96, help="Voxel bins along z")
    g.add_argument("--nt", type=int, default=8, help="Voxel bins along t")
    g.add_argument("--min_pts", type=int, default=8, help="Minimum detections per voxel to keep it")
    g.add_argument("--huber_delta", type=float, default=3.0, help="Huber delta for the data loss")

    g = p.add_argument_group("Network")
    g.add_argument("--hidden_layers", type=int, default=8)
    g.add_argument("--hidden_size", type=int, default=128)
    g.add_argument("--activation", default="tanh", choices=["tanh", "sine"])
    g.add_argument("--omega_0", type=float, default=30.0, help="Frequency scale for the 'sine' (SIREN) activation")

    g = p.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=50000, help="Total training epochs")
    g.add_argument("--data_only_epochs", type=int, default=35000, help="Data-only warmup before physics")
    g.add_argument("--lr", type=float, default=8e-4, help="Adam learning rate")
    g.add_argument("--beta", type=float, default=1.0, help="Physics loss weight")
    g.add_argument("--n_colloc", type=int, default=16384, help="Collocation points per epoch")
    g.add_argument("--physics_mode", default="full", choices=["full", "divergence_only", "none"])
    g.add_argument("--physics_ramp_epochs", type=int, default=12000,
                   help="Epochs over which the physics weight ramps up after the warmup")
    g.add_argument("--physics_ramp_start", type=float, default=0.1, help="Physics ramp start scale")
    g.add_argument("--physics_ramp_end", type=float, default=1.0, help="Physics ramp end scale")
    g.add_argument("--phys_target_ratio", type=float, default=0.5,
                   help="Auto start physics target: phys_loss ~= ratio * data_loss")
    g.add_argument("--pressure_gauge_weight", type=float, default=0.1,
                   help="Weight of the mean-pressure gauge constraint")
    g.add_argument("--floor_tolerance", type=float, default=0.10,
                   help="Data loss floor tolerance before physics is throttled")
    g.add_argument("--floor_decay", type=float, default=0.3, help="Data loss floor decay (->1 throttles less)")
    g.add_argument("--data_batch_size", type=int, default=50000)
    g.add_argument("--val_frac", type=float, default=0.1, help="Held out fraction for the fit guard")
    g.add_argument("--print_every", type=int, default=5000)
    g.add_argument("--seed_rng", type=int, default=101, help="Random seed for weights/collocation")

    # Re-track cost = w_pred*|predicted_pos - detection| + w_vel*|implied_vel - PINN_vel| + w_geo*|endpoint - detection|
    # Default 1/0/0 = pure PINN field cost, the in vivo setting gates on the field-predicted
    # next position, no velocity or geometric term. The in-silico Strategy B blend is 0.6/0.3/0.1
    g = p.add_argument_group("Re-tracking (PINN field cost)")
    g.add_argument("--w_pred", type=float, default=1.0,
                   help="Weight on predicted-position error (the PINN field cost)")
    g.add_argument("--w_vel", type=float, default=0.0,
                   help="Weight on velocity consistency |implied velocity - PINN velocity| (Strategy B uses 0.3)")
    g.add_argument("--w_geo", type=float, default=0.0,
                   help="Weight on raw geometric distance to the track endpoint (Strategy B uses 0.1)")
    g.add_argument("--retrack_gate", type=float, default=0.055, help="Candidate gating radius (normalized)")
    g.add_argument("--min_length", type=int, default=5, help="Minimum retained track length")
    g.add_argument("--cost_threshold", type=float, default=10.0, help="Assignment cost cutoff")

    g = p.add_argument_group("Post-processing")
    g.add_argument("--smooth_factor", type=int, default=20, help="Savitzky-Golay window")
    g.add_argument("--interp_factor", type=float, default=-1,
                   help="Interpolation step (<=0 auto from res/max_linking_distance)")
    g.add_argument("--res", type=float, default=10.0, help="Super-resolution factor")
    g.add_argument("--max_linking_distance", type=float, default=2.0)
    return p


def main():
    args = build_parser().parse_args()
    device = pick_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"== ULM-PINN == device={device}  input={os.path.basename(args.input)}", flush=True)

    # 1. load + normalize
    x_norm, orig_bounds = load_detections(args.input)
    print(f"   loaded {len(x_norm):,} detections", flush=True)

    if args.frame_skip > 1:
        uf = np.unique(x_norm[:, 2])
        keep = set(uf[::args.frame_skip].tolist())
        x_norm = x_norm[np.isin(x_norm[:, 2], list(keep))]
        t = x_norm[:, 2]
        x_norm[:, 2] = (t - t.min()) / (t.max() - t.min() + 1e-9)
        print(f"   frame_skip={args.frame_skip}: kept {len(x_norm):,} detections", flush=True)
    gate = args.seed_gate * args.frame_skip

    # 2. seed velocity prior
    t0 = time.time()
    seed = seed_tracks(x_norm, args.seed, gate, args.seed_min_length, args.seed_kalman_q)
    print(f"   seed: {len(seed):,} tracks ({args.seed}) in {time.time()-t0:.1f}s", flush=True)

    pos, vel = compute_track_velocities(seed, savgol_polyorder=2, use_vectorized=True,
                                        min_track_length=args.seed_min_length)
    vm = vel.mean(0, keepdims=True)
    vs = vel.std(0, keepdims=True) + 1e-9
    vel = np.clip(vel, vm - 2.5 * vs, vm + 2.5 * vs)

    # 3. velocity target
    pos_b, vel_b = pos, vel
    if args.bin == 1:
        binned = build_voxel_mean_velocity_field(pos, vel, nx=args.nx, nz=args.nz, nt=args.nt,
                                                 min_pts=args.min_pts)
        if len(binned["pos_kept"]) >= max(50, int(0.01 * len(pos))):
            pos_b, vel_b = binned["pos_kept"], binned["vel_target_kept"]
            print(f"   binned target: {len(pos_b):,} pts, coherent_fraction={binned['coherent_fraction']:.3f}",
                  flush=True)
        else:
            print(f"   WARNING: voxel binning kept only {len(binned['pos_kept'])} of {len(pos):,} points; "
                  f"using the raw per detection target instead. Lower --nx/--nz/--nt or --min_pts for sparse data.",
                  flush=True)

    if len(pos_b) > args.max_points:
        rs = np.random.RandomState(42)
        idx = np.sort(rs.choice(len(pos_b), args.max_points, replace=False))
        pos_b, vel_b = pos_b[idx], vel_b[idx]

    tr_idx, va_idx = train_val_split(len(pos_b), val_frac=args.val_frac, seed=0)
    pos_va, vel_va = pos_b[va_idx], vel_b[va_idx]
    x_data = torch.tensor(pos_b[tr_idx], dtype=torch.float32, device=device)
    v_meas = torch.tensor(vel_b[tr_idx], dtype=torch.float32, device=device)
    print(f"   train={len(tr_idx):,}  held out guard={len(va_idx):,}", flush=True)

    # 4. train
    torch.manual_seed(args.seed_rng)
    np.random.seed(args.seed_rng)
    model = PINN(input_dim=3, output_dim=3, hidden_layers=args.hidden_layers,
                 hidden_size=args.hidden_size, activation=args.activation, omega_0=args.omega_0).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def guard(ep, m, normalizer):
        met = evaluate_fit(m, pos_va, vel_va, normalizer, device)
        print(f"   [guard ep{ep:05d}] held out R^2={met['r2']:+.3f} "
              f"|v|pred={met['pred_speed_mean']:.3f} vs tgt {met['target_speed_mean']:.3f}", flush=True)

    data_only = args.epochs if args.physics_mode == "none" else args.data_only_epochs
    train_kwargs = dict(
        domain={"x": (0, 1), "z": (0, 1), "t": (0, 1)},
        epochs=args.epochs, data_only_epochs=data_only,
        n_colloc=args.n_colloc, beta=args.beta, physics_mode=args.physics_mode,
        physics_ramp_start=args.physics_ramp_start, physics_ramp_end=args.physics_ramp_end,
        physics_ramp_epochs=args.physics_ramp_epochs,
        physics_ramp_auto_start=True, physics_ramp_auto_target_ratio=args.phys_target_ratio,
        physics_ramp_auto_clip=(1e-12, 1.0),
        data_loss_floor=True, data_loss_floor_tolerance=args.floor_tolerance,
        data_loss_floor_decay=args.floor_decay,
        pressure_gauge_weight=args.pressure_gauge_weight,
        data_loss_weight=1.0, use_density_guided=True,
        data_batch_size=args.data_batch_size,
        data_loss_fn="huber", huber_delta=args.huber_delta, orig_bounds=orig_bounds,
        print_every=args.print_every, epoch_callback=guard,
    )
    # keep only kwargs the installed train_pinn accepts
    import inspect
    sig = set(inspect.signature(train_pinn).parameters.keys())
    train_kwargs = {k: v for k, v in train_kwargs.items() if k in sig}

    t0 = time.time()
    d_hist, p_hist, t_hist = train_pinn(model, optimizer, x_data, v_meas, **train_kwargs)
    print(f"   training done in {time.time()-t0:.1f}s; final data_loss={d_hist[-1]:.4e}", flush=True)

    # 5. re-track with the learned field, predictive cost
    model.eval()
    tracks = extract_tracks_predictive_pinn(
        x_norm, model, w_pred=args.w_pred, w_vel=args.w_vel, w_geo=args.w_geo,
        max_geo_radius=args.retrack_gate, min_length=args.min_length,
        cost_threshold=args.cost_threshold)
    print(f"   re-tracked: {len(tracks):,} tracks", flush=True)

    # 6. post-process + export
    interp_factor = args.interp_factor if args.interp_factor > 0 else None
    interp = interpolate_tracks(tracks, interp_factor=interp_factor, smooth_factor=args.smooth_factor,
                                res=args.res, max_linking_distance=args.max_linking_distance,
                                savgol_polyorder=2)
    unnorm = unnormalize_tracks(interp, orig_bounds)
    mat_path = os.path.join(args.out_dir, "tracks.mat")
    save_tracks_mat(unnorm, mat_path, min_length=args.min_length)

    # tracks normalized + bounds for the brain-map renderer
    bounds_arr = np.array([orig_bounds["X_min"], orig_bounds["X_max"],
                           orig_bounds["Z_min"], orig_bounds["Z_max"],
                           orig_bounds["T_min"], orig_bounds["T_max"]], dtype=np.float64)
    np.savez(os.path.join(args.out_dir, "tracks.npz"),
             tracks=tracks_to_rows(tracks), orig_bounds=bounds_arr)

    norm_stats = None
    if getattr(model, "normalizer", None) is not None:
        norm_stats = {k: [float(v[0]), float(v[1])] for k, v in model.normalizer.stats.items()}
    torch.save(dict(model_state_dict=model.state_dict(), normalizer_stats=norm_stats,
                    orig_bounds=orig_bounds, config=vars(args)),
               os.path.join(args.out_dir, "pinn_checkpoint.pt"))

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(dict(n_detections=int(len(x_norm)), n_tracks=int(len(tracks)),
                       final_data_loss=float(d_hist[-1]), config=vars(args)), f, indent=2, default=str)
    print(f"   saved tracks + checkpoint to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
