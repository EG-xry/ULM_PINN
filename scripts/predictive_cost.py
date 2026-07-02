#!/usr/bin/env python3
"""
Predictive PINN re-tracking.

Provides ``extract_tracks_predictive_pinn``: PINN guided re-tracking that uses
the learned velocity field to PREDICT the next position and penalises links
that deviate from the prediction.

Cost = w_pred * |predicted_pos - detection|
     + w_vel  * |v_implied - v_pinn|
     + w_geo  * |endpoint - detection|

where v_implied = (detection - endpoint) / dt

Candidate gating is centred on the PINN-predicted position (not the
track endpoint), so the search region follows the learned flow field.
"""

import time
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from pinn_model import PINN, DataNormalizer
from tracking import predict_velocities_batch_optimized


def extract_tracks_predictive_pinn(
    positions,
    model,
    w_pred=0.6,
    w_vel=0.3,
    w_geo=0.1,
    max_geo_radius=0.055,
    dt=1.0,
    batch_size=512,
    min_length=5,
    cost_threshold=10.0,
):
    """
    PINN guided tracking with predictive gating and velocity-consistency cost.

    Parameters
    ----------
    positions : np.ndarray (N, 3)
        Normalised detections [x, z, t] in [0, 1]^3.
    model : PINN
        Trained PINN model (eval mode).
    w_pred : float
        Weight for prediction error |predicted_pos - detection|.
    w_vel : float
        Weight for velocity consistency |v_implied - v_pinn|.
    w_geo : float
        Weight for raw geometric distance |endpoint - detection|.
    max_geo_radius : float
        Search radius around the PREDICTED position.
    dt : float
        Fallback time step (overridden by actual frame dt when available).
    batch_size : int
        PINN query batch size.
    min_length : int
        Minimum track length to keep.
    cost_threshold : float
        Maximum assignment cost.

    Returns
    -------
    tracks : list of list of [x, z, t]
    """
    print("=== Predictive PINN Tracking ===")
    print(f"  w_pred={w_pred:.2f}, w_vel={w_vel:.2f}, w_geo={w_geo:.2f}")
    print(f"  max_geo_radius={max_geo_radius}, min_length={min_length}, "
          f"cost_threshold={cost_threshold}")

    x_norm = np.asarray(positions, dtype=np.float32)
    device = next(model.parameters()).device
    model.eval()

    # --- Frame grouping ---
    unique_times = np.unique(np.round(x_norm[:, 2], decimals=6))
    unique_times.sort()
    frame_groups = defaultdict(list)
    for i in range(len(x_norm)):
        t_rounded = round(float(x_norm[i, 2]), 6)
        frame_groups[t_rounded].append(i)
    sorted_frames = sorted(frame_groups.keys())

    if len(sorted_frames) < 2:
        print("WARNING: fewer than 2 frames - nothing to track.")
        return []

    print(f"  {len(sorted_frames)} frames, {len(x_norm)} detections")

    # --- Data structures ---
    # track_id -> list of detection indices
    track_points = {}
    # detection_index -> track_id   active tracks = last frame's endpoints
    active_tracks = {}
    next_track_id = 0

    # Initialise with first frame
    for idx in frame_groups[sorted_frames[0]]:
        track_points[next_track_id] = [idx]
        active_tracks[idx] = next_track_id
        next_track_id += 1

    start_time = time.time()

    for fi in range(1, len(sorted_frames)):
        t_curr = sorted_frames[fi]
        t_prev = sorted_frames[fi - 1]
        dt_actual = t_curr - t_prev
        if dt_actual <= 0 or not np.isfinite(dt_actual):
            dt_actual = dt

        det_indices_curr = frame_groups[t_curr]
        if len(det_indices_curr) == 0:
            continue

        if not active_tracks:
            for idx in det_indices_curr:
                track_points[next_track_id] = [idx]
                active_tracks[idx] = next_track_id
                next_track_id += 1
            continue

        # track endpoints, last det per active track
        endpoint_det_ids = list(active_tracks.keys())
        track_ids_ordered = [active_tracks[d] for d in endpoint_det_ids]
        endpoints = x_norm[endpoint_det_ids]  # (M, 3)

        # Current frame detections
        det_positions = x_norm[det_indices_curr]  # (D, 3)
        det_xy = det_positions[:, :2]

        # --- 1. Query PINN for velocity at track endpoints ---
        u_pred, v_pred = predict_velocities_batch_optimized(
            model, endpoints, t_prev, batch_size, device
        )

        # --- 2. Compute predicted next positions ---
        predicted_xy = np.column_stack([
            endpoints[:, 0] + u_pred * dt_actual,
            endpoints[:, 1] + v_pred * dt_actual,
        ])  # (M, 2)

        endpoint_xy = endpoints[:, :2]  # (M, 2)

        # --- 3. Build candidate set around PREDICTED positions ---
        if len(det_xy) == 0 or len(predicted_xy) == 0:
            active_tracks = {}
            for idx in det_indices_curr:
                track_points[next_track_id] = [idx]
                active_tracks[idx] = next_track_id
                next_track_id += 1
            continue

        tree_det = cKDTree(det_xy)
        candidate_lists = tree_det.query_ball_point(predicted_xy, r=max_geo_radius, p=2)

        # flatten to sparse (row, col) pairs
        rows, cols = [], []
        for track_i, det_list in enumerate(candidate_lists):
            for det_j in det_list:
                rows.append(track_i)
                cols.append(det_j)

        if len(rows) == 0:
            active_tracks = {}
            for idx in det_indices_curr:
                track_points[next_track_id] = [idx]
                active_tracks[idx] = next_track_id
                next_track_id += 1
            continue

        rows = np.array(rows, dtype=int)
        cols = np.array(cols, dtype=int)

        # --- 4. Compute three cost components (vectorised) ---

        # 4a. Prediction error |predicted_pos - detection|
        pred_err = np.linalg.norm(predicted_xy[rows] - det_xy[cols], axis=1)

        # 4b. Geometric distance |endpoint - detection|
        geo_dist = np.linalg.norm(endpoint_xy[rows] - det_xy[cols], axis=1)

        # 4c. Velocity consistency |v_implied - v_pinn|
        #     v_implied = (detection - endpoint) / dt
        v_implied = (det_xy[cols] - endpoint_xy[rows]) / (dt_actual + 1e-12)
        v_pinn = np.column_stack([u_pred[rows], v_pred[rows]])
        vel_err = np.linalg.norm(v_implied - v_pinn, axis=1)

        combined_cost = w_pred * pred_err + w_vel * vel_err + w_geo * geo_dist

        # --- 5. Filter by cost threshold, build minimal dense matrix, solve ---
        valid = combined_cost < cost_threshold
        if not np.any(valid):
            active_tracks = {}
            for idx in det_indices_curr:
                track_points[next_track_id] = [idx]
                active_tracks[idx] = next_track_id
                next_track_id += 1
            continue

        v_rows = rows[valid]
        v_cols = cols[valid]
        v_costs = combined_cost[valid]

        unique_tracks = np.unique(v_rows)
        unique_dets = np.unique(v_cols)

        track_map = {t: i for i, t in enumerate(unique_tracks)}
        det_map = {d: i for i, d in enumerate(unique_dets)}

        big_cost = cost_threshold * 10.0
        cost_matrix = np.full(
            (len(unique_tracks), len(unique_dets)), big_cost, dtype=np.float32
        )
        for r, c, cost in zip(v_rows, v_cols, v_costs):
            cost_matrix[track_map[r], det_map[c]] = cost

        row_assign, col_assign = linear_sum_assignment(cost_matrix)

        # --- 6. Update tracks ---
        new_active = {}
        used_dets = set()

        for ra, ca in zip(row_assign, col_assign):
            if cost_matrix[ra, ca] >= cost_threshold:
                continue
            orig_track_i = unique_tracks[ra]
            orig_det_j = unique_dets[ca]
            track_id = track_ids_ordered[orig_track_i]
            det_global = det_indices_curr[orig_det_j]
            track_points[track_id].append(det_global)
            new_active[det_global] = track_id
            used_dets.add(orig_det_j)

        # new tracks for unassigned dets
        for local_j, global_j in enumerate(det_indices_curr):
            if local_j not in used_dets:
                track_points[next_track_id] = [global_j]
                new_active[global_j] = next_track_id
                next_track_id += 1

        active_tracks = new_active

        if fi % 100 == 0:
            elapsed = time.time() - start_time
            progress = fi / len(sorted_frames)
            eta = (elapsed / progress - elapsed) if progress > 0 else 0
            print(f"  Frame {fi}/{len(sorted_frames)-1}  "
                  f"active={len(active_tracks)}  ETA={eta/60:.1f}min")

    # --- Convert to output format ---
    tracks = []
    for tid, idx_list in track_points.items():
        if len(idx_list) < min_length:
            continue
        track = [list(x_norm[i]) for i in idx_list]
        tracks.append(track)

    total_time = time.time() - start_time
    print(f"  Complete: {len(tracks)} tracks "
          f"(>= {min_length} pts) in {total_time:.1f}s")
    if tracks:
        total_pts = sum(len(t) for t in tracks)
        print(f"  Points in tracks: {total_pts:,} / {len(x_norm):,} "
              f"({100*total_pts/len(x_norm):.1f}%)")
        print(f"  Avg track length: {total_pts/len(tracks):.1f}")

    return tracks


def load_model(checkpoint_path, device):
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    h = ckpt.get("model_hparams", {})
    model = PINN(
        input_dim=int(h.get("input_dim", 3)),
        output_dim=int(h.get("output_dim", 3)),
        hidden_layers=int(h.get("hidden_layers", 5)),
        hidden_size=int(h.get("hidden_size", 64)),
        activation=str(h.get("activation", "tanh")),
        omega_0=float(h.get("omega_0", 30)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    stats = ckpt.get("normalizer_stats", None)
    if stats is not None:
        model.normalizer = DataNormalizer(stats)

    model.eval()
    return model, ckpt
