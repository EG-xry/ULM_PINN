#!/usr/bin/env python3
"""
Kalman Filter Tracking Baseline for ULM Microbubble Tracking
=============================================================
Implements a standard constant velocity Kalman filter tracker with
Hungarian (linear sum) assignment - the classic approach used in ULM
literature (e.g., Christensen-Jeffries 2020, Heiles 2022).

Pipeline per frame:
  1. **Predict** all active track states forward (constant velocity model).
  2. **Gate**:   compute Euclidean distance between predicted positions
                 and new detections; reject pairs beyond max_distance.
  3. **Assign**: Hungarian algorithm on the gated cost matrix.
  4. **Update**: Kalman measurement update for matched tracks.
  5. **Birth**:  start new tracks from unmatched detections.
  6. **Death**:  kill tracks unmatched for > max_miss consecutive frames.
  7. **Filter**: keep only tracks with ≥ min_length detections.

State vector:  [x, z, vx, vz]   (position + velocity)
Measurement:   [x, z]           (position only)
"""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def kalman_track(x_norm, max_distance=0.06, min_length=5, max_miss=3,
                 process_noise_q=1e-4, measurement_noise_r=1e-4):
    """
    Standard constant velocity Kalman filter tracker with Hungarian
    assignment.

    Parameters
    ----------
    x_norm : np.ndarray (N, 3)
        Normalized detections [x, z, t] in [0,1]^3.
    max_distance : float
        Gating distance: max Euclidean distance (in normalized space)
        between a predicted track position and a detection to allow
        association.
    min_length : int
        Minimum track length to keep.
    max_miss : int
        Number of consecutive frames without a matched detection before
        a track is terminated.
    process_noise_q : float
        Process noise intensity for the constant velocity model.
        Controls how much velocity can change per step.
    measurement_noise_r : float
        Measurement noise variance (diagonal).

    Returns
    -------
    tracks : list of list of [x, z, t] tuples
        Same format as extract_tracks_hungarian_initial output.
    """

    # round t, avoid fp frame-grouping jitter
    unique_times = np.unique(np.round(x_norm[:, 2], decimals=6))
    unique_times.sort()

    frame_groups = defaultdict(list)
    for i in range(len(x_norm)):
        t_rounded = round(float(x_norm[i, 2]), 6)
        frame_groups[t_rounded].append(i)

    sorted_frames = sorted(frame_groups.keys())
    n_frames = len(sorted_frames)

    # median inter-frame dt, robust to gaps
    if n_frames > 1:
        dts = np.diff(sorted_frames)
        dt = float(np.median(dts))
    else:
        dt = 1.0

    # Kalman matrices, constant velocity model
    # State:       [x, z, vx, vz]
    # Transition:  x' = x + vx*dt, z' = z + vz*dt
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0,  1, 0],
        [0, 0,  0, 1],
    ], dtype=np.float64)

    # Measurement matrix:  observe [x, z]
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float64)

    # Process noise, piecewise white noise jerk model
    Q = process_noise_q * np.array([
        [dt**3/3, 0,       dt**2/2, 0      ],
        [0,       dt**3/3, 0,       dt**2/2],
        [dt**2/2, 0,       dt,      0      ],
        [0,       dt**2/2, 0,       dt     ],
    ], dtype=np.float64)

    # Measurement noise
    R = measurement_noise_r * np.eye(2, dtype=np.float64)

    # Initial covariance
    P0 = np.diag([measurement_noise_r,  measurement_noise_r,
                  process_noise_q * 10, process_noise_q * 10])

    # Each active track, dict with keys 'state', 'P', 'history', 'miss'
    active_tracks = []
    finished_tracks = []
    next_track_id = 0

    BIG_COST = 1e6  # cost for gated out assignments

    for frame_idx, t_key in enumerate(sorted_frames):
        det_indices = frame_groups[t_key]
        det_arr = x_norm[det_indices]  # (M, 3)
        n_det = len(det_arr)

        # PREDICT all active tracks
        predicted_positions = np.empty((len(active_tracks), 2))
        for i, trk in enumerate(active_tracks):
            trk['state'] = F @ trk['state']
            trk['P'] = F @ trk['P'] @ F.T + Q
            predicted_positions[i] = trk['state'][:2]

        n_trk = len(active_tracks)

        if n_trk == 0 and n_det == 0:
            continue

        if n_trk > 0 and n_det > 0:
            # large frames, KD-tree gated sparse assign
            # avoids O(n_trk*n_det) dense matrix
            _use_sparse = (n_trk * n_det) > 5_000_000  # ~2236 × 2236

            if _use_sparse:
                from scipy.spatial import cKDTree
                det_xy = det_arr[:, :2].astype(np.float64)
                tree = cKDTree(det_xy)
                neighbors = tree.query_ball_point(predicted_positions, r=max_distance)

                _rows, _cols, _costs = [], [], []
                for trk_idx, det_list in enumerate(neighbors):
                    for det_idx in det_list:
                        d = np.sqrt(np.sum((predicted_positions[trk_idx] - det_xy[det_idx]) ** 2))
                        _rows.append(trk_idx)
                        _cols.append(det_idx)
                        _costs.append(d)

                if len(_rows) > 0:
                    # reduced cost matrix, gated entries only
                    _rows = np.array(_rows, dtype=np.int32)
                    _cols = np.array(_cols, dtype=np.int32)
                    _costs = np.array(_costs, dtype=np.float64)

                    u_trks = np.unique(_rows)
                    u_dets = np.unique(_cols)
                    trk_map = {v: i for i, v in enumerate(u_trks)}
                    det_map = {v: i for i, v in enumerate(u_dets)}

                    # Dense matrix over involved pairs only
                    small_n = len(u_trks)
                    small_m = len(u_dets)
                    small_cost = np.full((small_n, small_m), BIG_COST, dtype=np.float64)
                    for r, c, co in zip(_rows, _cols, _costs):
                        small_cost[trk_map[r], det_map[c]] = co

                    sm_row, sm_col = linear_sum_assignment(small_cost)
                    row_ind = u_trks[sm_row]
                    col_ind = u_dets[sm_col]
                    _cost_lookup = small_cost
                    _trk_map = trk_map
                    _det_map = det_map
                else:
                    row_ind = np.array([], dtype=np.int32)
                    col_ind = np.array([], dtype=np.int32)
                    _cost_lookup = None
            else:
                # dense, small frames
                diff = predicted_positions[:, None, :] - det_arr[None, :, :2]
                cost_matrix = np.sqrt(np.sum(diff ** 2, axis=2))  # (n_trk, n_det)
                cost_matrix[cost_matrix > max_distance] = BIG_COST
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                _cost_lookup = None

            matched_trk = set()
            matched_det = set()

            for r, c in zip(row_ind, col_ind):
                # sparse reduced-matrix lookup; dense cost_matrix
                if _cost_lookup is not None:
                    if r in _trk_map and c in _det_map:
                        pair_cost = _cost_lookup[_trk_map[r], _det_map[c]]
                    else:
                        pair_cost = BIG_COST
                else:
                    pair_cost = cost_matrix[r, c]

                if pair_cost < BIG_COST:
                    matched_trk.add(r)
                    matched_det.add(c)

                    # KALMAN UPDATE
                    trk = active_tracks[r]
                    z_meas = det_arr[c, :2].astype(np.float64)
                    y_innov = z_meas - H @ trk['state']
                    S = H @ trk['P'] @ H.T + R
                    K = trk['P'] @ H.T @ np.linalg.inv(S)
                    trk['state'] = trk['state'] + K @ y_innov
                    trk['P'] = (np.eye(4) - K @ H) @ trk['P']
                    trk['miss'] = 0
                    trk['history'].append(
                        (float(det_arr[c, 0]),
                         float(det_arr[c, 1]),
                         float(det_arr[c, 2]))
                    )
        else:
            matched_trk = set()
            matched_det = set()

        # unmatched, bump miss, kill past max_miss
        tracks_to_keep = []
        for i, trk in enumerate(active_tracks):
            if i not in matched_trk:
                trk['miss'] += 1
                if trk['miss'] > max_miss:
                    finished_tracks.append(trk['history'])
                    continue
            tracks_to_keep.append(trk)
        active_tracks = tracks_to_keep

        # Birth, new tracks from unmatched detections
        for j in range(n_det):
            if j not in matched_det:
                state0 = np.array([
                    float(det_arr[j, 0]),
                    float(det_arr[j, 1]),
                    0.0,
                    0.0,
                ], dtype=np.float64)
                active_tracks.append({
                    'state': state0,
                    'P': P0.copy(),
                    'history': [(float(det_arr[j, 0]),
                                 float(det_arr[j, 1]),
                                 float(det_arr[j, 2]))],
                    'miss': 0,
                    'id': next_track_id,
                })
                next_track_id += 1

        if (frame_idx + 1) % 100 == 0 or frame_idx == n_frames - 1:
            print(f"  Frame {frame_idx+1}/{n_frames}: "
                  f"{len(active_tracks)} active tracks")

    # Flush remaining active tracks
    for trk in active_tracks:
        finished_tracks.append(trk['history'])

    all_tracks = [t for t in finished_tracks if len(t) >= min_length]

    # history tuples -> list-of-lists
    tracks_out = []
    for history in all_tracks:
        track = [[h[0], h[1], h[2]] for h in history]
        tracks_out.append(track)

    total_points = sum(len(t) for t in tracks_out)
    print(f"  Kalman tracking: {len(tracks_out)} tracks, "
          f"{total_points:,} points ({100*total_points/len(x_norm):.1f}% coverage)")

    return tracks_out
