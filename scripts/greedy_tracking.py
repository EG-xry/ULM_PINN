"""
greedy_tracking.py - Fast greedy nearest neighbor tracker for dense ULM data

Designed for high density in vivo data (>5K detections/frame) where optimal
Hungarian/LAPJV assignment is too slow. Uses KD-tree + sorted greedy matching

Matches the simpletracker.m workflow from PALA but with greedy instead of Munkres:
1. Group detections by frame
2. For each frame pair, find candidate pairs within max_linking_distance
3. Sort by distance, assign greedily (shortest first, no conflicts)
4. Track birth/death/gap closing
5. Filter by minimum length

Speed: ~0.5s per frame at 18.5K detections → ~7 min for 800 frames
"""

import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict


def greedy_nn_track(x_norm, max_distance=0.027, min_length=5, max_gap=0, k_neighbors=10):
    """
    Greedy nearest neighbor tracker
    
    Parameters
    ----------
    x_norm : np.ndarray (N, 3)
        Detections [x, z, t] in normalized [0,1]^3.
    max_distance : float
        Maximum linking distance (normalized space).
    min_length : int
        Minimum track length to keep.
    max_gap : int
        Maximum frame gap for gap closing (0 = no gap closing).
    k_neighbors : int
        Number of nearest neighbors to consider per track endpoint.
    
    Returns
    -------
    tracks : list of list of [x, z, t]
    """
    # Group by frame
    unique_times = np.unique(np.round(x_norm[:, 2], decimals=6))
    unique_times.sort()
    
    frame_groups = defaultdict(list)
    for i in range(len(x_norm)):
        t = round(float(x_norm[i, 2]), 6)
        frame_groups[t].append(i)
    
    sorted_frames = sorted(frame_groups.keys())
    n_frames = len(sorted_frames)
    
    if n_frames == 0:
        return []
    
    # Active tracks: list of {'history': [(x,z,t),...], 'last_pos': (x,z), 'miss': int}
    active_tracks = []
    finished_tracks = []

    first_indices = frame_groups[sorted_frames[0]]
    for idx in first_indices:
        pt = x_norm[idx]
        active_tracks.append({
            'history': [(float(pt[0]), float(pt[1]), float(pt[2]))],
            'last_pos': pt[:2].copy(),
            'miss': 0,
        })
    
    # Process subsequent frames
    for frame_idx in range(1, n_frames):
        det_indices = frame_groups[sorted_frames[frame_idx]]
        det_arr = x_norm[det_indices]
        n_det = len(det_arr)
        n_trk = len(active_tracks)
        
        if n_trk == 0 and n_det == 0:
            continue
        
        matched_trk = set()
        matched_det = set()
        
        if n_trk > 0 and n_det > 0:
            # Get track endpoints
            endpoints = np.array([t['last_pos'] for t in active_tracks])
            det_xy = det_arr[:, :2]
            
            # KD tree on detections
            tree = cKDTree(det_xy)
            
            # k nearest per endpoint
            k = min(k_neighbors, n_det)
            dists, idxs = tree.query(endpoints, k=k, distance_upper_bound=max_distance)
            
            # flatten to sorted candidate list
            if k == 1:
                dists = dists.reshape(-1, 1)
                idxs = idxs.reshape(-1, 1)
            
            valid = np.isfinite(dists) & (idxs < n_det)
            trk_ids = np.repeat(np.arange(n_trk), k)[valid.ravel()]
            det_ids = idxs.ravel()[valid.ravel()]
            pair_dists = dists.ravel()[valid.ravel()]
            
            # sort by distance; greedy = shortest first
            order = np.argsort(pair_dists)
            
            used_trk = np.zeros(n_trk, dtype=bool)
            used_det = np.zeros(n_det, dtype=bool)
            
            for idx in order:
                t = trk_ids[idx]
                d = det_ids[idx]
                if not used_trk[t] and not used_det[d]:
                    used_trk[t] = True
                    used_det[d] = True
                    matched_trk.add(t)
                    matched_det.add(d)
                    
                    # Update track
                    pt = det_arr[d]
                    active_tracks[t]['history'].append(
                        (float(pt[0]), float(pt[1]), float(pt[2])))
                    active_tracks[t]['last_pos'] = pt[:2].copy()
                    active_tracks[t]['miss'] = 0
        
        # Handle unmatched tracks
        new_active = []
        for i, trk in enumerate(active_tracks):
            if i not in matched_trk:
                trk['miss'] += 1
                if trk['miss'] > max_gap:
                    finished_tracks.append(trk['history'])
                    continue
            new_active.append(trk)
        
        # Birth, new tracks from unmatched detections
        for j in range(n_det):
            if j not in matched_det:
                pt = det_arr[j]
                new_active.append({
                    'history': [(float(pt[0]), float(pt[1]), float(pt[2]))],
                    'last_pos': pt[:2].copy(),
                    'miss': 0,
                })
        
        active_tracks = new_active

        if (frame_idx + 1) % 100 == 0:
            print(f"  Frame {frame_idx+1}/{n_frames}: {len(active_tracks)} active tracks")
    
    # Flush remaining
    for trk in active_tracks:
        finished_tracks.append(trk['history'])
    
    # Filter by min_length
    tracks = [t for t in finished_tracks if len(t) >= min_length]
    
    total_pts = sum(len(t) for t in tracks)
    print(f"  Greedy NN tracking: {len(tracks)} tracks, "
          f"{total_pts:,} points ({100*total_pts/len(x_norm):.1f}% coverage)")
    
    return tracks
