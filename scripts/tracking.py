import gc
import time

import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

lapjv_solver = None
LAP_PACKAGE = None
# if unique_tracks × unique_detections exceeds this, skip dense matrix, go greedy
# 400M entries (~1.6 GB float32) supports the full in vivo dataset
# (~18.5K tracks × 18.5K detections = 342M entries). The KD-tree pre-filtering in
# extract_tracks_hungarian_initial keeps the actual reduced matrix much smaller
# (only pairs within max_geo_radius), so even 18.5K/frame datasets reduce to
# ~18K × ~20 candidates = 360K entries, well within limits
SPARSE_ASSIGNMENT_OPTIMAL_MATRIX_THRESHOLD = 400_000_000
DENSE_MATRIX_MEMORY_LIMIT_MB = 1600  # per-frame ceiling; ~1.6 GB allows full in vivo


def _initialize_assignment_solver():
    """Attempt to import a LAP solver, falling back to scipy when needed."""
    try:
        from lapjv import lapjv as solver
        return solver, "lapjv"
    except ImportError:
        try:
            from lap import lapjv as solver
            return solver, "lap"
        except ImportError:
            return None, None


lapjv_solver, LAP_PACKAGE = _initialize_assignment_solver()

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

if lapjv_solver is not None:
    print(f"Using LAPJV solver from {LAP_PACKAGE}.")
elif linear_sum_assignment is not None:
    print("Using scipy.optimize.linear_sum_assignment as the assignment solver.")
else:
    print(
        "Error: No assignment solver available. Install lapjv, lap, or scipy to enable tracking."
    )


def get_assignment_solver_metadata():
    """
    Return assignment solver metadata for run provenance.
    """
    if lapjv_solver is not None:
        solver_mode = f"lapjv:{LAP_PACKAGE}"
    elif linear_sum_assignment is not None:
        solver_mode = "scipy_linear_sum_assignment"
    else:
        solver_mode = "unavailable"

    return {
        "solver_mode": solver_mode,
        "solver_matrix_threshold": int(SPARSE_ASSIGNMENT_OPTIMAL_MATRIX_THRESHOLD),
        "has_lapjv": bool(lapjv_solver is not None),
        "has_scipy": bool(linear_sum_assignment is not None),
    }


class EdgeCollector:
    """Collect edges as three parallel arrays instead of a list of tuples."""

    def __init__(self, initial_capacity=10000):
        self.capacity = initial_capacity
        self.size = 0
        self.rows = np.empty(initial_capacity, dtype=int)
        self.cols = np.empty(initial_capacity, dtype=int)
        self.data = np.empty(initial_capacity, dtype=np.float32)
    
    def _resize(self, new_capacity):
        """Resize arrays when needed."""
        new_rows = np.empty(new_capacity, dtype=int)
        new_cols = np.empty(new_capacity, dtype=int)
        new_data = np.empty(new_capacity, dtype=np.float32)
        
        new_rows[:self.size] = self.rows[:self.size]
        new_cols[:self.size] = self.cols[:self.size]
        new_data[:self.size] = self.data[:self.size]
        
        self.rows = new_rows
        self.cols = new_cols
        self.data = new_data
        self.capacity = new_capacity
    
    def add_edges(self, rows, cols, data):
        """Add multiple edges at once."""
        n_new = len(rows)
        if self.size + n_new > self.capacity:
            new_capacity = max(self.capacity * 2, self.size + n_new)
            self._resize(new_capacity)
        
        end_idx = self.size + n_new
        self.rows[self.size:end_idx] = rows
        self.cols[self.size:end_idx] = cols
        self.data[self.size:end_idx] = data
        self.size = end_idx
    
    def add_edge(self, row, col, data_val):
        """Add single edge."""
        if self.size >= self.capacity:
            self._resize(self.capacity * 2)
        
        self.rows[self.size] = row
        self.cols[self.size] = col
        self.data[self.size] = data_val
        self.size += 1
    
    def get_arrays(self):
        """Get the filled portions of the arrays."""
        return self.rows[:self.size], self.cols[:self.size], self.data[:self.size]
    
    def to_csr_matrix(self, shape):
        """Convert to CSR sparse matrix."""
        rows, cols, data = self.get_arrays()
        return csr_matrix((data, (rows, cols)), shape=shape)


def group_frames_optimized(positions, time_precision=0.001):
    """
    Group detections into frames by time.

    Returns:
        frame_times: Sorted unique frame times
        frame_slices: List of (start, end) indices for each frame
        sorted_positions: Positions sorted by time
    """
    positions = np.array(positions)

    rounded_times = np.round(positions[:, 2] / time_precision) * time_precision

    sort_indices = np.argsort(rounded_times)
    sorted_positions = positions[sort_indices]
    sorted_times = rounded_times[sort_indices]

    frame_times, frame_indices = np.unique(sorted_times, return_index=True)

    frame_slices = []
    for i in range(len(frame_indices)):
        start = frame_indices[i]
        end = frame_indices[i + 1] if i + 1 < len(frame_indices) else len(sorted_positions)
        frame_slices.append((start, end))
    
    return frame_times, frame_slices, sorted_positions


def generate_sparse_candidates(track_positions, detection_positions, max_radius,
                              frame_offset=0, detection_offset=0):
    """
    Generate sparse candidate pairs within max_radius using a KDTree.

    Returns:
        row_indices: Track indices
        col_indices: Detection indices
        distances: Geometric distances
    """
    if len(track_positions) == 0 or len(detection_positions) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])

    tree_det = cKDTree(detection_positions)

    pairs = tree_det.query_ball_point(track_positions, r=max_radius, p=2)

    total_pairs = sum(len(det_indices) for det_indices in pairs)

    if total_pairs == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])

    row_indices = np.empty(total_pairs, dtype=int)
    col_indices = np.empty(total_pairs, dtype=int)
    distances = np.empty(total_pairs, dtype=np.float32)

    write_idx = 0
    for track_idx, det_indices in enumerate(pairs):
        if det_indices:
            n_candidates = len(det_indices)
            end_idx = write_idx + n_candidates

            track_pos = track_positions[track_idx]
            candidate_positions = detection_positions[det_indices]
            dists = np.linalg.norm(candidate_positions - track_pos, axis=1)

            row_indices[write_idx:end_idx] = track_idx + frame_offset
            col_indices[write_idx:end_idx] = np.array(det_indices) + detection_offset
            distances[write_idx:end_idx] = dists
            
            write_idx = end_idx
    
    return row_indices, col_indices, distances


def solve_sparse_assignment(row_indices, col_indices, costs, cost_threshold, n_tracks, n_detections):
    """
    Solve assignment problem on sparse data without creating dense matrices.
    Uses LAPJV algorithm when available, falls back to scipy Hungarian or greedy.
    
    Returns:
        assigned_tracks: Track indices
        assigned_detections: Detection indices
    """
    if len(row_indices) == 0:
        return np.array([]), np.array([])

    valid_mask = costs < cost_threshold
    if not np.any(valid_mask):
        return np.array([]), np.array([])

    valid_rows = row_indices[valid_mask]
    valid_cols = col_indices[valid_mask]
    valid_costs = costs[valid_mask]

    unique_tracks     = np.unique(valid_rows)
    unique_detections = np.unique(valid_cols)
    n_t = len(unique_tracks)
    n_d = len(unique_detections)
    n_entries = n_t * n_d

    # safety guard 1 entry-count threshold
    # safety guard 2 explicit MB ceiling, catches unexpectedly large frames
    matrix_mb = n_entries * 4 / 1_000_000
    if n_entries >= SPARSE_ASSIGNMENT_OPTIMAL_MATRIX_THRESHOLD or matrix_mb > DENSE_MATRIX_MEMORY_LIMIT_MB:
        print(f"  WARNING: Using greedy assignment "
              f"(matrix {n_t}x{n_d}={n_entries:,}, {matrix_mb:.0f} MB)")
        return greedy_assignment_sparse(valid_rows, valid_cols, valid_costs, cost_threshold)

    # build dense cost matrix, only for candidate pairs found by KDTree
    SENTINEL = cost_threshold * 10
    track_map = {t: i for i, t in enumerate(unique_tracks)}
    det_map   = {d: i for i, d in enumerate(unique_detections)}

    small_matrix = np.full((n_t, n_d), SENTINEL, dtype=np.float32)
    for row, col, cost in zip(valid_rows, valid_cols, valid_costs):
        small_matrix[track_map[row], det_map[col]] = cost

    # --- LAPJV (preferred: O(n²)-O(n³), ~milliseconds for 3k×3k) ---
    # lapjv needs a square matrix; pad with SENTINEL so dummy
    # rows/cols absorb surplus assignments, real ones stay intact
    lapjv_ok = False
    if lapjv_solver is not None:
        try:
            if n_t != n_d:
                sq = max(n_t, n_d)
                sq_matrix = np.full((sq, sq), SENTINEL, dtype=np.float32)
                sq_matrix[:n_t, :n_d] = small_matrix
            else:
                sq_matrix = small_matrix

            _, row_asgn, _ = lapjv_solver(sq_matrix)

            assigned_small_tracks = []
            assigned_small_dets   = []
            for i, j in enumerate(row_asgn):
                if i < n_t and 0 <= j < n_d and small_matrix[i, j] < cost_threshold:
                    assigned_small_tracks.append(i)
                    assigned_small_dets.append(j)

            assigned_tracks      = unique_tracks[assigned_small_tracks]
            assigned_detections  = unique_detections[assigned_small_dets]
            lapjv_ok = True
            return assigned_tracks, assigned_detections
        except Exception:
            pass  # fall through to scipy fallback

    # --- scipy linear_sum_assignment (handles rectangular, but O(n³)) ---
    if not lapjv_ok and linear_sum_assignment is not None:
        try:
            r, c = linear_sum_assignment(small_matrix)
            assigned_tracks     = unique_tracks[r]
            assigned_detections = unique_detections[c]
            valid_final = small_matrix[r, c] < cost_threshold
            return assigned_tracks[valid_final], assigned_detections[valid_final]
        except Exception:
            pass

    # final greedy fallback, only if both optimal solvers failed
    print(f"  WARNING: Using greedy assignment (matrix {n_t}x{n_d}={n_entries:,})")
    return greedy_assignment_sparse(valid_rows, valid_cols, valid_costs, cost_threshold)


def predict_velocities_batch_optimized(model, positions, time_val, batch_size, device):
    """Optimized batch velocity prediction.
    
    Inputs (positions) are in raw normalised [0,1] space.  If the model has a
    `normalizer` attribute (DataNormalizer), inputs are converted to star-space
    before the forward pass.  The model's own ``forward()`` in eval-mode
    already calls ``normalizer.unstar()`` on the outputs, so we must NOT apply
    unstar again here (that would be a double-unstar bug).
    """
    model.eval()
    normalizer = getattr(model, 'normalizer', None)
    
    u_pred = np.zeros(len(positions))
    v_pred = np.zeros(len(positions))
    
    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
            end_idx = min(i + batch_size, len(positions))
            batch_pos = positions[i:end_idx]

            batch_input = np.column_stack([
                batch_pos[:, :2],
                np.full(len(batch_pos), time_val, dtype=np.float32)
            ])

            input_tensor = torch.tensor(batch_input, dtype=torch.float32, device=device)

            # convert inputs to star-space, the model expects star-space inputs
            if normalizer is not None:
                input_tensor = torch.cat([
                    normalizer.star(input_tensor[:, 0:1], 'x'),
                    normalizer.star(input_tensor[:, 1:2], 'z'),
                    normalizer.star(input_tensor[:, 2:3], 't'),
                ], dim=1)

            # In eval mode, model.forward() returns physical (unstarred) values
            # when a normalizer is attached.  Do NOT unstar again!
            u_batch, v_batch, _ = model(input_tensor)

            u_pred[i:end_idx] = u_batch.cpu().numpy().flatten()
            v_pred[i:end_idx] = v_batch.cpu().numpy().flatten()

    return u_pred, v_pred

def greedy_assignment_sparse(row_indices, col_indices, costs, cost_threshold):
    """Greedy assignment for sparse candidates."""
    if len(row_indices) == 0:
        return np.array([]), np.array([])

    sort_indices = np.argsort(costs)

    assigned_tracks = []
    assigned_detections = []
    used_tracks = set()
    used_detections = set()
    
    for idx in sort_indices:
        if costs[idx] >= cost_threshold:
            break
            
        track_idx = row_indices[idx]
        det_idx = col_indices[idx]
        
        if track_idx not in used_tracks and det_idx not in used_detections:
            assigned_tracks.append(track_idx)
            assigned_detections.append(det_idx)
            used_tracks.add(track_idx)
            used_detections.add(det_idx)
    
    return np.array(assigned_tracks), np.array(assigned_detections)

def perform_gap_closing_vectorized(sorted_positions, frame_times, frame_slices, edge_collector, 
                                 max_radius, max_gap_frames):
    """Vectorized gap closing using boolean masks and batch KDTree queries."""
    print("Starting gap closing...")

    rows, cols, _ = edge_collector.get_arrays()
    n_total = len(sorted_positions)

    used_points = np.zeros(n_total, dtype=bool)
    used_points[rows] = True
    used_points[cols] = True

    max_edges_per_batch = 10000
    batch_rows = np.zeros(max_edges_per_batch, dtype=int)
    batch_cols = np.zeros(max_edges_per_batch, dtype=int)
    batch_data = np.ones(max_edges_per_batch, dtype=np.float32)

    total_gap_edges = 0
    batch_count = 0

    frame_batch_size = min(50, len(frame_times) - max_gap_frames)
    
    for frame_batch_start in range(0, len(frame_times) - max_gap_frames, frame_batch_size):
        frame_batch_end = min(frame_batch_start + frame_batch_size, len(frame_times) - max_gap_frames)
        
        for frame_idx in range(frame_batch_start, frame_batch_end):
            curr_start, curr_end = frame_slices[frame_idx]
            current_indices = np.arange(curr_start, curr_end)

            unmatched_sources_mask = ~used_points[current_indices]
            unmatched_sources = current_indices[unmatched_sources_mask]

            if len(unmatched_sources) == 0:
                continue

            unmatched_source_positions = sorted_positions[unmatched_sources][:, :2]

            for gap in range(2, max_gap_frames + 1):
                target_frame_idx = frame_idx + gap
                if target_frame_idx >= len(frame_times):
                    break
                    
                target_start, target_end = frame_slices[target_frame_idx]
                target_indices = np.arange(target_start, target_end)

                unmatched_targets_mask = ~used_points[target_indices]
                unmatched_targets = target_indices[unmatched_targets_mask]

                if len(unmatched_targets) == 0:
                    continue

                unmatched_target_positions = sorted_positions[unmatched_targets][:, :2]

                if len(unmatched_sources) == 0 or len(unmatched_targets) == 0:
                    continue

                tree_targets = cKDTree(unmatched_target_positions)

                dists, idxs = tree_targets.query(unmatched_source_positions,
                                               distance_upper_bound=max_radius,
                                               k=1)

                valid_mask = (dists < max_radius) & (dists != np.inf)
                valid_source_indices = np.where(valid_mask)[0]
                valid_target_indices = idxs[valid_mask]

                if len(valid_source_indices) == 0:
                    continue

                for i in range(0, len(valid_source_indices), 1000):
                    end_i = min(i + 1000, len(valid_source_indices))
                    batch_sources = valid_source_indices[i:end_i]
                    batch_targets = valid_target_indices[i:end_i]

                    unique_targets = set()
                    valid_assignments = []

                    for src_local_idx, tgt_local_idx in zip(batch_sources, batch_targets):
                        src_idx = unmatched_sources[src_local_idx]
                        tgt_idx = unmatched_targets[tgt_local_idx]

                        # guard against reusing a target in-batch or a
                        # point already consumed elsewhere
                        if (tgt_idx not in unique_targets and
                            not used_points[src_idx] and
                            not used_points[tgt_idx]):

                            valid_assignments.append((src_idx, tgt_idx))
                            unique_targets.add(tgt_idx)

                    for src_idx, tgt_idx in valid_assignments:
                        if batch_count >= max_edges_per_batch:
                            edge_collector.add_edges(batch_rows[:batch_count],
                                                   batch_cols[:batch_count],
                                                   batch_data[:batch_count])
                            total_gap_edges += batch_count
                            batch_count = 0

                        batch_rows[batch_count] = src_idx
                        batch_cols[batch_count] = tgt_idx
                        batch_count += 1

                        used_points[src_idx] = True
                        used_points[tgt_idx] = True

                # Early exit once all sources in the current frame are matched
                unmatched_sources_mask = ~used_points[current_indices]
                unmatched_sources = current_indices[unmatched_sources_mask]
                if len(unmatched_sources) == 0:
                    break
                unmatched_source_positions = sorted_positions[unmatched_sources][:, :2]

        if frame_batch_start % (frame_batch_size * 2) == 0:
            gc.collect()

    if batch_count > 0:
        edge_collector.add_edges(batch_rows[:batch_count], 
                               batch_cols[:batch_count], 
                               batch_data[:batch_count])
        total_gap_edges += batch_count
    
    print(f"Gap closing complete: {total_gap_edges} gap edges added")
    return total_gap_edges

def reconstruct_tracks_from_edges_efficient(edge_collector, sorted_positions, n_total):
    """Reconstruct tracks by following precomputed next-node indices."""
    print("Starting track reconstruction...")

    if edge_collector.size == 0:
        return []

    rows, cols, data = edge_collector.get_arrays()

    # For each node, store the next node index (or -1 if none)
    next_indices = np.full(n_total, -1, dtype=int)
    next_indices[rows] = cols

    # Track starts are nodes with no incoming edge
    has_incoming = np.zeros(n_total, dtype=bool)
    has_incoming[cols] = True
    track_starts = np.where(~has_incoming & (next_indices != -1))[0]

    print(f"Found {len(track_starts)} potential track starts")

    max_tracks = min(len(track_starts), 100000)
    tracks = []
    used_points = np.zeros(n_total, dtype=bool)

    batch_size = 10000
    track_count = 0

    for batch_start in range(0, len(track_starts), batch_size):
        batch_end = min(batch_start + batch_size, len(track_starts))
        batch_starts = track_starts[batch_start:batch_end]

        for start_idx in batch_starts:
            if used_points[start_idx] or track_count >= max_tracks:
                continue

            max_track_length = 1000
            track_indices = np.full(max_track_length, -1, dtype=int)
            track_length = 0

            current_idx = start_idx
            track_indices[track_length] = current_idx
            track_length += 1
            used_points[current_idx] = True

            while next_indices[current_idx] != -1 and track_length < max_track_length:
                next_idx = next_indices[current_idx]
                if used_points[next_idx]:
                    break

                track_indices[track_length] = next_idx
                track_length += 1
                used_points[next_idx] = True
                current_idx = next_idx

            if track_length > 0:
                track = [tuple(sorted_positions[track_indices[i]]) for i in range(track_length)]
                tracks.append(track)
                track_count += 1

        if len(track_starts) > 10000 and batch_start % (batch_size * 5) == 0:
            progress = (batch_start + batch_size) / len(track_starts) * 100
            print(f"  Track reconstruction progress: {progress:.0f}% ({track_count} tracks)")

        if batch_start % (batch_size * 10) == 0:
            gc.collect()

    print(f"Track reconstruction complete: {len(tracks)} tracks built")
    return tracks


def manage_memory_for_large_datasets(tracks, threshold_points=100000000):
    """
    Memory management function for large datasets.
    
    Parameters:
        tracks: List of tracks
        threshold_points: Threshold for considering dataset as "large"
    
    Returns:
        is_large: Boolean indicating if dataset is large
        recommendations: List of memory management recommendations
    """
    total_points = sum(len(track) for track in tracks)
    is_large = total_points > threshold_points
    
    recommendations = []
    
    if is_large:
        print(f"⚠️  Large dataset detected: {total_points:,} total points")
        print("Memory management recommendations:")
        
        if total_points > 500000000:  # 500M points
            recommendations.append("Use batch_size=1000 for velocity statistics")
            recommendations.append("Use batch_size=500 for track statistics")
            recommendations.append("Consider skipping velocity statistics for interpolated tracks")
        elif total_points > 200000000:  # 200M points
            recommendations.append("Use batch_size=2000 for velocity statistics")
            recommendations.append("Use batch_size=1000 for track statistics")
        else:
            recommendations.append("Use batch_size=5000 for velocity statistics")
            recommendations.append("Use batch_size=2000 for track statistics")
        
        recommendations.append("Enable garbage collection between processing steps")
        recommendations.append("Monitor memory usage during processing")
        
        for rec in recommendations:
            print(f"  • {rec}")
    else:
        print(f"✓ Dataset size manageable: {total_points:,} total points")
    
    return is_large, recommendations

def calculate_velocity_statistics_memory_efficient(tracks, stage_name="tracks", orig_bounds=None, batch_size=1000):
    """
    Memory-efficient version of velocity statistics calculation that processes tracks in batches.
    
    Parameters:
        tracks: List of tracks (each track is a list of tuples (x, z, t))
        stage_name: String identifier for the stage (e.g., "raw", "interpolated")
        orig_bounds: Original bounds for coordinate scaling (optional)
        batch_size: Number of tracks to process at once
    
    Returns:
        velocity_stats: Dictionary containing velocity statistics
    """
    if not tracks:
        print(f"{stage_name.capitalize()} velocity statistics: No tracks found")
        return None
    
    print(f"\n{stage_name.capitalize()} Velocity Statistics (Memory Efficient):")

    total_points = 0
    vx_sum = 0.0
    vx_sum_sq = 0.0
    vz_sum = 0.0
    vz_sum_sq = 0.0
    speed_sum = 0.0
    speed_sum_sq = 0.0
    
    vx_min = float('inf')
    vx_max = float('-inf')
    vz_min = float('inf')
    vz_max = float('-inf')
    speed_min = float('inf')
    speed_max = float('-inf')

    n_batches = (len(tracks) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(tracks))
        batch_tracks = tracks[start_idx:end_idx]

        for track in batch_tracks:
            if len(track) < 2:
                continue  # need >=2 points for velocity

            track_array = np.array(track)
            x_coords = track_array[:, 0]
            z_coords = track_array[:, 1]
            t_coords = track_array[:, 2]

            vx = np.zeros(len(track))
            vz = np.zeros(len(track))

            for i in range(len(track)):
                if i == 0 and len(track) > 1:
                    # Forward difference at the first point
                    dt = t_coords[1] - t_coords[0]
                    if abs(dt) > 1e-8:
                        vx[i] = (x_coords[1] - x_coords[0]) / dt
                        vz[i] = (z_coords[1] - z_coords[0]) / dt
                elif i == len(track) - 1 and len(track) > 1:
                    # Backward difference at the last point
                    dt = t_coords[-1] - t_coords[-2]
                    if abs(dt) > 1e-8:
                        vx[i] = (x_coords[-1] - x_coords[-2]) / dt
                        vz[i] = (z_coords[-1] - z_coords[-2]) / dt
                elif len(track) > 2:
                    # Central difference for interior points
                    dt = t_coords[i+1] - t_coords[i-1]
                    if abs(dt) > 1e-8:
                        vx[i] = (x_coords[i+1] - x_coords[i-1]) / dt
                        vz[i] = (z_coords[i+1] - z_coords[i-1]) / dt

            vx = np.nan_to_num(vx, nan=0.0, posinf=0.0, neginf=0.0)
            vz = np.nan_to_num(vz, nan=0.0, posinf=0.0, neginf=0.0)

            speeds = np.sqrt(vx**2 + vz**2)

            valid_mask = np.isfinite(vx) & np.isfinite(vz) & np.isfinite(speeds)
            if np.any(valid_mask):
                vx_valid = vx[valid_mask]
                vz_valid = vz[valid_mask]
                speeds_valid = speeds[valid_mask]

                n_valid = len(vx_valid)
                total_points += n_valid

                vx_sum += np.sum(vx_valid)
                vx_sum_sq += np.sum(vx_valid**2)
                vz_sum += np.sum(vz_valid)
                vz_sum_sq += np.sum(vz_valid**2)
                speed_sum += np.sum(speeds_valid)
                speed_sum_sq += np.sum(speeds_valid**2)

                vx_min = min(vx_min, np.min(vx_valid))
                vx_max = max(vx_max, np.max(vx_valid))
                vz_min = min(vz_min, np.min(vz_valid))
                vz_max = max(vz_max, np.max(vz_valid))
                speed_min = min(speed_min, np.min(speeds_valid))
                speed_max = max(speed_max, np.max(speeds_valid))

        if n_batches > 10 and batch_idx % (n_batches // 10) == 0:
            progress = (batch_idx + 1) / n_batches * 100
            print(f"  Processing velocity statistics: {progress:.0f}% complete")

    if total_points == 0:
        print("  No valid velocities found")
        return None

    vx_mean = vx_sum / total_points
    vz_mean = vz_sum / total_points
    speed_mean = speed_sum / total_points
    
    vx_std = np.sqrt((vx_sum_sq / total_points) - vx_mean**2)
    vz_std = np.sqrt((vz_sum_sq / total_points) - vz_mean**2)
    speed_std = np.sqrt((speed_sum_sq / total_points) - speed_mean**2)
    
    velocity_stats = {
        'count': total_points,
        'vx_min': vx_min,
        'vx_max': vx_max,
        'vx_mean': vx_mean,
        'vx_std': vx_std,
        'vz_min': vz_min,
        'vz_max': vz_max,
        'vz_mean': vz_mean,
        'vz_std': vz_std,
        'speed_min': speed_min,
        'speed_max': speed_max,
        'speed_mean': speed_mean,
        'speed_std': speed_std,
        # note percentiles require full data, so we'll estimate them
        'speed_p50': speed_mean,  # Estimate as mean
        'speed_p95': speed_mean + 2 * speed_std,  # Rough estimate
        'speed_p99': speed_mean + 3 * speed_std   # Rough estimate
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
    print(f"    Note: Percentiles are estimates for memory efficiency")
    
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

def calculate_track_statistics_memory_efficient(tracks, stage_name="tracks", batch_size=1000):
    """
    Memory-efficient version of track statistics calculation that processes tracks in batches.
    
    Parameters:
        tracks: List of tracks (each track is a list of tuples (x, z, t))
        stage_name: String identifier for the stage (e.g., "raw", "interpolated")
        batch_size: Number of tracks to process at once
    
    Returns:
        track_stats: Dictionary containing track statistics
    """
    if not tracks:
        print(f"{stage_name.capitalize()} track statistics: No tracks found")
        return None
    
    print(f"\n{stage_name.capitalize()} Track Statistics (Memory Efficient):")

    total_tracks = len(tracks)
    total_points = 0
    length_sum = 0
    length_sum_sq = 0

    track_length_min = float('inf')
    track_length_max = float('-inf')

    n_batches = (len(tracks) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(tracks))
        batch_tracks = tracks[start_idx:end_idx]

        for track in batch_tracks:
            track_length = len(track)
            total_points += track_length
            length_sum += track_length
            length_sum_sq += track_length**2

            track_length_min = min(track_length_min, track_length)
            track_length_max = max(track_length_max, track_length)

        if n_batches > 10 and batch_idx % (n_batches // 10) == 0:
            progress = (batch_idx + 1) / n_batches * 100
            print(f"  Processing track statistics: {progress:.0f}% complete")

    if total_tracks == 0:
        print("  No tracks found")
        return None

    mean_length = length_sum / total_tracks
    length_std = np.sqrt((length_sum_sq / total_tracks) - mean_length**2)

    median_length = mean_length  # rough estimate; see "Note" printed below
    
    track_stats = {
        'count': total_tracks,
        'total_points': total_points,
        'length_min': track_length_min,
        'length_max': track_length_max,
        'length_mean': mean_length,
        'length_median': median_length,
        'length_std': length_std
    }
    
    # Print track statistics
    print(f"  Number of tracks: {track_stats['count']:,}")
    print(f"  Total points: {track_stats['total_points']:,}")
    print(f"  Track lengths - Min: {track_stats['length_min']}, Max: {track_stats['length_max']}")
    print(f"  Track lengths - Mean: {track_stats['length_mean']:.2f}, Median: {track_stats['length_median']:.2f}")
    print(f"  Track lengths - Std: {track_stats['length_std']:.2f}")
    print(f"  Note: Median is estimated for memory efficiency")
    
    return track_stats


def verify_track_uniqueness(tracks, positions):
    """
    Verify that tracks do not overuse detections, accounting for repeated
    coordinates in the raw detection pool.

    Important: the challenge datasets can contain multiple detections with
    identical (x, z, t). A naive nearest neighbour lookup can collapse these
    to one index and falsely report reuse. Here we compare multiplicities:
    for each unique (x, z, t), track usage count must be <= detection count.

    Parameters:
        tracks: List of tracks (each track is a list of (x, z, t) tuples)
        positions: Original input positions (N, 3) array
        
    Returns:
        is_unique: Boolean indicating if all detection indices are unique
        duplicate_count: Number of duplicate indices found
        total_points: Total points in all tracks
        unique_points: Number of unique detection indices
    """
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] < 3:
        raise ValueError("positions must have shape (N, >=3)")

    # canonicalize to float32 triplets so keys match downstream track tuples
    pos_xyz = np.asarray(positions[:, :3], dtype=np.float32)
    pos_key = np.ascontiguousarray(pos_xyz).view(
        np.dtype((np.void, pos_xyz.dtype.itemsize * pos_xyz.shape[1]))
    ).reshape(-1)
    pos_unique, pos_counts = np.unique(pos_key, return_counts=True)
    pos_count_map = {k: int(c) for k, c in zip(pos_unique.tolist(), pos_counts.tolist())}

    # flatten all track points (x,z,t only)
    total_points = sum(len(track) for track in tracks)
    if total_points == 0:
        print("Track uniqueness verification (multiplicity aware):")
        print("  No points in tracks.")
        return True, 0, 0, 0

    track_xyz = np.empty((total_points, 3), dtype=np.float32)
    w = 0
    for track in tracks:
        for p in track:
            track_xyz[w, 0] = np.float32(p[0])
            track_xyz[w, 1] = np.float32(p[1])
            track_xyz[w, 2] = np.float32(p[2])
            w += 1

    tr_key = np.ascontiguousarray(track_xyz).view(
        np.dtype((np.void, track_xyz.dtype.itemsize * track_xyz.shape[1]))
    ).reshape(-1)
    tr_unique, tr_counts = np.unique(tr_key, return_counts=True)

    overused_points = 0
    overused_coords = 0
    missing_points = 0
    for k, c_used in zip(tr_unique.tolist(), tr_counts.tolist()):
        c_avail = pos_count_map.get(k, 0)
        if c_avail == 0:
            missing_points += int(c_used)
            overused_coords += 1
            overused_points += int(c_used)
            continue
        if c_used > c_avail:
            overused_coords += 1
            overused_points += int(c_used - c_avail)

    # "unique points" = track points matchable to available detections
    unique_points = int(total_points - overused_points)
    duplicate_count = int(overused_points)

    dataset_dup_points = int(len(pos_xyz) - len(pos_unique))

    print("Track uniqueness verification (multiplicity aware):")
    print(f"  Total points in tracks: {total_points:,}")
    print(f"  Matchable points: {unique_points:,}")
    print(f"  Overused points: {duplicate_count:,}")
    print(f"  Overused coordinate keys: {overused_coords:,}")
    print(f"  Missing-coordinate points: {missing_points:,}")
    print(f"  Original input points: {len(positions):,}")
    print(f"  Repeated-coordinate detections in input: {dataset_dup_points:,}")

    if duplicate_count > 0:
        print(f"  WARNING: {duplicate_count:,} points exceed available detection multiplicity.")
        return False, duplicate_count, total_points, unique_points

    print("  No multiplicity overuse detected.")
    return True, 0, total_points, unique_points


def extract_tracks_hungarian_initial(positions, time_precision=0.001, max_geo_radius=0.1,
                                    min_length=1, max_tracks_per_frame=500000, 
                                    cost_threshold=50.0, max_matrix_size=200000000, disable_gap_closing=False):
    """
    Optimized initial Hungarian tracking using sparse matrices and efficient data structures.
    Ensures each point is used only once across all tracks.
    """
    print(f"=== Initial Hungarian Tracking (Optimized) ===")
    print(f"Input points: {len(positions):,}")
    print(f"Parameters: max_radius={max_geo_radius}, cost_threshold={cost_threshold}")

    frame_times, frame_slices, sorted_positions = group_frames_optimized(positions, time_precision)

    if len(frame_times) == 0:
        print("Warning: No frames found")
        return []

    print(f"Processing {len(frame_times)} frames")

    n_total = len(sorted_positions)
    edge_collector = EdgeCollector(initial_capacity=n_total)

    active_tracks = {}  # detection_idx -> track_id
    next_track_id = 0

    # All points used in any track, to prevent reuse
    used_points = set()

    first_start, first_end = frame_slices[0]
    for i in range(first_start, first_end):
        active_tracks[i] = next_track_id
        used_points.add(i)
        next_track_id += 1

    print(f"Started with {len(active_tracks)} tracks")

    max_gap_frames = 2
    detection_tree = None

    for frame_idx in range(1, len(frame_times)):
        if not active_tracks:
            break

        curr_start, curr_end = frame_slices[frame_idx]
        current_detections = sorted_positions[curr_start:curr_end]

        if len(current_detections) == 0:
            continue

        track_endpoints = []
        track_indices = []
        track_ids = []

        for det_idx, track_id in active_tracks.items():
            track_endpoints.append(sorted_positions[det_idx][:2])
            track_indices.append(det_idx)
            track_ids.append(track_id)

        track_endpoints = np.array(track_endpoints)
        current_positions = current_detections[:, :2]

        row_indices, col_indices, distances = generate_sparse_candidates(
            track_endpoints, current_positions, max_geo_radius
        )

        if len(row_indices) == 0:
            # No candidates: start new tracks for unused detections
            active_tracks = {}
            for i in range(curr_start, curr_end):
                if i not in used_points:
                    active_tracks[i] = next_track_id
                    used_points.add(i)
                    next_track_id += 1
            continue

        n_tracks = len(track_endpoints)
        n_detections = len(current_detections)

        assigned_tracks, assigned_detections = solve_sparse_assignment(
            row_indices, col_indices, distances, cost_threshold, n_tracks, n_detections
        )

        new_active_tracks = {}
        used_detections = set()

        if len(assigned_tracks) > 0:
            old_det_indices = [track_indices[i] for i in assigned_tracks]
            new_det_indices = [curr_start + i for i in assigned_detections]
            edge_data = np.ones(len(assigned_tracks), dtype=np.float32)

            edge_collector.add_edges(np.array(old_det_indices),
                                   np.array(new_det_indices),
                                   edge_data)

            for track_idx, det_idx in zip(assigned_tracks, assigned_detections):
                track_id = track_ids[track_idx]
                new_det_idx = curr_start + det_idx
                new_active_tracks[new_det_idx] = track_id
                used_detections.add(det_idx)
                used_points.add(new_det_idx)

        # Start new tracks for unassigned, unused detections
        for det_idx in range(len(current_detections)):
            if det_idx not in used_detections:
                new_det_idx = curr_start + det_idx
                if new_det_idx not in used_points:
                    new_active_tracks[new_det_idx] = next_track_id
                    used_points.add(new_det_idx)
                    next_track_id += 1

        active_tracks = new_active_tracks

        if frame_idx % 100 == 0:
            print(f"Frame {frame_idx}/{len(frame_times)-1}: {len(active_tracks)} active tracks")

    if not disable_gap_closing and max_gap_frames > 0:
        print("Performing gap closing...")
        perform_gap_closing_vectorized(sorted_positions, frame_times, frame_slices,
                                     edge_collector, max_geo_radius, max_gap_frames)

    print("Reconstructing tracks...")
    tracks = reconstruct_tracks_from_edges_efficient(edge_collector, sorted_positions, n_total)

    filtered_tracks = [track for track in tracks if len(track) >= min_length]

    print(f"Optimized tracking complete: {len(filtered_tracks)} tracks (from {len(tracks)} total)")

    if filtered_tracks:
        total_points = sum(len(track) for track in filtered_tracks)
        avg_length = total_points / len(filtered_tracks)
        print(f"Points preserved: {total_points:,} / {len(positions):,} ({100*total_points/len(positions):.1f}%)")
        print(f"Average track length: {avg_length:.1f}")

        is_unique, duplicate_count, _, _ = verify_track_uniqueness(filtered_tracks, positions)
        
        if not is_unique:
            print(f"ERROR: Point reuse detected! This should not happen with the corrected algorithm.")
            print("Please check the gap closing and track reconstruction functions.")
    
    return filtered_tracks

def extract_tracks_lapjv_pinn(positions, model, 
                               w_geo=0.5, w_phys=0.5,
                               max_geo_radius=0.025, dt=1.0,
                               batch_size=1000, max_tracks_per_frame=100000, 
                               min_length=15, cost_threshold=10.0,
                               use_multi_gpu=True, memory_efficient=True,
                               max_matrix_size=100000000, max_candidates_per_track=5000,
                               max_total_candidates=250000, disable_gap_closing=False,
                               early_termination=True, adaptive_batch_sizing=True,
                               candidate_anchor="endpoint", candidate_blend_alpha=0.5):
    """
    PINN guided LAPJV tracking with early termination and adaptive batch sizing.
    """
    print("=== LAPJV + PINN Tracking (Ultra-Optimized) ===")
    print(f"Parameters: w_geo={w_geo:.2f}, w_phys={w_phys:.2f}, cost_threshold={cost_threshold:.2f}")
    candidate_anchor = str(candidate_anchor).lower()
    if candidate_anchor not in {"endpoint", "physics", "blend"}:
        raise ValueError(
            f"candidate_anchor must be one of {{'endpoint','physics','blend'}}, got: {candidate_anchor}"
        )
    candidate_blend_alpha = float(np.clip(candidate_blend_alpha, 0.0, 1.0))
    print(
        f"Candidate anchor mode: {candidate_anchor} "
        f"(blend_alpha={candidate_blend_alpha:.2f})"
    )
    
    if adaptive_batch_sizing and len(positions) > 1000000:
        print("Large dataset detected - adjusting parameters for optimal performance")
        batch_size = min(batch_size, 500)
        max_candidates_per_track = min(max_candidates_per_track, 2000)
        max_total_candidates = min(max_total_candidates, 100000)
        print(f"Adjusted batch_size={batch_size}, max_candidates_per_track={max_candidates_per_track}")

    time_precision = 0.001
    frame_times, frame_slices, sorted_positions = group_frames_optimized(positions, time_precision)

    if len(frame_times) == 0:
        print("Warning: No frames found")
        return []

    print(f"Processing {len(frame_times)} frames with {len(positions)} total detections")

    device = next(model.parameters()).device

    n_total = len(sorted_positions)
    edge_collector = EdgeCollector(initial_capacity=n_total)

    active_tracks = {}
    next_track_id = 0

    first_start, first_end = frame_slices[0]
    for i in range(first_start, first_end):
        active_tracks[i] = next_track_id
        next_track_id += 1

    print(f"Started with {len(active_tracks)} tracks")

    max_gap_frames = 2
    start_time = time.time()

    for frame_idx in range(1, len(frame_times)):
        if not active_tracks:
            break

        if frame_idx % 100 == 0:
            elapsed_time = time.time() - start_time
            progress = frame_idx / len(frame_times)
            if progress > 0:
                estimated_total = elapsed_time / progress
                remaining_time = estimated_total - elapsed_time
                print(f"Frame {frame_idx}/{len(frame_times)-1}: {len(active_tracks)} active tracks, "
                      f"ETA: {remaining_time/60:.1f} minutes")

        if early_termination and len(active_tracks) > 50000:
            print(f"Early termination: Too many active tracks ({len(active_tracks)}), stopping frame processing")
            break

        curr_start, curr_end = frame_slices[frame_idx]
        current_detections = sorted_positions[curr_start:curr_end]

        if len(current_detections) == 0:
            continue

        track_endpoints = []
        track_indices = []
        track_ids = []

        for det_idx, track_id in active_tracks.items():
            track_endpoints.append(sorted_positions[det_idx])
            track_indices.append(det_idx)
            track_ids.append(track_id)

        track_endpoints = np.array(track_endpoints)
        current_positions = current_detections[:, :2]

        u_pred, v_pred = predict_velocities_batch_optimized(
            model, track_endpoints, frame_times[frame_idx], batch_size, device
        )

        # actual dt from frame times, velocity is dx/dt in normalised coords,
        # so use the real time step, NOT the user-supplied dt (may be wrong)
        dt_actual = frame_times[frame_idx] - frame_times[frame_idx - 1]
        if dt_actual <= 0 or not np.isfinite(dt_actual):
            dt_actual = dt  # fallback to user supplied dt

        physics_predictions = np.column_stack([
            track_endpoints[:, 0] + u_pred * dt_actual,
            track_endpoints[:, 1] + v_pred * dt_actual
        ])

        # candidate gating center
        # endpoint mode reproduces legacy behavior
        # physics mode gates around predicted next positions, Kalman-style gating
        # blend mode interpolates between endpoint and predicted position
        if candidate_anchor == "endpoint":
            candidate_centers = track_endpoints[:, :2]
        elif candidate_anchor == "physics":
            candidate_centers = physics_predictions
        else:  # blend
            candidate_centers = (
                (1.0 - candidate_blend_alpha) * track_endpoints[:, :2]
                + candidate_blend_alpha * physics_predictions
            )

        geo_candidates = generate_sparse_candidates(
            candidate_centers, current_positions, max_geo_radius
        )
        
        if len(geo_candidates[0]) == 0:
            # No candidates, start new tracks
            active_tracks = {}
            for i in range(curr_start, curr_end):
                active_tracks[i] = next_track_id
                next_track_id += 1
            continue

        n_tracks = len(track_endpoints)
        n_detections = len(current_detections)

        row_indices, col_indices, geo_distances = geo_candidates

        phys_distances = np.linalg.norm(
            physics_predictions[row_indices] - current_positions[col_indices], axis=1
        )

        combined_costs = w_geo * geo_distances + w_phys * phys_distances

        assigned_tracks, assigned_detections = solve_sparse_assignment(
            row_indices, col_indices, combined_costs, cost_threshold, n_tracks, n_detections
        )

        new_active_tracks = {}
        used_detections = set()

        if len(assigned_tracks) > 0:
            old_det_indices = [track_indices[i] for i in assigned_tracks]
            new_det_indices = [curr_start + i for i in assigned_detections]
            edge_data = np.ones(len(assigned_tracks), dtype=np.float32)
            
            edge_collector.add_edges(np.array(old_det_indices),
                                   np.array(new_det_indices),
                                   edge_data)

            for track_idx, det_idx in zip(assigned_tracks, assigned_detections):
                track_id = track_ids[track_idx]
                new_det_idx = curr_start + det_idx
                new_active_tracks[new_det_idx] = track_id
                used_detections.add(det_idx)

        for det_idx in range(len(current_detections)):
            if det_idx not in used_detections:
                new_det_idx = curr_start + det_idx
                new_active_tracks[new_det_idx] = next_track_id
                next_track_id += 1

        active_tracks = new_active_tracks

        if frame_idx % 200 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Skip gap closing for large datasets to save time
    if not disable_gap_closing and max_gap_frames > 0 and len(frame_times) < 1000:
        print("Performing gap closing...")
        gap_start_time = time.time()
        perform_gap_closing_vectorized(sorted_positions, frame_times, frame_slices,
                                     edge_collector, max_geo_radius, max_gap_frames)
        gap_time = time.time() - gap_start_time
        print(f"Gap closing completed in {gap_time:.1f} seconds")
    elif len(frame_times) >= 1000:
        print("Skipping gap closing for large dataset (>1000 frames) to save time")

    print("Reconstructing tracks...")
    recon_start_time = time.time()
    tracks = reconstruct_tracks_from_edges_efficient(edge_collector, sorted_positions, n_total)
    recon_time = time.time() - recon_start_time
    print(f"Track reconstruction completed in {recon_time:.1f} seconds")

    filtered_tracks = [track for track in tracks if len(track) >= min_length]
    
    total_time = time.time() - start_time
    print(f"Ultra-optimized PINN tracking complete: {len(filtered_tracks)} tracks (from {len(tracks)} total)")
    print(f"Total processing time: {total_time/60:.1f} minutes")
    
    if filtered_tracks:
        total_points = sum(len(track) for track in filtered_tracks)
        avg_length = total_points / len(filtered_tracks)
        print(f"Points preserved: {total_points:,} / {len(positions):,} ({100*total_points/len(positions):.1f}%)")
        print(f"Average track length: {avg_length:.1f}")
    
    return filtered_tracks


def build_cost_matrix_geometric_only(track_positions, detection_positions, max_radius, cost_threshold):
    """Build cost matrix using only geometric distance for initial tracking."""
    n_tracks = len(track_positions)
    n_detections = len(detection_positions)
    
    if n_tracks == 0 or n_detections == 0:
        return np.array([]).reshape(0, 0)

    dx = track_positions[:, 0:1] - detection_positions[:, 0]
    dz = track_positions[:, 1:2] - detection_positions[:, 1]
    distances = np.sqrt(dx**2 + dz**2)

    # Out-of-range pairs get a prohibitively high cost
    cost_matrix = np.where(distances <= max_radius, distances, cost_threshold * 10)

    return cost_matrix.astype(np.float32)


def greedy_tracking_fallback_permissive(tracks, current_detections, max_radius):
    """Permissive greedy tracking fallback for initial stage."""
    new_tracks = []
    used_detections = set()
    
    if len(tracks) == 0:
        return [[det] for det in current_detections]

    track_endpoints = np.array([track[-1] for track in tracks])[:, :2]
    detection_positions = np.array(current_detections)[:, :2]

    tree = cKDTree(detection_positions)

    for track_idx, track in enumerate(tracks):
        track_pos = track_endpoints[track_idx]

        candidate_indices = tree.query_ball_point(track_pos, max_radius, p=2)

        if candidate_indices:
            for det_idx in candidate_indices:
                if det_idx not in used_detections:
                    new_tracks.append(track + [current_detections[det_idx]])
                    used_detections.add(det_idx)
                    break
            else:
                new_tracks.append(track)  # no available detection, track ends
        else:
            new_tracks.append(track)  # no nearby detections, track ends

    for det_idx, detection in enumerate(current_detections):
        if det_idx not in used_detections:
            new_tracks.append([detection])
    
    return new_tracks


def extract_tracks(positions, time_round=2, distance_threshold=0.05, min_length=15):
    """Original greedy tracking function, kept for compatibility."""
    groups = {}
    for pos in positions:
        frame = round(pos[2], time_round)
        groups.setdefault(frame, []).append((pos[0], pos[1], pos[2]))

    if not groups:
        return []

    frames = sorted(groups.keys())
    tracks = [[coord] for coord in groups[frames[0]]]

    for frame in frames[1:]:
        current_points = np.array([t[-1][:2] for t in tracks])
        next_coords = groups[frame]
        next_points = np.array([c[:2] for c in next_coords])

        tree = cKDTree(current_points)
        dists, idxs = tree.query(next_points, k=1, distance_upper_bound=distance_threshold)

        used = set()
        for i, (d, ti) in enumerate(zip(dists, idxs)):
            if d <= distance_threshold:
                tracks[ti].append(next_coords[i])
                used.add(i)

        for i, c in enumerate(next_coords):
            if i not in used:
                tracks.append([c])

    filtered_tracks = [track for track in tracks if len(track) >= min_length]
    
    return filtered_tracks


def compute_track_velocities(tracks, savgol_polyorder=2, use_vectorized=True, min_track_length=5):
    """
    Compute central-difference velocities for each track.
    
    Parameters:
        tracks: List of tracks
        savgol_polyorder: Polynomial order for Savitzky-Golay filter
        use_vectorized: Whether to use vectorized operations
        
    Returns:
        all_positions: Nx3 array of (x,z,t)
        all_velocities: Nx2 array of globally-normalised (u,v)
    """
    min_track_length = int(min_track_length)
    if min_track_length < 2:
        min_track_length = 2

    original_count = len(tracks)
    short_tracks = sum(1 for track in tracks if len(track) < min_track_length)
    if short_tracks > 0:
        print(f"Filtering out {short_tracks} tracks with <{min_track_length} points (of {original_count} total tracks)")
        print(f"Processing {original_count - short_tracks} tracks with ≥{min_track_length} points for velocity computation")
    
    all_pos = []
    all_vel = []
    
    if use_vectorized:
        for tr in tracks:
            arr = np.array(tr)
            if len(arr) < min_track_length:
                continue
            
            x, z, t = arr[:, 0], arr[:, 1], arr[:, 2]
            N = len(x)
            
            if N < 2:
                u = np.zeros(N)
                v = np.zeros(N)
            else:
                dt = np.diff(t)
                has_duplicate_t = np.any(np.isclose(dt, 0.0, atol=1e-8))
                if N < 3:
                    # np.gradient(..., edge_order=2) needs >=3 points;
                    # for 2-point tracks use a simple finite difference
                    u = np.zeros(N)
                    v = np.zeros(N)
                    dt_01 = t[1] - t[0]
                    if np.abs(dt_01) > 1e-8:
                        du = (x[1] - x[0]) / dt_01
                        dv = (z[1] - z[0]) / dt_01
                        u[:] = du
                        v[:] = dv
                elif has_duplicate_t:
                    u = np.zeros(N)
                    v = np.zeros(N)
                    if N > 2:
                        dt_center = t[2:] - t[:-2]
                        valid_dt = np.abs(dt_center) > 1e-8
                        u[1:-1][valid_dt] = (x[2:] - x[:-2])[valid_dt] / dt_center[valid_dt]
                        v[1:-1][valid_dt] = (z[2:] - z[:-2])[valid_dt] / dt_center[valid_dt]
                    if N > 1:
                        dt_fwd = t[1] - t[0]
                        dt_bwd = t[-1] - t[-2]
                        if np.abs(dt_fwd) > 1e-8:
                            u[0] = (x[1] - x[0]) / dt_fwd
                            v[0] = (z[1] - z[0]) / dt_fwd
                        if np.abs(dt_bwd) > 1e-8:
                            u[-1] = (x[-1] - x[-2]) / dt_bwd
                            v[-1] = (z[-1] - z[-2]) / dt_bwd
                else:
                    u = np.gradient(x, t, edge_order=2)
                    v = np.gradient(z, t, edge_order=2)
            
            u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            
            if N >= 5 and (np.any(np.abs(u) > 1e-10) or np.any(np.abs(v) > 1e-10)):
                try:
                    u = savgol_filter(u, 5, savgol_polyorder)
                    v = savgol_filter(v, 5, savgol_polyorder)
                    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
                    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    pass
            
            all_pos.append(arr)
            all_vel.append(np.column_stack([u, v]))
    else:
        for tr in tracks:
            arr = np.array(tr)
            if len(arr) < min_track_length:
                continue
            
            x, z, t = arr[:, 0], arr[:, 1], arr[:, 2]
            N = len(x)
            u = np.zeros(N)
            v = np.zeros(N)
            
            if N > 2:
                dt = t[2:] - t[:-2]
                valid = np.abs(dt) > 1e-8
                if np.any(valid):
                    u[1:-1][valid] = (x[2:] - x[:-2])[valid] / dt[valid]
                    v[1:-1][valid] = (z[2:] - z[:-2])[valid] / dt[valid]
            if N > 1:
                dt_fwd = t[1] - t[0]
                dt_bwd = t[-1] - t[-2]
                if np.abs(dt_fwd) > 1e-8:
                    u[0] = (x[1] - x[0]) / dt_fwd
                    v[0] = (z[1] - z[0]) / dt_fwd
                if np.abs(dt_bwd) > 1e-8:
                    u[-1] = (x[-1] - x[-2]) / dt_bwd
                    v[-1] = (z[-1] - z[-2]) / dt_bwd
            
            if N >= 5:
                try:
                    u = savgol_filter(u, 5, savgol_polyorder)
                    v = savgol_filter(v, 5, savgol_polyorder)
                except Exception:
                    pass
            
            u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            
            all_pos.append(arr)
            all_vel.append(np.column_stack([u, v]))
    
    if not all_pos:
        return np.empty((0, 3)), np.empty((0, 2))
    
    all_positions = np.vstack(all_pos)
    all_velocities = np.vstack(all_vel)
    
    # NOTE: do NOT z-score velocities here. The DataNormalizer inside
    # train_pinn already maps raw velocities into zero-mean/unit-std
    # "star" space. An extra z-score would corrupt predictions during
    # tracking because the PINN output (after unstar) would be in z-scored
    # units instead of true normalised coordinate velocity units;
    # only clean up NaN/Inf that might remain after differentiation
    all_velocities = np.nan_to_num(all_velocities, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Velocity stats (raw): u  mean={all_velocities[:,0].mean():.4e}  std={all_velocities[:,0].std():.4e}")
    print(f"                        v  mean={all_velocities[:,1].mean():.4e}  std={all_velocities[:,1].std():.4e}")
    
    return all_positions, all_velocities





def predict_velocities_batch(model, positions, time_val, batch_size, device, memory_efficient=True):
    """Predict velocities for a batch of positions.

    Same semantics as predict_velocities_batch_optimized: inputs are in raw
    normalised [0,1] space and outputs are in physical normalised coordinate
    velocity units.  The model's forward() already unstars in eval mode, so
    we must NOT unstar again here.
    """
    model.eval()

    u_pred = np.zeros(len(positions))
    v_pred = np.zeros(len(positions))

    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
            end_idx = min(i + batch_size, len(positions))
            batch_pos = positions[i:end_idx]

            batch_input_raw = np.column_stack([
                batch_pos[:, :2],
                np.full(len(batch_pos), time_val, dtype=np.float32)
            ])
            input_tensor = torch.tensor(batch_input_raw, dtype=torch.float32, device=device)

            # convert inputs to star-space, model expects star-space inputs
            if hasattr(model, 'normalizer') and model.normalizer is not None:
                x_star = model.normalizer.star(input_tensor[:, 0:1], 'x')
                z_star = model.normalizer.star(input_tensor[:, 1:2], 'z')
                t_star = model.normalizer.star(input_tensor[:, 2:3], 't')
                input_tensor = torch.cat([x_star, z_star, t_star], dim=1)
            
            # eval mode model.forward() returns physical unstarred values
            # Do NOT unstar again!
            u_batch, v_batch, _ = model(input_tensor)

            u_pred[i:end_idx] = u_batch.cpu().numpy().flatten()
            v_pred[i:end_idx] = v_batch.cpu().numpy().flatten()

            if memory_efficient:
                del input_tensor, u_batch, v_batch

    return u_pred, v_pred


def prefilter_candidates(track_positions, detection_positions, max_radius,
                        max_candidates_per_track=1000, max_total_candidates=50000):
    """Pre-filter candidates using KDTree for efficiency."""
    n_tracks = len(track_positions)
    n_detections = len(detection_positions)
    
    if n_tracks > 10000 or n_detections > 100000:
        return prefilter_candidates_aggressive(track_positions, detection_positions, max_radius)

    tree = cKDTree(detection_positions)

    track_candidates = []
    all_detection_candidates = set()

    for track_idx, track_pos in enumerate(track_positions):
        candidates = tree.query_ball_point(track_pos, r=max_radius, p=2)

        if candidates:
            track_candidates.append(track_idx)
            all_detection_candidates.update(candidates)

    detection_candidates = sorted(list(all_detection_candidates))
    
    print(f"Pre-filtering: {len(track_candidates)} tracks, {len(detection_candidates)} detection candidates")
    return track_candidates, detection_candidates


def prefilter_candidates_aggressive(track_positions, detection_positions, max_radius):
    """Aggressive filtering for very large datasets."""
    n_tracks = len(track_positions)
    n_detections = len(detection_positions)
    
    print(f"Processing ALL data without sampling: {n_tracks} tracks, {n_detections} detections")

    track_candidates_base = np.arange(n_tracks)
    detection_candidates_base = np.arange(n_detections)

    tree = cKDTree(detection_positions)

    track_candidates = []
    detection_candidates = set()

    for track_idx, track_pos in enumerate(track_positions):
        candidates_local = tree.query_ball_point(track_pos, r=max_radius, p=2)
        if candidates_local:
            track_candidates.append(track_candidates_base[track_idx])
            detection_candidates.update([detection_candidates_base[c] for c in candidates_local])
    
    detection_candidates = sorted(list(detection_candidates))
    
    print(f"Aggressive pre-filtering: {len(track_candidates)} tracks, {len(detection_candidates)} detection candidates")
    return track_candidates, detection_candidates


def build_cost_matrix_optimized(track_positions, physics_predictions, detection_positions, 
                               w_geo, w_phys, max_radius, cost_threshold):
    """Build cost matrix efficiently with vectorized operations."""
    n_tracks = len(track_positions)
    n_detections = len(detection_positions)
    
    if n_tracks == 0 or n_detections == 0:
        return np.array([]).reshape(0, 0)

    chunk_threshold = 10_000_000
    if n_tracks * n_detections > chunk_threshold:
        print(f"Large cost matrix ({n_tracks} x {n_detections}), using memory-efficient computation")
        return build_cost_matrix_chunked(track_positions, physics_predictions, detection_positions,
                                       w_geo, w_phys, max_radius, cost_threshold)

    dx = track_positions[:, 0:1] - detection_positions[:, 0]
    dz = track_positions[:, 1:2] - detection_positions[:, 1]
    geo_distances = np.sqrt(dx**2 + dz**2)

    dx_phys = physics_predictions[:, 0:1] - detection_positions[:, 0]
    dz_phys = physics_predictions[:, 1:2] - detection_positions[:, 1]
    phys_distances = np.sqrt(dx_phys**2 + dz_phys**2)

    cost_matrix = w_geo * geo_distances + w_phys * phys_distances

    # Out-of-range pairs get a prohibitively high cost
    invalid_mask = geo_distances > max_radius
    cost_matrix[invalid_mask] = cost_threshold * 10

    cost_matrix = np.nan_to_num(cost_matrix, nan=cost_threshold * 10,
                               posinf=cost_threshold * 10, neginf=cost_threshold * 10)
    
    return cost_matrix


def build_cost_matrix_chunked(track_positions, physics_predictions, detection_positions,
                            w_geo, w_phys, max_radius, cost_threshold, chunk_size=1000):
    """Build cost matrix in chunks for memory efficiency."""
    n_tracks = len(track_positions)
    n_detections = len(detection_positions)
    cost_matrix = np.full((n_tracks, n_detections), cost_threshold * 10, dtype=np.float32)

    for i in range(0, n_tracks, chunk_size):
        end_i = min(i + chunk_size, n_tracks)

        dx = track_positions[i:end_i, 0:1] - detection_positions[:, 0]
        dz = track_positions[i:end_i, 1:2] - detection_positions[:, 1]
        geo_dist_chunk = np.sqrt(dx**2 + dz**2)

        dx_phys = physics_predictions[i:end_i, 0:1] - detection_positions[:, 0]
        dz_phys = physics_predictions[i:end_i, 1:2] - detection_positions[:, 1]
        phys_dist_chunk = np.sqrt(dx_phys**2 + dz_phys**2)

        valid_mask = geo_dist_chunk <= max_radius
        cost_chunk = w_geo * geo_dist_chunk + w_phys * phys_dist_chunk

        cost_matrix[i:end_i][valid_mask] = cost_chunk[valid_mask]

    return cost_matrix


def greedy_assignment_fallback(cost_matrix, cost_threshold):
    """Fallback greedy assignment if LAPJV fails."""
    row_indices = []
    col_indices = []
    used_cols = set()

    flat_indices = np.unravel_index(np.argsort(cost_matrix.ravel()), cost_matrix.shape)

    for row, col in zip(flat_indices[0], flat_indices[1]):
        if cost_matrix[row, col] >= cost_threshold:
            break
        if row not in [r for r in row_indices] and col not in used_cols:
            row_indices.append(row)
            col_indices.append(col)
            used_cols.add(col)
    
    return np.array(row_indices), np.array(col_indices)


def greedy_tracking_fallback(tracks, current_detections, max_radius):
    """Simple greedy tracking fallback when LAPJV fails."""
    new_tracks = []
    used_detections = set()

    track_endpoints = np.array([track[-1] for track in tracks])[:, :2]
    detection_positions = np.array(current_detections)[:, :2]

    tree = cKDTree(detection_positions)

    for track_idx, track in enumerate(tracks):
        track_pos = track_endpoints[track_idx]

        distances, indices = tree.query(track_pos, k=min(10, len(detection_positions)))

        # Normalize scalar results (k=1) to lists
        if np.isscalar(distances):
            distances = [distances]
            indices = [indices]

        assigned = False
        for dist, det_idx in zip(distances, indices):
            if dist <= max_radius and det_idx not in used_detections:
                new_tracks.append(track + [current_detections[det_idx]])
                used_detections.add(det_idx)
                assigned = True
                break

        if not assigned:
            new_tracks.append(track)

    for det_idx, detection in enumerate(current_detections):
        if det_idx not in used_detections:
            new_tracks.append([detection])
    
    print(f"Greedy fallback: {len(new_tracks)} tracks ({len(used_detections)} assignments)")
    return new_tracks 
