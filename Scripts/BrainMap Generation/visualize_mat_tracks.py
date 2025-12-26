"""
Visualize tracks from MATLAB .mat file format.
This script loads tracks from a .mat file (containing Track_tot_1, Track_tot_2, Track_tot_3)
and creates visualizations showing:
- Lines representing complete tracks
- Dots representing microbubbles at each frame
"""

import argparse
import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def load_tracks_from_mat(mat_file):
    """
    Load tracks from MATLAB .mat file.
    
    Parameters:
        mat_file: Path to .mat file containing Track_tot_1, Track_tot_2, Track_tot_3
        
    Returns:
        tracks: List of track arrays, each with shape (N, 5) containing [x, z, vx, vz, timeline]
    """
    print(f"Loading tracks from: {mat_file}", flush=True)
    
    try:
        mat_data = loadmat(mat_file)
    except Exception as e:
        raise FileNotFoundError(f"Error loading .mat file: {e}")
    
    # Extract tracks from all three groups
    tracks = []
    group_counts = {}
    
    for group_name in ['Track_tot_1', 'Track_tot_2', 'Track_tot_3']:
        if group_name not in mat_data:
            continue
            
        group_tracks = mat_data[group_name]
        count = 0
        
        # Handle MATLAB cell array format (nested arrays)
        if group_tracks.dtype == object:
            # MATLAB cell array - each element is a track
            # Handle both 1D and 2D cell arrays
            if group_tracks.ndim == 2:
                # 2D cell array (most common MATLAB format)
                for i in range(group_tracks.shape[0]):
                    for j in range(group_tracks.shape[1]):
                        track = group_tracks[i, j]
                        if track.size > 0 and track.ndim == 2:
                            tracks.append(track)
                            count += 1
            elif group_tracks.ndim == 1:
                # 1D cell array
                for i in range(len(group_tracks)):
                    track = group_tracks[i]
                    if track.size > 0 and track.ndim == 2:
                        tracks.append(track)
                        count += 1
        else:
            # Not a cell array - check if it's a list/array of arrays
            if group_tracks.ndim == 1:
                # 1D array of arrays
                for track in group_tracks:
                    if track.size > 0 and track.ndim == 2:
                        tracks.append(track)
                        count += 1
            elif group_tracks.ndim == 2:
                # Single 2D array (one track)
                if group_tracks.size > 0:
                    tracks.append(group_tracks)
                    count = 1
        
        group_counts[group_name] = count
        if count > 0:
            print(f"  Loaded {count} tracks from {group_name}", flush=True)
    
    print(f"Total tracks loaded: {len(tracks)}", flush=True)
    
    if len(tracks) == 0:
        available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        raise ValueError(f"No tracks found in .mat file. Expected Track_tot_1, Track_tot_2, Track_tot_3. "
                        f"Available keys: {available_keys}")
    
    return tracks


def plot_tracks_static(tracks, output_file=None, line_width=0.5, dot_size=1.0, show_dots=True):
    """
    Create a static plot showing all tracks as lines and optionally dots.
    
    Parameters:
        tracks: List of track arrays, each with shape (N, 5) containing [x, z, vx, vz, timeline]
        output_file: Optional path to save the figure
        line_width: Width of track lines
        dot_size: Size of microbubble dots
        show_dots: Whether to show dots at each point along tracks
    """
    print("Creating static track visualization...", flush=True)
    
    # Find bounds so the aspect ratio is preserved across all tracks
    x_min, x_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')
    
    for track in tracks:
        if track.shape[0] > 0:
            x_min = min(x_min, track[:, 0].min())
            x_max = max(x_max, track[:, 0].max())
            z_min = min(z_min, track[:, 1].min())
            z_max = max(z_max, track[:, 1].max())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='w')
    
    # Plot each track
    # Important for performance: if dots are enabled, collect all points and scatter once,
    # rather than creating thousands of PathCollection objects (one per track).
    all_dot_x = []
    all_dot_z = []
    for i, track in enumerate(tracks):
        if track.shape[0] == 0:
            continue
            
        x_coords = track[:, 0]
        z_coords = track[:, 1]
        
        # Plot line for track
        ax.plot(x_coords, z_coords, 'b-', linewidth=line_width, alpha=0.6)
        
        # Plot dots if requested
        if show_dots:
            all_dot_x.append(x_coords)
            all_dot_z.append(z_coords)

    if show_dots and all_dot_x:
        # Use a single scatter with no edge so dots render as solid filled (not hollow)
        dot_x = np.concatenate(all_dot_x)
        dot_z = np.concatenate(all_dot_z)
        ax.scatter(
            dot_x,
            dot_z,
            c='red',
            s=dot_size**2,
            alpha=0.4,
            edgecolors='none',
            linewidths=0,
            zorder=2,
        )
    
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Z Coordinate", fontsize=12)
    ax.set_title(f"Track Visualization ({len(tracks)} tracks)", fontsize=14)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved static visualization to: {output_file}", flush=True)
    else:
        plt.show()
    
    plt.close()


def create_frame_animation(tracks, output_file=None, line_width=0.5, dot_size=3.0, interval=200, show_trails=True, max_frames=None):
    """
    Create an animated visualization showing tracks frame by frame.
    
    Parameters:
        tracks: List of track arrays, each with shape (N, 5) containing [x, z, vx, vz, timeline]
        output_file: Optional path to save the animation (GIF)
        line_width: Width of track lines
        dot_size: Size of microbubble dots
        interval: Animation interval in milliseconds
        show_trails: Whether to show complete track trails up to current frame
    """
    print("Creating frame-by-frame animation...", flush=True)
    
    # Extract all unique frame times
    all_times = set()
    for track in tracks:
        if track.shape[0] > 0:
            all_times.update(track[:, 4])  # timeline is column 4
    
    if len(all_times) == 0:
        raise ValueError("No time information found in tracks")
    
    # Sort all available timeline values so the animation plays chronologically
    frames = sorted(all_times)
    print(f"Found {len(frames)} unique frames", flush=True)
    
    # Downsample frames if requested
    if max_frames is not None and len(frames) > max_frames:
        print(f"Downsampling from {len(frames)} to {max_frames} frames", flush=True)
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]
        print(f"Using {len(frames)} frames for animation", flush=True)
    
    # Warn if too many frames (animation will be very slow/large)
    if len(frames) > 10000:
        print(f"Warning: {len(frames)} frames is very large. Animation may take a long time and create a huge file.", flush=True)
        print("Consider using --max-frames to limit the number of frames or --static-only for faster visualization.", flush=True)
    
    # Find bounds
    x_min, x_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')
    
    for track in tracks:
        if track.shape[0] > 0:
            x_min = min(x_min, track[:, 0].min())
            x_max = max(x_max, track[:, 0].max())
            z_min = min(z_min, track[:, 1].min())
            z_max = max(z_max, track[:, 1].max())
    
    # Organize tracks by frame - much more efficient: iterate through points, not frames
    # Build a lookup of frame_time -> list of track slices visible in that frame
    tracks_by_frame = {}
    for track_id, track in enumerate(tracks):
        if track.shape[0] == 0:
            continue
        
        x_coords = track[:, 0]
        z_coords = track[:, 1]
        times = track[:, 4]
        
        # Iterate through each point in the track, group by frame time
        for i in range(len(times)):
            frame_time = times[i]
            
            if frame_time not in tracks_by_frame:
                tracks_by_frame[frame_time] = []
            
            # Find if this track already has an entry at this frame
            track_entry = None
            for entry in tracks_by_frame[frame_time]:
                if entry['track_id'] == track_id:
                    track_entry = entry
                    break
            
            if track_entry is None:
                # First point for this track at this frame
                tracks_by_frame[frame_time].append({
                    'track_id': track_id,
                    'x': [x_coords[i]],
                    'z': [z_coords[i]],
                    'all_x': x_coords,
                    'all_z': z_coords,
                    'all_times': times
                })
            else:
                # Append to existing entry for this track at this frame
                track_entry['x'].append(x_coords[i])
                track_entry['z'].append(z_coords[i])
    
    # Convert lists to numpy arrays for efficiency
    for frame_time in tracks_by_frame:
        for track_data in tracks_by_frame[frame_time]:
            track_data['x'] = np.array(track_data['x'])
            track_data['z'] = np.array(track_data['z'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='w')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Z Coordinate", fontsize=12)
    title = ax.set_title("Frame 0", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Store line objects for trails - need to track all tracks, not just current frame
    all_track_data = {}
    for track_id, track in enumerate(tracks):
        if track.shape[0] > 0:
            all_track_data[track_id] = {
                'x': track[:, 0],
                'z': track[:, 1],
                'times': track[:, 4]
            }
    
    # Hold references to Line2D objects so they can be removed on each update call
    trail_lines = {}
    # Solid filled dots (no marker edge) to avoid hollow-looking points
    current_dots = ax.scatter(
        [],
        [],
        c='red',
        s=dot_size**2,
        alpha=0.8,
        zorder=10,
        edgecolors='none',
        linewidths=0,
    )
    
    def init():
        # Initialize animation artists to an empty state before playback starts
        current_dots.set_offsets(np.empty((0, 2)))
        title.set_text("Frame 0")
        return [current_dots, title]
    
    def update(frame_idx):
        frame_time = frames[frame_idx]
        
        # Clear previous trails
        for line in trail_lines.values():
            line.remove()
        trail_lines.clear()
        
        # Collect current microbubbles
        current_positions = []
        
        # Plot trails for all tracks up to current frame
        if show_trails:
            for track_id, track_data in all_track_data.items():
                all_x = track_data['x']
                all_z = track_data['z']
                all_times = track_data['times']
                
                # Show trail up to current frame
                mask = all_times <= frame_time
                if np.sum(mask) > 1:
                    line, = ax.plot(all_x[mask], all_z[mask], 'b-', 
                                   linewidth=line_width, alpha=0.4, zorder=1)
                    trail_lines[track_id] = line
        
        # Collect current frame positions
        for track_data in tracks_by_frame.get(frame_time, []):
            current_positions.extend([
                [x, z] for x, z in zip(track_data['x'], track_data['z'])
            ])
        
        # Update dots
        if current_positions:
            current_dots.set_offsets(np.array(current_positions))
        else:
            current_dots.set_offsets(np.empty((0, 2)))
        
        title.set_text(f"Frame {frame_idx + 1}/{len(frames)} (time={frame_time:.3f})")
        
        return [current_dots, title] + list(trail_lines.values())
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), init_func=init,
        interval=interval, blit=False, repeat=True
    )
    
    if output_file:
        print(f"Saving animation to: {output_file}", flush=True)
        ani.save(output_file, writer='pillow', fps=1000/interval)
        print(f"Animation saved!", flush=True)
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize tracks from MATLAB .mat file format. "
                    "Creates static plot with all tracks and optional frame-by-frame animation."
    )
    parser.add_argument("mat_file", type=str)
    parser.add_argument("--output-static", type=str, default=None,
                       help="Output file for static visualization (e.g., 'tracks.png')")
    parser.add_argument("--output-animation", type=str, default=None,
                       help="Output file for animation (e.g., 'tracks.gif')")
    parser.add_argument("--line-width", type=float, default=0.25,
                       help="Width of track lines (default: 0.5)")
    parser.add_argument("--dot-size", type=float, default=0.0,
                       help="Size of microbubble dots (default: 1.0)")
    parser.add_argument("--no-dots", action="store_true",
                       help="Don't show dots in static visualization")
    parser.add_argument("--no-trails", action="store_true",
                       help="Don't show track trails in animation")
    parser.add_argument("--animation-interval", type=int, default=200,
                       help="Animation interval in milliseconds (default: 200)")
    parser.add_argument("--static-only", action="store_true",
                       help="Only create static visualization")
    parser.add_argument("--animation-only", action="store_true",
                       help="Only create animation")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum number of frames to include in animation (downsamples if needed)")
    
    args = parser.parse_args()
    
    # Load tracks from the MATLAB file into Python-friendly numpy arrays
    tracks = load_tracks_from_mat(args.mat_file)
    
    # Create static visualization
    if not args.animation_only:
        static_output = args.output_static
        if static_output is None and not args.static_only:
            # Default output name
            base_name = os.path.splitext(os.path.basename(args.mat_file))[0]
            static_output = f"{base_name}_static.png"
        
        # Render a static PNG containing the full set of detected tracks
        plot_tracks_static(
            tracks, 
            output_file=static_output,
            line_width=args.line_width,
            dot_size=args.dot_size,
            show_dots=not args.no_dots
        )
    
    # Create animation
    if not args.static_only:
        animation_output = args.output_animation
        if animation_output is None:
            # Default output name
            base_name = os.path.splitext(os.path.basename(args.mat_file))[0]
            animation_output = f"{base_name}_animation.gif"
        
        # Build and optionally downsample an animation that reveals tracks over time
        create_frame_animation(
            tracks,
            output_file=animation_output,
            line_width=args.line_width,
            dot_size=args.dot_size,
            interval=args.animation_interval,
            show_trails=not args.no_trails,
            max_frames=args.max_frames
        )
    
    print("Visualization complete!", flush=True)


if __name__ == "__main__":
    main()

