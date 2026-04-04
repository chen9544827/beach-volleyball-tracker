# core/static_player_filter.py
# -*- coding: utf-8 -*-
"""
Static Player Filter - Remove stationary off-court persons from tracking JSON.

Detects and removes player detections that remain stationary for extended
periods (e.g., photographers, referees, ball retrievers). These persons
are identified by their lack of movement over a sliding window.

Algorithm:
1. Build position tracks by nearest-neighbor matching across frames
2. For each track, use a sliding window to detect static segments
3. Remove static detections from player_detections
4. Protection: always keep at least min_players_per_frame per frame

Usage:
    from core.static_player_filter import filter_static_players
    filtered_data = filter_static_players(json_data)

CLI:
    python -m core.static_player_filter --json tracking.json --output filtered.json
"""

import math
from typing import Dict, List, Set, Tuple


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _build_tracks(
    frames: List[Dict],
    match_threshold_px: float = 50.0,
) -> List[List[Tuple[int, float, float, int]]]:
    """Build position tracks by nearest-neighbor matching across frames.

    Args:
        frames: List of frame dicts with 'frame_id' and 'player_detections'.
        match_threshold_px: Max distance to link a detection to an existing track.

    Returns:
        List of tracks. Each track is a list of (frame_id, cx, cy, det_index).
    """
    tracks: List[List[Tuple[int, float, float, int]]] = []
    # Last known position for each active track: track_idx -> (cx, cy, last_frame_id)
    active: Dict[int, Tuple[float, float, int]] = {}

    for frame in frames:
        frame_id = frame.get("frame_id", 0)
        players = frame.get("player_detections", [])

        # Collect current detections
        current_dets: List[Tuple[float, float, int]] = []
        for det_idx, det in enumerate(players):
            cp = det.get("center_point")
            if cp and len(cp) >= 2:
                current_dets.append((float(cp[0]), float(cp[1]), det_idx))

        # Match current detections to active tracks (greedy nearest-neighbor)
        matched_tracks: Set[int] = set()
        matched_dets: Set[int] = set()

        # Build candidate pairs sorted by distance
        pairs = []
        for d_i, (cx, cy, _) in enumerate(current_dets):
            for t_idx, (tx, ty, last_fid) in active.items():
                # Only match if track was seen recently (within 5 frames gap)
                if frame_id - last_fid > 5:
                    continue
                dist = _distance((cx, cy), (tx, ty))
                if dist < match_threshold_px:
                    pairs.append((dist, d_i, t_idx))

        pairs.sort(key=lambda x: x[0])

        for dist, d_i, t_idx in pairs:
            if d_i in matched_dets or t_idx in matched_tracks:
                continue
            matched_dets.add(d_i)
            matched_tracks.add(t_idx)
            cx, cy, det_idx = current_dets[d_i]
            tracks[t_idx].append((frame_id, cx, cy, det_idx))
            active[t_idx] = (cx, cy, frame_id)

        # Unmatched detections start new tracks
        for d_i, (cx, cy, det_idx) in enumerate(current_dets):
            if d_i not in matched_dets:
                t_idx = len(tracks)
                tracks.append([(frame_id, cx, cy, det_idx)])
                active[t_idx] = (cx, cy, frame_id)

    return tracks


def _find_static_detections(
    tracks: List[List[Tuple[int, float, float, int]]],
    window_size: int,
    max_displacement_px: float,
) -> Set[Tuple[int, int]]:
    """Find (frame_id, det_index) pairs that are static.

    Args:
        tracks: Output from _build_tracks.
        window_size: Sliding window size in frames.
        max_displacement_px: Max displacement to consider static.

    Returns:
        Set of (frame_id, det_index) to remove.
    """
    static_set: Set[Tuple[int, int]] = set()

    for track in tracks:
        if len(track) < window_size:
            continue

        for start in range(len(track) - window_size + 1):
            window = track[start : start + window_size]

            # Check that window spans enough actual frames
            first_fid = window[0][0]
            last_fid = window[-1][0]
            # Allow some gaps but window should roughly span window_size frames
            if last_fid - first_fid > window_size * 2:
                continue

            # Compute mean position
            mean_x = sum(p[1] for p in window) / len(window)
            mean_y = sum(p[2] for p in window) / len(window)

            # Compute max displacement from mean
            max_disp = max(
                _distance((p[1], p[2]), (mean_x, mean_y)) for p in window
            )

            if max_disp < max_displacement_px:
                for fid, _, _, det_idx in window:
                    static_set.add((fid, det_idx))

    return static_set


def filter_static_players(
    json_data: dict,
    static_seconds: float = 5.0,
    max_displacement_px: float = 30.0,
    min_players_per_frame: int = 4,
) -> dict:
    """Remove static (non-moving) player detections from tracking JSON.

    Identifies persons who remain stationary for extended periods
    (photographers, referees, etc.) and removes them.

    Args:
        json_data: Full tracking JSON with 'metadata' and 'frames'.
        static_seconds: Duration threshold for static detection (default 5s).
        max_displacement_px: Max displacement at 720p to be static (default 30px).
        min_players_per_frame: Minimum players to keep per frame (default 4).

    Returns:
        Modified json_data with static players removed (modified in-place).
    """
    metadata = json_data.get("metadata", {})
    frames = json_data.get("frames", [])

    if not frames:
        return json_data

    fps = metadata.get("fps", 25.0)
    video_height = metadata.get("frame_height") or 720
    resolution_scale = video_height / 720.0

    window_size = max(1, int(fps * static_seconds))
    scaled_displacement = max_displacement_px * resolution_scale
    scaled_match_threshold = 50.0 * resolution_scale

    # Build tracks
    tracks = _build_tracks(frames, match_threshold_px=scaled_match_threshold)

    # Find static detections
    static_set = _find_static_detections(tracks, window_size, scaled_displacement)

    if not static_set:
        return json_data

    # Remove static detections, respecting min_players_per_frame
    total_removed = 0
    for frame in frames:
        frame_id = frame.get("frame_id", 0)
        players = frame.get("player_detections", [])

        if not players:
            continue

        # Find which detection indices are static in this frame
        static_indices = set()
        for det_idx in range(len(players)):
            if (frame_id, det_idx) in static_set:
                static_indices.add(det_idx)

        if not static_indices:
            continue

        # Ensure we keep at least min_players_per_frame
        remaining = len(players) - len(static_indices)
        if remaining < min_players_per_frame:
            # Keep the most confident static detections to meet minimum
            need_to_keep = min_players_per_frame - remaining
            static_list = sorted(
                static_indices,
                key=lambda i: players[i].get("confidence", 0),
                reverse=True,
            )
            for i in range(min(need_to_keep, len(static_list))):
                static_indices.discard(static_list[i])

        if static_indices:
            frame["player_detections"] = [
                det
                for i, det in enumerate(players)
                if i not in static_indices
            ]
            total_removed += len(static_indices)

    # Add filter metadata
    metadata["static_player_filter"] = {
        "removed_detections": total_removed,
        "static_seconds": static_seconds,
        "max_displacement_px": max_displacement_px,
        "window_size": window_size,
        "tracks_found": len(tracks),
    }

    return json_data


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Remove static off-court persons from tracking JSON"
    )
    parser.add_argument(
        "--json", required=True, help="Input tracking JSON file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file (default: overwrite input)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Static duration threshold in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Max displacement in pixels at 720p (default: 30.0)",
    )
    parser.add_argument(
        "--min-players",
        type=int,
        default=4,
        help="Minimum players to keep per frame (default: 4)",
    )

    args = parser.parse_args()

    input_path = args.json
    output_path = args.output or input_path

    print(f"[INFO] Loading {input_path} ...")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

    total_frames = len(data.get("frames", []))
    before_counts = [
        len(fr.get("player_detections", []))
        for fr in data.get("frames", [])
    ]

    print(f"[INFO] Frames: {total_frames}")
    if before_counts:
        print(
            f"[INFO] Players per frame before: "
            f"avg={sum(before_counts)/len(before_counts):.1f}, "
            f"min={min(before_counts)}, max={max(before_counts)}"
        )

    filter_static_players(
        data,
        static_seconds=args.seconds,
        max_displacement_px=args.threshold,
        min_players_per_frame=args.min_players,
    )

    after_counts = [
        len(fr.get("player_detections", []))
        for fr in data.get("frames", [])
    ]
    filter_info = data.get("metadata", {}).get("static_player_filter", {})

    print(f"[INFO] Tracks found: {filter_info.get('tracks_found', 0)}")
    print(f"[INFO] Removed detections: {filter_info.get('removed_detections', 0)}")
    if after_counts:
        print(
            f"[INFO] Players per frame after: "
            f"avg={sum(after_counts)/len(after_counts):.1f}, "
            f"min={min(after_counts)}, max={max(after_counts)}"
        )

    print(f"[INFO] Saving to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print("[OK] Done.")
