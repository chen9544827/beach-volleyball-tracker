# tools/prepare_court_dataset.py
# -*- coding: utf-8 -*-
"""
Court Keypoint Dataset Preparation Tool

Scans court_configs/ for existing configs, extracts frames from corresponding videos,
and converts court_config annotations to YOLO keypoint format labels.

6 Keypoints:
    0: far_left    - court far-side left corner
    1: far_right   - court far-side right corner
    2: near_left   - court near-side left corner
    3: near_right  - court near-side right corner
    4: net_left    - net left endpoint (interpolated)
    5: net_right   - net right endpoint (interpolated)

Usage:
    python tools/prepare_court_dataset.py \
        --config-dir court_configs \
        --video-dir input_video/original_video \
        --output dataset/court_keypoints \
        --frames-per-video 5 \
        --val-ratio 0.2

    # Also supports segment directory for shorter clips
    python tools/prepare_court_dataset.py \
        --config-dir court_configs \
        --segment-dir output_data/video_segments \
        --output dataset/court_keypoints
"""

import os
import sys
import json
import cv2
import argparse
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.filename_parser import scan_video_directory, group_videos, parse_filename


def sort_corners(polygon):
    """
    Sort 4 polygon points into: far_left, far_right, near_left, near_right.

    Uses same logic as CourtZones: sort by Y to get top/bottom pairs,
    then sort each pair by X.

    Returns:
        (far_left, far_right, near_left, near_right) as tuples
    """
    pts = [tuple(p) for p in polygon]
    sorted_by_y = sorted(pts, key=lambda p: p[1])
    top_points = sorted(sorted_by_y[:2], key=lambda p: p[0])
    bottom_points = sorted(sorted_by_y[2:], key=lambda p: p[0])
    return top_points[0], top_points[1], bottom_points[0], bottom_points[1]


def compute_net_endpoints(far_left, far_right, near_left, near_right, net_y):
    """
    Compute net left/right endpoints by interpolating court boundary edges at net_y.

    Returns:
        ((net_left_x, net_y), (net_right_x, net_y))
    """
    def intersect_at_y(p1, p2, y):
        if abs(p2[1] - p1[1]) < 1e-6:
            return p1[0]
        t = (y - p1[1]) / (p2[1] - p1[1])
        t = max(0.0, min(1.0, t))
        return p1[0] + (p2[0] - p1[0]) * t

    net_lx = intersect_at_y(far_left, near_left, net_y)
    net_rx = intersect_at_y(far_right, near_right, net_y)
    return (net_lx, net_y), (net_rx, net_y)


def config_to_keypoints(court_config):
    """
    Convert court_config to 6 keypoints.

    Returns:
        List of 6 (x, y) tuples: [far_left, far_right, near_left, near_right, net_left, net_right]
        or None if config is invalid.
    """
    polygon = court_config.get('court_boundary_polygon', [])
    net_y = court_config.get('net_y')
    if len(polygon) < 4 or net_y is None:
        return None

    far_left, far_right, near_left, near_right = sort_corners(polygon)
    net_left, net_right = compute_net_endpoints(far_left, far_right, near_left, near_right, net_y)

    return [far_left, far_right, near_left, near_right, net_left, net_right]


def keypoints_to_yolo_label(keypoints, img_w, img_h, margin_ratio=0.10):
    """
    Convert 6 keypoints to YOLO keypoint label format.

    Format: <class> <cx> <cy> <w> <h> <kp0_x> <kp0_y> <vis> ... <kp5_x> <kp5_y> <vis>
    All coordinates normalized to [0, 1].

    Args:
        keypoints: List of 6 (x, y) tuples
        img_w: Image width in pixels
        img_h: Image height in pixels
        margin_ratio: Bounding box margin as ratio of bbox size

    Returns:
        Label string (single line)
    """
    xs = [kp[0] for kp in keypoints]
    ys = [kp[1] for kp in keypoints]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add margin
    w_margin = (x_max - x_min) * margin_ratio
    h_margin = (y_max - y_min) * margin_ratio
    x_min = max(0, x_min - w_margin)
    y_min = max(0, y_min - h_margin)
    x_max = min(img_w, x_max + w_margin)
    y_max = min(img_h, y_max + h_margin)

    # Normalized bbox center + size
    cx = ((x_min + x_max) / 2) / img_w
    cy = ((y_min + y_max) / 2) / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    # Build label parts
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]

    for kp in keypoints:
        kx = kp[0] / img_w
        ky = kp[1] / img_h
        # Clip to valid range
        kx = max(0.0, min(1.0, kx))
        ky = max(0.0, min(1.0, ky))
        # visibility: 2 = visible
        parts.append(f"{kx:.6f} {ky:.6f} 2")

    return " ".join(parts)


def extract_frames(video_path, num_frames=5, positions=None):
    """
    Extract frames from video at specified relative positions.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        positions: List of relative positions (0.0-1.0), defaults to evenly spaced

    Returns:
        List of (frame_idx, frame_image) tuples
    """
    if positions is None:
        # Avoid very start/end (may have black frames)
        positions = [0.2 + i * 0.1 for i in range(num_frames)]
        # Clip to [0.1, 0.9]
        positions = [max(0.1, min(0.9, p)) for p in positions]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    frames = []
    for pos in positions:
        frame_idx = int(total_frames * pos)
        frame_idx = max(0, min(total_frames - 1, frame_idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append((frame_idx, frame))

    cap.release()
    return frames


def find_video_for_group(group_key, video_dir=None, segment_dir=None):
    """
    Find a video file belonging to the given group_key.

    Searches segment_dir first (shorter clips), then video_dir.

    Returns:
        Video file path, or None
    """
    videos = find_all_videos_for_group(group_key, video_dir, segment_dir)
    return videos[0] if videos else None


def find_all_videos_for_group(group_key, video_dir=None, segment_dir=None):
    """
    Find ALL video files belonging to the given group_key.

    Searches both segment_dir and video_dir. For segments, also searches
    by group_key subdirectory name (segments may not have parseable filenames).

    Returns:
        List of video file paths
    """
    video_exts = ('.mp4', '.avi', '.mkv', '.mov')
    found = []

    # Search segment_dir: look in group_key subdirectory if it exists
    if segment_dir and os.path.isdir(segment_dir):
        group_subdir = os.path.join(segment_dir, group_key)
        if os.path.isdir(group_subdir):
            for root, dirs, files in os.walk(group_subdir):
                for f in sorted(files):
                    if any(f.lower().endswith(ext) for ext in video_exts):
                        found.append(os.path.join(root, f))

        # Also try filename-based matching
        if not found:
            for root, dirs, files in os.walk(segment_dir):
                for f in sorted(files):
                    if not any(f.lower().endswith(ext) for ext in video_exts):
                        continue
                    parsed = parse_filename(f)
                    if parsed and parsed.get('group_key') == group_key:
                        full_path = os.path.join(root, f)
                        if full_path not in found:
                            found.append(full_path)

    # Search video_dir by filename parsing
    if video_dir and os.path.isdir(video_dir):
        for root, dirs, files in os.walk(video_dir):
            for f in sorted(files):
                if not any(f.lower().endswith(ext) for ext in video_exts):
                    continue
                parsed = parse_filename(f)
                if parsed and parsed.get('group_key') == group_key:
                    full_path = os.path.join(root, f)
                    if full_path not in found:
                        found.append(full_path)

    return found


def prepare_dataset(config_dir, output_dir, video_dir=None, segment_dir=None,
                    frames_per_video=5, frames_per_segment=1, val_ratio=0.2,
                    max_segments_per_group=None, skip_groups=None):
    """
    Main dataset preparation pipeline.

    Extracts frames from original videos AND all segment clips to maximize
    dataset size. Each segment contributes frames_per_segment frames.

    Args:
        config_dir: Directory containing court_config JSON files
        output_dir: Output dataset directory
        video_dir: Original video directory
        segment_dir: Video segments directory
        frames_per_video: Frames to extract per original video (default: 5)
        frames_per_segment: Frames to extract per segment clip (default: 1)
        val_ratio: Fraction of groups for validation set
        max_segments_per_group: Max segment clips per group (None=unlimited)
        skip_groups: Set of group_keys to skip (e.g. aerial views)
    """
    print("=" * 70)
    print("Court Keypoint Dataset Preparation")
    print("=" * 70)

    if skip_groups is None:
        skip_groups = set()

    # Scan config files
    config_files = [f for f in os.listdir(config_dir)
                    if f.endswith('.json') and not f.startswith('.')]
    print(f"Found {len(config_files)} config files in {config_dir}")

    if not config_files:
        print("[ERROR] No court config files found")
        return

    # Setup output dirs - clear old data first
    for split in ('train', 'val'):
        for subdir in ('images', 'labels'):
            folder = os.path.join(output_dir, subdir, split)
            if os.path.isdir(folder):
                for f in os.listdir(folder):
                    os.remove(os.path.join(folder, f))
            os.makedirs(folder, exist_ok=True)

    # Collect valid configs
    all_samples = []
    skipped = 0

    for config_file in sorted(config_files):
        group_key = os.path.splitext(config_file)[0]

        if group_key in skip_groups:
            print(f"  [SKIP] {group_key} (in skip list)")
            skipped += 1
            continue

        config_path = os.path.join(config_dir, config_file)

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                court_config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [WARNING] Cannot read {config_file}: {e}")
            skipped += 1
            continue

        keypoints = config_to_keypoints(court_config)
        if keypoints is None:
            print(f"  [WARNING] Invalid config (missing polygon/net_y): {config_file}")
            skipped += 1
            continue

        all_samples.append((group_key, config_path, court_config, keypoints))

    print(f"Valid configs: {len(all_samples)}, Skipped: {skipped}")

    if not all_samples:
        print("[ERROR] No valid configs to process")
        return

    # Shuffle and split by group
    random.shuffle(all_samples)
    val_count = max(1, int(len(all_samples) * val_ratio))
    val_groups = set(s[0] for s in all_samples[:val_count])

    print(f"Val groups: {val_groups}")
    print()

    total_images = 0
    train_count = 0
    val_count_actual = 0

    for group_key, config_path, court_config, keypoints in all_samples:
        split = 'val' if group_key in val_groups else 'train'
        group_images = 0

        # Find ALL videos for this group (original + segments)
        all_videos = find_all_videos_for_group(group_key, video_dir, segment_dir)

        if not all_videos:
            print(f"  [SKIP] No video found for {group_key}")
            continue

        # Separate original videos vs segments
        original_videos = []
        segment_videos = []
        for vpath in all_videos:
            if segment_dir and segment_dir in vpath:
                segment_videos.append(vpath)
            else:
                original_videos.append(vpath)

        # Limit segments if requested
        if max_segments_per_group and len(segment_videos) > max_segments_per_group:
            random.shuffle(segment_videos)
            segment_videos = segment_videos[:max_segments_per_group]

        # Process original videos (more frames each)
        for vidx, video_path in enumerate(original_videos):
            frames = extract_frames(video_path, num_frames=frames_per_video)
            for i, (frame_idx, frame_img) in enumerate(frames):
                img_h, img_w = frame_img.shape[:2]
                label_str = keypoints_to_yolo_label(keypoints, img_w, img_h)

                img_name = f"{group_key}_orig{vidx}_f{frame_idx:06d}_{i}.jpg"
                label_name = f"{group_key}_orig{vidx}_f{frame_idx:06d}_{i}.txt"

                cv2.imwrite(os.path.join(output_dir, 'images', split, img_name), frame_img)
                with open(os.path.join(output_dir, 'labels', split, label_name), 'w') as f:
                    f.write(label_str + '\n')
                group_images += 1

        # Process segment clips (fewer frames each, but many segments)
        for sidx, seg_path in enumerate(segment_videos):
            # For segments, sample from middle to avoid transition frames
            frames = extract_frames(seg_path, num_frames=frames_per_segment,
                                    positions=[0.5] if frames_per_segment == 1
                                    else None)
            for i, (frame_idx, frame_img) in enumerate(frames):
                img_h, img_w = frame_img.shape[:2]
                label_str = keypoints_to_yolo_label(keypoints, img_w, img_h)

                img_name = f"{group_key}_seg{sidx:03d}_f{frame_idx:06d}_{i}.jpg"
                label_name = f"{group_key}_seg{sidx:03d}_f{frame_idx:06d}_{i}.txt"

                cv2.imwrite(os.path.join(output_dir, 'images', split, img_name), frame_img)
                with open(os.path.join(output_dir, 'labels', split, label_name), 'w') as f:
                    f.write(label_str + '\n')
                group_images += 1

        total_images += group_images
        if split == 'train':
            train_count += group_images
        else:
            val_count_actual += group_images

        print(f"  [OK] {group_key}: {len(original_videos)} orig + "
              f"{len(segment_videos)} segs = {group_images} frames -> {split}")

    # Write dataset YAML
    yaml_path = os.path.join(output_dir, 'court_keypoints.yaml')
    abs_output = os.path.abspath(output_dir)
    yaml_content = f"""# Court Keypoint Detection Dataset
# Auto-generated by prepare_court_dataset.py

path: {abs_output}
train: images/train
val: images/val

kpt_shape: [6, 3]
flip_idx: [1, 0, 3, 2, 5, 4]

nc: 1
names: ['court']
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print()
    print("=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    print(f"Total images:  {total_images}")
    print(f"  Train:       {train_count}")
    print(f"  Val:         {val_count_actual}")
    print(f"YAML config:   {yaml_path}")
    print(f"Output dir:    {abs_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare YOLO keypoint dataset from court configs + videos"
    )
    parser.add_argument("--config-dir", type=str, default="court_configs",
                        help="Directory containing court_config JSON files")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Original video directory")
    parser.add_argument("--segment-dir", type=str, default=None,
                        help="Video segments directory (preferred, shorter clips)")
    parser.add_argument("--output", type=str, default="dataset/court_keypoints",
                        help="Output dataset directory")
    parser.add_argument("--frames-per-video", type=int, default=5,
                        help="Frames to extract per original video (default: 5)")
    parser.add_argument("--frames-per-segment", type=int, default=1,
                        help="Frames to extract per segment clip (default: 1)")
    parser.add_argument("--max-segments", type=int, default=None,
                        help="Max segment clips per group (default: unlimited)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation set ratio (default: 0.2)")
    parser.add_argument("--skip-groups", type=str, nargs='*', default=None,
                        help="Group keys to skip (e.g. Hamburg_WT19_C1 for aerial view)")

    args = parser.parse_args()

    if not args.video_dir and not args.segment_dir:
        parser.error("At least one of --video-dir or --segment-dir is required")

    if not os.path.isdir(args.config_dir):
        print(f"[ERROR] Config directory not found: {args.config_dir}")
        sys.exit(1)

    skip_groups = set(args.skip_groups) if args.skip_groups else set()

    prepare_dataset(
        config_dir=args.config_dir,
        output_dir=args.output,
        video_dir=args.video_dir,
        segment_dir=args.segment_dir,
        frames_per_video=args.frames_per_video,
        frames_per_segment=args.frames_per_segment,
        val_ratio=args.val_ratio,
        max_segments_per_group=args.max_segments,
        skip_groups=skip_groups,
    )


if __name__ == "__main__":
    main()
