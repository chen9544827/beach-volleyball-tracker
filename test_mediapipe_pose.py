# test_mediapipe_pose.py
# -*- coding: utf-8 -*-
"""
MediaPipe Pose Landmarker test script.

Tests multi-person pose detection on beach volleyball video segments
using the MediaPipe Tasks API (PoseLandmarker).

Outputs an annotated video with pose landmarks drawn on detected persons,
plus per-frame detection statistics.

Usage:
    python test_mediapipe_pose.py
    python test_mediapipe_pose.py --video path/to/video.mp4
    python test_mediapipe_pose.py --video path/to/video.mp4 --num-poses 6
"""

import os
import sys
import time
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pose_landmarker_heavy.task")

# Default test video (Chetumal match005 - has far-side player issues)
DEFAULT_VIDEO = os.path.join(
    PROJECT_ROOT, "output_data", "video_segments", "Chetumal_WT18_C1",
    "FIVB-BVB-WT18-Chetumal-4Star-251018-C1-MD-W-005-Walsh-Jennings-Sweat-USA-Huber-Hubscher-SUI",
    "normal_segments", "segment_011_Team2.mp4"
)

# Colors for different detected persons (BGR)
PERSON_COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 165, 255),  # Orange
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (255, 255, 0),  # Cyan
]

# MediaPipe Pose landmark connections (33 landmarks)
POSE_CONNECTIONS = vision.PoseLandmarksConnections.POSE_LANDMARKS


def draw_landmarks_on_image(frame, detection_result):
    """Draw pose landmarks for all detected persons on the frame.

    Args:
        frame: BGR image (numpy array).
        detection_result: PoseLandmarkerResult from MediaPipe.

    Returns:
        Annotated frame, number of persons detected.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    num_persons = len(detection_result.pose_landmarks)

    for person_idx, landmarks in enumerate(detection_result.pose_landmarks):
        color = PERSON_COLORS[person_idx % len(PERSON_COLORS)]
        lighter_color = tuple(min(255, c + 80) for c in color)

        # Convert normalized landmarks to pixel coordinates
        pts = []
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            vis = lm.visibility if hasattr(lm, 'visibility') else 1.0
            pts.append((px, py, vis))

        # Draw connections
        for conn in POSE_CONNECTIONS:
            start_idx, end_idx = conn.start, conn.end
            if start_idx >= len(pts) or end_idx >= len(pts):
                continue
            x1, y1, v1 = pts[start_idx]
            x2, y2, v2 = pts[end_idx]
            # Only draw if both points are visible enough
            if v1 > 0.3 and v2 > 0.3:
                cv2.line(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw landmark points
        for pt_idx, (px, py, vis) in enumerate(pts):
            if vis > 0.3:
                radius = 4 if vis > 0.6 else 2
                cv2.circle(annotated, (px, py), radius, lighter_color, -1)

        # Draw person label with bounding box
        visible_pts = [(px, py) for px, py, v in pts if v > 0.3]
        if visible_pts:
            xs = [p[0] for p in visible_pts]
            ys = [p[1] for p in visible_pts]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            # Draw bounding box
            cv2.rectangle(annotated, (x_min - 5, y_min - 5),
                          (x_max + 5, y_max + 5), color, 1)
            # Label
            label = f"P{person_idx + 1}"
            avg_vis = np.mean([v for _, _, v in pts if v > 0.1])
            label += f" ({avg_vis:.2f})"
            cv2.putText(annotated, label, (x_min - 5, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Frame stats overlay
    cv2.putText(annotated, f"MediaPipe Pose: {num_persons} person(s)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated, num_persons


def process_video(video_path, output_path, model_path, num_poses=6,
                  min_detection_conf=0.3, min_tracking_conf=0.3):
    """Process video with MediaPipe PoseLandmarker.

    Args:
        video_path: Input video file path.
        output_path: Output annotated video path.
        model_path: Path to pose_landmarker .task model.
        num_poses: Maximum number of poses to detect.
        min_detection_conf: Minimum detection confidence.
        min_tracking_conf: Minimum tracking confidence.
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("Download with:")
        print("  python -c \"import urllib.request; "
              "urllib.request.urlretrieve("
              "'https://storage.googleapis.com/mediapipe-models/"
              "pose_landmarker/pose_landmarker_heavy/float16/latest/"
              "pose_landmarker_heavy.task', 'models/pose_landmarker_heavy.task')\"")
        sys.exit(1)

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    # --- Initialize PoseLandmarker ---
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_detection_conf,
        min_pose_presence_confidence=min_detection_conf,
        min_tracking_confidence=min_tracking_conf,
        output_segmentation_masks=False,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # --- Open video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video: {os.path.basename(video_path)}")
    print(f"[INFO] Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")
    print(f"[INFO] Model: {os.path.basename(model_path)}")
    print(f"[INFO] num_poses={num_poses}, min_det_conf={min_detection_conf}, "
          f"min_track_conf={min_tracking_conf}")

    # --- Setup output video ---
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- Process frames ---
    frame_idx = 0
    detection_counts = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect poses (VIDEO mode requires timestamp in ms)
        timestamp_ms = int(frame_idx * 1000 / fps)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw landmarks
        annotated, num_persons = draw_landmarks_on_image(frame, result)
        detection_counts.append(num_persons)

        writer.write(annotated)
        frame_idx += 1

        # Progress
        if frame_idx % 50 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed
            eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
            avg_det = np.mean(detection_counts[-50:])
            print(f"  [{frame_idx}/{total_frames}] "
                  f"{fps_actual:.1f} fps, ETA: {eta:.0f}s, "
                  f"avg_persons: {avg_det:.1f}")

    cap.release()
    writer.release()
    landmarker.close()

    elapsed = time.time() - start_time
    fps_actual = frame_idx / elapsed if elapsed > 0 else 0

    # --- Print statistics ---
    print(f"\n{'='*60}")
    print(f"  MediaPipe Pose Landmarker Results")
    print(f"{'='*60}")
    print(f"  Total frames: {frame_idx}")
    print(f"  Processing speed: {fps_actual:.1f} fps")
    print(f"  Processing time: {elapsed:.1f}s")
    print(f"")
    if detection_counts:
        counts = np.array(detection_counts)
        print(f"  Persons detected per frame:")
        print(f"    Average: {counts.mean():.2f}")
        print(f"    Max:     {counts.max()}")
        print(f"    Min:     {counts.min()}")
        print(f"    Median:  {np.median(counts):.0f}")
        # Distribution
        for n in range(0, max(counts) + 1):
            pct = 100.0 * np.sum(counts == n) / len(counts)
            bar = '#' * int(pct / 2)
            print(f"    {n} persons: {np.sum(counts == n):>4d} frames ({pct:5.1f}%) {bar}")
    print(f"")
    print(f"  Output: {output_path}")
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Test MediaPipe Pose Landmarker on beach volleyball video")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO,
                        help="Input video path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: auto-generated)")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Pose landmarker model path")
    parser.add_argument("--num-poses", type=int, default=6,
                        help="Max number of poses to detect (default: 6)")
    parser.add_argument("--min-det-conf", type=float, default=0.3,
                        help="Min detection confidence (default: 0.3)")
    parser.add_argument("--min-track-conf", type=float, default=0.3,
                        help="Min tracking confidence (default: 0.3)")
    args = parser.parse_args()

    # Auto-generate output path
    if args.output is None:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output_dir = os.path.join(PROJECT_ROOT, "test_venue_output", "mediapipe_test")
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"{video_name}_mediapipe_pose.mp4")

    process_video(
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        num_poses=args.num_poses,
        min_detection_conf=args.min_det_conf,
        min_tracking_conf=args.min_track_conf,
    )


if __name__ == "__main__":
    main()
