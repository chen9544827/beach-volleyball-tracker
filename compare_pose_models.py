#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare yolov8m-pose vs yolo26m-pose on the same video.
Generates side-by-side or individual annotated videos.
"""

import os
import sys
import time
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from ultralytics import YOLO

# COCO keypoint connections for skeleton drawing
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12),  # torso
    (11, 12),  # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Colors for keypoints by body part
KP_COLORS = {
    'head': (0, 255, 255),      # yellow
    'arm': (0, 165, 255),       # orange
    'wrist': (0, 0, 255),       # red (important for serve)
    'leg': (255, 165, 0),       # blue
    'ankle': (255, 0, 255),     # magenta (important for jump serve)
    'torso': (0, 255, 0),       # green
}

def get_kp_color(idx):
    if idx in [0, 1, 2, 3, 4]:
        return KP_COLORS['head']
    elif idx in [5, 6, 11, 12]:
        return KP_COLORS['torso']
    elif idx in [7, 8]:
        return KP_COLORS['arm']
    elif idx in [9, 10]:
        return KP_COLORS['wrist']
    elif idx in [13, 14]:
        return KP_COLORS['leg']
    elif idx in [15, 16]:
        return KP_COLORS['ankle']
    return (255, 255, 255)


def draw_detections(frame, results, model_name):
    """Draw bounding boxes, keypoints, and skeleton on frame."""
    annotated = frame.copy()

    if not results or len(results) == 0:
        return annotated

    r = results[0]
    n_det = 0

    if r.boxes is not None and r.keypoints is not None:
        boxes = r.boxes
        kps_data = r.keypoints.data.cpu().numpy()  # [N, 17, 3]

        for i in range(len(boxes)):
            n_det += 1
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

            # Draw box
            color = (0, 255, 0) if conf > 0.5 else (0, 200, 200)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f'{conf:.2f}', (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw skeleton
            if i < len(kps_data):
                person_kps = kps_data[i]  # [17, 3]

                # Draw connections
                for (a, b) in SKELETON:
                    if person_kps[a][2] > 0.3 and person_kps[b][2] > 0.3:
                        pt1 = (int(person_kps[a][0]), int(person_kps[a][1]))
                        pt2 = (int(person_kps[b][0]), int(person_kps[b][1]))
                        cv2.line(annotated, pt1, pt2, (200, 200, 200), 1)

                # Draw keypoints
                for k in range(17):
                    kx, ky, kc = person_kps[k]
                    if kc > 0.3:
                        pt = (int(kx), int(ky))
                        kcolor = get_kp_color(k)
                        radius = 4 if k in [9, 10, 15, 16] else 3  # larger for wrist/ankle
                        cv2.circle(annotated, pt, radius, kcolor, -1)

    # Draw model name and detection count
    cv2.rectangle(annotated, (0, 0), (350, 35), (0, 0, 0), -1)
    cv2.putText(annotated, f'{model_name} | {n_det} persons',
               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated


def generate_videos(video_path, output_dir, num_frames=300, imgsz=1280):
    """Generate annotated videos for both models."""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

    model_configs = [
        ('yolov8m-pose.pt', 'yolov8m-pose'),
        ('yolo26m-pose.pt', 'yolo26m-pose'),
    ]

    os.makedirs(output_dir, exist_ok=True)

    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(num_frames, total)
    cap.release()

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Read all frames first
    print(f"Reading {num_frames} frames from {video_name}...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"  Read {len(frames)} frames ({width}x{height}, {fps:.1f}fps)")

    for model_file, model_name in model_configs:
        model_path = os.path.join(models_dir, model_file)
        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found: {model_path}")
            continue

        print(f"\nProcessing with {model_name}...")
        model = YOLO(model_path)

        out_path = os.path.join(output_dir, f'{video_name}_{model_name}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        times = []
        for i, frame in enumerate(frames):
            t0 = time.perf_counter()
            results = model(frame, conf=0.15, classes=[0], verbose=False,
                          imgsz=imgsz, max_det=100, iou=0.6)
            t1 = time.perf_counter()

            if i >= 5:  # skip warmup
                times.append(t1 - t0)

            annotated = draw_detections(frame, results, model_name)

            # Add FPS overlay
            if times:
                current_fps = 1.0 / times[-1] if len(times) > 5 else 1.0 / np.mean(times)
                cv2.putText(annotated, f'FPS: {current_fps:.1f}',
                           (width - 180, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Frame counter
            cv2.putText(annotated, f'Frame {i+1}/{len(frames)}',
                       (width - 220, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            writer.write(annotated)

            if (i + 1) % 50 == 0:
                avg_fps = 1.0 / np.mean(times) if times else 0
                print(f"  Frame {i+1}/{len(frames)} | Avg FPS: {avg_fps:.1f}")

        writer.release()
        avg_fps = 1.0 / np.mean(times) if times else 0
        print(f"  Saved: {out_path}")
        print(f"  Average FPS: {avg_fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Compare pose models - generate videos')
    parser.add_argument('--video', type=str, required=True, help='Video path')
    parser.add_argument('--output', type=str, default='model_comparison', help='Output directory')
    parser.add_argument('--frames', type=int, default=300, help='Number of frames')
    parser.add_argument('--imgsz', type=int, default=1280, help='Inference image size')
    args = parser.parse_args()

    generate_videos(args.video, args.output, args.frames, args.imgsz)


if __name__ == '__main__':
    main()
