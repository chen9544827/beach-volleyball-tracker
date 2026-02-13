# test_yolo_det_comparison.py
# -*- coding: utf-8 -*-
"""
Compare player detection: YOLOv8m-pose vs YOLO11m detection model.
Tests multiple imgsz to find the best configuration for far-side players.

Outputs a side-by-side annotated video + detection statistics.
"""

import os
import sys
import time
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_VIDEO = os.path.join(
    PROJECT_ROOT, "output_data", "video_segments", "Chetumal_WT18_C1",
    "FIVB-BVB-WT18-Chetumal-4Star-251018-C1-MD-W-005-Walsh-Jennings-Sweat-USA-Huber-Hubscher-SUI",
    "normal_segments", "segment_011_Team2.mp4"
)

# Colors
COLOR_POSE = (0, 255, 0)      # Green - pose model
COLOR_DET = (0, 165, 255)     # Orange - detection model
COLOR_DET_ONLY = (0, 0, 255)  # Red - det-only (not found by pose)


def detect_with_pose(frame, model, imgsz=640, conf=0.15):
    """Run YOLOv8m-pose and return detections with keypoints."""
    results = model(frame, conf=conf, classes=[0], verbose=False,
                    imgsz=imgsz, max_det=100, iou=0.45)
    detections = []
    if results and results[0].boxes:
        has_kpts = results[0].keypoints is not None
        for i in range(len(results[0].boxes)):
            box = results[0].boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf_val = float(box.conf[0].cpu().numpy())
            keypoints = []
            if has_kpts:
                kpts = results[0].keypoints[i]
                if kpts.xy is not None and kpts.conf is not None:
                    kpts_xy = kpts.xy[0].cpu().numpy()
                    kpts_conf = kpts.conf[0].cpu().numpy()
                    for k in range(kpts_xy.shape[0]):
                        keypoints.append((
                            int(kpts_xy[k, 0]),
                            int(kpts_xy[k, 1]),
                            float(kpts_conf[k])
                        ))
            detections.append({
                "box": [x1, y1, x2, y2],
                "conf": conf_val,
                "center": ((x1+x2)//2, (y1+y2)//2),
                "keypoints": keypoints,
            })
    return detections


def detect_with_det(frame, model, imgsz=640, conf=0.15):
    """Run YOLO11m detection and return person detections."""
    results = model(frame, conf=conf, classes=[0], verbose=False,
                    imgsz=imgsz, max_det=100, iou=0.45)
    detections = []
    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf_val = float(box.conf[0].cpu().numpy())
            detections.append({
                "box": [x1, y1, x2, y2],
                "conf": conf_val,
                "center": ((x1+x2)//2, (y1+y2)//2),
            })
    return detections


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0:
        return 0.0
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter)


def find_det_only(pose_dets, det_dets, iou_thresh=0.4):
    """Find detections in det_dets that don't overlap with any pose_dets."""
    det_only = []
    for dd in det_dets:
        matched = False
        for pd in pose_dets:
            if compute_iou(dd["box"], pd["box"]) > iou_thresh:
                matched = True
                break
        if not matched:
            det_only.append(dd)
    return det_only


# COCO-17 skeleton connections for YOLO pose
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # Head
    (5, 6),                                     # Shoulders
    (5, 7), (7, 9),                             # Left arm
    (6, 8), (8, 10),                            # Right arm
    (5, 11), (6, 12),                           # Torso
    (11, 12),                                   # Hips
    (11, 13), (13, 15),                         # Left leg
    (12, 14), (14, 16),                         # Right leg
]

# Keypoint colors by region
KPT_COLORS = {
    "head": (255, 200, 0),     # Cyan-ish
    "arm": (0, 200, 255),      # Orange
    "leg": (200, 0, 255),      # Purple
    "torso": (0, 255, 200),    # Green-cyan
}

def _kpt_region(idx):
    if idx <= 4:
        return "head"
    if idx in (5, 6, 11, 12):
        return "torso"
    if idx in (7, 8, 9, 10):
        return "arm"
    return "leg"


def draw_detections(frame, pose_dets, det_dets, det_only, label_prefix=""):
    """Draw all detections on frame with pose keypoints."""
    annotated = frame.copy()

    # Draw det-model detections (orange, thin) - background layer
    for d in det_dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_DET, 1)

    # Draw pose detections (green) with keypoints
    for d in pose_dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_POSE, 2)
        cv2.putText(annotated, f"P {d['conf']:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_POSE, 1)

        kpts = d.get("keypoints", [])
        if kpts:
            # Draw skeleton connections
            for (a, b) in SKELETON_CONNECTIONS:
                if a < len(kpts) and b < len(kpts):
                    ax, ay, ac = kpts[a]
                    bx, by, bc = kpts[b]
                    if ac > 0.3 and bc > 0.3 and ax > 0 and bx > 0:
                        cv2.line(annotated, (ax, ay), (bx, by),
                                 COLOR_POSE, 1, cv2.LINE_AA)

            # Draw keypoint dots
            for k_idx, (kx, ky, kc) in enumerate(kpts):
                if kc > 0.3 and kx > 0 and ky > 0:
                    color = KPT_COLORS[_kpt_region(k_idx)]
                    radius = 4 if kc > 0.6 else 2
                    cv2.circle(annotated, (kx, ky), radius, color, -1)

    # Highlight det-only (red, thick) - these are the ones pose missed
    for d in det_only:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_DET_ONLY, 3)
        cv2.putText(annotated, f"DET-ONLY {d['conf']:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DET_ONLY, 2)

    # Stats overlay
    cv2.putText(annotated, f"{label_prefix}Pose:{len(pose_dets)} Det:{len(det_dets)} "
                f"Det-only:{len(det_only)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO pose vs detection model")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO)
    parser.add_argument("--pose-model", type=str,
                        default=os.path.join(PROJECT_ROOT, "models", "yolov8m-pose.pt"))
    parser.add_argument("--det-model", type=str,
                        default=os.path.join(PROJECT_ROOT, "models", "yolo11m.pt"))
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280],
                        help="imgsz values to test (default: 640 1280)")
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    # Load models
    print("[INFO] Loading models...")
    pose_model = YOLO(args.pose_model)
    det_model = YOLO(args.det_model)

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video: {os.path.basename(args.video)}")
    print(f"[INFO] {w}x{h}, {fps:.0f}fps, {total_frames} frames")
    print(f"[INFO] Testing imgsz: {args.imgsz}")

    output_dir = os.path.join(PROJECT_ROOT, "test_venue_output", "yolo_det_comparison")
    os.makedirs(output_dir, exist_ok=True)

    # For each imgsz, process full video
    for imgsz in args.imgsz:
        print(f"\n{'='*60}")
        print(f"  imgsz = {imgsz}")
        print(f"{'='*60}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        out_path = os.path.join(output_dir, f"{video_name}_compare_imgsz{imgsz}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (w, h))

        stats = {"pose_counts": [], "det_counts": [], "det_only_counts": []}
        frame_idx = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pose_dets = detect_with_pose(frame, pose_model, imgsz=imgsz, conf=args.conf)
            det_dets = detect_with_det(frame, det_model, imgsz=imgsz, conf=args.conf)
            det_only = find_det_only(pose_dets, det_dets)

            stats["pose_counts"].append(len(pose_dets))
            stats["det_counts"].append(len(det_dets))
            stats["det_only_counts"].append(len(det_only))

            annotated = draw_detections(frame, pose_dets, det_dets, det_only,
                                        label_prefix=f"imgsz={imgsz} ")
            writer.write(annotated)
            frame_idx += 1

            if frame_idx % 50 == 0:
                elapsed = time.time() - t0
                fps_actual = frame_idx / elapsed
                avg_do = np.mean(stats["det_only_counts"][-50:])
                print(f"  [{frame_idx}/{total_frames}] {fps_actual:.1f}fps "
                      f"avg_det_only: {avg_do:.1f}")

        writer.release()
        elapsed = time.time() - t0

        # Print stats
        pc = np.array(stats["pose_counts"])
        dc = np.array(stats["det_counts"])
        doc = np.array(stats["det_only_counts"])
        print(f"\n  Results (imgsz={imgsz}):")
        print(f"    Speed: {frame_idx/elapsed:.1f} fps ({elapsed:.1f}s)")
        print(f"    Pose model:  avg={pc.mean():.2f}, max={pc.max()}")
        print(f"    Det model:   avg={dc.mean():.2f}, max={dc.max()}")
        print(f"    Det-only:    avg={doc.mean():.2f}, max={doc.max()} "
              f"(frames with >0: {np.sum(doc>0)}/{len(doc)})")
        print(f"    Output: {out_path}")
        if os.path.exists(out_path):
            print(f"    Size: {os.path.getsize(out_path)/1024/1024:.1f} MB")

    cap.release()
    print(f"\n[DONE] All results in: {output_dir}")


if __name__ == "__main__":
    main()
