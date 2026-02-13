# tools/predict_court.py
# -*- coding: utf-8 -*-
"""
Court Keypoint Detection - Inference + Visualization

Use trained YOLO keypoint model to detect court boundary from video/image,
and output annotated images showing:
  - Green polygon: court boundary (4 corners)
  - Yellow line: net position (2 endpoints)
  - Purple semi-transparent: auto-generated exclusion zones
  - Keypoint labels with confidence scores

Usage:
    # Single video (sample multiple frames)
    python tools/predict_court.py --input input_video/original_video/match.mp4 --model models/court_best.pt

    # Single image
    python tools/predict_court.py --input frame.jpg --model models/court_best.pt

    # Directory of videos
    python tools/predict_court.py --input input_video/original_video --model models/court_best.pt

    # Custom output directory
    python tools/predict_court.py --input video.mp4 --model models/court_best.pt --output predict_output

    # Also export court_config JSON
    python tools/predict_court.py --input video.mp4 --model models/court_best.pt --export-config
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.auto_court_detector import AutoCourtDetector, save_court_config

# Keypoint indices (must match auto_court_detector.py)
KP_FAR_LEFT = 0
KP_FAR_RIGHT = 1
KP_NEAR_LEFT = 2
KP_NEAR_RIGHT = 3
KP_NET_LEFT = 4
KP_NET_RIGHT = 5
NUM_KEYPOINTS = 6
KP_NAMES = ['far_left', 'far_right', 'near_left', 'near_right', 'net_left', 'net_right']


def draw_detection(frame, keypoints, per_kp_conf=None, confidence=0.0,
                   num_detections=1, court_config=None):
    """
    Draw full detection visualization on frame.

    Args:
        frame: BGR image (will be copied)
        keypoints: (6, 2) array of keypoint positions
        per_kp_conf: (6,) per-keypoint confidence, or None
        confidence: Overall detection confidence
        num_detections: Number of frames used for detection
        court_config: Optional generated court_config (for exclusion zones)

    Returns:
        Annotated frame copy
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Colors
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)
    PURPLE = (180, 0, 180)
    RED = (0, 0, 255)

    kps = keypoints.astype(int)

    # 1. Draw exclusion zones first (semi-transparent purple, behind everything)
    if court_config and court_config.get('exclusion_zones'):
        overlay = vis.copy()
        for zone in court_config['exclusion_zones']:
            poly = np.array(zone['polygon'], dtype=np.int32)
            cv2.fillPoly(overlay, [poly], PURPLE)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
        # Draw exclusion zone borders
        for zone in court_config['exclusion_zones']:
            poly = np.array(zone['polygon'], dtype=np.int32)
            cv2.polylines(vis, [poly], True, PURPLE, 1)

    # 2. Draw court boundary polygon (green, thick)
    court_poly = np.array([kps[KP_FAR_LEFT], kps[KP_FAR_RIGHT],
                           kps[KP_NEAR_RIGHT], kps[KP_NEAR_LEFT]])
    cv2.polylines(vis, [court_poly], True, GREEN, 3)

    # 3. Draw net line (yellow, thick)
    cv2.line(vis, tuple(kps[KP_NET_LEFT]), tuple(kps[KP_NET_RIGHT]), YELLOW, 3)

    # 4. Draw net_y horizontal reference line (yellow, thin, dashed-like)
    if court_config and court_config.get('net_y'):
        net_y = court_config['net_y']
        for x_start in range(0, w, 20):
            x_end = min(x_start + 10, w)
            cv2.line(vis, (x_start, net_y), (x_end, net_y), (0, 200, 200), 1)

    # 5. Draw keypoints with labels
    kp_colors = [GREEN, GREEN, BLUE, BLUE, YELLOW, YELLOW]
    for i in range(NUM_KEYPOINTS):
        px, py = int(keypoints[i][0]), int(keypoints[i][1])
        color = kp_colors[i]

        # Circle with outline
        cv2.circle(vis, (px, py), 8, (0, 0, 0), -1)  # black outline
        cv2.circle(vis, (px, py), 6, color, -1)

        # Label with confidence
        if per_kp_conf is not None:
            label = f"{KP_NAMES[i]} ({per_kp_conf[i]:.2f})"
        else:
            label = KP_NAMES[i]

        # Position label to avoid overlap
        label_x = px + 12
        label_y = py - 8
        if i in (KP_FAR_RIGHT, KP_NEAR_RIGHT, KP_NET_RIGHT):
            # Right-side points: put label to the left
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_x = px - text_size[0] - 12

        # Background rectangle for readability
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(vis, (label_x - 2, label_y - text_size[1] - 4),
                      (label_x + text_size[0] + 2, label_y + 4), (0, 0, 0), -1)
        cv2.putText(vis, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 6. Info panel (top-left)
    info_lines = [
        f"Confidence: {confidence:.3f}",
        f"Frames used: {num_detections}",
    ]
    if court_config:
        info_lines.append(f"net_y: {court_config.get('net_y', 'N/A')}")
        n_zones = len(court_config.get('exclusion_zones', []))
        info_lines.append(f"Exclusion zones: {n_zones}")

    panel_h = 25 * len(info_lines) + 10
    cv2.rectangle(vis, (0, 0), (280, panel_h), (0, 0, 0), -1)
    for i, line in enumerate(info_lines):
        cv2.putText(vis, line, (10, 22 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    return vis


def process_video(detector, video_path, output_dir, sample_frames=5,
                  export_config=False, margin_lr=0.15, margin_far=0.30,
                  margin_near=0.25):
    """
    Process a single video: detect court, save annotated frames.

    Args:
        detector: AutoCourtDetector instance
        video_path: Path to video
        output_dir: Output directory for images
        sample_frames: Number of frames to sample
        export_config: Whether to also save court_config JSON
        margin_lr/far/near: Exclusion zone margin ratios
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n  Processing: {basename}")

    # Multi-frame detection
    detection = detector.detect_from_video(video_path, sample_frames=sample_frames)

    if detection.get('keypoints') is None:
        print(f"    [FAIL] No court detected")
        return

    conf = detection['confidence']
    n_det = detection['num_detections']
    print(f"    Confidence: {conf:.3f} ({n_det}/{sample_frames} frames)")

    # Generate court config
    court_config = detector.generate_court_config(
        detection,
        margin_lr=margin_lr,
        margin_far=margin_far,
        margin_near=margin_near,
    )

    # Read frames and draw
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample positions for output
    positions = [0.2, 0.4, 0.6]
    saved = 0

    for pos in positions:
        frame_idx = int(total_frames * pos)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        vis = draw_detection(
            frame,
            detection['keypoints'],
            per_kp_conf=detection.get('per_kp_confidence'),
            confidence=conf,
            num_detections=n_det,
            court_config=court_config,
        )

        out_path = os.path.join(output_dir, f"{basename}_f{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, vis)
        saved += 1

    cap.release()
    print(f"    [OK] Saved {saved} annotated frames to {output_dir}")

    # Export config if requested
    if export_config and court_config:
        config_path = os.path.join(output_dir, f"{basename}_court_config.json")
        save_court_config(court_config, config_path)
        print(f"    [OK] Config saved: {config_path}")


def process_image(detector, image_path, output_dir, export_config=False,
                  margin_lr=0.15, margin_far=0.30, margin_near=0.25):
    """Process a single image."""
    basename = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n  Processing: {basename}")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"    [FAIL] Cannot read image")
        return

    h, w = frame.shape[:2]
    det = detector.detect_single_frame(frame)

    if det is None:
        print(f"    [FAIL] No court detected")
        return

    conf = det['confidence']
    print(f"    Confidence: {conf:.3f}")

    # Build detection dict compatible with generate_court_config
    detection = {
        'keypoints': det['keypoints'][:, :2],
        'confidence': conf,
        'per_kp_confidence': det['keypoints'][:, 2],
        'num_detections': 1,
        'frame_w': w,
        'frame_h': h,
    }

    court_config = detector.generate_court_config(
        detection, frame_w=w, frame_h=h,
        margin_lr=margin_lr, margin_far=margin_far, margin_near=margin_near,
    )

    vis = draw_detection(
        frame, detection['keypoints'],
        per_kp_conf=detection.get('per_kp_confidence'),
        confidence=conf, num_detections=1,
        court_config=court_config,
    )

    out_path = os.path.join(output_dir, f"{basename}_detected.jpg")
    cv2.imwrite(out_path, vis)
    print(f"    [OK] Saved: {out_path}")

    if export_config and court_config:
        config_path = os.path.join(output_dir, f"{basename}_court_config.json")
        save_court_config(court_config, config_path)
        print(f"    [OK] Config saved: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Court keypoint detection - inference + visualization"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input video, image, or directory")
    parser.add_argument("--model", type=str, default="models/court_best.pt",
                        help="Path to trained model (default: models/court_best.pt)")
    parser.add_argument("--output", type=str, default="predict_output",
                        help="Output directory for annotated images")
    parser.add_argument("--sample-frames", type=int, default=5,
                        help="Frames to sample per video (default: 5)")
    parser.add_argument("--export-config", action="store_true",
                        help="Also export court_config JSON")
    parser.add_argument("--margin-lr", type=float, default=0.15,
                        help="Left/right exclusion margin ratio")
    parser.add_argument("--margin-far", type=float, default=0.30,
                        help="Far-side exclusion margin ratio")
    parser.add_argument("--margin-near", type=float, default=0.25,
                        help="Near-side exclusion margin ratio")

    args = parser.parse_args()

    # Load model
    print("=" * 60)
    print("Court Keypoint Detection")
    print("=" * 60)

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print("  Train with: python tools/train_court_detector.py")
        sys.exit(1)

    detector = AutoCourtDetector(args.model)
    print(f"Model loaded: {args.model}")

    os.makedirs(args.output, exist_ok=True)

    video_exts = ('.mp4', '.avi', '.mkv', '.mov')
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')

    # Collect input files
    input_files = []
    if os.path.isdir(args.input):
        for f in sorted(os.listdir(args.input)):
            ext = os.path.splitext(f)[1].lower()
            if ext in video_exts or ext in image_exts:
                input_files.append(os.path.join(args.input, f))
    elif os.path.isfile(args.input):
        input_files.append(args.input)
    else:
        print(f"[ERROR] Input not found: {args.input}")
        sys.exit(1)

    print(f"Input files: {len(input_files)}")
    print(f"Output dir:  {args.output}")

    # Process each file
    for fpath in input_files:
        ext = os.path.splitext(fpath)[1].lower()
        if ext in video_exts:
            process_video(detector, fpath, args.output,
                          sample_frames=args.sample_frames,
                          export_config=args.export_config,
                          margin_lr=args.margin_lr,
                          margin_far=args.margin_far,
                          margin_near=args.margin_near)
        elif ext in image_exts:
            process_image(detector, fpath, args.output,
                          export_config=args.export_config,
                          margin_lr=args.margin_lr,
                          margin_far=args.margin_far,
                          margin_near=args.margin_near)

    print()
    print("=" * 60)
    print(f"Done! Results saved to: {os.path.abspath(args.output)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
