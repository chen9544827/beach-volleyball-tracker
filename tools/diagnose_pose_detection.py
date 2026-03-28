#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pose Detection Failure Diagnostic Tool

Diagnoses WHY pose detection fails on far-side players:
  - Distance problem (players too small) -> recommend SAHI
  - Net occlusion problem (players behind net) -> recommend better augmentation

Uses a very low confidence threshold (default 0.05) to capture all potential
detections including ones normally filtered out.

Usage:
    python tools/diagnose_pose_detection.py \
        --video output_data/test_segment/segment_001_Team1.mp4 \
        --court-config court_configs/Edmonton_WT19_C4.json \
        --model models/yolov8m-pose.pt \
        --conf 0.05 \
        --frames 300 \
        --output output/diagnose_baseline
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from core.sahi_pose_detector import SahiPoseDetector


# ---------------------------------------------------------------------------
# Court config helpers
# ---------------------------------------------------------------------------

def load_court_config(path):
    """Load court_config JSON and extract key Y coordinates."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    polygon = cfg.get('court_boundary_polygon', [])
    net_y = cfg.get('net_y', None)

    far_y = None
    near_y = None

    if len(polygon) >= 4:
        pts = sorted(polygon, key=lambda p: p[1])  # sort by Y
        # Top 2 = far side (small Y) -> far baseline = their max Y
        far_y = max(pts[0][1], pts[1][1])
        # Bottom 2 = near side (large Y) -> near baseline = their min Y
        near_y = min(pts[2][1], pts[3][1])

    return cfg, net_y, far_y, near_y


def classify_zone(center_y, net_y, far_y, near_y):
    """Classify detection into court zone by center_y."""
    margin_far = 80
    margin_near = 100
    net_margin = 80

    if center_y < far_y + margin_far:
        # Could also overlap with net_area if net_y is close
        if abs(center_y - net_y) < net_margin:
            return 'net_area'
        return 'far_side'
    elif abs(center_y - net_y) < net_margin:
        return 'net_area'
    elif center_y > near_y - margin_near:
        return 'near_side'
    else:
        return 'mid_court'


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def bbox_color(conf):
    """Return BGR color based on confidence level."""
    if conf >= 0.3:
        return (0, 255, 0)    # green - normal threshold
    elif conf >= 0.1:
        return (0, 255, 255)  # yellow - low confidence
    else:
        return (0, 0, 255)    # red - very low confidence


def draw_dashed_hline(img, y, color, thickness=1, dash_len=15):
    """Draw a horizontal dashed line."""
    h, w = img.shape[:2]
    x = 0
    while x < w:
        x_end = min(x + dash_len, w)
        cv2.line(img, (x, y), (x_end, y), color, thickness)
        x += dash_len * 2


def normalize_yolo_results(results):
    """Convert YOLO results object to list of {conf, x1, y1, x2, y2} dicts."""
    boxes_list = []
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None:
            for i in range(len(r.boxes)):
                conf = float(r.boxes.conf[i])
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy().astype(int)
                boxes_list.append({'conf': conf, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    return boxes_list


def normalize_sahi_results(sahi_dets):
    """Convert SahiPoseDetector output to list of {conf, x1, y1, x2, y2} dicts."""
    boxes_list = []
    for det in sahi_dets:
        x1, y1, x2, y2 = [int(v) for v in det['box_coords']]
        boxes_list.append({'conf': det['confidence'], 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    return boxes_list


def draw_frame(frame, boxes_list, net_y, far_y, near_y, court_config, zone_counts):
    """
    Draw annotated frame with color-coded bounding boxes.

    boxes_list: normalized list of {conf, x1, y1, x2, y2} dicts.
    zone_counts: dict to accumulate per-zone detection counts for this frame.
    Returns annotated frame and list of detection dicts for this frame.
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    # --- Draw court boundary polygon ---
    polygon = court_config.get('court_boundary_polygon', [])
    if polygon:
        pts = np.array(polygon, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 200, 0), 2)

    # --- Draw net_y dashed line ---
    if net_y is not None:
        draw_dashed_hline(vis, int(net_y), (0, 255, 255), thickness=2)
        cv2.putText(vis, f'net_y={net_y}', (5, int(net_y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # --- Draw far/near baseline guides (faint) ---
    if far_y is not None:
        cv2.line(vis, (0, int(far_y)), (w, int(far_y)), (100, 180, 100), 1)
        cv2.putText(vis, f'far_y={int(far_y)}', (5, int(far_y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 180, 100), 1)
    if near_y is not None:
        cv2.line(vis, (0, int(near_y)), (w, int(near_y)), (100, 180, 100), 1)
        cv2.putText(vis, f'near_y={int(near_y)}', (5, int(near_y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 180, 100), 1)

    frame_detections = []

    for b in boxes_list:
        conf = b['conf']
        x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = (x2 - x1) * (y2 - y1)

        zone = classify_zone(cy, net_y, far_y, near_y)

        color = bbox_color(conf)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f'{conf:.2f}', (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        zone_counts[zone] = zone_counts.get(zone, 0) + 1

        frame_detections.append({
            'conf': conf,
            'area': area,
            'cx': cx,
            'cy': cy,
            'zone': zone,
        })

    # --- Zone count overlay (top-left) ---
    overlay_lines = [
        f"far:  {zone_counts.get('far_side', 0)}",
        f"net:  {zone_counts.get('net_area', 0)}",
        f"mid:  {zone_counts.get('mid_court', 0)}",
        f"near: {zone_counts.get('near_side', 0)}",
    ]
    bg_h = len(overlay_lines) * 20 + 8
    cv2.rectangle(vis, (0, 0), (120, bg_h), (0, 0, 0), -1)
    for li, txt in enumerate(overlay_lines):
        cv2.putText(vis, txt, (5, 18 + li * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Legend (bottom-left) ---
    legend = [
        ('green  conf>=0.3', (0, 255, 0)),
        ('yellow 0.1-0.3',   (0, 255, 255)),
        ('red    <0.1',      (0, 0, 255)),
    ]
    leg_y = h - len(legend) * 20 - 5
    cv2.rectangle(vis, (0, leg_y - 4), (170, h), (0, 0, 0), -1)
    for li, (txt, col) in enumerate(legend):
        cv2.putText(vis, txt, (5, leg_y + li * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    return vis, frame_detections


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

ZONES = ['far_side', 'net_area', 'mid_court', 'near_side']


def compute_stats(all_detections, total_frames):
    """
    Compute per-zone statistics from all detections across frames.

    all_detections: list of detection dicts (each has conf, area, zone)
    Returns dict keyed by zone.
    """
    from collections import defaultdict

    zone_confs = defaultdict(list)
    zone_areas = defaultdict(list)
    zone_high  = defaultdict(int)   # conf >= 0.3
    zone_total = defaultdict(int)

    for det in all_detections:
        z = det['zone']
        zone_confs[z].append(det['conf'])
        zone_areas[z].append(det['area'])
        zone_total[z] += 1
        if det['conf'] >= 0.3:
            zone_high[z] += 1

    stats = {}
    for z in ZONES:
        confs = zone_confs[z]
        areas = zone_areas[z]
        total = zone_total[z]
        high  = zone_high[z]

        stats[z] = {
            'total_detections': total,
            'avg_per_frame': total / max(total_frames, 1),
            'avg_conf': float(np.mean(confs)) if confs else 0.0,
            'avg_area': float(np.mean(areas)) if areas else 0.0,
            'high_conf_ratio': high / total if total > 0 else 0.0,
        }

    return stats


def generate_conclusions(stats):
    """Generate diagnostic conclusions from per-zone stats."""
    lines = []
    far  = stats.get('far_side',  {})
    net  = stats.get('net_area',  {})
    near = stats.get('near_side', {})

    far_conf  = far.get('avg_conf', 0)
    near_conf = near.get('avg_conf', 0)
    far_area  = far.get('avg_area', 0)
    near_area = near.get('avg_area', 0)
    net_high  = net.get('high_conf_ratio', 0)
    near_high = near.get('high_conf_ratio', 0)

    distance_issue = False
    occlusion_issue = False

    if near_conf > 0 and far_conf < near_conf * 0.5:
        lines.append(
            f'-> 遠端平均信心度 ({far_conf:.2f}) 遠低於近端 ({near_conf:.2f})：'
            '[距離/尺寸] 問題可能性高'
        )
        distance_issue = True

    if near_area > 0 and far_area < near_area * 0.5:
        lines.append(
            f'-> 遠端平均 bbox 面積 ({far_area:,.0f}px^2) << 近端 ({near_area:,.0f}px^2)：'
            '遠端球員確實較小 -> 建議 SAHI'
        )
        distance_issue = True

    if near_high > 0 and net_high < near_high * 0.5:
        lines.append(
            f'-> 網子附近高信心比例 ({net_high:.0%}) 顯著低於近端 ({near_high:.0%})：'
            '[遮擋] 問題可能性高 -> 建議增強資料'
        )
        occlusion_issue = True

    if not lines:
        lines.append('-> 各區域信心度差異不顯著，需人工檢視標註影片')

    lines.append('')
    if distance_issue and occlusion_issue:
        lines.append('=> 建議：同時啟用 SAHI 並改進網遮擋增強資料')
    elif distance_issue:
        lines.append('=> 建議：啟用 SAHI（core/sahi_pose_detector.py 已實作）')
    elif occlusion_issue:
        lines.append('=> 建議：改進 fine-tune 增強策略（更真實網格 pattern、更多遮擋資料）')

    return lines


def format_stats_table(stats, total_frames):
    """Format stats as a text table."""
    header = f"{'區域':<12}| {'平均偵測/幀':>10} | {'平均信心度':>10} | {'平均bbox面積(px^2)':>18} | {'高信心比例(>=0.3)':>16}"
    sep = '-' * len(header)
    rows = [header, sep]

    zone_labels = {
        'far_side':  'far_side  ',
        'net_area':  'net_area  ',
        'mid_court': 'mid_court ',
        'near_side': 'near_side ',
    }

    for z in ZONES:
        s = stats.get(z, {})
        avg_f  = s.get('avg_per_frame', 0)
        avg_c  = s.get('avg_conf', 0)
        avg_a  = s.get('avg_area', 0)
        high_r = s.get('high_conf_ratio', 0)
        label  = zone_labels.get(z, z)
        rows.append(
            f'{label:<12}| {avg_f:>10.2f} | {avg_c:>10.2f} | {avg_a:>18,.0f} | {high_r:>16.0%}'
        )

    return '\n'.join(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_diagnosis(video_path, court_config_path, model_path, conf_thresh,
                  max_frames, output_dir, imgsz=1280, iou_thresh=0.7,
                  use_sahi=False, sahi_slice=512, sahi_overlap=0.2):

    os.makedirs(output_dir, exist_ok=True)

    # Load court config
    court_cfg, net_y, far_y, near_y = load_court_config(court_config_path)
    print(f'[INFO] court_config loaded: net_y={net_y}, far_y={far_y:.1f}, near_y={near_y:.1f}')

    # Load model
    print(f'[INFO] Loading model: {model_path}')
    model = YOLO(model_path)

    # Setup SAHI wrapper if requested
    sahi_detector = None
    if use_sahi:
        sahi_detector = SahiPoseDetector(
            model,
            slice_size=sahi_slice,
            overlap_ratio=sahi_overlap,
            conf_thresh=conf_thresh,
            iou_thresh=0.5,
        )
        print(f'[INFO] SAHI enabled: slice_size={sahi_slice}, overlap={sahi_overlap}')
    else:
        print(f'[INFO] Mode: standard inference, imgsz={imgsz}, iou={iou_thresh}')

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open video: {video_path}')
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(max_frames, total)

    print(f'[INFO] Video: {width}x{height} @ {fps:.1f}fps, processing {n_frames}/{total} frames')

    # Setup video writer
    video_out_path = os.path.join(output_dir, 'diagnose_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

    all_detections = []
    frame_idx = 0

    while frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if sahi_detector is not None:
            sahi_dets = sahi_detector.detect(frame, conf_thresh=conf_thresh)
            boxes_list = normalize_sahi_results(sahi_dets)
        else:
            results = model(frame, conf=conf_thresh, imgsz=imgsz,
                            iou=iou_thresh, verbose=False)
            boxes_list = normalize_yolo_results(results)

        zone_counts = {}
        vis, frame_dets = draw_frame(frame, boxes_list, net_y, far_y, near_y,
                                     court_cfg, zone_counts)

        all_detections.extend(frame_dets)
        writer.write(vis)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f'  [{frame_idx}/{n_frames}] detections so far: {len(all_detections)}')

    cap.release()
    writer.release()
    print(f'[INFO] Annotated video saved: {video_out_path}')

    # Compute stats
    stats = compute_stats(all_detections, n_frames)
    conclusions = generate_conclusions(stats)
    table = format_stats_table(stats, n_frames)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    mode_str = f'SAHI (slice={sahi_slice}, overlap={sahi_overlap})' if use_sahi else f'standard (imgsz={imgsz}, iou={iou_thresh})'

    report_lines = [
        f'=== Pose Detection Diagnostic Report ===',
        f'Model     : {model_name}',
        f'Mode      : {mode_str}',
        f'Video     : {os.path.basename(video_path)}',
        f'Court cfg : {os.path.basename(court_config_path)}',
        f'Conf thr  : {conf_thresh}',
        f'Frames    : {n_frames}',
        f'net_y={net_y}, far_y={far_y:.1f}, near_y={near_y:.1f}',
        f'Total detections: {len(all_detections)}',
        '',
        f'=== 區域偵測統計（共 {n_frames} 幀）===',
        table,
        '',
        '=== 自動診斷結論 ===',
    ] + conclusions

    report_text = '\n'.join(report_lines)

    # Print to console
    print()
    print(report_text)

    # Save to file
    stats_path = os.path.join(output_dir, 'diagnose_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(report_text + '\n')
    print(f'\n[INFO] Stats saved: {stats_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Diagnose pose detection failures (distance vs net occlusion)'
    )
    p.add_argument('--video', required=True, help='Input video path')
    p.add_argument('--court-config', required=True, help='court_config JSON path')
    p.add_argument('--model', default='models/yolov8m-pose.pt',
                   help='Pose model path (default: models/yolov8m-pose.pt)')
    p.add_argument('--conf', type=float, default=0.05,
                   help='Detection confidence threshold (default: 0.05)')
    p.add_argument('--frames', type=int, default=300,
                   help='Number of frames to process (default: 300)')
    p.add_argument('--output', default='output/diagnose',
                   help='Output directory (default: output/diagnose)')
    p.add_argument('--imgsz', type=int, default=1280,
                   help='Inference image size for standard mode (default: 1280)')
    p.add_argument('--iou', type=float, default=0.7,
                   help='NMS IoU threshold (default: 0.7, lower = more aggressive merging)')
    p.add_argument('--sahi', action='store_true',
                   help='Use SAHI sliced inference instead of standard inference')
    p.add_argument('--sahi-slice', type=int, default=512,
                   help='SAHI slice size in pixels (default: 512)')
    p.add_argument('--sahi-overlap', type=float, default=0.2,
                   help='SAHI overlap ratio between slices (default: 0.2)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_diagnosis(
        video_path=args.video,
        court_config_path=args.court_config,
        model_path=args.model,
        conf_thresh=args.conf,
        max_frames=args.frames,
        output_dir=args.output,
        imgsz=args.imgsz,
        iou_thresh=args.iou,
        use_sahi=args.sahi,
        sahi_slice=args.sahi_slice,
        sahi_overlap=args.sahi_overlap,
    )
