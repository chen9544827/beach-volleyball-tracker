# tools/diagnose_serve_frames.py
# -*- coding: utf-8 -*-
"""
Serve detection diagnostic tool.

For each detected serve event, outputs annotated frames:
- 3 frames before hit
- Hit frame
- 3 frames after hit
- Reception frame (if detected)

Annotations include:
- Ball position (yellow circle)
- Player bounding boxes (green=server, blue=others)
- Court boundary (green polygon)
- Net line (yellow dashed)
- Exclusion zones (purple semi-transparent)
- Serve zone / Reception zone labels
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.serve_detector import analyze_serve_events_v2
from core.server_identifier import analyze_serve_player, determine_serving_side
from core.court_zones import CourtZones
from core.reception_detector import ReceptionDetector
from core.jump_serve_detector import classify_serve_type
from core.data_validator import safe_load_json


def draw_frame_annotations(frame, frame_data, court_config, server_idx=None,
                           ball_highlight=True, label="", reception_pos=None):
    """Draw annotations on a single frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Draw court boundary
    if court_config and court_config.get('court_boundary_polygon'):
        pts = np.array(court_config['court_boundary_polygon'], dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

    # Draw net_y
    if court_config and court_config.get('net_y'):
        net_y = court_config['net_y']
        cv2.line(vis, (0, net_y), (w, net_y), (0, 255, 255), 1, cv2.LINE_AA)

    # Draw exclusion zones
    if court_config and court_config.get('exclusion_zones'):
        overlay = vis.copy()
        for zone in court_config['exclusion_zones']:
            poly = np.array(zone['polygon'], dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (180, 0, 180))
        cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)

    if not frame_data:
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return vis

    # Draw players
    players = frame_data.get('player_detections', [])
    for idx, player in enumerate(players):
        box = player.get('box_coords')
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        if idx == server_idx:
            color = (0, 255, 0)  # Green = server
            thickness = 3
        else:
            color = (255, 150, 0)  # Blue = others
            thickness = 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(vis, f"P{idx}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw ball
    balls = frame_data.get('ball_detections', [])
    if balls:
        best_ball = max(balls, key=lambda b: b.get('confidence', 0))
        cp = best_ball.get('center_point')
        if cp and len(cp) >= 2:
            bx, by = int(cp[0]), int(cp[1])
            radius = 12 if ball_highlight else 8
            color = (0, 255, 255) if ball_highlight else (0, 200, 200)
            cv2.circle(vis, (bx, by), radius, color, 3)
            cv2.circle(vis, (bx, by), 3, color, -1)
            conf = best_ball.get('confidence', 0)
            cv2.putText(vis, f"ball({conf:.2f})", (bx + 15, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw reception position
    if reception_pos:
        rx, ry = int(reception_pos[0]), int(reception_pos[1])
        cv2.circle(vis, (rx, ry), 15, (0, 0, 255), 3)
        cv2.putText(vis, "RECEPTION", (rx + 20, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Label
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return vis


def process_segment(video_path, json_path, court_config, output_dir, segment_name):
    """Process one segment: detect serve and output diagnostic frames."""
    # Load tracking data
    data, err = safe_load_json(json_path)
    if err or not data or 'frames' not in data:
        print(f"  [SKIP] Cannot load: {json_path}")
        return

    frames_data = data['frames']
    metadata = data.get('metadata', {})
    total_frames = len(frames_data)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Detect serve
    events = analyze_serve_events_v2(frames_data, metadata, first_only=True)
    if not events:
        print(f"  [SKIP] No serve detected")
        cap.release()
        return

    event = events[0]
    hit_frame = event.get('hit_frame_id', 0)
    toss_frame = event.get('toss_start_frame', 0)
    apex_frame = event.get('apex_frame', 0)

    # Identify server
    net_y = court_config.get('net_y', 0)
    exclusion_zones = court_config.get('exclusion_zones', [])
    server_info = analyze_serve_player(
        frames_data, event,
        method='lookback',
        exclusion_zones=exclusion_zones
    )
    server_idx = server_info.get('final_server_index')

    # Determine serving side from ball position at hit frame
    hit_frame_data = frames_data[hit_frame] if hit_frame < total_frames else None
    ball_pos_at_hit = None
    if hit_frame_data and hit_frame_data.get('ball_detections'):
        best_ball = max(hit_frame_data['ball_detections'], key=lambda b: b.get('confidence', 0))
        cp = best_ball.get('center_point')
        if cp and len(cp) >= 2:
            ball_pos_at_hit = (cp[0], cp[1])

    if ball_pos_at_hit and net_y > 0:
        serving_side = determine_serving_side(ball_pos_at_hit, metadata.get('height', 720), net_y / metadata.get('height', 720))
    else:
        serving_side = 'near'  # default

    # Get image height for resolution scaling
    image_height = metadata.get('height', 720)

    # Detect reception
    reception_result = {}
    if net_y > 0:
        court_zones = None
        if court_config.get('court_boundary_polygon') and len(court_config['court_boundary_polygon']) >= 4:
            try:
                court_zones = CourtZones(court_config)
            except Exception:
                pass

        detector = ReceptionDetector()
        reception_result = detector.analyze_reception(
            frames_data, event, net_y, serving_side, court_zones,
            image_height=image_height
        )

    reception_frame = reception_result.get('reception_frame')
    reception_pos = reception_result.get('reception_position')
    reception_zone = reception_result.get('reception_zone')

    # Detect jump serve
    jump_result = classify_serve_type(
        frames_data, event, server_info,
        court_config=court_config,
        image_height=image_height
    )

    # Determine frames to output
    # Toss, Apex, Hit-3, Hit-2, Hit-1, Hit, Hit+1, Hit+2, Hit+3, Reception
    output_frames = {}

    if toss_frame and toss_frame > 0:
        output_frames[toss_frame] = f"TOSS (frame {toss_frame})"
    if apex_frame and apex_frame > 0:
        output_frames[apex_frame] = f"APEX (frame {apex_frame})"

    for offset in [-3, -2, -1, 0, 1, 2, 3]:
        f = hit_frame + offset
        if 0 <= f < total_frames:
            if offset < 0:
                tag = f"HIT{offset}"
            elif offset == 0:
                tag = ">>> HIT <<<"
            else:
                tag = f"HIT+{offset}"
            output_frames[f] = f"{tag} (frame {f})"

    if reception_frame and reception_frame > 0:
        output_frames[reception_frame] = f"RECEPTION zone={reception_zone} (frame {reception_frame})"

    # Sort frames
    sorted_frames = sorted(output_frames.items())

    # Create output directory
    seg_dir = os.path.join(output_dir, segment_name)
    os.makedirs(seg_dir, exist_ok=True)

    # Output info text
    info = {
        'segment': segment_name,
        'total_frames': total_frames,
        'toss_frame': toss_frame,
        'apex_frame': apex_frame,
        'hit_frame': hit_frame,
        'hit_speed': event.get('hit_speed', 0),
        'server_index': server_idx,
        'server_confidence': server_info.get('confidence', 0),
        'serving_side': serving_side,
        'is_jump_serve': jump_result.get('is_jump_serve', False),
        'serve_type': jump_result.get('serve_type', 'unknown'),
        'jump_height': jump_result.get('jump_height', 0),
        'jump_confidence': jump_result.get('confidence', 0),
        'reception_detected': reception_result.get('reception_detected', False),
        'reception_frame': reception_frame,
        'reception_zone': reception_zone,
        'reception_position': reception_pos,
        'image_height': image_height,
    }

    with open(os.path.join(seg_dir, 'info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # Read and save annotated frames
    serve_type_str = jump_result.get('serve_type', '?')
    jump_h = jump_result.get('jump_height', 0)
    print(f"  Toss: {toss_frame}, Apex: {apex_frame}, Hit: {hit_frame}, "
          f"Server: P{server_idx} ({serving_side}), "
          f"Type: {serve_type_str} (h={jump_h:.1f}px), "
          f"Reception: {'zone ' + str(reception_zone) if reception_zone else 'N/A'}")

    for frame_idx, label in sorted_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_data = frames_data[frame_idx] if frame_idx < total_frames else None
        is_hit = (frame_idx == hit_frame)
        is_reception = (frame_idx == reception_frame)
        r_pos = reception_pos if is_reception else None

        vis = draw_frame_annotations(
            frame, frame_data, court_config,
            server_idx=server_idx,
            ball_highlight=is_hit,
            label=label,
            reception_pos=r_pos,
        )

        # Add serve info panel at bottom
        panel_h = 40
        panel = np.zeros((panel_h, vis.shape[1], 3), dtype=np.uint8)
        info_text = (f"Server: P{server_idx} | Side: {serving_side} | "
                     f"Hit speed: {event.get('hit_speed', 0):.1f} | "
                     f"Confidence: {server_info.get('confidence', 0):.2f}")
        cv2.putText(panel, info_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        vis = np.vstack([vis, panel])

        filename = f"{frame_idx:06d}_{label.split('(')[0].strip().replace(' ', '_').replace('>', '').replace('<', '')}.jpg"
        cv2.imwrite(os.path.join(seg_dir, filename), vis)

    cap.release()
    print(f"  [OK] Saved {len(sorted_frames)} frames to {seg_dir}")


def main():
    parser = argparse.ArgumentParser(description='Serve detection diagnostic tool')
    parser.add_argument('--video-dir', required=True, help='Video segments directory')
    parser.add_argument('--json-dir', required=True, help='Tracking JSON directory')
    parser.add_argument('--court-config', required=True, help='Court config JSON')
    parser.add_argument('--output', default='serve_diagnose_output', help='Output directory')
    args = parser.parse_args()

    # Load court config
    court_config, err = safe_load_json(args.court_config)
    if err or not court_config:
        print(f"[ERROR] Cannot load court config: {args.court_config} ({err})")
        return

    print("=" * 60)
    print("Serve Detection Diagnostic")
    print("=" * 60)
    print(f"Court config: {args.court_config}")
    print(f"net_y: {court_config.get('net_y')}")
    print()

    # Find matching video-json pairs
    videos = {}
    for f in sorted(os.listdir(args.video_dir)):
        if f.endswith('.mp4'):
            name = os.path.splitext(f)[0]
            videos[name] = os.path.join(args.video_dir, f)

    count = 0
    for name, video_path in videos.items():
        json_path = os.path.join(args.json_dir, f"{name}_all_frames_data_with_pose.json")
        if not os.path.exists(json_path):
            continue

        count += 1
        print(f"[{count}] {name}")
        process_segment(video_path, json_path, court_config, args.output, name)
        print()

    print(f"Done! {count} segments processed -> {args.output}")


if __name__ == '__main__':
    main()
