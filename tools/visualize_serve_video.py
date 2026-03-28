# tools/visualize_serve_video.py
# -*- coding: utf-8 -*-
"""
產生標注發球分析的輸出影片

從 batch_test_summary.json 讀取發球事件，
對每個片段產生標注影片（含球、球員、發球階段、接球等標示）
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.data_validator import safe_load_json, validate_center_point
from core.server_identifier import get_keypoint


# ── 顏色常數 ──────────────────────────────────────────────
COLOR_BALL      = (0, 255, 255)   # 黃色
COLOR_SERVER    = (0, 255, 0)     # 綠色
COLOR_PLAYER    = (255, 100, 0)   # 藍橘色
COLOR_WRIST     = (0, 0, 255)     # 紅色
COLOR_ANKLE     = (0, 165, 255)   # 橘色
COLOR_NET       = (255, 255, 0)   # 青黃
COLOR_EXCLUSION = (255, 0, 255)   # 紫色
COLOR_TOSS      = (0, 220, 255)   # 淺黃
COLOR_HIT       = (0, 80, 255)    # 橘紅
COLOR_RECEPTION = (0, 255, 120)   # 淺綠


def load_tracking_json_frames(json_path: str) -> dict:
    """載入 tracking JSON，回傳以 frame_id 為鍵的字典"""
    data, err = safe_load_json(json_path)
    if data is None:
        raise RuntimeError(f"無法載入 JSON: {err}")
    frames = data.get('frames', [])
    return {fr['frame_id']: fr for fr in frames if 'frame_id' in fr}


def find_server_center(frames_by_id: dict, found_frame_id: int, server_index: int) -> list | None:
    """從 FOUND 幀取得發球員中心點"""
    if found_frame_id is None:
        return None
    fr = frames_by_id.get(found_frame_id)
    if not fr:
        return None
    players = fr.get('player_detections', [])
    if 0 <= server_index < len(players):
        return validate_center_point(players[server_index].get('center_point'))
    return None


def match_server_in_frame(players: list, server_center: list, tolerance: float = 150.0) -> int | None:
    """在當前幀中找出最接近 server_center 的球員 index"""
    if not server_center:
        return None
    best_idx, best_dist = None, float('inf')
    for i, p in enumerate(players):
        c = validate_center_point(p.get('center_point'))
        if not c:
            continue
        d = ((c[0] - server_center[0])**2 + (c[1] - server_center[1])**2) ** 0.5
        if d < best_dist and d < tolerance:
            best_idx, best_dist = i, d
    return best_idx


def get_phase_label(frame_id: int, toss_frame: int, hit_frame: int, reception_frame: int | None) -> tuple[str, tuple]:
    """依幀號回傳階段標籤與顏色"""
    if toss_frame is not None and frame_id < toss_frame:
        return "PRE-TOSS", (80, 80, 80)
    if toss_frame is not None and hit_frame is not None and toss_frame <= frame_id <= hit_frame:
        if frame_id == toss_frame:
            return "TOSS", COLOR_TOSS
        if frame_id == hit_frame:
            return "HIT", COLOR_HIT
        return "IN-FLIGHT", (200, 200, 0)
    if hit_frame is not None and reception_frame is not None and hit_frame < frame_id <= reception_frame:
        if frame_id == reception_frame:
            return "RECEPTION", COLOR_RECEPTION
        return "AFTER-HIT", (0, 200, 100)
    return "POST-SERVE", (60, 60, 60)


def draw_frame(frame: np.ndarray, frame_data: dict | None, frame_id: int,
               serve_result: dict, server_center: list | None,
               court_config: dict | None) -> np.ndarray:
    """在單幀上繪製所有標注"""
    h, w = frame.shape[:2]

    toss_frame     = serve_result.get('toss_frame')
    hit_frame      = serve_result.get('hit_frame')
    reception_frame = serve_result.get('reception_frame')
    jump_result_type = serve_result.get('serve_type', 'unknown')
    jump_height    = serve_result.get('jump_height', 0)

    # ── 排除區域（紫色邊框）──────────────────────────
    if court_config:
        for zone in court_config.get('exclusion_zones', []):
            poly = zone.get('polygon') if isinstance(zone, dict) else zone
            if poly:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(frame, [pts], True, COLOR_EXCLUSION, 2)

    # ── 網子位置（水平虛線）──────────────────────────
    if court_config and court_config.get('net_y'):
        ny = int(court_config['net_y'])
        for x in range(0, w, 30):
            cv2.line(frame, (x, ny), (min(x + 15, w), ny), COLOR_NET, 2)
        cv2.putText(frame, "NET", (10, ny - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NET, 1)

    # ── 球偵測 ────────────────────────────────────────
    if frame_data:
        balls = frame_data.get('ball_detections', [])
        for bd in balls:
            bp = validate_center_point(bd.get('center_point'))
            if bp:
                cv2.circle(frame, (int(bp[0]), int(bp[1])), 14, COLOR_BALL, 3)
                conf = bd.get('confidence', 0)
                cv2.putText(frame, f"{conf:.2f}", (int(bp[0]) + 15, int(bp[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BALL, 1)

    # ── 球員偵測 ──────────────────────────────────────
    if frame_data:
        players = frame_data.get('player_detections', [])
        srv_idx = match_server_in_frame(players, server_center)

        for i, p in enumerate(players):
            box = p.get('box_coords')
            if not box or len(box) < 4:
                continue

            is_server = (i == srv_idx)
            color = COLOR_SERVER if is_server else COLOR_PLAYER
            label = "SERVER" if is_server else f"P{i}"

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # 手腕關鍵點
            for kp_idx in [9, 10]:
                kp = get_keypoint(p, kp_idx)
                if kp:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 6, COLOR_WRIST, -1)

            # 腳踝關鍵點（只顯示發球員）
            if is_server:
                for kp_idx in [15, 16]:
                    kp = get_keypoint(p, kp_idx)
                    if kp:
                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 6, COLOR_ANKLE, -1)

    # ── 關鍵幀垂直彩條 ──────────────────────────────
    overlay = frame.copy()
    if frame_id == toss_frame:
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 220, 255), -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    elif frame_id == hit_frame:
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 80, 255), -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    elif frame_id == reception_frame:
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 120), -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    # ── 發球類型標籤（右上角）────────────────────────
    if jump_result_type == 'jump':
        type_label = f"JUMP SERVE  h={jump_height:.0f}px"
        type_bg    = (0, 200, 200)
        type_fg    = (0, 0, 0)
    else:
        type_label = "STANDING SERVE"
        type_bg    = (80, 80, 80)
        type_fg    = (255, 255, 255)

    (tw, th), base = cv2.getTextSize(type_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    tx = w - tw - 20
    ty = 35
    cv2.rectangle(frame, (tx - 6, ty - th - 6), (tx + tw + 6, ty + base + 6), type_bg, -1)
    cv2.putText(frame, type_label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, type_fg, 2)

    # ── 階段標籤（左上角）────────────────────────────
    phase_str, phase_color = get_phase_label(frame_id, toss_frame, hit_frame, reception_frame)
    (pw, ph), pb = cv2.getTextSize(phase_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    cv2.rectangle(frame, (8, 8), (pw + 20, ph + 20), (0, 0, 0), -1)
    cv2.putText(frame, phase_str, (14, ph + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, phase_color, 3)

    # ── 幀資訊（左下角）──────────────────────────────
    info_lines = [
        f"Frame: {frame_id}",
        f"Toss:{toss_frame}  Hit:{hit_frame}  Rec:{reception_frame}",
        f"Conf:{serve_result.get('confidence', 0):.2f}  "
        f"Zone:{serve_result.get('serve_zone')}  "
        f"Side:{serve_result.get('serving_side')}",
    ]
    for j, line in enumerate(info_lines):
        y_pos = h - 20 - (len(info_lines) - 1 - j) * 26
        cv2.putText(frame, line, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

    return frame


def visualize_segment(video_path: str, json_path: str, serve_result: dict,
                      output_path: str, court_config: dict | None,
                      context_frames: int = 60) -> bool:
    """
    對單個片段產生標注影片

    Args:
        context_frames: 發球前後各保留的幀數
    """
    toss_frame     = serve_result.get('toss_frame') or 0
    hit_frame      = serve_result.get('hit_frame') or toss_frame
    reception_frame = serve_result.get('reception_frame')
    found_frame    = serve_result.get('found_frame') or toss_frame
    server_index   = serve_result.get('server_index') or 0
    is_no_serve    = serve_result.get('status') != 'success'

    if is_no_serve:
        # no_serve 片段：輸出全段影片
        start_frame = 0
        end_frame   = 999999  # 會被 total-1 截斷
    else:
        end_event  = reception_frame if reception_frame else hit_frame
        start_frame = max(0, toss_frame - context_frames)
        end_frame   = end_event + context_frames if end_event else hit_frame + context_frames

    print(f"  載入追蹤資料...")
    frames_by_id = load_tracking_json_frames(json_path)
    server_center = find_server_center(frames_by_id, found_frame, server_index)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] 無法開啟影片: {video_path}")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(end_frame, total - 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current = start_frame
    written = 0

    print(f"  輸出影格範圍: {start_frame} ~ {end_frame}  ({end_frame - start_frame + 1} 幀)")

    while current <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_data = frames_by_id.get(current)
        frame = draw_frame(frame, frame_data, current, serve_result,
                           server_center, court_config)
        out.write(frame)
        current += 1
        written += 1

    cap.release()
    out.release()
    print(f"  [OK] 已儲存: {output_path}  ({written} 幀)")
    return True


def main():
    parser = argparse.ArgumentParser(description='產生發球分析標注影片')
    parser.add_argument('--summary',     required=True, help='batch_test_summary.json 路徑')
    parser.add_argument('--video-dir',   required=True, help='原始影片目錄')
    parser.add_argument('--json-dir',    required=True, help='追蹤 JSON 目錄')
    parser.add_argument('--output-dir',  default='output/serve_videos', help='輸出影片目錄')
    parser.add_argument('--court-config', default=None, help='court_config.json 路徑（選用）')
    parser.add_argument('--context',     type=int, default=60, help='發球前後各保留的幀數（預設 60）')
    parser.add_argument('--segments',    nargs='*', help='指定片段名稱（不指定則處理全部）')
    parser.add_argument('--all',         action='store_true', help='包含 no_serve 片段（輸出完整影片）')
    args = parser.parse_args()

    # 載入 summary
    with open(args.summary, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    court_config = None
    if args.court_config and os.path.exists(args.court_config):
        with open(args.court_config, 'r', encoding='utf-8') as f:
            court_config = json.load(f)
        print(f"場地設定已載入: net_y={court_config.get('net_y')}")

    results = summary.get('results', [])
    if args.segments:
        results = [r for r in results if r.get('video_name') in args.segments]

    print(f"\n處理 {len(results)} 個片段...\n")

    success, fail = 0, 0
    for res in results:
        name   = res.get('video_name')
        status = res.get('status')

        if status != 'success':
            if not args.all:
                print(f"  [SKIP] {name}: status={status}")
                continue
            print(f"  [NO-SERVE] {name}: 輸出完整追蹤影片")

        video_path = os.path.join(args.video_dir, name + '.mp4')
        json_path  = os.path.join(args.json_dir,
                                  name + '_all_frames_data_with_pose.json')

        if not os.path.exists(video_path):
            print(f"  [SKIP] {name}: 找不到影片 {video_path}")
            fail += 1
            continue
        if not os.path.exists(json_path):
            print(f"  [SKIP] {name}: 找不到 JSON {json_path}")
            fail += 1
            continue

        output_path = os.path.join(args.output_dir, name + '_serve_annotated.mp4')
        print(f"[{name}]")

        ok = visualize_segment(video_path, json_path, res, output_path,
                               court_config, args.context)
        if ok:
            success += 1
        else:
            fail += 1

    print(f"\n完成: {success} 成功 / {fail} 失敗")
    print(f"輸出目錄: {args.output_dir}")


if __name__ == '__main__':
    main()
