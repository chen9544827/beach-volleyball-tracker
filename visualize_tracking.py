# visualize_tracking.py
# -*- coding: utf-8 -*-
"""
球追蹤視覺化工具

將追蹤結果標記在影片上，方便檢查追蹤品質
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from collections import deque

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def load_tracking_data(json_path: str) -> tuple:
    """載入追蹤數據

    Returns:
        (frames_dict, metadata) where frames_dict maps frame_id -> frame data
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})

    # 轉換為 frame_id -> data 的字典
    frames_data = data.get('frames', data)
    if isinstance(frames_data, list):
        return {item.get('frame_id', i): item for i, item in enumerate(frames_data)}, metadata
    return frames_data, metadata


def get_ball_position(frame_data: dict) -> tuple:
    """
    從幀數據中取得球位置
    
    Returns:
        (x, y, confidence) 或 None
    """
    if not frame_data:
        return None
    
    ball_detections = frame_data.get('ball_detections', [])
    if not ball_detections:
        return None
    
    # 過濾背景區域的球
    valid_balls = [b for b in ball_detections 
                   if not b.get('is_in_background_zone', False)]
    
    if not valid_balls:
        return None
    
    # 取信心度最高的球
    best_ball = max(valid_balls, key=lambda b: b.get('confidence', 0))
    
    # 取得位置
    box = best_ball.get('box_coords')
    if box:
        x = (box[0] + box[2]) / 2
        y = (box[1] + box[3]) / 2
        conf = best_ball.get('confidence', 0)
        return (x, y, conf)
    
    center = best_ball.get('center_point')
    if center:
        conf = best_ball.get('confidence', 0)
        return (center[0], center[1], conf)
    
    return None


def draw_players(frame: np.ndarray, frame_data: dict) -> np.ndarray:
    """
    繪製球員標記

    Args:
        frame: 影格
        frame_data: 幀數據
    """
    if not frame_data:
        return frame

    players = frame_data.get('player_detections', [])

    for i, player in enumerate(players):
        # 取得球員邊界框
        box = player.get('box_coords')
        if not box or len(box) < 4:
            continue

        x1, y1, x2, y2 = map(int, box)
        conf = player.get('confidence', 0)

        # 繪製邊界框（藍色）
        color = (255, 0, 0)  # 藍色
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 顯示信心度
        label = f"P{i+1}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 如果有姿態關鍵點，繪製
        pose_keypoints = player.get('pose_keypoints')
        if pose_keypoints and len(pose_keypoints) > 0:
            # 繪製關鍵點（簡化版，只顯示高信心度的點）
            for kp in pose_keypoints:
                if len(kp) >= 3 and kp[2] > 0.5:  # 信心度 > 0.5
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # 綠色點

    return frame


def draw_rejected_players(frame: np.ndarray, frame_data: dict) -> np.ndarray:
    """Draw rejected (off-court) player detections as red boxes with X marker."""
    if not frame_data:
        return frame

    rejected = frame_data.get('rejected_detections', [])
    for det in rejected:
        box = det.get('box_coords')
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255)  # red
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Draw X across the box
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        cv2.line(frame, (x2, y1), (x1, y2), color, 2)
        conf = det.get('confidence', 0)
        cv2.putText(frame, f"OUT {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return frame


def draw_auto_court_boundary(frame: np.ndarray, auto_court_info: dict) -> np.ndarray:
    """Draw the auto-estimated court trapezoid boundary and net line."""
    if not auto_court_info:
        return frame

    boundary = auto_court_info.get('boundary')
    if boundary and len(boundary) >= 4:
        pts = np.array(boundary, dtype=np.int32)
        # Semi-transparent green fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 200, 0))
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        # Border
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, "AutoCourt", (pts[0][0] + 5, pts[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    net_y = auto_court_info.get('net_y')
    if net_y is not None:
        h, w = frame.shape[:2]
        net_y_int = int(net_y)
        dash_len = 15
        for x_start in range(0, w, dash_len * 2):
            x_end = min(x_start + dash_len, w)
            cv2.line(frame, (x_start, net_y_int), (x_end, net_y_int),
                     (0, 200, 0), 1)
        cv2.putText(frame, f"AutoNet y={net_y_int}", (10, net_y_int - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

    return frame


def draw_ball_marker(frame: np.ndarray, position: tuple,
                     trail: deque, frame_id: int,
                     show_trail: bool = True,
                     show_info: bool = True) -> np.ndarray:
    """
    在影格上繪製球標記

    Args:
        frame: 影格
        position: (x, y, confidence)
        trail: 軌跡歷史
        frame_id: 幀號
        show_trail: 是否顯示軌跡
        show_info: 是否顯示資訊
    """
    if position is None:
        # 沒有偵測到球，顯示提示（右上角）
        h, w = frame.shape[:2]
        text = f"Frame {frame_id} - No ball"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = w - text_size[0] - 10
        cv2.putText(frame, text, (text_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame
    
    x, y, conf = position
    x, y = int(x), int(y)
    
    # 繪製軌跡（漸變效果）
    if show_trail and len(trail) > 1:
        trail_list = list(trail)
        for i in range(1, len(trail_list)):
            if trail_list[i-1] is None or trail_list[i] is None:
                continue
            
            # 計算顏色漸變（越舊越淡）
            alpha = i / len(trail_list)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))  # 從紅到綠
            thickness = max(1, int(3 * alpha))
            
            pt1 = (int(trail_list[i-1][0]), int(trail_list[i-1][1]))
            pt2 = (int(trail_list[i][0]), int(trail_list[i][1]))
            cv2.line(frame, pt1, pt2, color, thickness)
    
    # 繪製球的位置（圓圈 + 十字）
    # 外圈（根據信心度變色）
    if conf > 0.7:
        color = (0, 255, 0)  # 綠色 - 高信心度
    elif conf > 0.4:
        color = (0, 255, 255)  # 黃色 - 中信心度
    else:
        color = (0, 0, 255)  # 紅色 - 低信心度
    
    cv2.circle(frame, (x, y), 15, color, 2)
    cv2.circle(frame, (x, y), 3, color, -1)  # 中心點
    
    # 十字線
    cv2.line(frame, (x - 20, y), (x + 20, y), color, 1)
    cv2.line(frame, (x, y - 20), (x, y + 20), color, 1)
    
    # 顯示資訊（右上角）
    if show_info:
        h, w = frame.shape[:2]
        info_text = f"Frame {frame_id} | Ball: ({x}, {y}) | Conf: {conf:.2f}"
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = w - text_size[0] - 10

        # 繪製黑色背景
        cv2.rectangle(frame, (text_x - 5, 10), (w - 5, 35), (0, 0, 0), -1)
        # 繪製文字
        cv2.putText(frame, info_text, (text_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def calculate_speed(trail: deque) -> float:
    """計算當前速度"""
    if len(trail) < 2:
        return 0
    
    trail_list = list(trail)
    if trail_list[-1] is None or trail_list[-2] is None:
        return 0
    
    dx = trail_list[-1][0] - trail_list[-2][0]
    dy = trail_list[-1][1] - trail_list[-2][1]
    return np.sqrt(dx*dx + dy*dy)


def draw_court_config(frame: np.ndarray, court_config: dict) -> np.ndarray:
    """
    Draw court boundary, exclusion zones, and net line on frame.

    Args:
        frame: video frame
        court_config: court configuration dict
    """
    if not court_config:
        return frame

    # Draw court boundary (cyan, dashed-like thin line)
    boundary = court_config.get('court_boundary_polygon')
    if boundary and len(boundary) >= 3:
        pts = np.array(boundary, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
        # Label
        cv2.putText(frame, "Court", (pts[0][0] + 5, pts[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Draw exclusion zones (purple, semi-transparent fill)
    for i, zone in enumerate(court_config.get('exclusion_zones', [])):
        polygon = zone.get('polygon') if isinstance(zone, dict) else zone
        if not polygon:
            continue
        pts = np.array(polygon, dtype=np.int32)
        # Semi-transparent fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (200, 0, 200))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        # Border
        cv2.polylines(frame, [pts], isClosed=True, color=(200, 0, 200), thickness=2)
        # Label
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(frame, f"Excl-{i+1}", (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 200), 1)

    # Draw net line (white dashed)
    net_y = court_config.get('net_y')
    if net_y is not None:
        h, w = frame.shape[:2]
        # Draw dashed line
        dash_len = 20
        for x_start in range(0, w, dash_len * 2):
            x_end = min(x_start + dash_len, w)
            cv2.line(frame, (x_start, int(net_y)), (x_end, int(net_y)),
                     (255, 255, 255), 1)
        cv2.putText(frame, "Net", (10, int(net_y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame


def visualize_video(video_path: str, json_path: str, output_path: str = None,
                    show_trail: bool = True, trail_length: int = 30,
                    show_speed: bool = True, playback_speed: float = 1.0,
                    start_frame: int = 0, end_frame: int = None,
                    court_config: dict = None, show_center: bool = False):
    """
    視覺化追蹤結果
    
    Args:
        video_path: 輸入影片路徑
        json_path: 追蹤結果 JSON 路徑
        output_path: 輸出影片路徑（None 則只預覽）
        show_trail: 是否顯示軌跡
        trail_length: 軌跡長度
        show_speed: 是否顯示速度
        playback_speed: 播放速度倍率
        start_frame: 起始幀
        end_frame: 結束幀
    """
    # 載入追蹤數據
    print(f"載入追蹤數據: {json_path}")
    tracking_data, json_metadata = load_tracking_data(json_path)
    print(f"  共 {len(tracking_data)} 幀數據")

    # Load auto_court info from metadata (if available)
    auto_court_info = json_metadata.get('auto_court')
    if auto_court_info:
        print(f"  AutoCourt boundary detected, net_y={auto_court_info.get('net_y', '?')}")
    
    # 開啟影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"影片資訊: {width}x{height}, {fps:.1f} FPS, {total_frames} 幀")

    # 計算中心點（用於 show_center）
    center_point = None
    if show_center:
        if court_config:
            boundary = court_config.get('court_boundary_polygon')
            if boundary and len(boundary) == 4:
                pts = np.array(boundary, dtype=np.float32)
                raw_center = np.mean(pts, axis=0)
                court_h = max(pts[:, 1]) - min(pts[:, 1])
                center_point = (int(raw_center[0]), int(raw_center[1] + court_h * 0.1))
                raw_center_pt = (int(raw_center[0]), int(raw_center[1]))
                print(f"  Court center (raw): {raw_center_pt}")
                print(f"  Court center (shifted +10%): {center_point}")
        if center_point is None:
            center_point = (width // 2, int(height / 2 + height * 0.1))
            raw_center_pt = (width // 2, height // 2)
            print(f"  Frame center (raw): {raw_center_pt}")
            print(f"  Frame center (shifted +10%): {center_point}")

    # 設定結束幀
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    
    # 設定輸出（輸出模式必須指定 output_path）
    if output_path is None:
        print("錯誤：請使用 --output 指定輸出影片路徑")
        print("例如：python visualize_tracking.py --video input.mp4 --json data.json --output output.mp4")
        cap.release()
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"輸出影片: {output_path}")
    
    # 軌跡歷史
    trail = deque(maxlen=trail_length)
    
    # 跳到起始幀
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_id = start_frame
    
    print(f"\n開始處理 (幀 {start_frame} - {end_frame})...")
    
    while frame_id < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 取得幀數據
        frame_data = tracking_data.get(frame_id) or tracking_data.get(str(frame_id))

        # 繪製場地配置（排除區域、邊界、網線）
        frame = draw_court_config(frame, court_config)

        # 繪製 AutoCourt 估算邊界
        frame = draw_auto_court_boundary(frame, auto_court_info)

        # 繪製被排除的球員（紅色框 + X）
        frame = draw_rejected_players(frame, frame_data)

        # 繪製中心點標記
        if show_center and center_point is not None:
            cx, cy = center_point
            # 十字線
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)
            # 圓圈
            cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 2)
            # 標籤
            cv2.putText(frame, "CENTER", (cx + 12, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 繪製球員（先繪製，在底層）
        frame = draw_players(frame, frame_data)

        # 取得球位置
        position = get_ball_position(frame_data)

        # 更新軌跡
        if position:
            trail.append((position[0], position[1]))
        else:
            trail.append(None)

        # 繪製球標記（後繪製，在上層）
        frame = draw_ball_marker(frame, position, trail, frame_id, show_trail)
        
        # 顯示速度（右上角第二行）
        if show_speed and position:
            h, w = frame.shape[:2]
            speed = calculate_speed(trail)
            speed_text = f"Speed: {speed:.1f} px/frame"
            text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = w - text_size[0] - 10

            # 繪製黑色背景
            cv2.rectangle(frame, (text_x - 5, 40), (w - 5, 65), (0, 0, 0), -1)
            # 繪製文字
            cv2.putText(frame, speed_text, (text_x, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 輸出影片
        writer.write(frame)
        
        frame_id += 1
        
        # 進度顯示
        if frame_id % 100 == 0:
            progress = (frame_id - start_frame) / (end_frame - start_frame) * 100
            print(f"  處理進度: {progress:.1f}% (幀 {frame_id})")
    
    # 清理
    cap.release()
    writer.release()
    print(f"\n輸出完成: {output_path}")
    print("完成！")


def main():
    parser = argparse.ArgumentParser(description="球追蹤視覺化工具")
    parser.add_argument("--video", type=str, required=True, help="輸入影片路徑")
    parser.add_argument("--json", type=str, required=True, help="追蹤結果 JSON 路徑")
    parser.add_argument("--output", type=str, default=None, help="輸出影片路徑（不指定則只預覽）")
    parser.add_argument("--no-trail", action="store_true", help="不顯示軌跡")
    parser.add_argument("--trail-length", type=int, default=30, help="軌跡長度")
    parser.add_argument("--no-speed", action="store_true", help="不顯示速度")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率")
    parser.add_argument("--start", type=int, default=0, help="起始幀")
    parser.add_argument("--end", type=int, default=None, help="結束幀")
    parser.add_argument("--court-config", type=str, default=None,
                        help="Court config JSON (draws boundary, exclusion zones, net)")
    parser.add_argument("--show-center", action="store_true",
                        help="Show the effective center point used for player selection")

    args = parser.parse_args()

    # Load court config if provided
    court_cfg = None
    if args.court_config and os.path.exists(args.court_config):
        with open(args.court_config, 'r', encoding='utf-8') as f:
            court_cfg = json.load(f)

    visualize_video(
        video_path=args.video,
        json_path=args.json,
        output_path=args.output,
        show_trail=not args.no_trail,
        trail_length=args.trail_length,
        show_speed=not args.no_speed,
        playback_speed=args.speed,
        start_frame=args.start,
        end_frame=args.end,
        court_config=court_cfg,
        show_center=args.show_center
    )


if __name__ == "__main__":
    main()