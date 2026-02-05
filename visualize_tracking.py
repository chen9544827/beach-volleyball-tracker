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


def load_tracking_data(json_path: str) -> dict:
    """載入追蹤數據"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 轉換為 frame_id -> data 的字典
    frames_data = data.get('frames', data)
    if isinstance(frames_data, list):
        return {item.get('frame_id', i): item for i, item in enumerate(frames_data)}
    return frames_data


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


def visualize_video(video_path: str, json_path: str, output_path: str = None,
                    show_trail: bool = True, trail_length: int = 30,
                    show_speed: bool = True, playback_speed: float = 1.0,
                    start_frame: int = 0, end_frame: int = None):
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
    tracking_data = load_tracking_data(json_path)
    print(f"  共 {len(tracking_data)} 幀數據")
    
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
    
    args = parser.parse_args()
    
    visualize_video(
        video_path=args.video,
        json_path=args.json,
        output_path=args.output,
        show_trail=not args.no_trail,
        trail_length=args.trail_length,
        show_speed=not args.no_speed,
        playback_speed=args.speed,
        start_frame=args.start,
        end_frame=args.end
    )


if __name__ == "__main__":
    main()