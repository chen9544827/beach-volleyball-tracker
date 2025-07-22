# -*- coding: utf-8 -*-
# video_processing/event_analyzer.py (The Correct, Final Robust Version)

import numpy as np
from collections import deque

def get_ball_center(frame_data):
    """從一幀的資料中安全地獲取球的中心點。"""
    if frame_data and frame_data.get('ball_detections'):
        best_ball = max(frame_data['ball_detections'], key=lambda b: b.get('confidence', 0))
        box = best_ball['box_coords']
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    return None

def find_serve_by_toss_and_hit(all_frames_data, config):
    """
    【最終穩健版偵測邏輯】
    透過「驗證過的垂直拋球軌跡」後接「高速位移」來偵測發球。
    """
    min_toss_frames = config.get("min_toss_frames", 3)
    vertical_ratio_thresh = config.get("vertical_ratio_thresh", 2.0)
    hit_v_thresh = config.get("hit_v_thresh", 35.0)
    max_frames_between_toss_and_hit = config.get("max_frames_between_toss_and_hit", 75)
    max_lost_frames = config.get("max_lost_frames", 8)

    serve_events = []
    state = "SEARCHING_TOSS"
    toss_info = {}
    ball_history = deque(maxlen=min_toss_frames + 1)

    print("\n[最終穩健版智慧邏輯] 正在搜尋「垂直拋球 -> 高速擊球」事件序列...")

    for i in range(len(all_frames_data)):
        curr_ball_pos = get_ball_center(all_frames_data[i])
        ball_history.append(curr_ball_pos)

        if state == "SEARCHING_TOSS":
            if len(ball_history) < ball_history.maxlen or None in list(ball_history):
                continue

            start_pos, end_pos = ball_history[0], ball_history[-1]
            avg_velocity = (end_pos - start_pos) / (ball_history.maxlen - 1)
            avg_vx, avg_vy = avg_velocity[0], avg_velocity[1]

            is_upward = avg_vy < 0
            vertical_ratio = abs(avg_vy) / (abs(avg_vx) + 1e-6)
            is_vertical = vertical_ratio > vertical_ratio_thresh

            if is_upward and is_vertical:
                print(f"  > [第 {i} 幀] 確認垂直拋球軌跡 (V_ratio: {vertical_ratio:.1f}) -> 進入 等待高速擊球 狀態")
                state = "AWAITING_HIT"
                toss_info = {'toss_frame': i, 'lost_frames_count': 0}
                ball_history.clear()

        elif state == "AWAITING_HIT":
            if (i - toss_info['toss_frame']) > max_frames_between_toss_and_hit:
                state = "SEARCHING_TOSS"; ball_history.clear(); continue
            
            if curr_ball_pos is None:
                toss_info['lost_frames_count'] += 1
                if toss_info['lost_frames_count'] > max_lost_frames:
                    state = "SEARCHING_TOSS"; ball_history.clear()
                continue
            
            toss_info['lost_frames_count'] = 0

            if len(ball_history) > 1 and ball_history[-2] is not None:
                speed = np.linalg.norm(curr_ball_pos - ball_history[-2])
                if speed > hit_v_thresh:
                    print(f"  > [第 {i} 幀][偵測到高速位移] (速度: {speed:.1f}) -> 判定為發球！")
                    
                    server_player = None
                    if all_frames_data[i].get('player_detections'):
                        server_player = min(all_frames_data[i]['player_detections'],
                            key=lambda p: np.linalg.norm(np.array(p['center_point']) - curr_ball_pos))

                    serve_events.append({
                        "frame_id": i, "event_type": "SERVE", "toss_frame": toss_info['toss_frame'],
                        "hit_position": curr_ball_pos.tolist(), "hit_speed": speed,
                        "server_player_data": server_player
                    })
                    state = "SEARCHING_TOSS"
                    ball_history.clear()

    return serve_events