# analysis/jump_serve_analyzer.py (v6 - 增加偵錯資訊回傳)

import numpy as np

def get_player_center(player_data):
    """ 從球員資料中獲取中心點。 """
    if not player_data or 'center_point' not in player_data:
        return None
    return np.array(player_data['center_point'])

def analyze_jump_serve_by_pose(all_frames_data, server_data, hit_frame_id, config):
    """
    【位移分析法】
    回傳結果以及一個包含偵錯資訊的字典。
    """
    jump_height_threshold = config.get('jump_height_threshold', 20)
    action_window = range(max(0, hit_frame_id - 35), hit_frame_id)
    
    # 初始化偵錯資訊字典
    debug_info = {
        "status": "Started",
        "action_window": list(action_window),
        "hip_y_trajectory": [],
        "crouch_y": None,
        "peak_y": None,
        "displacement": None
    }

    if not server_data or not server_data.get('pose_keypoints'):
        debug_info["status"] = "Error: No pose_keypoints in initial server data."
        return 'Unknown (No pose_keypoints)', debug_info

    server_initial_pos = get_player_center(server_data)
    if server_initial_pos is None:
        debug_info["status"] = "Error: No center point in initial server data."
        return 'Unknown (No Center)', debug_info

    hip_y_trajectory = []
    for frame_idx in action_window:
        if frame_idx >= len(all_frames_data): continue
        frame_data = all_frames_data[frame_idx]
        players = frame_data.get('player_detections', [])
        if not players: continue

        closest_player = min(players, key=lambda p: np.linalg.norm(get_player_center(p) - server_initial_pos if get_player_center(p) is not None else float('inf')))
        
        if np.linalg.norm(get_player_center(closest_player) - server_initial_pos) > 150 or not closest_player.get('pose_keypoints'):
            continue

        kpts = np.array(closest_player['pose_keypoints'])
        if kpts.shape[0] < 17 or kpts.shape[1] < 3: continue

        left_hip_y = kpts[11, 1] if kpts[11, 2] > 0.3 else np.nan
        right_hip_y = kpts[12, 1] if kpts[12, 2] > 0.3 else np.nan
        
        avg_hip_y = np.nanmean([left_hip_y, right_hip_y])
        
        if not np.isnan(avg_hip_y):
            hip_y_trajectory.append(avg_hip_y)
    
    debug_info["hip_y_trajectory"] = hip_y_trajectory
    
    if len(hip_y_trajectory) < 5:
        debug_info["status"] = f"Error: Insufficient Trajectory Data (found {len(hip_y_trajectory)} points)."
        return "Unknown (Insufficient Trajectory)", debug_info

    crouch_y = max(hip_y_trajectory)
    peak_y = min(hip_y_trajectory)
    vertical_displacement = crouch_y - peak_y

    debug_info.update({
        "crouch_y": crouch_y,
        "peak_y": peak_y,
        "displacement": vertical_displacement,
        "threshold": jump_height_threshold
    })
    
    if vertical_displacement > jump_height_threshold:
        debug_info["status"] = "Success: Judged as JUMP."
        return "Jump", debug_info
    else:
        debug_info["status"] = "Success: Judged as STANDING."
        return "Standing", debug_info