# video_processing/event_analyzer.py
import numpy as np
import cv2

# --- 姿態關鍵點索引 ---
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

def is_player_behind_baseline(player_center, court_polygon):
    if court_polygon is None or len(court_polygon) < 4: return True 
    point = (float(player_center[0]), float(player_center[1]))
    if cv2.pointPolygonTest(np.array(court_polygon, dtype=np.int32), point, False) >= 0: return False
    return True

def find_serve_by_pose_and_toss(all_frames_data, config, court_polygon):
    """
    [最終穩定版]
    實作了最穩健的狀態機，能正確理解從「持球準備」到「拋球」再到「擊球」的完整動作序列。
    """
    # --- 參數設定 ---
    wrist_dist_thresh = config.get("wrist_dist_thresh", 50)
    min_pose_held_frames = config.get("min_pose_held_frames", 3)
    toss_upward_vel_thresh = config.get("toss_upward_vel_thresh", 5) 
    max_horizontal_ratio = config.get("max_horizontal_ratio", 1.5) 
    min_toss_validation_frames = config.get("min_toss_validation_frames", 4)
    min_toss_validation_height = config.get("min_toss_validation_height", 80)
    hit_dist_thresh = config.get("hit_dist_thresh", 80)
    max_lost_frames_tolerance = config.get("max_lost_frames", 15)
    reacquisition_radius = config.get("reacquisition_radius", 150)
    max_validation_lost_frames = config.get("max_validation_lost_frames", 5)
    
    # --- 狀態機 ---
    state = "SEARCHING"
    pose_confirmation_frame = -1
    toss_data = {}

    print("\n[智慧推理邏輯-最終穩定版] 正在搜尋發球動作序列...")
    
    for i in range(1, len(all_frames_data)):
        prev_frame_data = all_frames_data[i-1]; curr_frame_data = all_frames_data[i]
        players, balls = curr_frame_data.get('player_detections', []), curr_frame_data.get('ball_detections', [])
        
        if state == "SEARCHING":
            if not players or balls: continue
            for player in players:
                if not is_player_behind_baseline(player['center_point'], court_polygon): continue
                kpts = player.get("pose_keypoints")
                if not kpts or len(kpts) < 17: continue
                l_wrist, r_wrist, l_shoulder_y = np.array(kpts[LEFT_WRIST][:2]), np.array(kpts[RIGHT_WRIST][:2]), kpts[LEFT_SHOULDER][1]
                if kpts[LEFT_WRIST][2] > 0.4 and kpts[RIGHT_WRIST][2] > 0.4:
                    wrist_dist = np.linalg.norm(l_wrist - r_wrist)
                    if wrist_dist < wrist_dist_thresh and l_wrist[1] > (l_shoulder_y + 10):
                        print(f"  > [第 {i} 幀][偵測到發球區姿勢] -> 進入 確認姿勢 狀態")
                        state = "CONFIRMING_POSE"; pose_confirmation_frame = i
                        toss_data = {'server_id': player['center_point'], 'server_info': player}; break
        
        elif state == "CONFIRMING_POSE":
            if (i - pose_confirmation_frame) >= min_pose_held_frames:
                print(f"  > [第 {i} 幀][確認姿勢成功] -> 進入 耐心等待拋球 狀態")
                state = "WAITING_FOR_TOSS"
            elif not players: print(f"  > [第 {i} 幀][重設] 確認期間球員消失"); state = "SEARCHING"
        
        elif state == "WAITING_FOR_TOSS":
            if balls:
                curr_ball = max(balls, key=lambda b: b['confidence']); curr_ball_pos = np.array(curr_ball['center_point'])
                server_pos = np.array(toss_data['server_id'])
                if np.linalg.norm(curr_ball_pos - server_pos) < reacquisition_radius:
                    ball_vy, ball_vx = 0, 0
                    if prev_frame_data.get('ball_detections'):
                        prev_ball_pos = min([b['center_point'] for b in prev_frame_data['ball_detections']], key=lambda p: np.linalg.norm(np.array(p) - curr_ball_pos), default=curr_ball_pos)
                        ball_vy = prev_ball_pos[1] - curr_ball_pos[1]; ball_vx = curr_ball_pos[0] - prev_ball_pos[0]
                    if ball_vy > toss_upward_vel_thresh:
                        if abs(ball_vx) > ball_vy * max_horizontal_ratio:
                            print(f"  > [第 {i} 幀][忽略] 偵測到水平移動 (vx: {ball_vx:.1f}, vy: {ball_vy:.1f})")
                            continue
                        print(f"  > [第 {i} 幀][偵測到垂直拋球] (vy: {ball_vy:.1f}) -> 進入 驗證軌跡 狀態")
                        state = "VALIDATING_TOSS"; toss_data['validation_start_frame'] = i
                        toss_data['validation_lost_frames'] = 0
            if (i - pose_confirmation_frame) > 150: print(f"  > [第 {i} 幀][重設] 等待拋球超時"); state = "SEARCHING"

        elif state == "VALIDATING_TOSS":
            if not balls:
                toss_data['validation_lost_frames'] = toss_data.get('validation_lost_frames', 0) + 1
                if toss_data['validation_lost_frames'] > max_validation_lost_frames:
                    print(f"    ---> [重設] 驗證期間球失蹤太久"); state = "WAITING_FOR_TOSS"
                continue
            toss_data['validation_lost_frames'] = 0
            curr_ball = max(balls, key=lambda b: b['confidence']); curr_ball_pos = np.array(curr_ball['center_point']); ball_vy = 0
            if prev_frame_data.get('ball_detections'):
                prev_ball_pos = min([b['center_point'] for b in prev_frame_data['ball_detections']], key=lambda p: np.linalg.norm(np.array(p) - curr_ball_pos), default=curr_ball_pos)
                ball_vy = prev_ball_pos[1] - curr_ball_pos[1]
            if ball_vy < -1: print(f"  > [第 {i} 幀][重設] 拋球軌跡不持續"); state = "WAITING_FOR_TOSS"; continue
            frames_since_validation = i - toss_data.get('validation_start_frame', i)
            height_gain_since_validation = toss_data.get('validation_start_pos', curr_ball_pos)[1] - curr_ball_pos[1]
            print(f"  [第 {i} 幀][驗證中] 持續: {frames_since_validation}/{min_toss_validation_frames} 幀, 高度: {height_gain_since_validation:.1f}/{min_toss_validation_height} px")
            if frames_since_validation >= min_toss_validation_frames and height_gain_since_validation >= min_toss_validation_height:
                print(f"  > [第 {i} 幀][確認拋球] 軌跡驗證成功！-> 進入 等待頂點 狀態")
                state = "AWAITING_APEX"; toss_data['frames_lost_counter'] = 0

        elif state == "AWAITING_APEX":
            if not balls:
                toss_data['frames_lost_counter'] = getattr(toss_data, 'frames_lost_counter', 0) + 1
                if toss_data['frames_lost_counter'] > max_lost_frames_tolerance: print(f"    ---> [重設] 等待頂點期間球失蹤太久"); state = "SEARCHING"
                continue
            toss_data['frames_lost_counter'] = 0; ball_vy = 0
            if prev_frame_data.get('ball_detections'):
                curr_ball = max(balls, key=lambda b: b['confidence'])
                prev_ball_pos = min([b['center_point'] for b in prev_frame_data['ball_detections']], key=lambda p: np.linalg.norm(np.array(p) - np.array(curr_ball['center_point'])), default=curr_ball['center_point'])
                ball_vy = np.array(curr_ball['center_point'])[1] - np.array(prev_ball_pos)[1]
            if ball_vy > 1:
                print(f"  > [第 {i} 幀][到達頂點] 球已開始下落，進入 等待擊球 狀態。"); state = "AWAITING_HIT"
            elif (i - toss_data.get('validation_start_frame', i)) > 60: print(f"  > [第 {i} 幀][重設] 等待頂點超時"); state = "SEARCHING"

        elif state == "AWAITING_HIT":
            if not balls:
                toss_data['frames_lost_counter'] = getattr(toss_data, 'frames_lost_counter', 0) + 1
                if toss_data['frames_lost_counter'] > max_lost_frames_tolerance: print(f"    ---> [重設] 等待擊球期間球失蹤太久"); state = "SEARCHING"
                continue
            toss_data['frames_lost_counter'] = 0; current_server_data = None
            original_server_id = toss_data.get('server_id')
            if original_server_id and players:
                candidates = [(p, np.linalg.norm(np.array(p['center_point']) - np.array(original_server_id))) for p in players]
                valid_candidates = [cand for cand in candidates if cand[1] < reacquisition_radius]
                if valid_candidates: current_server_data, _ = min(valid_candidates, key=lambda item: item[1])
            if current_server_data:
                toss_data['server_id'] = current_server_data['center_point']
                curr_ball = max(balls, key=lambda b: b['confidence']); curr_ball_pos = np.array(curr_ball['center_point'])
                dist_to_server = np.linalg.norm(np.array(current_server_data['center_point']) - curr_ball_pos)
                print(f"  [第 {i} 幀][等待擊球] 追蹤中... [距離: {dist_to_server:.1f} (需 < {hit_dist_thresh})]")
                if dist_to_server < hit_dist_thresh:
                    print(f"  [成功!] 條件滿足，偵測到擊球！")
                    return [{"frame_id": i, "event_type": "SERVE", "server_player_data": current_server_data, "ball_position": list(curr_ball_pos)}]
            else:
                toss_data['frames_lost_counter'] = getattr(toss_data, 'frames_lost_counter', 0) + 1
                if toss_data['frames_lost_counter'] > max_lost_frames_tolerance: print(f"    ---> [重設] 目標(球員)失蹤太久"); state = "SEARCHING"
            if (i - pose_confirmation_frame) > 240: print(f"  > [第 {i} 幀][重設] 整個序列等待超時"); state = "SEARCHING"
                
    return []