# video_processing/event_analyzer.py
import cv2
import numpy as np
import json # 如果將來這些函數內部需要處理JSON，雖然目前主要由調用方處理

# --- 輔助函數 ---
def get_player_center(player_box_input): # 將參數名改為 player_box_input 更清晰
    """
    計算邊界框的中心點。
    player_box_input: 一個包含至少四個數值 (x1, y1, x2, y2) 的列表或元組。
                      額外的元素（如信心度）將被忽略。
    """
    if not player_box_input or len(player_box_input) < 4:
        # print("警告 (get_player_center): 輸入的 player_box_input 無效或長度不足4。")
        return None, None 
    
    # 只取前四個元素作為座標
    coords = player_box_input[:4]
    
    try:
        # 將座標轉換為浮點數以便計算
        x1, y1, x2, y2 = map(float, coords)
    except ValueError:
        # print("警告 (get_player_center): Box 座標無法轉換為浮點數。")
        return None, None
        
    return (x1 + x2) / 2, (y1 + y2) / 2
    

def get_baselines_from_ordered_corners(court_polygon_points):
    if court_polygon_points is None or len(court_polygon_points) != 4:
        # print("錯誤 (get_baselines_from_ordered_corners): court_polygon_points 必須是4個點。")
        return None, None
    points = [tuple(p) for p in court_polygon_points]
    P0, P1, P2, P3 = points[0], points[1], points[2], points[3] # P0:左上, P1:左下, P2:右下, P3:右上
    far_baseline = (P0, P3)
    near_baseline = (P1, P2)
    return far_baseline, near_baseline

def is_point_outside_line_segment_extended(point, line_p1, line_p2, court_center_approx, is_far_baseline, x_margin_factor=0.15, y_margin_pixels=10):
    px, py = point
    x1, y1 = line_p1
    x2, y2 = line_p2

    min_baseline_x = min(x1, x2)
    max_baseline_x = max(x1, x2)
    baseline_width_x = max_baseline_x - min_baseline_x
    if baseline_width_x < 1: baseline_width_x = 1
    x_expansion = baseline_width_x * x_margin_factor
    if not (min_baseline_x - x_expansion <= px <= max_baseline_x + x_expansion):
        return False

    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    D_point = A * px + B * py + C

    if court_center_approx is None:
        if is_far_baseline: return py < min(y1, y2) - y_margin_pixels
        else: return py > max(y1, y2) + y_margin_pixels
        
    cx_court, cy_court = court_center_approx
    D_center = A * cx_court + B * cy_court + C

    if D_point == 0: return False
    is_outside_of_line = (D_point * D_center < 0)
    if not is_outside_of_line: return False

    # 避免 A 和 B 同時為0（雖然對於線段不太可能）
    denominator_sq = A**2 + B**2
    if denominator_sq == 0: return False 
    distance_to_line = abs(D_point) / np.sqrt(denominator_sq)
    return distance_to_line >= y_margin_pixels

def is_player_behind_baseline(player_center, court_polygon_points, frame_height, frame_width, court_center_approx_for_normal_check):
    if player_center is None or player_center[0] is None or player_center[1] is None:
        return False
    px, py = player_center

    if court_polygon_points is None: return False
    court_poly_np = np.array(court_polygon_points, dtype=np.int32)
    if court_poly_np.ndim != 2 or court_poly_np.shape[0] != 4 or court_poly_np.shape[1] != 2:
        return False

    if cv2.pointPolygonTest(court_poly_np, (float(px), float(py)), False) >= 0:
        return False 

    far_baseline, near_baseline = get_baselines_from_ordered_corners(court_polygon_points)
    if far_baseline is None or near_baseline is None:
        return False 

    if is_point_outside_line_segment_extended(
        (px,py), far_baseline[0], far_baseline[1], 
        court_center_approx_for_normal_check, is_far_baseline=True, 
        y_margin_pixels=10): # 可以將 y_margin_pixels 設為可配置的參數
        return True
    if is_point_outside_line_segment_extended(
        (px,py), near_baseline[0], near_baseline[1], 
        court_center_approx_for_normal_check, is_far_baseline=False,
        y_margin_pixels=10): # 可以將 y_margin_pixels 設為可配置的參數
        return True
    return False

# --- 主要分析函數 ---
def find_serve_events(all_frames_data, court_geometry, fps, frame_w, frame_h):
    serve_events = []
    if not court_geometry or "court_boundary_polygon" not in court_geometry:
        print("錯誤: find_serve_events - 未提供有效的場地邊界定義。")
        return serve_events

    court_polygon = court_geometry["court_boundary_polygon"]
    if not isinstance(court_polygon, list) or len(court_polygon) != 4:
        print("錯誤: find_serve_events - court_boundary_polygon 無效 (需要4個點的列表)。")
        return serve_events

    court_center_approx = None
    try:
        poly_np = np.array(court_polygon, dtype=np.int32)
        M = cv2.moments(poly_np)
        if M["m00"] != 0:
            court_center_approx = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        elif court_polygon: 
            court_center_approx = (
                int(sum(p[0] for p in court_polygon) / len(court_polygon)),
                int(sum(p[1] for p in court_polygon) / len(court_polygon))
            )
    except Exception as e:
        print(f"警告 (find_serve_events): 計算場地中心時出錯: {e}")

    # --- 參數設定 ---
    BALL_MIN_SPEED_START_SQ = 10**2 
    PLAYER_BALL_PROXIMITY_THRESH = 70
    MIN_FRAMES_FOR_SERVE_SETUP = int(fps * 0.3) 
    MIN_BALL_FLIGHT_FRAMES_AFTER_SERVE = int(fps * 0.2)
    SERVE_COOLDOWN_FRAMES = int(fps * 3)
    last_serve_frame_idx = -SERVE_COOLDOWN_FRAMES 

    potential_server_info = None 
    
    for i in range(len(all_frames_data)):
        current_frame_data = all_frames_data[i]
        frame_idx = current_frame_data["frame_id"]

        if frame_idx < last_serve_frame_idx + SERVE_COOLDOWN_FRAMES:
            potential_server_info = None 
            continue

        current_ball_infos = current_frame_data.get("ball_detections", [])
        current_player_infos = current_frame_data.get("player_detections", [])

        found_potential_this_frame = False
        if not current_player_infos:
            if potential_server_info and frame_idx > potential_server_info.get('last_seen_frame_idx', frame_idx) + int(fps*0.2):
                 potential_server_info = None
            continue

        for p_info_dict in current_player_infos:
            player_box_for_center_calc = p_info_dict.get("box_coords")
            if not player_box_for_center_calc or len(player_box_for_center_calc) < 4:
                continue
            # get_player_center 期望的是一個至少包含4個座標的序列
            player_center_x, player_center_y = get_player_center(player_box_for_center_calc) 
            if player_center_x is None: continue

            player_center = (player_center_x, player_center_y)

            if is_player_behind_baseline(player_center, court_polygon, frame_h, frame_w, court_center_approx):
                ball_is_near_this_player = False
                if current_ball_infos:
                    ball_info = current_ball_infos[0] 
                    dist_player_ball = np.sqrt((player_center[0] - ball_info["center_x"])**2 + \
                                               (player_center[1] - ball_info["center_y"])**2)
                    if dist_player_ball < PLAYER_BALL_PROXIMITY_THRESH:
                        ball_is_near_this_player = True
                
                if ball_is_near_this_player:
                    found_potential_this_frame = True
                    if potential_server_info and \
                       potential_server_info.get('player_box_coords') == p_info_dict["box_coords"]:
                        potential_server_info['ball_nearby_frames'] += 1
                        potential_server_info['last_seen_frame_idx'] = frame_idx
                    else:
                        potential_server_info = {
                            'player_info_dict': p_info_dict,
                            'player_box_coords': p_info_dict["box_coords"], 
                            'player_center': player_center,
                            'frame_idx_setup_start': frame_idx, 
                            'ball_nearby_frames': 1,
                            'last_seen_frame_idx': frame_idx
                        }
                    break 
        
        if not found_potential_this_frame and potential_server_info:
            if frame_idx > potential_server_info.get('last_seen_frame_idx', frame_idx) + int(fps*0.2): # 0.2秒內沒再滿足就清除
                potential_server_info = None
        
        if potential_server_info and \
           potential_server_info['ball_nearby_frames'] >= MIN_FRAMES_FOR_SERVE_SETUP:
            if i + 1 < len(all_frames_data):
                next_frame_data = all_frames_data[i+1]
                next_ball_infos = next_frame_data.get("ball_detections", [])

                if current_ball_infos and next_ball_infos:
                    current_ball_of_interest_info = current_ball_infos[0] 
                    next_ball_of_interest_info = next_ball_infos[0]     
                    cb_pos = (current_ball_of_interest_info["center_x"], current_ball_of_interest_info["center_y"])
                    nb_pos = (next_ball_of_interest_info["center_x"], next_ball_of_interest_info["center_y"])
                    ball_speed_sq = (nb_pos[0] - cb_pos[0])**2 + (nb_pos[1] - cb_pos[1])**2
                    
                    if ball_speed_sq > BALL_MIN_SPEED_START_SQ:
                        dist_server_ball_start = np.sqrt(
                            (potential_server_info['player_center'][0] - cb_pos[0])**2 + \
                            (potential_server_info['player_center'][1] - cb_pos[1])**2)

                        if dist_server_ball_start < PLAYER_BALL_PROXIMITY_THRESH:
                            ball_flies_consistently = True
                            if i + MIN_BALL_FLIGHT_FRAMES_AFTER_SERVE < len(all_frames_data):
                                for k_fly in range(1, MIN_BALL_FLIGHT_FRAMES_AFTER_SERVE + 1):
                                    future_frame_data = all_frames_data[i+k_fly]
                                    if not future_frame_data.get("ball_detections", []):
                                        ball_flies_consistently = False; break
                            else:
                                ball_flies_consistently = False 

                            if ball_flies_consistently:
                                serve_event_data = {
                                    "serve_frame_idx": frame_idx, 
                                    "serving_player_info": potential_server_info['player_info_dict'],
                                    "ball_start_position": cb_pos,
                                    "ball_initial_motion_vector": (nb_pos[0] - cb_pos[0], nb_pos[1] - cb_pos[1])
                                }
                                serve_events.append(serve_event_data)
                                print(f"** 偵測到發球事件! 影格: {frame_idx}, "
                                      f"發球員 Box: {potential_server_info['player_info_dict'].get('box_coords')} **")
                                last_serve_frame_idx = frame_idx
                                potential_server_info = None 
    return serve_events