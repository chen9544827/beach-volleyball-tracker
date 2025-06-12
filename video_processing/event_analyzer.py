# video_processing/event_analyzer.py
import os
import json
import numpy as np
import cv2
import argparse

# --- 輔助函數 ---
def get_player_center(bbox):
    """計算球員邊界框的中心點"""
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

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
def find_serve_events(frame_data, court_config, pose_analyzer=None):
    """分析發球事件 - 簡化版"""
    serve_events = []
    ball_trajectory = []
    first_serve_found = False
    
    print(f"開始分析 {len(frame_data)} 幀的發球事件...")
    
    for frame_idx, frame in enumerate(frame_data):
        # 檢查是否有球
        if 'ball_detections' in frame and frame['ball_detections']:
            ball = frame['ball_detections'][0]
            ball_y = ball['center_y']
            
            # 記錄球的軌跡
            ball_trajectory.append({
                'frame': frame_idx,
                'position': (ball['center_x'], ball['center_y'])
            })
            
            # 如果球在網子上方，且是第一次長時間飛行
            if ball_y < court_config['net_y'] and not first_serve_found:
                # 檢查是否為長時間飛行
                if len(ball_trajectory) > 5:  # 可以調整這個閾值
                    serve_events.append({
                        'start_frame': frame_idx,
                        'end_frame': frame_idx,
                        'ball_position': (ball['center_x'], ball['center_y']),
                        'trajectory': ball_trajectory
                    })
                    first_serve_found = True
                    print(f"找到發球事件！發生在第 {frame_idx} 幀")
                    break
    
    print(f"分析完成，共找到 {len(serve_events)} 個發球事件")
    return serve_events

def is_serve(frame_data, net_y):
    """判斷是否為發球 - 判斷球的垂直和水平移動"""
    # 檢查是否有球
    if not frame_data.get('ball_detections'):
        return False
    
    # 取得最可能的球（信心度最高的）
    ball_detections = frame_data['ball_detections']
    if not ball_detections:
        return False
    
    ball = max(ball_detections, key=lambda x: x['confidence'])
    ball_pos = (ball['center_x'], ball['center_y'])
    
    # 檢查球的信心度
    if ball['confidence'] < 0.3:  # 降低信心度要求
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='分析排球影片中的發球事件')
    parser.add_argument('--input', required=True, help='輸入的 JSON 檔案路徑')
    parser.add_argument('--output_dir', required=True, help='輸出目錄')
    parser.add_argument('--court_config', required=True, help='場地設定檔路徑')
    parser.add_argument('--fps', type=float, default=25, help='影片的 FPS (預設: 25.0)')
    args = parser.parse_args()

    # 載入場地設定
    with open(args.court_config, 'r', encoding='utf-8') as f:
        court_config = json.load(f)
    net_y = court_config.get('net_y')
    if net_y is None:
        print("錯誤：場地設定檔中缺少 net_y 參數！")
        return

    # 載入追蹤資料
    with open(args.input, 'r', encoding='utf-8') as f:
        tracking_data = json.load(f)

    # 分析發球事件
    serve_events = []
    frames = tracking_data  # 直接使用追蹤資料，因為它已經是幀的列表
    print(f"開始分析 {len(frames)} 幀的發球事件...")

    # 發球檢測參數
    MIN_RISE_FRAMES = 5    # 最小上升幀數 (拋球階段)
    MIN_FALL_FRAMES = 3    # 最小下降幀數 (擊球前)
    MIN_PEAK_HEIGHT = 200  # 最小頂點高度 (像素)
    MAX_SERVE_DURATION = 2.0 # 最大發球持續時間 (秒)
    SERVE_AREA_X = [100, 1800] # 發球區域X範圍
    SERVE_AREA_Y = [300, 700]  # 發球區域Y範圍
    
    serve_events = []
    ball_history = []
    state = "IDLE"  # 狀態機: IDLE, TOSS, PEAK, HIT
    
    # 狀態追蹤變量
    toss_start_frame = None
    peak_frame = None
    hit_frame = None
    last_y = None
    upward_count = 0
    downward_count = 0

    for i in range(len(frames)):
        frame_data = frames[i]
        frame_id = frame_data.get('frame_id', i)
        
        # 獲取球位置 (跳過無球幀)
        if not frame_data.get('ball_detections'):
            continue
        ball = max(frame_data['ball_detections'], key=lambda x: x['confidence'])
        x, y = ball['center_x'], ball['center_y']
        ball_history.append({'frame': frame_id, 'x': x, 'y': y})
        
        # 狀態機處理
        if state == "IDLE":
            # 檢測拋球開始 (Y連續上升)
            if last_y is not None:
                if y < last_y: 
                    upward_count += 1
                else:
                    upward_count = 0
                    
                # 進入拋球階段條件
                if upward_count >= MIN_RISE_FRAMES and SERVE_AREA_X[0] <= x <= SERVE_AREA_X[1] and SERVE_AREA_Y[0] <= y <= SERVE_AREA_Y[1]:
                    state = "TOSS"
                    toss_start_frame = i - upward_count
                    print(f"檢測到拋球開始: 幀 {toss_start_frame}, 位置 ({x:.1f}, {y:.1f})")
            
        elif state == "TOSS":
            # 檢測頂點 (Y變化由正轉負)
            if last_y is not None and y < last_y:
                downward_count += 1
            else:
                downward_count = 0
                
            # 進入頂點狀態條件
            if downward_count >= MIN_FALL_FRAMES:
                state = "PEAK"
                peak_frame = i - downward_count
                print(f"檢測到軌跡頂點: 幀 {peak_frame}, 高度 {last_y:.1f}")
                
        elif state == "PEAK":
            # 檢測擊球點 (快速水平移動)
            if len(ball_history) > 5:
                dx = x - ball_history[-5]['x']
                dy = y - ball_history[-5]['y']
                
                # 擊球特徵: 快速水平移動 + 垂直下降
                if abs(dx) > 30 and dy < -20:  
                    state = "HIT"
                    hit_frame = i
                    # 計算發球持續時間
                    duration = (hit_frame - toss_start_frame) / 30.0
                    
                    if duration <= MAX_SERVE_DURATION and toss_start_frame >= 0 and toss_start_frame < len(ball_history):
                        print(f"檢測到擊球: 幀 {hit_frame}, 位置 ({x:.1f}, {y:.1f})")
                        
                        # 自動判斷發球方
                        serving_player_id = determine_serving_player(
                            frame_data, court_config, (x, y)
                        )
                        
                        serve_events.append({
                            'toss_frame': toss_start_frame,
                            'peak_frame': peak_frame,
                            'hit_frame': hit_frame,
                            'toss_position': (ball_history[toss_start_frame]['x'], ball_history[toss_start_frame]['y']),
                            'peak_position': (ball_history[peak_frame]['x'], ball_history[peak_frame]['y']),
                            'hit_position': (x, y),
                            'duration': duration,
                            'serving_player_id': serving_player_id
                        })
                    
                    # 重置狀態
                    state = "IDLE"
                    toss_start_frame = None
                    peak_frame = None
                    hit_frame = None
                    upward_count = 0
                    downward_count = 0
        
        last_y = y

    print(f"分析完成，共找到 {len(serve_events)} 個發球事件")

    # 只輸出最早的發球-擊球事件
    if serve_events:
        earliest_event = serve_events[0]
        print("\n最早的發球-擊球事件：")
        print(f"發球開始：幀 {earliest_event['start_frame']}, 位置 ({earliest_event['start_position'][0]:.1f}, {earliest_event['start_position'][1]:.1f})")
        print(f"擊球：幀 {earliest_event['hit_frame']}, 位置 ({earliest_event['hit_position'][0]:.1f}, {earliest_event['hit_position'][1]:.1f})")
        print(f"持續幀數：{(earliest_event['hit_frame'] - earliest_event['start_frame'] ):.0f}")
        #print(f"水平位移：{earliest_event['horizontal_displacement']:.1f} 像素")
        #print(f"垂直移動：{earliest_event['vertical_movement']:.1f} 像素")

def determine_serving_player(frame_data, court_config, hit_position):
    """自動判斷發球球員"""
    if not frame_data.get('player_detections'):
        return None
        
    # 取得場地幾何資訊
    court_poly = court_config.get('court_boundary_polygon')
    baseline_y = court_config.get('baseline_y')
    if not court_poly or baseline_y is None:
        return None
        
    # 篩選後場球員
    backcourt_players = []
    for player in frame_data['player_detections']:
        player_y = player['center_point'][1]
        # 球員在底線後方20像素內視為後場球員
        if player_y > baseline_y - 20:
            backcourt_players.append(player)
            
    # 如果沒有後場球員，返回None
    if not backcourt_players:
        return None
        
    # 選擇最接近擊球點的球員
    hit_x, hit_y = hit_position
    serving_player = min(backcourt_players, 
                        key=lambda p: abs(p['center_point'][0] - hit_x))
    return serving_player.get('player_id', None)

# 在主要循環中加入發球方判斷
# ... [在擊球點檢測代碼塊中] ...
    if x_change > X_HIT_THRESHOLD and y_change < Y_HIT_THRESHOLD:
        # 判斷發球方
        hit_frame_data = frames[hit_frame]
        serving_player_id = determine_serving_player(
            hit_frame_data, court_config, hit_point
        )
        
        # 記錄完整的發球事件
        serve_events.append({
            'start_frame': serve_start_frame,
            'start_time': serve_start_frame / 30.0,
            'start_position': serve_start_pos,
            'highest_point_frame': highest_point_frame,
            'highest_point_time': highest_point_frame / 30.0,
            'highest_point': highest_point,
            'hit_frame': hit_frame,
            'hit_time': hit_frame / 30.0,
            'hit_position': hit_point,
            'duration': duration,
            'horizontal_displacement': horizontal_displacement,
            'vertical_movement': vertical_movement,
            'serving_player_id': serving_player_id  # 新增發球方ID
        })

# ... [後續代碼保持不變] ...

# 儲存分析結果
    output_file = os.path.join(args.output_dir, "serve_events_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(serve_events, f, indent=2)
    
    print(f"分析完成，結果已保存到：{output_file}")
    print(f"共檢測到 {len(serve_events)} 個發球事件")
    
    # 輸出每個發球事件的幀數資訊
    if serve_events:
        print("\n發球事件詳細資訊：")
        for i, event in enumerate(serve_events, 1):
            print(f"\n發球事件 {i}:")
            print(f"  開始幀數: {event['start_frame']}")
            print(f"  最高點幀數: {event['highest_point_frame']}")
            print(f"  擊球幀數: {event['hit_frame']}")
            print(f"  持續幀數: {event['hit_frame'] - event['start_frame']}")
            print(f"  水平位移: {event['horizontal_displacement']:.1f} 像素")
            print(f"  垂直移動: {event['vertical_movement']:.1f} 像素")

if __name__ == '__main__':
    main()
