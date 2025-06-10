# video_processing/event_analyzer.py
import os
import json
import numpy as np
import cv2
from serve_pose_analyzer import ServePoseAnalyzer
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

    # 只分析一次發球（自動尋找上拋-最高點-擊球，累積x位移）
    vertical_threshold = 30  # 垂直移動閾值
    horizontal_threshold = 50  # 水平移動閾值
    max_serve_duration = 2.0  # 發球最大持續時間（秒）
    min_vertical_movement = 100  # 最小垂直移動距離（像素）
    max_start_y = 200  # 發球開始時的最大y值（像素）

    min_y = None
    min_y_frame = None
    min_y_time = None
    min_y_x = None
    serve_start_frame = None
    serve_start_time = None
    serve_start_x = None
    serve_start_y = None
    found_serve = False
    after_highest_x = None
    after_highest_time = None
    last_y = None
    vertical_movement = False
    vertical_movement_distance = 0  # 追蹤垂直移動距離
    for i in range(len(frames)):
        frame_data = frames[i]
        if not frame_data.get('ball_detections'):
            continue
        ball = max(frame_data['ball_detections'], key=lambda x: x['confidence'])
        x, y = ball['center_x'], ball['center_y']
        t = i / args.fps
        
        # 輸出9秒附近的球軌跡
        if 8.5 <= t <= 10.0:
            if last_y is not None:
                dy = y - last_y
                print(f"時間 {t:.2f}秒: 球位置 ({x:.1f}, {y:.1f}), 垂直變化 {dy:.2f}")
            last_y = y
            continue
            
        if last_y is not None:
            dy = y - last_y
            # 檢查是否有明顯垂直上拋（y連續下降）
            if dy < -vertical_threshold:
                vertical_movement = True
                vertical_movement_distance += abs(dy)  # 累積垂直移動距離
        last_y = y
        
        # 檢查發球開始位置
        if serve_start_frame is None and y > max_start_y:
            continue
            
        if not vertical_movement:
            continue
            
        if serve_start_frame is None:
            # 找到拋球開始（y開始下降）
            serve_start_frame = i
            serve_start_time = t
            serve_start_x = x
            serve_start_y = y
            min_y = y
            min_y_frame = i
            min_y_time = t
            min_y_x = x
            after_highest_x = None
            after_highest_time = None
        else:
            # 只要y持續下降就更新最高點
            if y < min_y:
                min_y = y
                min_y_frame = i
                min_y_time = t
                min_y_x = x
                after_highest_x = None
                after_highest_time = None
            # 一旦y開始上升，進入擊球判斷
            elif y > min_y + vertical_threshold:
                # 檢查垂直移動距離是否足夠
                if vertical_movement_distance < min_vertical_movement:
                    # 重置狀態，繼續尋找
                    serve_start_frame = None
                    serve_start_time = None
                    serve_start_x = None
                    serve_start_y = None
                    min_y = None
                    min_y_frame = None
                    min_y_time = None
                    min_y_x = None
                    after_highest_x = None
                    after_highest_time = None
                    vertical_movement = False
                    vertical_movement_distance = 0
                    continue
                    
                # 累積x位移
                if after_highest_x is None:
                    after_highest_x = x
                    after_highest_time = t
                dx_cum = abs(x - min_y_x)
                duration = t - min_y_time
                if duration <= max_serve_duration and dx_cum > horizontal_threshold:
                    print(f"自動判斷發球：")
                    print(f"  拋球開始：{serve_start_time:.2f}秒 (frame {serve_start_frame}, x={serve_start_x:.1f}, y={serve_start_y:.1f})")
                    print(f"  最高點：{min_y_time:.2f}秒 (frame {min_y_frame}, x={min_y_x:.1f}, y={min_y:.1f})")
                    print(f"  擊球：{t:.2f}秒 (frame {i}, x={x:.1f}, y={y:.1f})")
                    print(f"  最高點到擊球時間：{duration:.2f}秒，水平累積位移：{dx_cum:.1f}像素")
                    print(f"  垂直移動距離：{vertical_movement_distance:.1f}像素")
                    serve_events.append({
                        'start_frame': serve_start_frame,
                        'start_time': serve_start_time,
                        'highest_point_frame': min_y_frame,
                        'highest_point_time': min_y_time,
                        'highest_point': [min_y_x, min_y],
                        'hit_frame': i,
                        'hit_time': t,
                        'hit_position': [x, y],
                        'duration': duration,
                        'horizontal_displacement': dx_cum,
                        'vertical_movement': vertical_movement_distance
                    })
                    found_serve = True
                    break
                elif duration > max_serve_duration:
                    # 超過時間限制，重置
                    serve_start_frame = None
                    serve_start_time = None
                    serve_start_x = None
                    serve_start_y = None
                    min_y = None
                    min_y_frame = None
                    min_y_time = None
                    min_y_x = None
                    after_highest_x = None
                    after_highest_time = None
                    vertical_movement = False
                    vertical_movement_distance = 0

    print(f"分析完成，共找到 {len(serve_events)} 個發球事件")

    # 儲存分析結果
    output_path = os.path.join(args.output_dir, 'serve_events_analysis.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serve_events, f, indent=2)
    print(f"分析完成，結果已保存到：{output_path}")
    print(f"共檢測到 {len(serve_events)} 個發球事件")

if __name__ == '__main__':
    main()