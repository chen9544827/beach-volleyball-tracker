# video_processing/track_ball_and_player.py (v15 最終正確版 - 根據您的原始碼修正)

# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import traceback
import numpy as np
from ultralytics import YOLO

# --- 關鍵修正：定義正確的模型路徑與檔名 ---
# 獲取此腳本檔案所在的目錄 (e.g., .../video_processing)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# 獲取專案的根目錄 (e.g., .../beach-volleyball-tracker)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# 構造 'models' 資料夾的絕對路徑
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# 根據您提供的正確檔名
PLAYER_MODEL_NAME = 'yolov8s-pose.pt'
BALL_MODEL_NAME = 'ball_best.pt'

# --- 修改後的偵測邏輯 - 整合 ByteTrack 追蹤 ---
def detect_ball(frame, ball_model, conf_thresh, background_ball_zones, use_tracking=False, tracker_config=None):
    """
    球體偵測函數 - 支援追蹤模式
    
    Args:
        frame: 影片幀
        ball_model: YOLO 球體偵測模型
        conf_thresh: 置信度閾值
        background_ball_zones: 背景球過濾區域
        use_tracking: 是否使用追蹤模式 (預設 False 保持向下相容)
        tracker_config: 追蹤器配置檔路徑 (例如 'configs/bytetrack_ball.yaml')
    """
    import cv2
    detected_balls = []
    try:
        if use_tracking and tracker_config:
            # 使用追蹤模式
            results = ball_model.track(
                frame, 
                conf=conf_thresh, 
                classes=[0], 
                persist=True,
                tracker=tracker_config,
                verbose=False
            )
        else:
            # 使用原始偵測模式(向下相容)
            results = ball_model(frame, conf=conf_thresh, classes=[0], verbose=False)
        
        if not results or not results[0].boxes: return detected_balls
        
        for box in results[0].boxes:
            if box.xyxy is None or len(box.xyxy) == 0: continue
            coords = box.xyxy[0].cpu().numpy()
            if len(coords) < 4: continue
            x1, y1, x2, y2 = map(int, coords)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # 提取 track_id (如果有的話)
            track_id = -1
            if use_tracking and box.id is not None:
                track_id = int(box.id[0].cpu().numpy())
            
            is_in_background_zone = False
            if background_ball_zones:
                for zone in background_ball_zones:
                    if zone.get('x1') is not None and zone['x1'] <= center_x <= zone['x2'] and zone['y1'] <= center_y <= zone['y2']:
                        is_in_background_zone = True; break
            
            detected_balls.append({
                "track_id": track_id,
                "box_coords": [x1, y1, x2, y2], 
                "confidence": float(box.conf[0].cpu().numpy()),
                "center_point": [center_x, center_y], 
                "is_in_background_zone": is_in_background_zone
            })
    except Exception as e: 
        print(f"!! Exception in detect_ball: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    return detected_balls

def detect_and_filter_players(frame, player_pose_model, conf_thresh, court_boundary_np, exclusion_zones_np, court_center_xy, use_tracking=False, tracker_config=None):
    """
    球員偵測與過濾函數 - 支援追蹤模式
    
    Args:
        frame: 影片幀
        player_pose_model: YOLO 姿態偵測模型
        conf_thresh: 置信度閾值
        court_boundary_np: 場地邊界多邊形
        exclusion_zones_np: 排除區域列表
        court_center_xy: 場地中心座標
        use_tracking: 是否使用追蹤模式 (預設 False 保持向下相容)
        tracker_config: 追蹤器配置檔路徑 (例如 'configs/bytetrack_player.yaml')
    """
    import cv2
    all_candidates = []
    try:
        if use_tracking and tracker_config:
            # 使用追蹤模式
            results = player_pose_model.track(
                frame,
                conf=conf_thresh,
                classes=[0],
                persist=True,
                tracker=tracker_config,
                verbose=False
            )
        else:
            # 使用原始偵測模式(向下相容)
            results = player_pose_model(frame, conf=conf_thresh, classes=[0], verbose=False)
        
        if not results or not results[0].boxes or not results[0].keypoints: return all_candidates
        
        for i in range(len(results[0].boxes)):
            box, kpts = results[0].boxes[i], results[0].keypoints[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            center_pt = (float((x1+x2)/2), float((y1+y2)/2))
            
            # 提取 track_id (如果有的話)
            track_id = -1
            if use_tracking and box.id is not None:
                track_id = int(box.id[0].cpu().numpy())
            
            in_exclusion = False
            if exclusion_zones_np:
                for zone_np in exclusion_zones_np:
                    if cv2.pointPolygonTest(zone_np, center_pt, False) >= 0: in_exclusion = True; break
            if in_exclusion: continue
            
            is_inside = cv2.pointPolygonTest(court_boundary_np, center_pt, False) >= 0 if court_boundary_np is not None else False
            dist_to_center = np.linalg.norm(np.array(center_pt) - np.array(court_center_xy)) if court_center_xy else float('inf')
            
            keypoints_xyc_list = []
            if kpts.xy is not None and kpts.conf is not None:
                kpts_xy, kpts_conf = kpts.xy[0].cpu().numpy(), kpts.conf[0].cpu().numpy()
                for kp_idx in range(kpts_xy.shape[0]): 
                    keypoints_xyc_list.append([float(kpts_xy[kp_idx, 0]), float(kpts_xy[kp_idx, 1]), float(kpts_conf[kp_idx])])
            
            all_candidates.append({
                "track_id": track_id,
                "box_coords": [x1, y1, x2, y2], 
                "confidence": float(box.conf[0].cpu().numpy()),
                "center_point": list(center_pt), 
                "is_inside_court": bool(is_inside),
                "distance_to_center": float(dist_to_center), 
                "pose_keypoints": keypoints_xyc_list
            })
    except Exception as e: 
        print(f"!! Exception in detect_and_filter_players: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    
    all_candidates.sort(key=lambda p: (not p['is_inside_court'], p['distance_to_center']))
    return all_candidates[:4]

# --- 【v15 簡化與修正】---
# 移除所有繪圖和不必要的存檔邏輯
# 將原本的 main() 函數改造成一個簡單、專注於產生JSON的單一入口函數

def run_tracking_and_save_json(video_path, output_dir, use_tracking=False):
    """
    專門被 run_first_hit_analysis.py 呼叫的單一入口函數。
    它的唯一目標就是處理影片並產生一個 JSON 檔案。
    
    Args:
        video_path: 輸入影片路徑
        output_dir: 輸出目錄
        use_tracking: 是否啟用 ByteTrack 追蹤 (預設 False 保持向下相容)
    """
    import cv2 # 延遲導入
    
    # 步驟 1: 載入模型 (使用修正後的絕對路徑)
    player_model_path = os.path.join(MODELS_DIR, PLAYER_MODEL_NAME)
    ball_model_path = os.path.join(MODELS_DIR, BALL_MODEL_NAME)
    
    if not os.path.exists(player_model_path): raise FileNotFoundError(f"找不到選手模型: {player_model_path}")
    if not os.path.exists(ball_model_path): raise FileNotFoundError(f"找不到排球模型: {ball_model_path}")

    player_model = YOLO(player_model_path)
    ball_model = YOLO(ball_model_path)
    
    # 設定追蹤器配置檔路徑
    ball_tracker_config = os.path.join(PROJECT_ROOT, 'configs', 'bytetrack_ball.yaml') if use_tracking else None
    player_tracker_config = os.path.join(PROJECT_ROOT, 'configs', 'bytetrack_player.yaml') if use_tracking else None
    
    if use_tracking:
        print(f"[追蹤模式] 使用 ByteTrack 配置:", file=sys.stdout)
        print(f"  球體: {ball_tracker_config}", file=sys.stdout)
        print(f"  球員: {player_tracker_config}", file=sys.stdout)
    
    # 步驟 2: 載入影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"無法開啟影片檔案: {video_path}")
    
    # 步驟 3: 幀處理迴圈
    all_frames_data = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 為了簡化，我們暫時不使用 court_config.json 的過濾功能
        # 如果需要，可以將其作為參數傳遞進來
        balls = detect_ball(frame, ball_model, 0.3, [], use_tracking=use_tracking, tracker_config=ball_tracker_config)
        players = detect_and_filter_players(frame, player_model, 0.3, None, [], None, use_tracking=use_tracking, tracker_config=player_tracker_config)
        
        all_frames_data.append({"frame_id": frame_idx, "ball_detections": balls, "player_detections": players})
        frame_idx += 1
    
    cap.release()
    
    # 步驟 4: 正確地儲存 JSON 檔案
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    # 確保輸出目錄存在 (主腳本會提供完整的路徑)
    os.makedirs(output_dir, exist_ok=True)
    json_output_path = os.path.join(output_dir, f"{video_base_name}_all_frames_data_with_pose.json")

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_frames_data, f, indent=2)
    
    tracking_status = "with ByteTrack" if use_tracking else "without tracking"
    print(f"JSON saved to {json_output_path} ({tracking_status})", file=sys.stdout)


# --- 【v16 - 整合 ByteTrack 追蹤】---
# 支援向下相容:預設不啟用追蹤,透過 --use_tracking 參數啟用
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_tracking", action="store_true", help="啟用 ByteTrack 追蹤 (預設關閉以保持向下相容)")
    args = parser.parse_args()

    try:
        run_tracking_and_save_json(args.input, args.output_dir, use_tracking=args.use_tracking)
    except Exception as e:
        # 將任何錯誤都打印到標準錯誤流，以便主腳本捕捉
        print(f"FATAL ERROR in track_ball_and_player.py: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1) # 以非零代碼退出，明確表示失敗