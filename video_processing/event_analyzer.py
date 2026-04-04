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

# --- 保留您原始的偵測邏輯，不做任何修改 ---
def detect_ball(frame, ball_model, conf_thresh, background_ball_zones):
    # (此函數與您提供的版本完全相同)
    import cv2
    detected_balls = []
    try:
        results = ball_model(frame, conf=conf_thresh, classes=[0], verbose=False)
        if not results or not results[0].boxes: return detected_balls
        for box in results[0].boxes:
            if box.xyxy is None or len(box.xyxy) == 0: continue
            coords = box.xyxy[0].cpu().numpy()
            if len(coords) < 4: continue
            x1, y1, x2, y2 = map(int, coords)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            is_in_background_zone = False
            if background_ball_zones:
                for zone in background_ball_zones:
                    if zone.get('x1') is not None and zone['x1'] <= center_x <= zone['x2'] and zone['y1'] <= center_y <= zone['y2']:
                        is_in_background_zone = True; break
            detected_balls.append({
                "box_coords": [x1, y1, x2, y2], "confidence": float(box.conf[0].cpu().numpy()),
                "center_point": [center_x, center_y], "is_in_background_zone": is_in_background_zone
            })
    except Exception as e: print(f"!! Exception in detect_ball: {e}", file=sys.stderr)
    return detected_balls

def detect_and_filter_players(frame, player_pose_model, conf_thresh, court_boundary_np, exclusion_zones_np, court_center_xy):
    # (此函數與您提供的版本完全相同)
    import cv2
    all_candidates = []
    try:
        results = player_pose_model(frame, conf=conf_thresh, classes=[0], verbose=False)
        if not results or not results[0].boxes or not results[0].keypoints: return all_candidates
        for i in range(len(results[0].boxes)):
            box, kpts = results[0].boxes[i], results[0].keypoints[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            center_pt = (float((x1+x2)/2), float((y1+y2)/2))
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
                for kp_idx in range(kpts_xy.shape[0]): keypoints_xyc_list.append([float(kpts_xy[kp_idx, 0]), float(kpts_xy[kp_idx, 1]), float(kpts_conf[kp_idx])])
            all_candidates.append({
                "box_coords": [x1, y1, x2, y2], "confidence": float(box.conf[0].cpu().numpy()),
                "center_point": list(center_pt), "is_inside_court": bool(is_inside),
                "distance_to_center": float(dist_to_center), "pose_keypoints": keypoints_xyc_list
            })
    except Exception as e: print(f"!! Exception in detect_and_filter_players: {e}", file=sys.stderr)
    all_candidates.sort(key=lambda p: (not p['is_inside_court'], p['distance_to_center']))
    return all_candidates[:4]

# --- 【v15 簡化與修正】---
# 移除所有繪圖和不必要的存檔邏輯
# 將原本的 main() 函數改造成一個簡單、專注於產生JSON的單一入口函數

def run_tracking_and_save_json(video_path, output_dir):
    """
    專門被 run_first_hit_analysis.py 呼叫的單一入口函數。
    它的唯一目標就是處理影片並產生一個 JSON 檔案。
    """
    import cv2 # 延遲導入
    
    # 步驟 1: 載入模型 (使用修正後的絕對路徑)
    player_model_path = os.path.join(MODELS_DIR, PLAYER_MODEL_NAME)
    ball_model_path = os.path.join(MODELS_DIR, BALL_MODEL_NAME)
    
    if not os.path.exists(player_model_path): raise FileNotFoundError(f"找不到選手模型: {player_model_path}")
    if not os.path.exists(ball_model_path): raise FileNotFoundError(f"找不到排球模型: {ball_model_path}")

    player_model = YOLO(player_model_path)
    ball_model = YOLO(ball_model_path)
    
    # 步驟 2: 載入影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"無法開啟影片檔案: {video_path}")
    
    # 步驟 3: 幀處理迴圈 (使用您原始的偵測邏輯)
    all_frames_data = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 為了簡化，我們暫時不使用 court_config.json 的過濾功能
        # 如果需要，可以將其作為參數傳遞進來
        balls = detect_ball(frame, ball_model, 0.3, [])
        players = detect_and_filter_players(frame, player_model, 0.3, None, [], None)
        
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
    
    print(f"JSON saved to {json_output_path}", file=sys.stdout)


# --- 【v15 簡化與修正】---
# 將原本複雜的 main() 函數替換成現在這個更簡單的版本
# 它只負責解析從主腳本傳來的參數，並呼叫上面的核心函數
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    try:
        run_tracking_and_save_json(args.input, args.output_dir)
    except Exception as e:
        # 將任何錯誤都打印到標準錯誤流，以便主腳本捕捉
        print(f"FATAL ERROR in track_ball_and_player.py: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1) # 以非零代碼退出，明確表示失敗