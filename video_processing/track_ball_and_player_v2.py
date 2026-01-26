# video_processing/track_ball_and_player_v2.py
# -*- coding: utf-8 -*-
"""
改進版球員與球追蹤腳本

改進內容：
1. 整合 BallTracker - 支援遮擋預測
2. 輸出增強 - 包含追蹤狀態和預測標記
3. 效能優化選項 - 可跳幀偵測
4. 更好的錯誤處理

用法：
    python track_ball_and_player_v2.py --input video.mp4 --output_dir output/
    
    # 使用跳幀加速（每 2 幀偵測一次）
    python track_ball_and_player_v2.py --input video.mp4 --output_dir output/ --detection_interval 2
"""

import os
import sys

# 解決 OpenMP 重複載入問題
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import argparse
import traceback
import time
import numpy as np
from typing import Dict, List, Any, Optional

# 添加專案根目錄到 path
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO

# 嘗試導入核心模組
try:
    from core.ball_tracker import BallTracker, create_tracker_from_config
    HAS_BALL_TRACKER = True
except ImportError:
    print("[警告] 無法導入 BallTracker，將使用基礎追蹤模式", file=sys.stderr)
    HAS_BALL_TRACKER = False


# --- 模型路徑配置 ---
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
PLAYER_MODEL_NAME = 'yolov8s-pose.pt'
BALL_MODEL_NAME = 'ball_best.pt'


def detect_ball(frame, ball_model, conf_thresh: float, background_ball_zones: List[Dict]) -> List[Dict]:
    """
    偵測球的位置
    
    Args:
        frame: 影像幀
        ball_model: YOLO 模型
        conf_thresh: 信心度閾值
        background_ball_zones: 背景球過濾區域
        
    Returns:
        偵測結果列表
    """
    import cv2
    detected_balls = []
    
    try:
        results = ball_model(frame, conf=conf_thresh, classes=[0], verbose=False)
        
        if not results or not results[0].boxes:
            return detected_balls
        
        for box in results[0].boxes:
            if box.xyxy is None or len(box.xyxy) == 0:
                continue
            
            coords = box.xyxy[0].cpu().numpy()
            if len(coords) < 4:
                continue
            
            x1, y1, x2, y2 = map(int, coords)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # 檢查是否在背景區域
            is_in_background_zone = False
            if background_ball_zones:
                for zone in background_ball_zones:
                    if (zone.get('x1') is not None and 
                        zone['x1'] <= center_x <= zone['x2'] and 
                        zone['y1'] <= center_y <= zone['y2']):
                        is_in_background_zone = True
                        break
            
            detected_balls.append({
                "box_coords": [x1, y1, x2, y2],
                "confidence": float(box.conf[0].cpu().numpy()),
                "center_point": [center_x, center_y],
                "is_in_background_zone": is_in_background_zone
            })
    
    except Exception as e:
        print(f"!! Exception in detect_ball: {e}", file=sys.stderr)
    
    return detected_balls


def detect_and_filter_players(
    frame, 
    player_pose_model, 
    conf_thresh: float,
    court_boundary_np: Optional[np.ndarray],
    exclusion_zones_np: List[np.ndarray],
    court_center_xy: Optional[tuple]
) -> List[Dict]:
    """
    偵測並過濾球員
    
    Args:
        frame: 影像幀
        player_pose_model: YOLO Pose 模型
        conf_thresh: 信心度閾值
        court_boundary_np: 場地邊界多邊形
        exclusion_zones_np: 排除區域列表
        court_center_xy: 場地中心點
        
    Returns:
        球員偵測結果（最多 4 人）
    """
    import cv2
    all_candidates = []
    
    try:
        results = player_pose_model(frame, conf=conf_thresh, classes=[0], verbose=False)
        
        if not results or not results[0].boxes or not results[0].keypoints:
            return all_candidates
        
        for i in range(len(results[0].boxes)):
            box = results[0].boxes[i]
            kpts = results[0].keypoints[i]
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            center_pt = (float((x1 + x2) / 2), float((y1 + y2) / 2))
            
            # 排除區域檢查
            in_exclusion = False
            if exclusion_zones_np:
                for zone_np in exclusion_zones_np:
                    if cv2.pointPolygonTest(zone_np, center_pt, False) >= 0:
                        in_exclusion = True
                        break
            
            if in_exclusion:
                continue
            
            # 場內檢查
            is_inside = False
            if court_boundary_np is not None:
                is_inside = cv2.pointPolygonTest(court_boundary_np, center_pt, False) >= 0
            
            # 計算到中心的距離
            dist_to_center = float('inf')
            if court_center_xy:
                dist_to_center = np.linalg.norm(
                    np.array(center_pt) - np.array(court_center_xy)
                )
            
            # 提取關鍵點
            keypoints_xyc_list = []
            if kpts.xy is not None and kpts.conf is not None:
                kpts_xy = kpts.xy[0].cpu().numpy()
                kpts_conf = kpts.conf[0].cpu().numpy()
                for kp_idx in range(kpts_xy.shape[0]):
                    keypoints_xyc_list.append([
                        float(kpts_xy[kp_idx, 0]),
                        float(kpts_xy[kp_idx, 1]),
                        float(kpts_conf[kp_idx])
                    ])
            
            all_candidates.append({
                "box_coords": [x1, y1, x2, y2],
                "confidence": float(box.conf[0].cpu().numpy()),
                "center_point": list(center_pt),
                "is_inside_court": bool(is_inside),
                "distance_to_center": float(dist_to_center),
                "pose_keypoints": keypoints_xyc_list
            })
    
    except Exception as e:
        print(f"!! Exception in detect_and_filter_players: {e}", file=sys.stderr)
    
    # 排序：場內優先，然後按距離
    all_candidates.sort(key=lambda p: (not p['is_inside_court'], p['distance_to_center']))
    
    # 保留最多 6 人（4 球員 + 可能的裁判等），避免漏掉被遮擋的球員
    return all_candidates[:6]


def run_tracking_v2(
    video_path: str,
    output_dir: str,
    court_config: Dict = None,
    ball_conf_thresh: float = 0.3,
    player_conf_thresh: float = 0.15,  # 降低閾值以偵測被遮擋的球員
    detection_interval: int = 1,
    use_ball_tracker: bool = True,
    max_occlusion_frames: int = 15,
    verbose: bool = True
) -> str:
    """
    執行改進版追蹤
    
    Args:
        video_path: 影片路徑
        output_dir: 輸出目錄
        court_config: 場地配置（包含邊界、排除區域等）
        ball_conf_thresh: 球偵測信心度閾值
        player_conf_thresh: 球員偵測信心度閾值
        detection_interval: 偵測間隔（每 N 幀偵測一次）
        use_ball_tracker: 是否使用 BallTracker
        max_occlusion_frames: 最大遮擋幀數
        verbose: 是否輸出詳細日誌
        
    Returns:
        輸出 JSON 檔案路徑
    """
    import cv2
    
    # --- 載入模型 ---
    player_model_path = os.path.join(MODELS_DIR, PLAYER_MODEL_NAME)
    ball_model_path = os.path.join(MODELS_DIR, BALL_MODEL_NAME)
    
    if not os.path.exists(player_model_path):
        raise FileNotFoundError(f"找不到選手模型: {player_model_path}")
    if not os.path.exists(ball_model_path):
        raise FileNotFoundError(f"找不到排球模型: {ball_model_path}")
    
    if verbose:
        print(f"[追蹤] 載入模型...")
    
    player_model = YOLO(player_model_path)
    ball_model = YOLO(ball_model_path)
    
    # --- 解析場地配置 ---
    court_boundary_np = None
    exclusion_zones_np = []
    court_center_xy = None
    background_ball_zones = []
    
    if court_config:
        # 場地邊界
        boundary = court_config.get('court_boundary_polygon')
        if boundary and len(boundary) == 4:
            court_boundary_np = np.array(boundary, dtype=np.float32)
            # 計算場地中心
            court_center_xy = tuple(np.mean(court_boundary_np, axis=0))
        
        # 排除區域
        for zone in court_config.get('exclusion_zones', []):
            if zone.get('polygon'):
                exclusion_zones_np.append(np.array(zone['polygon'], dtype=np.float32))
        
        # 背景球區域
        background_ball_zones = court_config.get('background_ball_zones', [])
    
    # --- 初始化追蹤器 ---
    ball_tracker = None
    if use_ball_tracker and HAS_BALL_TRACKER:
        ball_tracker = create_tracker_from_config({
            'max_occlusion_frames': max_occlusion_frames
        })
        if verbose:
            print(f"[追蹤] 使用 BallTracker（最大遮擋幀數: {max_occlusion_frames}）")
    else:
        if verbose:
            print(f"[追蹤] 使用基礎偵測模式")
    
    # --- 開啟影片 ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"無法開啟影片檔案: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if verbose:
        print(f"[追蹤] 影片: {os.path.basename(video_path)}")
        print(f"[追蹤] 總幀數: {total_frames}, FPS: {fps:.1f}")
        print(f"[追蹤] 偵測間隔: 每 {detection_interval} 幀")
    
    # --- 幀處理迴圈 ---
    all_frames_data = []
    frame_idx = 0
    last_ball_detection = None
    last_player_detections = []
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 決定是否執行偵測
        should_detect = (frame_idx % detection_interval == 0)
        
        if should_detect:
            # 執行偵測
            ball_detections = detect_ball(
                frame, ball_model, ball_conf_thresh, background_ball_zones
            )
            player_detections = detect_and_filter_players(
                frame, player_model, player_conf_thresh,
                court_boundary_np, exclusion_zones_np, court_center_xy
            )
            
            last_ball_detection = ball_detections
            last_player_detections = player_detections
        else:
            # 使用上一次的偵測結果
            ball_detections = last_ball_detection or []
            player_detections = last_player_detections
        
        # --- 球追蹤處理 ---
        tracking_result = None
        if ball_tracker is not None:
            # 選擇最佳球偵測
            valid_balls = [
                b for b in ball_detections
                if not b.get('is_in_background_zone', False)
            ]
            best_ball = None
            if valid_balls:
                best_ball = max(valid_balls, key=lambda b: b.get('confidence', 0))
            
            # 更新追蹤器
            tracking_result = ball_tracker.update(best_ball, frame_idx)
            
            # 如果追蹤器提供了預測位置，添加到偵測結果
            if tracking_result['position'] and tracking_result['is_predicted']:
                # 添加預測的球位置
                predicted_ball = {
                    "box_coords": [
                        int(tracking_result['position'][0] - 10),
                        int(tracking_result['position'][1] - 10),
                        int(tracking_result['position'][0] + 10),
                        int(tracking_result['position'][1] + 10)
                    ],
                    "confidence": tracking_result['confidence'],
                    "center_point": [int(tracking_result['position'][0]), 
                                     int(tracking_result['position'][1])],
                    "is_in_background_zone": False,
                    "is_predicted": True,
                    "tracker_state": tracking_result['state']
                }
                ball_detections = [predicted_ball] + ball_detections
        
        # --- 組裝幀數據 ---
        frame_data = {
            "frame_id": frame_idx,
            "ball_detections": ball_detections,
            "player_detections": player_detections
        }
        
        # 添加追蹤信息
        if tracking_result:
            frame_data["ball_tracking"] = {
                "state": tracking_result['state'],
                "is_predicted": tracking_result['is_predicted'],
                "confidence": tracking_result['confidence'],
                "velocity": tracking_result['velocity'],
                "speed": tracking_result['speed'],
                "occlusion_frames": tracking_result['occlusion_frames']
            }
        
        all_frames_data.append(frame_data)
        frame_idx += 1
        
        # 進度顯示
        if verbose and frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed
            eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
            print(f"[追蹤] 進度: {frame_idx}/{total_frames} "
                  f"({100*frame_idx/total_frames:.1f}%) "
                  f"速度: {fps_actual:.1f} fps, ETA: {eta:.0f}s")
    
    cap.release()
    
    # --- 統計追蹤結果 ---
    if ball_tracker and verbose:
        trajectory = ball_tracker.get_trajectory()
        predicted_count = sum(1 for t in trajectory if t[3])
        total_count = len(trajectory)
        print(f"[追蹤] 完成！總幀數: {total_count}, 預測幀數: {predicted_count} "
              f"({100*predicted_count/total_count:.1f}%)" if total_count > 0 else "[追蹤] 完成！")
    
    # --- 儲存 JSON ---
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    json_output_path = os.path.join(output_dir, f"{video_base_name}_all_frames_data_with_pose.json")
    
    # 添加元數據
    output_data = {
        "metadata": {
            "video_path": video_path,
            "total_frames": frame_idx,
            "fps": fps,
            "detection_interval": detection_interval,
            "use_ball_tracker": use_ball_tracker and HAS_BALL_TRACKER,
            "processing_time": time.time() - start_time
        },
        "frames": all_frames_data
    }
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    if verbose:
        print(f"[追蹤] JSON 已儲存: {json_output_path}")
    
    return json_output_path


# === 為了向後兼容，保留原有的函數簽名 ===
def run_tracking_and_save_json(video_path: str, output_dir: str):
    """
    向後兼容的接口（供 run_analysis_all_in_one.py 呼叫）
    """
    return run_tracking_v2(
        video_path=video_path,
        output_dir=output_dir,
        use_ball_tracker=HAS_BALL_TRACKER,
        verbose=False
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="改進版球員與球追蹤腳本")
    parser.add_argument("--input", type=str, required=True, help="輸入影片路徑")
    parser.add_argument("--output_dir", type=str, required=True, help="輸出目錄")
    parser.add_argument("--court_config", type=str, help="場地配置 JSON 檔案")
    parser.add_argument("--detection_interval", type=int, default=1, 
                        help="偵測間隔（每 N 幀偵測一次，預設 1）")
    parser.add_argument("--max_occlusion", type=int, default=15,
                        help="最大遮擋幀數（預設 15）")
    parser.add_argument("--no_tracker", action="store_true",
                        help="停用 BallTracker（使用基礎偵測模式）")
    parser.add_argument("--quiet", action="store_true", help="安靜模式")
    
    args = parser.parse_args()
    
    # 載入場地配置
    court_config = None
    if args.court_config and os.path.exists(args.court_config):
        with open(args.court_config, 'r', encoding='utf-8') as f:
            court_config = json.load(f)
    
    try:
        output_path = run_tracking_v2(
            video_path=args.input,
            output_dir=args.output_dir,
            court_config=court_config,
            detection_interval=args.detection_interval,
            use_ball_tracker=not args.no_tracker,
            max_occlusion_frames=args.max_occlusion,
            verbose=not args.quiet
        )
        print(f"完成！輸出: {output_path}")
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)