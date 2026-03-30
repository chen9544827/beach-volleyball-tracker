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
    print("[WARNING] Cannot import BallTracker, using basic tracking mode", file=sys.stderr)
    HAS_BALL_TRACKER = False

try:
    from core.sahi_pose_detector import SahiPoseDetector
    HAS_SAHI = True
except ImportError:
    HAS_SAHI = False

try:
    from core.static_ball_filter import StaticBallFilter
    HAS_STATIC_FILTER = True
except ImportError:
    HAS_STATIC_FILTER = False

try:
    from core.auto_court_estimator import AutoCourtEstimator
    HAS_AUTO_COURT = True
except ImportError:
    HAS_AUTO_COURT = False


# --- 模型路徑配置 ---
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
PLAYER_MODEL_NAME = 'yolov8m-pose.pt'
PLAYER_DET_MODEL_NAME = 'yolo11m.pt'
BALL_MODEL_NAME = 'ball_best.pt'
DEFAULT_IMGSZ = 1280


def detect_ball(frame, ball_model, conf_thresh: float, background_ball_zones: List[Dict],
                resolution_scale: float = 1.0, imgsz: int = None) -> List[Dict]:
    """
    偵測球的位置

    Args:
        frame: 影像幀
        ball_model: YOLO 模型
        conf_thresh: 信心度閾值
        background_ball_zones: 背景球過濾區域
        resolution_scale: 解析度縮放因子 (height / 720)

    Returns:
        偵測結果列表
    """
    import cv2
    detected_balls = []

    # A1: Ball size limits (px @720p, scaled by resolution)
    min_ball_size = 8 * resolution_scale
    max_ball_size = 50 * resolution_scale

    try:
        results = ball_model(frame, conf=conf_thresh, classes=[0], verbose=False, imgsz=imgsz or DEFAULT_IMGSZ)

        if not results or not results[0].boxes:
            return detected_balls

        for box in results[0].boxes:
            if box.xyxy is None or len(box.xyxy) == 0:
                continue

            coords = box.xyxy[0].cpu().numpy()
            if len(coords) < 4:
                continue

            x1, y1, x2, y2 = map(int, coords)

            # A1: Size filter - reject too small or too large detections
            ball_w = x2 - x1
            ball_h = y2 - y1
            ball_size = max(ball_w, ball_h)
            if ball_size < min_ball_size or ball_size > max_ball_size:
                continue

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
    court_center_xy: Optional[tuple],
    frame_size: Optional[tuple] = None,
    iou_thresh: float = 0.6,
    imgsz: int = None,
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
        frame_size: (width, height) 用於無 court_config 時計算畫面中心

    Returns:
        球員偵測結果（最多 6 人）
    """
    import cv2
    all_candidates = []

    has_court_config = court_boundary_np is not None

    # 決定中心點：有 court_config 用場地中心，否則用畫面中心
    effective_center = court_center_xy
    if effective_center is None and frame_size is not None:
        effective_center = (frame_size[0] / 2, frame_size[1] / 2 + frame_size[1] * 0.1)

    try:
        results = player_pose_model(frame, conf=0.15, classes=[0], verbose=False, imgsz=imgsz or DEFAULT_IMGSZ, max_det=100, iou=iou_thresh)

        if not results or not results[0].boxes or not results[0].keypoints:
            return all_candidates

        for i in range(len(results[0].boxes)):
            box = results[0].boxes[i]
            kpts = results[0].keypoints[i]

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            center_pt = (float((x1 + x2) / 2), float((y1 + y2) / 2))

            # 排除區域檢查（僅在有 court_config 時執行）
            if has_court_config and exclusion_zones_np:
                in_exclusion = False
                for zone_np in exclusion_zones_np:
                    if cv2.pointPolygonTest(zone_np, center_pt, False) >= 0:
                        in_exclusion = True
                        break
                if in_exclusion:
                    continue

            # 場內檢查（僅在有 court_config 時執行）
            is_inside = False
            if has_court_config:
                is_inside = cv2.pointPolygonTest(court_boundary_np, center_pt, False) >= 0

            # 計算到中心的距離
            dist_to_center = float('inf')
            if effective_center:
                dist_to_center = np.linalg.norm(
                    np.array(center_pt) - np.array(effective_center)
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

    if has_court_config:
        # 有 court_config：場內優先，然後按距離
        all_candidates.sort(key=lambda p: (not p['is_inside_court'], p['distance_to_center']))
    else:
        # 無 court_config：僅按距離畫面中心排序
        all_candidates.sort(key=lambda p: p['distance_to_center'])

    # 取距離中心最近的 6 人
    return all_candidates[:6]


def _compute_iou(box1, box2):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter)


def _get_pose_for_crop(frame, pose_model, box, scale_factor=2.0, pad=15, imgsz: int = None):
    """
    裁切球員邊界框區域，放大後跑 pose model 取得關鍵點。

    Args:
        frame: 原始影格
        pose_model: YOLO Pose 模型
        box: [x1, y1, x2, y2] 球員框
        scale_factor: 裁切區域放大倍率
        pad: 邊界填充像素

    Returns:
        keypoints list [[x, y, conf], ...] 已映射回原始座標，失敗時返回空 list
    """
    import cv2
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad)
    y2p = min(h, y2 + pad)
    crop = frame[y1p:y2p, x1p:x2p]
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return []
    new_w = int(crop.shape[1] * scale_factor)
    new_h = int(crop.shape[0] * scale_factor)
    crop_up = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    try:
        results = pose_model(crop_up, conf=0.03, classes=[0], verbose=False,
                             imgsz=imgsz or DEFAULT_IMGSZ, max_det=5)
        if (results and results[0].boxes is not None and
                results[0].keypoints is not None and len(results[0].boxes) > 0):
            kpts = results[0].keypoints[0]
            if kpts.xy is not None and kpts.conf is not None:
                kpts_xy = kpts.xy[0].cpu().numpy()
                kpts_conf = kpts.conf[0].cpu().numpy()
                keypoints = []
                for kp_idx in range(kpts_xy.shape[0]):
                    orig_x = float(kpts_xy[kp_idx, 0] / scale_factor) + x1p
                    orig_y = float(kpts_xy[kp_idx, 1] / scale_factor) + y1p
                    keypoints.append([orig_x, orig_y, float(kpts_conf[kp_idx])])
                return keypoints
    except Exception:
        pass
    return []


def detect_supplementary_players(
    frame,
    det_model,
    pose_detections: List[Dict],
    iou_thresh: float = 0.4,
    pose_model=None,
    imgsz: int = None,
) -> List[Dict]:
    """
    Use YOLO detection model to find players missed by pose model.
    Returns only detections that don't overlap with existing pose detections.
    If pose_model is provided, runs pose estimation on each found player crop
    to obtain keypoints.
    """
    try:
        results = det_model(frame, conf=0.15, classes=[0], verbose=False,
                            imgsz=imgsz or DEFAULT_IMGSZ, max_det=100, iou=0.6)
    except Exception as e:
        print(f"!! Exception in detect_supplementary_players: {e}", file=sys.stderr)
        return []

    if not results or not results[0].boxes:
        return []

    det_results = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        det_results.append({
            "box_coords": [x1, y1, x2, y2],
            "confidence": float(box.conf[0].cpu().numpy()),
            "center_point": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
            "pose_keypoints": [],
            "detection_source": "det_model",
        })

    # Filter out detections that overlap with pose detections
    supplementary = []
    for det in det_results:
        matched = False
        for pose_det in pose_detections:
            if _compute_iou(det["box_coords"], pose_det["box_coords"]) > iou_thresh:
                matched = True
                break
        if not matched:
            supplementary.append(det)

    # 補跑 pose model 取得關鍵點（裁切 + 放大 2x）
    if pose_model is not None:
        for det in supplementary:
            kps = _get_pose_for_crop(frame, pose_model, det['box_coords'],
                                     scale_factor=4.0, imgsz=640)
            if kps:
                det['pose_keypoints'] = kps
                det['detection_source'] = 'det_model_with_pose'

    return supplementary


def detect_far_side_players(
    frame,
    pose_model,
    det_model,
    existing_detections: List[Dict],
    net_y: float,
    scale_factor: float = 2.0,
    iou_thresh: float = 0.6,
    imgsz: int = None,
) -> List[Dict]:
    """
    Far-side retry: crop the region above net_y, upscale, and re-detect
    with very low confidence to find players occluded by the net.

    Only triggered when fewer than 2 players are detected above net_y.

    Args:
        frame: Full frame image.
        pose_model: YOLO Pose model.
        det_model: YOLO Detection model (or None).
        existing_detections: Already detected players.
        net_y: Y coordinate of the net.
        scale_factor: How much to upscale the cropped region.

    Returns:
        List of new detections (coordinates mapped back to original frame).
    """
    import cv2

    # Count existing players in far side (above net_y)
    far_players = [d for d in existing_detections
                   if d["center_point"][1] < net_y]
    if len(far_players) >= 2:
        return []

    # Crop far-side region: from top to net_y + margin
    h, w = frame.shape[:2]
    margin = int((net_y - 0) * 0.15)  # 15% below net for partially visible players
    crop_bottom = min(int(net_y + margin), h)
    crop = frame[0:crop_bottom, :]

    if crop.shape[0] < 20:
        return []

    # Upscale the crop
    new_w = int(w * scale_factor)
    new_h = int(crop.shape[0] * scale_factor)
    crop_upscaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    new_detections = []

    # Try pose model on upscaled crop with very low conf
    try:
        results = pose_model(crop_upscaled, conf=0.03, classes=[0], verbose=False,
                             imgsz=imgsz or DEFAULT_IMGSZ, max_det=50, iou=iou_thresh)
        if results and results[0].boxes is not None and results[0].keypoints is not None:
            for i in range(len(results[0].boxes)):
                box = results[0].boxes[i]
                kpts = results[0].keypoints[i]

                # Map coordinates back to original frame
                x1 = int(box.xyxy[0][0].cpu().numpy() / scale_factor)
                y1 = int(box.xyxy[0][1].cpu().numpy() / scale_factor)
                x2 = int(box.xyxy[0][2].cpu().numpy() / scale_factor)
                y2 = int(box.xyxy[0][3].cpu().numpy() / scale_factor)
                center_pt = [float((x1 + x2) / 2), float((y1 + y2) / 2)]

                # Map keypoints back
                keypoints_xyc_list = []
                if kpts.xy is not None and kpts.conf is not None:
                    kpts_xy = kpts.xy[0].cpu().numpy()
                    kpts_conf = kpts.conf[0].cpu().numpy()
                    for kp_idx in range(kpts_xy.shape[0]):
                        keypoints_xyc_list.append([
                            float(kpts_xy[kp_idx, 0] / scale_factor),
                            float(kpts_xy[kp_idx, 1] / scale_factor),
                            float(kpts_conf[kp_idx])
                        ])

                new_detections.append({
                    "box_coords": [x1, y1, x2, y2],
                    "confidence": float(box.conf[0].cpu().numpy()),
                    "center_point": center_pt,
                    "pose_keypoints": keypoints_xyc_list,
                    "detection_source": "far_side_retry",
                })
    except Exception:
        pass

    # Also try det model if available
    if det_model is not None:
        try:
            results = det_model(crop_upscaled, conf=0.03, classes=[0], verbose=False,
                                imgsz=imgsz or DEFAULT_IMGSZ, max_det=50, iou=iou_thresh)
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1 = int(box.xyxy[0][0].cpu().numpy() / scale_factor)
                    y1 = int(box.xyxy[0][1].cpu().numpy() / scale_factor)
                    x2 = int(box.xyxy[0][2].cpu().numpy() / scale_factor)
                    y2 = int(box.xyxy[0][3].cpu().numpy() / scale_factor)

                    new_detections.append({
                        "box_coords": [x1, y1, x2, y2],
                        "confidence": float(box.conf[0].cpu().numpy()),
                        "center_point": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                        "pose_keypoints": [],
                        "detection_source": "far_side_retry_det",
                    })
        except Exception:
            pass

    # 對 det_model 找到但無關鍵點的結果，補跑 pose model（裁切原始幀區域）
    for nd in new_detections:
        if not nd.get('pose_keypoints'):
            kps = _get_pose_for_crop(frame, pose_model, nd['box_coords'],
                                     scale_factor=4.0, imgsz=640)
            if kps:
                nd['pose_keypoints'] = kps
                nd['detection_source'] = nd['detection_source'] + '_with_pose'

    # Deduplicate: remove new detections that overlap with existing ones
    filtered = []
    for nd in new_detections:
        overlap = False
        for ed in existing_detections:
            if _compute_iou(nd["box_coords"], ed["box_coords"]) > 0.3:
                overlap = True
                break
        if not overlap:
            filtered.append(nd)

    return filtered


def _post_filter_players(
    raw_detections: List[Dict],
    court_boundary_np: Optional[np.ndarray],
    exclusion_zones_np: List[np.ndarray],
    court_center_xy: Optional[tuple],
    frame_size: Optional[tuple] = None,
) -> List[Dict]:
    """Apply exclusion zones, court boundary check, and sorting to SAHI detections.

    Mirrors the filtering logic of detect_and_filter_players but operates on
    pre-detected results from SahiPoseDetector.
    """
    import cv2

    has_court_config = court_boundary_np is not None

    # 決定中心點：有 court_config 用場地中心，否則用畫面中心
    effective_center = court_center_xy
    if effective_center is None and frame_size is not None:
        effective_center = (frame_size[0] / 2, frame_size[1] / 2 + frame_size[1] * 0.1)

    filtered = []
    for det in raw_detections:
        center_pt = tuple(det["center_point"])

        # Exclusion zone check (only with court_config)
        if has_court_config and exclusion_zones_np:
            in_exclusion = False
            for zone_np in exclusion_zones_np:
                if cv2.pointPolygonTest(zone_np, center_pt, False) >= 0:
                    in_exclusion = True
                    break
            if in_exclusion:
                continue

        # Court boundary check (only with court_config)
        is_inside = False
        if has_court_config:
            is_inside = cv2.pointPolygonTest(court_boundary_np, center_pt, False) >= 0

        # Distance to center
        dist_to_center = float('inf')
        if effective_center:
            dist_to_center = float(np.linalg.norm(
                np.array(center_pt) - np.array(effective_center)
            ))

        det["is_inside_court"] = bool(is_inside)
        det["distance_to_center"] = dist_to_center
        filtered.append(det)

    if has_court_config:
        # Sort: inside court first, then by distance
        filtered.sort(key=lambda p: (not p['is_inside_court'], p['distance_to_center']))
    else:
        # No court_config: sort by distance to frame center only
        filtered.sort(key=lambda p: p['distance_to_center'])

    return filtered[:6]


def run_tracking_v2(
    video_path: str,
    output_dir: str,
    court_config: Dict = None,
    ball_conf_thresh: float = 0.45,
    player_conf_thresh: float = 0.01,  # 降低閾值以偵測被遮擋的球員和遠距離球員
    detection_interval: int = 1,
    use_ball_tracker: bool = True,
    max_occlusion_frames: int = 15,
    use_sahi: bool = False,
    sahi_slice_size: int = 512,
    sahi_overlap: float = 0.2,
    player_model_name: str = None,
    player_iou: float = 0.6,
    imgsz: int = None,
    verbose: bool = True,
    early_stop_after_serve: bool = True,
    early_stop_buffer_frames: int = 150,
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
        use_sahi: 是否使用 SAHI 切片偵測球員（改善遠端偵測）
        sahi_slice_size: SAHI 切片大小（像素）
        sahi_overlap: SAHI 切片重疊比例
        verbose: 是否輸出詳細日誌
        
    Returns:
        輸出 JSON 檔案路徑
    """
    import cv2
    
    # --- 載入模型 ---
    _player_name = player_model_name if player_model_name else PLAYER_MODEL_NAME
    player_model_path = os.path.join(MODELS_DIR, _player_name)
    ball_model_path = os.path.join(MODELS_DIR, BALL_MODEL_NAME)

    if not os.path.exists(player_model_path):
        raise FileNotFoundError(f"找不到選手模型: {player_model_path}")
    if not os.path.exists(ball_model_path):
        raise FileNotFoundError(f"找不到排球模型: {ball_model_path}")

    if verbose:
        print(f"[追蹤] 載入模型... (pose={_player_name}, iou={player_iou})")
    
    player_model = YOLO(player_model_path)
    ball_model = YOLO(ball_model_path)

    # Load supplementary detection model for players missed by pose model
    player_det_model = None
    player_det_model_path = os.path.join(MODELS_DIR, PLAYER_DET_MODEL_NAME)
    if os.path.exists(player_det_model_path):
        player_det_model = YOLO(player_det_model_path)
        if verbose:
            print(f"[DET] Supplementary detection model loaded: {PLAYER_DET_MODEL_NAME}")
    else:
        if verbose:
            print(f"[DET] {PLAYER_DET_MODEL_NAME} not found, using pose model only")
    
    # --- 解析場地配置 ---
    court_boundary_np = None
    exclusion_zones_np = []
    court_center_xy = None
    background_ball_zones = []
    net_y = None
    
    if court_config:
        # 場地邊界
        boundary = court_config.get('court_boundary_polygon')
        if boundary and len(boundary) == 4:
            court_boundary_np = np.array(boundary, dtype=np.float32)
            # 計算場地中心（向下偏移 10% 場地高度，補償透視效果）
            raw_center = np.mean(court_boundary_np, axis=0)
            court_h = max(court_boundary_np[:, 1]) - min(court_boundary_np[:, 1])
            court_center_xy = (float(raw_center[0]), float(raw_center[1] + court_h * 0.1))
        
        # 排除區域
        for zone in court_config.get('exclusion_zones', []):
            if zone.get('polygon'):
                exclusion_zones_np.append(np.array(zone['polygon'], dtype=np.float32))
        
        # 背景球區域
        background_ball_zones = court_config.get('background_ball_zones', [])

        # 網子 Y 座標（用於遠端二次偵測）
        net_y = court_config.get('net_y')
    
    # --- 初始化 SAHI Pose Detector ---
    sahi_detector = None
    if use_sahi and HAS_SAHI:
        sahi_detector = SahiPoseDetector(
            model=player_model,
            slice_size=sahi_slice_size,
            overlap_ratio=sahi_overlap,
            conf_thresh=player_conf_thresh,
        )
        if verbose:
            print(f"[SAHI] Enabled (slice={sahi_slice_size}, overlap={sahi_overlap})")
    elif use_sahi and not HAS_SAHI:
        if verbose:
            print(f"[SAHI] Requested but module not available, falling back to standard")

    # --- 初始化追蹤器 ---
    ball_tracker = None
    if use_ball_tracker and HAS_BALL_TRACKER:
        ball_tracker = create_tracker_from_config({
            'max_occlusion_frames': max_occlusion_frames
        })
        if verbose:
            print(f"[追蹤] BallTracker enabled (max_occlusion={max_occlusion_frames})")
    else:
        if verbose:
            print(f"[追蹤] Basic detection mode")
    
    # --- 開啟影片 ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"無法開啟影片檔案: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- Resolution scale for pixel thresholds ---
    resolution_scale = video_height / 720.0 if video_height > 0 else 1.0

    # --- imgsz: inference resolution (default 1280, can be set higher for small far-side players) ---
    _imgsz = imgsz or DEFAULT_IMGSZ

    # --- 畫面中心 fallback（無 court_config 時使用）---
    frame_size = (video_width, video_height)
    if court_center_xy is None:
        # 向下偏移 10% 畫面高度，補償透視效果（近端較大）
        court_center_xy = (video_width / 2, video_height / 2 + video_height * 0.1)
    if net_y is None:
        # 估計網子位置：標準側視角約在畫面 40% 高度
        net_y = video_height * 0.4

    # --- 初始化靜態球濾波器 ---
    static_filter = None
    if HAS_STATIC_FILTER:
        static_filter = StaticBallFilter(resolution_scale=resolution_scale)
        if verbose:
            print(f"[StaticFilter] Enabled (cell={static_filter.cell_size}px)")

    # --- 初始化自動場地估算器（無 court_config 時使用）---
    auto_court = None
    if court_boundary_np is None and HAS_AUTO_COURT:
        auto_court = AutoCourtEstimator(
            frame_width=video_width,
            frame_height=video_height,
            min_confidence=0.5,
            buffer_size=50,
            min_samples=10,
        )
        if verbose:
            print(f"[AutoCourt] Enabled (no court_config, using near-player estimation)")

    if verbose:
        print(f"[追蹤] Video: {os.path.basename(video_path)}")
        print(f"[追蹤] Frames: {total_frames}, FPS: {fps:.1f}")
        print(f"[追蹤] Detection interval: every {detection_interval} frame(s)")
        print(f"[追蹤] Inference imgsz: {_imgsz}")
    
    # --- Early Stop：初始化發球狀態機（偵測到擊球後再處理 buffer_frames 幀即停止）---
    _serve_detector_es = None
    _es_hit_found = False
    _es_countdown = -1
    _es_prev_ball = None
    if early_stop_after_serve:
        try:
            from core.serve_detector import ServeDetector
            _es_cfg = {'image_height': video_height}
            if net_y is not None:
                _es_cfg['net_y'] = net_y
            _serve_detector_es = ServeDetector(_es_cfg)
            if verbose:
                print(f"[早停] 已啟用，偵測到擊球後再處理 {early_stop_buffer_frames} 幀即停止")
        except Exception as _e:
            if verbose:
                print(f"[早停] 無法初始化 ServeDetector，停用早停: {_e}")

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
                frame, ball_model, ball_conf_thresh, background_ball_zones,
                resolution_scale=resolution_scale, imgsz=_imgsz
            )

            # Static ball filter: update history and remove static detections
            if static_filter is not None:
                static_filter.update(frame_idx, ball_detections)
                ball_detections = static_filter.filter(ball_detections)

            # Player detection: SAHI or standard
            if sahi_detector is not None:
                raw_players = sahi_detector.detect(frame, conf_thresh=player_conf_thresh)
                # Apply exclusion zones, court boundary, and sorting
                player_detections = _post_filter_players(
                    raw_players, court_boundary_np, exclusion_zones_np, court_center_xy,
                    frame_size=frame_size
                )
            else:
                player_detections = detect_and_filter_players(
                    frame, player_model, player_conf_thresh,
                    court_boundary_np, exclusion_zones_np, court_center_xy,
                    frame_size=frame_size,
                    iou_thresh=player_iou,
                    imgsz=_imgsz,
                )

            # Supplementary detection: find players missed by pose model
            if player_det_model is not None:
                supplementary = detect_supplementary_players(
                    frame, player_det_model, player_detections,
                    pose_model=player_model,
                    imgsz=_imgsz,
                )
                if supplementary:
                    # Apply same filtering as pose detections
                    supplementary = _post_filter_players(
                        supplementary, court_boundary_np, exclusion_zones_np, court_center_xy,
                        frame_size=frame_size
                    )
                    player_detections.extend(supplementary)
                    # 合併後重新排序並截取前 6 人
                    if court_boundary_np is not None:
                        player_detections.sort(key=lambda p: (not p.get('is_inside_court', False), p.get('distance_to_center', float('inf'))))
                    else:
                        player_detections.sort(key=lambda p: p.get('distance_to_center', float('inf')))
                    player_detections = player_detections[:6]

            # --- Auto court estimation + filtering (no court_config) ---
            effective_net_y = net_y
            if auto_court is not None:
                auto_court.update(frame_idx, player_detections)
                if auto_court.is_ready:
                    player_detections = auto_court.filter_detections(
                        player_detections, margin=20.0
                    )
                    # Use auto-estimated net_y
                    if auto_court.net_y is not None:
                        effective_net_y = auto_court.net_y

            # --- Hard filter with court_config boundary + directional margin ---
            # X (sideline): tight margin to exclude coaches/referees at the sides
            # Y (baseline): loose margin to allow servers standing behind baseline
            elif court_boundary_np is not None:
                court_xs = court_boundary_np[:, 0]
                court_ys = court_boundary_np[:, 1]
                court_w = float(max(court_xs) - min(court_xs))
                court_h = float(max(court_ys) - min(court_ys))
                x_margin = court_w * 0.08   # 8% sideline margin
                y_margin = court_h * 0.30   # 30% baseline margin (serving area)
                x_min = float(min(court_xs)) - x_margin
                x_max = float(max(court_xs)) + x_margin
                y_min = float(min(court_ys)) - y_margin
                y_max = float(max(court_ys)) + y_margin
                kept = []
                for det in player_detections:
                    cx, cy = det['center_point']
                    if x_min <= cx <= x_max and y_min <= cy <= y_max:
                        kept.append(det)
                player_detections = kept

            # Far-side retry: if < 2 players above net, crop+upscale far region
            if effective_net_y is not None:
                far_retry = detect_far_side_players(
                    frame, player_model, player_det_model,
                    player_detections, effective_net_y,
                    iou_thresh=player_iou,
                    imgsz=_imgsz,
                )
                if far_retry:
                    # Filter far-retry results through auto_court or court boundary
                    if auto_court is not None and auto_court.is_ready:
                        far_retry = auto_court.filter_detections(far_retry, margin=20.0)
                    elif court_boundary_np is not None:
                        court_w = max(court_boundary_np[:, 0]) - min(court_boundary_np[:, 0])
                        court_margin = court_w * 0.15
                        far_retry = [
                            d for d in far_retry
                            if cv2.pointPolygonTest(court_boundary_np,
                                (float(d['center_point'][0]), float(d['center_point'][1])), True)
                            >= -court_margin
                        ]
                    if far_retry:
                        player_detections.extend(far_retry)
                        player_detections.sort(key=lambda p: p.get('distance_to_center', float('inf')))
                        player_detections = player_detections[:6]

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
                # A3: Trajectory consistency - prefer ball close to last position
                proximity_threshold = 80 * resolution_scale
                edge_margin = 30 * resolution_scale  # pixels from frame edge
                last_pos = ball_tracker.get_last_position() if hasattr(ball_tracker, 'get_last_position') else None

                # Check if last position was near frame edge (ball likely exited)
                ball_exited = False
                if last_pos is not None:
                    lx, ly = float(last_pos[0]), float(last_pos[1])
                    if (lx < edge_margin or lx > video_width - edge_margin or
                            ly < edge_margin or ly > video_height - edge_margin):
                        ball_exited = True

                if last_pos is not None:
                    # Find balls within proximity of last known position
                    nearby = []
                    for b in valid_balls:
                        cp = b.get('center_point', [0, 0])
                        dist = ((cp[0] - last_pos[0])**2 + (cp[1] - last_pos[1])**2)**0.5
                        if dist < proximity_threshold:
                            nearby.append((dist, b))
                    if nearby:
                        # Pick closest nearby ball
                        nearby.sort(key=lambda x: x[0])
                        best_ball = nearby[0][1]
                    elif ball_exited:
                        # Ball was near edge and no nearby detection found
                        # -> ball likely left frame, skip all detections to avoid jumping
                        best_ball = None
                    else:
                        # No nearby ball, fall back to highest confidence
                        best_ball = max(valid_balls, key=lambda b: b.get('confidence', 0))
                else:
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

        # Add rejected detections for visualization
        if auto_court is not None and auto_court.rejected_detections:
            frame_data["rejected_detections"] = auto_court.rejected_detections
        
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

        # --- Early Stop：嘗試偵測擊球事件，命中後倒數 buffer_frames 幀停止 ---
        if _serve_detector_es is not None and should_detect:
            _es_curr_ball = None
            if tracking_result and tracking_result.get('position') is not None:
                _es_curr_ball = np.array(tracking_result['position'])
            elif best_ball is not None:
                cp = best_ball.get('center_point')
                if cp:
                    _es_curr_ball = np.array(cp)

            if not _es_hit_found:
                _es_event = _serve_detector_es.process_frame(
                    _es_prev_ball, _es_curr_ball, frame_idx,
                    player_detections=player_detections
                )
                if _es_event:
                    _es_hit_found = True
                    _es_countdown = early_stop_buffer_frames
                    if verbose:
                        print(f"[早停] 幀 {frame_idx} 偵測到擊球，"
                              f"再處理 {early_stop_buffer_frames} 幀後停止追蹤")
            _es_prev_ball = _es_curr_ball

        if _es_hit_found:
            if _es_countdown <= 0:
                if verbose:
                    print(f"[早停] 觸發於幀 {frame_idx}，"
                          f"跳過剩餘 {total_frames - frame_idx} 幀")
                break
            _es_countdown -= 1

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
        print(f"[追蹤] Done! Frames: {total_count}, Predicted: {predicted_count} "
              f"({100*predicted_count/total_count:.1f}%)" if total_count > 0 else "[追蹤] Done!")

    if static_filter and verbose:
        static_zones = static_filter.get_static_zones()
        if static_zones:
            print(f"[StaticFilter] Detected {len(static_zones)} static zone(s)")
    
    # --- 儲存 JSON ---
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    json_output_path = os.path.join(output_dir, f"{video_base_name}_all_frames_data_with_pose.json")
    
    # 添加元數據
    fps_scale = fps / 25.0 if fps > 0 else 1.0
    output_data = {
        "metadata": {
            "video_path": video_path,
            "total_frames": frame_idx,
            "fps": fps,
            "detection_interval": detection_interval,
            "use_ball_tracker": use_ball_tracker and HAS_BALL_TRACKER,
            "processing_time": time.time() - start_time,
            "video_context": {
                "width": video_width,
                "height": video_height,
                "fps": fps,
                "resolution_scale": round(resolution_scale, 4),
                "fps_scale": round(fps_scale, 4),
            },
            "frame_height": video_height,
        },
        "frames": all_frames_data
    }

    # Add auto_court info to metadata
    if auto_court is not None and auto_court.is_ready:
        output_data["metadata"]["auto_court"] = {
            "boundary": auto_court.boundary.tolist(),
            "net_y": auto_court.net_y,
        }
        if verbose:
            print(f"[AutoCourt] Boundary estimated, net_y={auto_court.net_y:.0f}")
    
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
    parser.add_argument("--court-config-dir", type=str, default=None,
                        help="court_config 目錄，自動按檔名 group_key 匹配")
    parser.add_argument("--detection_interval", type=int, default=1,
                        help="偵測間隔（每 N 幀偵測一次，預設 1）")
    parser.add_argument("--max_occlusion", type=int, default=15,
                        help="最大遮擋幀數（預設 15）")
    parser.add_argument("--no_tracker", action="store_true",
                        help="停用 BallTracker（使用基礎偵測模式）")
    parser.add_argument("--use_sahi", action="store_true",
                        help="Enable SAHI sliced inference for player detection")
    parser.add_argument("--sahi_slice_size", type=int, default=512,
                        help="SAHI slice size in pixels (default 512)")
    parser.add_argument("--sahi_overlap", type=float, default=0.2,
                        help="SAHI slice overlap ratio (default 0.2)")
    parser.add_argument("--pose-model", type=str, default=None,
                        help=f"球員 Pose 模型檔名（預設: {PLAYER_MODEL_NAME}）")
    parser.add_argument("--player-iou", type=float, default=0.6,
                        help="球員偵測 NMS IoU 門檻（預設 0.6，降低可抑制同人重複框）")
    parser.add_argument("--quiet", action="store_true", help="安靜模式")
    
    args = parser.parse_args()
    
    # 載入場地配置
    court_config = None
    if args.court_config and os.path.exists(args.court_config):
        with open(args.court_config, 'r', encoding='utf-8') as f:
            court_config = json.load(f)
    elif getattr(args, 'court_config_dir', None):
        # 自動按 group_key 匹配 court_config
        try:
            from core.filename_parser import parse_filename, extract_group_key_from_path
            parsed = parse_filename(os.path.basename(args.input))
            group_key = None
            if parsed and parsed.get('group_key'):
                group_key = parsed['group_key']
            else:
                # fallback: 從路徑中提取 group_key
                group_key = extract_group_key_from_path(args.input)
            if group_key:
                auto_path = os.path.join(args.court_config_dir, f"{group_key}.json")
                if os.path.exists(auto_path):
                    with open(auto_path, 'r', encoding='utf-8') as f:
                        court_config = json.load(f)
                    print(f"[AutoMatch] court_config: {auto_path}")
                else:
                    print(f"[WARNING] No court_config for group {group_key}")
            else:
                print(f"[WARNING] Cannot determine group_key for {os.path.basename(args.input)}")
        except Exception as e:
            print(f"[WARNING] court_config auto-match failed: {e}")
    
    try:
        output_path = run_tracking_v2(
            video_path=args.input,
            output_dir=args.output_dir,
            court_config=court_config,
            detection_interval=args.detection_interval,
            use_ball_tracker=not args.no_tracker,
            max_occlusion_frames=args.max_occlusion,
            use_sahi=args.use_sahi,
            sahi_slice_size=args.sahi_slice_size,
            sahi_overlap=args.sahi_overlap,
            player_model_name=args.pose_model,
            player_iou=args.player_iou,
            verbose=not args.quiet
        )
        print(f"完成！輸出: {output_path}")
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)