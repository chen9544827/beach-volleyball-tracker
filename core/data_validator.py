"""
資料驗證模組

提供追蹤資料、場地設定等的驗證功能，採用混合策略：
- 關鍵欄位：嚴格驗證（必須存在且格式正確）
- 次要欄位：寬鬆處理（允許缺失，記錄警告）
"""

import json
import os
import logging
from typing import Dict, List, Optional, Tuple, Any

from .error_messages import format_error, format_warning, ValidationError


class DataValidator:
    """資料驗證器類別"""

    def __init__(self, verbose: bool = True):
        """
        初始化驗證器

        Args:
            verbose: 是否輸出詳細驗證訊息
        """
        self.verbose = verbose
        self.warnings: List[str] = []

    def validate_tracking_json(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        驗證追蹤 JSON 資料

        採用混合驗證策略：
        - 關鍵欄位（frames, frame_id, metadata）必須存在
        - 次要欄位（ball_detections, player_detections）允許缺失

        Args:
            data: 追蹤資料字典

        Returns:
            (is_valid, errors): 驗證是否通過及錯誤訊息列表
        """
        errors = []
        self.warnings = []

        # === 關鍵欄位驗證（嚴格） ===

        # 1. 檢查 frames 欄位
        if 'frames' not in data:
            errors.append(format_error('missing_frames'))
            return False, errors

        frames = data['frames']
        if not isinstance(frames, list):
            errors.append("'frames' 欄位必須是列表類型")
            return False, errors

        if len(frames) == 0:
            errors.append(format_error('missing_frames'))
            return False, errors

        # 2. 檢查 metadata 欄位
        if 'metadata' not in data:
            errors.append(format_error('missing_metadata'))
            return False, errors

        metadata = data['metadata']
        if not isinstance(metadata, dict):
            errors.append("'metadata' 欄位必須是字典類型")
            return False, errors

        # 3. 檢查 metadata 必要子欄位
        if 'video_path' not in metadata:
            errors.append(format_error('missing_video_path'))
            return False, errors

        if 'fps' not in metadata:
            errors.append(format_error('missing_fps'))
            return False, errors

        try:
            fps = float(metadata['fps'])
            if fps <= 0:
                errors.append(f"metadata.fps 必須是正數，當前值: {fps}")
                return False, errors
        except (ValueError, TypeError):
            errors.append(f"metadata.fps 必須是數字類型，當前值: {metadata['fps']}")
            return False, errors

        # 4. 驗證每個幀的 frame_id
        valid_frame_count = 0
        for idx, frame in enumerate(frames):
            if not isinstance(frame, dict):
                errors.append(f"幀 {idx} 必須是字典類型")
                continue

            if 'frame_id' not in frame:
                errors.append(format_error('invalid_frame_id', index=idx))
                continue

            try:
                frame_id = int(frame['frame_id'])
                if frame_id < 0:
                    errors.append(f"幀 {idx} 的 frame_id 不能是負數: {frame_id}")
                    continue
            except (ValueError, TypeError):
                errors.append(f"幀 {idx} 的 frame_id 必須是整數類型")
                continue

            valid_frame_count += 1

        if valid_frame_count == 0:
            errors.append(format_error('no_valid_frame_data'))
            return False, errors

        # === 次要欄位驗證（寬鬆，僅警告） ===

        incomplete_frames = 0
        for frame in frames:
            if not isinstance(frame, dict) or 'frame_id' not in frame:
                continue

            frame_id = frame['frame_id']

            # 檢查 ball_detections（可選，允許空列表）
            if 'ball_detections' not in frame:
                self.warnings.append(
                    format_warning('partial_ball_detections', frame_id=frame_id)
                )
                incomplete_frames += 1
            elif not isinstance(frame['ball_detections'], list):
                self.warnings.append(
                    f"警告: 幀 {frame_id} 的 ball_detections 不是列表類型"
                )

            # 檢查 player_detections（可選，允許空列表）
            if 'player_detections' not in frame:
                self.warnings.append(
                    format_warning('partial_player_detections', frame_id=frame_id)
                )
                incomplete_frames += 1
            elif not isinstance(frame['player_detections'], list):
                self.warnings.append(
                    f"警告: 幀 {frame_id} 的 player_detections 不是列表類型"
                )

        # 如果超過 50% 的幀不完整，輸出統計警告
        if incomplete_frames > len(frames) * 0.5:
            self.warnings.append(
                f"警告: 有 {incomplete_frames}/{len(frames)} ({incomplete_frames/len(frames)*100:.1f}%) 的幀資料不完整（可能因遮擋）"
            )

        # 輸出警告訊息
        if self.verbose and self.warnings:
            for warning in self.warnings[:5]:  # 只顯示前 5 個警告
                logging.warning(warning)
            if len(self.warnings) > 5:
                logging.warning(f"... 還有 {len(self.warnings) - 5} 個警告")

        # 如果有嚴重錯誤，返回失敗
        if errors:
            return False, errors

        return True, []

    def validate_court_config(self, config: Dict) -> Tuple[bool, List[str]]:
        """
        驗證場地設定資料

        Args:
            config: 場地設定字典

        Returns:
            (is_valid, errors): 驗證是否通過及錯誤訊息列表
        """
        errors = []
        self.warnings = []

        if not config:
            errors.append("場地設定為空")
            return False, errors

        # 檢查 court_boundary_polygon
        if 'court_boundary_polygon' not in config:
            errors.append(format_error('missing_court_boundary'))
        else:
            polygon = config['court_boundary_polygon']
            if not isinstance(polygon, list) or len(polygon) < 3:
                errors.append("court_boundary_polygon 必須是至少包含 3 個點的列表")

        # 檢查 exclusion_zones（可選）
        if 'exclusion_zones' not in config:
            self.warnings.append(format_warning('no_exclusion_zones'))
        else:
            zones = config['exclusion_zones']
            if not isinstance(zones, list):
                errors.append("exclusion_zones 必須是列表類型")

        if self.verbose and self.warnings:
            for warning in self.warnings:
                logging.warning(warning)

        return len(errors) == 0, errors

    def validate_frame_data(self, frame_data: Dict, frame_id: int) -> bool:
        """
        驗證單一幀資料的有效性（寬鬆驗證）

        Args:
            frame_data: 幀資料字典
            frame_id: 幀 ID

        Returns:
            是否有效（只有嚴重錯誤才返回 False）
        """
        if not frame_data:
            return False

        if 'frame_id' not in frame_data:
            return False

        # ball_detections 和 player_detections 可以是空列表
        return True


def safe_load_json(json_path: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    安全載入 JSON 檔案，處理所有可能的錯誤

    Args:
        json_path: JSON 檔案路徑

    Returns:
        (data, error_message): 成功時返回 (data, None)，失敗時返回 (None, error_message)
    """
    # 檢查檔案是否存在
    if not os.path.exists(json_path):
        error_msg = format_error('file_not_found', path=json_path)
        return None, error_msg

    # 檢查是否為檔案
    if not os.path.isfile(json_path):
        error_msg = f"路徑不是檔案: {json_path}"
        return None, error_msg

    # 嘗試讀取 JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 檢查是否為空
        if data is None:
            error_msg = format_error('json_empty_file', path=json_path)
            return None, error_msg

        return data, None

    except json.JSONDecodeError as e:
        error_msg = format_error('json_decode_error', path=json_path, error=str(e))
        return None, error_msg

    except UnicodeDecodeError as e:
        error_msg = format_error('file_read_error', path=json_path, error=f"編碼錯誤: {e}")
        return None, error_msg

    except Exception as e:
        error_msg = format_error('file_read_error', path=json_path, error=str(e))
        return None, error_msg


def get_keypoint(pose_keypoints: List, keypoint_idx: int, confidence_threshold: float = 0.3) -> Optional[Tuple[float, float]]:
    """
    安全取得姿態關鍵點座標

    Args:
        pose_keypoints: 關鍵點列表 [[x, y, conf], ...]
        keypoint_idx: 關鍵點索引（COCO-17 格式）
        confidence_threshold: 信心度閾值

    Returns:
        (x, y) 或 None（如果關鍵點不存在或信心度過低）
    """
    if not pose_keypoints:
        return None

    if keypoint_idx < 0 or keypoint_idx >= len(pose_keypoints):
        return None

    keypoint = pose_keypoints[keypoint_idx]
    if not keypoint or len(keypoint) < 3:
        return None

    x, y, conf = keypoint[0], keypoint[1], keypoint[2]

    if conf < confidence_threshold:
        return None

    return (float(x), float(y))


def validate_center_point(point: Any) -> Optional[Tuple[float, float]]:
    """
    驗證中心點座標的有效性

    Args:
        point: 可能是座標的物件

    Returns:
        (x, y) 或 None（如果無效）
    """
    if not point:
        return None

    if not isinstance(point, (list, tuple)):
        return None

    if len(point) < 2:
        return None

    try:
        x = float(point[0])
        y = float(point[1])
        return (x, y)
    except (ValueError, TypeError, IndexError):
        return None
