# core/reception_detector.py
# -*- coding: utf-8 -*-
"""
接球偵測模組

從發球擊球幀開始，追蹤球軌跡直到對方接球。

偵測流程：
    擊球幀 -> 球飛行追蹤 -> 球到達對方半場 ->
    找到最近球員 = 接球員 -> 記錄接球區域

判斷條件（組合判斷）：
1. 球進入對方半場（跨過 net_y）
2. 球與某球員距離很近
3. 球速突然改變（方向或大小）
4. 時間限制：擊球後 30-90 幀
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any

from .data_validator import validate_center_point


# 預設閾值（以 720p 為基準）
DEFAULT_CONFIG = {
    'min_frames_after_hit': 10,      # 擊球後最少等幾幀才開始偵測接球
    'max_frames_after_hit': 90,      # 擊球後最多追蹤幾幀
    'ball_player_distance': 80.0,    # 球與球員距離閾值 (px @720p)
    'speed_change_ratio': 0.4,       # 速度變化比例閾值（下降超過此比例視為接觸）
    'min_speed_for_change': 5.0,     # 最小速度才計算速度變化
    'direction_change_angle': 60.0,  # 方向變化角度閾值（度）
}


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """兩點間歐氏距離"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _speed(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """兩幀間速度"""
    return _distance(p1, p2)


def _velocity(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """兩幀間速度向量"""
    return (p2[0] - p1[0], p2[1] - p1[1])


def _angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """兩向量夾角（度）"""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def _get_ball_center(frame_data: Dict) -> Optional[Tuple[float, float]]:
    """從幀資料提取球中心位置"""
    if not frame_data or not frame_data.get('ball_detections'):
        return None
    balls = frame_data['ball_detections']
    if not balls:
        return None
    # 取信心度最高的球
    best = max(balls, key=lambda b: b.get('confidence', 0))
    cp = best.get('center_point')
    return validate_center_point(cp)


def _get_player_positions(frame_data: Dict) -> List[Tuple[int, Tuple[float, float]]]:
    """
    從幀資料提取所有球員位置

    Returns:
        [(player_index, (x, y)), ...]
    """
    if not frame_data or not frame_data.get('player_detections'):
        return []
    results = []
    for idx, player in enumerate(frame_data['player_detections']):
        cp = player.get('center_point')
        pos = validate_center_point(cp)
        if pos:
            results.append((idx, pos))
    return results


def _find_nearest_player(
    ball_pos: Tuple[float, float],
    players: List[Tuple[int, Tuple[float, float]]],
    max_distance: float,
    target_side: str,
    net_y: float
) -> Optional[Tuple[int, Tuple[float, float], float]]:
    """
    找到距離球最近的對方球員

    Args:
        ball_pos: 球位置
        players: [(index, (x,y)), ...]
        max_distance: 最大距離閾值
        target_side: 接球方 'far' 或 'near'
        net_y: 網線 Y 座標

    Returns:
        (player_index, player_position, distance) 或 None
    """
    best = None
    best_dist = float('inf')

    for idx, pos in players:
        # 過濾：只考慮接球方的球員
        if target_side == 'far' and pos[1] >= net_y:
            continue
        if target_side == 'near' and pos[1] < net_y:
            continue

        dist = _distance(ball_pos, pos)
        if dist < best_dist and dist < max_distance:
            best_dist = dist
            best = (idx, pos, dist)

    return best


class ReceptionDetector:
    """接球偵測器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: 偵測參數配置
        """
        cfg = DEFAULT_CONFIG.copy()
        if config:
            cfg.update(config)

        self.min_frames = cfg['min_frames_after_hit']
        self.max_frames = cfg['max_frames_after_hit']
        self.ball_player_dist = cfg['ball_player_distance']
        self.speed_change_ratio = cfg['speed_change_ratio']
        self.min_speed = cfg['min_speed_for_change']
        self.direction_change_angle = cfg['direction_change_angle']

    def analyze_reception(
        self,
        frames_data: List[Dict],
        serve_event: Dict,
        net_y: float,
        serving_side: str,
        court_zones=None,
        image_height: int = 720,
    ) -> Dict[str, Any]:
        """
        從擊球幀開始追蹤球，偵測接球事件

        Args:
            frames_data: 所有幀的追蹤資料
            serve_event: 發球事件字典，需含 'hit_frame_id'
            net_y: 網線 Y 座標
            serving_side: 發球方 'far' 或 'near'
            court_zones: CourtZones 實例（可選，用於判斷接球區域）

        Returns:
            {
                'reception_detected': bool,
                'reception_frame': int or None,
                'reception_position': [x, y] or None,
                'reception_zone': int or None,  # 1-6
                'receiver_index': int or None,
                'receiver_position': [x, y] or None,
                'time_to_reception': int or None,  # 幀數
                'ball_crossed_net': bool,
                'confidence': float,  # 0-1
            }
        """
        result = {
            'reception_detected': False,
            'reception_frame': None,
            'reception_position': None,
            'reception_zone': None,
            'receiver_index': None,
            'receiver_position': None,
            'time_to_reception': None,
            'ball_crossed_net': False,
            'confidence': 0.0,
            'is_ace': False,
            'landing_zone': None,
            'landing_position': None,
        }

        hit_frame_id = serve_event.get('hit_frame_id')
        if hit_frame_id is None:
            return result

        # 解析度縮放因子（所有像素閾值以 720p 為基準）
        resolution_scale = image_height / 720.0
        scaled_ball_player_dist = self.ball_player_dist * resolution_scale

        # 接球方 = 發球方的對面
        receiving_side = 'near' if serving_side == 'far' else 'far'

        # 追蹤球軌跡
        total_frames = len(frames_data)
        start_idx = hit_frame_id + self.min_frames
        end_idx = min(hit_frame_id + self.max_frames, total_frames - 1)

        if start_idx >= total_frames:
            return result

        ball_crossed_net = False
        prev_ball_pos = None
        prev_prev_ball_pos = None
        prev_speed = None
        post_net_ball_positions = []  # 記錄過網後的球位置（供 ace 落點偵測使用）

        for frame_idx in range(hit_frame_id + 1, end_idx + 1):
            if frame_idx >= total_frames:
                break

            frame = frames_data[frame_idx]
            ball_pos = _get_ball_center(frame)

            if ball_pos is None:
                # 球丟失，清除上一幀位置以避免跳幀計算錯誤
                prev_prev_ball_pos = prev_ball_pos
                prev_ball_pos = None
                continue

            # 檢查球是否跨過網線
            if not ball_crossed_net:
                if serving_side == 'near' and ball_pos[1] < net_y:
                    ball_crossed_net = True
                    result['ball_crossed_net'] = True
                elif serving_side == 'far' and ball_pos[1] >= net_y:
                    ball_crossed_net = True
                    result['ball_crossed_net'] = True

            # 追蹤過網後的球位置（供 ace 落點偵測使用）
            if ball_crossed_net:
                post_net_ball_positions.append((frame_idx, ball_pos))

            # 只在球跨過網線且超過最小幀數後才偵測接球
            frames_since_hit = frame_idx - hit_frame_id
            if frames_since_hit < self.min_frames:
                prev_prev_ball_pos = prev_ball_pos
                prev_ball_pos = ball_pos
                continue

            # 計算速度和方向變化
            speed_changed = False
            direction_changed = False
            current_speed = None

            if prev_ball_pos is not None:
                current_speed = _speed(prev_ball_pos, ball_pos)
                current_vel = _velocity(prev_ball_pos, ball_pos)

                if prev_speed is not None and prev_speed > self.min_speed:
                    speed_ratio = current_speed / prev_speed
                    if speed_ratio < (1.0 - self.speed_change_ratio):
                        speed_changed = True

                if prev_prev_ball_pos is not None:
                    prev_vel = _velocity(prev_prev_ball_pos, prev_ball_pos)
                    angle = _angle_between(prev_vel, current_vel)
                    if angle > self.direction_change_angle:
                        direction_changed = True

            # 找最近的接球方球員
            players = _get_player_positions(frame)
            nearest = _find_nearest_player(
                ball_pos, players, scaled_ball_player_dist, receiving_side, net_y
            )

            # 組合判斷
            proximity_match = nearest is not None
            motion_match = speed_changed or direction_changed
            crossed = ball_crossed_net

            # 判斷是否接球
            # 條件：(球靠近球員) AND (球已過網 OR 動作變化)
            if proximity_match and (crossed or motion_match):
                confidence = self._compute_confidence(
                    proximity_match, motion_match, crossed,
                    nearest[2] if nearest else self.ball_player_dist,
                    frames_since_hit
                )

                result['reception_detected'] = True
                result['reception_frame'] = frame_idx
                result['reception_position'] = list(ball_pos)
                result['receiver_index'] = nearest[0]
                result['receiver_position'] = list(nearest[1])
                result['time_to_reception'] = frames_since_hit
                result['confidence'] = confidence

                # 計算接球區域
                if court_zones is not None:
                    zone = court_zones.get_reception_zone(ball_pos, receiving_side)
                    result['reception_zone'] = zone

                # 有接球：落點等於接球位置（非 ace）
                result['is_ace'] = False
                result['landing_zone'] = result.get('reception_zone')
                result['landing_position'] = list(ball_pos)
                return result

            # 更新歷史
            prev_prev_ball_pos = prev_ball_pos
            prev_ball_pos = ball_pos
            if current_speed is not None:
                prev_speed = current_speed

        # Ace 落點偵測：球過網但無人接球
        if ball_crossed_net and post_net_ball_positions:
            # 找 Y 值最大的幀（球最接近地面）
            _, landing_pos = max(post_net_ball_positions, key=lambda x: x[1][1])
            result['is_ace'] = True
            result['landing_position'] = list(landing_pos)
            if court_zones is not None:
                result['landing_zone'] = court_zones.get_reception_zone(landing_pos, receiving_side)

        return result

    def _compute_confidence(
        self,
        proximity: bool,
        motion: bool,
        crossed_net: bool,
        distance: float,
        frames_since_hit: int
    ) -> float:
        """
        計算接球偵測的信心度

        Args:
            proximity: 球是否靠近球員
            motion: 球是否有速度/方向變化
            crossed_net: 球是否已過網
            distance: 球到接球員距離
            frames_since_hit: 擊球後幀數

        Returns:
            信心度 0.0-1.0
        """
        score = 0.0

        # 靠近球員 (最重要)
        if proximity:
            # 距離越近信心度越高
            dist_ratio = 1.0 - min(distance / self.ball_player_dist, 1.0)
            score += 0.4 * (0.5 + 0.5 * dist_ratio)

        # 速度/方向變化
        if motion:
            score += 0.3

        # 球過網
        if crossed_net:
            score += 0.2

        # 時間合理性 (30-60 幀最合理)
        if 20 <= frames_since_hit <= 60:
            score += 0.1
        elif frames_since_hit <= 90:
            score += 0.05

        return min(1.0, score)
