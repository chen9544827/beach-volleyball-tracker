# core/server_identifier.py
# -*- coding: utf-8 -*-
"""
發球員識別模組

根據擊球幀的球位置和球員位置，判斷是哪個球員在發球
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple


# COCO 17 關鍵點索引
KEYPOINT_NAMES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye', 
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}


def determine_serving_side(ball_position: Tuple[float, float], 
                           image_height: int = 720,
                           net_position_ratio: float = 0.35) -> str:
    """
    根據球的位置判斷是哪一方在發球
    
    Args:
        ball_position: 球的位置 (x, y)
        image_height: 畫面高度
        net_position_ratio: 網子在畫面中的相對位置（預設 0.35，表示網子在畫面上方 35% 處）
        
    Returns:
        'far' (對面/畫面上方) 或 'near' (這邊/畫面下方)
    """
    ball_y = ball_position[1]
    net_y = image_height * net_position_ratio
    
    if ball_y < net_y:
        return 'far'  # 球在網子上方，對面發球
    else:
        return 'near'  # 球在網子下方，這邊發球


def find_holding_frame(
    frames_data: List[Dict],
    toss_frame_id: int,
    max_lookback: int = 60,
    ball_stationary_threshold: float = 5.0,
    min_stationary_frames: int = 3
) -> Tuple[Optional[int], Optional[Tuple[float, float]]]:
    """
    從拋球幀往回找，找到球員持球等待的幀
    
    持球特徵：
    - 球的位置幾乎不動（速度很低）
    - 持續數幀
    
    Args:
        frames_data: 所有幀資料
        toss_frame_id: 拋球幀 ID
        max_lookback: 最多往回看幾幀
        ball_stationary_threshold: 球靜止的速度閾值（像素/幀）
        min_stationary_frames: 最少連續靜止幾幀才算持球
        
    Returns:
        (持球幀 ID, 持球位置) 或 (None, None)
    """
    # 建立 frame_id 到 index 的映射
    frame_index_map = {f['frame_id']: i for i, f in enumerate(frames_data)}
    
    if toss_frame_id not in frame_index_map:
        return None, None
    
    # 從拋球幀往回找
    start_frame = max(0, toss_frame_id - max_lookback)
    
    # 收集球的位置
    ball_positions = []
    for frame_id in range(start_frame, toss_frame_id + 1):
        if frame_id not in frame_index_map:
            ball_positions.append(None)
            continue
            
        frame = frames_data[frame_index_map[frame_id]]
        ball_detections = frame.get('ball_detections', [])
        
        if ball_detections:
            # 取信心度最高的球
            best_ball = max(ball_detections, key=lambda b: b.get('confidence', 0))
            if not best_ball.get('is_in_background_zone', False):
                ball_positions.append({
                    'frame_id': frame_id,
                    'position': best_ball.get('center_point')
                })
            else:
                ball_positions.append(None)
        else:
            ball_positions.append(None)
    
    # 計算速度並找靜止區間
    stationary_frames = []
    
    for i in range(1, len(ball_positions)):
        if ball_positions[i] is None or ball_positions[i-1] is None:
            continue
        
        pos1 = ball_positions[i-1]['position']
        pos2 = ball_positions[i]['position']
        
        if pos1 is None or pos2 is None:
            continue
        
        # 計算速度
        speed = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
        
        if speed < ball_stationary_threshold:
            stationary_frames.append({
                'frame_id': ball_positions[i]['frame_id'],
                'position': ball_positions[i]['position'],
                'speed': speed
            })
    
    # 找最後一段連續靜止區間（最接近拋球的持球時刻）
    if len(stationary_frames) < min_stationary_frames:
        # 靜止幀不夠，退回使用拋球幀前幾幀
        for i in range(len(ball_positions) - 1, -1, -1):
            if ball_positions[i] is not None:
                return ball_positions[i]['frame_id'], tuple(ball_positions[i]['position'])
        return None, None
    
    # 從後往前找連續靜止區間
    best_holding_frame = None
    best_holding_position = None
    
    consecutive_count = 0
    for i in range(len(stationary_frames) - 1, -1, -1):
        if i > 0:
            frame_diff = stationary_frames[i]['frame_id'] - stationary_frames[i-1]['frame_id']
            if frame_diff <= 2:  # 允許有 1 幀間隔
                consecutive_count += 1
            else:
                if consecutive_count >= min_stationary_frames - 1:
                    # 找到了足夠長的靜止區間
                    best_holding_frame = stationary_frames[i]['frame_id']
                    best_holding_position = tuple(stationary_frames[i]['position'])
                    break
                consecutive_count = 0
        
        if consecutive_count >= min_stationary_frames - 1:
            best_holding_frame = stationary_frames[i]['frame_id']
            best_holding_position = tuple(stationary_frames[i]['position'])
    
    # 如果沒找到連續靜止區間，使用最後一個靜止幀
    if best_holding_frame is None and stationary_frames:
        best_holding_frame = stationary_frames[-1]['frame_id']
        best_holding_position = tuple(stationary_frames[-1]['position'])
    
    return best_holding_frame, best_holding_position


def find_server_by_holding_position(
    frames_data: List[Dict],
    holding_frame_id: int,
    holding_position: Tuple[float, float],
    max_distance: float = 100.0
) -> Dict[str, Any]:
    """
    根據持球位置找發球員
    
    Args:
        frames_data: 所有幀資料
        holding_frame_id: 持球幀 ID
        holding_position: 持球位置
        max_distance: 球員中心到球的最大距離
        
    Returns:
        發球員資訊
    """
    # 找到持球幀
    holding_frame = None
    for frame in frames_data:
        if frame.get('frame_id') == holding_frame_id:
            holding_frame = frame
            break
    
    if holding_frame is None:
        return {
            'server_index': None,
            'server': None,
            'confidence': 0,
            'holding_frame_id': holding_frame_id,
            'holding_position': holding_position,
            'error': 'Holding frame not found'
        }
    
    players = holding_frame.get('player_detections', [])
    
    if not players:
        return {
            'server_index': None,
            'server': None,
            'confidence': 0,
            'holding_frame_id': holding_frame_id,
            'holding_position': holding_position,
            'error': 'No players detected in holding frame'
        }
    
    # 找離球最近的球員
    best_player = None
    best_index = None
    best_distance = float('inf')
    
    for i, player in enumerate(players):
        center = player.get('center_point', [0, 0])
        distance = calculate_distance(holding_position, tuple(center))
        
        if distance < best_distance:
            best_distance = distance
            best_player = player
            best_index = i
    
    # 計算信心度
    if best_distance <= max_distance:
        confidence = max(0, 1 - best_distance / max_distance)
    else:
        confidence = 0.1  # 距離太遠，低信心度
    
    return {
        'server_index': best_index,
        'server': best_player,
        'confidence': round(confidence, 2),
        'holding_frame_id': holding_frame_id,
        'holding_position': holding_position,
        'distance_to_ball': round(best_distance, 1)
    }


def find_server_with_lookback(
    frames_data: List[Dict],
    start_frame_id: int,
    max_lookback: int = 90,
    overlap_threshold: float = 70.0,
    exclusion_zones: List[List[Tuple[int, int]]] = None
) -> Dict[str, Any]:
    """
    從指定幀往回找，直到找到有球員與球重疊的幀
    
    Args:
        frames_data: 所有幀資料
        start_frame_id: 起始幀 ID（通常是拋球幀）
        max_lookback: 最多往回找幾幀
        overlap_threshold: 判定為「重疊」的最大距離（像素）
        exclusion_zones: 排除區域列表，每個區域是多邊形頂點列表
        
    Returns:
        發球員資訊
    """
    import cv2
    
    # 建立 frame_id 到 frame 的映射
    frame_map = {f['frame_id']: f for f in frames_data}
    
    # 將排除區域轉換為 numpy 陣列（用於 pointPolygonTest）
    exclusion_zones_np = []
    if exclusion_zones:
        for zone in exclusion_zones:
            if isinstance(zone, dict) and 'polygon' in zone:
                # 格式: {"polygon": [[x1,y1], [x2,y2], ...]}
                exclusion_zones_np.append(np.array(zone['polygon'], dtype=np.float32))
            elif isinstance(zone, list):
                # 格式: [[x1,y1], [x2,y2], ...] 或 [(x1,y1), (x2,y2), ...]
                exclusion_zones_np.append(np.array(zone, dtype=np.float32))
    
    # 從 start_frame_id 往回找
    for frame_id in range(start_frame_id, max(0, start_frame_id - max_lookback), -1):
        if frame_id not in frame_map:
            continue
        
        frame = frame_map[frame_id]
        ball_detections = frame.get('ball_detections', [])
        players = frame.get('player_detections', [])
        
        # 需要有球和球員
        if not ball_detections or not players:
            continue
        
        # 取得球位置（排除背景球）
        valid_balls = [b for b in ball_detections if not b.get('is_in_background_zone', False)]
        if not valid_balls:
            continue
        
        best_ball = max(valid_balls, key=lambda b: b.get('confidence', 0))
        ball_position = best_ball.get('center_point')
        
        if not ball_position:
            continue
        
        # 找離球最近的球員（排除在排除區域內的人）
        best_player = None
        best_index = None
        best_distance = float('inf')
        
        for i, player in enumerate(players):
            center = player.get('center_point', [0, 0])
            
            # 檢查是否在排除區域內
            in_exclusion = False
            if exclusion_zones_np:
                for zone_np in exclusion_zones_np:
                    if cv2.pointPolygonTest(zone_np, tuple(center), False) >= 0:
                        in_exclusion = True
                        break
            
            if in_exclusion:
                continue
            
            distance = calculate_distance(tuple(ball_position), tuple(center))
            
            if distance < best_distance:
                best_distance = distance
                best_player = player
                best_index = i
        
        # 如果沒有找到有效球員（全被排除了），繼續往回找
        if best_player is None:
            continue
        
        # 檢查是否重疊（距離小於閾值）
        if best_distance <= overlap_threshold:
            # 找到了！
            confidence = max(0.5, 1 - best_distance / overlap_threshold)
            return {
                'server_index': best_index,
                'server': best_player,
                'confidence': round(confidence, 2),
                'found_frame_id': frame_id,
                'ball_position': ball_position,
                'distance_to_ball': round(best_distance, 1),
                'method': 'lookback',
                'frames_searched': start_frame_id - frame_id
            }
    
    # 找不到重疊的幀，返回失敗
    return {
        'server_index': None,
        'server': None,
        'confidence': 0,
        'found_frame_id': None,
        'error': f'No overlapping player found within {max_lookback} frames'
    }


def filter_players_by_side(players: List[Dict], 
                           serving_side: str,
                           image_height: int = 720,
                           net_position_ratio: float = 0.35) -> List[Dict]:
    """
    根據發球方過濾球員，只保留該半場的球員
    
    Args:
        players: 所有偵測到的人
        serving_side: 'far' 或 'near'
        image_height: 畫面高度
        net_position_ratio: 網子位置比例
        
    Returns:
        該半場的球員列表（包含原始索引）
    """
    if not players:
        return []
    
    net_y = image_height * net_position_ratio
    filtered = []
    
    for i, player in enumerate(players):
        center = player.get('center_point', [0, 0])
        player_y = center[1]
        
        player_with_idx = player.copy()
        player_with_idx['original_index'] = i
        
        if serving_side == 'far':
            # 對面發球，找畫面上方的球員
            if player_y < net_y + 100:  # 網子上方（加一些容錯）
                filtered.append(player_with_idx)
        else:
            # 這邊發球，找畫面下方的球員
            if player_y > net_y + 50:  # 網子下方
                filtered.append(player_with_idx)
    
    return filtered


def get_keypoint(player: Dict, keypoint_idx: int, min_confidence: float = 0.3) -> Optional[Tuple[float, float]]:
    """
    取得球員的特定關鍵點位置
    
    Args:
        player: 球員資料
        keypoint_idx: 關鍵點索引
        min_confidence: 最小信心度
        
    Returns:
        (x, y) 或 None
    """
    keypoints = player.get('pose_keypoints', [])
    if keypoint_idx >= len(keypoints):
        return None
    
    kp = keypoints[keypoint_idx]
    if len(kp) >= 3 and kp[2] >= min_confidence:
        return (kp[0], kp[1])
    return None


def get_player_wrists(player: Dict) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """取得球員的左右手腕位置"""
    left_wrist = get_keypoint(player, 9)
    right_wrist = get_keypoint(player, 10)
    return left_wrist, right_wrist


def get_highest_wrist(player: Dict) -> Optional[Tuple[float, float]]:
    """取得球員舉最高的手腕位置（y 最小）"""
    left_wrist, right_wrist = get_player_wrists(player)
    
    if left_wrist is None and right_wrist is None:
        return None
    elif left_wrist is None:
        return right_wrist
    elif right_wrist is None:
        return left_wrist
    else:
        # 返回 y 較小的（較高的）
        return left_wrist if left_wrist[1] < right_wrist[1] else right_wrist


def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """計算兩點距離"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def filter_court_players(players: List[Dict], image_height: int = 720, 
                          court_margin: float = 0.15) -> List[Dict]:
    """
    過濾出場上球員，排除裁判、球僮等
    
    沙灘排球場地特徵：
    - 球員在底線附近（畫面上方或下方）
    - 裁判在網旁（畫面中間偏上）
    - 球僮在場外
    
    Args:
        players: 所有偵測到的人
        image_height: 畫面高度
        court_margin: 邊界比例（上下各 15% 為底線區域）
        
    Returns:
        過濾後的球員列表（保留原始索引資訊）
    """
    if not players:
        return []
    
    filtered = []
    
    # 計算邊界
    top_boundary = image_height * 0.35      # 上方 35% 以上為上半場底線區域
    bottom_boundary = image_height * 0.55   # 下方 55% 以下為下半場底線區域
    middle_top = image_height * 0.20        # 網前區域上界
    middle_bottom = image_height * 0.45     # 網前區域下界
    
    for i, player in enumerate(players):
        center = player.get('center_point', [0, 0])
        y = center[1]
        
        # 保留原始索引
        player_with_idx = player.copy()
        player_with_idx['original_index'] = i
        
        # 判斷是否在球員區域（底線附近）
        # 排除網前中間區域的人（可能是裁判）
        is_in_middle = middle_top < y < middle_bottom
        
        if not is_in_middle:
            # 不在網前區域，可能是球員
            filtered.append(player_with_idx)
        else:
            # 在網前區域，檢查是否有舉手動作（發球員會舉手）
            # 如果手腕位置很高，可能還是發球員
            left_wrist = get_keypoint(player, 9)
            right_wrist = get_keypoint(player, 10)
            
            highest_wrist_y = float('inf')
            if left_wrist:
                highest_wrist_y = min(highest_wrist_y, left_wrist[1])
            if right_wrist:
                highest_wrist_y = min(highest_wrist_y, right_wrist[1])
            
            # 如果手腕很高（在頭部以上），可能是發球員
            nose = get_keypoint(player, 0, min_confidence=0.1)
            if nose and highest_wrist_y < nose[1] - 50:
                # 手在頭上方 50 像素以上，可能是發球
                filtered.append(player_with_idx)
    
    return filtered


def identify_server(
    ball_position: Tuple[float, float],
    players: List[Dict],
    ball_velocity: Tuple[float, float] = None,
    method: str = 'combined',
    image_height: int = 720,
    filter_by_serving_side: bool = True
) -> Dict[str, Any]:
    """
    識別發球員
    
    Args:
        ball_position: 球的位置 (x, y)
        players: 球員列表
        ball_velocity: 球的速度向量 (vx, vy)，用於確認飛行方向
        method: 判斷方法
            - 'nearest': 找離球最近的球員
            - 'highest_hand': 找手舉最高的球員
            - 'combined': 綜合判斷（預設）
        image_height: 畫面高度（用於過濾）
        filter_by_serving_side: 是否根據發球方過濾球員
    
    Returns:
        {
            'server_index': 發球員在 players 中的索引,
            'server': 發球員資料,
            'confidence': 判斷信心度,
            'method_used': 使用的方法,
            'serving_side': 發球方 ('far' 或 'near'),
            'details': 詳細資訊
        }
    """
    if not players:
        return {
            'server_index': None,
            'server': None,
            'confidence': 0,
            'method_used': method,
            'serving_side': None,
            'details': 'No players detected'
        }
    
    # 1. 判斷發球方（根據球的位置）
    serving_side = determine_serving_side(ball_position, image_height)
    
    # 2. 根據發球方過濾球員
    if filter_by_serving_side:
        side_players = filter_players_by_side(players, serving_side, image_height)
        
        # 如果該半場沒有偵測到球員，可能是被網子遮擋
        if not side_players:
            # 退回使用所有球員，但標記為低信心度
            side_players = [{'original_index': i, **p} for i, p in enumerate(players)]
            side_filter_failed = True
        else:
            side_filter_failed = False
    else:
        side_players = [{'original_index': i, **p} for i, p in enumerate(players)]
        side_filter_failed = False
    
    results = []
    
    for player in side_players:
        original_idx = player.get('original_index', 0)
        player_center = player.get('center_point', [0, 0])
        highest_wrist = get_highest_wrist(player)
        
        # 計算各項指標
        center_distance = calculate_distance(ball_position, tuple(player_center))
        
        wrist_distance = float('inf')
        if highest_wrist:
            wrist_distance = calculate_distance(ball_position, highest_wrist)
        
        wrist_height = float('inf')
        if highest_wrist:
            wrist_height = highest_wrist[1]
        
        results.append({
            'index': original_idx,
            'player': player,
            'center_distance': center_distance,
            'wrist_distance': wrist_distance,
            'wrist_height': wrist_height,
            'highest_wrist': highest_wrist
        })
    
    # 根據方法選擇發球員
    if method == 'nearest':
        # 方法1：找離球最近的球員（用中心點）
        best = min(results, key=lambda x: x['center_distance'])
        confidence = max(0, 1 - best['center_distance'] / 500)  # 距離越近信心度越高
        
    elif method == 'highest_hand':
        # 方法2：找手舉最高的球員
        valid_results = [r for r in results if r['wrist_height'] < float('inf')]
        if not valid_results:
            # 沒有有效的手腕資料，退回用最近距離
            best = min(results, key=lambda x: x['center_distance'])
            confidence = 0.3
        else:
            best = min(valid_results, key=lambda x: x['wrist_height'])
            confidence = 0.7
            
    else:  # combined
        # 方法3：綜合判斷
        # 評分 = 手腕距離權重 + 手腕高度權重
        for r in results:
            score = 0
            
            # 手腕距離分數（越近越好）
            if r['wrist_distance'] < float('inf'):
                wrist_dist_score = max(0, 1 - r['wrist_distance'] / 300)
                score += wrist_dist_score * 0.6
            else:
                # 沒有手腕資料，用中心距離
                center_dist_score = max(0, 1 - r['center_distance'] / 500)
                score += center_dist_score * 0.3
            
            # 手腕高度分數（越高越好）
            if r['wrist_height'] < float('inf'):
                # 假設發球時手腕在 y=100-400 的範圍
                height_score = max(0, 1 - (r['wrist_height'] - 100) / 400)
                score += height_score * 0.4
            
            r['score'] = score
        
        best = max(results, key=lambda x: x.get('score', 0))
        confidence = best.get('score', 0)
    
    # 如果有球速資訊，驗證飛行方向
    direction_verified = None
    if ball_velocity:
        vx, vy = ball_velocity
        ball_y = ball_position[1]
        player_y = best['player'].get('center_point', [0, 0])[1]
        
        # 球應該往遠離發球員的方向飛
        # 如果球員在畫面下方（y大），球應該往上飛（vy < 0）
        # 如果球員在畫面上方（y小），球應該往下飛（vy > 0）
        if player_y > 400:  # 球員在畫面下方
            direction_verified = vy < 0
        else:  # 球員在畫面上方
            direction_verified = vy > 0
        
        if direction_verified:
            confidence = min(1.0, confidence + 0.1)
        else:
            confidence = max(0, confidence - 0.2)
    
    return {
        'server_index': best['index'],
        'server': best['player'],
        'confidence': round(confidence, 2),
        'method_used': method,
        'serving_side': serving_side,
        'side_filter_failed': side_filter_failed if 'side_filter_failed' in dir() else False,
        'wrist_position': best.get('highest_wrist'),
        'wrist_to_ball_distance': best.get('wrist_distance'),
        'direction_verified': direction_verified,
        'details': {
            'all_candidates': [
                {
                    'index': r['index'],
                    'center_distance': round(r['center_distance'], 1),
                    'wrist_distance': round(r['wrist_distance'], 1) if r['wrist_distance'] < float('inf') else None,
                    'wrist_height': round(r['wrist_height'], 1) if r['wrist_height'] < float('inf') else None,
                    'score': round(r.get('score', 0), 2)
                }
                for r in results
            ]
        }
    }


def analyze_serve_player(
    frames_data: List[Dict],
    serve_event: Dict,
    method: str = 'lookback',
    image_height: int = 720,
    exclusion_zones: List = None
) -> Dict[str, Any]:
    """
    分析發球事件，找出發球員
    
    Args:
        frames_data: 所有幀資料
        serve_event: 發球事件（包含 toss_start_frame, hit_frame_id 等）
        method: 判斷方法
            - 'lookback': 從拋球幀往回找球員與球重疊的幀（推薦）
            - 'toss_frame': 使用拋球幀判斷
            - 'hit_frame': 使用擊球幀判斷
        image_height: 畫面高度
        exclusion_zones: 排除區域列表（來自 court_config.json）
        
    Returns:
        發球員識別結果
    """
    toss_frame_id = serve_event.get('toss_start_frame')
    hit_frame_id = serve_event.get('hit_frame_id')
    toss_position = serve_event.get('toss_position')
    hit_position = serve_event.get('hit_position')
    hit_velocity = serve_event.get('hit_velocity', [0, 0])
    
    result = {
        'method': method,
        'toss_frame_id': toss_frame_id,
        'hit_frame_id': hit_frame_id,
        'found_frame_id': None,
        'ball_position': None,
        'final_server_index': None,
        'final_confidence': 0,
        'server': None,
    }
    
    # 方法 1：Lookback（推薦）- 從拋球幀往回找
    if method == 'lookback' and toss_frame_id is not None:
        lookback_result = find_server_with_lookback(
            frames_data=frames_data,
            start_frame_id=toss_frame_id,
            max_lookback=90,  # 往回找最多 90 幀（約 3-4 秒）
            overlap_threshold=100.0,  # 球員中心到球 100 像素內算重疊
            exclusion_zones=exclusion_zones
        )
        
        result['lookback_result'] = lookback_result
        result['found_frame_id'] = lookback_result.get('found_frame_id')
        result['ball_position'] = lookback_result.get('ball_position')
        result['final_server_index'] = lookback_result.get('server_index')
        result['final_confidence'] = lookback_result.get('confidence', 0)
        result['server'] = lookback_result.get('server')
        result['distance_to_ball'] = lookback_result.get('distance_to_ball')
        result['frames_searched'] = lookback_result.get('frames_searched', 0)
        
        if lookback_result.get('server_index') is not None:
            return result
        
        # 如果 lookback 失敗，退回使用拋球幀
        method = 'toss_frame'
    
    # 方法 2：使用拋球幀
    if method == 'toss_frame' and toss_frame_id is not None and toss_position is not None:
        toss_frame_data = None
        for frame in frames_data:
            if frame.get('frame_id') == toss_frame_id:
                toss_frame_data = frame
                break
        
        if toss_frame_data:
            players = toss_frame_data.get('player_detections', [])
            toss_result = identify_server(
                ball_position=tuple(toss_position),
                players=players,
                ball_velocity=None,
                method='combined',
                image_height=image_height
            )
            result['toss_result'] = toss_result
            result['found_frame_id'] = toss_frame_id
            result['ball_position'] = toss_position
            result['final_server_index'] = toss_result['server_index']
            result['final_confidence'] = toss_result['confidence']
            result['server'] = toss_result.get('server')
            return result
    
    # 方法 3：使用擊球幀（最後手段）
    if hit_frame_id is not None and hit_position is not None:
        hit_frame_data = None
        for frame in frames_data:
            if frame.get('frame_id') == hit_frame_id:
                hit_frame_data = frame
                break
        
        if hit_frame_data:
            players = hit_frame_data.get('player_detections', [])
            hit_result = identify_server(
                ball_position=tuple(hit_position),
                players=players,
                ball_velocity=tuple(hit_velocity) if hit_velocity else None,
                method='combined',
                image_height=image_height
            )
            result['hit_result'] = hit_result
            result['found_frame_id'] = hit_frame_id
            result['ball_position'] = hit_position
            result['final_server_index'] = hit_result['server_index']
            result['final_confidence'] = hit_result['confidence']
            result['server'] = hit_result.get('server')
    
    return result


if __name__ == "__main__":
    # 簡單測試
    print("測試發球員識別...")
    
    # 模擬資料
    test_players = [
        {
            'center_point': [400, 600],
            'pose_keypoints': [
                [400, 500, 0.9],  # nose
                [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],  # eyes, ears
                [380, 520, 0.9], [420, 520, 0.9],  # shoulders
                [360, 480, 0.9], [440, 480, 0.9],  # elbows
                [350, 420, 0.9], [450, 350, 0.9],  # wrists - 右手舉高
                [380, 600, 0.9], [420, 600, 0.9],  # hips
                [380, 700, 0.9], [420, 700, 0.9],  # knees
                [380, 800, 0.9], [420, 800, 0.9],  # ankles
            ]
        },
        {
            'center_point': [800, 600],
            'pose_keypoints': [
                [800, 550, 0.9],  # nose
                [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],  # eyes, ears
                [780, 570, 0.9], [820, 570, 0.9],  # shoulders
                [760, 600, 0.9], [840, 600, 0.9],  # elbows
                [750, 620, 0.9], [850, 620, 0.9],  # wrists - 雙手放下
                [780, 650, 0.9], [820, 650, 0.9],  # hips
                [780, 750, 0.9], [820, 750, 0.9],  # knees
                [780, 850, 0.9], [820, 850, 0.9],  # ankles
            ]
        }
    ]
    
    # 球在第一個球員上方
    ball_pos = (420, 320)
    
    result = identify_server(ball_pos, test_players, method='combined')
    
    print(f"\n球位置: {ball_pos}")
    print(f"發球員索引: {result['server_index']}")
    print(f"信心度: {result['confidence']}")
    print(f"手腕位置: {result['wrist_position']}")
    print(f"手腕到球距離: {result['wrist_to_ball_distance']:.1f}")
    print("\n所有候選:")
    for c in result['details']['all_candidates']:
        print(f"  球員 {c['index']}: 中心距離={c['center_distance']}, "
              f"手腕距離={c['wrist_distance']}, 手腕高度={c['wrist_height']}, "
              f"分數={c['score']}")