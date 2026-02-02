# core/jump_serve_detector.py
# -*- coding: utf-8 -*-
"""
跳發偵測模組

分析發球員的腳踝位置變化，判斷是跳發還是站發

跳發特徵：
- 發球員在擊球前會起跳
- 腳踝 Y 座標會明顯上升（Y 值變小）
- 起跳時間通常在拋球後、擊球前

站發特徵：
- 發球員雙腳保持在地面
- 腳踝 Y 座標相對穩定
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .server_identifier import get_keypoint


# 關鍵點索引
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_HIP = 11
RIGHT_HIP = 12


def get_ankle_positions(player: Dict, min_confidence: float = 0.3) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """
    取得球員的左右腳踝位置
    
    Args:
        player: 球員資料
        min_confidence: 最小信心度
        
    Returns:
        (left_ankle, right_ankle) 各為 (x, y) 或 None
    """
    left_ankle = get_keypoint(player, LEFT_ANKLE, min_confidence)
    right_ankle = get_keypoint(player, RIGHT_ANKLE, min_confidence)
    return left_ankle, right_ankle


def get_lowest_ankle_y(player: Dict, min_confidence: float = 0.3) -> Optional[float]:
    """
    取得球員最低的腳踝 Y 座標（最接近地面）
    
    注意：Y 座標越大表示越接近畫面底部（地面）
    
    Args:
        player: 球員資料
        min_confidence: 最小信心度
        
    Returns:
        最低腳踝的 Y 座標，或 None
    """
    left_ankle, right_ankle = get_ankle_positions(player, min_confidence)
    
    if left_ankle is None and right_ankle is None:
        return None
    elif left_ankle is None:
        return right_ankle[1]
    elif right_ankle is None:
        return left_ankle[1]
    else:
        # 返回 Y 較大的（較低的，更接近地面）
        return max(left_ankle[1], right_ankle[1])


def get_average_ankle_y(player: Dict, min_confidence: float = 0.3) -> Optional[float]:
    """
    取得球員腳踝的平均 Y 座標
    
    Args:
        player: 球員資料
        min_confidence: 最小信心度
        
    Returns:
        腳踝平均 Y 座標，或 None
    """
    left_ankle, right_ankle = get_ankle_positions(player, min_confidence)
    
    if left_ankle is None and right_ankle is None:
        return None
    elif left_ankle is None:
        return right_ankle[1]
    elif right_ankle is None:
        return left_ankle[1]
    else:
        return (left_ankle[1] + right_ankle[1]) / 2


def find_server_in_frame(
    frame_data: Dict, 
    server_center: Tuple[float, float],
    max_distance: float = 150.0
) -> Optional[Dict]:
    """
    在幀中找到發球員
    
    Args:
        frame_data: 幀資料
        server_center: 發球員的中心座標（從 FOUND 幀）
        max_distance: 最大匹配距離
        
    Returns:
        發球員資料，或 None
    """
    players = frame_data.get('player_detections', [])
    
    best_player = None
    min_dist = float('inf')
    
    for player in players:
        center = player.get('center_point', [0, 0])
        dist = np.sqrt((center[0] - server_center[0])**2 + (center[1] - server_center[1])**2)
        
        if dist < min_dist and dist < max_distance:
            min_dist = dist
            best_player = player
    
    return best_player


def analyze_jump_serve(
    frames_data: List[Dict],
    serve_event: Dict,
    server_result: Dict,
    jump_threshold: float = 30.0,
    min_jump_frames: int = 3,
    ground_y: float = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    分析是否為跳發
    
    Args:
        frames_data: 所有幀資料
        serve_event: 發球事件（包含 toss_start_frame, hit_frame_id）
        server_result: 發球員識別結果（包含 server 資訊）
        jump_threshold: 跳躍高度閾值（像素）
        min_jump_frames: 最少維持跳躍的幀數
        ground_y: 地面 Y 座標（從場地設定取得，若無則自動估算）
        verbose: 是否輸出詳細資訊
        
    Returns:
        跳發分析結果
    """
    result = {
        'is_jump_serve': False,
        'confidence': 0.0,
        'jump_height': 0.0,
        'baseline_ankle_y': None,
        'min_ankle_y': None,
        'jump_start_frame': None,
        'peak_jump_frame': None,
        'ankle_trajectory': [],
        'analysis_frames': 0,
        'error': None
    }
    
    # 取得發球員資訊
    server_info = server_result.get('server')
    if not server_info:
        result['error'] = 'No server information'
        return result
    
    server_center = server_info.get('center_point')
    if not server_center:
        result['error'] = 'No server center point'
        return result
    
    # 取得關鍵幀
    toss_frame_id = serve_event.get('toss_start_frame')
    hit_frame_id = serve_event.get('hit_frame_id')
    found_frame_id = server_result.get('found_frame_id')
    
    if toss_frame_id is None or hit_frame_id is None:
        result['error'] = 'Missing toss or hit frame'
        return result
    
    # 建立 frame_id 到資料的映射（驗證 frame_id 存在）
    frame_map = {f['frame_id']: f for f in frames_data if 'frame_id' in f}

    if not frame_map:
        result['error'] = '無有效幀資料（所有幀都缺少 frame_id）'
        return result
    
    # 分析範圍：從 FOUND 幀（或拋球前 30 幀）到擊球幀
    if found_frame_id:
        start_frame = found_frame_id
    else:
        start_frame = max(0, toss_frame_id - 30)
    
    end_frame = hit_frame_id + 5  # 擊球後再看幾幀
    
    if verbose:
        print(f"[跳發分析] 分析範圍: frame {start_frame} ~ {end_frame}")
        print(f"[跳發分析] 發球員中心: {server_center}")
    
    # 收集腳踝軌跡
    ankle_trajectory = []
    
    for frame_id in range(start_frame, end_frame + 1):
        if frame_id not in frame_map:
            continue
        
        frame_data = frame_map[frame_id]
        server_player = find_server_in_frame(frame_data, server_center)
        
        if server_player is None:
            continue
        
        ankle_y = get_average_ankle_y(server_player)
        
        if ankle_y is not None:
            ankle_trajectory.append({
                'frame_id': frame_id,
                'ankle_y': ankle_y,
                'player_center': server_player.get('center_point')
            })
    
    result['ankle_trajectory'] = ankle_trajectory
    result['analysis_frames'] = len(ankle_trajectory)

    # 驗證腳踝資料是否充足
    if not ankle_trajectory:
        result['error'] = '無腳踝軌跡資料'
        return result

    if len(ankle_trajectory) < 5:
        result['error'] = f'腳踝資料不足（{len(ankle_trajectory)} 幀），需要至少 5 幀'
        return result

    # 分析腳踝 Y 座標變化
    ankle_y_values = [t['ankle_y'] for t in ankle_trajectory]
    frame_ids = [t['frame_id'] for t in ankle_trajectory]
    
    # 計算基準線（地面位置）
    # 優先使用場地設定的 ground_y，否則用前幾幀平均
    if ground_y is not None:
        baseline_ankle_y = ground_y
        if verbose:
            print(f"[跳發分析] 使用場地設定的地面 Y: {ground_y:.1f}")
    else:
        # 使用起始幀的腳踝位置作為基準線
        if not ankle_y_values:
            result['error'] = '無有效腳踝 Y 座標資料'
            return result

        baseline_frames = min(5, len(ankle_y_values) // 3)
        baseline_ankle_y = np.mean(ankle_y_values[:baseline_frames])
        if verbose:
            print(f"[跳發分析] 使用動態估算的地面 Y: {baseline_ankle_y:.1f}")

    result['baseline_ankle_y'] = float(baseline_ankle_y)

    # 找到最高點（Y 最小）
    if not ankle_y_values:
        result['error'] = '無有效腳踝 Y 座標資料'
        return result

    min_ankle_y = min(ankle_y_values)
    min_ankle_idx = ankle_y_values.index(min_ankle_y)
    result['min_ankle_y'] = float(min_ankle_y)
    result['peak_jump_frame'] = frame_ids[min_ankle_idx]
    
    # 計算跳躍高度（基準線 - 最高點）
    jump_height = baseline_ankle_y - min_ankle_y
    result['jump_height'] = float(jump_height)
    
    if verbose:
        print(f"[跳發分析] 基準腳踝 Y: {baseline_ankle_y:.1f}")
        print(f"[跳發分析] 最低腳踝 Y: {min_ankle_y:.1f}")
        print(f"[跳發分析] 跳躍高度: {jump_height:.1f} 像素")
    
    # 判斷是否為跳發
    # 條件 1：跳躍高度超過閾值
    if jump_height < jump_threshold:
        result['confidence'] = jump_height / jump_threshold * 0.5
        if verbose:
            print(f"[跳發分析] 跳躍高度不足 ({jump_height:.1f} < {jump_threshold})")
        return result
    
    # 條件 2：跳躍必須在擊球前後發生
    peak_frame = frame_ids[min_ankle_idx]
    frames_before_hit = hit_frame_id - peak_frame
    
    if frames_before_hit < -10 or frames_before_hit > 30:
        # 跳躍時間點不對（太早或太晚）
        result['confidence'] = 0.3
        if verbose:
            print(f"[跳發分析] 跳躍時間點異常 (peak={peak_frame}, hit={hit_frame_id})")
        return result
    
    # 條件 3：檢查是否有連續的跳躍幀
    jump_frames = []
    for i, (fid, y) in enumerate(zip(frame_ids, ankle_y_values)):
        if baseline_ankle_y - y > jump_threshold * 0.5:  # 至少一半高度
            jump_frames.append(fid)
    
    # 找連續的跳躍幀
    max_consecutive = 1
    current_consecutive = 1
    jump_start = jump_frames[0] if jump_frames else None
    best_sequence_start_idx = 0  # 追蹤最長序列的起始索引

    for i in range(1, len(jump_frames)):
        if jump_frames[i] - jump_frames[i-1] <= 2:  # 允許跳過 1 幀
            current_consecutive += 1
            if current_consecutive > max_consecutive:
                max_consecutive = current_consecutive
                best_sequence_start_idx = i - current_consecutive + 1
        else:
            current_consecutive = 1

    # 關鍵修復：檢查迴圈結束後的最後序列
    if current_consecutive >= max_consecutive:
        max_consecutive = current_consecutive
        best_sequence_start_idx = len(jump_frames) - current_consecutive

    # 使用最長序列的起始索引設定 jump_start
    if jump_frames:
        jump_start = jump_frames[best_sequence_start_idx]

    result['jump_start_frame'] = jump_start
    
    if max_consecutive < min_jump_frames:
        result['confidence'] = 0.5
        if verbose:
            print(f"[跳發分析] 連續跳躍幀不足 ({max_consecutive} < {min_jump_frames})")
        return result
    
    # 所有條件滿足，判定為跳發
    result['is_jump_serve'] = True
    
    # 計算信心度
    height_score = min(1.0, jump_height / (jump_threshold * 2))
    duration_score = min(1.0, max_consecutive / (min_jump_frames * 2))
    timing_score = 1.0 if 0 <= frames_before_hit <= 15 else 0.7
    
    result['confidence'] = (height_score * 0.5 + duration_score * 0.3 + timing_score * 0.2)
    
    if verbose:
        print(f"[跳發分析] [OK] 判定為跳發")
        print(f"[跳發分析] 信心度: {result['confidence']:.2f}")
        print(f"[跳發分析] 跳躍開始: frame {jump_start}")
        print(f"[跳發分析] 最高點: frame {peak_frame}")
        print(f"[跳發分析] 連續跳躍幀: {max_consecutive}")
    
    return result


def classify_serve_type(
    frames_data: List[Dict],
    serve_event: Dict,
    server_result: Dict,
    court_config: Dict = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    分類發球類型（跳發 vs 站發）
    
    這是主要的對外接口
    
    Args:
        frames_data: 所有幀資料
        serve_event: 發球事件
        server_result: 發球員識別結果
        court_config: 場地設定（用於估算地面位置）
        verbose: 是否輸出詳細資訊
        
    Returns:
        {
            'serve_type': 'jump' 或 'standing',
            'is_jump_serve': bool,
            'confidence': float,
            'jump_height': float,
            'details': {...}
        }
    """
    # 從場地設定估算地面 Y 座標
    ground_y = None
    if court_config:
        boundary = court_config.get('court_boundary_polygon', [])
        if len(boundary) >= 4:
            # 取得場地的上下邊界
            y_values = [pt[1] for pt in boundary]
            ground_y_far = min(y_values)   # 遠端（畫面上方）
            ground_y_near = max(y_values)  # 近端（畫面下方）
            
            # 根據發球員位置判斷使用哪個地面
            server_info = server_result.get('server')
            if server_info:
                server_y = server_info.get('center_point', [0, 0])[1]
                mid_y = (ground_y_far + ground_y_near) / 2
                
                if server_y < mid_y:
                    ground_y = ground_y_far  # 發球員在遠端
                else:
                    ground_y = ground_y_near  # 發球員在近端
    
    analysis = analyze_jump_serve(
        frames_data=frames_data,
        serve_event=serve_event,
        server_result=server_result,
        jump_threshold=30.0,  # 30 像素作為跳躍閾值
        min_jump_frames=3,
        ground_y=ground_y,
        verbose=verbose
    )
    
    return {
        'serve_type': 'jump' if analysis['is_jump_serve'] else 'standing',
        'is_jump_serve': analysis['is_jump_serve'],
        'confidence': analysis['confidence'],
        'jump_height': analysis['jump_height'],
        'baseline_ankle_y': analysis['baseline_ankle_y'],
        'min_ankle_y': analysis['min_ankle_y'],
        'ground_y': ground_y,
        'jump_start_frame': analysis['jump_start_frame'],
        'peak_jump_frame': analysis['peak_jump_frame'],
        'analysis_frames': analysis['analysis_frames'],
        'error': analysis['error'],
        'details': {
            'ankle_trajectory': analysis['ankle_trajectory']
        }
    }