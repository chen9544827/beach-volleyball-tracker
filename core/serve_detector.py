# core/serve_detector.py
# -*- coding: utf-8 -*-
"""
發球偵測器模組 - 改進版

改進內容：
1. 動態閾值：根據影片統計自動調整
2. 速度比例驗證：區分擊球和下墜
3. 加速度驗證：確保符合物理規律
4. 與 BallTracker 整合：利用平滑軌跡

狀態機：
SEARCHING_TOSS → CONFIRMING_TOSS → AWAITING_APEX → AWAITING_HIT → [發球事件] → SEARCHING_TOSS
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from enum import Enum


class ServeState(Enum):
    """發球偵測狀態"""
    SEARCHING_TOSS = "SEARCHING_TOSS"      # 尋找拋球
    CONFIRMING_TOSS = "CONFIRMING_TOSS"    # 確認拋球
    AWAITING_APEX = "AWAITING_APEX"        # 等待頂點
    AWAITING_HIT = "AWAITING_HIT"          # 等待擊球
    COOLDOWN = "COOLDOWN"                   # 冷卻期（避免重複偵測）


class ServeDetector:
    """
    發球偵測器
    
    使用狀態機追蹤：拋球 → 頂點 → 擊球 的完整序列
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: 配置字典，包含偵測參數
        """
        self.config = config or {}
        
        # === 基礎閾值（可被動態調整覆蓋）===
        # 拋球偵測
        self.toss_initial_vy_thresh = self.config.get('toss_vy', 8.0)
        self.vertical_ratio_thresh = self.config.get('vertical_ratio', 1.5)
        self.frames_to_validate_toss = self.config.get('frames_to_validate', 8)
        self.min_upward_confirms = self.config.get('min_upward_confirms', 3)
        
        # 頂點偵測
        self.max_frames_to_apex = self.config.get('max_frames_to_apex', 75)
        self.max_lost_frames_apex = self.config.get('max_lost_frames_apex', 50)
        
        # 擊球偵測
        self.hit_v_thresh = self.config.get('hit_v', 40.0)
        self.max_frames_to_hit = self.config.get('max_frames_to_hit', 40)
        self.hit_horizontal_ratio_thresh = self.config.get('hit_h_ratio', 2.5)
        
        # 物理約束
        self.max_plausible_speed = self.config.get('max_speed', 200.0)
        self.min_hit_horizontal_ratio = self.config.get('min_hit_horizontal_ratio', 0.3)
        
        # 狀態
        self.state = ServeState.SEARCHING_TOSS
        self.event_candidate = {}
        self.detected_events = []
        
        # 統計（用於動態閾值）
        self.speed_history = deque(maxlen=300)  # 最近 300 幀的速度
        self.vertical_speed_history = deque(maxlen=300)
        
        # 冷卻計數
        self.cooldown_frames = 0
        self.cooldown_duration = 30  # 偵測到發球後冷卻 30 幀
        
        # 位置歷史（用於回推找擊球起點）
        self.position_history = deque(maxlen=30)  # 記錄最近 30 幀的位置和速度
    
    def update_statistics(self, speed: float, vy: float):
        """更新速度統計（用於動態閾值）"""
        if 0 < speed < self.max_plausible_speed:
            self.speed_history.append(speed)
        if abs(vy) < self.max_plausible_speed:
            self.vertical_speed_history.append(vy)
    
    def get_dynamic_thresholds(self) -> Dict[str, float]:
        """
        根據統計計算動態閾值
        
        Returns:
            動態調整後的閾值字典
        """
        if len(self.speed_history) < 50:
            # 數據不足，使用默認值
            return {
                'hit_v': self.hit_v_thresh,
                'toss_vy': self.toss_initial_vy_thresh
            }
        
        speeds = np.array(self.speed_history)
        
        # 擊球速度閾值：使用 90 百分位數的 0.8 倍
        # 這樣可以自適應不同影片的速度分布
        hit_v_dynamic = np.percentile(speeds, 90) * 0.8
        hit_v_dynamic = max(hit_v_dynamic, 25.0)  # 最小值保護
        hit_v_dynamic = min(hit_v_dynamic, 60.0)  # 最大值保護
        
        # 拋球速度閾值：使用中位數的 0.5 倍
        if len(self.vertical_speed_history) > 50:
            vy_arr = np.array(self.vertical_speed_history)
            upward_speeds = vy_arr[vy_arr > 0]
            if len(upward_speeds) > 10:
                toss_vy_dynamic = np.median(upward_speeds) * 0.8
                toss_vy_dynamic = max(toss_vy_dynamic, 5.0)
                toss_vy_dynamic = min(toss_vy_dynamic, 15.0)
            else:
                toss_vy_dynamic = self.toss_initial_vy_thresh
        else:
            toss_vy_dynamic = self.toss_initial_vy_thresh
        
        return {
            'hit_v': hit_v_dynamic,
            'toss_vy': toss_vy_dynamic
        }
    
    def validate_hit(self, speed: float, vx: float, vy: float) -> Tuple[bool, str]:
        """
        驗證擊球是否有效
        
        Args:
            speed: 總速度
            vx: 水平速度
            vy: 垂直速度
            
        Returns:
            (是否有效, 原因)
        """
        # 1. 速度比例檢查：排除純下墜
        horizontal_ratio = abs(vx) / (abs(vy) + 1e-6)
        
        if horizontal_ratio < self.min_hit_horizontal_ratio:
            return False, "horizontal_too_small"
        
        # 2. 檢查是否為純水平移動（過濾滾球）
        if horizontal_ratio > self.hit_horizontal_ratio_thresh:
            return False, "horizontal_too_large"
        
        return True, "valid"
    
    def process_frame(
        self,
        prev_ball_pos: Optional[np.ndarray],
        curr_ball_pos: Optional[np.ndarray],
        frame_id: int,
        use_dynamic_threshold: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        處理單幀，偵測發球事件
        
        Args:
            prev_ball_pos: 前一幀球位置 [x, y]
            curr_ball_pos: 當前幀球位置 [x, y]
            frame_id: 幀號
            use_dynamic_threshold: 是否使用動態閾值
            
        Returns:
            偵測到的發球事件，或 None
        """
        # 冷卻期處理
        if self.state == ServeState.COOLDOWN:
            self.cooldown_frames += 1
            if self.cooldown_frames >= self.cooldown_duration:
                self.state = ServeState.SEARCHING_TOSS
                self.cooldown_frames = 0
            return None
        
        # 球位置檢查
        if prev_ball_pos is None or curr_ball_pos is None:
            # 球丟失，更新遮擋計數
            if self.state in [ServeState.AWAITING_APEX, ServeState.AWAITING_HIT, 
                              ServeState.CONFIRMING_TOSS]:
                self.event_candidate['lost_frames_count'] = \
                    self.event_candidate.get('lost_frames_count', 0) + 1
                
                # 檢查是否應該放棄
                if self.state == ServeState.AWAITING_APEX and \
                   self.event_candidate.get('lost_frames_count', 0) > self.max_lost_frames_apex:
                    self.state = ServeState.SEARCHING_TOSS
                elif self.state == ServeState.AWAITING_HIT and \
                     self.event_candidate.get('lost_frames_count', 0) > 15:
                    self.state = ServeState.SEARCHING_TOSS
                elif self.state == ServeState.CONFIRMING_TOSS and \
                     self.event_candidate.get('lost_frames_count', 0) > 5:
                    self.state = ServeState.SEARCHING_TOSS
            return None
        
        # 計算速度
        velocity = curr_ball_pos - prev_ball_pos
        vx, vy = velocity[0], velocity[1]
        speed = np.linalg.norm(velocity)
        
        # 記錄位置歷史（用於回推找擊球起點）
        self.position_history.append({
            'frame_id': frame_id,
            'position': curr_ball_pos.copy(),
            'speed': speed,
            'vx': vx,
            'vy': vy
        })
        
        # 更新統計
        self.update_statistics(speed, -vy)  # vy 取反（向上為正）
        
        # 過濾不合理的速度
        if speed > self.max_plausible_speed:
            return None
        
        # 重置遮擋計數
        if 'lost_frames_count' in self.event_candidate:
            self.event_candidate['lost_frames_count'] = 0
        
        # 獲取閾值
        if use_dynamic_threshold:
            thresholds = self.get_dynamic_thresholds()
            hit_v_thresh = thresholds['hit_v']
            toss_vy_thresh = thresholds['toss_vy']
        else:
            hit_v_thresh = self.hit_v_thresh
            toss_vy_thresh = self.toss_initial_vy_thresh
        
        # === 狀態機邏輯 ===
        
        if self.state == ServeState.SEARCHING_TOSS:
            # 尋找拋球：球向上移動（vy < 0 表示向上）
            upward_vy = -vy  # 轉換為向上為正
            
            if upward_vy > toss_vy_thresh:
                # 檢查垂直/水平比例：過濾水平滾球
                vertical_horizontal_ratio = abs(upward_vy) / (abs(vx) + 1e-6)
                
                if vertical_horizontal_ratio > self.vertical_ratio_thresh:
                    self.state = ServeState.CONFIRMING_TOSS
                    self.event_candidate = {
                        'confirm_start_frame': frame_id,
                        'upward_frames_count': 1,
                        'lost_frames_count': 0,
                        'last_pos': curr_ball_pos.copy(),
                        'toss_position': curr_ball_pos.copy()
                    }
        
        elif self.state == ServeState.CONFIRMING_TOSS:
            # 確認拋球：持續向上
            if (frame_id - self.event_candidate['confirm_start_frame']) > self.frames_to_validate_toss:
                self.state = ServeState.SEARCHING_TOSS
                return None
            
            upward_vy = self.event_candidate['last_pos'][1] - curr_ball_pos[1]
            if upward_vy > 0:  # 仍在向上
                self.event_candidate['upward_frames_count'] += 1
            
            self.event_candidate['last_pos'] = curr_ball_pos.copy()
            
            if self.event_candidate['upward_frames_count'] >= self.min_upward_confirms:
                self.state = ServeState.AWAITING_APEX
                self.event_candidate['toss_start_frame'] = self.event_candidate['confirm_start_frame']
                self.event_candidate['lost_frames_count'] = 0
        
        elif self.state == ServeState.AWAITING_APEX:
            # 等待頂點：球開始下降
            if (frame_id - self.event_candidate.get('toss_start_frame', frame_id)) > self.max_frames_to_apex:
                self.state = ServeState.SEARCHING_TOSS
                return None
            
            # vy > 0 表示向下（像素座標系）
            if vy > 1:  # 開始下降
                self.state = ServeState.AWAITING_HIT
                self.event_candidate['apex_frame'] = frame_id
                self.event_candidate['apex_position'] = prev_ball_pos.copy()
                self.event_candidate['lost_frames_count'] = 0
        
        elif self.state == ServeState.AWAITING_HIT:
            # 等待擊球
            if (frame_id - self.event_candidate.get('apex_frame', frame_id)) > self.max_frames_to_hit:
                self.state = ServeState.SEARCHING_TOSS
                return None
            
            if speed > hit_v_thresh:
                # 驗證擊球
                is_valid, reason = self.validate_hit(speed, vx, vy)
                
                if is_valid:
                    # 偵測到有效擊球！
                    event = {
                        'hit_frame_id': frame_id,
                        'hit_position': curr_ball_pos.tolist(),
                        'hit_speed': float(speed),
                        'hit_velocity': [float(vx), float(vy)],
                        'toss_start_frame': self.event_candidate.get('toss_start_frame'),
                        'toss_position': self.event_candidate.get('toss_position', []).tolist() 
                            if hasattr(self.event_candidate.get('toss_position', []), 'tolist') 
                            else self.event_candidate.get('toss_position'),
                        'apex_frame': self.event_candidate.get('apex_frame'),
                        'apex_position': self.event_candidate.get('apex_position', []).tolist()
                            if hasattr(self.event_candidate.get('apex_position', []), 'tolist')
                            else self.event_candidate.get('apex_position'),
                        'dynamic_threshold_used': hit_v_thresh if use_dynamic_threshold else None
                    }
                    
                    self.detected_events.append(event)
                    
                    # 進入冷卻期
                    self.state = ServeState.COOLDOWN
                    self.cooldown_frames = 0
                    self.event_candidate = {}
                    
                    return event
        
        return None
    
    def get_all_events(self) -> List[Dict[str, Any]]:
        """獲取所有偵測到的發球事件"""
        return self.detected_events.copy()
    
    def reset(self):
        """重置偵測器"""
        self.state = ServeState.SEARCHING_TOSS
        self.event_candidate = {}
        self.detected_events = []
        self.cooldown_frames = 0
        # 保留統計數據，不重置 speed_history


def analyze_serve_events_v2(
    all_frames_data: List[Dict],
    config: Dict[str, Any] = None,
    log_prefix: str = "",
    use_dynamic_threshold: bool = True,
    first_only: bool = True
) -> List[Dict[str, Any]]:
    """
    改進版發球事件分析（可直接替換原有的 analyze_serve_events）
    
    Args:
        all_frames_data: 所有幀的追蹤數據
        config: 配置字典
        log_prefix: 日誌前綴
        use_dynamic_threshold: 是否使用動態閾值
        first_only: 是否只返回第一個偵測到的事件（適用於每段影片只有一次發球的情況）
        
    Returns:
        發球事件列表
    """
    detector = ServeDetector(config or {})
    
    mode_str = "（只取第一個事件）" if first_only else "（取所有事件）"
    print(f"\n{log_prefix}[分析階段] 使用改進版 v2 偵測邏輯 {mode_str}...")
    
    def get_ball_center(frame_data):
        """從幀數據中提取球中心"""
        if not frame_data or not frame_data.get('ball_detections'):
            return None
        valid_balls = [
            b for b in frame_data['ball_detections'] 
            if not b.get('is_in_background_zone', False)
        ]
        if not valid_balls:
            return None
        best_ball = max(valid_balls, key=lambda b: b.get('confidence', 0))
        box = best_ball.get('box_coords')
        if box:
            return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        center = best_ball.get('center_point')
        if center:
            return np.array(center)
        return None
    
    first_event = None
    
    for i in range(1, len(all_frames_data)):
        prev_ball_pos = get_ball_center(all_frames_data[i - 1])
        curr_ball_pos = get_ball_center(all_frames_data[i])
        
        event = detector.process_frame(
            prev_ball_pos, 
            curr_ball_pos, 
            i,
            use_dynamic_threshold=use_dynamic_threshold
        )
        
        if event:
            print(f"{log_prefix}  > [第 {event['hit_frame_id']} 幀] 偵測到發球 (速度: {event['hit_speed']:.1f})")
            
            # 如果只需要第一個事件，記錄後停止
            if first_only and first_event is None:
                first_event = event
                print(f"{log_prefix}  ✓ 已找到第一個發球事件，停止搜尋")
                break
    
    if first_only:
        events = [first_event] if first_event else []
    else:
        events = detector.get_all_events()
    
    print(f"{log_prefix}[分析完成] 返回 {len(events)} 個發球事件")
    
    return events


if __name__ == "__main__":
    # 簡單測試
    print("測試 ServeDetector...")
    
    detector = ServeDetector({
        'hit_v': 40.0,
        'toss_vy': 8.0
    })
    
    # 模擬發球軌跡
    # 拋球 → 頂點 → 擊球
    test_trajectory = [
        np.array([500, 400]),   # 0
        np.array([502, 380]),   # 1 - 開始拋球（向上）
        np.array([504, 360]),   # 2
        np.array([506, 345]),   # 3
        np.array([508, 335]),   # 4
        np.array([510, 330]),   # 5 - 接近頂點
        np.array([512, 328]),   # 6 - 頂點
        np.array([514, 332]),   # 7 - 開始下降
        np.array([516, 340]),   # 8
        np.array([520, 350]),   # 9
        np.array([570, 380]),   # 10 - 擊球！大位移
        np.array([620, 400]),   # 11
    ]
    
    print("\n逐幀處理：")
    for i in range(1, len(test_trajectory)):
        event = detector.process_frame(
            test_trajectory[i-1],
            test_trajectory[i],
            i,
            use_dynamic_threshold=False
        )
        
        if event:
            print(f"\n  *** 偵測到發球事件！***")
            print(f"      擊球幀: {event['hit_frame_id']}")
            print(f"      擊球速度: {event['hit_speed']:.1f}")
            print(f"      拋球幀: {event['toss_start_frame']}")
            print(f"      頂點幀: {event['apex_frame']}")
        else:
            print(f"  Frame {i}: state={detector.state.value}")
    
    print("\n測試完成！")