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

        # 解析度縮放（以 720p 為基準）
        image_height = self.config.get('image_height', 720)
        resolution_scale = image_height / 720.0

        # 最小拋球高度：排除 rally 球短暫向上的 false positive
        # 真正的發球拋球至少要上升 60px（@720p）
        # 注意：toss_position 從 prev_ball_pos 記錄以減少因偵測延遲造成的低估
        self.min_toss_height = self.config.get('min_toss_height', 60.0) * resolution_scale

        # [Method A] 球員距離約束：拋球起點需在某個球員上半身附近
        # 軟性約束：若無球員資訊則不拒絕；避免假陽性（rally 中球不在發球員旁）
        # 注意：因攝影機透視角度使場地空間座標判斷不可靠，改用球員相對距離
        self.player_proximity_thresh = self.config.get('player_proximity_thresh', 150.0) * resolution_scale

        # [Method C] net_y 約束：拋球起點需在 net_y + margin 以上（far side / near net）
        # 真實發球：球在遠端底線附近拋起，Y 值接近或小於 net_y
        # 假陽性：接球後球從近端（大 Y）反彈上升，Y 遠大於 net_y
        # 注意：net_y 的 Y 座標來自 court_config，是水平線，不受透視扭曲影響
        #       margin 預設 130px（@720p）提供緩衝，避免 net_y 校準誤差
        #       比舊版 120px 多 10px，防止真實發球球第一次偵測點恰好在邊界附近被誤拒
        self.net_y = self.config.get('net_y', None)  # None = 不啟用此約束
        self.net_y_toss_margin = self.config.get('net_y_toss_margin', 130.0) * resolution_scale
        
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
    
    def _estimate_contact_frame(self, detected_frame_id, detected_pos, hit_v_thresh):
        """
        回推估算實際擊球接觸幀

        球速超過閾值時，球通常已飛離發球員一段距離（尤其擊球瞬間
        手臂遮擋導致數幀無球偵測，重新偵測時球已遠離）。

        回溯 position_history 找到速度突然增加前的最後慢速幀，
        使用其下一幀作為更接近實際擊球的時間點。

        Returns:
            (estimated_frame_id, estimated_position, original_frame_id)
        """
        if len(self.position_history) < 3:
            return detected_frame_id, detected_pos, detected_frame_id

        slow_threshold = hit_v_thresh * 0.5
        history_len = len(self.position_history)
        search_start = max(0, history_len - 15)

        # 從後往前搜尋最後一個慢速幀
        for i in range(history_len - 2, search_start - 1, -1):
            entry = self.position_history[i]
            if entry['speed'] < slow_threshold:
                # 找到慢速幀，使用下一幀作為估算的擊球幀
                next_idx = i + 1
                if next_idx < history_len:
                    next_entry = self.position_history[next_idx]
                    return next_entry['frame_id'], next_entry['position'], detected_frame_id
                return entry['frame_id'], entry['position'], detected_frame_id

        # 沒找到明確的慢速幀，使用前一個 history 項
        prev = self.position_history[-2]
        return prev['frame_id'], prev['position'], detected_frame_id

    def _is_near_player(self, ball_pos: np.ndarray, player_detections) -> bool:
        """
        [Method A] 檢查球是否在某個球員的上半身附近。
        軟性約束：
          - 無球員資訊（player_detections 為空）→ 不拒絕
          - 所有球員都在球的對側半場（球 X 超出所有球員 X 範圍）→ 不拒絕
            （發球員可能在畫面邊緣未被偵測到，避免誤拒真實發球）
        設計理由：
          - 真實拋球：球在發球員手部附近，但發球員可能未被偵測到（邊緣位置）
          - Rally 誤判：球在中場飛行，通常已遠離最近的球員（測試用）
          - 攝影機透視角度使場地座標判斷不可靠，只用相對距離
        注意：此方法目前為備用軟性約束，主要過濾由 Method D + Method C 提供
        """
        if not player_detections:
            return True  # 無球員資訊 → 不拒絕

        # 計算所有球員的 X 範圍
        x_coords = []
        for player in player_detections:
            box = player.get('box_coords')
            if box and len(box) >= 4:
                x_coords.append((box[0] + box[2]) / 2)

        # 若球的 X 超出所有偵測球員的 X 範圍（發球員在畫面邊緣未被偵測）→ 不拒絕
        if x_coords:
            min_px, max_px = min(x_coords), max(x_coords)
            margin_x = self.player_proximity_thresh
            if ball_pos[0] < min_px - margin_x or ball_pos[0] > max_px + margin_x:
                return True  # 球在球員群體外側 → 可能是邊緣發球員，不拒絕

        for player in player_detections:
            box = player.get('box_coords')
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = box[:4]
            # 上半身中心點：bbox 上方 40% 處（手腕 / 肩膀 / 擊球手臂範圍）
            px = (x1 + x2) / 2
            py = y1 + (y2 - y1) * 0.4
            dist = np.sqrt((ball_pos[0] - px) ** 2 + (ball_pos[1] - py) ** 2)
            if dist < self.player_proximity_thresh:
                return True
        return False

    def _was_ball_descending_recently(self, frame_id: int, window: int = 10,
                                       min_entries: int = 3,
                                       threshold_pct: float = 0.6) -> bool:
        """
        [Method D] 檢查 position_history 最近 N 幀中球是否在持續下降（Y 增加）。

        設計邏輯：
          - 接球假陽性：接球員接球前，球從遠端飛向近端（Y 持續增加），
            然後被接球後反彈向上 → 觸發 CONFIRMING_TOSS。
            此時 position_history 的近期軌跡呈 Y 增加（下降）趨勢。
          - 真實發球：球在發球員手中（VballNet 無法偵測），拋起時才首次出現，
            position_history 缺乏足夠的「前置下降」資料 → len(recent) < min_entries。
          - 注意：window=10 幀足夠判斷接球前的下降軌跡，
            但不會追溯太久而錯誤抓到更早的無關下降段。

        Returns:
            True  = 最近有下降軌跡，視為接球反彈假陽性（應拒絕）
            False = 歷史不足或沒有明確下降趨勢（可能是真實發球）
        """
        # 取 position_history 中 frame_id-window ~ frame_id-1 的資料
        recent = sorted(
            [e for e in self.position_history
             if frame_id - window <= e['frame_id'] < frame_id],
            key=lambda e: e['frame_id']
        )
        if len(recent) < min_entries:
            return False  # 歷史不足，不拒絕（可能是真實發球剛出現）

        y_vals = [e['position'][1] for e in recent]
        n_transitions = len(y_vals) - 1
        # 計算 Y 增加（球下降）的轉換次數（容差 3px 避免雜訊影響）
        n_increasing = sum(
            1 for i in range(n_transitions) if y_vals[i + 1] > y_vals[i] + 3
        )

        # 超過 threshold_pct（預設 60%）的轉換都是下降 → 視為接球軌跡
        return n_increasing >= max(2, n_transitions * threshold_pct)

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
        use_dynamic_threshold: bool = True,
        player_detections: Optional[list] = None
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
                # 檢查垂直/水平比例：過濾水平滾球 & 接球弧線（需 > 2.5，更嚴格）
                vertical_horizontal_ratio = abs(upward_vy) / (abs(vx) + 1e-6)

                if vertical_horizontal_ratio > self.vertical_ratio_thresh:
                    # [Method D] 下降軌跡檢查（主要過濾器）：
                    # 若最近幀 Y 持續增加（球下降朝近端），視為接球反彈假陽性
                    # 真實發球：球在發球員手中無法偵測，拋起後才首次出現，history 不足
                    # 接球假陽性：ball 從遠端飛向近端（Y 增加）後被接起，history 有明確下降段
                    if self._was_ball_descending_recently(frame_id):
                        return None

                    # [Method C] net_y 約束（備用過濾器）：拋球起點需在 net_y + margin 以上
                    # 問題：球上升後觸發 CONFIRMING_TOSS 時，curr_ball_pos 已高於起點。
                    # 解法：只看最近 15 幀的 position_history（避免前一分的舊數據污染），
                    #       比取全部 history 更可靠，避免早期 rally 的大 Y 值干擾判斷。
                    # 真實發球：
                    #   (a) 球剛出現（history 近 15 幀為空）→ ref_y = prev_ball_pos（球起點）
                    #   (b) 球從 net 附近拋起 → ref_y 小，不超過 threshold
                    # 假陽性：球在近端被接球後上升 → 最近 15 幀有 Y >> net_y 的記錄
                    if self.net_y is not None:
                        # 只取最近 15 幀的 history（排除太舊的前幾分數據）
                        recent_15 = [e for e in self.position_history
                                     if frame_id - 15 <= e['frame_id'] < frame_id]
                        if recent_15:
                            max_hist_y = max(e['position'][1] for e in recent_15)
                            ref_y = max(max_hist_y, prev_ball_pos[1])
                        else:
                            # 近 15 幀無歷史（球剛出現），只用 prev_ball_pos
                            ref_y = prev_ball_pos[1]
                        if ref_y > self.net_y + self.net_y_toss_margin:
                            # 球從近端深處上升，不可能是發球拋球
                            return None

                    # [Method A] 球員距離約束：拋球起點需在某球員上半身附近
                    # 軟性約束：若無球員資料則通過
                    if self._is_near_player(prev_ball_pos, player_detections):
                        self.state = ServeState.CONFIRMING_TOSS
                        self.event_candidate = {
                            'confirm_start_frame': frame_id,
                            'upward_frames_count': 1,
                            'lost_frames_count': 0,
                            'last_pos': curr_ball_pos.copy(),
                            # 用 prev_ball_pos 記錄拋球起點，減少因偵測延遲的高度低估
                            'toss_position': prev_ball_pos.copy()
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
                # 檢查拋球高度是否足夠（過濾 rally 球短暫向上的 false positive）
                toss_start_y = self.event_candidate.get('toss_position', curr_ball_pos)[1]
                apex_y = prev_ball_pos[1]
                toss_height = toss_start_y - apex_y  # 正值 = 球向上升

                if toss_height < self.min_toss_height:
                    # 拋球高度不足，視為 false positive（如 rally 擊球）
                    self.state = ServeState.SEARCHING_TOSS
                    self.event_candidate = {}
                    return None

                self.state = ServeState.AWAITING_HIT
                self.event_candidate['apex_frame'] = frame_id
                self.event_candidate['apex_position'] = prev_ball_pos.copy()
                self.event_candidate['lost_frames_count'] = 0
                self.event_candidate['toss_height_px'] = float(toss_height)
                self.event_candidate['min_toss_height_px'] = float(self.min_toss_height)
        
        elif self.state == ServeState.AWAITING_HIT:
            # 等待擊球
            if (frame_id - self.event_candidate.get('apex_frame', frame_id)) > self.max_frames_to_hit:
                self.state = ServeState.SEARCHING_TOSS
                return None
            
            if speed > hit_v_thresh:
                # 驗證擊球
                is_valid, reason = self.validate_hit(speed, vx, vy)
                
                if is_valid:
                    # 回推實際擊球接觸幀
                    est_frame, est_pos, orig_frame = self._estimate_contact_frame(
                        frame_id, curr_ball_pos, hit_v_thresh
                    )
                    est_pos_list = est_pos.tolist() if hasattr(est_pos, 'tolist') else list(est_pos)

                    # 偵測到有效擊球！
                    event = {
                        'hit_frame_id': est_frame,
                        'hit_frame_id_detected': orig_frame,
                        'hit_position': est_pos_list,
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
                        'dynamic_threshold_used': hit_v_thresh if use_dynamic_threshold else None,
                        'toss_height_px': self.event_candidate.get('toss_height_px'),
                        'min_toss_height_px': self.event_candidate.get('min_toss_height_px'),
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
    first_only: bool = True,
    image_height: int = 720
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
    cfg = dict(config or {})
    cfg.setdefault('image_height', image_height)
    detector = ServeDetector(cfg)
    
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
        
        # [Method A] 傳入當前幀的球員偵測（軟性約束，無資料時自動跳過）
        curr_players = all_frames_data[i].get('player_detections', []) if all_frames_data[i] else []

        event = detector.process_frame(
            prev_ball_pos,
            curr_ball_pos,
            i,
            use_dynamic_threshold=use_dynamic_threshold,
            player_detections=curr_players
        )
        
        if event:
            orig = event.get('hit_frame_id_detected', event['hit_frame_id'])
            if orig != event['hit_frame_id']:
                print(f"{log_prefix}  > [第 {event['hit_frame_id']} 幀] 偵測到發球 (速度: {event['hit_speed']:.1f}, 原始偵測幀: {orig})")
            else:
                print(f"{log_prefix}  > [第 {event['hit_frame_id']} 幀] 偵測到發球 (速度: {event['hit_speed']:.1f})")
            
            # 如果只需要第一個事件，記錄後停止
            if first_only and first_event is None:
                first_event = event
                print(f"{log_prefix}  [OK] 已找到第一個發球事件，停止搜尋")
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