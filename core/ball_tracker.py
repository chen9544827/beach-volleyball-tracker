# core/ball_tracker.py
# -*- coding: utf-8 -*-
"""
球追蹤器模組 - 解決遮擋問題的核心

功能：
1. KalmanBallFilter: 卡爾曼濾波器，平滑軌跡並預測位置
2. ParabolaPredictor: 拋物線預測器，處理短期遮擋
3. BallTracker: 整合追蹤器，管理球的狀態

使用方式：
    tracker = BallTracker()
    for frame in frames:
        detection = detect_ball(frame)  # 可能為 None（被遮擋）
        tracked_ball = tracker.update(detection)
        # tracked_ball 包含預測位置，即使偵測失敗
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Dict, Any


class KalmanBallFilter:
    """
    2D 卡爾曼濾波器用於球的位置追蹤
    
    狀態向量: [x, y, vx, vy] (位置 + 速度)
    觀測向量: [x, y] (僅位置)
    
    特點：
    - 考慮重力加速度（沙灘排球物理）
    - 自適應過程雜訊（根據運動狀態調整）
    """
    
    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 1.0):
        """
        初始化卡爾曼濾波器
        
        Args:
            process_noise: 過程雜訊強度（越大越信任觀測）
            measurement_noise: 觀測雜訊強度（越大越信任預測）
        """
        # 狀態向量 [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # 狀態協方差矩陣
        self.P = np.eye(4) * 1000  # 初始不確定性很高
        
        # 狀態轉移矩陣 (dt=1 frame)
        self.F = np.array([
            [1, 0, 1, 0],   # x = x + vx
            [0, 1, 0, 1],   # y = y + vy
            [0, 0, 1, 0],   # vx = vx
            [0, 0, 0, 1],   # vy = vy
        ], dtype=float)
        
        # 觀測矩陣 (只觀測位置)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        # 過程雜訊協方差
        self.Q = np.eye(4) * process_noise
        self.Q[2, 2] = process_noise * 2  # vx 不確定性更高
        self.Q[3, 3] = process_noise * 2  # vy 不確定性更高
        
        # 觀測雜訊協方差
        self.R = np.eye(2) * measurement_noise
        
        # 重力加速度（像素/幀²，需根據實際調整）
        # 30fps 影片中，重力約為 0.5-1.0 像素/幀²
        self.gravity = 0.5
        
        self.initialized = False
    
    def initialize(self, x: float, y: float, vx: float = 0, vy: float = 0):
        """初始化狀態"""
        self.state = np.array([x, y, vx, vy])
        self.P = np.eye(4) * 100  # 降低初始不確定性
        self.initialized = True
    
    def predict(self) -> np.ndarray:
        """
        預測下一幀的位置
        
        Returns:
            預測的 [x, y] 位置
        """
        if not self.initialized:
            return None
        
        # 狀態預測：x = Fx + 重力影響
        self.state = self.F @ self.state
        self.state[3] += self.gravity  # vy 受重力影響（向下為正）
        
        # 協方差預測
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state[:2].copy()
    
    def update(self, measurement: Optional[np.ndarray]) -> np.ndarray:
        """
        根據觀測更新狀態
        
        Args:
            measurement: 觀測到的 [x, y] 位置，None 表示未觀測到
            
        Returns:
            更新後的 [x, y] 位置
        """
        if not self.initialized:
            if measurement is not None:
                self.initialize(measurement[0], measurement[1])
                return measurement
            return None
        
        if measurement is None:
            # 沒有觀測，只用預測
            # 增加不確定性
            self.P *= 1.1
            return self.state[:2].copy()
        
        # 卡爾曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 狀態更新
        z = np.array(measurement)
        y = z - self.H @ self.state  # 觀測殘差
        self.state = self.state + K @ y
        
        # 協方差更新
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state[:2].copy()
    
    def get_velocity(self) -> np.ndarray:
        """獲取當前速度估計 [vx, vy]"""
        return self.state[2:4].copy() if self.initialized else np.zeros(2)
    
    def get_speed(self) -> float:
        """獲取當前速度大小"""
        vel = self.get_velocity()
        return np.linalg.norm(vel)
    
    def reset(self):
        """重置濾波器"""
        self.state = np.zeros(4)
        self.P = np.eye(4) * 1000
        self.initialized = False


class ParabolaPredictor:
    """
    拋物線預測器 - 用於短期遮擋時的位置預測
    
    基於物理原理：排球軌跡近似拋物線
    y = y0 + vy0*t + 0.5*g*t²
    x = x0 + vx0*t
    """
    
    def __init__(self, history_length: int = 10):
        """
        Args:
            history_length: 保留的歷史位置數量
        """
        self.positions = deque(maxlen=history_length)
        self.timestamps = deque(maxlen=history_length)  # 幀號
        self.gravity = 0.5  # 像素/幀²
    
    def add_position(self, x: float, y: float, frame_id: int):
        """添加觀測到的位置"""
        self.positions.append(np.array([x, y]))
        self.timestamps.append(frame_id)
    
    def predict(self, target_frame: int) -> Optional[np.ndarray]:
        """
        預測指定幀的位置
        
        Args:
            target_frame: 目標幀號
            
        Returns:
            預測的 [x, y] 位置，或 None（數據不足）
        """
        if len(self.positions) < 3:
            return None
        
        # 使用最近的 3-5 個點擬合
        n_points = min(5, len(self.positions))
        positions = list(self.positions)[-n_points:]
        frames = list(self.timestamps)[-n_points:]
        
        # 計算速度（使用最近兩點）
        dt = frames[-1] - frames[-2]
        if dt == 0:
            dt = 1
        
        vx = (positions[-1][0] - positions[-2][0]) / dt
        vy = (positions[-1][1] - positions[-2][1]) / dt
        
        # 預測
        t = target_frame - frames[-1]  # 時間差（幀數）
        
        pred_x = positions[-1][0] + vx * t
        pred_y = positions[-1][1] + vy * t + 0.5 * self.gravity * t * t
        
        return np.array([pred_x, pred_y])
    
    def clear(self):
        """清空歷史"""
        self.positions.clear()
        self.timestamps.clear()


class BallTracker:
    """
    球追蹤器 - 整合偵測、濾波和預測
    
    主要功能：
    1. 管理球的追蹤狀態（追蹤中/遮擋中/丟失）
    2. 在偵測失敗時使用預測
    3. 提供平滑的軌跡
    
    狀態轉換：
    TRACKING ─(偵測失敗)→ OCCLUDED ─(持續失敗)→ LOST
        ↑                     │
        └────(偵測成功)────────┘
    """
    
    # 追蹤狀態
    STATE_TRACKING = "TRACKING"      # 正常追蹤
    STATE_OCCLUDED = "OCCLUDED"      # 被遮擋，使用預測
    STATE_LOST = "LOST"              # 完全丟失
    
    def __init__(
        self,
        max_occlusion_frames: int = 15,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0
    ):
        """
        Args:
            max_occlusion_frames: 最大允許遮擋幀數
            process_noise: 卡爾曼濾波過程雜訊
            measurement_noise: 卡爾曼濾波觀測雜訊
        """
        self.kalman = KalmanBallFilter(process_noise, measurement_noise)
        self.parabola = ParabolaPredictor()
        
        self.state = self.STATE_LOST
        self.current_frame = 0
        self.occlusion_start_frame = 0
        self.max_occlusion_frames = max_occlusion_frames
        
        # 追蹤歷史
        self.trajectory = []  # [(frame_id, x, y, is_predicted), ...]
        self.last_detection = None
        self.consecutive_misses = 0
    
    def update(
        self,
        detection: Optional[Dict[str, Any]],
        frame_id: int
    ) -> Dict[str, Any]:
        """
        更新追蹤狀態
        
        Args:
            detection: 球的偵測結果，格式同 detect_ball() 返回
                      None 表示該幀未偵測到球
            frame_id: 當前幀號
            
        Returns:
            追蹤結果字典：
            {
                "position": [x, y],      # 位置（可能是預測）
                "velocity": [vx, vy],    # 速度估計
                "speed": float,          # 速度大小
                "is_predicted": bool,    # 是否為預測值
                "state": str,            # 追蹤狀態
                "confidence": float,     # 信心度
                "occlusion_frames": int  # 遮擋幀數
            }
        """
        self.current_frame = frame_id
        
        # 提取球的中心位置
        ball_pos = None
        confidence = 0.0
        if detection is not None:
            if isinstance(detection, dict):
                if 'center_point' in detection:
                    ball_pos = np.array(detection['center_point'], dtype=float)
                    confidence = detection.get('confidence', 0.5)
                elif 'box_coords' in detection:
                    box = detection['box_coords']
                    ball_pos = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])
                    confidence = detection.get('confidence', 0.5)
            elif isinstance(detection, (list, tuple, np.ndarray)) and len(detection) >= 2:
                ball_pos = np.array(detection[:2], dtype=float)
                confidence = 0.5
        
        # 卡爾曼預測
        predicted_pos = self.kalman.predict()
        
        # 根據是否有偵測結果更新狀態
        if ball_pos is not None:
            # 有偵測結果
            self.consecutive_misses = 0
            
            # 如果之前在遮擋狀態，驗證偵測是否合理
            if self.state == self.STATE_OCCLUDED and predicted_pos is not None:
                distance = np.linalg.norm(ball_pos - predicted_pos)
                if distance > 100:  # 距離預測太遠，可能是誤偵測
                    # 降低信心度但仍使用偵測結果
                    confidence *= 0.5
            
            # 卡爾曼更新
            updated_pos = self.kalman.update(ball_pos)
            
            # 更新拋物線預測器
            self.parabola.add_position(ball_pos[0], ball_pos[1], frame_id)
            
            # 更新狀態
            self.state = self.STATE_TRACKING
            self.last_detection = ball_pos
            
            # 記錄軌跡
            self.trajectory.append((frame_id, updated_pos[0], updated_pos[1], False))
            
            return {
                "position": updated_pos.tolist(),
                "velocity": self.kalman.get_velocity().tolist(),
                "speed": self.kalman.get_speed(),
                "is_predicted": False,
                "state": self.state,
                "confidence": confidence,
                "occlusion_frames": 0
            }
        
        else:
            # 沒有偵測結果
            self.consecutive_misses += 1
            
            if self.state == self.STATE_TRACKING:
                # 剛開始遮擋
                self.state = self.STATE_OCCLUDED
                self.occlusion_start_frame = frame_id
            
            occlusion_frames = frame_id - self.occlusion_start_frame
            
            if self.state == self.STATE_OCCLUDED:
                if occlusion_frames > self.max_occlusion_frames:
                    # 遮擋太久，判定為丟失
                    self.state = self.STATE_LOST
                    self.kalman.reset()
                    self.parabola.clear()
                    return {
                        "position": None,
                        "velocity": [0, 0],
                        "speed": 0,
                        "is_predicted": True,
                        "state": self.state,
                        "confidence": 0,
                        "occlusion_frames": occlusion_frames
                    }
                
                # 使用拋物線預測
                parabola_pred = self.parabola.predict(frame_id)
                
                # 卡爾曼更新（無觀測）
                kalman_pred = self.kalman.update(None)
                
                # 融合兩種預測（偏向拋物線，因為更符合物理）
                if parabola_pred is not None and kalman_pred is not None:
                    # 加權融合，越久越信任卡爾曼（因為拋物線假設會累積誤差）
                    w = min(occlusion_frames / 10, 0.5)  # 最多 50% 權重給卡爾曼
                    final_pred = (1 - w) * parabola_pred + w * kalman_pred
                elif parabola_pred is not None:
                    final_pred = parabola_pred
                elif kalman_pred is not None:
                    final_pred = kalman_pred
                else:
                    final_pred = self.last_detection
                
                if final_pred is not None:
                    # 計算信心度（隨遮擋時間遞減）
                    pred_confidence = max(0.1, 1.0 - occlusion_frames / self.max_occlusion_frames)
                    
                    self.trajectory.append((frame_id, final_pred[0], final_pred[1], True))
                    
                    return {
                        "position": final_pred.tolist() if isinstance(final_pred, np.ndarray) else list(final_pred),
                        "velocity": self.kalman.get_velocity().tolist(),
                        "speed": self.kalman.get_speed(),
                        "is_predicted": True,
                        "state": self.state,
                        "confidence": pred_confidence,
                        "occlusion_frames": occlusion_frames
                    }
            
            # 完全丟失狀態
            return {
                "position": None,
                "velocity": [0, 0],
                "speed": 0,
                "is_predicted": True,
                "state": self.state,
                "confidence": 0,
                "occlusion_frames": self.consecutive_misses
            }
    
    def get_trajectory(self, last_n: Optional[int] = None) -> List[Tuple]:
        """
        獲取軌跡歷史
        
        Args:
            last_n: 只返回最近 N 個點，None 返回全部
            
        Returns:
            [(frame_id, x, y, is_predicted), ...]
        """
        if last_n is None:
            return self.trajectory.copy()
        return self.trajectory[-last_n:]
    
    def get_last_position(self) -> Optional[np.ndarray]:
        """Get the last known ball position, or None if no trajectory."""
        if not self.trajectory:
            return None
        t = self.trajectory[-1]
        return np.array([t[1], t[2]])

    def get_recent_positions(self, n: int = 5) -> List[np.ndarray]:
        """獲取最近 N 個位置（用於速度計算等）"""
        recent = self.trajectory[-n:] if len(self.trajectory) >= n else self.trajectory
        return [np.array([t[1], t[2]]) for t in recent]
    
    def reset(self):
        """重置追蹤器"""
        self.kalman.reset()
        self.parabola.clear()
        self.state = self.STATE_LOST
        self.trajectory = []
        self.last_detection = None
        self.consecutive_misses = 0


# ============================================================
# 輔助函數：與現有 track_ball_and_player.py 整合
# ============================================================

def create_tracker_from_config(config: Dict[str, Any] = None) -> BallTracker:
    """
    根據配置創建追蹤器
    
    Args:
        config: 配置字典，可包含：
            - max_occlusion_frames: 最大遮擋幀數 (default: 15)
            - process_noise: 過程雜訊 (default: 1.0)
            - measurement_noise: 觀測雜訊 (default: 1.0)
    """
    if config is None:
        config = {}
    
    return BallTracker(
        max_occlusion_frames=config.get('max_occlusion_frames', 15),
        process_noise=config.get('process_noise', 1.0),
        measurement_noise=config.get('measurement_noise', 1.0)
    )


def process_frame_with_tracking(
    tracker: BallTracker,
    ball_detections: List[Dict],
    frame_id: int,
    background_ball_zones: List[Dict] = None
) -> Dict[str, Any]:
    """
    處理單幀的球偵測，整合追蹤器
    
    Args:
        tracker: BallTracker 實例
        ball_detections: detect_ball() 的返回結果
        frame_id: 幀號
        background_ball_zones: 背景球過濾區域
        
    Returns:
        追蹤結果
    """
    # 過濾背景區域的球
    valid_balls = [
        b for b in ball_detections 
        if not b.get('is_in_background_zone', False)
    ]
    
    # 選擇信心度最高的球
    best_detection = None
    if valid_balls:
        best_detection = max(valid_balls, key=lambda b: b.get('confidence', 0))
    
    # 更新追蹤器
    return tracker.update(best_detection, frame_id)


if __name__ == "__main__":
    # 簡單測試
    print("測試 BallTracker...")
    
    tracker = BallTracker(max_occlusion_frames=10)
    
    # 模擬軌跡：拋物線 + 遮擋
    test_positions = [
        {"center_point": [100, 300], "confidence": 0.9},  # frame 0
        {"center_point": [120, 280], "confidence": 0.9},  # frame 1
        {"center_point": [140, 265], "confidence": 0.9},  # frame 2
        {"center_point": [160, 255], "confidence": 0.9},  # frame 3
        {"center_point": [180, 250], "confidence": 0.9},  # frame 4 (頂點)
        None,  # frame 5 (遮擋開始)
        None,  # frame 6
        None,  # frame 7
        {"center_point": [260, 290], "confidence": 0.8},  # frame 8 (恢復)
        {"center_point": [280, 310], "confidence": 0.9},  # frame 9
    ]
    
    print("\n逐幀追蹤結果：")
    for i, det in enumerate(test_positions):
        result = tracker.update(det, i)
        pos_str = f"({result['position'][0]:.1f}, {result['position'][1]:.1f})" if result['position'] else "None"
        print(f"  Frame {i}: pos={pos_str}, state={result['state']}, "
              f"predicted={result['is_predicted']}, confidence={result['confidence']:.2f}")
    
    print("\n軌跡歷史：")
    for t in tracker.get_trajectory():
        pred_mark = "*" if t[3] else " "
        print(f"  {pred_mark} Frame {t[0]}: ({t[1]:.1f}, {t[2]:.1f})")
    
    print("\n測試完成！")
