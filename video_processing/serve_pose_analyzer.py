import numpy as np
import cv2
import math

# COCO 關鍵點索引定義
KEYPOINTS = {
    'nose': 0,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

class ServePoseAnalyzer:
    CONFIDENCE_THRESHOLD = 0.5
    CONSECUTIVE_FRAMES_REQUIRED = 5
    MIN_PHASES_REQUIRED = 3
    
    # 定義發球階段
    SERVE_PHASES = {
        'preparation': '準備階段',
        'jump': '跳躍階段',
        'contact': '擊球階段',
        'follow_through': '跟進階段'
    }
    
    # 定義角度閾值
    ANGLE_THRESHOLDS = {
        'arm_angle': {
            'min': 90,  # 最小手臂角度
            'max': 180  # 最大手臂角度
        },
        'knee_angle': {
            'min': 90,  # 最小膝蓋角度
            'max': 180  # 最大膝蓋角度
        },
        'jump_height': {
            'min': 0.1,  # 最小跳躍高度（相對於身高）
            'max': 0.5   # 最大跳躍高度（相對於身高）
        },
        'contact_height': {
            'min': 0.8,  # 最小擊球高度（相對於身高）
            'max': 1.2   # 最大擊球高度（相對於身高）
        }
    }
    
    def _validate_keypoints(self, keypoints):
        """驗證關鍵點的有效性"""
        if not keypoints or len(keypoints) < 17:  # YOLOv8-pose 有 17 個關鍵點
            return False
            
        # 檢查關鍵點的置信度
        for kp in keypoints:
            if len(kp) < 3 or kp[2] < self.CONFIDENCE_THRESHOLD:
                return False
                
        return True
        
    def _get_keypoint_coords(self, keypoints, idx):
        """安全地獲取關鍵點座標"""
        if idx < len(keypoints):
            return keypoints[idx][0], keypoints[idx][1]
        return None, None
        
    def _calculate_angle(self, p1, p2, p3):
        """計算三個點形成的角度"""
        if None in (p1, p2, p3):
            return None
            
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        v1_mag = (v1[0]**2 + v1[1]**2)**0.5
        v2_mag = (v2[0]**2 + v2[1]**2)**0.5
        
        if v1_mag == 0 or v2_mag == 0:
            return None
            
        cos_angle = dot_product / (v1_mag * v2_mag)
        cos_angle = max(-1, min(1, cos_angle))  # 確保在 [-1, 1] 範圍內
        
        return math.degrees(math.acos(cos_angle))
        
    def analyze_serve_pose(self, keypoints):
        """分析發球姿勢"""
        if not self._validate_keypoints(keypoints):
            return None, 0
            
        # 獲取關鍵點座標
        left_shoulder = self._get_keypoint_coords(keypoints, 5)
        right_shoulder = self._get_keypoint_coords(keypoints, 6)
        left_elbow = self._get_keypoint_coords(keypoints, 7)
        right_elbow = self._get_keypoint_coords(keypoints, 8)
        left_wrist = self._get_keypoint_coords(keypoints, 9)
        right_wrist = self._get_keypoint_coords(keypoints, 10)
        left_hip = self._get_keypoint_coords(keypoints, 11)
        right_hip = self._get_keypoint_coords(keypoints, 12)
        left_knee = self._get_keypoint_coords(keypoints, 13)
        right_knee = self._get_keypoint_coords(keypoints, 14)
        left_ankle = self._get_keypoint_coords(keypoints, 15)
        right_ankle = self._get_keypoint_coords(keypoints, 16)
        
        # 計算手臂角度
        left_arm_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # 計算膝蓋角度
        left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        
        # 計算跳躍高度（相對於身高）
        shoulder_height = (left_shoulder[1] + right_shoulder[1]) / 2
        ankle_height = (left_ankle[1] + right_ankle[1]) / 2
        height = abs(shoulder_height - ankle_height)
        jump_height = max(0, (ankle_height - shoulder_height) / height)
        
        # 計算擊球高度（相對於身高）
        wrist_height = (left_wrist[1] + right_wrist[1]) / 2
        contact_height = (shoulder_height - wrist_height) / height
        
        # 判斷當前發球階段
        current_phase = None
        if jump_height < self.ANGLE_THRESHOLDS['jump_height']['min']:
            current_phase = 'preparation'
        elif jump_height >= self.ANGLE_THRESHOLDS['jump_height']['min']:
            current_phase = 'jump'
        elif contact_height >= self.ANGLE_THRESHOLDS['contact_height']['min']:
            current_phase = 'contact'
        else:
            current_phase = 'follow_through'
            
        # 計算品質分數
        quality_score = 0
        if current_phase == 'preparation':
            # 檢查手臂和膝蓋角度
            if (left_arm_angle and right_arm_angle and
                self.ANGLE_THRESHOLDS['arm_angle']['min'] <= left_arm_angle <= self.ANGLE_THRESHOLDS['arm_angle']['max'] and
                self.ANGLE_THRESHOLDS['arm_angle']['min'] <= right_arm_angle <= self.ANGLE_THRESHOLDS['arm_angle']['max']):
                quality_score += 25
            if (left_knee_angle and right_knee_angle and
                self.ANGLE_THRESHOLDS['knee_angle']['min'] <= left_knee_angle <= self.ANGLE_THRESHOLDS['knee_angle']['max'] and
                self.ANGLE_THRESHOLDS['knee_angle']['min'] <= right_knee_angle <= self.ANGLE_THRESHOLDS['knee_angle']['max']):
                quality_score += 25
        elif current_phase == 'jump':
            # 檢查跳躍高度
            if self.ANGLE_THRESHOLDS['jump_height']['min'] <= jump_height <= self.ANGLE_THRESHOLDS['jump_height']['max']:
                quality_score += 50
        elif current_phase == 'contact':
            # 檢查擊球高度和手臂角度
            if self.ANGLE_THRESHOLDS['contact_height']['min'] <= contact_height <= self.ANGLE_THRESHOLDS['contact_height']['max']:
                quality_score += 25
            if (left_arm_angle and right_arm_angle and
                self.ANGLE_THRESHOLDS['arm_angle']['min'] <= left_arm_angle <= self.ANGLE_THRESHOLDS['arm_angle']['max'] and
                self.ANGLE_THRESHOLDS['arm_angle']['min'] <= right_arm_angle <= self.ANGLE_THRESHOLDS['arm_angle']['max']):
                quality_score += 25
        elif current_phase == 'follow_through':
            # 檢查手臂角度和跟進動作
            if (left_arm_angle and right_arm_angle and
                self.ANGLE_THRESHOLDS['arm_angle']['min'] <= left_arm_angle <= self.ANGLE_THRESHOLDS['arm_angle']['max'] and
                self.ANGLE_THRESHOLDS['arm_angle']['min'] <= right_arm_angle <= self.ANGLE_THRESHOLDS['arm_angle']['max']):
                quality_score += 50
                
        return current_phase, quality_score
        
    def analyze_serve_sequence(self, keypoints_sequence):
        """分析發球序列"""
        phases = []
        quality_scores = []
        
        for keypoints in keypoints_sequence:
            phase, quality_score = self.analyze_serve_pose(keypoints)
            if phase:
                phases.append(phase)
                quality_scores.append(quality_score)
                
        return phases, quality_scores
        
    def detect_serve_phase(self, keypoints_sequence):
        """檢測發球階段"""
        phases, quality_scores = self.analyze_serve_sequence(keypoints_sequence)
        
        if not phases:
            return None
            
        # 計算每個階段的出現次數
        phase_counts = {}
        for phase in phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            
        # 找出最常見的階段
        most_common_phase = max(phase_counts.items(), key=lambda x: x[1])[0]
        
        return most_common_phase 