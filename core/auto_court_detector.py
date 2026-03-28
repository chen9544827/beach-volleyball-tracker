# core/auto_court_detector.py
# -*- coding: utf-8 -*-
"""
Auto Court Detector - YOLO keypoint model for court boundary detection.

Uses a trained YOLO-pose model to detect 6 court keypoints:
    0: far_left, 1: far_right, 2: near_left, 3: near_right,
    4: net_left, 5: net_right

From these keypoints, generates a complete court_config including:
    - court_boundary_polygon (4 corners)
    - net_y (average of net endpoints)
    - exclusion_zones (auto-generated from boundary margins)
    - background_ball_zones (empty, can be added manually)

Usage:
    detector = AutoCourtDetector("models/court_best.pt")
    result = detector.detect_from_video("input_video/match.mp4")
    if result['confidence'] >= 0.7:
        config = detector.generate_court_config(result, frame_w=1280, frame_h=720)
        # config is ready to save as court_config.json
"""

import os
import cv2
import json
import numpy as np
from typing import Dict, List, Optional, Tuple


# Keypoint indices
KP_FAR_LEFT = 0
KP_FAR_RIGHT = 1
KP_NEAR_LEFT = 2
KP_NEAR_RIGHT = 3
KP_NET_LEFT = 4
KP_NET_RIGHT = 5
NUM_KEYPOINTS = 6

# Keypoint names for display
KP_NAMES = ['far_left', 'far_right', 'near_left', 'near_right', 'net_left', 'net_right']


class AutoCourtDetector:
    """
    YOLO keypoint-based court boundary detector.

    Detects court corners and net endpoints from video frames,
    then generates complete court_config with auto exclusion zones.
    """

    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Args:
            model_path: Path to trained YOLO keypoint model (.pt)
            device: Device for inference ('auto', 'cuda', 'cpu')
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        from ultralytics import YOLO
        self.model = YOLO(model_path)

        if device == 'auto':
            self._device = None  # Let YOLO decide
        else:
            self._device = device

    def detect_single_frame(self, frame: np.ndarray, conf_threshold: float = 0.3) -> Optional[Dict]:
        """
        Detect court keypoints from a single frame.

        Args:
            frame: BGR image (numpy array)
            conf_threshold: Minimum detection confidence

        Returns:
            Dict with 'keypoints' (6x3 array: x, y, conf), 'confidence', 'bbox'
            or None if no detection.
        """
        kwargs = {'conf': conf_threshold, 'verbose': False}
        if self._device is not None:
            kwargs['device'] = self._device

        results = self.model(frame, **kwargs)

        if not results or len(results) == 0:
            return None

        result = results[0]
        if result.keypoints is None or len(result.keypoints.data) == 0:
            return None

        # Take highest confidence detection
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_idx = int(boxes.conf.argmax())
        best_conf = float(boxes.conf[best_idx])

        kps = result.keypoints.data[best_idx].cpu().numpy()  # (num_kp, 3)
        bbox = boxes.xyxy[best_idx].cpu().numpy()  # [x1, y1, x2, y2]

        if kps.shape[0] < NUM_KEYPOINTS:
            return None

        return {
            'keypoints': kps[:NUM_KEYPOINTS],  # (6, 3): x, y, conf
            'confidence': best_conf,
            'bbox': bbox,
        }

    def detect_from_video(self, video_path: str, sample_frames: int = 5,
                          conf_threshold: float = 0.3) -> Dict:
        """
        Detect court keypoints from multiple video frames and take median.

        Samples frames at evenly spaced positions (avoiding start/end),
        runs detection on each, and takes the median keypoint position
        for robustness.

        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample
            conf_threshold: Minimum detection confidence per frame

        Returns:
            Dict with:
                'keypoints': (6, 2) median keypoint positions
                'confidence': Average confidence across detections
                'per_kp_confidence': (6,) per-keypoint average confidence
                'num_detections': Number of successful detections
                'frame_w': Frame width
                'frame_h': Frame height
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'confidence': 0.0, 'num_detections': 0,
                    'keypoints': None, 'error': f'Cannot open video: {video_path}'}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0:
            cap.release()
            return {'confidence': 0.0, 'num_detections': 0,
                    'keypoints': None, 'error': 'Empty video'}

        # Sample positions (avoid first/last 10%)
        positions = [0.1 + i * 0.8 / max(1, sample_frames - 1)
                     for i in range(sample_frames)]

        all_kps = []  # List of (6, 3) arrays
        all_confs = []

        for pos in positions:
            frame_idx = int(total_frames * pos)
            frame_idx = max(0, min(total_frames - 1, frame_idx))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            det = self.detect_single_frame(frame, conf_threshold)
            if det is not None:
                all_kps.append(det['keypoints'])
                all_confs.append(det['confidence'])

        cap.release()

        if not all_kps:
            return {'confidence': 0.0, 'num_detections': 0,
                    'keypoints': None, 'frame_w': frame_w, 'frame_h': frame_h}

        # Compute median keypoints across detections
        kps_array = np.array(all_kps)  # (N, 6, 3)
        median_xy = np.median(kps_array[:, :, :2], axis=0)  # (6, 2)
        per_kp_conf = np.mean(kps_array[:, :, 2], axis=0)   # (6,)
        avg_conf = float(np.mean(all_confs))

        return {
            'keypoints': median_xy,
            'confidence': avg_conf,
            'per_kp_confidence': per_kp_conf,
            'num_detections': len(all_kps),
            'frame_w': frame_w,
            'frame_h': frame_h,
        }

    def generate_court_config(self, detection: Dict,
                              frame_w: int = None, frame_h: int = None,
                              margin_lr: float = 0.15,
                              margin_far: float = 0.30,
                              margin_near: float = 0.25) -> Optional[Dict]:
        """
        Generate complete court_config from detection results.

        Args:
            detection: Result from detect_from_video()
            frame_w: Frame width (overrides detection value)
            frame_h: Frame height (overrides detection value)
            margin_lr: Left/right exclusion margin as ratio of court width at that height
            margin_far: Far-side (top) exclusion margin as ratio of court pixel height
            margin_near: Near-side (bottom) exclusion margin as ratio of court pixel height

        Returns:
            Complete court_config dict, or None if detection is invalid
        """
        keypoints = detection.get('keypoints')
        if keypoints is None:
            return None

        fw = frame_w or detection.get('frame_w', 1280)
        fh = frame_h or detection.get('frame_h', 720)

        # Extract corner points (as integer pixels for config compatibility)
        far_left = [int(round(keypoints[KP_FAR_LEFT][0])),
                     int(round(keypoints[KP_FAR_LEFT][1]))]
        far_right = [int(round(keypoints[KP_FAR_RIGHT][0])),
                      int(round(keypoints[KP_FAR_RIGHT][1]))]
        near_left = [int(round(keypoints[KP_NEAR_LEFT][0])),
                      int(round(keypoints[KP_NEAR_LEFT][1]))]
        near_right = [int(round(keypoints[KP_NEAR_RIGHT][0])),
                       int(round(keypoints[KP_NEAR_RIGHT][1]))]

        # court_boundary_polygon: [top_left, bottom_left, bottom_right, top_right]
        # Match the format from court_config_generator.py
        court_boundary = [far_left, near_left, near_right, far_right]

        # Net Y: average of two net endpoints
        net_y = int(round((keypoints[KP_NET_LEFT][1] + keypoints[KP_NET_RIGHT][1]) / 2))

        # Generate exclusion zones
        exclusion_zones = self._generate_exclusion_zones(
            keypoints, fw, fh, margin_lr, margin_far, margin_near
        )

        config = {
            'court_boundary_polygon': court_boundary,
            'exclusion_zones': exclusion_zones,
            'net_y': net_y,
            'background_ball_zones': [],
        }

        return config

    def _generate_exclusion_zones(self, keypoints: np.ndarray,
                                   frame_w: int, frame_h: int,
                                   margin_lr: float = 0.15,
                                   margin_far: float = 0.30,
                                   margin_near: float = 0.25) -> List[Dict]:
        """
        Auto-generate exclusion zones from court boundary.

        Creates up to 6 exclusion zones:
        - Left/Right: areas outside active region on left/right sides
        - Top/Bottom: areas above far baseline / below near baseline
        - Net-post left/right: referee stand areas around net endpoints

        Args:
            keypoints: (6, 2) keypoint positions
            frame_w: Frame width
            frame_h: Frame height
            margin_lr: Left/right margin as ratio of court width at that height
            margin_far: Far-side margin as ratio of court pixel height
            margin_near: Near-side margin as ratio of court pixel height

        Returns:
            List of exclusion zone dicts with 'polygon' key
        """
        fl = keypoints[KP_FAR_LEFT]
        fr = keypoints[KP_FAR_RIGHT]
        nl = keypoints[KP_NEAR_LEFT]
        nr = keypoints[KP_NEAR_RIGHT]
        net_l = keypoints[KP_NET_LEFT]
        net_r = keypoints[KP_NET_RIGHT]

        # Court pixel height
        court_h = max(abs(nl[1] - fl[1]), abs(nr[1] - fr[1]))
        if court_h < 20:
            return []

        # Court widths at far and near sides
        far_width = abs(fr[0] - fl[0])
        near_width = abs(nr[0] - nl[0])

        # Expanded active region corners
        far_expand_y = margin_far * court_h
        far_expand_x = margin_lr * far_width
        near_expand_y = margin_near * court_h
        near_expand_x = margin_lr * near_width

        # Active region corners (trapezoid, expanded from court)
        active_fl = (max(0, fl[0] - far_expand_x), max(0, fl[1] - far_expand_y))
        active_fr = (min(frame_w, fr[0] + far_expand_x), max(0, fr[1] - far_expand_y))
        active_nl = (max(0, nl[0] - near_expand_x), min(frame_h, nl[1] + near_expand_y))
        active_nr = (min(frame_w, nr[0] + near_expand_x), min(frame_h, nr[1] + near_expand_y))

        zones = []
        min_size = 5  # Minimum exclusion zone dimension (lowered to catch tight edges)

        # Left exclusion zone: frame left edge to active left boundary
        left_x_far = active_fl[0]
        left_x_near = active_nl[0]
        if max(left_x_far, left_x_near) > min_size:
            zone = {
                'polygon': [
                    [0, int(round(active_fl[1]))],
                    [int(round(left_x_far)), int(round(active_fl[1]))],
                    [int(round(left_x_near)), int(round(active_nl[1]))],
                    [0, int(round(active_nl[1]))],
                ]
            }
            zones.append(zone)

        # Right exclusion zone: active right boundary to frame right edge
        right_x_far = active_fr[0]
        right_x_near = active_nr[0]
        if (frame_w - min(right_x_far, right_x_near)) > min_size:
            zone = {
                'polygon': [
                    [int(round(right_x_far)), int(round(active_fr[1]))],
                    [frame_w, int(round(active_fr[1]))],
                    [frame_w, int(round(active_nr[1]))],
                    [int(round(right_x_near)), int(round(active_nr[1]))],
                ]
            }
            zones.append(zone)

        # Top exclusion zone: frame top to active far boundary
        top_y = min(active_fl[1], active_fr[1])
        if top_y > min_size:
            zone = {
                'polygon': [
                    [0, 0],
                    [frame_w, 0],
                    [frame_w, int(round(top_y))],
                    [0, int(round(top_y))],
                ]
            }
            zones.append(zone)

        return zones

    def visualize_detection(self, frame: np.ndarray, detection: Dict,
                            court_config: Dict = None) -> np.ndarray:
        """
        Draw detection results on a frame for verification.

        Args:
            frame: BGR image
            detection: Result from detect_from_video() or detect_single_frame()
            court_config: Optional generated court_config (to show exclusion zones)

        Returns:
            Annotated frame copy
        """
        vis = frame.copy()
        keypoints = detection.get('keypoints')
        if keypoints is None:
            cv2.putText(vis, "No detection", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return vis

        # Draw court boundary (green polygon)
        corners = keypoints[:4].astype(int)
        # Order: far_left -> far_right -> near_right -> near_left (for drawing)
        court_poly = np.array([corners[0], corners[1], corners[3], corners[2]])
        cv2.polylines(vis, [court_poly], True, (0, 255, 0), 2)

        # Draw net line (yellow)
        net_l = keypoints[KP_NET_LEFT].astype(int)[:2]
        net_r = keypoints[KP_NET_RIGHT].astype(int)[:2]
        cv2.line(vis, tuple(net_l), tuple(net_r), (0, 255, 255), 2)

        # Draw keypoints with labels
        colors = [
            (0, 255, 0),     # far_left - green
            (0, 255, 0),     # far_right - green
            (255, 0, 0),     # near_left - blue
            (255, 0, 0),     # near_right - blue
            (0, 255, 255),   # net_left - yellow
            (0, 255, 255),   # net_right - yellow
        ]

        for i in range(NUM_KEYPOINTS):
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            conf = detection.get('per_kp_confidence')
            conf_str = f" {conf[i]:.2f}" if conf is not None else ""
            cv2.circle(vis, (x, y), 6, colors[i], -1)
            cv2.putText(vis, f"{KP_NAMES[i]}{conf_str}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)

        # Draw exclusion zones if court_config provided (purple, semi-transparent)
        if court_config and court_config.get('exclusion_zones'):
            overlay = vis.copy()
            for zone in court_config['exclusion_zones']:
                poly = np.array(zone['polygon'], dtype=np.int32)
                cv2.fillPoly(overlay, [poly], (180, 0, 180))
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

        # Show confidence
        conf = detection.get('confidence', 0)
        n_det = detection.get('num_detections', 1)
        cv2.putText(vis, f"Conf: {conf:.3f} ({n_det} frames)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return vis


def save_court_config(config: Dict, output_path: str):
    """Save court_config to JSON file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
