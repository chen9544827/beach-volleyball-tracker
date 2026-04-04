# core/sahi_pose_detector.py
# -*- coding: utf-8 -*-
"""
SAHI (Slicing Aided Hyper Inference) Pose Detector

Standard SAHI discards keypoints during merge. This custom wrapper preserves
all 17 COCO pose keypoints through slice-based inference and keypoint-aware NMS.

Usage:
    detector = SahiPoseDetector(model_path, slice_size=512, overlap_ratio=0.2)
    detections = detector.detect(frame, conf_thresh=0.15)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class SahiPoseDetector:
    """SAHI wrapper for YOLO Pose models with keypoint preservation."""

    def __init__(
        self,
        model,
        slice_size: int = 512,
        overlap_ratio: float = 0.2,
        conf_thresh: float = 0.15,
        iou_thresh: float = 0.5,
    ):
        """
        Args:
            model: Loaded YOLO Pose model instance.
            slice_size: Size of each square slice in pixels.
            overlap_ratio: Overlap between adjacent slices (0-1).
            conf_thresh: Minimum detection confidence.
            iou_thresh: IoU threshold for NMS merging.
        """
        self.model = model
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def detect(self, frame, conf_thresh: Optional[float] = None) -> List[Dict]:
        """
        Run sliced inference on a frame.

        Args:
            frame: BGR image (numpy array).
            conf_thresh: Override default confidence threshold.

        Returns:
            List of detection dicts with keys:
                box_coords, confidence, center_point, pose_keypoints
        """
        if conf_thresh is None:
            conf_thresh = self.conf_thresh

        h, w = frame.shape[:2]
        slices = self._create_slices(h, w)

        all_detections = []
        for (x_off, y_off, x_end, y_end) in slices:
            crop = frame[y_off:y_end, x_off:x_end]
            dets = self._detect_slice(crop, conf_thresh)
            # Translate to global coordinates
            for det in dets:
                det = self._translate_to_global(det, x_off, y_off)
                # Clip to frame boundaries
                det["box_coords"] = [
                    max(0, det["box_coords"][0]),
                    max(0, det["box_coords"][1]),
                    min(w, det["box_coords"][2]),
                    min(h, det["box_coords"][3]),
                ]
                all_detections.append(det)

        # Also run full-frame inference to catch large/centered players
        full_dets = self._detect_slice(frame, conf_thresh)
        all_detections.extend(full_dets)

        # Merge with keypoint-aware NMS
        merged = self._merge_detections(all_detections)
        return merged

    def _create_slices(self, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Generate overlapping slice coordinates.

        Returns:
            List of (x_offset, y_offset, x_end, y_end) tuples.
        """
        step = int(self.slice_size * (1 - self.overlap_ratio))
        slices = []

        for y in range(0, h, step):
            for x in range(0, w, step):
                x_end = min(x + self.slice_size, w)
                y_end = min(y + self.slice_size, h)
                # Adjust start if slice is too small at edges
                x_start = max(0, x_end - self.slice_size)
                y_start = max(0, y_end - self.slice_size)
                slices.append((x_start, y_start, x_end, y_end))

        return slices

    def _detect_slice(self, crop, conf_thresh: float) -> List[Dict]:
        """Run YOLO Pose on a single crop.

        Returns:
            List of detections in local (crop) coordinates.
        """
        detections = []
        try:
            results = self.model(
                crop, conf=conf_thresh, classes=[0], verbose=False,
                imgsz=640, max_det=100, iou=0.45
            )
            if not results or not results[0].boxes or not results[0].keypoints:
                return detections

            for i in range(len(results[0].boxes)):
                box = results[0].boxes[i]
                kpts = results[0].keypoints[i]

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())

                keypoints_xyc = []
                if kpts.xy is not None and kpts.conf is not None:
                    kpts_xy = kpts.xy[0].cpu().numpy()
                    kpts_conf = kpts.conf[0].cpu().numpy()
                    for kp_idx in range(kpts_xy.shape[0]):
                        keypoints_xyc.append([
                            float(kpts_xy[kp_idx, 0]),
                            float(kpts_xy[kp_idx, 1]),
                            float(kpts_conf[kp_idx])
                        ])

                detections.append({
                    "box_coords": [x1, y1, x2, y2],
                    "confidence": conf,
                    "center_point": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                    "pose_keypoints": keypoints_xyc,
                })
        except Exception as e:
            import sys
            print(f"!! SAHI slice detection error: {e}", file=sys.stderr)

        return detections

    def _translate_to_global(self, det: Dict, x_off: int, y_off: int) -> Dict:
        """Translate detection coordinates from slice-local to global frame."""
        x1, y1, x2, y2 = det["box_coords"]
        det["box_coords"] = [x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off]
        det["center_point"] = [
            det["center_point"][0] + x_off,
            det["center_point"][1] + y_off,
        ]
        for kp in det["pose_keypoints"]:
            if kp[2] > 0:  # Only translate valid keypoints
                kp[0] += x_off
                kp[1] += y_off
        return det

    def _merge_detections(self, detections: List[Dict]) -> List[Dict]:
        """Keypoint-aware NMS: merge overlapping detections, keep best keypoints.

        For each pair with IoU > threshold, keep the higher-confidence detection
        but merge in any higher-confidence individual keypoints from the other.
        """
        if not detections:
            return []

        # Sort by confidence descending
        detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        keep = [True] * len(detections)

        for i in range(len(detections)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(detections)):
                if not keep[j]:
                    continue
                iou = self._compute_iou(
                    detections[i]["box_coords"], detections[j]["box_coords"]
                )
                if iou > self.iou_thresh:
                    # Merge keypoints: keep higher conf per joint
                    self._merge_keypoints(detections[i], detections[j])
                    keep[j] = False

        return [d for d, k in zip(detections, keep) if k]

    def _merge_keypoints(self, primary: Dict, secondary: Dict):
        """Merge keypoints from secondary into primary, keeping higher confidence per joint."""
        kp_pri = primary.get("pose_keypoints", [])
        kp_sec = secondary.get("pose_keypoints", [])
        if not kp_pri or not kp_sec:
            return
        for idx in range(min(len(kp_pri), len(kp_sec))):
            if kp_sec[idx][2] > kp_pri[idx][2]:
                kp_pri[idx] = kp_sec[idx][:]

    @staticmethod
    def _compute_iou(box1: List, box2: List) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
