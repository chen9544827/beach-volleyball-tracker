# core/auto_court_estimator.py
# -*- coding: utf-8 -*-
"""
Auto Court Estimator - Dynamically estimate court boundary from near-side players.

Uses the two near-side players (high confidence, below net) as anchor points
to estimate the full court area as a trapezoid. This allows filtering out
off-court detections (spectators, referees, photographers) without needing
a manually configured court_config.

Algorithm:
1. Each frame: find 2 high-confidence players in the lower half (near side)
2. Accumulate positions in a sliding window for stability
3. From near-side player positions, estimate:
   - Court width (near and far side, with perspective)
   - Net Y position
   - Full court trapezoid boundary
4. Use the trapezoid to hard-reject off-court detections

Usage:
    estimator = AutoCourtEstimator(frame_width=1280, frame_height=720)
    # In tracking loop:
    estimator.update(frame_id, player_detections)
    if estimator.is_ready:
        filtered = [p for p in players if estimator.is_inside(p['center_point'])]
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple


class AutoCourtEstimator:
    """Dynamically estimate court boundary using near-side players as anchors."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        min_confidence: float = 0.5,
        buffer_size: int = 50,
        min_samples: int = 10,
        near_width_multiplier: float = 2.8,
        perspective_ratio: float = 0.65,
    ):
        """
        Args:
            frame_width: Video frame width in pixels.
            frame_height: Video frame height in pixels.
            min_confidence: Minimum detection confidence for near-side players.
            buffer_size: Sliding window size for position averaging.
            min_samples: Minimum samples before computing boundary.
            near_width_multiplier: Court width / player span ratio.
            perspective_ratio: Far-side width / near-side width ratio.
        """
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.min_confidence = min_confidence
        self.min_samples = min_samples
        self.near_width_mult = near_width_multiplier
        self.perspective_ratio = perspective_ratio

        self._near_positions: deque = deque(maxlen=buffer_size)
        self._boundary: Optional[np.ndarray] = None
        self._estimated_net_y: Optional[float] = None
        self._frame_count = 0
        self._rejected_this_frame: List[Dict] = []

    def update(self, frame_id: int, player_detections: List[Dict]):
        """Update estimator with current frame's player detections.

        Finds the 2 most reliable near-side players and accumulates positions.

        Args:
            frame_id: Current frame index.
            player_detections: All player detections for this frame.
        """
        self._frame_count += 1
        self._rejected_this_frame = []

        # Filter: high confidence + lower half of frame (near side)
        near_candidates = [
            d for d in player_detections
            if d.get("confidence", 0) >= self.min_confidence
            and d["center_point"][1] > self.frame_h * 0.4
        ]

        if len(near_candidates) < 2:
            return

        # Sort by distance to near-side center (center-x, 60% height)
        cx = self.frame_w / 2
        cy = self.frame_h * 0.6
        near_candidates.sort(
            key=lambda d: (d["center_point"][0] - cx) ** 2
            + (d["center_point"][1] - cy) ** 2
        )

        p1 = near_candidates[0]["center_point"]
        p2 = near_candidates[1]["center_point"]

        # Ensure p1 is left, p2 is right (for consistency)
        if p1[0] > p2[0]:
            p1, p2 = p2, p1

        self._near_positions.append((p1, p2))

        if len(self._near_positions) >= self.min_samples:
            self._compute_boundary()

    def _compute_boundary(self):
        """Compute trapezoid boundary from accumulated near-side positions."""
        # Use median for robustness against outliers
        left_xs = [p[0][0] for p in self._near_positions]
        left_ys = [p[0][1] for p in self._near_positions]
        right_xs = [p[1][0] for p in self._near_positions]
        right_ys = [p[1][1] for p in self._near_positions]

        avg_lx = float(np.median(left_xs))
        avg_ly = float(np.median(left_ys))
        avg_rx = float(np.median(right_xs))
        avg_ry = float(np.median(right_ys))

        # Near-side parameters
        near_center_x = (avg_lx + avg_rx) / 2
        near_span = abs(avg_rx - avg_lx)
        near_width = near_span * self.near_width_mult
        near_y = max(avg_ly, avg_ry)

        # Court pixel height estimate (near bottom to near top of frame)
        court_pixel_h = near_y - self.frame_h * 0.05
        if court_pixel_h < 50:
            return

        # Net Y: approximately 45% of court height above near-side
        net_y = near_y - court_pixel_h * 0.45

        # Far-side parameters (perspective shrink)
        far_width = near_width * self.perspective_ratio
        far_y = net_y - court_pixel_h * 0.30

        # Build trapezoid with margin
        margin_x = near_width * 0.25
        margin_y = court_pixel_h * 0.15

        self._boundary = np.array(
            [
                [near_center_x - far_width / 2 - margin_x, far_y - margin_y],
                [near_center_x + far_width / 2 + margin_x, far_y - margin_y],
                [near_center_x + near_width / 2 + margin_x, near_y + margin_y],
                [near_center_x - near_width / 2 - margin_x, near_y + margin_y],
            ],
            dtype=np.float32,
        )

        self._estimated_net_y = net_y

    @property
    def is_ready(self) -> bool:
        """Whether the estimator has computed a valid boundary."""
        return self._boundary is not None

    @property
    def boundary(self) -> Optional[np.ndarray]:
        """Trapezoid boundary as np.ndarray (4, 2), or None."""
        return self._boundary

    @property
    def net_y(self) -> Optional[float]:
        """Estimated net Y coordinate, or None."""
        return self._estimated_net_y

    @property
    def rejected_detections(self) -> List[Dict]:
        """Detections rejected by filter_detections() in the last call."""
        return self._rejected_this_frame

    def is_inside(self, point, margin: float = 0) -> bool:
        """Check if a point is inside the estimated court boundary.

        Args:
            point: (x, y) coordinate.
            margin: Extra tolerance in pixels (negative = stricter).

        Returns:
            True if inside or boundary not ready yet.
        """
        if self._boundary is None:
            return True
        dist = cv2.pointPolygonTest(
            self._boundary, (float(point[0]), float(point[1])), True
        )
        return dist >= -margin

    def filter_detections(
        self, player_detections: List[Dict], margin: float = 20.0
    ) -> List[Dict]:
        """Filter player detections, keeping only those inside the court boundary.

        Also stores rejected detections in self.rejected_detections.

        Args:
            player_detections: All detections to filter.
            margin: Extra tolerance in pixels.

        Returns:
            Filtered detections (inside court only).
        """
        if not self.is_ready:
            self._rejected_this_frame = []
            return player_detections

        kept = []
        rejected = []
        for det in player_detections:
            if self.is_inside(det["center_point"], margin=margin):
                kept.append(det)
            else:
                rejected.append(det)

        self._rejected_this_frame = rejected
        return kept
