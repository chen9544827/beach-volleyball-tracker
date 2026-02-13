# core/static_ball_filter.py
# -*- coding: utf-8 -*-
"""
Automatic Static Ball Filter

Detects and removes persistent false-positive ball detections caused by
static objects (court logos, markings, equipment) that are consistently
detected as volleyballs across many frames.

Uses a spatial grid to efficiently track detection frequency per location.
When a grid cell accumulates detections in >30 of the last 100 frames,
all future detections in that cell are suppressed.

Usage:
    static_filter = StaticBallFilter()
    # In tracking loop:
    static_filter.update(frame_id, ball_detections)
    filtered = static_filter.filter(ball_detections)
"""

from collections import defaultdict, deque
from typing import List, Dict, Tuple


class StaticBallFilter:
    """Filter out ball detections at positions that appear static over time."""

    def __init__(
        self,
        cell_size: int = 15,
        buffer_size: int = 100,
        min_static_frames: int = 15,
        resolution_scale: float = 1.0,
        warmup_frames: int = 50,
        warmup_threshold: int = 8,
    ):
        """
        Args:
            cell_size: Grid cell size in pixels (at 720p baseline).
            buffer_size: Sliding window size in frames.
            min_static_frames: Minimum frames a cell must appear in to be
                considered static (within the buffer window).
            resolution_scale: Scale factor for pixel thresholds (height / 720).
            warmup_frames: During first N frames, use warmup_threshold instead.
            warmup_threshold: Lower threshold during warmup for faster detection.
        """
        self.cell_size = max(1, int(cell_size * resolution_scale))
        self.buffer_size = buffer_size
        self.min_static_frames = min_static_frames
        self.warmup_frames = warmup_frames
        self.warmup_threshold = warmup_threshold

        # grid_key -> deque of frame_ids where detection occurred
        self._history: Dict[Tuple[int, int], deque] = defaultdict(
            lambda: deque(maxlen=buffer_size)
        )
        self._current_frame_id = -1

    def _to_grid(self, x: int, y: int) -> Tuple[int, int]:
        """Quantize pixel position to grid cell."""
        return (x // self.cell_size, y // self.cell_size)

    def update(self, frame_id: int, detections: List[Dict]):
        """Record detection positions for the current frame.

        Args:
            frame_id: Current frame index.
            detections: Ball detections for this frame.
        """
        self._current_frame_id = frame_id
        seen_cells = set()

        for det in detections:
            cp = det.get("center_point")
            if not cp or len(cp) < 2:
                continue
            cell = self._to_grid(int(cp[0]), int(cp[1]))
            if cell not in seen_cells:
                seen_cells.add(cell)
                history = self._history[cell]
                # Only add once per frame per cell
                if not history or history[-1] != frame_id:
                    history.append(frame_id)

    def _is_static(self, cell: Tuple[int, int]) -> bool:
        """Check if a grid cell is considered static."""
        history = self._history.get(cell)
        if not history:
            return False

        # Count frames within the recent window
        cutoff = self._current_frame_id - self.buffer_size
        recent_count = sum(1 for fid in history if fid > cutoff)

        # Use lower threshold during warmup for faster static detection
        threshold = self.min_static_frames
        if self._current_frame_id < self.warmup_frames:
            threshold = self.warmup_threshold

        return recent_count >= threshold

    def filter(self, detections: List[Dict]) -> List[Dict]:
        """Remove detections at static positions.

        Args:
            detections: Ball detections for the current frame.

        Returns:
            Filtered detections with static objects removed.
        """
        filtered = []
        for det in detections:
            cp = det.get("center_point")
            if not cp or len(cp) < 2:
                filtered.append(det)
                continue
            cell = self._to_grid(int(cp[0]), int(cp[1]))
            if not self._is_static(cell):
                filtered.append(det)
        return filtered

    def get_static_zones(self) -> List[Dict]:
        """Export currently detected static zones.

        Returns:
            List of dicts with keys: grid_x, grid_y, pixel_x, pixel_y, count
        """
        zones = []
        cutoff = self._current_frame_id - self.buffer_size
        for cell, history in self._history.items():
            recent_count = sum(1 for fid in history if fid > cutoff)
            if recent_count >= self.min_static_frames:
                zones.append({
                    "grid_x": cell[0],
                    "grid_y": cell[1],
                    "pixel_x": cell[0] * self.cell_size + self.cell_size // 2,
                    "pixel_y": cell[1] * self.cell_size + self.cell_size // 2,
                    "count": recent_count,
                })
        return zones
