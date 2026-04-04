# core/video_context.py
# -*- coding: utf-8 -*-
"""
影片參數記錄（輕量版）

僅記錄影片規格到 metadata，不做全面正規化。
依分組測試效果，效果差的分組跳過。
"""


class VideoContext:
    """影片參數記錄（輕量版，僅記錄不正規化）"""

    REFERENCE_HEIGHT = 720
    REFERENCE_FPS = 25.0

    def __init__(self, width: int, height: int, fps: float):
        """
        Args:
            width: 影片寬度
            height: 影片高度
            fps: 影片幀率
        """
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.resolution_scale = self.height / self.REFERENCE_HEIGHT
        self.fps_scale = self.fps / self.REFERENCE_FPS

    @classmethod
    def from_video_capture(cls, cap) -> 'VideoContext':
        """
        從 cv2.VideoCapture 建立

        Args:
            cap: cv2.VideoCapture 物件

        Returns:
            VideoContext 實例
        """
        import cv2
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        return cls(width, height, fps)

    def to_dict(self) -> dict:
        """轉為字典（用於 metadata）"""
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "resolution_scale": round(self.resolution_scale, 4),
            "fps_scale": round(self.fps_scale, 4),
        }

    def __repr__(self):
        return (f"VideoContext({self.width}x{self.height}, "
                f"{self.fps:.1f}fps, scale={self.resolution_scale:.2f})")
