# core/court_zones.py
# -*- coding: utf-8 -*-
"""
場地分區計算器

將沙排場地分成標準區域：
- 發球區：端線後方，左/中/右 3 區
- 接球區：每半場 6 區（前排左/中/右 + 後排左/中/右）

使用 court_boundary_polygon (4 點) + net_y 自動計算所有分區。
支援透視變形（上方遠端比下方近端窄）。

座標系統：Y 軸向下遞增（Y 值越大 = 畫面越下方）
- far side (對面場地) = 畫面上方 = Y 值較小
- near side (本方場地) = 畫面下方 = Y 值較大
"""

import logging
from typing import Dict, List, Optional, Tuple


def _lerp(p1: Tuple[float, float], p2: Tuple[float, float], t: float) -> Tuple[float, float]:
    """線性內插兩點"""
    return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)


def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    射線法判斷點是否在多邊形內

    Args:
        point: (x, y)
        polygon: 多邊形頂點列表
    """
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


class CourtZones:
    """
    場地分區計算器

    從 court_config 的 4 個頂點 + net_y 計算所有分區。

    court_boundary_polygon 頂點順序（預期）：
        [top_left, bottom_left, bottom_right, top_right]
    其中 top = 靠近 net_y 的遠端，bottom = 靠近畫面底部的近端

    分區編號：
        前排（靠網）：1=左, 2=中, 3=右
        後排（遠網）：4=左, 5=中, 6=右
        發球區：1=左, 2=中, 3=右
    """

    def __init__(self, court_config: dict):
        """
        從 court_config 建立分區

        Args:
            court_config: 場地設定字典，需包含：
                - court_boundary_polygon: 4 個頂點 [[x,y], ...]
                - net_y: 網子 Y 座標
        """
        polygon = court_config.get('court_boundary_polygon', [])
        if len(polygon) < 4:
            raise ValueError("court_boundary_polygon must have at least 4 points")

        self.net_y = court_config.get('net_y', 274)
        self.raw_polygon = [tuple(p) for p in polygon]

        # 將頂點分為上邊（far/遠端）和下邊（near/近端）
        # 按 Y 座標排序，Y 小的兩點為上邊（far），Y 大的兩點為下邊（near）
        sorted_by_y = sorted(self.raw_polygon, key=lambda p: p[1])
        top_points = sorted(sorted_by_y[:2], key=lambda p: p[0])   # 按 X 排序
        bottom_points = sorted(sorted_by_y[2:], key=lambda p: p[0])  # 按 X 排序

        # 場地四角
        self.far_left = top_points[0]      # 遠端左上
        self.far_right = top_points[1]     # 遠端右上
        self.near_left = bottom_points[0]  # 近端左下
        self.near_right = bottom_points[1] # 近端右下

        # 計算網線位置的左右端點（在 net_y 高度處內插）
        self._compute_net_endpoints()

        # 預計算所有分區多邊形
        self._zones = {}
        self._compute_half_zones('far')
        self._compute_half_zones('near')
        self._compute_serve_zones('far')
        self._compute_serve_zones('near')

    def _compute_net_endpoints(self):
        """計算網線與場地邊界的交點"""
        # 左邊界線: far_left -> near_left
        # 右邊界線: far_right -> near_right
        # 在 net_y 高度處找左右交點

        def intersect_at_y(p1, p2, y):
            """在給定 Y 座標處找邊界線的 X 座標"""
            if abs(p2[1] - p1[1]) < 1e-6:
                return p1[0]
            t = (y - p1[1]) / (p2[1] - p1[1])
            t = max(0.0, min(1.0, t))
            return p1[0] + (p2[0] - p1[0]) * t

        self.net_left_x = intersect_at_y(self.far_left, self.near_left, self.net_y)
        self.net_right_x = intersect_at_y(self.far_right, self.near_right, self.net_y)
        self.net_left = (self.net_left_x, self.net_y)
        self.net_right = (self.net_right_x, self.net_y)

    def _compute_half_zones(self, side: str):
        """
        計算某半場的 6 個區域

        Args:
            side: 'far' (遠端/上方) 或 'near' (近端/下方)
        """
        if side == 'far':
            # 遠端半場: net_line -> far_line
            bl = self.net_left
            br = self.net_right
            tl = self.far_left
            tr = self.far_right
        else:
            # 近端半場: net_line -> near_line
            tl = self.net_left
            tr = self.net_right
            bl = self.near_left
            br = self.near_right

        # 前排（靠網）和後排（遠網）的分界線 = 半場中線
        # 對 far side: 前排靠近 net_y（下方），後排靠近 far line（上方）
        # 對 near side: 前排靠近 net_y（上方），後排靠近 near line（下方）
        mid_left = _lerp(tl, bl, 0.5)
        mid_right = _lerp(tr, br, 0.5)

        # 左/中/右三等分線
        # 在每條水平線上三等分
        def third_points(left, right):
            p1 = _lerp(left, right, 1/3)
            p2 = _lerp(left, right, 2/3)
            return p1, p2

        # 各行的三等分點
        top_t1, top_t2 = third_points(tl, tr)
        mid_t1, mid_t2 = third_points(mid_left, mid_right)
        bot_t1, bot_t2 = third_points(bl, br)

        if side == 'far':
            # far side: 前排 = 靠網 (bottom = net side), 後排 = 遠離網 (top = far side)
            # 前排 1-3 (靠網)
            self._zones[('far', 1)] = [bl, bot_t1, mid_t1, mid_left]        # 前排左
            self._zones[('far', 2)] = [bot_t1, bot_t2, mid_t2, mid_t1]     # 前排中
            self._zones[('far', 3)] = [bot_t2, br, mid_right, mid_t2]      # 前排右
            # 後排 4-6 (遠離網)
            self._zones[('far', 4)] = [mid_left, mid_t1, top_t1, tl]       # 後排左
            self._zones[('far', 5)] = [mid_t1, mid_t2, top_t2, top_t1]     # 後排中
            self._zones[('far', 6)] = [mid_t2, mid_right, tr, top_t2]      # 後排右
        else:
            # near side: 前排 = 靠網 (top = net side), 後排 = 遠離網 (bottom = near side)
            # 前排 1-3 (靠網)
            self._zones[('near', 1)] = [tl, top_t1, mid_t1, mid_left]      # 前排左
            self._zones[('near', 2)] = [top_t1, top_t2, mid_t2, mid_t1]    # 前排中
            self._zones[('near', 3)] = [top_t2, tr, mid_right, mid_t2]     # 前排右
            # 後排 4-6 (遠離網)
            self._zones[('near', 4)] = [mid_left, mid_t1, bot_t1, bl]      # 後排左
            self._zones[('near', 5)] = [mid_t1, mid_t2, bot_t2, bot_t1]    # 後排中
            self._zones[('near', 6)] = [mid_t2, mid_right, br, bot_t2]     # 後排右

    def _compute_serve_zones(self, side: str):
        """
        計算某端的發球區（端線後方延伸）

        發球區在端線外側延伸約 50px（或場地高度的 5%）
        分為左/中/右 3 區

        Args:
            side: 'far' 或 'near'
        """
        # 場地高度（近端到遠端的 Y 距離）
        court_height = abs(self.near_left[1] - self.far_left[1])
        extend = max(50, court_height * 0.08)  # 延伸距離

        if side == 'far':
            # far 端發球區: 在 far line 上方延伸
            inner_left = self.far_left
            inner_right = self.far_right
            # 向上延伸
            outer_left = (inner_left[0], inner_left[1] - extend)
            outer_right = (inner_right[0], inner_right[1] - extend)
        else:
            # near 端發球區: 在 near line 下方延伸
            inner_left = self.near_left
            inner_right = self.near_right
            # 向下延伸
            outer_left = (inner_left[0], inner_left[1] + extend)
            outer_right = (inner_right[0], inner_right[1] + extend)

        # 三等分
        inner_t1 = _lerp(inner_left, inner_right, 1/3)
        inner_t2 = _lerp(inner_left, inner_right, 2/3)
        outer_t1 = _lerp(outer_left, outer_right, 1/3)
        outer_t2 = _lerp(outer_left, outer_right, 2/3)

        self._zones[(side, 'serve_1')] = [inner_left, inner_t1, outer_t1, outer_left]    # 左
        self._zones[(side, 'serve_2')] = [inner_t1, inner_t2, outer_t2, outer_t1]         # 中
        self._zones[(side, 'serve_3')] = [inner_t2, inner_right, outer_right, outer_t2]   # 右

    def get_serve_zone(self, position: Tuple[float, float], side: str) -> int:
        """
        判斷發球員在哪個發球區

        Args:
            position: (x, y) 發球員位置
            side: 'far' 或 'near'

        Returns:
            1=左, 2=中, 3=右, 0=無法判定
        """
        for zone_id in (1, 2, 3):
            key = (side, f'serve_{zone_id}')
            if key in self._zones:
                if _point_in_polygon(position, self._zones[key]):
                    return zone_id

        # 如果不在嚴格的發球區內，用 X 座標做粗略判斷
        return self._fallback_serve_zone(position, side)

    def _fallback_serve_zone(self, position: Tuple[float, float], side: str) -> int:
        """用 X 座標粗略判斷發球區"""
        if side == 'far':
            left_x = self.far_left[0]
            right_x = self.far_right[0]
        else:
            left_x = self.near_left[0]
            right_x = self.near_right[0]

        if abs(right_x - left_x) < 1e-6:
            return 0

        ratio = (position[0] - left_x) / (right_x - left_x)
        if ratio < 1/3:
            return 1
        elif ratio < 2/3:
            return 2
        else:
            return 3

    def get_reception_zone(self, position: Tuple[float, float], side: str) -> int:
        """
        判斷接球位置在哪個區

        Args:
            position: (x, y) 接球位置
            side: 'far' 或 'near'

        Returns:
            1-6 (前排左/中/右=1-3, 後排左/中/右=4-6), 0=無法判定
        """
        for zone_id in range(1, 7):
            key = (side, zone_id)
            if key in self._zones:
                if _point_in_polygon(position, self._zones[key]):
                    return zone_id

        # 如果不在嚴格區域內，用相對位置做粗略判斷
        return self._fallback_reception_zone(position, side)

    def _fallback_reception_zone(self, position: Tuple[float, float], side: str) -> int:
        """用相對位置粗略判斷接球區"""
        x, y = position

        if side == 'far':
            top_y = self.far_left[1]
            bot_y = self.net_y
            left_x = self.far_left[0]
            right_x = self.far_right[0]
        else:
            top_y = self.net_y
            bot_y = self.near_left[1]
            left_x = self.near_left[0]
            right_x = self.near_right[0]

        if abs(bot_y - top_y) < 1e-6 or abs(right_x - left_x) < 1e-6:
            return 0

        y_ratio = (y - top_y) / (bot_y - top_y)
        x_ratio = (x - left_x) / (right_x - left_x)
        x_ratio = max(0.0, min(1.0, x_ratio))

        # 左/中/右
        if x_ratio < 1/3:
            col = 0  # 左
        elif x_ratio < 2/3:
            col = 1  # 中
        else:
            col = 2  # 右

        # 前排/後排
        if side == 'far':
            # far: 前排靠網(bottom), 後排遠網(top)
            is_front = y_ratio > 0.5
        else:
            # near: 前排靠網(top), 後排遠網(bottom)
            is_front = y_ratio < 0.5

        if is_front:
            return col + 1  # 前排: 1, 2, 3
        else:
            return col + 4  # 後排: 4, 5, 6

    def get_zone_polygon(self, side: str, zone_id) -> Optional[List[Tuple[float, float]]]:
        """
        取得某區域的多邊形座標（用於視覺化）

        Args:
            side: 'far' 或 'near'
            zone_id: 1-6 (接球區) 或 'serve_1'/'serve_2'/'serve_3' (發球區)

        Returns:
            多邊形頂點列表，或 None
        """
        key = (side, zone_id)
        return self._zones.get(key)

    def get_all_zone_polygons(self, side: str) -> Dict:
        """
        取得某半場所有區域的多邊形

        Args:
            side: 'far' 或 'near'

        Returns:
            {zone_id: polygon_points, ...}
        """
        result = {}
        for key, polygon in self._zones.items():
            if key[0] == side:
                result[key[1]] = polygon
        return result

    def get_side_for_position(self, position: Tuple[float, float]) -> str:
        """
        判斷位置在哪一側

        Args:
            position: (x, y)

        Returns:
            'far' 或 'near'
        """
        return 'far' if position[1] < self.net_y else 'near'
