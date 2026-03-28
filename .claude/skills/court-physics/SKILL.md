---
name: court-physics
description: 當需要計算場地邊界、判斷球是否過網、修改接球/發球區域邏輯，或處理 3D 透視畸變問題時使用。
---
# 場地物理與攝影機視角限制

## 攝影機視角限制 (透視畸變)
攝影機從側面拍攝的透視畸變使得 X/Y 像素無法直接對應 3D 場地：
* **不能**用球的絕對像素座標判斷是否在「發球區（端線後方）」。
* **不能**用球的 X 座標判斷是否在左/中/右發球區。
* **可以**用 `net_y` (水平分割線) 判斷球在遠端或近端半場。
* **注意**：`net_y` 在攝影機視角中通常小於遠端底線 Y 值 (網子頂部在場地線上方)，這是正常現象。

## 自動場地偵測與分區
* `core/auto_court_detector.py` 使用 YOLO (`court_best.pt`) 偵測 6 個關鍵點：far_left(0), far_right(1), near_left(2), near_right(3), net_left(4), net_right(5)。
* 排除區擴展邊距比例：lr=0.15, far=0.30, near=0.25。
* `core/court_zones.py` 將場地分為 3 個發球區與 6 個接球區。
* **`effective_net_y`**：當 `net_y` 在邊界外時，使用 `far_y + (near_y - far_y) * 0.45` 計算有效網高用於分區，原始 `net_y` 僅用於球過網判斷。