---
name: action-detection
description: 當需要修改發球、跳發、接球、動作偵測的邏輯條件與閾值時使用。
---
# 動作偵測狀態機

## 發球偵測 (`core/serve_detector.py`)

狀態機：`SEARCHING_TOSS → CONFIRMING_TOSS → AWAITING_APEX → AWAITING_HIT → [Event] → COOLDOWN`

* **假陽性排除**：Rally 中接球反彈上升易誤判為拋球。使用最近 10 幀 Y 下降趨勢 (Method D) 與 `net_y + 130px` 限制 (Method C, 只看最近 15 幀) 來排除。
* **`min_toss_height` 限制**：在 `AWAITING_APEX` 驗證拋球高度 `< 60px @720p` 視為假陽性，重置為 `SEARCHING_TOSS`。真正發球通常 > 100px，rally 球通常 < 60px。
* 冷卻期 30 幀防止重複偵測。

## 發球員識別：回溯法 (`core/server_identifier.py`)

* 從拋球幀往回搜尋最多 90 幀，找第一個與球重疊的球員 (距離 `< overlap_threshold`)，並排除在排除區域內的球員。

## 接球與跳發偵測

* **跳發 (`core/jump_serve_detector.py`)**：追蹤腳踝 Y 座標（COCO kp 15, 16），跳躍高度需 `> 30px` (@720p) 且連續 3 幀以上。
* **接球 (`core/reception_detector.py`)**：偵測球跨過 net_y 後，尋找距離 `< 80px @720p` 的最近接球員。組合判斷：球員接近 + 球速/方向改變。

## 共通原則

* 球員索引在不同幀之間並不穩定，務必使用位置匹配 (容差 150px @720p)。
* 所有像素閾值以 720p 為基準，自動縮放：`resolution_scale = image_height / 720.0`
* 修改閾值時同時更新靜態閾值**和**動態閾值計算；使用 `diagnose_serve.py` 測試。

## 困難片段跳過條件

資料量大（2000 部），跳過比硬分析更有價值：

| 條件 | 動作 |
|------|------|
| 球偵測率 < 30% | 跳過 |
| 片段 < 5 秒 | 跳過 |
| 無球員偵測 | 跳過 |
| 發球信心度 < 0.4 | 跳過 |
| 追蹤中斷 > 50% 幀 | 跳過 |
