---
name: vballnet-model
description: 當需要處理排球追蹤、修改 VballNet 推理邏輯、調整模型閾值或比較 YOLO 與 VballNet 時使用。
---
# VballNet 模型與推理知識

## 球偵測方案比較
* YOLO (ball_best.pt)：單幀 BGR 輸入，偵測率約 30%，快速。
* VballNetV1b_best 單段：seq9 灰階疊加輸入，偵測率約 62.6%。
* **VballNetV1b_best 兩段 (推薦)**：seq9 灰階疊加 + 遠側 45% 裁切，偵測率約 67.7%。
* VballNetV1_150：3 幀 RGB 疊加，偵測率約 31.1% (官方原始模型)。

## 兩段式推理 (Two-pass inference)
* 推薦使用 `fast-volleyball-tracking-inference/src/inference_onnx_twopass.py`，效能約 83fps。
* 關鍵參數：`--far_crop_ratio 0.45` (裁切比例)、`--far_merge_threshold 0.85`。

## 追蹤整合與限制
* 整合腳本 (`tools/merge_vball_detections.py`) 參數：`BALL_BOX_RADIUS=12`，`DEFAULT_CONFIDENCE=0.85`，`margin_ratio=0.05`。
* **失球問題限制**：發球員擊球瞬間速度最快，VballNet 常在此時失去球的追蹤，導致真實發球的擊球幀無法偵測，這是根本性限制。