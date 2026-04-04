# CLAUDE.md

本文件為 Claude Code (claude.ai/code) 在此專案中工作時提供指引。

## 語言規範

**所有回覆必須使用繁體中文（Traditional Chinese）。** 無論使用者用什麼語言提問，一律以繁體中文回答。

## 專案概述

Beach Volleyball Tracker 是一套 Python 電腦視覺系統，處理約 2000 部 FIVB 沙灘排球影片（每部約 30 分鐘），使用 YOLO + VballNet 追蹤排球與球員，自動偵測發球事件、跳發類型、接球位置，並匯出 Excel/CSV 分析結果。

**GPU：** RTX 4080 Laptop + RTX 5060 ｜ **分組鍵：** `venue_year_court`（如 `Edmonton_WT19_C4`）

---

## 開發流程

### 步驟 1：ROI 配置（每個場地組一次）

```bash
python video_processing/roi_config_generator_v2.py \
    --video input_video/original_video/Edmonton_match1.mp4 \
    --output roi_configs/videos/Edmonton_match1_roi_config.json

python video_processing/batch_assign_venues.py \
    --video-dir input_video/original_video \
    --venues-dir roi_configs/venues \
    --output-dir roi_configs/videos
```

### 步驟 2：場地設定（每個場地組一次）

```bash
# 自動偵測（推薦）
python batch_court_config.py \
    --video-dir input_video/original_video \
    --output-dir court_configs \
    --auto --verify

# 手動 GUI（fallback）
python court_definition/court_config_generator.py \
    --video_path input_video/segment_001.mp4 \
    --output_path court_configs/Edmonton_WT19_C4.json
```

### 步驟 3：影片分割

```bash
source "C:/Users/Aa954/anaconda3/etc/profile.d/conda.sh" && conda activate base

python batch_video_slicing.py \
    --video-dir input_video/original_video \
    --roi-config-dir roi_configs \
    --output-dir output_data/video_segments_batch
```

### 步驟 4：追蹤 + 發球分析 + 結果匯出

```bash
python batch_test_serve.py \
    --video-dir input_video/analyze_serve \
    --json-dir tracking_output \
    --output results \
    --court-config court_config.json \
    --export-excel
```

---

## 測試

```bash
# 單元測試（共 44 個）
python test/test_filename_parser.py       # 檔名解析（12 案例）
python test/test_court_zones.py           # 場地分區（15 案例）
python test/test_jump_serve_logic.py      # 跳發邏輯（7 案例）
python test/test_data_validator.py        # 資料驗證（10 案例）
```

---

## 關鍵實作細節

### Python 環境

- **執行前必須啟動：** `source "C:/Users/Aa954/anaconda3/etc/profile.d/conda.sh" && conda activate base`
- Windows 需設定 `KMP_DUPLICATE_LIB_OK=TRUE` 和 `workers=0`
- **TensorFlow 環境**（VballNet 訓練/export）：`conda activate tff_env`（Python 3.11, TF 2.16+, Keras 3.x, tf2onnx 1.16.1）

**模型檔案：**
- `models/ball_best.pt` — 自訓練 YOLO 球偵測
- `models/yolov8m-pose.pt` — YOLOv8 Medium 姿態估計
- `models/court_best.pt` — 場地邊界 keypoint 偵測（6 keypoints）

### JSON 讀取與驗證約定（全域規範）

- 追蹤資料 JSON 可能數百 MB，**除非必要，只讀取前 50 行**：`Read tool, limit: 50`
- 一律使用 `safe_load_json()` 替代直接 `json.load()`（返回 `(data, error)` tuple）
- 使用 `validate_center_point()` 驗證座標；在 `min()`/`max()` 前先確認列表非空
- 捕捉 `ValidationError` 並提供明確訊息

### 全域開發規範

1. **Y 軸向下遞增**（Y 越大 = 畫面越下方）；遠端 = 畫面上方（小 Y），近端 = 畫面下方（大 Y）
2. **所有像素閾值必須傳入 `image_height` 做解析度縮放**，基準 720p（`scale = image_height / 720.0`）
3. 大規模處理**不保存逐幀追蹤 JSON**（避免 900GB）；只保存分析結果（< 10KB/影片）
4. 新增追蹤功能時務必整合 `BallTracker` 狀態機，並遵守 `exclusion_zones`
5. 使用 `get_keypoint()` 輔助函數並設定信心度閾值

### 修改發球偵測時

1. 同時更新靜態閾值**和**動態閾值計算
2. 使用 `diagnose_serve.py` 測試參數變更
3. 冷卻期（30 幀）可防止重複偵測
4. **確保閾值乘以 `resolution_scale`**

---

## 已知限制

1. **網子遮擋**：遠端球員被網子遮擋時偵測率下降
2. **攝影機角度依賴**：不同視角需各自設定 court_config（可自動偵測）
3. **最大追蹤間隔**：15 幀（可透過 `--max-occlusion` 調整）
4. **球員 ID 不穩定**：同一球員在不同幀索引可能改變，務必用位置匹配
5. **Windows 編碼**：所有輸出訊息避免 Unicode 特殊字符（使用 [OK], [ERROR]）
6. **Hamburg_WT19_C1**：空拍/俯瞰視角，需跳過（net_y 無意義，球員偵測無效）
7. **VballNet 失球**：擊球瞬間速度最快，VballNet 常失去追蹤，屬根本性限制

---

## Git 分支規範

- `main` — 穩定分支，PR 目標
- `發球偵測測試` — 當前開發分支
