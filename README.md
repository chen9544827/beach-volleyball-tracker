# Beach Volleyball Tracker（沙灘排球影片分析系統）

基於 Python 的電腦視覺系統，自動分析 FIVB 沙灘排球比賽影片——追蹤球與球員、偵測發球事件、分析接球位置，並將結果匯出為 Excel/CSV。

## 功能總覽

| 功能 | 說明 | 狀態 |
|------|------|------|
| 球偵測（YOLO） | 單幀 YOLO 排球偵測，偵測率約 30% | 完成 |
| 球偵測（VballNet） | 時序 9 幀模型，偵測率約 50% | 完成 |
| 球員偵測 | 球員位置與姿態估計 | 完成 |
| 場地設定 | 互動式場地邊界與排除區域設定 | 完成 |
| 自動場地偵測 | YOLO keypoint 模型自動偵測場地邊界 | 完成 |
| 發球偵測 | 識別發球事件（拋球 → 頂點 → 擊球） | 完成 |
| 假陽性過濾 | `min_toss_height` 過濾 rally 球誤觸發 | 完成 |
| 發球員識別 | 回溯法識別發球球員 | 完成 |
| 跳發偵測 | 依腳踝關鍵點軌跡分類跳發 / 站發 | 完成 |
| 解析度縮放 | 所有像素閾值自動依影片解析度縮放 | 完成 |
| 資料驗證 | 混合策略輸入驗證系統 | 完成 |
| 檔名解析 | FIVB 檔名解析 + 影片自動分組 | 完成 |
| 場地分區 | 每側 3 個發球區 + 6 個接球區 | 完成 |
| 接球偵測 | 擊球後追蹤球軌跡偵測接球 | 完成 |
| 結果匯出 | CSV/Excel 結構化輸出（30+ 欄位） | 完成 |
| 標注影片輸出 | 每片段產生含階段標籤的標注 MP4 | 完成 |
| 影片元資料 | 輕量影片元資料記錄 | 完成 |
| 批次分段 | 自動 ROI 匹配 + 影片分割管線 | 完成 |

---

## 專案結構

```
beach-volleyball-tracker/
├── core/                              # 核心模組
│   ├── __init__.py
│   ├── ball_tracker.py                # 球追蹤器（卡爾曼濾波 + 拋物線預測）
│   ├── serve_detector.py              # 發球偵測狀態機（拋球→頂點→擊球）
│   ├── server_identifier.py           # 發球員識別（回溯法）
│   ├── jump_serve_detector.py         # 跳發偵測（解析度感知）
│   ├── auto_court_detector.py         # 自動場地邊界偵測（YOLO keypoint）
│   ├── auto_court_estimator.py        # 從關鍵點估算場地邊界
│   ├── data_validator.py              # 資料驗證器
│   ├── error_messages.py              # 錯誤訊息系統
│   ├── filename_parser.py             # FIVB 檔名解析 + 影片分組
│   ├── court_zones.py                 # 場地分區系統（發球 3 區 + 接球 6 區）
│   ├── reception_detector.py          # 接球偵測器（解析度感知）
│   ├── result_exporter.py             # CSV/Excel 結果匯出
│   ├── static_ball_filter.py          # 靜態球假陽性過濾
│   ├── static_player_filter.py        # 靜止人員過濾（攝影師/裁判）
│   ├── sahi_pose_detector.py          # SAHI 分塊姿態偵測
│   └── video_context.py               # 輕量影片元資料
│
├── video_processing/                  # 影片處理
│   ├── track_ball_and_player_v2.py    # 主要追蹤管線
│   ├── video_slicer_by_score.py       # 依分數變化分割影片
│   ├── roi_config_generator_v2.py     # ROI 設定 GUI 工具
│   └── batch_assign_venues.py         # 批次場地指定
│
├── court_definition/                  # 場地定義工具
│   └── court_config_generator.py      # 互動式場地設定 GUI
│
├── tools/                             # 訓練與工具腳本
│   ├── prepare_court_dataset.py       # 產生 YOLO keypoint 訓練資料
│   ├── train_court_detector.py        # 場地偵測模型訓練腳本
│   ├── merge_vball_detections.py      # 合併 VballNet CSV 至 tracking JSON（支援 V1b 格式）
│   ├── vball_csv_to_cvat.py           # VballNet CSV → CVAT XML 預標註（fine-tune 資料準備）
│   ├── convert_cvat_to_tracknet.py    # CVAT XML → vball-net 訓練格式（含座標縮放 + 幀提取）
│   ├── visualize_serve_video.py       # 產生發球分析標注影片（MP4）
│   ├── diagnose_serve_frames.py       # 發球偵測診斷工具
│   ├── diagnose_pose_detection.py     # 姿態偵測診斷工具
│   └── add_net_overlay.py             # 網子覆蓋工具
│
├── fast-volleyball-tracking-inference/ # VballNet 時序球偵測子專案
│   ├── main.py                         # 推理入口
│   └── src/                            # VballNet 原始碼（ONNX 模型）
│
├── batch_court_config.py              # 批次場地設定（自動/手動/驗證）
├── batch_tracking.py                  # 批次追蹤腳本
├── batch_test_serve.py                # 批次發球分析（主要管線）
├── batch_segment_pipeline.py          # 批次影片分段管線
├── batch_video_slicing.py             # 批次影片分割
├── diagnose_serve.py                  # 單影片發球診斷
├── visualize_tracking.py              # 追蹤結果視覺化
│
├── models/                            # 模型檔案（不含於 repo）
│   ├── ball_best.pt                   # 排球偵測模型（YOLO）
│   ├── court_best.pt                  # 場地邊界 keypoint 模型
│   ├── yolov8m-pose.pt                # 球員姿態估計模型
│   └── vb-models/                     # VballNet 官方原始模型
│       ├── VballNetV1_150.keras        # VballNetV1（seq3 RGB，有源檔）
│       ├── VballNetV1_150_h288_w512.onnx
│       ├── VballNetFastV1_155.keras    # VballNetFastV1 輕量版
│       └── VballNetFastV1_155_h288_w512.onnx
│
├── dataset/                           # 訓練資料集
│   └── court_keypoints/               # 場地 keypoint 資料集（YOLO 格式）
│
├── court_configs/                     # 各場地組的場地設定
├── roi_configs/                       # ROI 設定
│   ├── venues/                        # 場地模板（可重複使用）
│   └── videos/                        # 逐影片設定
│
├── test/                              # 單元測試（共 44 個）
│   ├── test_filename_parser.py        # 檔名解析（12 案例）
│   ├── test_court_zones.py            # 場地分區（15 案例）
│   ├── test_jump_serve_logic.py       # 跳發邏輯（7 案例）
│   └── test_data_validator.py         # 資料驗證（10 案例）
│
├── input_video/                       # 輸入影片
├── output_data/                       # 處理輸出
├── runs/                              # 訓練記錄
├── CLAUDE.md                          # Claude Code 專案指引
└── README.md
```

---

## 環境需求

### Python 環境
```bash
# 建議使用 Anaconda base 環境
source "C:/Users/Aa954/anaconda3/etc/profile.d/conda.sh" && conda activate base
```

### 套件安裝
```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python        # 請勿使用 opencv-python-headless
pip install numpy
pip install openpyxl              # Excel 匯出（選用，未安裝時退回 CSV）
```

---

## 球偵測方案比較

| 方案 | 類型 | 偵測率 | 說明 |
|------|------|--------|------|
| YOLO（`ball_best.pt`） | 單幀 | ~30% | 自訓練，速度快 |
| VballNet V1b_best 單段 | seq9 灰階 | ~62.6% | 基礎 |
| **VballNet V1b_best 兩段** | **seq9 灰階 + crop** | **~67.7%** | **最佳，建議使用** |
| VballNetV1_150（`models/vb-models/`） | seq3 RGB | ~31.1% | 官方原始模型，有 .keras 源檔 |
| VballNetFastV1_155（`models/vb-models/`） | seq3 RGB | ~10.4% | 輕量版，精度差 |
| TrackNetV3（qaz812345） | 時序 | ~7% | 羽毛球訓練，不適用排球 |

> **兩段推理原理：** 遠端球在 512×288 下僅 2~3px；裁切畫面上方 45%（遠側）後縮放至 512×288，遠側球放大約 2.2倍，偵測率 +5pp。

### 方案 A：YOLO（單幀，偵測率約 30%）
```bash
python batch_tracking.py \
    --video-dir input_video/analyze_serve \
    --output-dir tracking_output \
    --court-config court_config.json
```

### 方案 B：VballNet 兩段推理（偵測率 ~68%）— 建議使用
```bash
# 步驟 1：執行 VballNet 兩段推理
cd fast-volleyball-tracking-inference/src
python inference_onnx_twopass.py \
    --video_path ../../output_data/segment_001.mp4 \
    --model_path ../models/VballNetV1b_seq9_grayscale_best.onnx \
    --output_dir ../../output/vball_csv \
    --only_csv --far_crop_ratio 0.45

# 步驟 2：合併至 tracking JSON
cd ../..
python tools/merge_vball_detections.py \
    --json-dir output/tracking_json \
    --csv-dir output/vball_csv \
    --output-dir output/tracking_vball_merged \
    --court-config court_configs/Edmonton_WT19_C4.json
```

VballNet 兩段推理在遠側大幅改善：seg_002 +10.3pp，seg_028 +7.4pp，平均 +5pp。

---

## 完整處理流程

```bash
# 步驟 1：解析檔名並分組（試跑確認）
python batch_segment_pipeline.py --video-dir input_video/original_video --dry-run

# 步驟 2：設定 ROI 設定（缺少時自動開啟 GUI）
python batch_segment_pipeline.py --video-dir input_video/original_video --roi-only

# 步驟 3：依分數變化分割影片
python batch_segment_pipeline.py --video-dir input_video/original_video

# 步驟 4：設定場地設定
# 自動偵測（建議）：
python batch_court_config.py \
    --video-dir input_video/original_video \
    --output-dir court_configs \
    --auto --verify

# 手動 GUI（備用）：
python court_definition/court_config_generator.py \
    --video_path input_video/segment_001.mp4 \
    --output_path court_config.json

# 步驟 5：批次追蹤
python batch_tracking.py \
    --video-dir input_video/analyze_serve \
    --output-dir tracking_output \
    --court-config court_config.json

# 步驟 6：批次發球分析 + 接球偵測 + 結果匯出
python batch_test_serve.py \
    --video-dir input_video/analyze_serve \
    --json-dir tracking_output \
    --output results \
    --court-config court_config.json \
    --export-excel

# 步驟 7（選用）：產生標注影片
python tools/visualize_serve_video.py \
    --summary results/batch_test_summary.json \
    --video-dir input_video/analyze_serve \
    --json-dir tracking_output \
    --output-dir output/serve_videos \
    --court-config court_config.json \
    --context 60
```

---

## 自動場地偵測

YOLO keypoint 模型偵測 6 個場地關鍵點並自動產生 court_config：

| 索引 | 關鍵點 | 說明 |
|------|--------|------|
| 0 | far_left | 遠端左角 |
| 1 | far_right | 遠端右角 |
| 2 | near_left | 近端左角 |
| 3 | near_right | 近端右角 |
| 4 | net_left | 網子左端 |
| 5 | net_right | 網子右端 |

### 模型訓練
```bash
# 1. 從現有 court_config 產生訓練資料
python tools/prepare_court_dataset.py \
    --config-dir court_configs \
    --video-dir input_video/original_video \
    --segment-dir output_data/video_segments \
    --output dataset/court_keypoints

# 2. 訓練模型
python tools/train_court_detector.py

# 3. 自動偵測並驗證
python batch_court_config.py \
    --video-dir input_video/original_video \
    --output-dir court_configs \
    --auto --verify
```

### 排除區域自動生成

排除區域從場地邊界自動計算：
- 左右邊距：場地寬度的 15%
- 遠端邊距：場地高度的 30%（含計分板區域）
- 近端邊距：場地高度的 25%
- 擴展邊界外到畫面邊緣的區域均為排除區域

---

## FIVB 影片檔名格式

影片命名遵循 FIVB 世界巡迴賽 / VIS 系統慣例：

```
FIVB_BVB_WT19_Edmonton_3Star_1718_C4_QT_W_007_Strauss_T_...
|    |   |    |         |      |    |  |  | |
|    |   |    |         |      |    |  |  | +-- 場次編號 (007)
|    |   |    |         |      |    |  |  +---- 性別 (W=女子, M=男子)
|    |   |    |         |      |    |  +------- 輪次 (QT/MD/SF/F)
|    |   |    |         |      |    +---------- 場地編號 (C1-C4)
|    |   |    |         |      +--------------- 日期範圍
|    |   |    |         +---------------------- 星級 (3Star/4Star/5Star)
|    |   |    +-------------------------------- 站點名稱
|    |   +------------------------------------- 賽季 (WT18/WT19)
|    +----------------------------------------- Beach VolleyBall
+---------------------------------------------- FIVB
```

底線（`_`）與連字號（`-`）兩種分隔符均支援。

**分組鍵：** `venue_year_court`（例如：`Edmonton_WT19_C4`）

---

## 場地分區系統

標準沙灘排球場地分區，用於發球與接球分析：

```
              網子
  +-----+-----+-----+
  |  1  |  2  |  3  |  遠端（畫面上方）
  |  左 |  中 |  右 |  接球區 = 前排 1-3
  +-----+-----+-----+
  |  4  |  5  |  6  |  接球區 = 後排 4-6
  |  左 |  中 |  右 |
  +-----+-----+-----+
              網子
  +-----+-----+-----+
  |  1  |  2  |  3  |  近端（畫面下方）
  |  左 |  中 |  右 |  接球區 = 前排 1-3
  +-----+-----+-----+
  |  4  |  5  |  6  |  接球區 = 後排 4-6
  |  左 |  中 |  右 |
  +-----+-----+-----+

發球區（端線後方）：左 / 中 / 右 = 3 個區域
```

分區從 `court_boundary_polygon`（4 點）+ `net_y` 自動計算，並考慮透視變形。

**effective_net_y**：當 `net_y` 在場地邊界 Y 範圍外時（攝影機視角下網子頂部在遠端底線上方，屬正常現象），自動計算 `effective_net_y = far_y + (near_y - far_y) * 0.45` 用於區域劃分；原始 `net_y` 僅用於球過網判斷。

---

## 結果匯出格式

`batch_test_serve.py --export-excel` 輸出結構化資料：

**每個發球事件一行，欄位包含：**
- 基本資訊：video_name, venue, year, court, gender, round, match_number, star_level, group_key
- 發球分析：serve_detected, toss_frame, hit_frame, hit_speed, server_index, confidence, serve_type, is_jump_serve, jump_height
- 發球區域：serve_zone (1-3), serving_side (near/far)
- 接球分析：reception_detected, reception_frame, reception_zone (1-6), receiver_index, time_to_reception
- 品質指標：quality_grade (A/B/C/F), ball_detection_rate, status

**品質等級：**
- A 級：偵測率 > 70%
- B 級：偵測率 50-70%
- C 級：偵測率 30-50%（結果可能不可靠）
- F 級：偵測率 < 30%（建議排除）

---

## 核心演算法

### 發球員識別 — 回溯法（Lookback Method）

從拋球幀往回搜尋最多 90 幀，找到第一個球員與球重疊（距離 < 閾值）的幀，排除位於排除區域內的球員。

### 發球偵測狀態機

```
SEARCHING_TOSS → CONFIRMING_TOSS → AWAITING_APEX → AWAITING_HIT → [事件] → COOLDOWN
```

在 `AWAITING_APEX` 階段額外驗證拋球高度（`min_toss_height = 60px @720p`），過濾 rally 球短暫向上移動造成的假陽性。

### 接球偵測

擊球後追蹤球軌跡，直到球過網並在附近找到接球員。使用球員接近距離 + 球速/方向改變的組合判斷。

### 跳發偵測

分析腳踝關鍵點 Y 軸軌跡（FOUND 幀至擊球幀）。跳躍閾值：最小 30px 高度（720p 基準），自動縮放，需連續 3 幀以上超過閾值。

### 解析度縮放

所有像素閾值依影片解析度自動縮放：
```python
resolution_scale = image_height / 720.0
# 跳躍閾值：    30px * scale
# 重疊閾值：    70px * scale
# 球員距離：    80px * scale
# 匹配容差：   150px * scale
# 拋球高度：    60px * scale
```

---

## 單元測試

```bash
# 共 44 個測試
python test/test_filename_parser.py      # 檔名解析（12 案例）
python test/test_court_zones.py          # 場地分區（15 案例）
python test/test_jump_serve_logic.py     # 跳發邏輯（7 案例）
python test/test_data_validator.py       # 資料驗證（10 案例）
```

---

## 已知限制

1. **網子遮擋**：遠端球員被網子遮擋時偵測率下降
2. **攝影機角度**：不同視角需各自設定 court_config（可自動偵測）
3. **最大追蹤間隔**：15 幀（可透過 `--max-occlusion` 調整）
4. **球員 ID 不穩定**：同一球員在不同幀的索引可能改變，務必用位置匹配
5. **Windows 編碼**：輸出訊息避免 Unicode 特殊字符（使用 [OK], [ERROR]）
6. **Hamburg_WT19_C1**：空拍俯瞰視角，需跳過（與標準側面視角差異過大）

---

## 更新日誌

### v5.0（2026-02-28）— VballNet 整合與假陽性過濾

**新功能：**
- 整合 VballNet 時序球偵測（`fast-volleyball-tracking-inference/`）
  - 9 幀時序上下文，ONNX 模型，偵測率 ~50%（vs YOLO ~30%）
  - `tools/merge_vball_detections.py`：將 VballNet ball.csv 合併至 tracking JSON（含場地 X 範圍濾鏡）
- `core/serve_detector.py` 新增 `min_toss_height` 參數：過濾 rally 球假陽性（預設 60px @720p，自動縮放）
- `analyze_serve_events_v2()` 新增 `image_height` 參數，實現完整解析度感知發球偵測
- `tools/visualize_serve_video.py`：產生含階段標籤、球員標注、網子線、排除區域的標注影片

**改善：**
- 發球分析成功率：2/4 → 4/4（測試片段，使用 VballNet）
- 修復 `batch_test_serve.py` 中 `image_height` UnboundLocalError

### v4.0（2026-02-13）— 自動場地偵測與解析度縮放

**新模組：**
- `core/auto_court_detector.py`：YOLO keypoint 自動場地邊界偵測（6 關鍵點）
- `core/auto_court_estimator.py`：從關鍵點估算場地邊界
- `tools/prepare_court_dataset.py`：從現有 court_config 產生訓練資料
- `tools/train_court_detector.py`：場地偵測模型訓練腳本
- `batch_court_config.py`：批次場地設定（--auto/--verify/--min-confidence）

**改善：**
- 所有像素閾值解析度縮放（跳發、接球、發球員識別）
- CourtZones 新增 `effective_net_y`（net_y 在遠端底線上方時的正確分區處理）
- 從場地邊界自動生成排除區域（比例邊距）

### v3.0（2026-02-07）— 分析管線擴充

**新模組：**
- `core/filename_parser.py`：FIVB 檔名解析 + 影片自動分組
- `core/court_zones.py`：含透視校正的每側 3 發球區 + 6 接球區
- `core/reception_detector.py`：擊球後球軌跡追蹤偵測接球
- `core/result_exporter.py`：30+ 欄位 CSV/Excel 匯出
- `core/video_context.py`：輕量影片元資料（解析度、FPS）
- `batch_segment_pipeline.py`：自動 ROI 匹配 + 影片分段

**測試：** 44 個單元測試（12+15+7+10），通過率 100%

### v2.1（2026-02-02）— 穩定性與驗證

- 修復跳發連續序列演算法
- 修復 Windows cp950 編碼問題
- 新增資料驗證系統（`core/data_validator.py`）

### v2.0（2025-01-26）

- 回溯法識別發球員
- 場地排除區域
- 批次追蹤與分析腳本

### v1.0（2025-01-23）

- 基本球追蹤、球員偵測、發球偵測

---

## 開發 Roadmap

| Phase | Task | Status |
|-------|------|--------|
| **P0** | FIVB filename parsing + video grouping | Done |
| **P1** | Court zone system (3 serve + 6 reception zones) | Done |
| **P2** | Reception detection (ball tracking after hit) | Done |
| **P3** | Pass quality assessment | Pending |
| **P4** | CSV/Excel result export | Done |
| **P5** | VideoContext lightweight metadata | Done |
| **P6** | Auto court detection (YOLO keypoint) | Done |
| **P7** | Resolution scaling for all thresholds | Done |
| **P8** | VballNet temporal ball detection integration | Done |
| **P9** | min_toss_height false positive filter | Done |
| **P10** | Annotated video output tool | Done |
| **P11** | VballNet CVAT pre-annotation workflow | Done |
| **P12** | Pose keypoint coverage via 4x crop scale | Done |
| **P13** | VballNet two-pass inference (62.6%→67.7%) | Done |
| **Batch** | Automated segmentation pipeline | Done |
| **Next** | Large-scale batch processing (multi-GPU) | Pending |
| **Next** | VballNet fine-tune on beach volleyball data | Pending |

---

## 資料結構參考

### 場地設定 (`court_config.json`)

```json
{
  "court_boundary_polygon": [[x,y], [x,y], [x,y], [x,y]],
  "exclusion_zones": [{"polygon": [[x,y], ...]}],
  "net_y": 280,
  "background_ball_zones": [...]
}
```

> `net_y` 在攝影機視角中通常小於遠端底線 Y 值（網子頂部在場地線上方），這是正常現象。

### 追蹤資料格式 (`*_all_frames_data_with_pose.json`)

```json
{
  "metadata": {
    "video_path": "...", "total_frames": 744,
    "fps": 25.0, "image_width": 1920, "image_height": 1080
  },
  "frames": [
    {
      "frame_id": 0,
      "ball_detections": [{"box_coords": [...], "confidence": 0.85, "center_point": [x, y]}],
      "player_detections": [{"box_coords": [...], "confidence": 0.92, "center_point": [x, y], "pose_keypoints": [[x, y, conf], ...]}]
    }
  ]
}
```

### COCO-17 關鍵點索引（常用）

| 索引 | 部位 | 用途 |
|------|------|------|
| 9, 10 | 左/右手腕 | 發球偵測 |
| 13, 14 | 左/右膝蓋 | 跳躍分析 |
| 15, 16 | 左/右腳踝 | 跳發偵測（主要） |

---

## 授權

MIT License
