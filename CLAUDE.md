# CLAUDE.md

本文件為 Claude Code (claude.ai/code) 在此專案中工作時提供指引。

## 專案概述

Beach Volleyball Tracker (沙灘排球影片分析系統) 是一個基於 Python 的電腦視覺系統，用於處理 **~2000 部 FIVB 沙灘排球巡迴賽影片**（每部約 30 分鐘，橫跨兩年賽事），自動：
- 使用 YOLO 模型追蹤排球和球員位置
- 偵測發球事件（拋球 → 頂點 → 擊球）
- 使用「回溯法（Lookback Method）」識別發球員
- 分類發球類型為跳發或站發
- 偵測排球落點位置（開發中）
- 使用卡爾曼濾波（Kalman Filtering）和拋物線軌跡預測處理遮擋
- 最終彙整成 Excel/CSV 供後續分析

### 專案規模與挑戰

| 項目 | 數值 |
|------|------|
| 影片總數 | ~2000 部 |
| 每部時長 | ~30 分鐘 |
| 總時長 | ~1000 小時 |
| 場地數量 | 待確認（預估 20-60 個場地組） |
| GPU 資源 | RTX 4080 Laptop + RTX 5060 |
| 預估處理時間 | ~15 天（雙 GPU + 跳幀） |
| 最終輸出 | Excel/CSV |

### 影片檔名格式

影片命名包含豐富的元資料（依 FIVB 世界巡迴賽 / VIS 系統慣例），可自動解析場地和分組：
```
FIVB_BVB_WT19_Edmonton_3Star_1718_C4_QT_W_007_Strauss_T_...
|    |   |    |         |      |    |  |  | |
|    |   |    |         |      |    |  |  | +-- 場次編號 (Match 7)
|    |   |    |         |      |    |  |  +---- 性別 (W=女子, M=男子)
|    |   |    |         |      |    |  +------- 輪次 (QT=資格賽, MD=主賽, SF=準決賽, F=決賽)
|    |   |    |         |      |    +---------- 場地編號 Court (C1-C4)
|    |   |    |         |      +--------------- 日期範圍（比賽天數，格式不固定）
|    |   |    |         +---------------------- 站點星級 (3Star/4Star/5Star)
|    |   |    +-------------------------------- 比賽站點名
|    |   +------------------------------------- 賽季 (WT18=World Tour 2018, WT19=2019)
|    +----------------------------------------- Beach VolleyBall (沙灘排球)
+---------------------------------------------- FIVB (國際排球總會)

後續段落：球員姓名_國家代碼（兩隊），最末可能有 clip/set index（如 _2）
```

**分組邏輯：** 配置單位 = 「站點 + 賽季 + 場地編號」（如 `Edmonton_WT19_C4`）
- 同場比賽固定視角，但兩年同場地可能不同視角
- 同一組的影片共享 court_config 和 ROI config
- 有兩種檔名分隔符變體（底線 `_` 和連字號 `-`）

---

## 完整處理管線

```
[一次性設定]
影片目錄 → 檔名解析 → 場地分組
                          ↓
                   為每組設定配置
                   - ROI config（分數區域）
                   - court_config（場地邊界）
                          ↓
[批次處理]
原始影片 → 分割（依分數變化） → 小片段
  (CPU 多進程)                     ↓
                           追蹤+分析（流式處理）
                           (多 GPU 並行)
                                   ↓
                           品質評分 + 困難片段跳過
                                   ↓
[結果彙整]
所有結果 → Excel/CSV → 品質報告
```

---

## 開發流程

### 步驟 1: ROI 配置（分數區域，每個場地組一次）

使用場地模板系統（推薦），同一場地的影片共享 ROI 設定：

```bash
# 互動式設定（支援場地選擇/創建）
python video_processing/roi_config_generator_v2.py \
    --video input_video/original_video/Edmonton_match1.mp4 \
    --output roi_configs/videos/Edmonton_match1_roi_config.json

# 批次指定已有場地模板
python video_processing/batch_assign_venues.py \
    --video-dir input_video/original_video \
    --venues-dir roi_configs/venues \
    --output-dir roi_configs/videos
```

**目錄結構：**
```
roi_configs/
├── venues/                      # 場地模板（可重複使用）
│   ├── Edmonton.json
│   └── Gstaad.json
└── videos/                      # 影片配置（連結到場地）
    ├── Edmonton_match1_roi_config.json
    └── Gstaad_match1_roi_config.json
```

### 步驟 2: 場地設定（court_config，每個場地組一次）

```bash
python court_definition/court_config_generator.py \
    --video_path input_video/segment_001.mp4 \
    --output_path court_configs/Edmonton_WT19_C4.json
```

court_config 提供：場地邊界、排除區域（裁判等）、網子位置、背景球過濾。
落點偵測需要 court_config 來判斷有效/出界。

### 步驟 3: 影片分割

```bash
source "C:/Users/Aa954/anaconda3/etc/profile.d/conda.sh" && conda activate base

# 單一影片
python video_processing/video_slicer_by_score.py \
    --input input_video/original_video/full_match.mp4 \
    --roi_config roi_configs/match_roi_config.json \
    --output_dir output_data/video_segments_with_score/match_name

# 批次分割
python batch_video_slicing.py \
    --video-dir input_video/original_video \
    --roi-config-dir roi_configs \
    --output-dir output_data/video_segments_batch
```

### 步驟 4: 追蹤 + 發球分析

```bash
# 批次追蹤
python batch_tracking.py \
    --video-dir input_video/analyze_serve \
    --output-dir test_output \
    --court-config court_config.json

# 發球分析
python batch_test_serve.py \
    --video-dir input_video/analyze_serve \
    --json-dir test_output \
    --output batch_test_output \
    --court-config court_config.json
```

### 步驟 5: 結果彙整

```bash
# 發球分析 + 接球偵測 + CSV/Excel 匯出
python batch_test_serve.py \
    --video-dir input_video/analyze_serve \
    --json-dir test_output \
    --output batch_test_output \
    --court-config court_config.json \
    --export-excel
```

---

## 核心架構

### 狀態機設計

**發球偵測** (`core/serve_detector.py`) 使用有限狀態機：
```
SEARCHING_TOSS → CONFIRMING_TOSS → AWAITING_APEX → AWAITING_HIT → [Event] → COOLDOWN
```
- 根據影片統計動態調整閾值
- 使用物理約束（垂直比例、加速度）驗證發球
- 與 BallTracker 整合以獲得平滑軌跡

**球追蹤器** (`core/ball_tracker.py`) 管理追蹤狀態：
```
TRACKING ─(偵測失敗)→ OCCLUDED ─(持續失敗)→ LOST
    ↑                      │
    └──────(偵測成功)───────┘
```
- 使用卡爾曼濾波平滑位置
- 在遮擋期間使用拋物線預測（預設最多 15 幀）
- 當兩種方法都可用時融合預測結果

### 發球員識別 - 回溯法（Lookback Method）

`core/server_identifier.py` 從拋球幀往回搜尋最多 90 幀，找到第一個球員與球重疊的幀（距離 < overlap_threshold），排除在排除區域內的球員。

### 跳發偵測

`core/jump_serve_detector.py` 分析腳踝關鍵點軌跡：
- 從 FOUND 幀追蹤腳踝 Y 座標至擊球幀
- 跳躍閾值：最小 30px 高度（720p 基準）、3 幀以上連續
- **修復紀錄 (2026-02-02)**：修復連續序列偵測演算法

### 場地分區系統

`core/court_zones.py` 將場地分為標準沙排區域：
- 3 個發球區（左/中/右，端線後方）
- 6 個接球區（前排 1-3 + 後排 4-6，每側）
- 從 `court_boundary_polygon` (4 點) + `net_y` 自動計算
- 考慮透視變形（遠端比近端窄）

### 接球偵測

`core/reception_detector.py` 從擊球幀追蹤球軌跡：
- 偵測球跨過 net_y（進入對方半場）
- 找到最近的接球員（距離 < 80px）
- 組合判斷：球員接近 + 球速/方向改變
- 輸出：接球幀、接球區域(1-6)、接球員、接球時間

### 檔名解析與影片分組

`core/filename_parser.py` 自動解析 FIVB 影片檔名：
- 支援底線和連字號兩種格式
- 提取：場地、賽季、場地編號、性別、輪次、場次
- 按 `venue_year_court` 自動分組
- `scan_video_directory()` 掃描目錄、`group_videos()` 分組

### 批次分段管線

`batch_segment_pipeline.py` 自動化影片分段流程：
- 掃描影片目錄 -> 解析檔名 -> 分組 -> 匹配 ROI -> 分段
- 支援 `--dry-run`、`--roi-only`、`--slice-only` 模式
- 缺少 ROI 時自動啟動 GUI 設定工具

### 結果匯出

`core/result_exporter.py` 匯出分析結果：
- CSV/Excel 格式，30+ 欄位
- 包含：基本資訊、發球分析、發球區、接球分析、品質指標
- `export_summary_json()` 輸出統計摘要

### 資料驗證系統

系統採用**混合驗證策略**（`core/data_validator.py`）：
- 關鍵欄位嚴格驗證（frames, frame_id, metadata）
- 次要欄位寬鬆處理（ball_detections, player_detections）
- 使用 `safe_load_json()` 載入所有 JSON 檔案
- 使用 `validate_center_point()` 驗證座標

---

## 重要常數與分辨率正規化

### 當前閾值（以 720p 為基準）

**重要：所有像素閾值都需要分辨率正規化**（待實作）

```python
# 正規化公式：actual_threshold = base_threshold * (actual_height / 720)
REFERENCE_HEIGHT = 720

# core/serve_detector.py
toss_vy: 8.0              # 拋球垂直速度閾值 (px/frame @720p)
hit_v: 40.0               # 擊球速度閾值 (px/frame @720p)
max_frames_to_apex: 75    # 從拋球到頂點的最大幀數

# core/server_identifier.py
overlap_threshold: 70.0   # 球員與球重疊的最大距離 (px @720p)
max_lookback: 90          # 向後搜尋的最大幀數

# core/jump_serve_detector.py
jump_threshold: 30.0      # 最小跳躍高度 (px @720p)
min_jump_frames: 3        # 超過閾值的最小連續幀數
```

### 困難片段跳過條件

資料量大（2000 部），跳過比硬分析更有價值：
- 球偵測率 < 30% → 跳過
- 片段 < 5 秒 → 跳過
- 無球員偵測 → 跳過
- 發球信心度 < 0.4 → 跳過
- 追蹤中斷 > 50% 幀 → 跳過

---

## 資料結構

### 場地設定 (court_config.json)
```json
{
  "court_boundary_polygon": [[x,y], ...],
  "exclusion_zones": [{"name": "...", "polygon": [[x,y], ...]}],
  "net_y": 280,
  "background_ball_zones": [...]
}
```

### 追蹤資料 (*_all_frames_data_with_pose.json)
```json
{
  "metadata": {"video_path": "...", "total_frames": 744, "fps": 25.0},
  "frames": [
    {
      "frame_id": 0,
      "ball_detections": [{"box_coords": [...], "confidence": 0.85, "center_point": [x, y]}],
      "player_detections": [{"box_coords": [...], "confidence": 0.92, "center_point": [x, y], "pose_keypoints": [[x, y, conf], ...]}]
    }
  ]
}
```

### COCO-17 關鍵點索引
```python
9, 10: 左/右手腕    # 用於發球偵測
13, 14: 左/右膝蓋   # 用於跳躍分析
15, 16: 左/右腳踝   # 跳發偵測的主要關鍵點
```

---

## 關鍵實作細節

### Python 環境

- 本專案需要在 **Anaconda base 環境** 中執行
- Anaconda 安裝路徑：`C:\Users\Aa954\anaconda3\`
- 執行腳本前：`source "C:/Users/Aa954/anaconda3/etc/profile.d/conda.sh" && conda activate base`

**模型檔案：**
- `models/ball_best.pt` - 自訓練 YOLO 球偵測模型
- `models/yolov8m-pose.pt` - YOLOv8 Medium 姿態估計模型（已從 s 升級為 m）

### JSON 文件讀取最佳實踐

**重要：節省 Token 消耗**
- 追蹤資料 JSON（`*_all_frames_data_with_pose.json`）可能數百 MB
- **除非必要，只讀取前 50 行查看格式**：`Read tool, limit: 50`
- 完整讀取僅在需要統計、搜尋特定幀時使用

### 儲存策略

**大規模處理時不保存逐幀追蹤 JSON**（避免 900GB）：
- 追蹤完成後即時分析，只保存分析結果（<10KB/影片）
- 如需重新分析，重新追蹤比保存更實際
- 開發/測試階段可保存完整 JSON

### 關鍵整合要點

**新增追蹤功能時**：
1. 務必整合 `BallTracker` 狀態機
2. 遵守場地設定中的排除區域
3. 使用 `get_keypoint()` 輔助函數並設定信心度閾值
4. 記住：Y 軸向下遞增（Y 值越大 = 畫面越下方）
5. 使用 `safe_load_json()` 載入所有 JSON 檔案
6. 使用 `validate_center_point()` 驗證座標
7. **所有像素閾值必須支援分辨率正規化**

**修改發球偵測時**：
1. 同時更新靜態閾值**和**動態閾值計算
2. 使用 `diagnose_serve.py` 測試參數變更
3. 冷卻期（30 幀）可防止重複偵測
4. **確保閾值乘以 resolution_scale**

**處理球員追蹤時**：
1. 球員索引在不同幀之間**並不穩定** - 務必使用位置匹配
2. 匹配容差：150px（@720p）

**資料驗證**：
1. 使用 `safe_load_json()` 替代直接 `json.load()`
2. 使用 `DataValidator` 驗證關鍵欄位
3. 空值檢查：在 `min()`/`max()` 前檢查列表非空
4. 捕捉 `ValidationError` 並提供明確訊息

---

## 輸出格式

### 圖片標註
- 綠色框 = 發球員
- 藍色框 = 其他球員
- 黃色圈 = 球
- 紫色多邊形 = 排除區域
- 紅點 = 手腕關鍵點
- 橘點 = 腳踝關鍵點（僅發球員）

### Excel/CSV 輸出格式

**發球明細表：每個發球事件一行**
```
基本資訊: video_name, venue, year, court, gender, round, match_number, star_level, group_key
發球分析: serve_detected, toss_frame, hit_frame, hit_speed, server_index, confidence, serve_type, is_jump_serve, jump_height
發球區域: serve_zone (1-3), serving_side (near/far)
接球分析: reception_detected, reception_frame, reception_zone (1-6), receiver_index, time_to_reception
品質指標: quality_grade (A/B/C/F), ball_detection_rate, status
```

**品質等級：**
- A 級：偵測率 > 70%
- B 級：偵測率 50-70%
- C 級：偵測率 30-50%（結果可能不可靠）
- F 級：偵測率 < 30%（建議排除）

---

## 測試

```bash
# 單元測試（共 44 個）
python test/test_filename_parser.py       # 檔名解析（12 案例）
python test/test_court_zones.py           # 場地分區（15 案例）
python test/test_jump_serve_logic.py      # 跳發邏輯（7 案例）
python test/test_data_validator.py        # 資料驗證（10 案例）

# 回歸測試
python batch_test_serve.py \
    --video-dir input_video/analyze_serve \
    --json-dir test_output \
    --output verification_output \
    --court-config court_config.json
```

---

## 已知限制

1. **分辨率依賴**：所有像素閾值以 720p 校準，不同分辨率需正規化（待實作）
2. **網子遮擋**：對面場地球員被網子遮擋時偵測率下降
3. **攝影機角度依賴**：不同角度需各自設定 court_config
4. **最大追蹤間隔**：15 幀（可透過 `--max-occlusion` 調整）
5. **球員 ID 不穩定**：同一球員在不同幀索引可能改變
6. **Windows 編碼**：所有輸出訊息避免 Unicode 特殊字符（使用 [OK], [ERROR]）

## Git 分支結構

- `main` - 穩定分支
- `發球偵測測試` - 當前開發分支

## 開發 Roadmap

| Phase | Task | Status |
|-------|------|--------|
| **P0** | FIVB filename parsing + video grouping | Done |
| **P1** | Court zone system (3 serve + 6 reception zones) | Done |
| **P2** | Reception detection (ball tracking after hit) | Done |
| **P3** | Pass quality assessment | Pending (needs requirement discussion) |
| **P4** | CSV/Excel result export | Done |
| **P5** | VideoContext lightweight metadata | Done |
| **Batch** | Automated segmentation pipeline | Done |
| **Next** | Court config setup per venue group | In Progress |
| **Next** | Large-scale batch processing (multi-GPU) | Pending |
