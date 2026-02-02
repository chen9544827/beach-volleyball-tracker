# CLAUDE.md

本文件為 Claude Code (claude.ai/code) 在此專案中工作時提供指引。

## 專案概述

Beach Volleyball Tracker (沙灘排球影片分析系統) 是一個基於 Python 的電腦視覺系統，可自動：
- 使用 YOLO 模型追蹤排球和球員位置
- 偵測發球事件（拋球 → 頂點 → 擊球）
- 使用「回溯法（Lookback Method）」識別發球員
- 分類發球類型為跳發或站發
- 使用卡爾曼濾波（Kalman Filtering）和拋物線軌跡預測處理遮擋

## 開發流程

### 三步驟分析流程

所有影片分析都遵循以下順序：

```bash
# 步驟 1: 場地設定（一次性互動設定）
python court_definition/court_config_generator.py \
    --video_path input_video/segment_001.mp4 \
    --output_path court_config.json

# 步驟 2: 批次追蹤（生成 JSON 追蹤資料）
python batch_tracking.py \
    --video-dir input_video/analyze_serve \
    --output-dir test_output \
    --court-config court_config.json

# 步驟 3: 發球分析（輸出標註圖片及發球偵測結果）
python batch_test_serve.py \
    --video-dir input_video/analyze_serve \
    --json-dir test_output \
    --output batch_test_output \
    --court-config court_config.json
```

### 測試個別元件

```bash
# 測試發球偵測診斷
python diagnose_serve.py --input test_output/segment_029_all_frames_data_with_pose.json

# 測試單一影片的發球員識別
python test_server_identification.py

# 測試追蹤視覺化
python visualize_tracking.py
```

## 核心架構

### 狀態機設計

**發球偵測** (`core/serve_detector.py`) 使用有限狀態機：
```
SEARCHING_TOSS → CONFIRMING_TOSS → AWAITING_APEX → AWAITING_HIT → [Event] → COOLDOWN
（尋找拋球）    （確認拋球）        （等待頂點）      （等待擊球）    （事件）   （冷卻期）
```
- 根據影片統計動態調整閾值
- 使用物理約束（垂直比例、加速度）驗證發球
- 與 BallTracker 整合以獲得平滑軌跡

**球追蹤器** (`core/ball_tracker.py`) 管理追蹤狀態：
```
TRACKING ─(偵測失敗)→ OCCLUDED ─(持續失敗)→ LOST
（追蹤中）              （遮擋中）              （丟失）
    ↑                      │
    └──────(偵測成功)───────┘
```
- 使用卡爾曼濾波平滑位置
- 在遮擋期間使用拋物線預測（預設最多 15 幀）
- 當兩種方法都可用時融合預測結果

### 發球員識別 - 回溯法（Lookback Method）

「回溯法」(`core/server_identifier.py`) 解決了在拋球/擊球時球可能已經離手的挑戰：

1. 從拋球幀開始
2. 往回搜尋最多 90 幀
3. 找到第一個球員與球重疊的幀（距離 < 100px）
4. 排除在排除區域內的球員（裁判、工作人員）

這是識別發球員的**主要方法**。

### 跳發偵測

`core/jump_serve_detector.py` 分析腳踝關鍵點軌跡：
- 從 FOUND 幀追蹤腳踝 Y 座標至擊球幀
- 從場地設定或動態估算計算基準線
- 跳躍閾值：最小 30px 高度、3 幀以上連續
- 驗證時機：跳躍頂點應出現在擊球幀附近
- **修復紀錄 (2026-02-02)**：修復連續序列偵測演算法，正確處理最長序列在結尾的情況

### 資料驗證系統 (2026-02-02 新增)

系統採用**混合驗證策略**，平衡嚴格性與容錯能力：

**核心模組：**
- `core/data_validator.py` - 驗證器核心邏輯
- `core/error_messages.py` - 統一的繁體中文錯誤訊息

**驗證策略：**
1. **關鍵欄位（嚴格驗證）**：
   - `frames` (list) - 必須存在且非空
   - `frame_id` (int) - 每幀必須有有效 ID
   - `metadata.video_path` (str) - 必須存在
   - `metadata.fps` (float) - 必須是正數
   - 缺失時拋出 `ValidationError`

2. **次要欄位（寬鬆處理）**：
   - `ball_detections` - 允許空列表（遮擋情況）
   - `player_detections` - 允許空列表（遮擋情況）
   - `pose_keypoints` - 個別關鍵點可能缺失
   - 缺失時記錄警告但繼續處理

**使用方式：**
```python
from core.data_validator import DataValidator, safe_load_json
from core.error_messages import ValidationError

# 載入 JSON
data, error = safe_load_json("tracking_data.json")
if error:
    print(f"載入失敗: {error}")
    return

# 驗證資料
validator = DataValidator(verbose=True)
is_valid, errors = validator.validate_tracking_json(data)
if not is_valid:
    raise ValidationError("資料驗證失敗", errors)
```

**輔助函數：**
```python
from core.data_validator import get_keypoint, validate_center_point

# 安全取得關鍵點（自動處理信心度）
left_ankle = get_keypoint(pose_keypoints, 15, confidence_threshold=0.3)

# 驗證中心點座標
center = validate_center_point(player.get('center_point'))
if center:
    x, y = center
```

## 資料結構

### 場地設定 (court_config.json)
```json
{
  "court_boundary_polygon": [[x,y], ...],  // 場地四個角
  "exclusion_zones": [
    {
      "name": "右上角裁判區",
      "polygon": [[x,y], ...]
    }
  ],
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
      "ball_detections": [{
        "box_coords": [x1, y1, x2, y2],
        "confidence": 0.85,
        "center_point": [x, y],
        "is_in_background_zone": false
      }],
      "player_detections": [{
        "box_coords": [x1, y1, x2, y2],
        "confidence": 0.92,
        "center_point": [x, y],
        "pose_keypoints": [[x, y, conf], ...]  // 17 個 COCO 關鍵點
      }]
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

## 關鍵實作細節

### Python 環境
- Python 3.8+
- **重要**：必須使用 `opencv-python` 而非 `opencv-python-headless`（場地設定需要 GUI）
- `models/` 目錄中的模型：
  - `best.pt` - 自訓練 YOLO 球偵測模型
  - `yolov8x-pose.pt` - YOLOv8 姿態估計模型

### 重要常數
```python
# core/serve_detector.py
toss_vy: 8.0              # 拋球垂直速度閾值
hit_v: 40.0               # 擊球速度閾值
max_frames_to_apex: 75    # 從拋球到頂點的最大幀數

# core/server_identifier.py
overlap_threshold: 100    # 球員與球重疊的最大距離（像素）
max_lookback: 90          # 向後搜尋的最大幀數

# core/jump_serve_detector.py
jump_threshold: 30.0      # 最小跳躍高度（像素）
min_jump_frames: 3        # 超過閾值的最小連續幀數
```

### 關鍵整合要點

**新增追蹤功能時**：
1. 務必整合 `BallTracker` 狀態機 - 絕不繞過它
2. 遵守場地設定中的排除區域
3. 使用 `get_keypoint()` 輔助函數並設定信心度閾值
4. 記住：Y 軸向下遞增（Y 值越大 = 畫面越下方）
5. **使用 `safe_load_json()` 載入所有 JSON 檔案**
6. **使用 `validate_center_point()` 驗證所有座標**

**修改發球偵測時**：
1. 同時更新靜態閾值**和**動態閾值計算
2. 使用 `diagnose_serve.py` 測試參數變更
3. 冷卻期（30 幀）可防止重複偵測

**處理球員追蹤時**：
1. 球員索引在不同幀之間**並不穩定** - 務必使用位置匹配
2. 使用 FOUND 幀的發球員中心點進行跨幀匹配
3. 匹配容差：球員識別為 150px

**資料驗證最佳實踐**：
1. **載入檔案**：使用 `safe_load_json()` 替代直接 `json.load()`
2. **驗證結構**：使用 `DataValidator` 驗證關鍵欄位
3. **空值檢查**：在 `min()`/`max()` 前檢查列表非空
4. **座標驗證**：使用 `validate_center_point()` 處理可能的 None 值
5. **錯誤處理**：捕捉 `ValidationError` 並提供明確訊息

## 輸出慣例

### 圖片標註
- 綠色框 = 發球員
- 藍色框 = 其他球員
- 黃色圈 = 球
- 紫色多邊形 = 排除區域
- 紅點 = 手腕關鍵點
- 橘點 = 腳踝關鍵點（僅發球員）

### 發球類型標籤 (batch_test_serve.py)
- **"JUMP SERVE"** - 黃色背景，顯示跳躍高度
- **"STANDING"** - 灰色背景

### 輸出檔案
```
batch_test_output/
  {video_name}_server_FOUND.jpg   # 重要：這是判定幀
  {video_name}_server_TOSS.jpg    # 僅供參考
  {video_name}_server_HIT.jpg     # 僅供參考
  batch_test_summary.json         # 統計與批次結果
```

## 測試

### 單元測試

```bash
# 測試跳發邏輯修復（7 個測試案例）
python test/test_jump_serve_logic.py

# 測試資料驗證器（10 個測試案例）
python test/test_data_validator.py
```

### 回歸測試

```bash
# 完整批次測試
python batch_test_serve.py \
    --video-dir input_video/analyze_serve \
    --json-dir test_output \
    --output verification_output \
    --court-config court_config.json
```

**驗證標準：**
- 單元測試：17/17 必須通過
- 批次測試：所有影片成功偵測
- 無編碼錯誤（Windows cp950 相容）
- 錯誤訊息為繁體中文

## 已知限制

1. **網子遮擋**：對面場地的球員可能被網子部分遮擋
2. **多球干擾**：透過 `background_ball_zones` 過濾背景球
3. **攝影機角度依賴**：不同角度需要透過 court_config_generator 設定新的排除區域
4. **最大追蹤間隔**：15 幀（可透過 `--max-occlusion` 調整）
5. **Windows 編碼**：所有輸出訊息必須避免 Unicode 特殊字符（使用 ASCII 標籤如 [OK], [ERROR]）

## Git 分支結構

- `main` - 穩定分支（用於 PR）
- `發球偵測測試` - 當前分支，用於發球偵測開發

## 最近更新

### 2026-02-02：跳發邏輯修復與驗證系統

**修復：**
- 跳發偵測連續序列演算法（最長序列在結尾時的 bug）
- 所有 Windows cp950 編碼問題（移除 Unicode 特殊字符）

**新增：**
- 完整資料驗證系統（`core/data_validator.py`, `core/error_messages.py`）
- 17 個單元測試（100% 通過）
- 詳細實作文件（`IMPLEMENTATION_SUMMARY.md`, `VERIFICATION_CHECKLIST.md`）

**整合：**
- `batch_test_serve.py` - 驗證 + 編碼修復
- `batch_tracking.py` - 驗證整合
- `core/serve_detector.py` - 編碼修復
- `core/jump_serve_detector.py` - 邏輯修復 + 驗證

**測試結果：**
- 單元測試：17/17 通過
- 批次測試：10/10 影片成功（100%）
- 跳發偵測：9/10 (90%)，站發：1/10 (10%)
