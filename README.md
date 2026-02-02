# Beach Volleyball Tracker 🏐

沙灘排球影片分析系統 - 自動追蹤球、偵測發球事件、識別發球員

## 功能概述

| 功能 | 說明 | 狀態 |
|------|------|------|
| 球追蹤 | 使用 YOLO 模型追蹤排球位置 | ✅ 完成 |
| 球員偵測 | 偵測球員位置和骨架姿態 | ✅ 完成 |
| 場地設定 | 互動式設定場地邊界、排除區域 | ✅ 完成 |
| 發球偵測 | 識別發球事件（拋球→擊球） | ✅ 完成 |
| 發球員識別 | 判斷是哪位球員發球（Lookback 方法） | ✅ 完成 |
| 批次處理 | 批次追蹤和分析多個影片 | ✅ 完成 |
| 跳發偵測 | 判斷發球是否為跳發 | ✅ 完成 |
| 資料驗證 | 完整的輸入驗證系統（混合策略） | ✅ 完成 |

---

## 專案結構

```
beach-volleyball-tracker/
├── core/                              # 核心模組
│   ├── __init__.py
│   ├── ball_tracker.py                # 球追蹤器（YOLO + 軌跡預測）
│   ├── serve_detector.py              # 發球偵測器（拋球→頂點→擊球）
│   ├── server_identifier.py           # 發球員識別（Lookback 方法）
│   ├── jump_serve_detector.py         # 跳發偵測器
│   ├── data_validator.py              # 資料驗證器（2026-02-02 新增）
│   └── error_messages.py              # 錯誤訊息系統（2026-02-02 新增）
│
├── video_processing/                  # 影片處理
│   └── track_ball_and_player_v2.py    # 主要追蹤流程
│
├── court_definition/                  # 場地定義工具
│   └── court_config_generator.py      # 互動式場地設定工具
│
├── batch_tracking.py                  # 批次追蹤腳本
├── batch_test_serve.py                # 批次發球員識別腳本
├── court_config.json                  # 場地設定檔（排除區域）
│
├── test/                              # 測試目錄（2026-02-02 新增）
│   ├── test_jump_serve_logic.py       # 跳發邏輯單元測試（7 個測試）
│   └── test_data_validator.py         # 資料驗證器測試（10 個測試）
│
├── test_server_identification.py      # 發球員識別測試腳本
├── visualize_tracking.py              # 追蹤結果視覺化工具
├── diagnose_serve.py                  # 發球偵測診斷工具
├── test_improvements.py               # 測試腳本
│
├── CLAUDE.md                          # Claude Code 專案指引
├── IMPLEMENTATION_SUMMARY.md          # 實作總結（2026-02-02）
├── VERIFICATION_CHECKLIST.md          # 驗證檢查表（2026-02-02）
│
├── models/                            # YOLO 模型（需自行放置）
│   ├── best.pt                        # 排球偵測模型
│   └── yolov8x-pose.pt                # 球員姿態模型
│
├── input_video/                       # 輸入影片目錄
├── test_output/                       # 追蹤輸出目錄（JSON）
├── batch_test_output/                 # 發球員識別輸出目錄（圖片）
└── README.md
```

---

## 安裝需求

### Python 環境
```bash
Python 3.8+
```

### 安裝套件
```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python   # 注意：不是 opencv-python-headless
pip install numpy
```

### YOLO 模型
- 排球偵測模型：`models/best.pt`
- 球員姿態模型：`models/yolov8x-pose.pt`

---

## 快速開始

### 完整流程（三步驟）

```bash
# Step 1: 設定場地排除區域（只需執行一次）
python court_definition/court_config_generator.py \
    --video_path input_video/analyze_serve/segment_001.mp4 \
    --output_path court_config.json

# Step 2: 批次追蹤所有影片
python batch_tracking.py \
    --video-dir input_video/analyze_serve \
    --output-dir test_output \
    --court-config court_config.json

# Step 3: 批次發球員識別
python batch_test_serve.py \
    --video-dir input_video/analyze_serve \
    --json-dir test_output \
    --output batch_test_output \
    --court-config court_config.json
```

---

## 詳細使用說明

### 1. 場地設定（court_config_generator.py）

互動式工具，用於定義：
- 場地邊界
- **排除區域**（裁判、工作人員、廣告區）
- 網子位置
- 背景球過濾區

```bash
python court_definition/court_config_generator.py \
    --video_path input_video/segment_001.mp4 \
    --output_path court_config.json
```

**操作說明：**
| 步驟 | 動作 | 按鍵 |
|------|------|------|
| 1 | 點擊場地四角（左上→左下→右下→右上） | 點擊後按 `q` 確認 |
| 2 | 框選排除區域（裁判位置） | 按 `a` 新增，`q` 確認，`n` 下一步 |
| 3 | 點擊網子位置 | 點擊後自動進入下一步 |
| 4 | 框選背景球區域（可選） | 按 `a` 新增，`n` 完成 |

**輸出檔案 (court_config.json)：**
```json
{
    "court_boundary_polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "exclusion_zones": [
        {
            "name": "右上角裁判區",
            "polygon": [[1050,0], [1050,280], [1456,280], [1456,0]]
        }
    ],
    "net_y": 280,
    "background_ball_zones": []
}
```

---

### 2. 批次追蹤（batch_tracking.py）

追蹤所有影片中的球和球員，輸出 JSON 檔案。

```bash
python batch_tracking.py \
    --video-dir input_video/analyze_serve \
    --output-dir test_output \
    --court-config court_config.json
```

**參數說明：**
| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--video-dir` | 影片目錄 | (必填) |
| `--output-dir` | 輸出目錄 | (必填) |
| `--court-config` | 場地設定 JSON | None |
| `--detection-interval` | 偵測間隔（每 N 幀） | 1 |
| `--max-occlusion` | 最大遮擋幀數 | 15 |
| `--no-tracker` | 停用球追蹤器 | False |
| `--quiet` | 安靜模式 | False |

**輸出：**
- `{video_name}_all_frames_data_with_pose.json` - 每個影片的追蹤數據

---

### 3. 批次發球員識別（batch_test_serve.py）

分析追蹤數據，識別發球員。

```bash
python batch_test_serve.py \
    --video-dir input_video/analyze_serve \
    --json-dir test_output \
    --output batch_test_output \
    --court-config court_config.json
```

**參數說明：**
| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--video-dir` | 影片目錄 | (必填) |
| `--json-dir` | JSON 追蹤數據目錄 | (必填) |
| `--output` | 輸出目錄 | batch_test_output |
| `--court-config` | 場地設定 JSON | None |
| `--no-images` | 不儲存圖片 | False |
| `--verbose` | 詳細模式 | False |

**輸出：**
- `{video_name}_server_FOUND.jpg` - 找到球員與球重疊的幀（**判斷依據**）
- `{video_name}_server_TOSS.jpg` - 拋球幀（參考）
- `{video_name}_server_HIT.jpg` - 擊球幀（參考）

---

## 核心演算法

### 發球員識別 - Lookback 方法

**問題**：在拋球幀或擊球幀時，球可能已經離開球員手中，難以判斷誰是發球員。

**解決方案**：從拋球幀**往回找**，直到找到球員與球**重疊**（距離 < 100 像素）的幀。

```
拋球幀 (frame 344)
    ↓
檢查：有球員與球重疊嗎？→ 沒有
    ↓
往回一幀 (frame 343)
    ↓
檢查：有球員與球重疊嗎？→ 沒有
    ↓
... 重複 ...
    ↓
找到！frame 310 有球員與球重疊
    ↓
該球員 = 發球員 ✅
```

**排除區域**：裁判、工作人員等會被排除，不會被誤判為發球員。

---

### 發球偵測流程

```
球軌跡分析
    ↓
偵測拋球（球向上移動，vy > 8）
    ↓
偵測頂點（球開始向下）
    ↓
偵測擊球（速度 > 40，水平移動）
    ↓
輸出發球事件
```

---

## JSON 數據格式

### 追蹤數據 (`*_all_frames_data_with_pose.json`)

```json
{
  "metadata": {
    "video_path": "input_video/segment_029.mp4",
    "total_frames": 744,
    "fps": 25.0
  },
  "frames": [
    {
      "frame_id": 0,
      "ball_detections": [
        {
          "box_coords": [650, 280, 680, 310],
          "confidence": 0.85,
          "center_point": [665, 295],
          "is_in_background_zone": false
        }
      ],
      "player_detections": [
        {
          "box_coords": [600, 400, 700, 700],
          "confidence": 0.92,
          "center_point": [650, 550],
          "pose_keypoints": [[x, y, conf], ...]
        }
      ]
    }
  ]
}
```

### 骨架關鍵點 (COCO 17-point)

| Index | 關鍵點 | 用途 |
|-------|--------|------|
| 0 | 鼻子 | - |
| 5, 6 | 左/右肩 | 身體位置 |
| 7, 8 | 左/右肘 | 手臂動作 |
| 9, 10 | 左/右手腕 | **發球判斷** |
| 11, 12 | 左/右髖 | 身體位置 |
| 13, 14 | 左/右膝 | 跳發偵測 |
| 15, 16 | 左/右腳踝 | **跳發偵測** |

---

## 輸出圖片說明

### FOUND 幀（判斷依據）
- 顯示找到球員與球重疊的幀
- 綠色框 = 發球員
- 藍色框 = 其他球員
- 黃色圈 = 球
- 紫色框 = 排除區域

### TOSS / HIT 幀（參考）
- 顯示拋球幀和擊球幀
- 用於驗證發球員識別結果

---

## 診斷工具

### 發球偵測診斷

```bash
python diagnose_serve.py \
    --input test_output/segment_029_all_frames_data_with_pose.json
```

**輸出：**
- 軌跡統計（偵測率、速度分佈）
- 潛在拋球序列
- 高速事件列表
- 參數調整建議

---

## 參數調整

### 發球偵測參數

```python
config = {
    'hit_v': 40.0,              # 擊球速度閾值
    'toss_vy': 8.0,             # 拋球垂直速度閾值
}
```

**調整建議：**
- 漏偵測發球 → 降低 `hit_v`
- 誤判太多 → 提高 `hit_v`
- 拋球偵測不穩 → 調整 `toss_vy`

### 發球員識別參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `overlap_threshold` | 球員與球重疊的距離閾值 | 100 像素 |
| `max_lookback` | 最多往回找幾幀 | 90 幀 |

---

## 已知限制

1. **網子遮擋**：對面場地的球員可能被網子遮擋，導致偵測不到
2. **快速移動**：極快的發球可能追蹤跳幀
3. **多球干擾**：畫面中有多顆球可能干擾
4. **攝影機角度**：不同角度需要重新設定排除區域

---

## 測試

### 單元測試（2026-02-02 新增）

```bash
# 測試跳發邏輯（7 個測試案例）
python test/test_jump_serve_logic.py

# 測試資料驗證器（10 個測試案例）
python test/test_data_validator.py
```

**測試覆蓋：**
- 跳發連續序列偵測（包含邊界情況）
- JSON 資料驗證（嚴格與寬鬆策略）
- 錯誤處理與容錯能力

### 回歸測試

```bash
# 完整批次測試
python batch_test_serve.py \
    --video-dir input_video/analyze_serve \
    --json-dir test_output \
    --output verification_output \
    --court-config court_config.json
```

---

## 待開發功能

- [x] ~~跳發偵測（分析腳踝位置變化）~~ ✅ 已完成
- [ ] 發球落點預測
- [ ] 發球速度估算（需要場地校準）
- [ ] GUI 介面
- [ ] 即時分析模式

---

## 更新日誌

### v2.1 (2026-02-02) - 穩定性與驗證系統

**🐛 錯誤修復：**
- 修復跳發偵測連續序列演算法（最長序列在結尾時的 bug）
- 修復所有 Windows cp950 編碼問題

**✨ 新功能：**
- 完整資料驗證系統（`core/data_validator.py`, `core/error_messages.py`）
  - 混合驗證策略：關鍵欄位嚴格驗證，次要欄位寬鬆處理
  - 統一的繁體中文錯誤訊息
  - 支援遮擋情況下的部分資料缺失
- 17 個單元測試（100% 通過率）
- 詳細文件：`IMPLEMENTATION_SUMMARY.md`, `VERIFICATION_CHECKLIST.md`

**🔧 整合改進：**
- `batch_test_serve.py` - 整合驗證系統
- `batch_tracking.py` - 整合驗證系統
- `core/serve_detector.py` - 編碼修復
- `core/jump_serve_detector.py` - 邏輯修復與驗證

**📊 測試結果：**
- 單元測試：17/17 通過
- 批次測試：10/10 影片成功（100%）
- 跳發偵測準確率：90%

### v2.0 (2025-01-26)
- ✅ 新增 Lookback 方法識別發球員
- ✅ 新增場地排除區域功能
- ✅ 新增批次追蹤腳本 (batch_tracking.py)
- ✅ 新增批次發球員識別腳本 (batch_test_serve.py)
- ✅ 新增場地設定工具 (court_config_generator.py)
- ✅ 輸出圖片顯示排除區域

### v1.0 (2025-01-23)
- ✅ 基本球追蹤功能
- ✅ 球員姿態偵測
- ✅ 發球偵測（拋球→擊球）
- ✅ 基本發球員識別

---

## License

MIT License

---

## 聯絡方式

如有問題或建議，請開 Issue 或聯繫作者。