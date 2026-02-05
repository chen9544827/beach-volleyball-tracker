# Multi-Venue ROI Configuration System - Implementation Summary

## 實作完成日期
2026-02-03

## 實作概述

實作了完整的多場地 ROI（Region of Interest）配置系統，允許為不同攝影機角度和場地設定專屬的分數顯示區域座標，解決了硬編碼 ROI 座標無法處理多場地影片的問題。

## 實作內容

### 新增檔案（5 個）

#### 1. `video_processing/roi_config_generator.py` (~250 行)
**功能：** 互動式 ROI 配置生成器

**特點：**
- 基於 `court_config_generator.py` 的設計模式
- 使用 OpenCV 滑鼠回調處理拖曳框選
- 實時視覺反饋（綠色框 = Team1，紅色框 = Team2）
- 自動驗證 ROI 尺寸（最小 10x10 像素）
- 輸出 JSON 格式配置檔案

**使用方式：**
```bash
python video_processing/roi_config_generator.py \
    --video input_video/original_video/match.mp4 \
    --output roi_configs/match_roi_config.json
```

#### 2. `video_processing/batch_roi_config_generator.py` (~150 行)
**功能：** 批次 ROI 配置生成器

**特點：**
- 依序為多個影片生成 ROI 配置
- 自動跳過已存在的配置檔案
- 支援用戶中斷和繼續
- 提供詳細的處理摘要

**使用方式：**
```bash
python video_processing/batch_roi_config_generator.py \
    --video-dir input_video/original_video \
    --output-dir roi_configs
```

#### 3. `batch_video_slicing.py` (~180 行)
**功能：** 批次影片分割腳本

**特點：**
- 自動匹配影片與 ROI 配置檔案
- 為每個影片創建獨立輸出目錄
- 跳過已處理的影片
- 支援 Anaconda 環境執行
- 提供詳細的成功/失敗報告

**使用方式：**
```bash
python batch_video_slicing.py \
    --video-dir input_video/original_video \
    --roi-config-dir roi_configs \
    --output-dir output_data/video_segments_batch
```

#### 4. `roi_configs/README.md`
**功能：** ROI 配置系統完整文件

**內容：**
- 配置檔案格式說明
- 檔案命名規則
- 使用範例
- 疑難排解指南

#### 5. `test/test_roi_config_system.py` (~200 行)
**功能：** ROI 配置系統驗證測試

**測試案例：**
1. 測試有效的 ROI 配置載入
2. 測試檔案不存在時的處理
3. 測試缺少必要欄位時的處理
4. 測試損壞的 JSON 處理
5. 測試 `preview_roi.py` 整合

**測試結果：** 5/5 通過 ✅

### 修改檔案（3 個）

#### 1. `video_processing/video_slicer_by_score.py`
**修改內容：**
- 添加 `import json`
- 新增 `load_roi_config()` 函數（~45 行）
- 添加 `--roi_config` 命令列參數
- 修改 `main()` 函數支援載入 ROI 配置
- 將硬編碼的 `SCORE_ROI_TEAM1` 和 `SCORE_ROI_TEAM2` 替換為變數

**向後相容性：** ✅
- 不指定 `--roi_config` 時使用預設值
- 現有命令繼續有效

#### 2. `video_processing/preview_roi.py`
**修改內容：**
- 添加 `import json`
- 新增 `load_roi_config()` 函數（~40 行）
- 添加 `--roi-config` 命令列參數
- 修改 `preview_roi()` 函數接受 `roi_config_path` 參數
- 支援載入自訂 ROI 配置

**向後相容性：** ✅
- 不指定 `--roi-config` 時使用預設值

#### 3. `test/test_roi_diff_analyzer.py`
**修改內容：**
- 添加 `import json` 和 `import sys`
- 新增 `load_roi_config()` 函數
- 支援 `--roi-config` 命令列參數
- 可從配置檔案載入 ROI 座標

**向後相容性：** ✅
- 保留原有的硬編碼 ROI 座標

### 文件更新（1 個）

#### `CLAUDE.md`
**更新章節：** 開發流程 > 前置步驟：影片分割

**新增內容：**
- 步驟 0-1: 為不同場地設定 ROI
- 步驟 0-2: 預覽 ROI 設定
- 步驟 0-3: 使用 ROI 配置分割影片
- ROI 配置檔案格式說明
- 批次處理工作流程

## JSON 配置格式

```json
{
  "video_name": "FIVB_BVB_WT19_Edmonton_3Star_...",
  "video_resolution": {
    "width": 1280,
    "height": 720
  },
  "score_roi_team1": {
    "x": 280,
    "y": 29,
    "width": 59,
    "height": 51,
    "label": "Team1 Score ROI"
  },
  "score_roi_team2": {
    "x": 287,
    "y": 92,
    "width": 59,
    "height": 50,
    "label": "Team2 Score ROI"
  },
  "notes": "Edmonton 戶外場地，分數顯示在左上角"
}
```

## 設計決策

### 1. 為什麼選擇 JSON 配置方案？

**優勢：**
- ✅ 完全遵循專案現有模式（與 `court_config.json` 一致）
- ✅ 支持版本管理和批量處理
- ✅ 易於批量腳本集成
- ✅ 可創建互動式生成工具

**替代方案（已排除）：**
- ❌ 命令行參數 - 冗長且不符合專案風格
- ❌ 自動偵測 ROI - 複雜度高，準確性不確定

### 2. 為什麼基於 `court_config_generator.py` 設計？

**理由：**
- 專案已有成熟的互動式配置生成模式
- 用戶已熟悉類似的操作流程
- 可複用相同的 UI/UX 設計模式
- 確保一致的用戶體驗

### 3. 為什麼保留預設值？

**向後相容性考量：**
- 不影響現有工作流程
- 允許用戶逐步遷移到新系統
- 減少破壞性變更

## 工作流程

### 完整批次處理流程

```bash
# 步驟 1: 批次生成 ROI 配置
python video_processing/batch_roi_config_generator.py \
    --video-dir input_video/original_video \
    --output-dir roi_configs

# 步驟 2: 批次分割影片
source "C:/Users/Aa954/anaconda3/etc/profile.d/conda.sh" && conda activate base
python batch_video_slicing.py \
    --video-dir input_video/original_video \
    --roi-config-dir roi_configs \
    --output-dir output_data/video_segments_batch

# 步驟 3: 繼續原有的分析流程
python batch_tracking.py --video-dir ... --court-config ...
python batch_test_serve.py --video-dir ... --json-dir ...
```

## 測試結果

### 單元測試
```
ROI 配置系統驗證測試
通過: 5/5 ✅
失敗: 0/5
```

**測試內容：**
1. ✅ 有效的 ROI 配置載入
2. ✅ 檔案不存在時的處理
3. ✅ 缺少必要欄位時的處理
4. ✅ 損壞的 JSON 處理
5. ✅ preview_roi.py 整合

### 集成測試
- ✅ `load_roi_config()` 函數在 `video_slicer_by_score.py` 中正常運作
- ✅ `load_roi_config()` 函數在 `preview_roi.py` 中正常運作
- ✅ 命令列參數正確解析
- ✅ 錯誤處理正確顯示訊息

## 已知限制

1. **GUI 依賴**：ROI 配置生成器需要 GUI 環境（使用 OpenCV 視窗）
2. **手動設定**：無法自動偵測 ROI 位置，需要用戶手動框選
3. **座標固定**：ROI 座標在影片播放期間不會動態調整
4. **單一配置**：每個影片只有一個 ROI 配置（假設分數顯示位置不變）

## 使用建議

### 首次設定
1. 使用 `roi_config_generator.py` 為每個場地設定一個 ROI 配置
2. 使用 `preview_roi.py` 確認 ROI 位置正確
3. 測試單一影片分割，確認分數偵測正常

### 批次處理
1. 使用 `batch_roi_config_generator.py` 批次生成配置
2. 使用 `batch_video_slicing.py` 批次分割影片
3. 繼續原有的分析流程

### 疑難排解
- 若 ROI 偵測不靈敏，調整 `--diff_threshold` 參數
- 若 ROI 位置不正確，刪除配置檔案重新生成
- 使用 `test/test_roi_diff_analyzer.py` 測試 ROI 差異值

## 未來改進方向

### 短期改進
1. **批次預覽工具**：一次預覽所有 ROI 配置
2. **配置驗證工具**：自動驗證 ROI 配置的有效性
3. **配置編輯器**：GUI 工具編輯現有配置

### 長期改進
1. **自動 ROI 偵測**：使用 OCR 或模板匹配自動找到分數區域
2. **動態 ROI 追蹤**：支援分數顯示位置移動的情況
3. **多 ROI 支援**：支援單一影片有多個分數顯示區域

## 檔案清單

### 新增檔案
```
video_processing/
├── roi_config_generator.py           # 互動式 ROI 配置生成器
├── batch_roi_config_generator.py     # 批次 ROI 配置生成器

batch_video_slicing.py                 # 批次影片分割腳本

roi_configs/
└── README.md                          # ROI 配置系統文件

test/
└── test_roi_config_system.py         # ROI 配置系統驗證測試

IMPLEMENTATION_ROI_CONFIG.md           # 本文件
```

### 修改檔案
```
video_processing/
├── video_slicer_by_score.py          # 支援 --roi_config 參數
└── preview_roi.py                    # 支援 --roi-config 參數

test/
└── test_roi_diff_analyzer.py         # 支援 --roi-config 參數

CLAUDE.md                              # 更新文件
```

## 提交訊息建議

```
實作多場地 ROI 配置支援系統

新增功能:
- 互動式 ROI 配置生成器 (roi_config_generator.py)
- 批次 ROI 配置生成器 (batch_roi_config_generator.py)
- 批次影片分割腳本 (batch_video_slicing.py)
- ROI 配置系統完整文件 (roi_configs/README.md)
- ROI 配置系統驗證測試 (test/test_roi_config_system.py)

修改:
- video_slicer_by_score.py: 支援 --roi_config 參數
- preview_roi.py: 支援 --roi-config 參數
- test_roi_diff_analyzer.py: 支援 --roi-config 參數
- CLAUDE.md: 更新影片分割流程文件

測試結果: 5/5 單元測試通過 ✅

向後相容: 不指定配置時使用預設值 ✅
```

## 參考資料

- **設計文件**：計畫檔案中的完整規劃
- **相關模組**：`court_config_generator.py`（設計範本）
- **相關文件**：`CLAUDE.md`（專案指引）

---

**實作完成**：2026-02-03
**測試狀態**：✅ 全部通過
**文件狀態**：✅ 已完成
