# 階段1+2 實作指南 - 記分板切片與 ByteTrack 追蹤

## 📋 新增功能總覽

本次更新整合了**階段1 (資料瘦身)** 與 **階段2 (追蹤升級)** 的核心功能：

### 🎯 階段1: 記分板 ROI 監控切片系統
- ✅ 互動式 ROI 標定工具
- ✅ 外部配置檔支援 (JSON)
- ✅ 批次處理多個長影片
- ✅ 自動產生切片報告

### 🎯 階段2: ByteTrack 追蹤整合
- ✅ YOLO 內建 ByteTrack 追蹤器
- ✅ 針對球體/球員的專屬配置
- ✅ 向下相容 (可選擇啟用/關閉)
- ✅ 追蹤品質評估工具

---

## 🚀 快速開始

### 步驟1: 標定記分板 ROI

使用互動式工具標記兩個記分板的位置：

```bash
python tools/scoreboard_roi_marker.py --video_path "path/to/reference_video.mp4" --output "scoreboard_config.json"
```

**操作流程:**
1. 視窗顯示影片首幀
2. 依序點擊 Team1 記分板的**左上角**和**右下角**
3. 依序點擊 Team2 記分板的**左上角**和**右下角**
4. 按 `s` 儲存配置，按 `r` 重新標記，按 `q` 退出

**輸出檔案:** `scoreboard_config.json`

---

### 步驟2: 執行影片切片

#### 選項A: 單一影片切片

使用配置檔:
```bash
python video_processing/video_slicer_by_score.py \
    --input "path/to/long_match.mp4" \
    --output_dir "output_data/segments" \
    --scoreboard_config "scoreboard_config.json" \
    --diff_threshold 9000
```

使用預設值 (硬編碼 ROI):
```bash
python video_processing/video_slicer_by_score.py \
    --input "path/to/long_match.mp4" \
    --output_dir "output_data/segments"
```

#### 選項B: 批次處理多個影片

```bash
python batch_slice_videos.py \
    --input_dir "input_video/raw_matches" \
    --output_dir "output_data/batch_results" \
    --scoreboard_config "scoreboard_config.json" \
    --workers 4 \
    --diff_threshold 9000
```

**參數說明:**
- `--workers`: 並行執行緒數 (建議不超過 CPU 核心數)
- `--diff_threshold`: SAD 閾值 (預設 9000)
- `--min_segment_duration`: 最小片段時長 (秒)
- `--long_segment_threshold`: 長片段閾值 (秒)

**輸出結果:**
- `{video_name}_segments/normal_segments/`: 正常片段
- `{video_name}_segments/long_segments/`: 異常長片段
- `master_slicing_report.txt`: 總體報告
- `master_slicing_report.csv`: CSV 格式報告

---

### 步驟3: 執行追蹤分析

#### 選項A: 不啟用追蹤 (原始逐幀檢測)

```bash
python video_processing/track_ball_and_player.py \
    --input "segment_001.mp4" \
    --output_dir "output_data/tracking"
```

#### 選項B: 啟用 ByteTrack 追蹤 ⭐ 推薦

```bash
python video_processing/track_ball_and_player.py \
    --input "segment_001.mp4" \
    --output_dir "output_data/tracking" \
    --use_tracking
```

**追蹤配置檔位置:**
- 球體: `configs/bytetrack_ball.yaml`
- 球員: `configs/bytetrack_player.yaml`

**JSON 輸出差異:**

未啟用追蹤:
```json
{
  "frame_id": 0,
  "ball_detections": [
    {
      "track_id": -1,  // 無追蹤 ID
      "box_coords": [100, 200, 150, 250],
      "confidence": 0.95
    }
  ]
}
```

啟用追蹤:
```json
{
  "frame_id": 0,
  "ball_detections": [
    {
      "track_id": 3,  // 持續追蹤 ID
      "box_coords": [100, 200, 150, 250],
      "confidence": 0.95
    }
  ]
}
```

---

### 步驟4: 評估追蹤品質

```bash
python tools/tracking_quality_checker.py \
    --json_path "output_data/tracking/segment_001_all_frames_data_with_pose.json"
```

**輸出指標:**
- 總軌跡數
- 平均軌跡長度
- 平均連續率 (目標 >95%)
- 疑似 ID 切換次數
- 軌跡長度分佈直方圖

**品質評級:**
- ✅ 優秀: 連續率 >95%
- ✓ 良好: 連續率 85-95%
- ⚠️ 中等: 連續率 70-85%
- ❌ 需改善: 連續率 <70%

---

## 🔧 追蹤器參數調優

### 球體追蹤配置 (`configs/bytetrack_ball.yaml`)

```yaml
track_high_thresh: 0.5      # 高置信度閾值
track_low_thresh: 0.1       # 低置信度閾值
new_track_thresh: 0.6       # 新軌跡建立閾值
track_buffer: 30            # 允許消失 30 幀 (~1秒)
match_thresh: 0.8           # IoU 匹配閾值
```

**調優建議:**
- **ID 切換過多**: 提高 `match_thresh` (0.8 → 0.85)
- **遺失追蹤過早**: 增加 `track_buffer` (30 → 50)
- **誤判新目標**: 提高 `new_track_thresh` (0.6 → 0.7)

### 球員追蹤配置 (`configs/bytetrack_player.yaml`)

```yaml
track_high_thresh: 0.4      # 球員較易辨識
track_low_thresh: 0.15      # 允許遮擋恢復
new_track_thresh: 0.5       # 新球員進場閾值
track_buffer: 50            # 更長緩衝時間
match_thresh: 0.7           # 考慮姿態變化
```

---

## 📊 完整工作流程範例

### 情境: 處理 10 小時比賽影片

```bash
# 1. 標定記分板 ROI (一次性操作)
python tools/scoreboard_roi_marker.py \
    --video_path "sample_match.mp4" \
    --output "scoreboard_config.json"

# 2. 批次切片所有長影片 (資料瘦身)
python batch_slice_videos.py \
    --input_dir "raw_matches/" \
    --output_dir "output_data/sliced_segments" \
    --scoreboard_config "scoreboard_config.json" \
    --workers 4

# 3. 對切片後的片段執行追蹤分析
python run_analysis_all_in_one.py \
    --input_folder "output_data/sliced_segments/match01_segments/normal_segments" \
    --court_config "court_config.json" \
    --workers 4

# 4. 評估追蹤品質 (抽樣檢查)
python tools/tracking_quality_checker.py \
    --json_path "volleyball_analysis_results/segment_001/tracking_output/segment_001_all_frames_data_with_pose.json"
```

---

## 🔍 疑難排解

### Q1: ROI 預覽視窗一片黑?
**A:** 影片編碼問題。嘗試使用 `ffmpeg` 重新編碼:
```bash
ffmpeg -i input.mp4 -c:v libx264 -preset fast output.mp4
```

### Q2: 切片產生過多短片段?
**A:** 調高 `--diff_threshold` (9000 → 12000) 或使用測試工具確認閾值:
```bash
python test/test_roi_diff_analyzer.py
```

### Q3: ByteTrack 啟用後反而效果變差?
**A:** 可能的原因:
1. 配置參數不適合您的場景 → 調整 YAML 配置
2. 模型置信度過低 → 檢查偵測品質
3. 影片幀率過低 → 調整 `track_buffer`

### Q4: 追蹤品質檢查顯示 "無追蹤數據"?
**A:** 確認追蹤時有使用 `--use_tracking` 參數。檢查 JSON 中的 `track_id` 欄位是否為 -1。

---

## 📁 新增檔案清單

```
beach-volleyball-tracker/
├── tools/
│   ├── scoreboard_roi_marker.py          # ROI 互動式標定工具
│   └── tracking_quality_checker.py       # 追蹤品質評估工具
├── configs/
│   ├── bytetrack_ball.yaml               # 球體追蹤配置
│   └── bytetrack_player.yaml             # 球員追蹤配置
├── batch_slice_videos.py                 # 批次切片腳本
└── video_processing/
    ├── video_slicer_by_score.py (已修改)  # 支援外部配置
    └── track_ball_and_player.py (已修改)  # 整合 ByteTrack
```

---

## 🎯 下一步計畫

- [ ] **階段3: Homography 映射** - 實作空間座標轉換
- [ ] **階段4: 邏輯實作** - 落點判定、In/Out 判斷
- [ ] **階段5: 量產部署** - CLI 平行化、Headless 執行
- [ ] **TensorRT 優化** - 模型加速 (2-3倍提升)

---

## 📝 版本記錄

- **v1.0** (2026-01-20): 初始實作階段1+2
  - ROI 標定工具
  - 批次切片系統
  - ByteTrack 整合
  - 追蹤品質評估

---

**聯絡資訊:** 如有問題請參考專案 README.md 或提交 Issue。
