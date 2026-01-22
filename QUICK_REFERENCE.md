# 🏐 快速參考卡 - 階段1+2 新功能

## 📌 核心工作流程

```
長影片 → [ROI標定] → [切片] → 片段影片 → [追蹤分析] → [品質評估] → 結果
```

---

## 🔧 命令速查

### 1️⃣ ROI 標定 (一次性操作)
```bash
python tools/scoreboard_roi_marker.py \
    --video_path "sample_video.mp4" \
    --output "scoreboard_config.json"
```
**互動操作**: 點擊兩個記分板的左上角和右下角

---

### 2️⃣ 單一影片切片
```bash
python video_processing/video_slicer_by_score.py \
    --input "long_match.mp4" \
    --scoreboard_config "scoreboard_config.json" \
    --diff_threshold 9000
```

---

### 3️⃣ 批次切片 (推薦)
```bash
python batch_slice_videos.py \
    --input_dir "raw_videos/" \
    --scoreboard_config "scoreboard_config.json" \
    --workers 4
```

---

### 4️⃣ 追蹤分析 (不啟用追蹤)
```bash
python video_processing/track_ball_and_player.py \
    --input "segment_001.mp4" \
    --output_dir "tracking_output/"
```

---

### 5️⃣ 追蹤分析 (啟用 ByteTrack) ⭐
```bash
python video_processing/track_ball_and_player.py \
    --input "segment_001.mp4" \
    --output_dir "tracking_output/" \
    --use_tracking
```

---

### 6️⃣ 品質評估
```bash
python tools/tracking_quality_checker.py \
    --json_path "tracking_output/segment_001_all_frames_data_with_pose.json"
```

---

## ⚙️ 關鍵參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--diff_threshold` | 9000 | SAD 閾值,控制切片敏感度 |
| `--workers` | 2 | 平行執行緒數 |
| `--min_segment_duration` | 10 | 最小片段時長 (秒) |
| `--use_tracking` | False | 啟用 ByteTrack 追蹤 |

---

## 📂 輸出結構

```
output_data/
├── batch_results/
│   ├── video1_segments/
│   │   ├── normal_segments/      # 正常片段
│   │   ├── long_segments/        # 長片段
│   │   └── slicing_summary.csv   # 單一影片報告
│   ├── master_slicing_report.txt # 總報告 (文字)
│   └── master_slicing_report.csv # 總報告 (CSV)
│
└── tracking_output/
    └── segment_001_all_frames_data_with_pose.json  # 追蹤數據
```

---

## 🎯 品質指標解讀

| 連續率 | 評級 | 建議 |
|--------|------|------|
| > 95% | ✅ 優秀 | 無需調整 |
| 85-95% | ✓ 良好 | 可接受 |
| 70-85% | ⚠️ 中等 | 考慮調整參數 |
| < 70% | ❌ 需改善 | 必須調整 |

---

## 🔍 疑難排解

### Q: 切片產生過多短片段?
```bash
# 提高閾值
--diff_threshold 12000  # (原 9000)
```

### Q: 追蹤 ID 切換頻繁?
```yaml
# 編輯 configs/bytetrack_ball.yaml
match_thresh: 0.85  # (原 0.8)
```

### Q: 球體追蹤過早消失?
```yaml
# 編輯 configs/bytetrack_ball.yaml
track_buffer: 50  # (原 30)
```

---

## 📚 完整文檔

- **使用指南**: [STAGE1_2_GUIDE.md](STAGE1_2_GUIDE.md)
- **實作總結**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **檔案清單**: [FILE_CHANGES.md](FILE_CHANGES.md)
- **專案總覽**: [README.md](README.md)

---

## ✅ 快速驗證

```bash
# 檢查檔案完整性
python quick_verify.py

# 查看配置檔範例
cat configs/bytetrack_ball.yaml
```

---

**提示**: 第一次使用建議從小規模測試開始 (1-2 個影片),確認效果後再批次處理。

**版本**: v1.0 (2026-01-20)
