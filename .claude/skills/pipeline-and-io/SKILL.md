---
name: pipeline-and-io
description: 當需要修改完整處理管線、批次分段流程、輸出格式（CSV/圖片標注）或輸出影片產生時使用。
---
# 管線與輸出系統

## 完整處理管線架構

```
[一次性設定]
影片目錄 → 檔名解析 → 場地分組
                      ↓
               為每組設定配置
               - ROI config（分數區域）
               - court_config（場地邊界，可自動偵測）
                      ↓
[批次處理]
原始影片 → 分割（依分數變化） → 小片段
(CPU 多進程)                    ↓
                       追蹤+分析（流式處理）
                       (多 GPU 並行)
                               ↓
                       品質評分 + 困難片段跳過
                               ↓
[結果彙整]
所有結果 → Excel/CSV → 品質報告
```

## 批次分段管線（`batch_segment_pipeline.py`）

- 掃描影片目錄 → 解析檔名 → 分組 → 匹配 ROI → 分段
- 支援 `--dry-run`、`--roi-only`、`--slice-only` 模式
- 缺少 ROI 時自動啟動 GUI 設定工具

```bash
# 試跑確認分組
python batch_segment_pipeline.py --video-dir input_video/original_video --dry-run
# 僅設定 ROI
python batch_segment_pipeline.py --video-dir input_video/original_video --roi-only
# 完整分段
python batch_segment_pipeline.py --video-dir input_video/original_video
```

## 結果匯出（`core/result_exporter.py`）

CSV/Excel 格式，30+ 欄位：
- **基本資訊**：video_name, venue, year, court, gender, round, match_number, star_level, group_key
- **發球分析**：serve_detected, toss_frame, hit_frame, hit_speed, server_index, confidence, serve_type, is_jump_serve, jump_height
- **發球區域**：serve_zone (1-3), serving_side (near/far)
- **接球分析**：reception_detected, reception_frame, reception_zone (1-6), receiver_index, time_to_reception
- **品質指標**：quality_grade (A/B/C/F), ball_detection_rate, status

`export_summary_json()` 輸出統計摘要。

**品質等級：** A(>70%) / B(50-70%) / C(30-50%, 不可靠) / F(<30%, 建議排除)

## 圖片標注色碼

- 綠色框 = 發球員
- 藍色框 = 其他球員
- 黃色圈 = 球
- 紫色多邊形 = 排除區域
- 紅點 = 手腕關鍵點
- 橘點 = 腳踝關鍵點（僅發球員）

## 輸出影片產生（`tools/visualize_serve_video.py`）

```bash
python tools/visualize_serve_video.py \
    --summary output/serve_analysis_vball/batch_test_summary.json \
    --video-dir output_data/test_segment \
    --json-dir output/tracking_vball_merged \
    --output-dir output/serve_videos \
    --court-config court_configs/Edmonton_WT19_C4.json \
    --context 60
```

標注內容：球（黃圓）、發球員（綠框）、其他球員（藍橘框）、網子（虛線）、排除區域（紫框）、階段標籤（PRE-TOSS / TOSS / IN-FLIGHT / HIT / AFTER-HIT / RECEPTION）、JUMP/STANDING 標籤。
