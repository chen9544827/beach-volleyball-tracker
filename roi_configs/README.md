# ROI Configuration Directory

此目錄用於儲存各場地/影片的 ROI（Region of Interest）配置檔案。

## 什麼是 ROI 配置？

ROI 配置定義了影片中分數顯示區域的位置。由於不同場地、攝影機角度或比賽轉播，分數顯示的位置會有所不同，因此需要為每個影片設定專屬的 ROI 座標。

## 檔案命名規則

ROI 配置檔案應與影片檔案對應：

```
影片檔案: FIVB_BVB_WT19_Edmonton_3Star_....mp4
配置檔案: FIVB_BVB_WT19_Edmonton_3Star_..._roi_config.json
```

命名格式：`{影片檔名}_roi_config.json`

## 配置檔案格式

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

## 如何生成 ROI 配置？

### 方法 1: 互動式生成器（單一影片）

```bash
python video_processing/roi_config_generator.py \
    --video input_video/original_video/your_video.mp4 \
    --output roi_configs/your_video_roi_config.json
```

操作步驟：
1. 程式會顯示影片第一幀
2. 用滑鼠拖曳框選 Team1 分數區域（綠色框）
3. 用滑鼠拖曳框選 Team2 分數區域（紅色框）
4. 按 's' 儲存配置
5. 按 'r' 重置重新開始
6. 按 'q' 退出

### 方法 2: 批次生成器（多個影片）

```bash
python video_processing/batch_roi_config_generator.py \
    --video-dir input_video/original_video \
    --output-dir roi_configs
```

此工具會依序為每個影片開啟互動視窗，讓你逐一設定 ROI。

## 如何使用 ROI 配置？

### 預覽 ROI

```bash
python video_processing/preview_roi.py \
    --input input_video/original_video/your_video.mp4 \
    --roi-config roi_configs/your_video_roi_config.json \
    --output roi_preview.jpg
```

### 分割影片（使用 ROI 配置）

```bash
python video_processing/video_slicer_by_score.py \
    --input input_video/original_video/your_video.mp4 \
    --roi_config roi_configs/your_video_roi_config.json \
    --output_dir output_data/video_segments/your_video
```

### 批次分割（多個影片）

```bash
python batch_video_slicing.py \
    --video-dir input_video/original_video \
    --roi-config-dir roi_configs \
    --output-dir output_data/video_segments_batch
```

## 注意事項

1. **ROI 座標格式**：所有座標使用 `(x, y, width, height)` 格式，單位為像素
2. **座標原點**：左上角為 (0, 0)，X 軸向右，Y 軸向下
3. **最小 ROI 尺寸**：建議至少 10x10 像素
4. **版本管理**：建議將 ROI 配置檔案加入 Git 版本控制
5. **向後相容**：若未指定 ROI 配置，系統會使用預設值

## 常見場地 ROI 範例

已設定的場地配置：

- **Edmonton** - 戶外場地，分數在左上角
- **Gstaad** - 室內場地，大型 LED 顯示屏
- **Chetumal** - 戶外場地
- **Jinjiang** - 戶外場地，藍色看台

## 疑難排解

### 問題：ROI 框選位置不正確

**解決方法**：
1. 使用 `preview_roi.py` 確認 ROI 位置
2. 如果位置錯誤，刪除配置檔案並重新生成
3. 確保框選時包含完整的分數顯示區域

### 問題：影片分割偵測不到分數變化

**解決方法**：
1. 使用 `test/test_roi_diff_analyzer.py` 測試 ROI 差異值
2. 調整 `--diff_threshold` 參數（預設 9000）
3. 確保 ROI 區域不要過大（避免包含非分數區域）

### 問題：找不到 ROI 配置檔案

**解決方法**：
1. 確認檔案命名格式：`{影片檔名}_roi_config.json`
2. 確認檔案位於正確的目錄中
3. 使用 `batch_roi_config_generator.py` 批次生成所有配置

## 技術細節

ROI 配置系統的實作細節請參考：

- `video_processing/roi_config_generator.py` - 互動式生成器
- `video_processing/batch_roi_config_generator.py` - 批次生成器
- `video_processing/video_slicer_by_score.py` - ROI 配置載入邏輯
- `batch_video_slicing.py` - 批次處理腳本

完整文件請參考 `CLAUDE.md` 的「前置步驟：影片分割」章節。
