# 場地模板系統使用指南

## 快速開始

### 情境 1：你有 5 個來自不同場地的影片

**步驟 1：為第一個 Edmonton 影片創建場地模板**
```bash
python video_processing/roi_config_generator_v2.py \
    --video input_video/original_video/Edmonton_match1.mp4 \
    --output roi_configs/videos/Edmonton_match1_roi_config.json
```

互動：
1. 目前沒有場地模板，按 **n** 創建新場地
2. 輸入場地名稱：`Edmonton`
3. 拖曳框選 Team1 和 Team2 ROI
4. 按 **s** 儲存

**步驟 2：為第二個 Edmonton 影片使用相同場地**
```bash
python video_processing/roi_config_generator_v2.py \
    --video input_video/original_video/Edmonton_match2.mp4 \
    --output roi_configs/videos/Edmonton_match2_roi_config.json
```

互動：
1. 看到場地列表顯示 `[1] Edmonton`
2. 按 **1** 選擇 Edmonton
3. ROI 框立即顯示在畫面上 ✨
4. 確認無誤後按 **s** 儲存（無需重新框選！）

**步驟 3：重複以上流程為其他場地創建模板**
- Gstaad: 按 **n** 創建新場地
- Chetumal: 按 **n** 創建新場地
- Jinjiang: 按 **n** 創建新場地

---

### 情境 2：你已經創建好所有場地模板，現在要快速為 20 個影片指定場地

```bash
python video_processing/batch_assign_venues.py \
    --video-dir input_video/original_video \
    --venues-dir roi_configs/venues \
    --output-dir roi_configs/videos \
    --preview  # 顯示預覽視窗
```

互動：
1. 看到影片 `Edmonton_match3.mp4`
2. 顯示場地列表：`[1] Edmonton [2] Gstaad [3] Chetumal [4] Jinjiang`
3. 按 **1** 選擇 Edmonton
4. （可選）查看預覽確認 ROI 位置
5. 按 **y** 確認，完成！

重複以上步驟為每個影片指定場地，**無需重新框選 ROI**。

---

## 工作流程對比

### 傳統方式（方案 B）
```
影片 1 → 框選 ROI → 儲存
影片 2 → 框選 ROI → 儲存（重複勞動！）
影片 3 → 框選 ROI → 儲存（重複勞動！）
...
影片 20 → 框選 ROI → 儲存（重複勞動！）
```

**時間：** 20 個影片 × 2 分鐘 = **40 分鐘**

### 場地模板方式（方案 A）
```
Edmonton 場地 → 框選 ROI → 儲存模板

影片 1 (Edmonton) → 選擇場地 → 儲存（5 秒）
影片 2 (Edmonton) → 選擇場地 → 儲存（5 秒）
影片 3 (Edmonton) → 選擇場地 → 儲存（5 秒）
...
影片 15 (Edmonton) → 選擇場地 → 儲存（5 秒）

Gstaad 場地 → 框選 ROI → 儲存模板

影片 16 (Gstaad) → 選擇場地 → 儲存（5 秒）
影片 17 (Gstaad) → 選擇場地 → 儲存（5 秒）
...
```

**時間：** 4 個場地 × 2 分鐘 + 20 個影片 × 5 秒 = **10 分鐘**

**節省時間：** 30 分鐘（75% 效率提升）✨

---

## 鍵盤快捷鍵

### roi_config_generator_v2.py

**選擇模式（載入現有場地後）：**
- `s` - 儲存配置
- `e` - 進入編輯模式（重新設定 ROI）
- `q` - 退出

**編輯模式（創建新場地或編輯現有場地）：**
- 拖曳滑鼠 - 框選 ROI
- `s` - 儲存配置
- `r` - 重置（清除 ROI 重新開始）
- `q` - 退出

### batch_assign_venues.py

- `1-9` - 選擇場地
- `s` - 跳過此影片
- `q` - 退出程式
- `y` - 確認選擇
- `n` - 取消選擇

---

## 目錄結構

```
roi_configs/
├── venues/                          # 場地模板（可重複使用）
│   ├── Edmonton.json               # 4 個模板
│   ├── Gstaad.json
│   ├── Chetumal.json
│   └── Jinjiang.json
│
└── videos/                          # 影片配置（連結到場地）
    ├── Edmonton_match1_roi_config.json      # venue: "Edmonton"
    ├── Edmonton_match2_roi_config.json      # venue: "Edmonton"
    ├── Edmonton_match3_roi_config.json      # venue: "Edmonton"
    ├── ...
    ├── Gstaad_match1_roi_config.json        # venue: "Gstaad"
    └── ...
```

## 優勢總結

✅ **避免重複勞動** - 同一場地只需設定一次 ROI
✅ **即時視覺反饋** - 切換場地時 ROI 框立即顯示
✅ **場地管理** - 集中管理所有場地配置
✅ **批次處理** - 快速為多個影片指定場地
✅ **易於維護** - 修改場地模板會影響所有使用該場地的影片
✅ **向後相容** - 仍然支援傳統的單一配置方式

## 常見問題

### Q1: 如何修改已存在的場地模板？

**方法 1：使用 roi_config_generator_v2.py**
```bash
python video_processing/roi_config_generator_v2.py \
    --video input_video/original_video/any_video.mp4 \
    --output roi_configs/videos/temp.json
```
1. 選擇要修改的場地
2. 按 **e** 進入編輯模式
3. 重新框選 ROI
4. 按 **s** 儲存（會覆蓋場地模板）

**方法 2：直接編輯 JSON**
```bash
# 編輯場地模板
vim roi_configs/venues/Edmonton.json
```

### Q2: 如何查看某個影片使用哪個場地？

```bash
cat roi_configs/videos/your_video_roi_config.json | grep venue
```

輸出：
```json
"venue": "Edmonton",
```

### Q3: 如何批次修改影片的場地？

使用 `batch_assign_venues.py` 重新指定場地即可。

### Q4: 場地模板和影片配置有什麼區別？

| 項目 | 場地模板 | 影片配置 |
|------|----------|----------|
| 位置 | `venues/` | `videos/` |
| 用途 | 可重複使用的 ROI 設定 | 連結影片到場地 |
| 內容 | `venue_name`, ROI | `video_name`, `venue`, ROI |
| 修改影響 | 所有使用該場地的影片 | 僅該影片 |

### Q5: 如果場地 ROI 略有不同怎麼辦？

**選項 1：** 創建新場地模板（例如 `Edmonton_v2`）
**選項 2：** 使用傳統方式為該影片單獨設定 ROI

---

## 工具對照表

| 工具 | 用途 | 輸出 |
|------|------|------|
| `roi_config_generator_v2.py` | 創建場地模板 + 影片配置 | `venues/*.json` + `videos/*.json` |
| `batch_assign_venues.py` | 批次指定場地 | `videos/*.json` |
| `roi_config_generator.py` | 傳統方式（單一配置） | `*.json` |
| `batch_roi_config_generator.py` | 傳統方式（批次） | `*.json` |

## 完整範例

假設你有以下影片：
```
input_video/original_video/
├── Edmonton_2024_Final.mp4
├── Edmonton_2024_SemiFinal.mp4
├── Gstaad_2024_Final.mp4
├── Chetumal_2024_Match1.mp4
└── Chetumal_2024_Match2.mp4
```

**步驟 1：創建 Edmonton 場地模板**
```bash
python video_processing/roi_config_generator_v2.py \
    --video input_video/original_video/Edmonton_2024_Final.mp4 \
    --output roi_configs/videos/Edmonton_2024_Final_roi_config.json
```
→ 按 **n** 創建新場地 "Edmonton"，框選 ROI，按 **s** 儲存

**步驟 2：為另一個 Edmonton 影片使用模板**
```bash
python video_processing/roi_config_generator_v2.py \
    --video input_video/original_video/Edmonton_2024_SemiFinal.mp4 \
    --output roi_configs/videos/Edmonton_2024_SemiFinal_roi_config.json
```
→ 按 **1** 選擇 "Edmonton"，ROI 立即顯示，按 **s** 儲存

**步驟 3：創建其他場地模板並批次指定**
```bash
# 創建 Gstaad 模板
python video_processing/roi_config_generator_v2.py \
    --video input_video/original_video/Gstaad_2024_Final.mp4 \
    --output roi_configs/videos/Gstaad_2024_Final_roi_config.json

# 創建 Chetumal 模板
python video_processing/roi_config_generator_v2.py \
    --video input_video/original_video/Chetumal_2024_Match1.mp4 \
    --output roi_configs/videos/Chetumal_2024_Match1_roi_config.json

# 批次指定剩餘影片
python video_processing/batch_assign_venues.py \
    --video-dir input_video/original_video \
    --venues-dir roi_configs/venues \
    --output-dir roi_configs/videos
```

完成！🎉
