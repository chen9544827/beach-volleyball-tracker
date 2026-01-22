# 階段1+2 實作完成總結

## ✅ 實作狀態

**日期**: 2026-01-20  
**狀態**: 已完成

---

## 📦 新增功能清單

### 🎯 階段1: 資料瘦身 (Data Reduction)

#### 1. **ROI 互動式標定工具**
- **檔案**: `tools/scoreboard_roi_marker.py`
- **功能**: 點擊式標記記分板位置，自動產生 JSON 配置
- **輸出**: `scoreboard_config.json`
- **特色**:
  - 視覺化預覽首幀
  - 即時繪製標記點
  - 錯誤重標機制 (按 'r')
  - 座標自動計算

#### 2. **切片腳本配置檔支援**
- **檔案**: `video_processing/video_slicer_by_score.py` (已修改)
- **新增功能**:
  - `--scoreboard_config` 參數
  - `load_scoreboard_config()` 函數
  - 自動 Fallback 到預設值
  - ROI 來源顯示 (配置檔 vs 預設值)

#### 3. **批次切片處理腳本**
- **檔案**: `batch_slice_videos.py`
- **功能**: 
  - 多執行緒平行處理
  - 遞迴搜尋影片檔案
  - 錯誤恢復機制
  - 總報告產生 (TXT + CSV)
- **輸出**:
  - `master_slicing_report.txt`
  - `master_slicing_report.csv`

---

### 🎯 階段2: 追蹤升級 (Tracking Upgrade)

#### 4. **ByteTrack 追蹤整合**
- **檔案**: `video_processing/track_ball_and_player.py` (已修改)
- **核心改動**:
  - `detect_ball()` 新增 `use_tracking` 參數
  - `detect_and_filter_players()` 新增 `use_tracking` 參數
  - `run_tracking_and_save_json()` 支援追蹤模式
  - JSON 輸出新增 `track_id` 欄位
- **向下相容**: 預設不啟用追蹤，透過 `--use_tracking` 啟用

#### 5. **ByteTrack 配置檔**
- **檔案**:
  - `configs/bytetrack_ball.yaml` (球體專用)
  - `configs/bytetrack_player.yaml` (球員專用)
- **針對性調整**:
  - 球體: 短暫遮擋容忍 (30幀)
  - 球員: 長時間遮擋容忍 (50幀)
  - 不同的匹配閾值策略

#### 6. **追蹤品質評估工具**
- **檔案**: `tools/tracking_quality_checker.py`
- **功能**:
  - 提取所有追蹤軌跡
  - 計算關鍵指標 (長度、連續率、ID切換)
  - 品質評級 (優秀/良好/中等/需改善)
  - 軌跡長度直方圖
- **輸出**: 終端報告 (未來可擴展至文件)

---

## 📊 技術亮點

### 1. **向下相容設計**
```python
# 追蹤模組可選擇性啟用
python track_ball_and_player.py --input video.mp4 --output_dir out/  # 舊版模式
python track_ball_and_player.py --input video.mp4 --output_dir out/ --use_tracking  # 新版模式
```

### 2. **配置外部化**
```json
{
  "score_roi_team1": {"x": 280, "y": 29, "w": 59, "h": 51},
  "score_roi_team2": {"x": 287, "y": 92, "w": 59, "h": 50},
  "recommended_diff_threshold": 9000
}
```

### 3. **平行處理架構**
```python
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(process_video, v): v for v in videos}
```

### 4. **追蹤數據結構**
```json
{
  "frame_id": 0,
  "ball_detections": [
    {
      "track_id": 3,  // 新增欄位
      "box_coords": [100, 200, 150, 250],
      "confidence": 0.95,
      "center_point": [125, 225],
      "is_in_background_zone": false
    }
  ],
  "player_detections": [
    {
      "track_id": 1,  // 新增欄位
      "box_coords": [300, 400, 400, 600],
      "confidence": 0.92,
      "pose_keypoints": [[x, y, conf], ...],
      ...
    }
  ]
}
```

---

## 📖 使用文檔

### 主要指南
- **`STAGE1_2_GUIDE.md`**: 完整使用教學
  - 快速開始
  - 參數說明
  - 工作流程範例
  - 疑難排解
  - 調優建議

### 測試腳本
- **`test_stage1_2_integration.py`**: 自動化測試 (6個測試案例)
- **`quick_verify.py`**: 快速驗證檔案完整性

---

## 🔧 關鍵技術決策

### 為什麼選擇 Ultralytics 內建 ByteTrack?
✅ **優點**:
- 零額外依賴
- API 簡單 (`.track()` vs `.predict()`)
- 官方維護，穩定性高
- 配置檔支援

❌ **替代方案 (未採用)**:
- 手動整合 ByteTrack: 需要額外依賴，複雜度高
- BoT-SORT: 速度較慢，不適合大規模處理

### 為什麼球體和球員使用不同配置?
**球體特性**:
- 運動速度快 (40-80 km/h)
- 容易被球網短暫遮擋
- 外觀一致性高

**球員特性**:
- 運動速度慢 (< 10 km/h)
- 可能長時間被其他球員遮擋
- 外觀變化大 (姿態、角度)

---

## ⚡ 效能優化策略

### 階段1: 資料瘦身效益
```
原始: 1000 小時影片
  ↓ [記分板觸發切片]
切片: ~200 小時有效片段 (減少 80%)
  ↓ [只對有效片段執行 YOLO]
效益: 節省 800 小時的 GPU 推理時間
```

### 階段2: 追蹤 vs 檢測
```
逐幀檢測模式:
  - 每幀執行完整 YOLO 推理
  - 無 ID 連續性
  - 計算成本: 100%

追蹤模式:
  - 首幀完整檢測，後續幀追蹤
  - ID 持續追蹤 (減少後續分析複雜度)
  - 計算成本: ~60-70% (估計)
```

---

## 🎯 下一階段規劃

### 階段3: 空間映射 (Spatial Mapping)
- [ ] Homography Matrix 計算
- [ ] 像素 → 真實座標轉換
- [ ] 遠端接球員位置映射 (1-6 號位)
- [ ] 落點 In/Out 判定基礎

### 階段4: 邏輯實作 (Logic Implementation)
- [ ] 跳發 vs 站立發球判定 (已有基礎)
- [ ] 球體軌跡拋物線擬合
- [ ] 落點檢測 (Bounce Detection)
- [ ] 遮擋軌跡補償

### 階段5: 量產部署 (Production)
- [ ] CLI 參數化設計
- [ ] Headless 執行模式
- [ ] 斷點續傳機制
- [ ] TensorRT 模型轉換
- [ ] Docker 容器化

---

## 📝 已知限制與改進方向

### 當前限制
1. **ROI 標定**: 需要手動點擊，無自動偵測
2. **追蹤配置**: 固定參數，未實作自適應調整
3. **錯誤處理**: 批次處理失敗時無自動重試
4. **報告格式**: 僅支援 TXT/CSV，無視覺化圖表

### 改進方向
1. **自動 ROI 偵測**: OCR + 模板匹配
2. **動態閾值**: 基於影片亮度/對比度自適應
3. **視覺化 Dashboard**: Web 介面監控進度
4. **模型量化**: INT8/FP16 加速推理

---

## 📈 測試覆蓋率

### 自動化測試 (test_stage1_2_integration.py)
- ✅ ROI 標定工具存在性
- ✅ 配置檔支援檢查
- ✅ ByteTrack 整合驗證
- ✅ 配置檔格式檢查
- ✅ 品質檢查工具驗證
- ✅ 批次腳本功能檢查

### 手動測試需求
- 🔲 實際影片 ROI 標定
- 🔲 切片功能端到端測試
- 🔲 追蹤模式效能對比
- 🔲 品質報告準確性驗證

---

## 🚀 快速開始命令

```bash
# 1. 標定 ROI
python tools/scoreboard_roi_marker.py --video_path sample.mp4 --output scoreboard_config.json

# 2. 批次切片
python batch_slice_videos.py --input_dir raw_videos/ --scoreboard_config scoreboard_config.json --workers 4

# 3. 追蹤分析 (啟用 ByteTrack)
python video_processing/track_ball_and_player.py --input segment_001.mp4 --output_dir tracking/ --use_tracking

# 4. 品質評估
python tools/tracking_quality_checker.py --json_path tracking/segment_001_all_frames_data_with_pose.json
```

---

## 📞 技術支援

如有問題，請參考:
1. **`STAGE1_2_GUIDE.md`** - 完整使用指南
2. **`README.md`** - 專案總覽
3. **程式碼內註解** - 詳細實作說明

---

**實作完成時間**: 2026-01-20  
**總程式碼行數**: ~1500+ 行  
**新增檔案數**: 8 個  
**修改檔案數**: 2 個  

🎉 **階段1+2 實作完成！**
