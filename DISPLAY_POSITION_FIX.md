# ROI 配置生成器顯示位置修正

## 問題描述

**原問題：** 場地名稱和提示文字顯示在左上角，會遮擋分數顯示區域，導致無法正確選取 ROI。

**影響範圍：**
- `roi_config_generator.py` - 提示文字在左上角 (10, 30)
- `roi_config_generator_v2.py` - 場地名稱在左上角 (10, 10)
- `preview_roi.py` - 座標資訊在左上角 (10, 30)
- `batch_assign_venues.py` - 場地名稱在左上角 (10, 10)

## 解決方案

**修改策略：** 將所有文字顯示移至右上角，避免遮擋左上角的分數區域。

### 修改詳情

#### 1. `roi_config_generator.py`

**修改位置：** 第 113-119 行

**修改前：**
```python
# 顯示當前步驟提示（左上角）
cv2.putText(display, prompt, (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
```

**修改後：**
```python
# 顯示當前步驟提示（右上角）
text_size = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
text_x = frame_width - text_size[0] - 10
text_y = 30

# 繪製背景（黑色半透明）
overlay = display.copy()
cv2.rectangle(overlay, (text_x - 5, text_y - 25),
             (frame_width - 5, text_y + 5), (0, 0, 0), -1)
cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

# 繪製文字
text_color = (0, 255, 0) if self.rois['team2'] else (255, 255, 255)
cv2.putText(display, prompt, (text_x, text_y),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
```

**特點：**
- ✅ 文字靠右對齊
- ✅ 黑色半透明背景增加可讀性
- ✅ 自動計算文字寬度

---

#### 2. `roi_config_generator_v2.py`

**修改位置：** 第 250-254 行

**修改前：**
```python
# 顯示場地名稱（左上角）
if self.current_venue:
    cv2.rectangle(display, (10, 10), (300, 50), (0, 0, 0), -1)
    cv2.putText(display, f"Venue: {self.current_venue}", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
```

**修改後：**
```python
# 顯示場地名稱（右上角）
if self.current_venue:
    venue_text = f"Venue: {self.current_venue}"
    text_size = cv2.getTextSize(venue_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = self.frame_width - text_size[0] - 20
    text_y = 35

    # 繪製背景（黑色）
    cv2.rectangle(display, (text_x - 10, 10), (self.frame_width - 10, 50), (0, 0, 0), -1)
    # 繪製文字
    cv2.putText(display, venue_text, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
```

**特點：**
- ✅ 場地名稱靠右顯示
- ✅ 黑色背景提高對比度
- ✅ 底部提示保持在螢幕下方（不會遮擋分數區）

---

#### 3. `preview_roi.py`

**修改位置：** 第 92-104 行

**修改前：**
```python
# 在畫面上顯示座標資訊（左上角）
info_text = [...]
y_offset = 30
for i, text in enumerate(info_text):
    cv2.putText(frame, text, (10, y_offset + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
```

**修改後：**
```python
# 在畫面上顯示座標資訊（右上角）
info_text = [...]
frame_height, frame_width = frame.shape[:2]

# 找出最長的文字寬度
max_width = 0
for text in info_text:
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    max_width = max(max_width, text_size[0])

# 繪製黑色背景（半透明）
overlay = frame.copy()
cv2.rectangle(overlay, (frame_width - max_width - 20, 10),
             (frame_width - 10, 120), (0, 0, 0), -1)
cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

# 繪製文字（右上角）
y_offset = 30
for i, text in enumerate(info_text):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = frame_width - text_size[0] - 15
    text_y = y_offset + i * 30
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
```

**特點：**
- ✅ 多行文字靠右對齊
- ✅ 根據最長文字計算背景寬度
- ✅ 半透明黑色背景

---

#### 4. `batch_assign_venues.py`

**修改位置：** 第 66-70 行

**修改前：**
```python
# 顯示場地名稱（左上角）
venue_name = venue_config.get('venue_name', 'Unknown')
cv2.rectangle(frame, (10, 10), (300, 50), (0, 0, 0), -1)
cv2.putText(frame, f"Venue: {venue_name}", (20, 35),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
```

**修改後：**
```python
# 顯示場地名稱（右上角）
venue_name = venue_config.get('venue_name', 'Unknown')
venue_text = f"Venue: {venue_name}"

frame_height, frame_width = frame.shape[:2]
text_size = cv2.getTextSize(venue_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
text_x = frame_width - text_size[0] - 20
text_y = 35

# 繪製背景（黑色）
cv2.rectangle(frame, (text_x - 10, 10), (frame_width - 10, 50), (0, 0, 0), -1)
# 繪製文字
cv2.putText(frame, venue_text, (text_x, text_y),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
```

---

## 測試驗證

### 測試腳本

新增測試腳本：`test/test_roi_display_position.py`

**測試項目：**
1. 場地名稱顯示位置
2. 提示文字顯示位置
3. 座標資訊顯示位置

**測試結果：**
```
測試 1: 場地名稱顯示位置
  文字位置: x=1039, y=35
  文字寬度: 221 pixels
  文字高度: 18 pixels

測試 2: 提示文字顯示位置
  文字位置: x=920, y=30
  文字寬度: 350 pixels
  文字高度: 16 pixels

測試 3: 座標資訊顯示位置
  最大文字寬度: 444 pixels

驗證結果:
  [OK] 場地名稱在右側
  [OK] 提示文字在右側
  [OK] 座標資訊在右側
  [OK] 文字不會遮擋分數區域
```

**測試圖片位置：** `test_output/roi_display_position/`
- `test_venue_name.jpg` - 場地名稱顯示測試
- `test_prompt.jpg` - 提示文字顯示測試
- `test_coordinates.jpg` - 座標資訊顯示測試

---

## 視覺效果對比

### 修改前（左上角）
```
┌─────────────────────────────────────┐
│ 場地名稱 ⬅️ 遮擋分數區！           │
│ [分數區域] ⬅️ 被文字遮擋           │
│                                     │
│                                     │
│                                     │
│                                     │
└─────────────────────────────────────┘
```

### 修改後（右上角）
```
┌─────────────────────────────────────┐
│ [分數區域] ✅             場地名稱 │
│                                     │
│                                     │
│                                     │
│                                     │
│                                     │
└─────────────────────────────────────┘
```

---

## 技術細節

### 1. 文字寬度自動計算
```python
text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
text_x = frame_width - text_size[0] - margin
```

### 2. 半透明背景
```python
overlay = frame.copy()
cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
```

### 3. 多行文字對齊
```python
# 找出最長文字
max_width = max(cv2.getTextSize(text, ...)[0][0] for text in texts)

# 逐行繪製，靠右對齊
for i, text in enumerate(texts):
    text_size = cv2.getTextSize(text, ...)[0]
    text_x = frame_width - text_size[0] - margin
    text_y = y_offset + i * line_height
    cv2.putText(frame, text, (text_x, text_y), ...)
```

---

## 影響範圍

### 修改檔案（4 個）
- ✅ `video_processing/roi_config_generator.py`
- ✅ `video_processing/roi_config_generator_v2.py`
- ✅ `video_processing/preview_roi.py`
- ✅ `video_processing/batch_assign_venues.py`

### 新增檔案（2 個）
- ✅ `test/test_roi_display_position.py` - 測試腳本
- ✅ `DISPLAY_POSITION_FIX.md` - 本文件

### 測試輸出
- ✅ `test_output/roi_display_position/*.jpg` - 測試圖片

---

## 向後相容性

✅ **完全向後相容** - 僅修改顯示位置，不影響任何功能邏輯

---

## 使用建議

1. **現有配置檔案** - 無需重新生成，可直接使用
2. **新配置生成** - 使用修正後的工具，不會遮擋分數區
3. **預覽檢查** - 使用 `preview_roi.py` 確認 ROI 位置

---

## 實作日期

2026-02-03

## 問題回報者

用戶反饋：「請幫我將 roi_config_generator 顯示名字的部分改到右上角，會擋住分數區無法選取」

## 修正完成

✅ 所有文字顯示已移至右上角，不會再遮擋分數區域
✅ 測試全部通過
✅ 視覺效果已驗證
