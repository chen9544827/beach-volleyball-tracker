# 修復完成總結：跳發邏輯與輸入驗證

## 修復日期
2026-02-02

## 修復內容

### ✅ 任務 1：修復跳發邏輯錯誤（已完成）

**檔案：** `core/jump_serve_detector.py` (lines 306-330)

**問題描述：**
- 原始演算法在最長連續序列位於結尾時無法正確識別
- `jump_start` 只在中間序列更新，導致結尾序列被忽略

**修復內容：**
1. 新增 `best_sequence_start_idx` 變數追蹤最長序列起始位置
2. 在迴圈中找到更長序列時更新起始索引
3. **關鍵修復**：迴圈結束後檢查最後序列是否為最長
4. 使用追蹤的索引正確設定 `jump_start`

**修復前後對比：**

```python
# 修復前（錯誤）
for i in range(1, len(jump_frames)):
    if jump_frames[i] - jump_frames[i-1] <= 2:
        current_consecutive += 1
        if current_consecutive > max_consecutive:
            max_consecutive = current_consecutive
    else:
        if current_consecutive == max_consecutive:  # ❌ 邏輯錯誤
            jump_start = jump_frames[i - current_consecutive + 1]
        current_consecutive = 1
# 缺少最後序列檢查 ❌

# 修復後（正確）
best_sequence_start_idx = 0  # ✅ 新增追蹤變數
for i in range(1, len(jump_frames)):
    if jump_frames[i] - jump_frames[i-1] <= 2:
        current_consecutive += 1
        if current_consecutive > max_consecutive:
            max_consecutive = current_consecutive
            best_sequence_start_idx = i - current_consecutive + 1  # ✅ 更新索引
    else:
        current_consecutive = 1

# ✅ 關鍵修復：檢查最後序列
if current_consecutive >= max_consecutive:
    max_consecutive = current_consecutive
    best_sequence_start_idx = len(jump_frames) - current_consecutive

if jump_frames:
    jump_start = jump_frames[best_sequence_start_idx]  # ✅ 使用正確索引
```

**測試案例：**
- [10,11,12, 30,31,32,33,34] → jump_start=30 ✅（修復前：11 ❌）
- [5,6, 20,21,22,23, 40,41] → jump_start=20 ✅
- [10,11,12,13,14] → jump_start=10 ✅
- [] → jump_start=None ✅

---

### ✅ 任務 2：建立驗證基礎設施（已完成）

#### 新檔案 1: `core/error_messages.py` (~170 行)

**功能：**
- 統一的繁體中文錯誤訊息範本
- 三類訊息：ERROR_MESSAGES、WARNING_MESSAGES、INFO_MESSAGES
- 格式化函數：`format_error()`, `format_warning()`, `format_info()`
- 自訂異常類別：`ValidationError`

**範例訊息：**
```python
ERROR_MESSAGES = {
    'file_not_found': '找不到檔案: {path}',
    'missing_frames': "JSON 中沒有 'frames' 欄位或 frames 為空",
    'insufficient_ankle_data': '腳踝資料不足 ({count} 幀)，需要至少 {required} 幀',
}
```

#### 新檔案 2: `core/data_validator.py` (~300 行)

**功能：**
- `DataValidator` 類別：混合驗證策略
  - **關鍵欄位嚴格驗證**：frames, frame_id, metadata.video_path, metadata.fps
  - **次要欄位寬鬆處理**：ball_detections, player_detections（允許空值）
  - 自動收集警告訊息

- `safe_load_json()` 函數：
  - 處理 FileNotFoundError, JSONDecodeError, UnicodeDecodeError
  - 返回 `(data, error_message)` 元組
  - 所有錯誤訊息使用繁體中文

- 輔助函數：
  - `get_keypoint()`: 安全取得姿態關鍵點
  - `validate_center_point()`: 驗證座標有效性
  - `validate_frame_data()`: 驗證單一幀資料

**驗證策略範例：**
```python
# 關鍵欄位缺失 → 驗證失敗
is_valid, errors = validator.validate_tracking_json(data_without_frames)
# is_valid = False, errors = ["JSON 中沒有 'frames' 欄位或 frames 為空"]

# 次要欄位缺失 → 通過但警告
is_valid, errors = validator.validate_tracking_json(data_without_ball)
# is_valid = True, errors = [], warnings = ["警告: 幀 0 的球偵測資料不完整"]
```

---

### ✅ 任務 3：整合驗證到 batch_test_serve.py（已完成）

**修改位置：**

1. **Lines 23-31** - 匯入驗證模組：
```python
from core.data_validator import DataValidator, safe_load_json, validate_center_point
from core.error_messages import ValidationError, format_error
import logging
```

2. **Lines 33-66** - 替換 `load_tracking_data()`：
```python
def load_tracking_data(json_path: str) -> dict:
    # 安全載入 JSON
    data, error = safe_load_json(json_path)
    if error:
        raise ValidationError(f"載入 JSON 失敗: {error}")

    # 驗證資料結構
    validator = DataValidator(verbose=False)
    is_valid, errors = validator.validate_tracking_json(data)

    if not is_valid:
        error_msg = f"資料驗證失敗:\n  - " + "\n  - ".join(errors)
        raise ValidationError(error_msg)

    return data
```

3. **Lines 68-89** - 替換 `load_court_config()`：
```python
def load_court_config(config_path: str) -> dict:
    if not config_path or not os.path.exists(config_path):
        logging.warning(f"找不到場地設定檔: {config_path}")
        return None

    data, error = safe_load_json(config_path)
    if error:
        logging.error(f"載入場地設定失敗: {error}")
        return None

    # 驗證場地設定結構
    validator = DataValidator(verbose=False)
    is_valid, errors = validator.validate_court_config(data)
    if not is_valid:
        logging.error(f"場地設定驗證失敗: {', '.join(errors)}")
        return None

    return data
```

4. **Line 133** - 增加 frame_data 驗證：
```python
players = frame_data.get('player_detections', []) if frame_data else []
```

5. **Lines 137-148** - 使用 `validate_center_point()`：
```python
server_center = validate_center_point(server_info.get('center_point'))
# ...
center = validate_center_point(player.get('center_point'))
if not center:
    continue
```

6. **Lines 279-296** - 增加異常處理：
```python
try:
    data = load_tracking_data(json_path)
    frames_data = data.get('frames', [])
    # ...
except ValidationError as e:
    result['status'] = 'error'
    result['error'] = str(e)
    return result
except Exception as e:
    result['status'] = 'error'
    result['error'] = f"載入資料時發生未預期的錯誤: {str(e)}"
    return result
```

---

### ✅ 任務 4：整合驗證到支援檔案（已完成）

#### 修改 1: `batch_tracking.py`

**Lines 22-52** - 替換 `load_court_config()`：
```python
from core.data_validator import safe_load_json, DataValidator
import logging

def load_court_config(config_path: str) -> dict:
    if not config_path or not os.path.exists(config_path):
        logging.warning(f"找不到場地設定檔: {config_path}")
        return None

    data, error = safe_load_json(config_path)
    if error:
        logging.error(f"載入場地設定失敗: {error}")
        return None

    # 驗證場地設定結構
    validator = DataValidator(verbose=False)
    is_valid, errors = validator.validate_court_config(data)
    if not is_valid:
        logging.error(f"場地設定驗證失敗: {', '.join(errors)}")
        return None

    return data
```

#### 修改 2: `core/jump_serve_detector.py`

**Lines 187-191** - 驗證 frame_id 存在：
```python
# 建立 frame_id 到資料的映射（驗證 frame_id 存在）
frame_map = {f['frame_id']: f for f in frames_data if 'frame_id' in f}

if not frame_map:
    result['error'] = '無有效幀資料（所有幀都缺少 frame_id）'
    return result
```

**Lines 227-233** - 空列表檢查：
```python
# 驗證腳踝資料是否充足
if not ankle_trajectory:
    result['error'] = '無腳踝軌跡資料'
    return result

if len(ankle_trajectory) < 5:
    result['error'] = f'腳踝資料不足（{len(ankle_trajectory)} 幀），需要至少 5 幀'
    return result
```

**Lines 250-256** - 在 min() 前檢查：
```python
else:
    # 使用起始幀的腳踝位置作為基準線
    if not ankle_y_values:
        result['error'] = '無有效腳踝 Y 座標資料'
        return result

    baseline_frames = min(5, len(ankle_y_values) // 3)
    baseline_ankle_y = np.mean(ankle_y_values[:baseline_frames])
```

**Lines 258-265** - 在 min() 前再次檢查：
```python
# 找到最高點（Y 最小）
if not ankle_y_values:
    result['error'] = '無有效腳踝 Y 座標資料'
    return result

min_ankle_y = min(ankle_y_values)
min_ankle_idx = ankle_y_values.index(min_ankle_y)
```

---

## 測試檔案

### 新檔案 1: `test/test_jump_serve_logic.py` (~180 行)

**測試案例：**
1. ✅ `test_longest_sequence_at_end()` - 最長序列在結尾（原始 bug）
2. ✅ `test_longest_sequence_in_middle()` - 最長序列在中間
3. ✅ `test_all_consecutive()` - 全部連續
4. ✅ `test_empty_list()` - 空列表處理
5. ✅ `test_single_frame()` - 單一幀
6. ✅ `test_with_gaps()` - 包含允許間隔
7. ✅ `test_multiple_equal_sequences()` - 多個相同長度序列

**執行方式：**
```bash
python test/test_jump_serve_logic.py
```

### 新檔案 2: `test/test_data_validator.py` (~250 行)

**測試案例：**
1. ✅ `test_valid_json_passes()` - 有效 JSON 通過驗證
2. ✅ `test_missing_frames_fails()` - 缺少 frames 時失敗
3. ✅ `test_missing_metadata_fails()` - 缺少 metadata 時失敗
4. ✅ `test_missing_ball_detections_warns()` - 缺少 ball_detections 時警告
5. ✅ `test_corrupt_json_fails()` - 損壞 JSON 優雅失敗
6. ✅ `test_file_not_found()` - 檔案不存在時優雅失敗
7. ✅ `test_get_keypoint_valid()` - 關鍵點提取驗證
8. ✅ `test_validate_center_point()` - 座標驗證
9. ✅ `test_invalid_fps()` - 無效 fps 值處理
10. ✅ `test_court_config_validation()` - 場地設定驗證

**執行方式：**
```bash
python test/test_data_validator.py
```

---

## 向後相容性

### ✅ 完全相容
- 函數簽名未改變
- 輸出格式保持一致
- 現有 JSON 檔案無需修改
- 正常資料的處理流程不受影響

### ⚠️ 行為變更（改進）
1. **更早失敗**：損壞的 JSON 或缺少關鍵欄位時會立即報錯
2. **更好的錯誤訊息**：使用繁體中文，訊息更明確
3. **容錯性提升**：遮擋導致的部分資料缺失不再導致崩潰

---

## 修復的問題列表

### 嚴重級別（已修復）
- ✅ 跳發邏輯錯誤（最長序列在結尾時失敗）
- ✅ JSON 載入無錯誤處理
- ✅ 幀資料存取無驗證
- ✅ 空列表操作（min/max 崩潰）
- ✅ 幀映射無檢查

### 高優先級（已修復）
- ✅ 配置結構無驗證
- ✅ 字典鏈式存取無 None 檢查
- ✅ 錯誤訊息語言不一致（現全部繁體中文）

---

## 使用範例

### 驗證追蹤資料
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
    print("驗證失敗:")
    for err in errors:
        print(f"  - {err}")
else:
    print("驗證通過！")
    if validator.warnings:
        print(f"警告數量: {len(validator.warnings)}")
```

### 安全取得關鍵點
```python
from core.data_validator import get_keypoint, validate_center_point

# 取得腳踝關鍵點
pose_keypoints = player.get('pose_keypoints')
left_ankle = get_keypoint(pose_keypoints, 15, confidence_threshold=0.3)

if left_ankle:
    x, y = left_ankle
    print(f"左腳踝位置: ({x}, {y})")
else:
    print("無法取得左腳踝資料（可能因遮擋或信心度過低）")

# 驗證中心點
center = validate_center_point(player.get('center_point'))
if center:
    x, y = center
    # 使用座標...
```

---

## 回歸測試步驟

執行以下測試確保無破壞性變更：

### 1. 單元測試
```bash
# 測試跳發邏輯修復
python test/test_jump_serve_logic.py

# 測試資料驗證器
python test/test_data_validator.py
```

### 2. 單一影片測試
```bash
python test_server_identification.py
```

### 3. 發球診斷測試
```bash
python diagnose_serve.py --input test_output/segment_029_all_frames_data_with_pose.json
```

### 4. 批次測試（完整流程）
```bash
python batch_test_serve.py \
    --video-dir input_video/analyze_serve \
    --json-dir test_output \
    --output regression_test_output \
    --court-config court_config.json
```

### 驗證標準
- ✅ 所有先前成功的偵測仍正常運作
- ✅ 錯誤訊息改為繁體中文
- ✅ 邊界案例不再崩潰（優雅失敗）
- ✅ 輸出圖片與修復前一致
- ✅ 跳發偵測準確性提升

---

## 檔案清單

### 新建檔案（4 個）
1. ✅ `core/error_messages.py` (~170 行)
2. ✅ `core/data_validator.py` (~300 行)
3. ✅ `test/test_jump_serve_logic.py` (~180 行)
4. ✅ `test/test_data_validator.py` (~250 行)

### 修改檔案（3 個）
1. ✅ `core/jump_serve_detector.py` - 修復邏輯 + 增加驗證
2. ✅ `batch_test_serve.py` - 整合驗證系統
3. ✅ `batch_tracking.py` - 整合驗證系統

---

## 總結

### 主要成果
1. **跳發偵測準確性提升** - 修復了最長序列在結尾時的 bug
2. **系統穩健性大幅提升** - 增加完整的輸入驗證和錯誤處理
3. **使用者體驗改善** - 所有錯誤訊息改為清晰的繁體中文
4. **容錯能力增強** - 支援遮擋情況下的部分資料缺失
5. **完全向後相容** - 不需要修改現有資料或腳本

### 預期效果
- ❌ 修復前：最長序列在結尾時錯誤識別，無驗證導致神秘崩潰
- ✅ 修復後：正確識別所有序列，優雅處理錯誤並提供明確訊息

### 建議後續步驟
1. 執行完整的回歸測試確保無破壞性變更
2. 更新 CLAUDE.md 記錄新的驗證系統
3. 考慮為其他腳本添加類似的驗證機制
4. 根據實際使用情況調整警告閾值
