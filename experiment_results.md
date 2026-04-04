# 實驗結果記錄

本文件記錄每次 pilot 測試的片段清單與分析結果，供跨 session 追蹤用。

---

## Pilot Test v1（修正前）

**執行時間：** 2026-04-01 09:50  
**腳本：** `run_pilot_test.py`  
**已知問題：** Chetumal 去重 bug（segment_002/005 各出現兩次）+ segment_002 VballNet 僅跑 9 幀（pipeline 重建前）  
**結果：** 成功 11/20（55%）、F 級 6/20（30%）

---

## Pilot Test v2（修正後）✅ 最新

**執行時間：** 2026-04-01（下午）  
**腳本：** `run_pilot_test.py`  
**修正項目：**
1. `collect_candidates()` 加入 `seen_names` 跨 match 目錄去重
2. Chetumal segment_002 VballNet CSV 重建（`output/vball_chetumal_fix/`），球偵測 1.2% → 79.4%

### 總覽

| 指標 | 數值 |
|------|------|
| 總片段數 | 20 |
| 成功偵測發球 | **12 (60%)** |
| 未偵測發球 | 7 |
| F 級比例（球偵測率 < 30%） | **3/20 (15%)** |
| 跳發 | 1 |
| 站立發球 | 11 |

### 各場地結果

| 場地 | 成功/總計 | 成功率 | no_serve | error/insuff |
|------|-----------|--------|----------|--------------|
| Chetumal_WT18_C1 | 5/5 | **100%** | 0 | 0 |
| Edmonton_WT19_C4 | 1/5 | **20%** | 4 | 0 |
| Gstaad_WT19_C1 | 4/5 | **80%** | 1 | 0 |
| Jinjiang_WT19_C1 | 2/5 | **40%** | 2 | 1 |

### 測試片段清單（共 20 個）

#### Chetumal_WT18_C1（5/5 成功）

| 片段 | 狀態 | 發球類型 | hit_frame | 信心度 | 球偵測率 | 說明 |
|------|------|----------|-----------|--------|----------|------|
| segment_005_Team1 | ✅ success | standing | 431 | 0.50 | 28.1% | 遠端發球，接球區 2 |
| segment_002_Team1 | ✅ success | standing | 179 | 0.50 | 79.4% | 近端發球，ace |
| segment_071_Team2 | ✅ success | standing | 400 | 0.54 | 63.0% | 近端發球，接球區 5 |
| segment_013_Team1 | ✅ success | standing | 193 | 0.88 | 71.5% | 近端發球，高信心度 |
| segment_004_Team1 | ✅ success | standing | 82 | 0.80 | 82.2% | 近端發球，ace |

#### Edmonton_WT19_C4（1/5 成功）

| 片段 | 狀態 | 發球類型 | hit_frame | 信心度 | 球偵測率 | 說明 |
|------|------|----------|-----------|--------|----------|------|
| segment_003_Team2 | ✅ success | standing | 448 | 0.20 | 42.4% | 遠端發球，低信心度 |
| segment_001_Team1 | ❌ no_serve | — | — | — | 61.1% | 球偵測正常但未找到發球 |
| segment_018_Team2 | ❌ no_serve | — | — | — | 32.0% | 球偵測正常但未找到發球 |
| segment_050_Team1 | ❌ no_serve | — | — | — | 55.0% | 球偵測正常但未找到發球 |
| segment_004_Team2 | ❌ no_serve | — | — | — | 34.9% | 球偵測正常但未找到發球 |

> ⚠️ Edmonton 4 個 no_serve 片段球偵測率均在 32~61%，疑為發球偵測演算法假陰性，待診斷

#### Gstaad_WT19_C1（4/5 成功）

| 片段 | 狀態 | 發球類型 | hit_frame | 信心度 | 球偵測率 | 說明 |
|------|------|----------|-----------|--------|----------|------|
| segment_010_Team1 | ✅ success | standing | 362 | 0.50 | 22.9% | 遠端發球 |
| segment_004_Team1 | ✅ success | jump | 53 | 0.50 | 33.9% | 跳發，近端 |
| segment_006_Team2 | ✅ success | standing | 157 | 0.50 | 68.7% | 遠端發球 |
| segment_019_Team1 | ✅ success | standing | 418 | 0.50 | 49.4% | 遠端發球，ace |
| segment_053_End_Of_Video | ❌ no_serve | — | — | — | 33.6% | 影片末段，可能無完整發球 |

#### Jinjiang_WT19_C1（2/5 成功）

| 片段 | 狀態 | 發球類型 | hit_frame | 信心度 | 球偵測率 | 說明 |
|------|------|----------|-----------|--------|----------|------|
| segment_005_Team1 | ✅ success | standing | 140 | 0.50 | 42.1% | 近端發球，接球區 5 |
| segment_063_Team2 | ✅ success | standing | 201 | 0.50 | 75.2% | 遠端發球 |
| segment_002_Team1 | ❌ insuff | — | — | — | 0.6% | VballNet 失球（僅 3 幀），根本性限制 |
| segment_049_Team2 | ❌ no_serve | — | — | — | 61.9% | 球偵測正常但未找到發球 |
| segment_023_Team1 | ❌ no_serve | — | — | — | 77.2% | 球偵測正常但未找到發球 |

### 輸出路徑
- CSV：`output/pilot_test_20seg/pilot_results.csv`
- 摘要 JSON：`output/pilot_test_20seg/pilot_summary.json`
- 發球圖片：`output/pilot_test_20seg/analysis/<venue>/serve_images/`
- 各場地分析：`output/pilot_test_20seg/analysis/<venue>/`

---

## Pilot Test v3：V1a vs V1b 對比（2026-04-04）

**執行腳本：** `run_pilot_v1a_compare.py`  
**目的：** 用 fine-tuned VballNetV1a 對相同 20 個片段重新分析，與 V1b 結果對比

### 總覽

| 指標 | V1b（原始） | V1a（fine-tuned） | 差異 |
|------|------------|-------------------|------|
| 成功率 | 12/20 (60%) | **15/20 (75%)** | **+15pp** |
| 改善 | — | 3 個片段 | 無退步 |
| 平均球偵測率 | ~51% | ~37% | **-14pp**（精準度提高，假陽性減少）|

### 各場地對比

| 場地 | V1b 成功 | V1a 成功 | 改善 | V1b 球率 | V1a 球率 |
|------|---------|---------|------|---------|---------|
| Chetumal_WT18_C1 | 5/5 | 5/5 | 0 | 64.8% | 41.4% |
| Edmonton_WT19_C4 | 1/5 | **3/5** | **+2** | 45.1% | 34.9% |
| Gstaad_WT19_C1 | 4/5 | 4/5 | 0 | 41.7% | 31.5% |
| Jinjiang_WT19_C1 | 2/5 | **3/5** | **+1** | 51.4% | 37.4% |

### 改善的片段

| 片段 | V1b | V1a | 備註 |
|------|-----|-----|------|
| Edmonton segment_018_Team2 | no_serve (32.0%, 62幀) | **success** (20.7%, 56幀) | 球率更低但仍找到發球 |
| Edmonton segment_050_Team1 | no_serve (55.0%, 148幀) | **success** (34.2%, 53幀) | 同上 |
| Jinjiang segment_023_Team1 | no_serve (77.2%, 31幀) | **success** (33.8%, 31幀) | 球率大降仍找到發球 |

> V1a 球偵測率較低但**精確度高**（假陽性少），使狀態機拋球軌跡確認更容易觸發。

### 仍失敗的片段（V1a）

| 片段 | V1a 狀態 | V1a 球率 | 最長連續 | 備註 |
|------|---------|---------|---------|------|
| Edmonton segment_001_Team1 | no_serve | 61.1% | 261 幀 | V1b/V1a 完全相同，純算法問題 |
| Edmonton segment_004_Team2 | no_serve | 16.3% | 36 幀 | V1a 球率大降，需更多訓練資料 |
| Gstaad segment_053_End_Of_Video | insuff | 5.5% | 5 幀 | V1a 對此片段球偵測極差 |
| Jinjiang segment_002_Team1 | no_serve | 33.7% | 72 幀 | V1b insufficient→V1a 找到球但無發球 |
| Jinjiang segment_049_Team2 | no_serve | 21.0% | 21 幀 | 球資料不足 |

### 異常現象

| 片段 | 問題 | 說明 |
|------|------|------|
| Gstaad segment_006_Team2 | 發球類型不同 | V1b: standing@157, V1a: jump@181 — 需視覺確認哪個正確 |
| Edmonton segment_001_Team1 | V1a/V1b 結果完全相同 | 61.1%/261幀結果一致，疑似 Edmonton_001 為訓練資料（V1a過擬合） |

### 輸出路徑
- 對比 CSV：`output/pilot_test_20seg_v1a/v1a_v1b_comparison.csv`
- 對比摘要：`output/pilot_test_20seg_v1a/v1a_v1b_summary.json`
- V1a 結果：`output/pilot_test_20seg_v1a/v1a_results.csv`

---

## 待調查問題與改善建議

### 問題清單（更新至 V1a 對比後）

| 優先度 | 問題 | 影響片段 | 狀態 |
|--------|------|----------|------|
| 🔴 緊急 | Edmonton segment_001：261 幀連續球資料，V1a/V1b 均找不到發球 | Edmonton: 001 | 待 debug 狀態機 |
| 🟠 高 | V1a 對 Edmonton_004 球偵測率驟降（16.3%）→ 泛化不足 | Edmonton: 004 | 需更多 Edmonton 訓練資料 |
| 🟠 高 | V1a 對 Gstaad_053 球偵測率驟降（5.5%）→ 泛化不足 | Gstaad: 053 | 需 Gstaad 訓練資料 |
| 🟡 中 | Gstaad_006 發球類型在 V1a/V1b 不一致（standing vs jump） | Gstaad: 006 | 需視覺確認 |
| 🟡 中 | Jinjiang_002 V1a 找到球（33.7%）但仍無發球 | Jinjiang: 002 | 算法問題 |
| 🟡 中 | V1a 整體球偵測率比 V1b 低 14pp（37% vs 51%）| 全部片段 | V1a 可能過於保守 |
| 🟢 低 | `run_pilot_test.py` 仍用 V1b（與 feedback 規定矛盾） | — | 需更新 VBALL_MODEL |

### 改善建議

#### 短期（算法層面）
1. **Debug Edmonton_001 狀態機**：用 `diagnose_serve.py` 逐幀追蹤，確認是哪個 Method 卡住
   - 已知：261 幀連續球、V1a/V1b 完全相同 → 純算法問題，與球模型無關
2. **修正 `run_pilot_test.py` 模型路徑**：改用 `VballNetV1a_finetuned_seq9_grayscale.onnx`

#### 中期（資料層面）
3. **擴充 V1a 微調資料集**：目前只有 3 個片段（~2,400 幀），需加入：
   - Edmonton 額外片段（segment_004_Team2 泛化差，需更多 Edmonton 樣本）
   - Gstaad 片段（完全沒有 Gstaad 訓練資料，segment_053 驟降至 5.5%）
4. **CVAT 標記優先序**：Gstaad 2 個片段 + Edmonton 2 個片段 → 重新微調 V1a

#### 長期（架構層面）
5. **信心度量化**：目前 confidence=0.5 是固定值，需根據球偵測的軌跡一致性計算真實信心度
6. **multi-serve 支援**：目前 `first_only=True` 只找片段中第一個發球，若開頭球資料差會遺漏後半段發球
