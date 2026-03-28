# Beach Volleyball Tracker - 開發進度記錄

---

## 2026-03-11（本次 session）

### VballNet Fine-tuning 完整流程

#### 資料集準備
- **CVAT 標記資料轉換**：使用 `tools/convert_cvat_to_tracknet.py` 將 3 個 FIVB 片段的 CVAT XML 標記轉換為 TrackNet CSV + 512×288 PNG 幀
  - `segment_001_Team1`：1092 幀，680 visible (62.3%)
  - `segment_002_Team1`：756 幀，481 visible (63.6%)
  - `segment_004_Team2`：600 幀，439 visible (73.2%)
- **作者資料集整合**：使用 `tools/prepare_vballnet_dataset.py` 合併作者原始資料集（70 clips）與我們的 CVAT 資料（3 clips）
  - Train：73 clips，25,798 幀（91.7% 有球），位於 `vball-net/data/frames/train/`
  - Test：19 clips，5,992 幀（91.8% 有球），位於 `vball-net/data/frames/test/`

#### 模型訓練
- **架構**：VballNetV1a（seq=9, grayscale, 243K params）
- **框架**：PyTorch（TensorFlow 在 Windows 不支援 GPU，改用 PyTorch CUDA 12.6）
- **訓練腳本**：`train_vballnet_pt.py`
  - Dataset：`VballCsvPngDataset`，多執行緒預載入（8 threads，17s 載入 3.6GB）
  - Loss：WeightedBCE
  - Optimizer：Adam (lr=1e-3) + ReduceLROnPlateau
  - GPU：RTX 4080 Laptop，每 epoch ~33 分鐘
- **結果**：訓練至 epoch 29，val_loss=0.000229，儲存於
  - `vball-net-pytorch/outputs/finetune_VballNetV1a_seq9_grayscale_20260311_002656/checkpoints/best.pth`

#### ONNX 匯出
- **腳本**：`export_vballnet_pt.py`（PyTorch → ONNX，`dynamo=False`）
- **輸出**：`fast-volleyball-tracking-inference/models/VballNetV1a_finetuned_seq9_grayscale.onnx`（985KB）

#### 模型比較（vs Ground Truth）

| 片段 | 模型 | F1 Score | Precision | Recall |
|------|------|----------|-----------|--------|
| seg001 | Fine-tuned V1a | **96.5%** | 96.4% | 96.6% |
| seg001 | Original V1b | 36.8% | 36.1% | 37.5% |
| seg002 | Fine-tuned V1a | **93.1%** | 93.7% | 92.5% |
| seg002 | Original V1b | 50.2% | 35.9% | 83.2% |
| seg004 | Fine-tuned V1a | **94.0%** | 97.5% | 90.8% |
| seg004 | Original V1b | 53.8% | 40.5% | 80.1% |

**重要發現**：原始 V1b 雖偵測率數字高（85-87%），但含大量 False Positive（假球）；Fine-tuned V1a 偵測精度大幅提升（平均 F1: 94.5% vs 46.9%）。

#### 影片輸出
- Fine-tuned 比較影片：`output/video_finetuned/<seg>/predict.mp4`
- Original 比較影片：`output/video_original/<seg>/predict.mp4`
- Edmonton 實際比賽片段（5 個）：`output/edmonton_inference/<seg>/predict.mp4`

| Edmonton 片段 | 幀數 | 偵測率 |
|--------------|------|--------|
| segment_001_Team1 | 1092 | 61.7% |
| segment_015_Team1 | 744 | 23.5% |
| segment_030_Team2 | 684 | 45.6% |
| segment_050_Team1 | 456 | 34.4% |
| segment_070_Team1 | 672 | 52.2% |

---

## 2026-03-15（本次 session）

### 多架構訓練支援

#### 腳本修改
- **`train_vballnet_pt.py`**：新增 `--model_name` 參數，支援 VballNetV1a / V1b / V1c 三種架構
  - V1b（1.03M params）：ASPP + deep supervision + EnhancedMotionPrompt，forward 回傳單一 tensor
  - V1c（490K params）：GRU 時序建模，forward 回傳 `(output, hn)` tuple（訓練時自動解包）
  - 輸出目錄命名自動帶入 model_name
- **`export_vballnet_pt.py`**：新增 `--model_name` 參數，支援 V1a/V1b/V1c 匯出
  - V1c 匯出時自動加 wrapper 去除 hn 輸出，確保 ONNX 介面一致（單一 output）
  - 從 checkpoint 路徑自動推斷模型架構

#### 模型參數比較
| 架構 | Params | 特點 |
|------|--------|------|
| VballNetV1a | 243K | 簡單 UNet，已 fine-tuned |
| VballNetV1b | 1.03M | ASPP + deep supervision + motion prompt |
| VballNetV1c | 490K | GRU 時序建模 |

### Fine-tuned V1a 多場地 Inference

使用 `VballNetV1a_finetuned_seq9_grayscale.onnx` + `inference_onnx_twopass.py`（`far_crop_ratio=0.45`）
對 4 個場地各 2 個片段進行推理，輸出至 `output/venue_inference_finetuned/`

| 場地 | 片段 | 幀數 | 偵測率 |
|------|------|------|--------|
| Chetumal_WT18_C1 (match003) | segment_002_Team1 | 9* | 100.0%* |
| Chetumal_WT18_C1 (match005) | segment_005_Team1 | 828 | 30.3% |
| Edmonton_WT19_C4 | segment_001_Team1 | 1092 | **61.7%** |
| Edmonton_WT19_C4 | segment_003_Team2 | 984 | 43.2% |
| Gstaad_WT19_C1 | segment_004_Team1 | 1440 | 39.5% |
| Gstaad_WT19_C1 | segment_010_Team1 | 1968 | 31.1% |
| Jinjiang_WT19_C1 | segment_002_Team1 | 9* | 44.4%* |
| Jinjiang_WT19_C1 | segment_005_Team1 | 756 | 43.8% |

\* 9 幀異常（路徑或影片問題，待調查）

---

## 環境設定記錄

### GPU 訓練問題解決
- TF 2.20 在 Windows 原生不支援 GPU（TF 2.11+ 僅支援 Linux/WSL2）
- **解決方案**：改用 PyTorch（`tff_env` 內有 PyTorch 2.9.1+cu126）
- 需在 import torch 前加：`os.add_dll_directory(torch_lib_path)` 才能找到 CUDA DLL

### 新增腳本
| 腳本 | 說明 |
|------|------|
| `train_vballnet_pt.py` | PyTorch 訓練主程式（PNG+CSV 格式，preload 到 RAM） |
| `train_vballnet.py` | TF 訓練啟動器（從根目錄執行，自動設定路徑） |
| `export_vballnet_pt.py` | PyTorch checkpoint → ONNX 匯出 |

---

## 待辦事項

- [ ] 增加更多 FIVB 比賽片段的 CVAT 標記（目前只有 3 個）→ 提升泛化能力
- [ ] 嘗試 VballNetV1b 架構（1M params，與原始模型架構相同）
- [ ] 嘗試 VballNetV1c（加入 GRU 時序建模）
- [ ] 完整 50 epochs 訓練（目前 29 epochs 後停止）
- [ ] 大規模批次處理（多 GPU，~2000 部影片）
- [ ] segment_015 偵測率只有 23.5%，需調查原因
