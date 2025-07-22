# 沙灘排球影片自動化分析專案 (Automated Beach Volleyball Video Analysis)

本專案旨在提供一個自動化的解決方案，用於分析沙灘排球比賽影片。系統能夠自動偵測、追蹤場中的球員與排球，並透過一套專為排球運動設計的智慧邏輯，準確識別出「發球」事件。

專案採用了最新的物件偵測模型 (YOLOv8) 進行追蹤，並透過一個高度容錯的事件分析引擎，來應對真實比賽中如攝影機角度變化、球體動態模糊、目標短暫離開畫面等複雜情況。

## 核心功能 (Core Features)

- **自動化物件追蹤**：使用 YOLOv8-Pose 模型，精準偵測並追蹤每一幀畫面中的球員位置與姿態，並使用特製的 YOLOv8 模型追蹤排球。
- **智慧發球偵測**：內建一套最終定義版的**四段式智慧狀態機** (`搜尋` -> `確認` -> `等待頂點與下降` -> `等待擊球`)，能夠準確識別符合排球物理特性的拋球與擊球動作，並有效過濾如球僮傳球等干擾事件。
- **高效平行處理**：利用 Python 的平行處理能力，腳本能夠調用您電腦的多個 CPU 核心，同時分析多個影片片段，大幅縮短處理大量影片所需的時間。
- **一鍵式總結報告**：在所有影片分析完畢後，自動生成一個總結資料夾，其中包含：
    -   每個影片首次擊球事件**前後 3 幀的關鍵畫格圖片**。
    -   一個 `first_hit_frames.txt` 文字檔，清晰記錄每個影片首次擊球的幀數。
- **可調校的偵測參數**：所有核心偵測邏輯的閾值（如拋球速度、擊球速度、垂直率等）均可透過命令列參數進行微調，方便您根據不同影片來源進行最佳化。
- **完整的日誌記錄**：為每一個處理的影片生成詳細的 `processing.log` 檔案，完整記錄追蹤與分析階段的所有輸出訊息，方便除錯與驗證。

## 專案運作流程 (Workflow)

本專案的核心是一個名為 `run_analysis_all_in_one.py` 的萬能整合腳本，其內部運作流程如下：

1.  **影片搜尋**：腳本會自動在您指定的輸入資料夾中，尋找所有支援的影片檔案（`.mp4`, `.mov` 等）。
2.  **任務分配**：腳本會為每一個影片檔案建立一個獨立的分析任務，並將這些任務分配給一個由多個 CPU 核心組成的處理池。
3.  **單一影片處理流程** (由每個核心獨立執行)：
    a. **階段一：物件追蹤**：在背景呼叫 `video_processing/track_ball_and_player.py`，對影片進行逐幀分析，偵測球與球員，並將所有追蹤數據（包括姿態關鍵點）儲存為一個 `.json` 檔案。
    b. **階段二：事件分析**：讀取上一步生成的 `.json` 檔案，並使用內建的「四段式智慧狀態機」對球的完整運動軌跡進行分析，找出所有符合發球定義的事件。
    c. **階段三：視覺化影片生成**：將偵測到的發球事件，以醒目的方式標註在原始影片上，並生成一個新的 `_analysis.mp4` 影片檔案。
4.  **總結報告生成**：在所有影片的處理任務都完成後，主程序會收集所有成功的偵測結果，自動執行「第一步：清理舊的腳本（非常重要！）」中的總結報告生成流程。

## 如何使用 (How to Use)

#### 1. 環境設定 (Environment Setup)
請先確您已安裝所有必要的 Python 套件。在專案根目錄下打開終端機，執行：
```bash
pip install -r requirements.txt
```
腳本也會在首次執行時，自動檢查並提示安裝 `tqdm` (進度條函式庫)。

#### 2. 準備影片 (Prepare Videos)
將所有您想要分析的沙灘排球影片片段，放入一個資料夾中，例如 `input_videos`。

#### 3. 執行整合分析腳本 (Run the All-in-One Script)
打開終端機，移動到專案的根目錄，然後執行以下命令。腳本將會自動開始處理。

- **基本執行** (使用預設參數，自動偵測 CPU 核心數進行平行處理):
  ```bash
  python run_analysis_all_in_one.py --input_folder "path/to/your/input_videos"
  ```
- **強制覆蓋舊結果並儲存所有原始畫格**:
  ```bash
  python run_analysis_all_in_one.py --input_folder "path/to/your/input_videos" --overwrite --save_all_frames
  ```

#### 4. 查看結果 (Check the Results)
處理完成後，所有結果將會儲存在您指定的輸出資料夾中（預設為 `volleyball_analysis_results`）。其結構如下：

```
volleyball_analysis_results/
├── first_hit_summary/                <-- ✨ 全新的總結報告資料夾
│   ├── segment_001_frame_00137_HIT.jpg
│   ├── segment_001_frame_00136_PRE_1.jpg
│   ├── ...
│   └── first_hit_frames.txt
│
├── segment_001/                        <-- 每個影片的獨立結果資料夾
│   ├── analysis_output/
│   │   └── segment_001_analysis.mp4    <-- 標註了發球事件的影片
│   ├── tracking_output/
│   │   └── segment_001/
│   │       └── ..._all_frames_data_with_pose.json  <-- 原始追蹤數據
│   └── processing.log                  <-- 完整的處理日誌
│
├── segment_002/
│   └── ...
│
└── summary_report.txt                  <-- 整體批次處理的成功/失敗摘要
```

## 參數調整與微調 (Parameter Adjustment & Tuning)

您可以透過命令列參數來微調偵測邏輯，以適應不同類型或品質的影片。

- `--workers <數量>`: 手動指定要使用的 CPU 核心數。
- `--hit_v <速度>`: 偵測擊球的最小瞬間速度（像素/幀）。預設 `40.0`。
- `--toss_vy <速度>`: 觸發「疑似拋球」的初始最小垂直速度。預設 `8.0`。
- `--vertical_ratio <比值>`: 拋球時，垂直速度必須是水平速度的最小倍數，用來過濾水平傳球。預設 `1.5`。
- `--hit_h_ratio <比值>`: 擊球時，水平速度相對於垂直速度的最大允許比值，用來過濾被誤判為擊球的水平傳球。預設 `2.5`。
- `--max_frames_to_apex <幀數>`: 從拋球開始，等待球到達頂點並開始下降的最長幀數，用以應對高拋球。預設 `75`。

**範例** (使用更嚴格的拋球判斷，並將擊球速度門檻降低):
```bash
python run_analysis_all_in_one.py --input_folder "path/to/your/input_videos" --vertical_ratio 2.0 --hit_v 35.0
```

## 模型訓練 (Model Training)

如果您發現物件追蹤模型（特別是球）在某些特定場景下表現不佳，您可以利用 `model_training/Fine_tuning.py` 腳本，使用少量您自己標註的「困難樣本」圖片，對現有的 `.pt` 模型進行微調，以提升其準確度。