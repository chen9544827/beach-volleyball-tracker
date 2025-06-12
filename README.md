# 沙灘排球分析專案 (Beach Volleyball Tracker)
本專案旨在利用電腦視覺和深度學習技術，對沙灘排球比賽影片進行分析，實現包括球員追蹤、球體追蹤與路徑分析、發球等事件偵測、以及基於分數變化的影片自動分割等功能。

## 主要功能

*   **場地定義**：透過點擊影片畫格，手動定義場地邊界、排除區域、網子位置及背景球過濾區域。設定儲存為 JSON 檔案。
*   **物件偵測模型微調**：使用 YOLO (yolo11s.pt) 模型，針對排球和球員偵測進行微調。
*   **發球事件偵測**：分析球的軌跡與球員動作（拋球、最高點、擊球），以識別發球事件。
*   **球體軌跡分析**：追蹤球的運動路徑並進行分析。
*   **跳發分析**：針對跳發球進行特定分析 (例如跳躍高度)。
*   **結果摘要生成**：將分析結果（如跳發事件、軌跡數據）匯總為 CSV 檔案。

## 專案結構

以下為主要資料夾的功能說明：

*   `court_definition/`: 用於定義場地幾何形狀（邊界、排除區、網高等）。
*   `input_video/`: 用於存放輸入的影片檔案。
*   `model_training/`: 包含物件偵測模型微調的腳本。
*   `video_processing/`: 包含核心的影片分析腳本，如事件偵測、球體追蹤等。

## 依賴套件

本專案主要依賴以下 Python 套件 (詳細列表請參考 `requirements.txt`):

*   `roboflow`
*   `ultralytics`
*   `opencv-python`
*   `numpy`
*   `torch`

## 安裝與使用說明

### 安裝 (Setup)

1.  **複製專案庫** (如果您尚未這樣做):
    ```bash
    git clone <專案URL>
    cd <專案目錄>
    ```
2.  **安裝依賴套件**:
    建議在 Python 虛擬環境中進行安裝。
    ```bash
    pip install -r requirements.txt
    ```
3.  **準備預訓練模型**:
    *   本專案的物件偵測模型微調 (`model_training/Fine_tuning.py`) 基於 YOLO 模型 (例如 `yolo11s.pt` 或其他 YOLOv8 預訓練權重)。
    *   請下載您選擇的預訓練 YOLO 模型權重檔案 (.pt)。
    *   建議將此檔案放置在專案的根目錄，或 `model_training/` 資料夾中。腳本 `model_training/Fine_tuning.py` 會需要引用此模型路徑。
4.  **模型微調資料集**:
    *   若要進行模型微調，請確保您的訓練資料集已依照 `model_training/combined_dataset/data.yaml` 檔案中的路徑和格式進行配置。此 YAML 檔案定義了訓練集、驗證集的圖像位置以及類別等資訊。

### 執行流程 (Execution Workflow)

1.  **定義場地 (`court_config.json`)**:
    *   執行以下命令，其中 `<影片路徑>` 是您要分析的影片檔案路徑 (或用於定義場地配置的代表性影片):
        ```bash
        python court_definition/court_config_generator.py <影片路徑>
        ```
    *   此腳本將開啟影片的第一幀，並引導您透過滑鼠點擊來定義以下場地區域：
        *   場地邊界 (Court boundaries)
        *   排除區域 (Exclusion zones)
        *   網子頂部和底部 (Net top and bottom for height calculation)
        *   背景球過濾區域 (Background ball filter area)
    *   完成後，相關座標將儲存於專案根目錄下的 `court_config.json` 檔案中。此檔案將被後續的分析腳本使用。

2.  **模型微調 (可選步驟)**:
    *   如果您需要基於自定義資料集對球和球員的偵測模型進行微調，請執行：
        ```bash
        python model_training/Fine_tuning.py
        ```
    *   此腳本會：
        *   讀取 `model_training/combined_dataset/data.yaml` 中設定的資料集。
        *   使用您在步驟 `安裝 (Setup) - 3` 中準備的預訓練 YOLO 模型作為基礎。
        *   開始微調訓練，並將新的模型權重儲存在 `runs/train/finetune_ball_player/weights/` 目錄下 (通常會有 `best.pt` 和 `last.pt`)。
    *   微調完成後，您可以在後續分析步驟中使用新產生的 `best.pt` 模型。

3.  **執行影片分析**:
    *   影片分析的核心流程通常包含物件追蹤、事件識別和摘要生成。以下為各主要組件的執行方式 (請注意，您可能需要根據實際的腳本整合情況調整執行順序或指令):
        *   **物件追蹤**:
            *   使用 `video_processing/` 目錄下的相關腳本 (例如 `track_ball_and_player.py` 或類似功能的腳本) 來處理輸入影片，並產生球和球員的追蹤數據 (通常是 JSON 或 CSV 格式)。此步驟會使用到微調後的模型 (如果進行了步驟2) 和 `court_config.json`。
            *   *(使用者應根據其具體的追蹤腳本名稱和參數填寫此處的指令)*
            *   例如: `python video_processing/track_ball_and_player.py --input_video <影片路徑> --output_dir <追蹤輸出目錄> --ball_model <球模型.pt> --player_model <球員模型.pt> --court_config court_config.json`
        *   **事件分析 (`event_analyzer.py`)**:
            *   此腳本分析追蹤數據以偵測發球等關鍵事件。
            *   執行指令範例:
                ```bash
                python video_processing/event_analyzer.py --input <追蹤資料.json> --output_dir <事件分析輸出目錄> --court_config court_config.json
                ```
            *   `<追蹤資料.json>` 是上一步物件追蹤產生的檔案路徑。
        *   **摘要生成 (`summary_generator.py`)**:
            *   此腳本將不同分析模組的結果 (如跳發事件、軌跡數據) 匯總成單一的摘要檔案。
            *   執行指令範例:
                ```bash
                python video_processing/summary_generator.py --jump_serve <跳發事件.json> --trajectory <軌跡分析.json> --output <最終摘要.csv>
                ```
            *   `<跳發事件.json>` 和 `<軌跡分析.json>` 分別是事件分析和軌跡分析模組的輸出。
    *   **重要提示**: 上述指令和腳本名稱可能需要根據您專案的最終結構進行調整。建議提供一個主要的執行腳本或更清晰的端到端執行流程說明。

4.  **查看結果**:
    *   分析結果通常會儲存在您在執行命令時指定的輸出目錄中。
    *   結果可能包含：
        *   詳細的幀級分析數據 (JSON 格式)。
        *   事件列表 (JSON 或 CSV 格式)。
        *   最終的比賽摘要報告 (CSV 格式)。

## 如何貢獻

歡迎對此專案做出貢獻！您可以透過以下方式參與：

*   回報問題 (Issues)
*   提交功能需求 (Feature Requests)
*   提交拉取請求 (Pull Requests) 進行程式碼改進或錯誤修復。

如果您有任何想法或建議，請隨時提出。

## 授權條款

此專案目前未指定授權條款。請考慮添加一個開源授權條款，例如 MIT License 或 Apache License 2.0。

## 功能特色

* **影片自動分割 (基於OCR分數辨識)：**
    * 自動辨識影片中的比分板分數。
    * 根據分數變化判斷有效比賽片段，並將長影片分割成多個短片段。
    * 可將過長或過短的片段進行分類儲存或刪除。
* **物件偵測與追蹤 (計劃中/進行中，基於YOLO)：**
    * 偵測畫面中的排球。
    * 偵測並追蹤運動員 (球員)。
    * (已討論) 球員篩選邏輯：基於與場地中心的距離選擇主要球員。
* **場地幾何定義 (手動配置)：**
    * 允許使用者手動點擊影片第一幀來定義球場邊界。
    * (已討論) 允許使用者手動定義分數板的ROI區域用於OCR。
* **比賽事件分析 (計劃中/進行中)：**
    * 偵測發球事件及識別發球員。
    * (未來可能) 分析球的軌跡與落點。
    * (未來可能) 判斷發球種類 (跳發/非跳發)。

## 專案結構
## 專案結構

- `beach-volleyball-tracker/`
  - `court_definition/`                 # 存放與場地定義、配置相關的腳本
    - `court_config_generator.py`     # 用於手動定義場地邊界和ROI
    - `court_config.json`             # (由上面腳本產生的場地設定檔)
  - `data_preparation/`                 # 存放與資料集下載、合併、轉換相關的腳本
    - `Download_dataset.py`
    - `combine_dataset.py`
    - `track_person_for_dataset.py`   # (原 track_person.py)
  - `model_training/`                   # 存放與模型訓練、微調相關的腳本
    - `Fine_tuning.py`
  - `video_processing/`                 # 存放處理影片、進行物件偵測、分割影片等核心功能的腳本
    - `video_slicer_by_score.py`      # 用OCR分割影片的腳本
    - `track_ball_and_player.py`      # 對短影片進行球和球員分析的腳本
    - `track_ball_only.py`            # (原 track_ball.py)
  - `models/`                           # 存放訓練好的模型權重檔案 (.pt)
    - `ball_best.pt`
    - `player_yolo.pt`
    - `yolo11s.pt`
    - `best.pt`
  - `output_data/`                      # 存放腳本執行後的輸出結果 (建議加入.gitignore)
    - `video_segments_output/`        # (由 video_slicer_by_score.py 產生)
      - `normal_segments/`
      - `long_segments/`
      - `temp_segments/`
    - `tracking_output/`              # (由 track_ball_and_player.py 產生)
      - `[video_name]_output/`
        - `frames/`
        - `labels/`
    - `dataset_output/`               # (由 data_preparation/ 中的腳本產生)
      - `output_player_
## 環境設定

1.  **複製專案：**
    ```bash
    git clone [你的專案Git URL]
    cd beach-volleyball-tracker
    ```

2.  **安裝Python依賴：**
    建議使用虛擬環境 (virtual environment)。
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    
    pip install -r requirements.txt
    ```
    主要的依賴套件包括 (詳見 `requirements.txt`)：
    * `opencv-python`
    * `ultralytics` (用於YOLO模型)
    * `numpy`
    * `torch`
    * `pytesseract` (用於OCR)

3.  **安裝 Tesseract OCR 引擎：**
    `pytesseract` 需要系統中已安裝 Tesseract OCR 引擎。
    * **Windows：** 從 [UB Mannheim 的 Tesseract 安裝程式](https://github.com/UB-Mannheim/tesseract/wiki) 下載並安裝。安裝時請確保選擇了 "English" (eng) 語言包。安裝後，可能需要將 Tesseract 的安裝路徑（例如 `C:\Program Files\Tesseract-OCR`）添加到系統的PATH環境變數，或者在腳本中指定 `pytesseract.pytesseract.tesseract_cmd` 的路徑。
    * **macOS：** `brew install tesseract tesseract-lang`
    * **Linux (Ubuntu/Debian)：** `sudo apt update && sudo apt install tesseract-ocr tesseract-ocr-eng`

4.  **準備模型檔案：**
    * 將你訓練好的YOLO模型權重檔案 (例如 `ball_best.pt`, `player_yolo.pt` 等) 放入 `models/` 資料夾。
    * 如果使用了預訓練的 `yolo11s.pt` (可能是yolov8s.pt或其他YOLO系列的預訓練權重)，也應放入 `models/` 資料夾或腳本可以訪問的路徑。

## 使用說明

### 1. 場地與ROI定義 (首次執行或需要更新時)

* 執行 `court_config_generator.py` 來為你的影片（或一類相似視角的影片）定義球場邊界和分數板ROI。
    ```bash
    # 從專案根目錄執行
    python court_definition/court_config_generator.py path/to/your_sample_video.mp4
    ```
    這會在專案根目錄下生成（或覆蓋） `court_config.json` 檔案。腳本會引導你手動點擊定義區域。請確保 `court_config.json` 中的分數板ROI座標 (`SCORE_ROI_TEAM1`, `SCORE_ROI_TEAM2`) 與 `video_slicer_by_score.py` 中硬編碼（或未來從設定檔讀取）的值一致。 *(註：目前 `video_slicer_by_score.py` 中ROI是硬編碼的，未來可以考慮讓它也從 `court_config.json` 讀取以保持一致性)*

### 2. 分割長時間影片 (基於OCR分數)

* 使用 `video_slicer_by_score.py` 將你的原始長影片分割成有效的比賽片段。
    ```bash
    # 從專案根目錄執行
    python video_processing/video_slicer_by_score.py \
        --input "path/to/your_long_match_video.mp4" \
        --output_dir "output_data/my_match_segments" \
        --min_segment_duration 10 \
        --long_segment_threshold 90 \
        --score_check_interval 1 \
        --no_score_change_timeout 120
    ```
    * `--input`: 原始長影片的路徑。
    * `--output_dir`: 分割後片段的儲存根目錄 (會在 `output_data/` 下)。
    * 其他參數用於控制分割邏輯，詳見腳本內的 `parse_arguments`。
    * 分割後的片段會存放在 `--output_dir` 下的 `normal_segments/` 和 `long_segments/` 子資料夾中。

### 3. 對分割後的短影片進行球與球員分析

* 使用 `track_ball_and_player.py` 處理由上一步產生的短影片片段。
    ```bash
    # 從專案根目錄執行
    python video_processing/track_ball_and_player.py \
        --input "output_data/my_match_segments/normal_segments/segment_001.mp4" \
        --output_dir "output_data/tracking_output" \
        --ball_model "models/ball_best.pt" \
        --player_model "models/player_yolo.pt" \
        --conf 0.3 
    ```
    * `--input`: 單個短影片片段的路徑。
    * `--output_dir`: 追蹤結果的儲存根目錄 (會在 `output_data/` 下)。
    * 腳本會讀取根目錄下的 `court_config.json` 進行場地相關的判斷。
    * 分析結果（例如帶有標註框的影片、每幀的物件座標等）會儲存在指定的輸出目錄下。

### 4. (其他腳本)

* `data_preparation/` 中的腳本用於下載或處理資料集。
* `model_training/Fine_tuning.py` 用於模型微調。
    請參考各腳本內部的說明或命令行參數 (`--help`) 來了解其具體用法。

## 未來展望與待辦事項

* [ ] **發球事件偵測：** 實作更精準的發球事件偵測邏輯。
* [ ] **球軌跡與落點分析：** 開發球體飛行軌跡的建模與落點預測。
* [ ] **發球種類識別：** 判斷跳發或非跳發。
* [ ] **將ROI設定統一到 `court_config.json`：** 讓 `video_slicer_by_score.py` 中的分數板ROI也從 `court_config.json` 讀取，而不是硬編碼。
* [ ] **更完善的錯誤處理和日誌記錄。**
* [ ] **(可選) 圖形化使用者介面 (GUI)。**

## 貢獻

歡迎提出問題、bug報告或功能請求。

## 授權

(如果你有特定的授權方式，可以在這裡說明，例如 MIT License)
