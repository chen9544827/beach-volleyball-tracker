beach-volleyball-tracker/
├── court_definition/                 # 存放與場地定義、配置相關的腳本
│   └── court_config_generator.py     # (我們之前討論的，用於手動定義場地邊界和ROI)
│   └── court_config.json             # (由上面腳本產生的場地設定檔)
│
├── data_preparation/                 # 存放與資料集下載、合併、轉換相關的腳本
│   └── Download_dataset.py           #
│   └── combine_dataset.py            #
│   └── track_person_for_dataset.py   # (原 track_person.py, 如果主要目的是生成純player資料集)
│
├── model_training/                   # 存放與模型訓練、微調相關的腳本
│   └── Fine_tuning.py                #
│
├── video_processing/                 # 存放處理影片、進行物件偵測、分割影片等核心功能的腳本
│   └── video_slicer_by_score.py      # (我們剛討論的，專門用OCR分割影片的腳本)
│   └── track_ball_and_player.py      # (用於對已分割的短影片進行分析)
│   └── track_ball_only.py            # (原 track_ball.py, 如果還有單獨追蹤球的需求)
│
├── models/                           # 存放訓練好的模型權重檔案 (.pt)
│   └── ball_best.pt                  # (來自 track_ball_and_player.py 的預設)
│   └── player_yolo.pt                # (來自 track_ball_and_player.py 的預設)
│   └── yolo11s.pt                    # (來自 Fine_tuning.py 和 track_person.py 的預設)
│   └── best.pt                       # (來自 track_ball.py 的預設)
│
├── output_data/                      # 存放腳本執行後的輸出結果
│   ├── video_segments_output/        # (由 video_slicer_by_score.py 產生)
│   │   ├── normal_segments/
│   │   └── long_segments/
│   │   └── temp_segments/
│   ├── tracking_output/              # (由 track_ball_and_player.py 產生，例如包含標註的影片或幀、標籤檔)
│   │   └── [video_name]_output/
│   │       ├── frames/
│   │       └── labels/
│   ├── dataset_output/               # (由 track_person_for_dataset.py 或 combine_dataset.py 產生)
│   │   └── output_player_dataset/
│   │       ├── images/
│   │       └── labels/
│   │   └── combined_dataset/         # (由 combine_dataset.py 產生)
│   │       ├── train/
│   │       ├── valid/
│   │       └── test/
│   │       └── data.yaml
│
├── utils/                            # (可選) 存放一些可能被多個腳本共用的輔助函數或類別
│   └── ocr_utils.py                  # (例如，可以把OCR相關的 preprocess 和 extract 函數放這裡)
│   └── video_utils.py                # (例如，影片讀寫、影格處理的通用函數)
│
├── .gitignore                        # Git忽略檔案列表 (建議加入 output_data/, models/ 可能下載的預訓練權重, __pycache__/, *.pyc 等)
├── requirements.txt                  #
└── README.md                         # 專案說明檔案
