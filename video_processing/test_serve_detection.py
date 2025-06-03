# video_processing/test_serve_detec tion.py

import json
import os
import sys
import cv2 # is_player_behind_baseline 會用到
import numpy as np # is_player_behind_baseline 和 find_serve_events 會用到

# --- 設定Python導入路徑 ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 從你的主腳本或分析模組導入需要的函數 ---
# 假設 find_serve_events 和其輔助函數在 track_ball_and_player.py 中
# 或者如果你把它們移到了例如 analysis_logic.py 中，就從那裡導入
try:
    from video_processing.track_ball_and_player import find_serve_events, get_player_center, is_player_behind_baseline
    from court_definition.court_config_generator import load_court_geometry # 用於載入 court_config.json
except ImportError as e:
    print(f"導入函數時發生錯誤: {e}")
    print("請確保 track_ball_and_player.py 和 court_config_generator.py 在正確的路徑並且包含所需的函數。")
    sys.exit(1)

def main_test_serve_detection():
    # --- 參數設定 ---
    # 1. 指定儲存的 all_frames_data.json 檔案路徑
    #    你需要將 'your_video_name_all_frames_data.json' 替換成你實際產生的檔案名
    #    這個路徑應該是相對於 project_root 的
    all_frames_data_path = os.path.join(project_root, "output_data", "tracking_output", 
                                        "segment_004", # 這是 track_ball_and_player.py 中 specific_output_dir_for_this_video 的一部分
                                        "segment_004_all_frames_data.json") # 假設影片名是 test.mp4

    # 2. 指定 court_config.json 檔案路徑
    config_file_path = os.path.join(project_root, "court_config.json")

    # 3. 影片的基本資訊 (需要與產生 all_frames_data 時的影片一致)
    #    這些值可以硬編碼用於測試，或者從某個地方讀取
    #    如果你的 all_frames_data 中沒有包含這些資訊，你需要手動提供
    test_video_fps = 30.0  # 假設值，請替換為你測試影片的實際FPS
    test_video_width = 640 # 假設值，請替換為實際寬度
    test_video_height = 640 # 假設值，請替換為實際高度

    # --- 載入測試數據 ---
    if not os.path.exists(all_frames_data_path):
        print(f"錯誤: 測試數據檔案 '{all_frames_data_path}' 不存在。")
        print("請先運行 track_ball_and_player.py 處理一個短影片並儲存 all_frames_data。")
        return

    if not os.path.exists(config_file_path):
        print(f"錯誤: 場地設定檔 '{config_file_path}' 不存在。")
        return

    try:
        with open(all_frames_data_path, 'r') as f:
            all_frames_data_loaded = json.load(f)
        print(f"成功從 '{all_frames_data_path}' 載入 {len(all_frames_data_loaded)} 幀的數據。")
    except Exception as e:
        print(f"載入 all_frames_data 時發生錯誤: {e}")
        return

    court_geometry_loaded = load_court_geometry(config_load_path=config_file_path)
    if court_geometry_loaded is None:
        print(f"無法從 '{config_file_path}' 載入場地幾何資訊。")
        return
    print(f"成功載入場地幾何資訊。")


    # --- 執行發球偵測 ---
    if all_frames_data_loaded and court_geometry_loaded:
        print(f"\n開始使用載入的數據進行發球事件分析...")
        
        serve_events_found = find_serve_events(
            all_frames_data_loaded, 
            court_geometry_loaded, 
            test_video_fps, 
            test_video_width, 
            test_video_height
        )
        
        if serve_events_found:
            print(f"\n共偵測到 {len(serve_events_found)} 個發球事件：")
            for idx, event in enumerate(serve_events_found):
                player_box = event['serving_player_info']['box_coords']
                ball_start_pos = event['ball_start_position']
                print(f"  事件 {idx+1}:")
                print(f"    發球影格 ID (serve_frame_idx): {event['serve_frame_idx']}")
                print(f"    發球球員 Box (x1,y1,x2,y2): ({player_box[0]}, {player_box[1]}, {player_box[2]}, {player_box[3]})")
                print(f"    發球球員信心度: {event['serving_player_info'].get('confidence', 'N/A'):.2f}")
                print(f"    球起始位置 (x,y): ({ball_start_pos[0]:.0f}, {ball_start_pos[1]:.0f})")
                if 'ball_initial_motion_vector' in event:
                    motion_vec = event['ball_initial_motion_vector']
                    print(f"    球初始運動向量 (dx,dy): ({motion_vec[0]:.1f}, {motion_vec[1]:.1f})")
        else:
            print("在此數據中未偵測到發球事件。")
    else:
        print("載入的數據或場地幾何資訊不完整，無法進行分析。")

if __name__ == "__main__":
    main_test_serve_detection()