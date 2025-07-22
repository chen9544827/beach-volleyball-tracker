# video_processing/create_debug_video.py
import cv2
import json
import os
import argparse
import numpy as np
import sys

# --- 專案路徑設定 ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def parse_args():
    parser = argparse.ArgumentParser(description="產生一個包含詳細偵錯資訊的視覺化影片。")
    parser.add_argument("--video_input", type=str, required=True, help="原始影片路徑")
    
    # --- ✨ 核心修正：將 json_input 改為選填 ---
    parser.add_argument("--json_input", type=str, default=None, 
                        help="對應影片的 JSON 數據檔案路徑。如果省略，將根據影片名稱自動推斷預設路徑。")
    
    parser.add_argument("--tracking_output_dir", type=str, default="output_data/tracking_output", 
                        help="追蹤腳本 (track_...) 的輸出根目錄。")
    parser.add_argument("--output_dir", type=str, default="output_data/debug_video", 
                        help="偵錯影片的輸出目錄")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # --- ✨ 核心修正：自動建構 JSON 路徑 ---
    json_file_path = args.json_input
    if json_file_path is None:
        video_base_name = os.path.splitext(os.path.basename(args.video_input))[0]
        # 根據 track_ball_and_player.py 的輸出結構來組合路徑
        json_file_path = os.path.join(
            project_root, 
            args.tracking_output_dir,
            video_base_name,
            f"{video_base_name}_all_frames_data_with_pose.json"
        )
        print(f"[資訊] 未提供 --json_input，將使用自動推斷的路徑:\n   {json_file_path}")

    # --- 載入數據 ---
    print(f"\n[資訊] 正在從 {json_file_path} 載入幀數據...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            all_frames_data = json.load(f)
    except FileNotFoundError:
        print(f"[錯誤] 找不到 JSON 檔案 '{json_file_path}'。")
        print("➡️  請先執行 track_ball_and_player.py 來產生此檔案。")
        return
    print("[成功] 數據載入成功！")

    # --- 初始化影片讀寫 ---
    print("\n[資訊] 正在產生視覺化偵錯影片...")
    cap = cv2.VideoCapture(args.video_input)
    if not cap.isOpened():
        print(f"[錯誤] 無法開啟影片檔案 '{args.video_input}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    video_base_name = os.path.splitext(os.path.basename(args.video_input))[0]
    output_video_path = os.path.join(output_dir, f"{video_base_name}_debug_info.mp4")
    
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # --- 逐幀繪製偵錯資訊 ---
    for i in range(len(all_frames_data)):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_data = all_frames_data[i]
        frame_id = frame_data['frame_id']
        
        cv2.putText(frame, f"Frame: {frame_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        balls = frame_data.get('ball_detections', [])
        players = frame_data.get('player_detections', [])
        
        best_ball = None
        if balls:
            best_ball = max(balls, key=lambda b: b['confidence'])
            ball_pos = np.array(best_ball['center_point'])
            
            ball_vx, ball_vy = 0, 0
            if i > 0 and all_frames_data[i-1].get('ball_detections'):
                prev_balls = all_frames_data[i-1]['ball_detections']
                # 找到前一幀中最接近當前球的球
                prev_ball_candidate = min(prev_balls, key=lambda b: np.linalg.norm(np.array(b['center_point']) - ball_pos), default=None)
                if prev_ball_candidate:
                    prev_ball_pos = np.array(prev_ball_candidate['center_point'])
                    ball_vx = ball_pos[0] - prev_ball_pos[0]
                    ball_vy = prev_ball_pos[1] - ball_pos[1] # 向上為正

            info_x, info_y = int(ball_pos[0]) + 15, int(ball_pos[1])
            cv2.putText(frame, "Ball Data:", (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"vy: {ball_vy:.1f}", (info_x, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"vx: {ball_vx:.1f}", (info_x, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if players and best_ball:
            ball_pos = np.array(best_ball['center_point']) # 確保 ball_pos 已定義
            for p_idx, player in enumerate(players):
                player_pos = np.array(player['center_point'])
                dist_to_ball = np.linalg.norm(player_pos - ball_pos)
                
                p_info_x, p_info_y = int(player['box_coords'][0]), int(player['box_coords'][1]) - 10
                cv2.putText(frame, f"P{p_idx} Dist: {dist_to_ball:.1f}", (p_info_x, p_info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"\n[成功] 視覺化偵錯影片已儲存至: {os.path.abspath(output_video_path)}")

if __name__ == '__main__':
    main()