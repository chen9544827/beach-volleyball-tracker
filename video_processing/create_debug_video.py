# video_processing/create_debug_video.py (路徑修正 + 視覺化增強版 v2)
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
    # ✨ 核心修正：將 type_str 修正為 str
    parser.add_argument("--video_input", type=str, required=True, help="原始影片路徑")
    parser.add_argument("--json_input", type=str, default=None,
                        help="對應影片的 JSON 數據檔案路徑。如果省略，將根據影片名稱自動推斷預設路徑。")
    parser.add_argument("--analysis_output_folder", type=str, default="volleyball_analysis_results",
                        help="主分析程式 (run_analysis_all_in_one.py) 的輸出根目錄。")
    parser.add_argument("--output_dir", type=str, default="output_data/debug_video",
                        help="偵錯影片的輸出目錄")
    return parser.parse_args()

def main():
    args = parse_args()

    json_file_path = args.json_input
    if json_file_path is None:
        video_base_name = os.path.splitext(os.path.basename(args.video_input))[0]
        json_file_path = os.path.join(
            project_root,
            args.analysis_output_folder,
            video_base_name,
            'tracking_output',
            f"{video_base_name}_all_frames_data_with_pose.json"
        )
        print(f"[資訊] 未提供 --json_input，將使用自動推斷的路徑:\n   {json_file_path}")

    print(f"\n[資訊] 正在從 {json_file_path} 載入幀數據...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            all_frames_data = json.load(f)
    except FileNotFoundError:
        print(f"[錯誤] 找不到 JSON 檔案 '{json_file_path}'。")
        print("➡️  請先執行 run_analysis_all_in_one.py 來產生此檔案。")
        return
    print("[成功] 數據載入成功！")

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
    output_video_path = os.path.join(output_dir, f"{video_base_name}_debug_ball_tracking.mp4")

    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for i in range(len(all_frames_data)):
        ret, frame = cap.read()
        if not ret: break

        frame_data = all_frames_data[i]
        frame_id = frame_data['frame_id']

        cv2.putText(frame, f"Frame: {frame_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        balls = frame_data.get('ball_detections', [])
        best_ball_of_frame = None
        if balls:
            best_ball_of_frame = max(balls, key=lambda b: b['confidence'])
            for ball_data in balls:
                box = ball_data['box_coords']
                center = tuple(map(int, ball_data['center_point']))
                conf = ball_data['confidence']
                box_color = (255, 191, 0) # 亮藍色
                if ball_data == best_ball_of_frame:
                    box_color = (0, 255, 0) # 綠色
                
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.putText(frame, f"{conf:.2f}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"\n[成功] 包含球標記的偵錯影片已儲存至: {os.path.abspath(output_video_path)}")

if __name__ == '__main__':
    main()