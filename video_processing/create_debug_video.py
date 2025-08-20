# video_processing/create_debug_video.py (v3 - 增加圖片序列儲存功能)
import cv2
import json
import os
import argparse
import numpy as np
import sys
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="產生包含詳細狀態機偵錯資訊的視覺化影片，並可選擇儲存圖片序列。")
    parser.add_argument("--video_input", type=str, required=True, help="原始影片路徑")
    parser.add_argument("--debug_log", type=str, required=True, help="由主程式產生的 _serve_debug_log.csv 檔案路徑")
    parser.add_argument("--output_dir", type=str, default="volleyball_analysis_results/debug_outputs", help="偵錯影片與圖片的輸出根目錄")
    # ✨ 核心修改 1: 新增命令列參數，讓使用者可以選擇是否儲存圖片
    parser.add_argument("--save_images", action="store_true", help="加上此旗標，會將每一幀都儲存為圖片。")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # --- 載入偵錯日誌 ---
    print(f"\n[資訊] 正在從 {args.debug_log} 載入偵錯日誌...")
    try:
        with open(args.debug_log, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            debug_data = {int(row['frame_id']): row for row in reader}
        print(f"[成功] 載入 {len(debug_data)} 筆幀數據！")
    except FileNotFoundError:
        print(f"[錯誤] 找不到偵錯日誌檔案 '{args.debug_log}'。")
        print("➡️  請先執行 run_analysis_all_in_one.py 來產生此檔案。")
        return

    # --- 初始化影片讀寫 ---
    cap = cv2.VideoCapture(args.video_input)
    if not cap.isOpened(): print(f"[錯誤] 無法開啟影片檔案 '{args.video_input}'"); return
    fps = cap.get(cv2.CAP_PROP_FPS); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_base_name = os.path.splitext(os.path.basename(args.video_input))[0]
    
    # 建立主輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 定義影片輸出路徑
    output_video_path = os.path.join(args.output_dir, f"{video_base_name}_debug.mp4")
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # ✨ 核心修改 2: 如果需要儲存圖片，建立專屬的圖片資料夾
    image_output_dir = None
    if args.save_images:
        image_output_dir = os.path.join(args.output_dir, f"{video_base_name}_debug_frames")
        os.makedirs(image_output_dir, exist_ok=True)
        print(f"[資訊] 偵錯圖片將儲存至: {image_output_dir}")

    print(f"\n[資訊] 正在產生視覺化偵錯影片...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        data = debug_data.get(frame_idx, {})
        
        # --- 在畫面上繪製所有偵錯資訊 ---
        state = data.get('state', 'UNKNOWN')
        speed = float(data.get('speed', 0)); vy = float(data.get('vy', 0)); vx = float(data.get('vx', 0)); h_ratio = float(data.get('h_ratio', 0)); lost_frames = int(data.get('lost_frames', 0))
        
        # 繪製背景板
        cv2.rectangle(frame, (w - 470, 10), (w - 10, 180), (0, 0, 0), -1)

        cv2.putText(frame, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        color = (0, 255, 0) if "SEARCHING" in state else (0, 165, 255) if "CONFIRM" in state else (0, 0, 255)
        cv2.putText(frame, f"State: {state}", (w - 450, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        text_y_start = 80
        cv2.putText(frame, f"Speed: {speed:.2f}", (w - 450, text_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"VY (Up+): {vy:.2f}", (w - 450, text_y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"H-Ratio: {h_ratio:.2f}", (w - 450, text_y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if lost_frames > 0:
            cv2.putText(frame, f"LostFrames: {lost_frames}", (w - 200, text_y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

        writer.write(frame)
        
        # ✨ 核心修改 3: 如果啟用，儲存當前幀為圖片
        if args.save_images and image_output_dir:
            image_path = os.path.join(image_output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(image_path, frame)

        frame_idx += 1

    cap.release(); writer.release()
    print(f"\n[成功] 視覺化偵錯影片已儲存至: {os.path.abspath(output_video_path)}")
    if args.save_images:
        print(f"[成功] 偵錯圖片序列已儲存至: {os.path.abspath(image_output_dir)}")

if __name__ == '__main__':
    main()