#!/usr/bin/env python3
"""
track_ball_optional_bgsub.py

排球检测脚本，带可选的背景运动过滤（MOG2）功能。

用法：
  python track_ball_optional_bgsub.py \
    --input your_video.mp4 \
    --output_dir output_video \
    --model model/best.pt \
    --conf 0.3 \
    --device 0 \
    [--disable_motion]  # 可选，不使用背景运动过滤

输出：
  视频和帧目录，以及 YOLO 格式的 .txt 标签。
"""
import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import json

# 設定專案根目錄
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="對沙灘排球影片進行球追蹤分析。")
    parser.add_argument("--input", type=str, required=True, help="輸入的短影片片段路徑")
    parser.add_argument("--output_dir", type=str, default="output_data/tracking_output", 
                        help="追蹤結果的輸出根目錄 (相對於專案根目錄)")
    parser.add_argument("--model", type=str, default="model/ball_best.pt", 
                        help="排球偵測模型路徑 (相對於專案根目錄)")
    parser.add_argument("--conf", type=float, default=0.3, help="物件偵測的置信度閾值")
    parser.add_argument("--device", type=str, default="0", help="推理設備: 'cpu' 或 GPU id")
    parser.add_argument("--config_file_name", type=str, default="court_config.json", 
                        help="場地設定檔名稱 (位於專案根目錄)")
    parser.add_argument("--disable_motion", action='store_true',default=True , help="禁用背景运动过滤，仅使用模型检测")#true:禁用過濾
    return parser.parse_args()

def detect_ball(frame, ball_model, conf_thresh, background_ball_zones=None):
    res_ball = ball_model(frame, conf=conf_thresh, classes=[0])[0]
    ball_boxes = []
    for box in res_ball.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 檢查是否在背景球過濾區域內
        if background_ball_zones:
            in_background_zone = False
            for zone in background_ball_zones:
                if (zone["x1"] <= center_x <= zone["x2"] and 
                    zone["y1"] <= center_y <= zone["y2"]):
                    in_background_zone = True
                    break
            if in_background_zone:
                continue
        
        conf_score = float(box.conf[0].cpu().numpy())
        ball_boxes.append((x1, y1, x2, y2, conf_score))
    return ball_boxes

def main():
    args = parse_args()
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_root = os.path.join(args.output_dir, base)
    os.makedirs(out_root, exist_ok=True)
    frames_dir = os.path.join(out_root, "frames")
    labels_dir = os.path.join(out_root, "labels")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 可选背景分割器
    if not args.disable_motion:
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = os.path.join(out_root, f"{base}_output.avi")
    writer = cv2.VideoWriter(out_vid, fourcc, fps, (w,h))
    if not writer.isOpened():
        raise RuntimeError(f"无法打开输出视频: {out_vid}")

    # 设置模型
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    model = YOLO(args.model).to(device)

    # 載入場地設定
    config_file_path = os.path.join(project_root, args.config_file_name)
    background_ball_zones = []
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            court_config = json.load(f)
            background_ball_zones = court_config.get("background_ball_zones", [])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 背景运动过滤
        if not args.disable_motion:
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 在處理每一幀時傳入背景球過濾區域
        ball_boxes = detect_ball(frame, model, args.conf, background_ball_zones)

        # 检测球 (class 0)
        txt_lines = []
        for box in ball_boxes:
            x1, y1, x2, y2, conf_score = box
            # 画框
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame, f"ball {conf_score:.2f}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
            # 添加标签
            cx = ((x1+x2)/2)/w; cy = ((y1+y2)/2)/h
            bw = (x2-x1)/w; bh = (y2-y1)/h
            txt_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # 写入输出
        writer.write(frame)
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        label_path = os.path.join(labels_dir, f"frame_{frame_idx:06d}.txt")
        with open(label_path, 'w') as f:
            f.write("\n".join(txt_lines))

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"检测完成，输出目录: {out_root}")

if __name__ == "__main__":
    main()
