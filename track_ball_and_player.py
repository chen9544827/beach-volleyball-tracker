#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       type=str, required=True,  help="输入视频文件")
    p.add_argument("--output_dir",  type=str, default="output_video", help="输出根目录")
    p.add_argument("--ball_model",  type=str, default="model/best.pt",help="球检测模型路径（你的 fine-tune best.pt）")
    p.add_argument("--player_model",type=str, default="model/yolo11s.pt",help="球员检测模型路径（如 yolo11s.pt）")
    p.add_argument("--conf",        type=float, default=0.3,   help="置信度阈值")
    p.add_argument("--device",      type=str, default="0",     help="推理设备: 'cpu' or GPU id")
    return p.parse_args()

def main():
    args = parse_args()

    # 基础准备：路径、模型、背景分割
    video_path = args.input
    base       = os.path.splitext(os.path.basename(video_path))[0]
    out_root   = os.path.join(args.output_dir, base)
    os.makedirs(out_root, exist_ok=True)
    frame_dir  = os.path.join(out_root, "frames")
    label_dir  = os.path.join(out_root, "labels")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    device = args.device
    if device.isdigit():
        device = f"cuda:{device}"

    # 载入模型
    ball_model   = YOLO(args.ball_model).to(device)
    player_model = YOLO(args.player_model).to(device)

    # 背景分割器
    fgbg   = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出视频
    out_vid = os.path.join(out_root, f"{base}_output.avi")
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    writer  = cv2.VideoWriter(out_vid, fourcc, fps, (w,h))
    if not writer.isOpened():
        raise RuntimeError(f"无法打开输出视频: {out_vid}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 前景分割 & 去噪
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 1) 球体检测 + 运动过滤
        res_ball = ball_model(frame, conf=args.conf, classes=[0])[0]
        ball_boxes = []
        for box in res_ball.boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
            roi = fgmask[y1:y2, x1:x2]
            if roi.size == 0: 
                continue
            motion_ratio = np.count_nonzero(roi) / roi.size
            if motion_ratio < 0.02:
                continue
            ball_boxes.append((x1,y1,x2,y2, float(box.conf[0].cpu().numpy())))

        # 2) 球员检测（不做运动过滤）
        res_player = player_model(frame, conf=args.conf, classes=[0])[0]
        player_boxes = []
        for box in res_player.boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
            player_boxes.append((x1,y1,x2,y2, float(box.conf[0].cpu().numpy())))

        # 3) 框绘制与标签保存
        # 保存原始帧
        frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)

        # 写标签
        txt_lines = []
        # volleyball: class 0
        for (x1,y1,x2,y2,conf) in ball_boxes:
            cx = ((x1+x2)/2)/w; cy = ((y1+y2)/2)/h
            bw = (x2-x1)/w; bh = (y2-y1)/h
            txt_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            # 画框
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame, f"ball {conf:.2f}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        # player: class 1
        for (x1,y1,x2,y2,conf) in player_boxes:
            cx = ((x1+x2)/2)/w; cy = ((y1+y2)/2)/h
            bw = (x2-x1)/w; bh = (y2-y1)/h
            txt_lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            # 画框
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, f"player {conf:.2f}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

        # 写入每帧标签文件
        label_path = os.path.join(label_dir, f"frame_{frame_idx:06d}.txt")
        with open(label_path, 'w') as f:
            f.write("\n".join(txt_lines))

        # 写入输出视频
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Finished! 结果保存在: {out_root}")

if __name__ == "__main__":
    main()
