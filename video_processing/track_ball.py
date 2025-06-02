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

def parse_args():
    parser = argparse.ArgumentParser(description="排球检测，支持可选背景运动过滤")
    parser.add_argument("--input",      type=str, required=True, help="输入视频路径")
    parser.add_argument("--output_dir", type=str, default="../output_data", help="输出父目录")
    parser.add_argument("--model",      type=str, default="../model/best.pt", help="模型权重路径")
    parser.add_argument("--conf",       type=float, default=0.3, help="检测置信度阈值")
    parser.add_argument("--device",     type=str, default="0", help="GPU 设备 ID 或 'cpu'")
    parser.add_argument("--disable_motion", action='store_true',default=True , help="禁用背景运动过滤，仅使用模型检测")#true:禁用過濾
    return parser.parse_args()


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

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 背景运动过滤
        if not args.disable_motion:
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 检测球 (class 0)
        res = model(frame, conf=args.conf, classes=[0])[0]
        txt_lines = []
        for box in res.boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf_score = float(box.conf[0].cpu().numpy())
            # 如果启用运动过滤，按阈值剔除静止球
            if not args.disable_motion:
                roi = fgmask[y1:y2, x1:x2]
                if roi.size == 0: continue
                motion_ratio = np.count_nonzero(roi) / roi.size
                if motion_ratio < 0.02: continue
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
