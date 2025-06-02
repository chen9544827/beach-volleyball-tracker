#!/usr/bin/env python3
# generate_player_dataset.py
# 该脚本从视频中自动生成“选手”（player）类别的YOLO格式数据集。
# 仅输出 player 标签，用于制作纯 player 数据集。
# 输出目录结构：
#   output_player_dataset/
#     images/
#     labels/

import os
import cv2
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Generate YOLO dataset from video with only player labels.")
    parser.add_argument("--input_video", type=str, required=True,help="输入视频文件路径")
    parser.add_argument("--output_dir", type=str, default="output_player_dataset",help="生成数据集的根目录")
    parser.add_argument("--player_model", type=str, default="model/yolo11s.pt",help="球员检测模型路径 (如 yolo11s.pt 或 COCO 预训练权重)")
    parser.add_argument("--conf", type=float, default=0.3,help="置信度阈值")
    parser.add_argument("--frame_step", type=int, default=1,help="每隔多少帧保存一次 (默认1: 每帧)")
    parser.add_argument("--device", type=str, default="0",help="推理设备: 'cpu' or GPU id")
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置设备
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    # 准备输出目录
    img_dir = os.path.join(args.output_dir, "images")
    lbl_dir = os.path.join(args.output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # 加载 player 模型
    player_model = YOLO(args.player_model).to(device)

    # 打开视频
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {args.input_video}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    save_idx  = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 跳过不需要处理的帧
        if frame_idx % args.frame_step != 0:
            frame_idx += 1
            continue

        # 推理并获取结果
        result = player_model(frame, conf=args.conf, classes=[0])[0]

        labels = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # 如果检测到 player，则保存图像和标签
        if labels:
            img_path = os.path.join(img_dir, f"{save_idx:06d}.jpg")
            lbl_path = os.path.join(lbl_dir, f"{save_idx:06d}.txt")
            cv2.imwrite(img_path, frame)
            with open(lbl_path, 'w') as f:
                f.write("\n".join(labels))
            save_idx += 1

        frame_idx += 1

    cap.release()
    print(f"Finished! 共保存 {save_idx} 帧到 {args.output_dir}")

if __name__ == "__main__":
    main()
