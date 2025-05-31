#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="同时使用两个模型：排球检测（带运动过滤）和选手检测（不带运动过滤）")
    p.add_argument("--input",        type=str, required=True,  help="输入视频文件路径")
    p.add_argument("--output_dir",   type=str, default="output_video", help="输出根目录")
    p.add_argument("--ball_model",   type=str, default="model/ball_best.pt", help="排球检测模型路径")
    p.add_argument("--player_model", type=str, default="model/player_yolo.pt", help="选手检测模型路径")
    p.add_argument("--conf",         type=float, default=0.3,   help="置信度阈值")
    p.add_argument("--device",       type=str, default="0",     help="推理设备: 'cpu' 或 GPU id")
    return p.parse_args()

def init_models(ball_model_path, player_model_path, device):
    """
    加载两个模型：ball_model 用于排球检测，player_model 用于选手检测
    """
    ball_model   = YOLO(ball_model_path).to(device)
    player_model = YOLO(player_model_path).to(device)
    return ball_model, player_model

def init_bg_subtractor():
    """
    初始化 MOG2 背景分割器及形态学内核，用于运动前景提取
    """
    fgbg   = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    return fgbg, kernel

def detect_ball(frame, ball_model, fgmask, kernel, conf_thresh):
    """
    在单帧图像上检测排球，并做运动过滤。
    返回 list of tuples: (x1, y1, x2, y2, confidence)
    """
    # 推理只保留 class 0 (排球)
    res_ball = ball_model(frame, conf=conf_thresh, classes=[0])[0]
    ball_boxes = []
    for box in res_ball.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        # 对应 fgmask 区域
        roi = fgmask[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        motion_ratio = np.count_nonzero(roi) / roi.size
        if motion_ratio < 0.02:
            continue  # 过滤静止球
        conf_score = float(box.conf[0].cpu().numpy())
        ball_boxes.append((x1, y1, x2, y2, conf_score))
    return ball_boxes

def detect_player(frame, player_model, conf_thresh):
    """
    在单帧图像上检测选手（person），不做运动过滤。
    返回 list of tuples: (x1, y1, x2, y2, confidence)
    """
    # 推理只保留 class 0 (COCO person 对应的索引通常是 0)
    res_player = player_model(frame, conf=conf_thresh, classes=[0])[0]
    player_boxes = []
    for box in res_player.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf_score = float(box.conf[0].cpu().numpy())
        player_boxes.append((x1, y1, x2, y2, conf_score))
    return player_boxes

def draw_and_save(frame, ball_boxes, player_boxes, w, h, frame_idx, frame_dir, label_dir):
    """
    在图像上绘制两个通道的框，并保存帧图与对应的 YOLO 格式标签文件
    """
    # 保存原始帧
    frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.jpg")
    cv2.imwrite(frame_path, frame)

    txt_lines = []
    # 绘制排球 (class 0)
    for (x1, y1, x2, y2, conf) in ball_boxes:
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        txt_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"ball {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 绘制选手 (class 1)
    for (x1, y1, x2, y2, conf) in player_boxes:
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        txt_lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"player {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 保存标签
    label_path = os.path.join(label_dir, f"frame_{frame_idx:06d}.txt")
    with open(label_path, 'w') as f:
        f.write("\n".join(txt_lines))

def main():
    args = parse_args()

    # 创建输出目录结构
    base     = os.path.splitext(os.path.basename(args.input))[0]
    out_root = os.path.join(args.output_dir, base)
    os.makedirs(out_root, exist_ok=True)
    frame_dir = os.path.join(out_root, "frames")
    label_dir = os.path.join(out_root, "labels")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # 设备选择
    device = args.device
    if device.isdigit():
        device = f"cuda:{device}"

    # 加载模型
    ball_model, player_model = init_models(args.ball_model, args.player_model, device)

    # 打开视频并获取参数
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出视频 writer
    out_vid = os.path.join(out_root, f"{base}_output.avi")
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    writer  = cv2.VideoWriter(out_vid, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法打开输出视频: {out_vid}")

    # 初始化背景分割器（用于排球检测的运动过滤）
    fgbg, kernel = init_bg_subtractor()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) 运动前景分割
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 2) 排球检测并做运动过滤
        ball_boxes = detect_ball(frame, ball_model, fgmask, kernel, args.conf)

        # 3) 选手检测（不做运动过滤）
        player_boxes = detect_player(frame, player_model, args.conf)

        # 4) 绘制、保存帧与标签
        draw_and_save(frame, ball_boxes, player_boxes, w, h, frame_idx, frame_dir, label_dir)

        # 5) 写入输出视频
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Finished! 结果保存在: {out_root}")

if __name__ == "__main__":
    main()
