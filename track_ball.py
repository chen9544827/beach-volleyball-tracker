import os
import cv2
import argparse
from ultralytics import YOLO
import numpy as np

# ---------- 检测含背景运动过滤的排球检测脚本 ----------
# 本脚本使用 BackgroundSubtractorMOG2 提取运动前景：
# 1. 学习背景模型，生成前景掩码 fgmask。
# 2. 对 fgmask 进行形态学去噪。
# 3. 对每个检测框裁剪 fgmask 区域 roi_mask。
# 4. 计算运动像素比例，过滤静止或伪检的球。
def main():
    parser = argparse.ArgumentParser(description="排球检测，静止球过滤，按视频创建输出文件夹")
    parser.add_argument("--input",      type=str, required=True, help="输入视频路径")
    parser.add_argument("--output_dir", type=str, default="output_video", help="输出父目录")
    parser.add_argument("--model",      type=str, default="model/best.pt", help="模型权重路径")
    parser.add_argument("--conf",       type=float, default=0.3, help="检测置信度阈值")
    parser.add_argument("--device",     type=str, default="0", help="GPU 设备 ID，示例 '0'")
    args = parser.parse_args()

    # 创建每个视频独立的输出文件夹
    base = os.path.splitext(os.path.basename(args.input))[0]
    video_out_dir = os.path.join(args.output_dir, base)
    os.makedirs(video_out_dir, exist_ok=True)

    # 路径配置
    output_video = os.path.join(video_out_dir, f"{base}_output.avi")
    frames_dir   = os.path.join(video_out_dir, "frames")
    labels_dir   = os.path.join(video_out_dir, "labels")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.input}")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化背景分割
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"无法打开 VideoWriter: {output_video}")

    # 加载模型到 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    model = YOLO(args.model).to('cuda')

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 运动前景分割 & 去噪
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 模型检测（volleyball 类 id=0）
        results = model(frame, conf=args.conf, classes=[0])[0]

        txt_lines = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            # 前景掩码区域
            roi_mask = fgmask[y1:y2, x1:x2]
            if roi_mask.size == 0:
                continue
            motion_ratio = np.count_nonzero(roi_mask) / float(roi_mask.size)
            if motion_ratio < 0.02:
                continue  # 过滤静止球

            # 绘制检测框
            conf_score = float(box.conf[0].cpu().numpy())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ball {conf_score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 生成 YOLO 标签
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            w  = (x2 - x1) / width
            h  = (y2 - y1) / height
            txt_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # 写视频帧
        out.write(frame)
        # 保存原始帧
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)

        # 保存标签文件
        label_path = os.path.join(labels_dir, f"frame_{frame_idx:06d}.txt")
        with open(label_path, 'w') as f:
            f.write("\n".join(txt_lines))

        frame_idx += 1

    cap.release()
    out.release()
    print(f"检测完成，输出文件夹: {video_out_dir}")

if __name__ == "__main__":
    main()
