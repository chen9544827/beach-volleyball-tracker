import os
import cv2
import argparse
from ultralytics import YOLO
import numpy as np

# ---------- 简单的排球检测示例（同时输出视频、逐帧原始图像及YOLO格式标签） ----------
def main():
    parser = argparse.ArgumentParser(description="使用自训练 YOLO11 模型进行排球检测，输出视频、逐帧原始图像及标签文件")
    parser.add_argument("--input",      type=str, required=True, help="输入视频路径")
    parser.add_argument("--output_dir", type=str, default="output_video", help="输出目录")
    parser.add_argument("--model",      type=str, default="runs/detect/train7/weights/best.pt", help="模型权重文件路径")
    parser.add_argument("--conf",       type=float, default=0.3, help="检测置信度阈值")
    parser.add_argument("--device",     type=str, default="0", help="GPU 设备 ID，例如 '0'")
    args = parser.parse_args()

    # 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    output_video = os.path.join(args.output_dir, f"{base}_output.avi")
    frames_dir   = os.path.join(args.output_dir, f"{base}_frames")
    labels_dir   = os.path.join(args.output_dir, f"{base}_labels")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.input}")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"无法打开 VideoWriter: {output_video}")

    # 设置 GPU 并加载模型
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    model = YOLO(args.model).to('cuda')

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 复制原始帧用于保存
        original_frame = frame.copy()

        # 只检测 volleyball 类 (class_id=0)
        results = model(frame, conf=args.conf, classes=[0])[0]

        # YOLO txt 标签内容
        txt_lines = []

        # 在 frame 上绘制检测框并生成标签行
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf_score = float(box.conf[0].cpu().numpy())
            # 画框在 frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ball {conf_score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # 生成归一化标签
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            w  = (x2 - x1) / width
            h  = (y2 - y1) / height
            txt_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # 写入带标记的视频帧
        out.write(frame)
        # 保存原始未标记帧到图像文件
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, original_frame)

        # 保存标签文件
        label_path = os.path.join(labels_dir, f"frame_{frame_idx:06d}.txt")
        with open(label_path, 'w') as f:
            f.write("\n".join(txt_lines))

        frame_idx += 1

    cap.release()
    out.release()
    print(f"检测完成，输出视频: {output_video}")
    print(f"帧原始图像已保存至: {frames_dir}")
    print(f"标签文件已保存至: {labels_dir}")

if __name__ == "__main__":
    main()
