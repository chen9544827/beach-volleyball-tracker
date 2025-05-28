import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker

# ---------- YOLO 检测结果 转 Norfair Detection ----------
def yolo_to_norfair(boxes):
    xywh    = boxes.xywh.cpu().numpy()    # [x_center, y_center, w, h]
    confs   = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    dets = []
    for (x, y, w, h), c, cls in zip(xywh, confs, classes):
        if cls == 32:  # COCO sports ball
            dets.append(Detection(np.array([x, y]), scores=np.array([c])))
    return dets


def main():
    parser = argparse.ArgumentParser(description="GPU 加速的沙滩排球跟踪与发球检测")
    parser.add_argument("--input",  type=str, required=True, help="输入视频路径")
    parser.add_argument("--output", type=str, default="output_video/ball_track_gpu.avi", help="输出视频路径")
    parser.add_argument("--model",  type=str, default="runs/train/exp0/weights/best.pt", help="自训练模型权重文件路径")
    parser.add_argument("--conf",   type=float, default=0.3, help="检测置信度阈值")
    parser.add_argument("--device", type=str, default="0", help="GPU 设备 id，例如 '0' 或 '0,1'")
    args = parser.parse_args()

    # 视频初始化
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.input}")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出文件 & 目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out    = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"无法打开 VideoWriter: {args.output}")

    # 加载 自训练 YOLO11 模型 并移动到 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    model = YOLO(args.model).to('cuda')

    # 初始化 Norfair 追踪器
    tracker = Tracker(distance_function="euclidean", distance_threshold=30)

    prev_has_ball = False
    serve_events  = []  # 记录发球帧序号
    frame_idx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) YOLO 推理 on GPU
        results = model(frame, conf=args.conf, classes=[32])[0]

        # 2) 绘制检测框
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf.cpu().numpy()[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ball {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # 3) Norfair 更新
        detections = yolo_to_norfair(results.boxes)
        tracks     = tracker.update(detections)
        has_ball   = len(tracks) > 0

        # 4) 发球检测
        if not prev_has_ball and has_ball:
            serve_frame = frame_idx
            serve_events.append(serve_frame)
            cv2.putText(frame, f"Serve frame: {serve_frame}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        prev_has_ball = has_ball

        # 5) 绘制轨迹 & ID
        for track in tracks:
            est = np.array(track.estimate).flatten()
            if est.size < 2:
                continue
            x, y = int(est[0]), int(est[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(track.id), (x+5, y+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            # 绘制历史轨迹
            history = [np.array(h).flatten() for h in track.estimate_history]
            for i in range(1, len(history)):
                p1 = tuple(history[i-1][:2].astype(int))
                p2 = tuple(history[i][:2].astype(int))
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

        # 写入输出
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("Tracking complete. Serve frames:", serve_events)

if __name__ == "__main__":
    main()
