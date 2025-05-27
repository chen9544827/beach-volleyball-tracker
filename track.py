import cv2
from ultralytics import YOLO
from norfair import Detection, Tracker
import numpy as np
import os

# ---------- YOLOv8 模型载入（GPU） ----------
model = YOLO("yolo11s.pt").to("cuda")

# ---------- 影片路径与输出路径 ----------
video_path = "input_video/test.mp4"
os.makedirs("output_video", exist_ok=True)
output_path = "output_video/tracked_output.avi"

# ---------- 打开 VideoCapture 与 VideoWriter ----------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"无法打开输入影片：{video_path}")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not out.isOpened():
    raise RuntimeError(f"无法打开 VideoWriter：{output_path}")

# ---------- Norfair 初始化 ----------
def yolo_to_norfair(boxes):
    xywh    = boxes.xywh.cpu().numpy()
    confs   = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    dets = []
    for (x, y, w, h), c, cls in zip(xywh, confs, classes):
        if cls == 32:  # COCO 中的 sports ball
            dets.append(Detection(np.array([x, y]), scores=np.array([c])))
    return dets

tracker = Tracker(distance_function="euclidean", distance_threshold=30)

# 手动维护每个 track.id 的轨迹历史
histories = {}  # { track_id: [ [x1,y1], [x2,y2], ... ] }

# ---------- 主循环 ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) 检测
    results = model(frame, device="0", imgsz=960, conf=0.1, iou=0.3, classes=[32])[0]

    # 2) 绘制检测框
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf.cpu().numpy()[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 3) Norfair 追踪
    detections = yolo_to_norfair(results.boxes)
    tracks     = tracker.update(detections)

    # 假设我们已经得到了 Norfair 的 tracks 列表
    for track in tracks:
        # 将 estimate 转成一维 numpy 数组
        est = np.array(track.estimate).flatten()
        # 如果确实至少有两个值，才解包
        if est.size < 2:
            continue
        x, y = est[0], est[1]

        # 绘制追踪点和 ID
        cv2.circle(frame, (int(x), int(y)), 4, (0,255,0), -1)
        cv2.putText(frame, str(track.id), (int(x)+5, int(y)+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # 绘制历史轨迹（假设 histories 是你自己维护的字典）
        pts = histories.setdefault(track.id, [])
        pts.append([x, y])
        for i in range(1, len(pts)):
            p1 = (int(pts[i-1][0]), int(pts[i-1][1]))
            p2 = (int(pts[i][0]),   int(pts[i][1]))
            cv2.line(frame, p1, p2, (255,0,0), 2)

    # 5) 写入输出视频
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("输出完成：", output_path)
