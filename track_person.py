import cv2
from ultralytics import YOLO
import os

# ---------- 參數設定 ----------
model = YOLO("yolo11s.pt")   # COCO person
person_class = [0]
conf_thresh  = 0.3

video_path = "input_video/test.mp4"
output_dir = "output_video"
os.makedirs(output_dir, exist_ok=True)  # 確保資料夾存在
output_path = os.path.join(output_dir, "person_demo.avi")

# ---------- 開啟影像流與 VideoWriter ----------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
if not out.isOpened():
    raise RuntimeError(f"無法開啟 VideoWriter: {output_path}")

# ---------- 主迴圈：偵測並寫入 ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=conf_thresh, classes=person_class)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf.cpu().numpy()[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"person {conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    out.write(frame)

# ---------- 收尾 ----------
cap.release()
out.release()
try:
    cv2.destroyAllWindows()
except:
    pass
print("Finished person detection demo →", output_path)

print("Finished person detection demo → person_demo.avi")
