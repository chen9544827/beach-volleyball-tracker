#!/usr/bin/env python3
import os, cv2, argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True, help="输入视频")
    p.add_argument("--output_dir", default="output_video")
    p.add_argument("--model",      default="runs/detect/ball_player_schemeB/weights/best.pt", help="best.pt 路径")
    p.add_argument("--conf",       type=float, default=0.2, help="置信度阈值（建议 0.1–0.3）")
    p.add_argument("--device",     default="0",  help="GPU ID 或 'cpu'")
    p.add_argument("--mode",       choices=["ball","player","both"], default="both")
    return p.parse_args()

def main():
    args = parse_args()
    # 输出
    base = os.path.splitext(os.path.basename(args.input))[0]
    out = os.path.join(args.output_dir, base); os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w,h = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(os.path.join(out, f"{base}_{args.mode}.avi"),
                             cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    # 加载模型
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    model  = YOLO(args.model).to(device)
    names  = model.names  # e.g. {0:'ball',1:'player'}

    # 根据 mode 设置 classes 过滤；both 时不传让它输出所有
    if args.mode == "ball":
        cls_filter = [i for i,n in names.items() if n=="ball"]
    elif args.mode == "player":
        cls_filter = [i for i,n in names.items() if n=="player"]
    else:
        cls_filter = None

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 推理
        if cls_filter is None:
            res = model(frame, conf=args.conf)[0]
        else:
            res = model(frame, conf=args.conf, classes=cls_filter)[0]

        # 遍历所有检测
        for box in res.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
            label = f"{names[cls]} {conf:.2f}"
            color = (0,0,255) if names[cls]=="ball" else (0,255,0)
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            cv2.putText(frame,label,(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print("Done:", out)

if __name__=="__main__":
    main()
