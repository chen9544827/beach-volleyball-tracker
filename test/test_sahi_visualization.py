"""
SAHI Detection Visualization

生成三種配置的視覺化影片用於對比：
1. Standard-640
2. SAHI-512
3. Large-1280

Usage:
    python test/test_sahi_visualization.py \
        --video "D:/VScode/beach-volleyball-tracker/output_data/video_segments_test/Edmonton/normal_segments/segment_001_Team1.mp4" \
        --court-config court_config.json \
        --output-dir test_output/visualizations
"""

import os
import sys
import cv2
import time
import json
import argparse
import numpy as np
from pathlib import Path

# 添加專案根目錄到路徑
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

# 嘗試導入 SAHI
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    HAS_SAHI = True
except ImportError:
    HAS_SAHI = False
    print("[警告] SAHI 未安裝，將跳過 SAHI 測試")


def detect_players_standard(frame, model, conf=0.15, imgsz=640):
    """標準 YOLO 檢測"""
    results = model(frame, conf=conf, classes=[0], verbose=False, imgsz=imgsz, max_det=100, iou=0.45)

    detections = []
    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_score = float(box.conf[0].cpu().numpy())
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf_score,
                'area': (x2 - x1) * (y2 - y1)
            })

    return detections


def detect_players_sahi(frame, sahi_model, slice_size=512, overlap=0.2):
    """SAHI 切片檢測"""
    if not HAS_SAHI:
        return []

    result = get_sliced_prediction(
        frame,
        sahi_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap
    )

    detections = []
    for obj_pred in result.object_prediction_list:
        bbox = obj_pred.bbox
        detections.append({
            'bbox': [int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)],
            'confidence': float(obj_pred.score.value),
            'area': bbox.area
        })

    return detections


def filter_by_court(detections, court_boundary_np):
    """根據場地邊界過濾檢測"""
    if court_boundary_np is None:
        return detections

    inside_court = []
    for det in detections:
        bbox = det['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        if cv2.pointPolygonTest(court_boundary_np, (center_x, center_y), False) >= 0:
            inside_court.append(det)

    return inside_court


def draw_detections(frame, detections, inside_court, config_name, fps, frame_num):
    """在影片幀上繪製檢測結果"""
    vis_frame = frame.copy()

    # 繪製場外檢測（灰色）
    outside_dets = [d for d in detections if d not in inside_court]
    for det in outside_dets:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
        cv2.putText(vis_frame, f"{det['confidence']:.2f}",
                   (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    # 繪製場內檢測（綠色）
    for det in inside_court:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_frame, f"{det['confidence']:.2f}",
                   (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 顯示資訊
    info_y = 30
    cv2.putText(vis_frame, f"Config: {config_name}",
               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(vis_frame, f"Frame: {frame_num}",
               (10, info_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_frame, f"FPS: {fps:.1f}",
               (10, info_y+75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_frame, f"Detections: {len(detections)} (Inside: {len(inside_court)})",
               (10, info_y+110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis_frame


def process_video_with_config(video_path, config, model, sahi_model, court_boundary_np, output_path):
    """處理影片並保存視覺化結果"""
    print(f"\n[處理] 配置: {config['name']}")
    print("=" * 80)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[錯誤] 無法打開影片: {video_path}")
        return None

    # 影片屬性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 創建輸出影片
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"[錯誤] 無法創建輸出影片: {output_path}")
        cap.release()
        return None

    frame_count = 0
    total_detections = 0
    inside_court_detections = 0
    total_time = 0

    print(f"[資訊] 影片: {width}x{height}, {fps} fps, {total_frames} 幀")
    print(f"[資訊] 輸出: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 執行檢測
        start_time = time.time()

        if config['type'] == 'standard':
            detections = detect_players_standard(frame, model, imgsz=config['imgsz'])
        elif config['type'] == 'sahi':
            detections = detect_players_sahi(frame, sahi_model,
                                            slice_size=config['slice_size'],
                                            overlap=config['overlap'])
        else:
            detections = []

        inference_time = time.time() - start_time
        total_time += inference_time

        # 過濾場內檢測
        inside_court = filter_by_court(detections, court_boundary_np)

        total_detections += len(detections)
        inside_court_detections += len(inside_court)

        # 計算當前 FPS
        current_fps = (frame_count + 1) / total_time if total_time > 0 else 0

        # 繪製檢測結果
        vis_frame = draw_detections(frame, detections, inside_court,
                                   config['name'], current_fps, frame_count)

        # 寫入輸出影片
        out.write(vis_frame)

        frame_count += 1

        if frame_count % 50 == 0:
            print(f"[進度] {frame_count}/{total_frames} - "
                  f"檢測: {len(detections)} | 場內: {len(inside_court)} | "
                  f"時間: {inference_time:.3f}s")

    cap.release()
    out.release()

    # 計算統計
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_detections = total_detections / frame_count if frame_count > 0 else 0
    avg_inside = inside_court_detections / frame_count if frame_count > 0 else 0

    result = {
        'config_name': config['name'],
        'total_frames': frame_count,
        'total_detections': total_detections,
        'inside_court_detections': inside_court_detections,
        'avg_detections_per_frame': round(avg_detections, 2),
        'avg_inside_per_frame': round(avg_inside, 2),
        'total_time_sec': round(total_time, 2),
        'avg_fps': round(avg_fps, 2),
        'output_video': output_path
    }

    print(f"\n[完成] {config['name']}:")
    print(f"  - 平均 FPS: {avg_fps:.2f}")
    print(f"  - 平均檢測數/幀: {avg_detections:.2f}")
    print(f"  - 平均場內檢測/幀: {avg_inside:.2f}")
    print(f"  - 總時間: {total_time:.2f}s")
    print(f"  - 輸出: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="SAHI Detection Visualization")
    parser.add_argument("--video", required=True, help="測試影片路徑")
    parser.add_argument("--court-config", help="場地配置 JSON")
    parser.add_argument("--output-dir", default="test_output/visualizations", help="輸出目錄")
    args = parser.parse_args()

    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)

    # 載入場地配置
    court_boundary_np = None
    if args.court_config and os.path.exists(args.court_config):
        with open(args.court_config, 'r', encoding='utf-8') as f:
            court_config = json.load(f)
            if 'court_boundary_polygon' in court_config:
                court_boundary_np = np.array(court_config['court_boundary_polygon'], dtype=np.float32)
                print(f"[資訊] 已載入場地配置: {args.court_config}")

    # 載入模型
    model_path = PROJECT_ROOT / 'models' / 'yolo26m-pose.pt'
    print(f"[載入] 模型: {model_path}")
    model = YOLO(str(model_path))

    # 初始化 SAHI 模型
    sahi_model = None
    if HAS_SAHI:
        print("[載入] 初始化 SAHI 模型...")
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=str(model_path),
            confidence_threshold=0.15,
            device='cuda:0'
        )

    # 測試配置
    configs = [
        {'name': 'Standard-640', 'type': 'standard', 'imgsz': 640},
    ]

    if HAS_SAHI:
        configs.append({'name': 'SAHI-512', 'type': 'sahi', 'slice_size': 512, 'overlap': 0.2})

    configs.append({'name': 'Large-1280', 'type': 'standard', 'imgsz': 1280})

    # 生成輸出檔名
    video_name = Path(args.video).stem

    # 處理每個配置
    results = []
    for config in configs:
        output_filename = f"{video_name}_{config['name']}.mp4"
        output_path = os.path.join(args.output_dir, output_filename)

        result = process_video_with_config(
            args.video, config, model, sahi_model,
            court_boundary_np, output_path
        )

        if result:
            results.append(result)

    # 保存結果摘要
    summary_path = os.path.join(args.output_dir, f"{video_name}_visualization_summary.json")
    summary = {
        'video': args.video,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[完成] 視覺化結果已保存到: {args.output_dir}")
    print(f"[完成] 摘要: {summary_path}")

    # 顯示對比
    print("\n" + "=" * 80)
    print("[摘要]")
    print("=" * 80)
    for result in results:
        print(f"\n{result['config_name']}:")
        print(f"  - 平均場內檢測/幀: {result['avg_inside_per_frame']}")
        print(f"  - 平均 FPS: {result['avg_fps']}")
        print(f"  - 輸出影片: {result['output_video']}")


if __name__ == "__main__":
    main()
