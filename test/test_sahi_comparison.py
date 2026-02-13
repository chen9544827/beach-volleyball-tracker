"""
SAHI vs Standard Detection Comparison Test

對比三種檢測配置：
1. Standard (imgsz=640) - 快速但可能漏小目標
2. SAHI (slice=512, overlap=0.2) - 切片推論
3. Large imgsz (1280) - 較大輸入尺寸

Usage:
    python test/test_sahi_comparison.py --video input_video/analyze_serve/segment_011.mp4 --court-config court_config.json
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


def main():
    parser = argparse.ArgumentParser(description="SAHI vs Standard Detection Comparison")
    parser.add_argument("--video", required=True, help="測試影片路徑")
    parser.add_argument("--court-config", help="場地配置 JSON")
    parser.add_argument("--output", default="test_output/sahi_comparison_results.json", help="輸出 JSON 路徑")
    parser.add_argument("--sample-frames", type=int, default=100, help="採樣幀數（預設 100）")
    args = parser.parse_args()

    # 載入場地配置
    court_boundary_np = None
    if args.court_config and os.path.exists(args.court_config):
        with open(args.court_config, 'r', encoding='utf-8') as f:
            court_config = json.load(f)
            if 'court_boundary_polygon' in court_config:
                court_boundary_np = np.array(court_config['court_boundary_polygon'], dtype=np.float32)

    # 載入模型
    model_path = PROJECT_ROOT / 'models' / 'yolo26m-pose.pt'
    print(f"[測試] 載入模型: {model_path}")
    model = YOLO(str(model_path))

    # 初始化 SAHI 模型
    sahi_model = None
    if HAS_SAHI:
        print("[測試] 初始化 SAHI 模型...")
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=str(model_path),
            confidence_threshold=0.15,
            device='cuda:0'
        )

    # 打開影片
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[錯誤] 無法打開影片: {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[測試] 影片資訊:")
    print(f"  - 總幀數: {total_frames}")
    print(f"  - FPS: {fps}")
    print(f"  - 解析度: {width}x{height}")

    # 採樣策略：均勻採樣
    sample_interval = max(1, total_frames // args.sample_frames)
    print(f"[測試] 採樣策略: 每 {sample_interval} 幀取一幀")

    # 測試配置
    configs = [
        {'name': 'Standard-640', 'type': 'standard', 'imgsz': 640},
    ]

    if HAS_SAHI:
        configs.append({'name': 'SAHI-512', 'type': 'sahi', 'slice_size': 512, 'overlap': 0.2})

    configs.append({'name': 'Large-1280', 'type': 'standard', 'imgsz': 1280})

    # 執行測試
    results = {}
    for config in configs:
        print(f"\n[測試] 配置: {config['name']}")
        print("=" * 80)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到開頭

        frame_count = 0
        sample_count = 0
        total_detections = 0
        inside_court_detections = 0
        total_time = 0
        inference_times = []

        while sample_count < args.sample_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_interval != 0:
                frame_count += 1
                continue

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
            inference_times.append(inference_time)
            total_time += inference_time

            # 過濾場內檢測
            inside_court = filter_by_court(detections, court_boundary_np)

            total_detections += len(detections)
            inside_court_detections += len(inside_court)

            sample_count += 1
            frame_count += 1

            if sample_count % 10 == 0:
                print(f"[進度] {sample_count}/{args.sample_frames} - "
                      f"檢測: {len(detections)} | 場內: {len(inside_court)} | "
                      f"時間: {inference_time:.3f}s")

        # 計算統計
        avg_fps = sample_count / total_time if total_time > 0 else 0
        avg_detections = total_detections / sample_count if sample_count > 0 else 0
        avg_inside = inside_court_detections / sample_count if sample_count > 0 else 0

        results[config['name']] = {
            'sample_frames': sample_count,
            'total_detections': total_detections,
            'inside_court_detections': inside_court_detections,
            'avg_detections_per_frame': round(avg_detections, 2),
            'avg_inside_per_frame': round(avg_inside, 2),
            'total_time_sec': round(total_time, 2),
            'avg_fps': round(avg_fps, 2),
            'avg_inference_ms': round(np.mean(inference_times) * 1000, 1),
            'std_inference_ms': round(np.std(inference_times) * 1000, 1)
        }

        print(f"\n[結果] {config['name']}:")
        print(f"  - 平均 FPS: {avg_fps:.2f}")
        print(f"  - 平均檢測數/幀: {avg_detections:.2f}")
        print(f"  - 平均場內檢測/幀: {avg_inside:.2f}")
        print(f"  - 總時間: {total_time:.2f}s")

    cap.release()

    # 保存結果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data = {
        'video': args.video,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'yolo26m-pose.pt',
        'video_info': {
            'total_frames': total_frames,
            'fps': fps,
            'resolution': f"{width}x{height}"
        },
        'results': results
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n[完成] 結果已保存: {args.output}")

    # 對比分析
    print("\n" + "=" * 80)
    print("[對比分析]")
    print("=" * 80)

    if len(results) >= 2:
        baseline = results['Standard-640']

        for name, result in results.items():
            if name == 'Standard-640':
                continue

            det_diff = result['avg_inside_per_frame'] - baseline['avg_inside_per_frame']
            det_pct = (det_diff / baseline['avg_inside_per_frame'] * 100) if baseline['avg_inside_per_frame'] > 0 else 0
            speed_ratio = result['avg_fps'] / baseline['avg_fps'] if baseline['avg_fps'] > 0 else 0

            print(f"\n{name} vs Standard-640:")
            print(f"  檢測差異: {det_diff:+.2f} ({det_pct:+.1f}%)")
            print(f"  速度比: {speed_ratio:.2f}x")

            # 決策建議
            if det_pct > 30 and result['avg_fps'] > 5:
                print(f"  ✅ 建議採用 {name}（檢測提升 > 30% 且速度 > 5 fps）")
            elif det_pct > 30:
                print(f"  ⚠️ {name} 檢測提升足夠但速度較慢（{result['avg_fps']:.2f} fps）")
            elif result['avg_fps'] > baseline['avg_fps']:
                print(f"  ⚠️ {name} 速度較快但檢測提升不足（{det_pct:+.1f}%）")
            else:
                print(f"  ❌ 不建議採用 {name}")


if __name__ == "__main__":
    main()
