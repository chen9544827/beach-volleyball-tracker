# tools/convert_cvat_to_tracknet.py
# -*- coding: utf-8 -*-
"""
將 CVAT 匯出的 XML 轉換為 vball-net 訓練用 CSV + PNG 幀

vball-net 訓練資料格式:
    data/frames/<mode>/<track_id>/        ← 512x288 PNG 幀（0.png, 1.png, ...）
    data/frames/<mode>/<track_id>_ball.csv ← 座標已縮放至 512x288 空間

用法:
    # 基本轉換（輸出 1280x720 座標 CSV）
    python tools/convert_cvat_to_tracknet.py \\
        --xml output/cvat_v1b/segment_028_Team1_cvat.xml \\
        --output output/vball_dataset

    # 完整 vball-net 訓練資料（縮放座標 + 提取 PNG 幀）
    python tools/convert_cvat_to_tracknet.py \\
        --xml output/cvat_v1b/segment_028_Team1_cvat.xml \\
        --video output_data/test_segment/segment_028_Team1.mp4 \\
        --output output/vball_dataset \\
        --scale-to-512 --extract-frames

    # 批次處理整個目錄
    python tools/convert_cvat_to_tracknet.py \\
        --xml-dir output/cvat_v1b \\
        --video-dir output_data/test_segment \\
        --output output/vball_dataset \\
        --scale-to-512 --extract-frames

座標縮放說明:
    CVAT 標註在原始影片解析度（如 1280x720）
    vball-net 訓練空間為 512x288
    scale_x = 512 / video_width   (e.g., 512/1280 = 0.4)
    scale_y = 288 / video_height  (e.g., 288/720  = 0.4)
"""

import os
import sys
import argparse
import csv
import glob
import xml.etree.ElementTree as ET

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# vball-net 訓練用推理解析度
TRAIN_WIDTH = 512
TRAIN_HEIGHT = 288


def parse_cvat_xml(xml_path: str):
    """
    解析 CVAT for Video 1.1 XML，提取 volleyball track 的點標註。

    Returns:
        dict: {frame_id: (x, y)} 有球的幀
        int: 總幀數
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 取得總幀數（從 meta/task/size）
    total_frames = None
    size_elem = root.find('./meta/task/size')
    if size_elem is not None:
        total_frames = int(size_elem.text)

    ball_positions = {}  # frame_id -> (x, y)

    for track in root.findall('track'):
        label = track.get('label', '')
        if label != 'volleyball':
            continue

        for points_elem in track.findall('points'):
            frame_id = int(points_elem.get('frame', 0))
            outside = int(points_elem.get('outside', 0))

            if outside == 1:
                continue  # 無球幀跳過

            pts_str = points_elem.get('points', '')
            if not pts_str:
                continue

            try:
                x_str, y_str = pts_str.split(',')
                x = float(x_str)
                y = float(y_str)
                ball_positions[frame_id] = (x, y)
            except (ValueError, IndexError):
                continue

    return ball_positions, total_frames


def generate_gaussian_heatmap(width: int, height: int, cx: float, cy: float,
                               sigma: float = 5.0) -> np.ndarray:
    """
    生成以 (cx, cy) 為中心的 Gaussian Heatmap（0-255 uint8）。
    """
    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    heatmap = (heatmap * 255).clip(0, 255).astype(np.uint8)
    return heatmap


def extract_frames_512(video_path: str, output_dir: str, total_frames: int = None) -> int:
    """
    從影片提取全部幀並縮放至 512x288 PNG
    輸出到 output_dir/<frame_id>.png（0-indexed）

    Returns:
        實際提取的幀數
    """
    import cv2
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if total_frames and count >= total_frames:
            break
        resized = cv2.resize(frame, (TRAIN_WIDTH, TRAIN_HEIGHT))
        cv2.imwrite(os.path.join(output_dir, f"{count}.png"), resized)
        count += 1
    cap.release()
    return count


def convert(xml_path: str, output_dir: str, video_path: str = None,
            sigma: float = 5.0, generate_heatmaps: bool = False,
            total_frames_override: int = None,
            scale_to_512: bool = False,
            extract_frames: bool = False):
    """
    CVAT XML -> vball-net 訓練用 CSV（+ 可選 PNG 幀提取）

    Args:
        xml_path: CVAT 匯出 XML 路徑
        output_dir: 輸出根目錄
        video_path: 影片路徑（scale_to_512 或 extract_frames 時需要）
        sigma: Gaussian sigma（像素，僅 generate_heatmaps 時使用）
        generate_heatmaps: 是否生成舊版 Heatmap（通常不需要，vball-net 在訓練中自動生成）
        total_frames_override: 手動指定總幀數
        scale_to_512: 是否將座標從原始解析度縮放到 512x288 訓練空間
        extract_frames: 是否提取 512x288 PNG 幀（vball-net 訓練所需）
    """
    ball_positions, total_frames_xml = parse_cvat_xml(xml_path)

    total_frames = total_frames_override or total_frames_xml
    img_w, img_h = 1280, 720

    # 取得影片資訊
    if video_path and os.path.exists(video_path):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if vid_frames > 0 and total_frames is None:
                total_frames = vid_frames
        except Exception:
            pass

    if total_frames is None:
        if ball_positions:
            total_frames = max(ball_positions.keys()) + 1
        else:
            print("[ERROR] 無法取得總幀數，請用 --total-frames 指定")
            return None

    # 座標縮放係數（1280x720 -> 512x288）
    scale_x = TRAIN_WIDTH / img_w if scale_to_512 else 1.0
    scale_y = TRAIN_HEIGHT / img_h if scale_to_512 else 1.0

    # 輸出目錄（vball-net 格式：frames/<mode>/<track_id>/）
    video_name = os.path.splitext(os.path.basename(xml_path))[0]
    video_name = video_name.replace('_cvat', '').replace('_corrected', '')
    dataset_dir = os.path.join(output_dir, video_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # 提取 512x288 PNG 幀
    if extract_frames:
        if not video_path or not os.path.exists(video_path):
            print("[WARNING] --extract-frames 需要 --video，已跳過幀提取")
        else:
            print(f"  提取 512x288 PNG 幀中...")
            n = extract_frames_512(video_path, dataset_dir, total_frames)
            print(f"  [OK] 提取 {n} 幀 -> {dataset_dir}/")

    # 生成 CSV（vball-net 格式: <track_id>_ball.csv 在同層目錄）
    csv_path = os.path.join(output_dir, f"{video_name}_ball.csv")
    rows = []
    count_vis1 = 0
    count_vis0 = 0

    for frame_id in range(total_frames):
        if frame_id in ball_positions:
            cx, cy = ball_positions[frame_id]
            # 縮放到訓練空間
            tx = int(round(cx * scale_x))
            ty = int(round(cy * scale_y))
            # 邊界保護
            tx = max(0, min(tx, TRAIN_WIDTH - 1 if scale_to_512 else img_w - 1))
            ty = max(0, min(ty, TRAIN_HEIGHT - 1 if scale_to_512 else img_h - 1))
            rows.append({'Frame': frame_id, 'Visibility': 1, 'X': tx, 'Y': ty})
            count_vis1 += 1

            if generate_heatmaps:
                import cv2
                hw = TRAIN_WIDTH if scale_to_512 else img_w
                hh = TRAIN_HEIGHT if scale_to_512 else img_h
                hm = generate_gaussian_heatmap(hw, hh, tx, ty, sigma)
                hm_dir = os.path.join(dataset_dir, 'heatmaps')
                os.makedirs(hm_dir, exist_ok=True)
                cv2.imwrite(os.path.join(hm_dir, f"{frame_id:06d}.png"), hm)
        else:
            rows.append({'Frame': frame_id, 'Visibility': 0, 'X': 0, 'Y': 0})
            count_vis0 += 1

            if generate_heatmaps:
                import cv2
                hw = TRAIN_WIDTH if scale_to_512 else img_w
                hh = TRAIN_HEIGHT if scale_to_512 else img_h
                hm = np.zeros((hh, hw), dtype=np.uint8)
                hm_dir = os.path.join(dataset_dir, 'heatmaps')
                os.makedirs(hm_dir, exist_ok=True)
                cv2.imwrite(os.path.join(hm_dir, f"{frame_id:06d}.png"), hm)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Frame', 'Visibility', 'X', 'Y'])
        writer.writeheader()
        writer.writerows(rows)

    det_rate = 100 * count_vis1 / total_frames if total_frames > 0 else 0
    coord_info = f"512x288 空間" if scale_to_512 else f"原始 {img_w}x{img_h} 空間"
    print(f"[OK] {video_name}")
    print(f"     ball.csv: {csv_path}  ({coord_info})")
    print(f"     總幀數={total_frames}, 有球={count_vis1}({det_rate:.1f}%), 無球={count_vis0}({100-det_rate:.1f}%)")

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="CVAT XML -> vball-net 訓練用 CSV（含座標縮放 + 幀提取）"
    )
    # 輸入：單一或批次
    parser.add_argument("--xml", type=str, default=None,
                        help="單一 CVAT 匯出 XML 路徑")
    parser.add_argument("--xml-dir", type=str, default=None,
                        help="含多個 _cvat.xml 的目錄（批次處理）")
    parser.add_argument("--video", type=str, default=None,
                        help="對應影片路徑（--xml 模式）")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="影片目錄（--xml-dir 批次模式，自動配對）")
    # 輸出
    parser.add_argument("--output", type=str, default="output/vball_dataset",
                        help="輸出根目錄（預設 output/vball_dataset）")
    # 處理選項
    parser.add_argument("--scale-to-512", action="store_true",
                        help="座標縮放至 512x288 訓練空間（vball-net 訓練必須）")
    parser.add_argument("--extract-frames", action="store_true",
                        help="提取 512x288 PNG 幀（vball-net 訓練必須）")
    parser.add_argument("--sigma", type=float, default=5.0,
                        help="Gaussian sigma（像素，預設 5.0）")
    parser.add_argument("--generate-heatmaps", action="store_true",
                        help="生成 Gaussian Heatmap PNG（通常不需要，vball-net 自動生成）")
    parser.add_argument("--total-frames", type=int, default=None,
                        help="手動指定影片總幀數（選填）")
    args = parser.parse_args()

    if args.xml:
        # 單一模式
        convert(
            xml_path=args.xml,
            output_dir=args.output,
            video_path=args.video,
            sigma=args.sigma,
            generate_heatmaps=args.generate_heatmaps,
            total_frames_override=args.total_frames,
            scale_to_512=args.scale_to_512,
            extract_frames=args.extract_frames,
        )
    elif args.xml_dir:
        # 批次模式
        xml_files = sorted(glob.glob(os.path.join(args.xml_dir, '*_cvat.xml')))
        if not xml_files:
            print(f"[ERROR] 找不到 _cvat.xml: {args.xml_dir}")
            return
        print(f"找到 {len(xml_files)} 個 XML 檔案\n")
        for xml_path in xml_files:
            seg_name = os.path.splitext(os.path.basename(xml_path))[0]
            seg_name = seg_name.replace('_cvat', '')
            video_path = None
            if args.video_dir:
                for ext in ['.mp4', '.avi', '.mov']:
                    candidate = os.path.join(args.video_dir, f"{seg_name}{ext}")
                    if os.path.exists(candidate):
                        video_path = candidate
                        break
            convert(
                xml_path=xml_path,
                output_dir=args.output,
                video_path=video_path,
                sigma=args.sigma,
                generate_heatmaps=args.generate_heatmaps,
                total_frames_override=args.total_frames,
                scale_to_512=args.scale_to_512,
                extract_frames=args.extract_frames,
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
