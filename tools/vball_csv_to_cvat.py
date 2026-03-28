# tools/vball_csv_to_cvat.py
# -*- coding: utf-8 -*-
"""
將 VballNet / V1b 的 _predict_ball.csv 轉換為 CVAT XML 預標註

比 export_to_cvat.py 更適合用於 fine-tuning 資料集準備：
- 直接讀取 V1b CSV（偵測率 ~62%，比 YOLO tracking JSON 更準確）
- 座標已是原始影片解析度（1280x720）

用法:
    # 單一 CSV
    python tools/vball_csv_to_cvat.py \
        --csv output/vball_v1b_csv/segment_001_Team1_predict_ball.csv \
        --video output_data/test_segment/segment_001_Team1.mp4 \
        --output output/cvat_v1b

    # 批次處理整個目錄
    python tools/vball_csv_to_cvat.py \
        --csv-dir output/vball_v1b_csv \
        --video-dir output_data/test_segment \
        --output output/cvat_v1b

CVAT 匯入步驟:
    1. 到 app.cvat.ai 建立 Project → 新增 Label "volleyball"（type: points）
    2. 建立 Task → 上傳對應 MP4 影片
    3. Actions -> Upload annotations -> Format: "CVAT 1.1" -> 選本工具輸出的 XML
    4. 標註人員逐幀修正:
       - 有球但沒點 -> 點擊補標
       - 點的位置偏移 -> 拖移修正
       - 無球但有點 -> 右鍵 Delete 或設 outside
    5. Actions -> Export annotations -> "CVAT for video 1.1"
    6. python tools/convert_cvat_to_tracknet.py --xml <匯出的 xml> \
           --video <對應影片> --output output/vball_dataset --scale-to-512
"""

import os
import sys
import argparse
import csv
import glob
import xml.etree.ElementTree as ET
import xml.dom.minidom

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_vball_csv(csv_path: str):
    """
    讀取 VballNet _predict_ball.csv

    Returns:
        dict: {frame_id: (x, y)} 有球的幀
        int: 總幀數
    """
    detections = {}
    total_frames = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = int(row['Frame'])
            visibility = int(row['Visibility'])
            total_frames = max(total_frames, frame_id + 1)
            if visibility == 1:
                x = float(row['X'])
                y = float(row['Y'])
                detections[frame_id] = (x, y)
    return detections, total_frames


def get_video_info(video_path: str):
    """取得影片總幀數和解析度"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return total, w, h
    except Exception:
        return None, 1280, 720


def csv_to_cvat_xml(csv_path: str, output_dir: str, video_path: str = None,
                    size_override: int = None) -> str:
    """
    V1b _predict_ball.csv -> CVAT XML Points Track

    Args:
        csv_path: VballNet predict_ball.csv 路徑
        output_dir: 輸出目錄
        video_path: 對應影片（用於取得解析度）
        size_override: 手動指定幀數（以 CVAT 顯示的幀數為準，解決幀數落差問題）

    Returns:
        輸出的 XML 路徑
    """
    detections, csv_frames = load_vball_csv(csv_path)

    img_w, img_h = 1280, 720
    if video_path and os.path.exists(video_path):
        _, img_w, img_h = get_video_info(video_path)  # 只取解析度，不取幀數

    # 總幀數優先順序：手動指定 > CSV 幀數
    # 注意：不同工具（OpenCV / VballNet / CVAT）計算幀數可能差 1，以 CVAT 顯示的為準
    if size_override is not None:
        total_frames = size_override
    else:
        total_frames = csv_frames
        print(f"  [提示] 若 CVAT 顯示幀數與此不同（{total_frames}），"
              f"請用 --size {total_frames-1} 或 --size {total_frames+1} 重新生成")

    # segment 名稱（去掉 _predict_ball 後綴）
    seg_name = os.path.splitext(os.path.basename(csv_path))[0]
    seg_name = seg_name.replace('_predict_ball', '')

    # 建立 CVAT XML
    root = ET.Element('annotations')
    ET.SubElement(root, 'version').text = '1.1'

    meta = ET.SubElement(root, 'meta')
    task = ET.SubElement(meta, 'task')
    ET.SubElement(task, 'name').text = seg_name
    ET.SubElement(task, 'size').text = str(total_frames)
    ET.SubElement(task, 'mode').text = 'interpolation'
    labels = ET.SubElement(task, 'labels')
    label = ET.SubElement(labels, 'label')
    ET.SubElement(label, 'name').text = 'volleyball'
    ET.SubElement(label, 'color').text = '#ffff00'
    attrs = ET.SubElement(label, 'attributes')
    attr = ET.SubElement(attrs, 'attribute')
    ET.SubElement(attr, 'name').text = 'needs_review'
    ET.SubElement(attr, 'mutable').text = 'True'
    ET.SubElement(attr, 'input_type').text = 'checkbox'
    ET.SubElement(attr, 'default_value').text = 'false'

    # 整條影片的球 track
    track = ET.SubElement(root, 'track')
    track.set('id', '0')
    track.set('label', 'volleyball')
    track.set('source', 'auto')

    count_detected = 0
    count_missing = 0

    for frame_id in range(total_frames):
        pts_elem = ET.SubElement(track, 'points')
        pts_elem.set('frame', str(frame_id))
        pts_elem.set('keyframe', '1')
        pts_elem.set('z_order', '0')
        pts_elem.set('occluded', '0')

        if frame_id in detections:
            x, y = detections[frame_id]
            pts_elem.set('outside', '0')
            pts_elem.set('points', f"{x:.2f},{y:.2f}")
            attr_elem = ET.SubElement(pts_elem, 'attribute')
            attr_elem.set('name', 'needs_review')
            attr_elem.text = 'false'
            count_detected += 1
        else:
            # 無偵測幀：outside=1（CVAT 不顯示點），needs_review=true 提醒標註員確認
            pts_elem.set('outside', '1')
            pts_elem.set('points', f"{img_w//2:.2f},{img_h//2:.2f}")
            attr_elem = ET.SubElement(pts_elem, 'attribute')
            attr_elem.set('name', 'needs_review')
            attr_elem.text = 'true'
            count_missing += 1

    # 格式化輸出
    xml_str = xml.dom.minidom.parseString(
        ET.tostring(root, encoding='unicode')
    ).toprettyxml(indent='  ', encoding=None)
    lines = xml_str.split('\n')
    if lines[0].startswith('<?xml'):
        lines[0] = '<?xml version="1.0" encoding="utf-8"?>'
    xml_str = '\n'.join(lines)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{seg_name}_cvat.xml")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)

    print(f"[OK] {seg_name}")
    print(f"     總幀數: {total_frames}  |  "
          f"有球(已預標): {count_detected} ({100*count_detected/total_frames:.1f}%)  |  "
          f"無球(needs_review): {count_missing} ({100*count_missing/total_frames:.1f}%)")
    print(f"     -> {out_path}")
    print()

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='VballNet predict_ball.csv -> CVAT XML 預標註'
    )
    # 單一模式
    parser.add_argument('--csv', type=str, default=None,
                        help='單一 _predict_ball.csv 路徑')
    parser.add_argument('--video', type=str, default=None,
                        help='對應影片路徑（取得精確總幀數用）')
    # 批次模式
    parser.add_argument('--csv-dir', type=str, default=None,
                        help='含多個 _predict_ball.csv 的目錄')
    parser.add_argument('--video-dir', type=str, default=None,
                        help='影片目錄（檔名與 CSV 前綴對應）')
    # 輸出
    parser.add_argument('--output', type=str, default='output/cvat_v1b',
                        help='輸出目錄（預設 output/cvat_v1b）')
    # 幀數修正
    parser.add_argument('--size', type=int, default=None,
                        help='手動指定總幀數（以 CVAT 顯示的幀數為準，解決幀數落差）')
    args = parser.parse_args()

    if args.csv:
        # 單一模式
        csv_to_cvat_xml(args.csv, args.output, args.video, size_override=args.size)

    elif args.csv_dir:
        # 批次模式
        csv_files = sorted(glob.glob(
            os.path.join(args.csv_dir, '*_predict_ball.csv')
        ))
        if not csv_files:
            print(f"[ERROR] 找不到 _predict_ball.csv: {args.csv_dir}")
            return

        print(f"找到 {len(csv_files)} 個 CSV 檔案\n")
        for csv_path in csv_files:
            seg_name = os.path.splitext(os.path.basename(csv_path))[0]
            seg_name = seg_name.replace('_predict_ball', '')

            # 自動尋找對應影片
            video_path = None
            if args.video_dir:
                for ext in ['.mp4', '.avi', '.mov']:
                    candidate = os.path.join(args.video_dir, f"{seg_name}{ext}")
                    if os.path.exists(candidate):
                        video_path = candidate
                        break

            # 批次模式下 --size 套用到所有片段（若各片段幀數不同請改用單一模式）
            csv_to_cvat_xml(csv_path, args.output, video_path, size_override=args.size)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
