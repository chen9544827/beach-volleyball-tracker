# tools/export_to_cvat.py
# -*- coding: utf-8 -*-
"""
將追蹤 JSON 轉換為 CVAT XML（Points Track 格式）預標註

輸出格式: CVAT for Video 1.1 XML，球作為 "track"，每幀一個 point
- outside="0" → 有球（已偵測到）
- outside="1" → 無球（需人工確認是否補標）

用法:
    python tools/export_to_cvat.py \\
        --json output/tracking_4x_crop \\
        --output output/cvat_annotations

CVAT 匯入步驟:
    1. CVAT 建立 Task → 上傳影片 → Label 新增 "volleyball"（type: any）
    2. Actions → Upload annotations → 選 "CVAT 1.1" → 選本工具輸出的 XML
    3. 逐幀檢查：有球但無點 → 點擊補標；有點但位置錯 → 拖移修正；無球但有點 → 刪除
    4. Actions → Export annotations → 選 "CVAT for video 1.1"
    5. python tools/convert_cvat_to_tracknet.py --xml <匯出的 xml> --output output/tracknet_dataset

訓練資料流程:
    annotation CSV (Frame, Visibility, X, Y)
          ↓ 訓練前處理
    Gaussian Heatmap (sigma=5px，以 X,Y 為中心)
          ↓
    TrackNet 訓練 (Input: N幀疊加, Output: Heatmap)
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
import xml.dom.minidom

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.data_validator import safe_load_json


def tracking_json_to_cvat_xml(json_path: str, output_dir: str, conf_thresh: float = 0.3):
    """
    將追蹤 JSON 轉換為 CVAT XML Points Track 格式。

    Args:
        json_path: 追蹤 JSON 路徑
        output_dir: 輸出目錄
        conf_thresh: 低於此信心度的偵測視為 needs_review
    """
    data, error = safe_load_json(json_path)
    if error:
        print(f"[ERROR] 載入失敗: {error}")
        return None

    metadata = data.get('metadata', {})
    frames = data.get('frames', [])
    if not frames:
        print("[ERROR] 無幀資料")
        return None

    total_frames = metadata.get('total_frames', len(frames))
    img_w = metadata.get('image_width', 1280)
    img_h = metadata.get('image_height', 720)
    video_name = os.path.splitext(os.path.basename(json_path))[0]
    video_name = video_name.replace('_all_frames_data_with_pose', '')

    # frame_id → ball center
    frames_dict = {f['frame_id']: f for f in frames}

    # --- 建立 CVAT XML ---
    root = ET.Element('annotations')
    ET.SubElement(root, 'version').text = '1.1'

    # meta
    meta = ET.SubElement(root, 'meta')
    task = ET.SubElement(meta, 'task')
    ET.SubElement(task, 'name').text = video_name
    ET.SubElement(task, 'size').text = str(total_frames)
    ET.SubElement(task, 'mode').text = 'interpolation'
    labels = ET.SubElement(task, 'labels')
    label = ET.SubElement(labels, 'label')
    ET.SubElement(label, 'name').text = 'volleyball'
    attrs = ET.SubElement(label, 'attributes')
    # needs_review attribute
    attr = ET.SubElement(attrs, 'attribute')
    ET.SubElement(attr, 'name').text = 'needs_review'
    ET.SubElement(attr, 'mutable').text = 'True'
    ET.SubElement(attr, 'input_type').text = 'checkbox'
    ET.SubElement(attr, 'default_value').text = 'false'

    # track（整部影片的球作為一條 track）
    track = ET.SubElement(root, 'track')
    track.set('id', '0')
    track.set('label', 'volleyball')
    track.set('source', 'auto')

    count_outside0 = 0  # 有球
    count_outside1 = 0  # 無球
    count_review = 0    # 需確認

    for frame_id in range(total_frames):
        frame = frames_dict.get(frame_id)
        balls = frame.get('ball_detections', []) if frame else []

        points_elem = ET.SubElement(track, 'points')
        points_elem.set('frame', str(frame_id))
        points_elem.set('keyframe', '1')
        points_elem.set('z_order', '0')
        points_elem.set('occluded', '0')

        if balls:
            best = max(balls, key=lambda b: b.get('confidence', 0))
            cx, cy = best.get('center_point', [0, 0])
            conf = best.get('confidence', 0)
            is_predicted = best.get('predicted', False)
            needs_review = is_predicted or conf < conf_thresh

            points_elem.set('outside', '0')
            points_elem.set('points', f"{cx:.2f},{cy:.2f}")
            if needs_review:
                count_review += 1

            # attribute
            attr_elem = ET.SubElement(points_elem, 'attribute')
            attr_elem.set('name', 'needs_review')
            attr_elem.text = 'true' if needs_review else 'false'

            count_outside0 += 1
        else:
            # 無球幀：outside=1，點設為影格中心（不顯示）
            points_elem.set('outside', '1')
            points_elem.set('points', f"{img_w//2:.2f},{img_h//2:.2f}")

            attr_elem = ET.SubElement(points_elem, 'attribute')
            attr_elem.set('name', 'needs_review')
            attr_elem.text = 'true'

            count_outside1 += 1

    # 漂亮縮排輸出
    xml_str = xml.dom.minidom.parseString(
        ET.tostring(root, encoding='unicode')
    ).toprettyxml(indent='  ', encoding=None)
    # 移除多餘的 <?xml?> 宣告行（minidom 會加）後重新加入 utf-8 版本
    lines = xml_str.split('\n')
    if lines[0].startswith('<?xml'):
        lines[0] = '<?xml version="1.0" encoding="utf-8"?>'
    xml_str = '\n'.join(lines)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{video_name}_cvat.xml")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)

    print(f"[OK] CVAT XML 已儲存: {out_path}")
    print(f"  總幀數       : {total_frames}")
    print(f"  有球幀(outside=0): {count_outside0} ({100*count_outside0/total_frames:.1f}%)")
    print(f"  無球幀(outside=1): {count_outside1} ({100*count_outside1/total_frames:.1f}%)  ← 需人工確認")
    print(f"  建議審查標註  : {count_review} 個 (低信心/預測補充)")
    print()

    return out_path


def main():
    parser = argparse.ArgumentParser(description="追蹤 JSON → CVAT XML Points Track 預標註")
    parser.add_argument("--json", type=str, required=True,
                        help="追蹤 JSON 路徑（或含多個 JSON 的目錄）")
    parser.add_argument("--output", type=str, default="output/cvat_annotations",
                        help="輸出目錄（預設 output/cvat_annotations）")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="低於此信心度標為 needs_review（預設 0.3）")
    args = parser.parse_args()

    if os.path.isdir(args.json):
        import glob
        json_files = sorted(glob.glob(
            os.path.join(args.json, '*_all_frames_data_with_pose.json')
        ))
        print(f"找到 {len(json_files)} 個 JSON 檔案\n")
        for jf in json_files:
            print(f"處理: {os.path.basename(jf)}")
            tracking_json_to_cvat_xml(jf, args.output, args.conf)
    else:
        tracking_json_to_cvat_xml(args.json, args.output, args.conf)


if __name__ == "__main__":
    main()
