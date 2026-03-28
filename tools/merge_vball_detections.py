# tools/merge_vball_detections.py
# -*- coding: utf-8 -*-
"""
VballNet CSV -> tracking JSON merger

VballNet の ball.csv を既存の tracking JSON に統合し、
ball_detections を VballNet 結果で置き換える。

使用法:
    python tools/merge_vball_detections.py \
        --json-dir output/tracking_test_all \
        --csv-dir output/vball_test \
        --output-dir output/tracking_vball_merged

"""

import os
import sys
import json
import argparse
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


BALL_BOX_RADIUS = 12  # VballNet X,Y を bbox に変換する際の半径 (px)
DEFAULT_CONFIDENCE = 0.85  # VballNet Visibility=1 の場合の擬似信心度


def filter_velocity_outliers(detections: dict, max_vel_px: float = 200.0,
                             image_height: int = 720) -> dict:
    """
    速度約束過濾：相鄰偵測幀間移動距離超出閾值的視為假陽性，標為不可見。

    排球物理約束：最快發球約 150km/h ≈ 118px/frame（@720p, 25fps）
    預設 200px/frame 已涵蓋快速殺球與發球，超過即視為跳幀假陽性。

    Args:
        detections: {frame_id: {'x', 'y', 'visible'}}
        max_vel_px:  每幀最大允許位移（@720p 基準，自動按解析度縮放）
        image_height: 影片高度（用於縮放閾值）
    Returns:
        過濾後的 detections（shallow copy，修改 visible 欄位）
    """
    scale = image_height / 720.0
    threshold = max_vel_px * scale

    result = {k: dict(v) for k, v in detections.items()}
    visible_frames = sorted(f for f, d in detections.items() if d.get('visible'))

    if len(visible_frames) < 2:
        return result

    # 逐對計算位移，超過 threshold * gap 的後一幀標為假陽性
    last_valid_frame = None
    for f in visible_frames:
        if last_valid_frame is None:
            last_valid_frame = f
            continue

        gap = f - last_valid_frame
        lv = result[last_valid_frame]
        cv = result[f]
        dist = ((cv['x'] - lv['x']) ** 2 + (cv['y'] - lv['y']) ** 2) ** 0.5
        max_dist = threshold * max(gap, 1)

        if dist > max_dist:
            # 超過速度上限 → 此幀為假陽性（last_valid_frame 保持不變）
            result[f]['visible'] = False
        else:
            last_valid_frame = f

    return result


def filter_island_detections(detections: dict, min_gap: int = 5,
                              min_neighbors: int = 1) -> dict:
    """
    孤島過濾：前後 min_gap 幀內沒有其他偵測的孤立點視為假陽性。

    單幀孤島通常是廣告板反光或球員揮手的瞬間假陽性。
    群聚的偵測（如連續追蹤到球）不受影響。

    Args:
        detections:    {frame_id: {'x', 'y', 'visible'}}
        min_gap:       前後多少幀內至少需要有 min_neighbors 個其他偵測
        min_neighbors: 鄰域內最少需要的偵測數（預設 1）
    Returns:
        過濾後的 detections
    """
    result = {k: dict(v) for k, v in detections.items()}
    visible_frames = sorted(f for f, d in detections.items() if d.get('visible'))
    visible_set = set(visible_frames)

    for f in visible_frames:
        neighbors = sum(
            1 for neighbor_f in range(f - min_gap, f + min_gap + 1)
            if neighbor_f != f and neighbor_f in visible_set
        )
        if neighbors < min_neighbors:
            result[f]['visible'] = False

    return result


def compute_court_x_range(court_config: dict, margin_ratio: float = 0.05) -> tuple:
    """
    court_boundary_polygon から X 軸の有効範囲を計算する。

    コートは遠端（narrow）と近端（wide）で幅が異なる。
    近端の幅を基準に margin を追加してコート外の false positive を除外。
    Y方向は上（トス後）まで許容するため制限しない。

    Returns:
        (x_min, x_max) — この範囲外の VballNet 検出は除外
    """
    polygon = court_config.get('court_boundary_polygon', [])
    if not polygon:
        return (0, 9999)

    xs = [pt[0] for pt in polygon]
    court_width = max(xs) - min(xs)
    margin = court_width * margin_ratio

    return (min(xs) - margin, max(xs) + margin)


def load_vball_csv(csv_path: str) -> dict:
    """
    VballNet の ball.csv を読み込む

    Returns:
        {frame_id: {'x': float, 'y': float, 'visible': bool}, ...}
    """
    detections = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = int(row['Frame'])
            visibility = int(row['Visibility'])
            if visibility == 1:
                x = float(row['X'])
                y = float(row['Y'])
                detections[frame_id] = {'x': x, 'y': y, 'visible': True}
            else:
                detections[frame_id] = {'visible': False}
    return detections


def make_ball_detection(x: float, y: float, radius: int = BALL_BOX_RADIUS) -> dict:
    """VballNet 座標から ball_detection dict を生成"""
    r = radius
    return {
        'box_coords': [x - r, y - r, x + r, y + r],
        'confidence': DEFAULT_CONFIDENCE,
        'center_point': [x, y],
        'source': 'vballnet'
    }


def merge_json_with_vball(json_path: str, csv_path: str, output_path: str,
                          court_config: dict = None,
                          velocity_filter: float = 0.0,
                          island_gap: int = 0) -> dict:
    """
    tracking JSON の ball_detections を VballNet 結果で置き換える

    court_config が与えられた場合、コート X 範囲外の VballNet 検出を除外して
    false positive（コート外の壁や広告板など）を抑制する。

    Returns:
        {
            'original_ball_rate': float,
            'vball_ball_rate': float,
            'filtered_count': int,
            'total_frames': int
        }
    """
    # JSON ロード
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frames = data.get('frames', [])
    if isinstance(frames, dict):
        frames = list(frames.values())

    # VballNet CSV ロード
    vball_data = load_vball_csv(csv_path)

    # 解析度取得（velocity filter の縮放用）
    image_height = data.get('metadata', {}).get('image_height', 720)

    # ---- 速度過濾 ----
    vel_filtered = 0
    if velocity_filter > 0:
        before = sum(1 for d in vball_data.values() if d.get('visible'))
        vball_data = filter_velocity_outliers(vball_data, max_vel_px=velocity_filter,
                                              image_height=image_height)
        after = sum(1 for d in vball_data.values() if d.get('visible'))
        vel_filtered = before - after

    # ---- 孤島過濾 ----
    island_filtered = 0
    if island_gap > 0:
        before = sum(1 for d in vball_data.values() if d.get('visible'))
        vball_data = filter_island_detections(vball_data, min_gap=island_gap)
        after = sum(1 for d in vball_data.values() if d.get('visible'))
        island_filtered = before - after

    # court X 範囲を計算
    x_min, x_max = (0, 9999)
    if court_config:
        x_min, x_max = compute_court_x_range(court_config, margin_ratio=0.05)

    total_frames = len(frames)
    orig_ball_count = 0
    vball_ball_count = 0
    filtered_count = 0

    for frame in frames:
        frame_id = frame.get('frame_id', 0)

        # 元のボール検出率計算
        if frame.get('ball_detections'):
            orig_ball_count += 1

        # VballNet 結果で置き換え
        vball_info = vball_data.get(frame_id)
        if vball_info and vball_info['visible']:
            x, y = vball_info['x'], vball_info['y']
            # コート X 範囲フィルタリング（false positive 除去）
            if x_min <= x <= x_max:
                frame['ball_detections'] = [make_ball_detection(x, y)]
                vball_ball_count += 1
            else:
                frame['ball_detections'] = []
                filtered_count += 1
        else:
            frame['ball_detections'] = []

    # metadata 更新
    if 'metadata' in data:
        data['metadata']['ball_source'] = 'vballnet'
        data['metadata']['vball_detection_rate'] = vball_ball_count / total_frames if total_frames > 0 else 0
        data['metadata']['vball_filtered_count'] = filtered_count
        if velocity_filter > 0:
            data['metadata']['vel_filtered'] = vel_filtered
        if island_gap > 0:
            data['metadata']['island_filtered'] = island_filtered

    # 出力
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))

    return {
        'original_ball_rate': orig_ball_count / total_frames if total_frames > 0 else 0,
        'vball_ball_rate': vball_ball_count / total_frames if total_frames > 0 else 0,
        'filtered_count': filtered_count,
        'vel_filtered': vel_filtered,
        'island_filtered': island_filtered,
        'total_frames': total_frames
    }


def main():
    parser = argparse.ArgumentParser(description='VballNet CSV -> tracking JSON merger')
    parser.add_argument('--json-dir', required=True, help='既存の tracking JSON ディレクトリ')
    parser.add_argument('--csv-dir', required=True, help='VballNet output ディレクトリ (ball.csv があるサブフォルダ)')
    parser.add_argument('--output-dir', required=True, help='マージ後の JSON 出力先')
    parser.add_argument('--court-config', default=None, help='court_config JSON（コート外 false positive 除去用）')
    parser.add_argument('--velocity-filter', type=float, default=0.0,
                        help='速度約束過濾：每幀最大位移 px（@720p，預設 0=停用，建議 200）')
    parser.add_argument('--island-gap', type=int, default=0,
                        help='孤島過濾：前後多少幀內須有其他偵測（預設 0=停用，建議 5）')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # court_config ロード
    court_config = None
    if args.court_config and os.path.exists(args.court_config):
        with open(args.court_config, 'r', encoding='utf-8') as f:
            court_config = json.load(f)
        polygon = court_config.get('court_boundary_polygon', [])
        if polygon:
            xs = [pt[0] for pt in polygon]
            x_min, x_max = compute_court_x_range(court_config, margin_ratio=0.05)
            print(f'[court filter] x_range: [{x_min:.0f}, {x_max:.0f}]  (court x: {min(xs):.0f}-{max(xs):.0f})')

    # JSON ファイルを列挙
    json_files = [f for f in os.listdir(args.json_dir) if f.endswith('.json')]

    print(f'[merge_vball] Found {len(json_files)} tracking JSONs')
    print()

    results = []
    for json_file in sorted(json_files):
        # セグメント名を抽出（_all_frames_data_with_pose.json を除去）
        seg_name = json_file.replace('_all_frames_data_with_pose.json', '')

        # CSV パスを構築（vball-net V1b 形式 優先、次に fast-volleyball 形式）
        csv_path_v1b = os.path.join(args.csv_dir, f'{seg_name}_predict_ball.csv')
        csv_path_old = os.path.join(args.csv_dir, seg_name, 'ball.csv')
        if os.path.exists(csv_path_v1b):
            csv_path = csv_path_v1b
        elif os.path.exists(csv_path_old):
            csv_path = csv_path_old
        else:
            print(f'[SKIP] {seg_name}: CSV not found at {csv_path_v1b} or {csv_path_old}')
            continue

        json_path = os.path.join(args.json_dir, json_file)
        output_path = os.path.join(args.output_dir, json_file)

        stats = merge_json_with_vball(
            json_path, csv_path, output_path,
            court_config=court_config,
            velocity_filter=args.velocity_filter,
            island_gap=args.island_gap,
        )

        extras = []
        if stats['filtered_count'] > 0:
            extras.append(f'court_x={stats["filtered_count"]}')
        if stats.get('vel_filtered', 0) > 0:
            extras.append(f'vel={stats["vel_filtered"]}')
        if stats.get('island_filtered', 0) > 0:
            extras.append(f'island={stats["island_filtered"]}')
        filtered_str = f'  filtered({", ".join(extras)})' if extras else ''
        print(f'[OK] {seg_name}')
        print(f'     YOLO ball: {stats["original_ball_rate"]:.1%}  ->  VballNet: {stats["vball_ball_rate"]:.1%}  ({stats["total_frames"]} frames){filtered_str}')
        results.append({'segment': seg_name, **stats})

    print()
    print(f'[done] Merged JSONs saved to: {args.output_dir}')

    return results


if __name__ == '__main__':
    main()
