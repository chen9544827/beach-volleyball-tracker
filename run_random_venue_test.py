# run_random_venue_test.py
# -*- coding: utf-8 -*-
"""
從各場地隨機挑選 normal_segment 執行發球分析
用法: python run_random_venue_test.py
"""
import os
import sys
import glob
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 設定
SEGMENTS_BASE = "output_data/video_segments"
TRACKING_BASE = "output/multi_venue_analysis/tracking_merged"
COURT_CONFIG_DIR = "court_configs"
OUTPUT_DIR = "output/random_venue_test"
NUM_SEGMENTS = 5
SEED = None  # None = 真隨機；設定整數可重現

from batch_test_serve import batch_test, load_court_config
from core.court_zones import CourtZones

def find_all_pairs():
    """找出所有 (venue, video_path, json_path, court_config_path) 組合"""
    pairs = []
    for venue_dir in sorted(glob.glob(os.path.join(SEGMENTS_BASE, "*"))):
        venue = os.path.basename(venue_dir)
        tracking_dir = os.path.join(TRACKING_BASE, venue)
        court_config_path = os.path.join(COURT_CONFIG_DIR, f"{venue}.json")

        if not os.path.isdir(tracking_dir):
            continue

        # 找所有 tracking JSON
        for json_path in sorted(glob.glob(os.path.join(tracking_dir, "*_all_frames_data_with_pose.json"))):
            seg_name = os.path.basename(json_path).replace("_all_frames_data_with_pose.json", "")
            # 在嵌套目錄中找對應影片
            video_pattern = os.path.join(venue_dir, "**", "normal_segments", f"{seg_name}.mp4")
            matches = glob.glob(video_pattern, recursive=True)
            if matches:
                pairs.append({
                    'venue': venue,
                    'video_path': matches[0],
                    'json_path': json_path,
                    'court_config_path': court_config_path if os.path.exists(court_config_path) else None,
                    'seg_name': seg_name,
                })

    return pairs

def main():
    if SEED is not None:
        random.seed(SEED)

    all_pairs = find_all_pairs()
    print(f"找到 {len(all_pairs)} 個可分析的 (影片+JSON) 配對：")
    for p in all_pairs:
        print(f"  [{p['venue']}] {p['seg_name']}")

    if len(all_pairs) == 0:
        print("[ERROR] 沒有可用的片段！")
        return

    # 每個 venue 最多抽 2 個，確保分散
    venue_groups = {}
    for p in all_pairs:
        venue_groups.setdefault(p['venue'], []).append(p)

    selected = []
    venues = list(venue_groups.keys())
    random.shuffle(venues)

    # 第一輪：每個 venue 各取 1
    for v in venues:
        if len(selected) >= NUM_SEGMENTS:
            break
        pool = venue_groups[v]
        selected.append(random.choice(pool))

    # 第二輪：若還不夠，從剩餘中補
    remaining = [p for p in all_pairs if p not in selected]
    random.shuffle(remaining)
    while len(selected) < NUM_SEGMENTS and remaining:
        selected.append(remaining.pop(0))

    print(f"\n隨機選出 {len(selected)} 個片段：")
    for p in selected:
        print(f"  [{p['venue']}] {p['seg_name']}")

    # 建立暫存目錄結構並執行分析（每個 venue 單獨跑）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import cv2
    from batch_test_serve import process_single_video

    all_results = []
    for i, p in enumerate(selected, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(selected)}] {p['venue']} / {p['seg_name']}")
        print(f"  影片: {p['video_path']}")
        print(f"  JSON: {p['json_path']}")

        court_config = None
        court_zones = None
        if p['court_config_path']:
            court_config = load_court_config(p['court_config_path'])
            if court_config:
                try:
                    court_zones = CourtZones(court_config)
                except Exception:
                    court_zones = None

        venue_output = os.path.join(OUTPUT_DIR, p['venue'])
        os.makedirs(venue_output, exist_ok=True)

        result = process_single_video(
            video_path=p['video_path'],
            json_path=p['json_path'],
            output_dir=venue_output,
            save_images=True,
            verbose=False,
            court_config=court_config,
            court_zones=court_zones,
        )
        result['venue'] = p['venue']
        all_results.append(result)

        status = result.get('status', '?')
        if status == 'success':
            serve_type = result.get('serve_type', '?')
            conf = result.get('confidence', 0)
            print(f"  [SUCCESS] {serve_type} 發球, 信心度={conf:.2f}, hit_frame={result.get('hit_frame')}")
            img = result.get('serve_moment_image')
            if img:
                print(f"  [IMAGE]   {img}")
        elif status == 'no_serve':
            print(f"  [WARNING] 未偵測到發球")
        else:
            print(f"  [ERROR]   {result.get('error', status)}")

    # 總結
    print(f"\n{'='*60}")
    print(f"總結: {len(all_results)} 個片段")
    success = [r for r in all_results if r['status'] == 'success']
    print(f"  成功: {len(success)}/{len(all_results)}")
    if success:
        print(f"  發球瞬間圖片存放目錄: {OUTPUT_DIR}/<venue>/serve_images/")

if __name__ == '__main__':
    main()
