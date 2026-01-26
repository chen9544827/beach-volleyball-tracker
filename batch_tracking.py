# batch_tracking.py
# -*- coding: utf-8 -*-
"""
批次追蹤影片，支援場地設定（排除區域）

用法:
    python batch_tracking.py --video-dir input_video/analyze_serve --output-dir test_output --court-config court_config.json
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from video_processing.track_ball_and_player_v2 import run_tracking_v2


def load_court_config(config_path: str) -> dict:
    """載入場地設定"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def batch_tracking(
    video_dir: str,
    output_dir: str,
    court_config_path: str = None,
    detection_interval: int = 1,
    use_ball_tracker: bool = True,
    max_occlusion_frames: int = 15,
    verbose: bool = True
):
    """
    批次追蹤所有影片
    
    Args:
        video_dir: 影片目錄
        output_dir: 輸出目錄
        court_config_path: 場地設定 JSON 路徑
        detection_interval: 偵測間隔
        use_ball_tracker: 是否使用球追蹤器
        max_occlusion_frames: 最大遮擋幀數
        verbose: 是否顯示詳細訊息
    """
    print("="*70)
    print("批次追蹤影片")
    print("="*70)
    print(f"影片目錄: {video_dir}")
    print(f"輸出目錄: {output_dir}")
    
    # 載入場地設定
    court_config = load_court_config(court_config_path)
    if court_config:
        exclusion_count = len(court_config.get('exclusion_zones', []))
        print(f"場地設定: {court_config_path}")
        print(f"  - 排除區域: {exclusion_count} 個")
        for i, zone in enumerate(court_config.get('exclusion_zones', [])):
            name = zone.get('name', f'Zone {i+1}')
            print(f"    [{i+1}] {name}")
    else:
        print(f"場地設定: 未指定（不排除任何區域）")
    print()
    
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 找出所有影片
    video_patterns = ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']
    video_files = []
    for pattern in video_patterns:
        video_files.extend(glob.glob(os.path.join(video_dir, pattern)))
    
    video_files = sorted(set(video_files))
    
    if not video_files:
        print("❌ 找不到影片檔案！")
        return
    
    print(f"找到 {len(video_files)} 個影片")
    print("-"*70)
    
    # 處理每個影片
    results = []
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n[{i}/{len(video_files)}] 處理: {video_name}")
        
        try:
            output_path = run_tracking_v2(
                video_path=video_path,
                output_dir=output_dir,
                court_config=court_config,
                detection_interval=detection_interval,
                use_ball_tracker=use_ball_tracker,
                max_occlusion_frames=max_occlusion_frames,
                verbose=verbose
            )
            
            results.append({
                'video_name': video_name,
                'status': 'success',
                'output': output_path
            })
            print(f"    ✅ 完成: {os.path.basename(output_path)}")
            
        except Exception as e:
            results.append({
                'video_name': video_name,
                'status': 'error',
                'error': str(e)
            })
            print(f"    ❌ 錯誤: {e}")
    
    # 統計結果
    print()
    print("="*70)
    print("統計結果")
    print("="*70)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    print(f"成功: {success_count}/{len(results)}")
    print(f"失敗: {error_count}/{len(results)}")
    
    if error_count > 0:
        print()
        print("失敗的影片:")
        for r in results:
            if r['status'] == 'error':
                print(f"  ❌ {r['video_name']}: {r['error']}")
    
    print()
    print("="*70)
    print("批次追蹤完成！")
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="批次追蹤影片（支援排除區域）")
    parser.add_argument("--video-dir", type=str, required=True,
                        help="影片目錄路徑")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="輸出目錄路徑")
    parser.add_argument("--court-config", type=str, default=None,
                        help="場地設定 JSON 路徑（包含排除區域）")
    parser.add_argument("--detection-interval", type=int, default=1,
                        help="偵測間隔（每 N 幀偵測一次，預設 1）")
    parser.add_argument("--max-occlusion", type=int, default=15,
                        help="最大遮擋幀數（預設 15）")
    parser.add_argument("--no-tracker", action="store_true",
                        help="停用球追蹤器")
    parser.add_argument("--quiet", action="store_true",
                        help="安靜模式（減少輸出）")
    
    args = parser.parse_args()
    
    batch_tracking(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        court_config_path=args.court_config,
        detection_interval=args.detection_interval,
        use_ball_tracker=not args.no_tracker,
        max_occlusion_frames=args.max_occlusion,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()