# tools/run_pilot_test.py
# -*- coding: utf-8 -*-
"""
Pilot Test Runner - 沙灘排球發球偵測管線前導測試

功能：
  1. 掃描 output_data/video_segments/{group_key}/{match_folder}/normal_segments/
  2. 使用 core/filename_parser.py 解析 match_folder 名稱取得比賽元資料
  3. 每個場地 (group_key) 隨機抽選 N 個 normal_segments（預設 5）
  4. 對每個 segment 執行追蹤 (run_tracking_v2) + 發球分析 (process_single_video)
  5. 匯出 CSV / Excel 詳細結果
  6. 額外輸出 pilot_summary.json，統計：
       - 成功偵測到發球的比例
       - 被判定為 F 級的比例
     (全體 + 每個場地分開統計)

Usage:
    python tools/run_pilot_test.py
    python tools/run_pilot_test.py --samples 3 --seed 0 --skip-tracking
    python tools/run_pilot_test.py --segments-dir path/to/segments --output-dir output/pilot
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# --- Path setup ---
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, PROJECT_ROOT)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from core.filename_parser import parse_filename
from core.data_validator import safe_load_json
from core.court_zones import CourtZones
from core.result_exporter import (
    build_result_row, export_to_csv, export_to_excel, compute_quality_grade
)

import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# --- Defaults ---
DEFAULT_SEGMENTS_DIR  = os.path.join(PROJECT_ROOT, 'output_data', 'video_segments')
DEFAULT_OUTPUT_DIR    = os.path.join(PROJECT_ROOT, 'output', 'pilot_test')
DEFAULT_COURT_DIR     = os.path.join(PROJECT_ROOT, 'court_configs')
DEFAULT_SAMPLES       = 5
DEFAULT_SEED          = 42


# ---------------------------------------------------------------------------
# 1. Directory discovery
# ---------------------------------------------------------------------------

def discover_segments(segments_dir: str) -> Dict[str, List[Tuple[str, str, Optional[Dict]]]]:
    """
    掃描 segments_dir，按 group_key 分組 normal_segments。

    Returns:
        {group_key: [(video_path, match_folder_name, parsed_match_info), ...]}
    """
    groups: Dict[str, List[Tuple[str, str, Optional[Dict]]]] = {}

    if not os.path.isdir(segments_dir):
        print(f"[ERROR] segments 目錄不存在: {segments_dir}")
        return groups

    for entry in sorted(os.listdir(segments_dir)):
        group_dir = os.path.join(segments_dir, entry)
        if not os.path.isdir(group_dir):
            continue
        # 跳過看起來像 FIVB match folder 的根目錄項（舊版扁平結構）
        if entry.upper().startswith('FIVB'):
            continue

        group_key = entry
        segment_records: List[Tuple[str, str, Optional[Dict]]] = []

        for match_folder in sorted(os.listdir(group_dir)):
            match_dir = os.path.join(group_dir, match_folder)
            if not os.path.isdir(match_dir):
                continue

            # 用 filename_parser 解析 match_folder 名稱
            parsed_match = parse_filename(match_folder)

            normal_dir = os.path.join(match_dir, 'normal_segments')
            if not os.path.isdir(normal_dir):
                continue

            for fname in sorted(os.listdir(normal_dir)):
                if fname.lower().endswith('.mp4'):
                    video_path = os.path.join(normal_dir, fname)
                    segment_records.append((video_path, match_folder, parsed_match))

        if segment_records:
            groups[group_key] = segment_records

    return groups


# ---------------------------------------------------------------------------
# 2. court_config loader
# ---------------------------------------------------------------------------

def load_court_config_for_group(group_key: str, court_dir: str) -> Optional[Dict]:
    config_path = os.path.join(court_dir, f"{group_key}.json")
    if not os.path.exists(config_path):
        return None
    data, err = safe_load_json(config_path)
    if err:
        print(f"  [WARN] 載入 court_config 失敗: {err}")
        return None
    return data


# ---------------------------------------------------------------------------
# 3. Pipeline runner for one segment
# ---------------------------------------------------------------------------

def run_segment(
    video_path: str,
    tracking_dir: str,
    analysis_dir: str,
    court_config: Optional[Dict],
    court_zones: Optional[CourtZones],
    skip_tracking: bool,
    verbose: bool,
) -> Dict:
    """
    對單一 segment 執行「追蹤 → 發球分析」完整管線。

    Returns:
        result dict (相容 process_single_video 回傳格式)
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = os.path.join(tracking_dir, f"{video_name}_all_frames_data_with_pose.json")

    # --- Step 1: Tracking ---
    if not os.path.exists(json_path):
        if skip_tracking:
            return _error_result(video_name, 'no_tracking_json',
                                 'Tracking JSON 不存在且已設定 --skip-tracking')
        print(f"    [追蹤] {video_name} ...", flush=True)
        try:
            from video_processing.track_ball_and_player_v2 import run_tracking_v2
            run_tracking_v2(
                video_path=video_path,
                output_dir=tracking_dir,
                court_config=court_config,
                verbose=verbose,
                early_stop_after_serve=True,
                early_stop_buffer_frames=150,
            )
        except Exception as e:
            return _error_result(video_name, 'tracking_error', str(e))

    # --- Step 2: Serve analysis ---
    try:
        from batch_test_serve import process_single_video
        result = process_single_video(
            video_path=video_path,
            json_path=json_path,
            output_dir=analysis_dir,
            save_images=False,
            verbose=verbose,
            court_config=court_config,
            court_zones=court_zones,
        )
    except Exception as e:
        result = _error_result(video_name, 'analysis_error', str(e))

    # 補齊 quality_grade（若 process_single_video 未設定）
    if 'quality_grade' not in result:
        result['quality_grade'] = compute_quality_grade(result.get('ball_detection_rate', 0))

    return result


def _error_result(video_name: str, status: str, error: str) -> Dict:
    return {
        'video_name': video_name,
        'status': status,
        'error': error,
        'serve_detected': False,
        'ball_detection_rate': 0.0,
        'quality_grade': 'F',
        'max_consecutive_ball_frames': 0,
    }


# ---------------------------------------------------------------------------
# 4. Summary computation
# ---------------------------------------------------------------------------

def _venue_stats(results: List[Dict]) -> Dict:
    total = len(results)
    if total == 0:
        return {'total_sampled': 0, 'serve_detected_count': 0,
                'serve_detection_rate': 0.0, 'f_grade_count': 0,
                'f_grade_rate': 0.0, 'error_count': 0}

    serve_count = sum(1 for r in results if r.get('serve_detected', False))
    f_count     = sum(1 for r in results if r.get('quality_grade') == 'F')
    err_count   = sum(1 for r in results
                      if r.get('status') in ('tracking_error', 'analysis_error',
                                             'no_tracking_json', 'insufficient_ball_data'))
    return {
        'total_sampled': total,
        'serve_detected_count': serve_count,
        'serve_detection_rate': round(serve_count / total, 4),
        'f_grade_count': f_count,
        'f_grade_rate': round(f_count / total, 4),
        'error_count': err_count,
    }


def compute_pilot_summary(
    all_results: List[Dict],
    per_venue: Dict[str, List[Dict]],
    params: Dict,
) -> Dict:
    return {
        'generated_at': datetime.now().isoformat(),
        'params': params,
        'overall': _venue_stats(all_results),
        'per_venue': {gk: _venue_stats(rs) for gk, rs in per_venue.items()},
        'segments': [
            {
                'video_name':             r.get('video_name', ''),
                'group_key':              r.get('group_key', ''),
                'match_folder':           r.get('match_folder', ''),
                'status':                 r.get('status', 'unknown'),
                'serve_detected':         r.get('serve_detected', False),
                'serve_type':             r.get('serve_type'),
                'is_jump_serve':          r.get('is_jump_serve'),
                'quality_grade':          r.get('quality_grade', '?'),
                'ball_detection_rate':    round(r.get('ball_detection_rate', 0), 4),
                'max_consecutive_ball':   r.get('max_consecutive_ball_frames', 0),
                'reception_detected':     r.get('reception_detected', False),
                'court_detection_quality': r.get('court_detection_quality'),
            }
            for r in all_results
        ],
    }


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Pilot test: sample segments per venue and run serve detection pipeline'
    )
    parser.add_argument('--segments-dir',    default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument('--output-dir',      default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--court-configs-dir', default=DEFAULT_COURT_DIR)
    parser.add_argument('--tracking-dir',    default=None,
                        help='Pre-computed tracking JSONs dir (default: <output-dir>/tracking)')
    parser.add_argument('--samples',         type=int, default=DEFAULT_SAMPLES,
                        help=f'Segments to sample per venue (default: {DEFAULT_SAMPLES})')
    parser.add_argument('--seed',            type=int, default=DEFAULT_SEED,
                        help=f'Random seed (default: {DEFAULT_SEED})')
    parser.add_argument('--skip-tracking',   action='store_true',
                        help='Skip tracking step (use existing JSONs only)')
    parser.add_argument('--verbose',         action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)

    print('=' * 70)
    print('Pilot Test Runner - Beach Volleyball Serve Detection')
    print('=' * 70)
    print(f'Segments dir   : {args.segments_dir}')
    print(f'Output dir     : {args.output_dir}')
    print(f'Court configs  : {args.court_configs_dir}')
    print(f'Samples/venue  : {args.samples}')
    print(f'Random seed    : {args.seed}')
    print(f'Skip tracking  : {args.skip_tracking}')
    print()

    # --- Discover ---
    all_groups = discover_segments(args.segments_dir)
    if not all_groups:
        print('[ERROR] 找不到任何 normal_segments，請確認目錄結構')
        sys.exit(1)

    print(f'發現 {len(all_groups)} 個場地 (group_key)：')
    for gk, recs in all_groups.items():
        print(f'  {gk}: {len(recs)} 個 normal_segments')
    print()

    # --- Sample ---
    sampled: Dict[str, List[Tuple[str, str, Optional[Dict]]]] = {}
    for gk, records in all_groups.items():
        n = min(args.samples, len(records))
        sampled[gk] = random.sample(records, n)

    print('抽選結果：')
    for gk, recs in sampled.items():
        print(f'  [{gk}]')
        for vpath, mfolder, _ in recs:
            print(f'    {os.path.basename(vpath)}  (from {mfolder[:50]}...)'
                  if len(mfolder) > 50 else
                  f'    {os.path.basename(vpath)}  (from {mfolder})')
    print()

    # --- Prepare dirs ---
    os.makedirs(args.output_dir, exist_ok=True)
    tracking_dir = args.tracking_dir or os.path.join(args.output_dir, 'tracking')
    analysis_dir = os.path.join(args.output_dir, 'analysis')
    os.makedirs(tracking_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    # --- Run pipeline ---
    all_results: List[Dict] = []
    per_venue_results: Dict[str, List[Dict]] = {gk: [] for gk in sampled}
    total_segs = sum(len(v) for v in sampled.values())
    processed = 0

    for gk, seg_records in sampled.items():
        print(f"\n{'='*60}")
        print(f'場地: {gk}  ({len(seg_records)} segments)')
        print(f"{'='*60}")

        court_config = load_court_config_for_group(gk, args.court_configs_dir)
        if court_config:
            cq = court_config.get('court_detection_quality', 'good')
            print(f'  court_config: {gk}.json  [quality={cq}]')
        else:
            print(f'  court_config: 未找到 {gk}.json，使用無設定模式')

        court_zones: Optional[CourtZones] = None
        if court_config:
            try:
                court_zones = CourtZones(court_config)
            except Exception as e:
                print(f'  [WARN] CourtZones 初始化失敗: {e}')

        for j, (video_path, match_folder, parsed_match) in enumerate(seg_records, 1):
            processed += 1
            seg_name = os.path.basename(video_path)
            print(f'\n  [{j}/{len(seg_records)}] ({processed}/{total_segs}) {seg_name}')

            result = run_segment(
                video_path=video_path,
                tracking_dir=tracking_dir,
                analysis_dir=analysis_dir,
                court_config=court_config,
                court_zones=court_zones,
                skip_tracking=args.skip_tracking,
                verbose=args.verbose,
            )

            # 補充元資料
            result['group_key']    = gk
            result['match_folder'] = match_folder
            # 將 parsed_match 的比賽資訊合入 result（供 build_result_row 使用）
            if parsed_match and not result.get('video_name', '').startswith('FIVB'):
                result.setdefault('_parsed_match_override', parsed_match)

            # 打印簡短狀態
            status   = result.get('status', 'unknown')
            serve_ok = '[SERVE]'    if result.get('serve_detected') else '[NO_SERVE]'
            grade    = f"[{result.get('quality_grade', '?')}]"
            rate     = result.get('ball_detection_rate', 0)
            consec   = result.get('max_consecutive_ball_frames', 0)
            print(f'    -> {status:<28} {serve_ok} {grade}  '
                  f'ball_rate={rate:.1%}  max_consec={consec}')

            all_results.append(result)
            per_venue_results[gk].append(result)

    # --- Export CSV / Excel ---
    print(f"\n{'='*70}")
    print('匯出結果...')
    try:
        export_rows = []
        for r in all_results:
            video_name = r.get('video_name', '')
            # 優先嘗試解析 segment 名稱；失敗時用 match_folder 解析結果
            parsed = parse_filename(video_name) or r.get('_parsed_match_override')
            # 最終備用：從 group_key 拆出欄位
            if parsed is None:
                parts = r.get('group_key', '__').split('_')
                parsed = {
                    'venue':     parts[0] if len(parts) > 0 else '',
                    'year':      parts[1] if len(parts) > 1 else '',
                    'court':     parts[2] if len(parts) > 2 else '',
                    'group_key': r.get('group_key', ''),
                }

            ball_rate = r.get('ball_detection_rate', 0)
            row = build_result_row(
                video_name=video_name,
                parsed_filename=parsed,
                serve_result=r,
                reception_result=r,
                ball_detection_rate=ball_rate,
                status=r.get('status', 'unknown'),
            )
            # 強制降級：auto court detection 品質不佳
            cq = r.get('court_detection_quality')
            if cq == 'unreliable':
                row['quality_grade'] = 'F'
            elif cq == 'degraded' and row.get('quality_grade') in ('A', 'B'):
                row['quality_grade'] = 'C'

            export_rows.append(row)

        csv_path = os.path.join(args.output_dir, 'pilot_results.csv')
        export_to_csv(export_rows, csv_path)
        print(f'  CSV  : {csv_path}')

        excel_path = os.path.join(args.output_dir, 'pilot_results.xlsx')
        export_to_excel(export_rows, excel_path)
        print(f'  Excel: {excel_path}')

    except Exception as e:
        print(f'  [WARN] CSV/Excel 匯出失敗: {e}')

    # --- pilot_summary.json ---
    params = {
        'segments_dir': args.segments_dir,
        'samples_per_venue': args.samples,
        'seed': args.seed,
        'skip_tracking': args.skip_tracking,
        'total_venues': len(sampled),
        'total_segments': total_segs,
    }
    summary = compute_pilot_summary(all_results, per_venue_results, params)
    summary_path = os.path.join(args.output_dir, 'pilot_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'  JSON : {summary_path}')

    # --- Console summary ---
    print()
    print('=' * 70)
    print('Pilot Summary')
    print('=' * 70)
    ov = summary['overall']
    print(f"{'總計':<12}: {ov['total_sampled']} segments")
    print(f"{'發球偵測率':<10}: {ov['serve_detection_rate']:.1%}"
          f"  ({ov['serve_detected_count']}/{ov['total_sampled']})")
    print(f"{'F 級比例':<11}: {ov['f_grade_rate']:.1%}"
          f"  ({ov['f_grade_count']}/{ov['total_sampled']})")
    print(f"{'錯誤數':<12}: {ov['error_count']}")
    print()
    print('per-venue:')
    header = f"  {'場地':<26} {'n':>3}  {'serve%':>7}  {'F%':>6}  {'errors':>6}"
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for gk, st in summary['per_venue'].items():
        print(f"  {gk:<26} {st['total_sampled']:>3}  "
              f"{st['serve_detection_rate']:>6.1%}  "
              f"{st['f_grade_rate']:>5.1%}  "
              f"{st['error_count']:>6}")


if __name__ == '__main__':
    main()
