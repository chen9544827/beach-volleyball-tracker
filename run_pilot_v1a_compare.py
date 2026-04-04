# run_pilot_v1a_compare.py
# -*- coding: utf-8 -*-
"""
用 fine-tuned VballNetV1a 對相同 20 個片段重新分析，並與 V1b 結果對比。

步驟：
  1. 讀取 V1b pilot 結果 CSV，取得 20 個片段清單
  2. 每個片段：重跑 VballNet V1a → merge（覆寫 ball_detections）
  3. 跑發球分析
  4. 輸出 V1a CSV + 對比報告
"""
import os
import sys
import csv
import json
import shutil
import subprocess
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ===== 設定 =====
VBALL_MODEL_V1A = "fast-volleyball-tracking-inference/models/VballNetV1a_finetuned_seq9_grayscale.onnx"
VBALL_SCRIPT    = "fast-volleyball-tracking-inference/src/inference_onnx_twopass.py"
MERGE_SCRIPT    = "tools/merge_vball_detections.py"
SEGMENTS_BASE   = "output_data/video_segments"
COURT_CONFIG_DIR = "court_configs"
OUTPUT_DIR      = "output/pilot_test_20seg_v1a"
V1B_RESULTS_CSV = "output/pilot_test_20seg/pilot_results.csv"

# 已有 V1b tracking JSON 的搜尋路徑（作為 YOLO base，ball_detections 會被覆寫）
YOLO_JSON_BASES = [
    "output/multi_venue_analysis/tracking_merged",
    "output/multi_venue_analysis/tracking",
    "output/pilot_test_20seg/_work/tracking_merged",
    "output/pilot_test_20seg/_work/yolo_tracking",
]
# ================


def log(msg):
    print(msg, flush=True)


def find_existing_json(venue, seg_name):
    """從所有已知路徑找 tracking JSON（作為 YOLO base）"""
    for base in YOLO_JSON_BASES:
        p = os.path.join(base, venue, f"{seg_name}_all_frames_data_with_pose.json")
        if os.path.exists(p):
            return p
    return None


def find_video_path(venue, seg_name):
    import glob
    pattern = os.path.join(SEGMENTS_BASE, venue, "**", "normal_segments", f"{seg_name}.mp4")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def run_vballnet_v1a(video_path, vball_csv_dir):
    cmd = [
        sys.executable, VBALL_SCRIPT,
        "--video_path", os.path.abspath(video_path),
        "--model_path", os.path.abspath(VBALL_MODEL_V1A),
        "--output_dir", os.path.abspath(vball_csv_dir),
        "--only_csv", "--far_crop_ratio", "0.45",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=os.path.dirname(os.path.abspath(__file__)))
    if r.returncode != 0:
        log(f"      [VballNet ERROR] {r.stderr[-300:]}")
        return False
    return True


def run_merge(yolo_json_dir, vball_csv_dir, merged_dir, court_config_path):
    cmd = [
        sys.executable, MERGE_SCRIPT,
        "--json-dir", yolo_json_dir,
        "--csv-dir", vball_csv_dir,
        "--output-dir", merged_dir,
        "--velocity-filter", "400",
        "--island-gap", "5",
    ]
    if court_config_path and os.path.exists(court_config_path):
        cmd += ["--court-config", court_config_path]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        log(f"      [Merge ERROR] {r.stderr[-300:]}")
        return False
    return True


def read_v1b_segments():
    """讀取 V1b pilot results CSV，回傳 20 個片段的 (venue, seg_name) 列表（去重）"""
    segments = []
    seen = set()
    with open(V1B_RESULTS_CSV, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key = (row['venue'], row['seg_name'])
            if key not in seen:
                seen.add(key)
                segments.append({'venue': row['venue'], 'seg_name': row['seg_name']})
    return segments


def prepare_v1a_json(seg, work_dir):
    """準備 V1a merged JSON：REUSE YOLO base + 重跑 V1a VballNet + merge"""
    venue    = seg['venue']
    seg_name = seg['seg_name']
    court_cfg = os.path.join(COURT_CONFIG_DIR, f"{venue}.json")
    court_cfg = court_cfg if os.path.exists(court_cfg) else None

    # 找影片
    video_path = find_video_path(venue, seg_name)
    if not video_path:
        log(f"      [SKIP] 找不到影片: {venue}/{seg_name}")
        return None

    # 找現有 YOLO JSON（ball_detections 會被 V1a 覆寫）
    yolo_json = find_existing_json(venue, seg_name)
    if not yolo_json:
        log(f"      [SKIP] 找不到 YOLO JSON: {venue}/{seg_name}")
        return None

    # 建立工作目錄
    yolo_dir   = os.path.join(work_dir, "yolo_base", venue)
    vball_dir  = os.path.join(work_dir, "vball_v1a", venue)
    merged_dir = os.path.join(work_dir, "merged_v1a", venue)
    os.makedirs(yolo_dir, exist_ok=True)
    os.makedirs(vball_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)

    # 複製 YOLO JSON 到工作目錄（merge 需要從目錄讀取）
    dst_json = os.path.join(yolo_dir, os.path.basename(yolo_json))
    if not os.path.exists(dst_json):
        shutil.copy2(yolo_json, dst_json)

    # 跑 V1a VballNet
    log(f"      [V1a] {seg_name}")
    if not run_vballnet_v1a(video_path, vball_dir):
        log(f"      [WARN] V1a 推理失敗，跳過")
        return None

    # 確認 VballNet CSV 存在
    vball_csv = os.path.join(vball_dir, seg_name, "ball.csv")
    if not os.path.exists(vball_csv):
        log(f"      [WARN] V1a CSV 未產生: {vball_csv}")
        # 檢查 vball_dir 下有什麼
        import glob as _glob
        found = _glob.glob(os.path.join(vball_dir, "**", "ball.csv"), recursive=True)
        log(f"      [INFO] 找到的 ball.csv: {found}")
        if not found:
            return None

    # Merge
    if not run_merge(yolo_dir, vball_dir, merged_dir, court_cfg):
        log(f"      [WARN] Merge 失敗")
        return None

    merged_json = os.path.join(merged_dir, f"{seg_name}_all_frames_data_with_pose.json")
    if os.path.exists(merged_json):
        return merged_json

    log(f"      [WARN] merged JSON 未產生: {merged_json}")
    return None


def run_serve_analysis(segments_with_json, output_dir):
    import cv2
    from batch_test_serve import process_single_video, load_court_config
    from core.court_zones import CourtZones

    os.makedirs(output_dir, exist_ok=True)
    results = []
    total = len(segments_with_json)
    for i, seg in enumerate(segments_with_json, 1):
        venue    = seg['venue']
        seg_name = seg['seg_name']
        video    = seg['video_path']
        json_p   = seg['json_path']
        court_cfg_path = os.path.join(COURT_CONFIG_DIR, f"{venue}.json")
        court_cfg_path = court_cfg_path if os.path.exists(court_cfg_path) else None

        log(f"  [{i}/{total}] {venue} / {seg_name}")

        court_config = load_court_config(court_cfg_path) if court_cfg_path else None
        court_zones  = None
        if court_config:
            try:
                court_zones = CourtZones(court_config)
            except Exception:
                pass

        venue_out = os.path.join(output_dir, venue)
        os.makedirs(venue_out, exist_ok=True)

        result = process_single_video(
            video_path=video,
            json_path=json_p,
            output_dir=venue_out,
            save_images=True,
            verbose=False,
            court_config=court_config,
            court_zones=court_zones,
        )
        result['venue']    = venue
        result['seg_name'] = seg_name
        results.append(result)

        status = result.get('status', '?')
        if status == 'success':
            log(f"    [OK] {result.get('serve_type','?')} serve  hit={result.get('hit_frame')}  "
                f"ball_rate={result.get('ball_detection_rate', 0):.1%}")
        elif status == 'no_serve':
            log(f"    [--] no_serve  ball_rate={result.get('ball_detection_rate', 0):.1%}")
        else:
            log(f"    [ERR] {result.get('error', status)}")

    return results


def read_v1b_results():
    """讀取 V1b 結果為 dict: (venue, seg_name) -> row"""
    results = {}
    with open(V1B_RESULTS_CSV, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key = (row['venue'], row['seg_name'])
            results[key] = row
    return results


def export_comparison(v1a_results, v1b_lookup, output_dir):
    """輸出對比 CSV 與 JSON summary"""
    fields_out = ["venue", "seg_name",
                  "v1b_status", "v1b_ball_rate", "v1b_max_consec", "v1b_serve_type", "v1b_hit_frame",
                  "v1a_status", "v1a_ball_rate", "v1a_max_consec", "v1a_serve_type", "v1a_hit_frame",
                  "changed", "direction"]

    rows = []
    for r in v1a_results:
        key = (r['venue'], r['seg_name'])
        v1b = v1b_lookup.get(key, {})

        v1a_status = r.get('status', '?')
        v1b_status = v1b.get('status', '?')
        v1a_rate   = r.get('ball_detection_rate', 0)
        v1b_rate   = float(v1b.get('ball_detection_rate', 0) or 0)
        v1a_consec = r.get('max_consecutive_ball_frames', 0)
        v1b_consec = int(v1b.get('max_consecutive_ball_frames', 0) or 0)

        changed = (v1a_status != v1b_status)
        if changed:
            if v1b_status != 'success' and v1a_status == 'success':
                direction = 'improved'
            elif v1b_status == 'success' and v1a_status != 'success':
                direction = 'regressed'
            else:
                direction = 'changed'
        else:
            direction = 'same'

        rows.append({
            "venue": r['venue'], "seg_name": r['seg_name'],
            "v1b_status": v1b_status,
            "v1b_ball_rate": f"{v1b_rate:.3f}",
            "v1b_max_consec": v1b_consec,
            "v1b_serve_type": v1b.get('serve_type', ''),
            "v1b_hit_frame": v1b.get('hit_frame', ''),
            "v1a_status": v1a_status,
            "v1a_ball_rate": f"{v1a_rate:.3f}",
            "v1a_max_consec": v1a_consec,
            "v1a_serve_type": r.get('serve_type', ''),
            "v1a_hit_frame": r.get('hit_frame', ''),
            "changed": str(changed),
            "direction": direction,
        })

    compare_csv = os.path.join(output_dir, "v1a_v1b_comparison.csv")
    with open(compare_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields_out)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log(f"[CSV] {compare_csv}")

    # Summary
    improved  = [r for r in rows if r['direction'] == 'improved']
    regressed = [r for r in rows if r['direction'] == 'regressed']
    same      = [r for r in rows if r['direction'] == 'same']

    v1a_success = [r for r in rows if r['v1a_status'] == 'success']
    v1b_success = [r for r in rows if r['v1b_status'] == 'success']

    summary = {
        "generated_at": datetime.now().isoformat(),
        "total": len(rows),
        "v1b_success": len(v1b_success),
        "v1b_success_rate": round(len(v1b_success) / max(len(rows), 1), 3),
        "v1a_success": len(v1a_success),
        "v1a_success_rate": round(len(v1a_success) / max(len(rows), 1), 3),
        "improved": len(improved),
        "regressed": len(regressed),
        "same": len(same),
        "improved_segments": [{"venue": r["venue"], "seg": r["seg_name"]} for r in improved],
        "regressed_segments": [{"venue": r["venue"], "seg": r["seg_name"]} for r in regressed],
        "per_venue": {},
    }

    venues = sorted(set(r['venue'] for r in rows))
    for v in venues:
        vr = [r for r in rows if r['venue'] == v]
        summary["per_venue"][v] = {
            "v1b_success": sum(1 for r in vr if r['v1b_status'] == 'success'),
            "v1a_success": sum(1 for r in vr if r['v1a_status'] == 'success'),
            "improved": sum(1 for r in vr if r['direction'] == 'improved'),
            "regressed": sum(1 for r in vr if r['direction'] == 'regressed'),
            "v1b_avg_ball_rate": round(sum(float(r['v1b_ball_rate']) for r in vr) / len(vr), 3),
            "v1a_avg_ball_rate": round(sum(float(r['v1a_ball_rate']) for r in vr) / len(vr), 3),
        }

    summary_path = os.path.join(output_dir, "v1a_v1b_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"[JSON] {summary_path}")
    return summary


def print_comparison(summary):
    log("\n" + "=" * 65)
    log("  V1a vs V1b 對比結果")
    log("=" * 65)
    log(f"  V1b 成功率: {summary['v1b_success']}/{summary['total']} "
        f"({summary['v1b_success_rate']:.1%})")
    log(f"  V1a 成功率: {summary['v1a_success']}/{summary['total']} "
        f"({summary['v1a_success_rate']:.1%})")
    log(f"  改善: {summary['improved']}  退步: {summary['regressed']}  不變: {summary['same']}")

    log("\n  --- 各場地 ---")
    for v, s in summary["per_venue"].items():
        tag = ""
        if s['improved'] > 0:
            tag += f"  +{s['improved']}改善"
        if s['regressed'] > 0:
            tag += f"  -{s['regressed']}退步"
        log(f"  {v}: V1b {s['v1b_success']}/5 → V1a {s['v1a_success']}/5{tag}"
            f"  ball_rate {s['v1b_avg_ball_rate']:.1%}→{s['v1a_avg_ball_rate']:.1%}")

    if summary["improved_segments"]:
        log("\n  改善的片段:")
        for s in summary["improved_segments"]:
            log(f"    + {s['venue']} / {s['seg']}")
    if summary["regressed_segments"]:
        log("\n  退步的片段:")
        for s in summary["regressed_segments"]:
            log(f"    - {s['venue']} / {s['seg']}")
    log("=" * 65)


def main():
    log("=" * 65)
    log("  V1a vs V1b Pilot 對比測試")
    log("=" * 65)

    if not os.path.exists(V1B_RESULTS_CSV):
        log(f"[ERROR] V1b 結果 CSV 不存在: {V1B_RESULTS_CSV}")
        sys.exit(1)
    if not os.path.exists(VBALL_MODEL_V1A):
        log(f"[ERROR] V1a 模型不存在: {VBALL_MODEL_V1A}")
        sys.exit(1)

    work_dir = os.path.join(OUTPUT_DIR, "_work")
    os.makedirs(work_dir, exist_ok=True)

    # 1. 讀取 20 個片段清單
    segments = read_v1b_segments()
    log(f"\n[Step 1] 讀取 {len(segments)} 個片段（來自 V1b pilot）")

    # 2. 重跑 V1a VballNet + merge
    log("\n[Step 2] V1a VballNet 推理 + merge...")
    segments_with_json = []
    for seg in segments:
        venue    = seg['venue']
        seg_name = seg['seg_name']
        video    = find_video_path(venue, seg_name)
        log(f"\n  {venue} / {seg_name}")
        json_path = prepare_v1a_json(seg, work_dir)
        if json_path and video:
            seg['json_path']   = json_path
            seg['video_path']  = video
            segments_with_json.append(seg)
        else:
            log(f"    [SKIP]")

    log(f"\n  共 {len(segments_with_json)}/{len(segments)} 個片段準備完成")

    # 3. 發球分析
    log("\n[Step 3] 發球分析...")
    analysis_dir = os.path.join(OUTPUT_DIR, "analysis")
    v1a_results  = run_serve_analysis(segments_with_json, analysis_dir)

    # 4. 輸出對比
    log("\n[Step 4] 輸出對比結果...")
    v1b_lookup = read_v1b_results()
    summary    = export_comparison(v1a_results, v1b_lookup, OUTPUT_DIR)

    # 5. 輸出完整 V1a CSV
    v1a_csv_path = os.path.join(OUTPUT_DIR, "v1a_results.csv")
    fields = ["venue", "seg_name", "status", "serve_detected", "serve_type",
              "hit_frame", "confidence", "serve_zone", "serving_side",
              "reception_detected", "reception_zone", "is_ace",
              "ball_detection_rate", "max_consecutive_ball_frames",
              "toss_height_px", "jump_height", "error"]
    with open(v1a_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in v1a_results:
            w.writerow(r)
    log(f"[CSV] {v1a_csv_path}")

    print_comparison(summary)


if __name__ == "__main__":
    main()
