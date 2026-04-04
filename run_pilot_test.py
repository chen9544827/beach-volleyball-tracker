# run_pilot_test.py
# -*- coding: utf-8 -*-
"""
Pilot 測試管線：每個場地各 5 個 normal_segment 全流程分析
步驟：
  1. 挑選 5 個片段（優先使用已有 tracking 的）
  2. 對缺少 tracking 的片段跑：YOLO tracking → VballNet 推理 → merge
  3. 跑發球分析 (batch_test_serve.py 邏輯)
  4. 輸出 batch_results.csv + pilot_summary.json
"""
import os
import sys
import glob
import json
import random
import subprocess
import shutil
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ========== 設定 ==========
SEGMENTS_BASE   = "output_data/video_segments"
TRACKING_BASE   = "output/multi_venue_analysis/tracking_merged"
COURT_CONFIG_DIR = "court_configs"
OUTPUT_DIR      = "output/pilot_test_20seg"
VBALL_MODEL     = "fast-volleyball-tracking-inference/models/VballNetV1a_finetuned_seq9_grayscale.onnx"
VBALL_SCRIPT    = "fast-volleyball-tracking-inference/src/inference_onnx_twopass.py"
MERGE_SCRIPT    = "tools/merge_vball_detections.py"
SEGMENTS_PER_VENUE = 5
SEED = 42
# ==========================

random.seed(SEED)


def log(msg):
    print(msg, flush=True)


def find_video(venue_dir, seg_name):
    pattern = os.path.join(venue_dir, "**", "normal_segments", f"{seg_name}.mp4")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def find_existing_json(venue, seg_name):
    """找已有的 tracking JSON（優先 merged，fallback tracking）"""
    for base in [
        "output/multi_venue_analysis/tracking_merged",
        "output/multi_venue_analysis/tracking",
    ]:
        p = os.path.join(base, venue, f"{seg_name}_all_frames_data_with_pose.json")
        if os.path.exists(p):
            return p
    return None


def run_yolo_tracking(video_path, output_dir, court_config_path):
    """跑 YOLO tracking，輸出 JSON 到 output_dir/<seg_name>/"""
    cmd = [
        sys.executable, "batch_tracking.py",
        "--video-dir", os.path.dirname(video_path),
        "--output-dir", output_dir,
        "--quiet",
    ]
    if court_config_path and os.path.exists(court_config_path):
        cmd += ["--court-config", court_config_path]
    log(f"      [YOLO] {os.path.basename(video_path)}")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        log(f"      [YOLO ERROR] {r.stderr[-300:]}")
        return False
    return True


def run_vballnet(video_path, vball_csv_dir):
    """跑 VballNet 兩段推理"""
    cmd = [
        sys.executable, VBALL_SCRIPT,
        "--video_path", os.path.abspath(video_path),
        "--model_path", os.path.abspath(VBALL_MODEL),
        "--output_dir", os.path.abspath(vball_csv_dir),
        "--only_csv", "--far_crop_ratio", "0.45",
    ]
    log(f"      [VballNet] {os.path.basename(video_path)}")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=os.path.dirname(os.path.abspath(__file__)))
    if r.returncode != 0:
        log(f"      [VballNet ERROR] {r.stderr[-300:]}")
        return False
    return True


def run_merge(yolo_json_dir, vball_csv_dir, merged_dir, court_config_path):
    """merge VballNet CSV 進 YOLO JSON"""
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
    log(f"      [Merge] {os.path.basename(yolo_json_dir)}")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        log(f"      [Merge ERROR] {r.stderr[-300:]}")
        return False
    return True


def prepare_tracking_json(seg_info, work_dir):
    """確保 tracking JSON 存在，不存在就跑完整追蹤管線。回傳 json_path 或 None"""
    venue      = seg_info["venue"]
    seg_name   = seg_info["seg_name"]
    video_path = seg_info["video_path"]
    court_cfg  = seg_info["court_config_path"]

    # 1. 已有就直接回傳
    existing = find_existing_json(venue, seg_name)
    if existing:
        log(f"      [REUSE] {existing}")
        return existing

    # 2. 跑 YOLO tracking
    yolo_dir    = os.path.join(work_dir, "yolo_tracking", venue)
    vball_dir   = os.path.join(work_dir, "vball_csv", venue)
    merged_dir  = os.path.join(work_dir, "tracking_merged", venue)
    os.makedirs(yolo_dir, exist_ok=True)
    os.makedirs(vball_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)

    # batch_tracking.py 需要影片目錄，且只處理該目錄下的 mp4
    # 使用暫存目錄放單個影片的 symlink / copy（Windows 用 copy）
    tmp_video_dir = os.path.join(work_dir, "tmp_video", venue, seg_name)
    os.makedirs(tmp_video_dir, exist_ok=True)
    dst_video = os.path.join(tmp_video_dir, os.path.basename(video_path))
    if not os.path.exists(dst_video):
        shutil.copy2(video_path, dst_video)

    if not run_yolo_tracking(dst_video, yolo_dir, court_cfg):
        return None

    # 找 YOLO 輸出 JSON
    yolo_json = os.path.join(yolo_dir, f"{seg_name}_all_frames_data_with_pose.json")
    if not os.path.exists(yolo_json):
        log(f"      [ERROR] YOLO JSON not found: {yolo_json}")
        return None

    # 3. VballNet
    if not run_vballnet(video_path, vball_dir):
        log(f"      [WARN] VballNet failed, using YOLO-only JSON")
        return yolo_json

    # 4. Merge
    if not run_merge(yolo_dir, vball_dir, merged_dir, court_cfg):
        log(f"      [WARN] Merge failed, using YOLO-only JSON")
        return yolo_json

    merged_json = os.path.join(merged_dir, f"{seg_name}_all_frames_data_with_pose.json")
    if os.path.exists(merged_json):
        return merged_json
    return yolo_json


def collect_candidates():
    """收集每個場地的候選 (venue, seg_name, video_path, court_config_path)"""
    venues = sorted([d for d in os.listdir(SEGMENTS_BASE)
                     if os.path.isdir(os.path.join(SEGMENTS_BASE, d))])
    venue_candidates = {}
    for venue in venues:
        venue_dir = os.path.join(SEGMENTS_BASE, venue)
        court_cfg = os.path.join(COURT_CONFIG_DIR, f"{venue}.json")
        court_cfg = court_cfg if os.path.exists(court_cfg) else None

        videos = sorted(glob.glob(os.path.join(venue_dir, "**", "normal_segments", "*.mp4"), recursive=True))
        has_json = []
        no_json  = []
        seen_names = set()  # 跨 match 目錄去重：同 seg_name 只保留第一個
        for v in videos:
            seg_name = os.path.splitext(os.path.basename(v))[0]
            if seg_name in seen_names:
                continue
            seen_names.add(seg_name)
            info = {
                "venue": venue,
                "seg_name": seg_name,
                "video_path": v,
                "court_config_path": court_cfg,
            }
            if find_existing_json(venue, seg_name):
                has_json.append(info)
            else:
                no_json.append(info)

        # 優先選已有 JSON 的，不足再從無 JSON 的補
        random.shuffle(has_json)
        random.shuffle(no_json)
        selected = (has_json + no_json)[:SEGMENTS_PER_VENUE]
        venue_candidates[venue] = selected
        log(f"  [{venue}] {len(has_json)} 已追蹤 / {len(no_json)} 未追蹤 → 選 {len(selected)} 個")
    return venue_candidates


def run_serve_analysis(segments_with_json, output_dir, work_dir):
    """對所有 (video_path, json_path, court_cfg) 跑發球分析，回傳 results list"""
    import cv2
    from batch_test_serve import process_single_video, load_court_config
    from core.court_zones import CourtZones

    os.makedirs(output_dir, exist_ok=True)
    results = []
    total = len(segments_with_json)
    for i, seg in enumerate(segments_with_json, 1):
        venue    = seg["venue"]
        seg_name = seg["seg_name"]
        video    = seg["video_path"]
        json_p   = seg["json_path"]
        court_cfg_path = seg["court_config_path"]

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
        result["venue"]    = venue
        result["seg_name"] = seg_name
        results.append(result)

        status = result.get("status", "?")
        if status == "success":
            st = result.get("serve_type", "?")
            c  = result.get("confidence", 0)
            log(f"    [OK] {st} 發球  hit_frame={result.get('hit_frame')}  conf={c:.2f}")
            img = result.get("serve_moment_image")
            if img:
                log(f"    [IMG] {img}")
        elif status == "no_serve":
            log(f"    [--] 未偵測到發球")
        else:
            log(f"    [ERR] {result.get('error', status)}")

    return results


def export_csv(results, csv_path):
    import csv
    fields = ["venue", "seg_name", "status", "serve_detected", "serve_type",
              "hit_frame", "confidence", "serve_zone", "serving_side",
              "reception_detected", "reception_zone", "is_ace",
              "ball_detection_rate", "max_consecutive_ball_frames",
              "toss_height_px", "jump_height", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    log(f"[CSV] {csv_path}")


def export_pilot_summary(results, summary_path):
    total   = len(results)
    success = [r for r in results if r.get("status") == "success"]
    no_serve = [r for r in results if r.get("status") == "no_serve"]
    errors   = [r for r in results if r.get("status") not in ("success", "no_serve")]
    insuff   = [r for r in results if r.get("status") == "insufficient_ball_data"]

    # F-grade: quality_grade == 'F'  or  ball_detection_rate < 0.3
    f_grade = [r for r in results
               if r.get("quality_grade") == "F"
               or r.get("ball_detection_rate", 1.0) < 0.3]

    # 每場地統計
    venues = sorted(set(r.get("venue", "?") for r in results))
    per_venue = {}
    for v in venues:
        vr = [r for r in results if r.get("venue") == v]
        vs = [r for r in vr if r.get("status") == "success"]
        per_venue[v] = {
            "total":   len(vr),
            "success": len(vs),
            "success_rate": round(len(vs) / max(len(vr), 1), 3),
            "no_serve": sum(1 for r in vr if r.get("status") == "no_serve"),
            "error":   sum(1 for r in vr if r.get("status") not in ("success", "no_serve")),
        }

    # 失敗片段
    failed = [
        {
            "venue":    r.get("venue"),
            "seg_name": r.get("seg_name"),
            "status":   r.get("status"),
            "reason":   r.get("error") or r.get("status"),
            "ball_detection_rate": round(r.get("ball_detection_rate", 0), 4),
        }
        for r in results if r.get("status") != "success"
    ]

    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_segments": total,
        "success_count":  len(success),
        "success_rate":   round(len(success) / max(total, 1), 3),
        "no_serve_count": len(no_serve),
        "error_count":    len(errors),
        "insufficient_ball_data_count": len(insuff),
        "f_grade_count":  len(f_grade),
        "f_grade_rate":   round(len(f_grade) / max(total, 1), 3),
        "jump_serve_count":    sum(1 for r in success if r.get("is_jump_serve")),
        "standing_serve_count": sum(1 for r in success if not r.get("is_jump_serve")),
        "avg_confidence": round(
            sum(r.get("confidence", 0) for r in success) / max(len(success), 1), 3),
        "per_venue": per_venue,
        "failed_segments": failed,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"[JSON] {summary_path}")
    return summary


def print_summary(summary):
    log("\n" + "="*65)
    log("  Pilot 測試摘要")
    log("="*65)
    log(f"  總片段數  : {summary['total_segments']}")
    log(f"  成功偵測  : {summary['success_count']} ({summary['success_rate']:.1%})")
    log(f"  未偵測發球: {summary['no_serve_count']}")
    log(f"  F 級比例  : {summary['f_grade_count']}/{summary['total_segments']} "
        f"({summary['f_grade_rate']:.1%})")
    log(f"  跳發 / 站發: {summary['jump_serve_count']} / {summary['standing_serve_count']}")

    log("\n  --- 各場地 ---")
    for v, s in summary["per_venue"].items():
        log(f"  {v}: {s['success']}/{s['total']} ({s['success_rate']:.1%}) "
            f"  no_serve={s['no_serve']}  error={s['error']}")

    if summary["failed_segments"]:
        log("\n  --- 失敗片段 ---")
        for f in summary["failed_segments"]:
            log(f"  [{f['venue']}] {f['seg_name']}")
            log(f"    status={f['status']}  ball_rate={f['ball_detection_rate']:.1%}")
            if f.get('reason') and f['reason'] != f['status']:
                reason = str(f['reason'])[:120]
                log(f"    reason={reason}")
    log("="*65)


def main():
    log("="*65)
    log("  Pilot 測試管線 (各場地 5 個 normal_segment)")
    log("="*65)

    work_dir = os.path.join(OUTPUT_DIR, "_work")
    os.makedirs(work_dir, exist_ok=True)

    # 1. 選段
    log("\n[Step 1] 選取片段...")
    venue_candidates = collect_candidates()

    # 2. 確保追蹤 JSON
    log("\n[Step 2] 準備 tracking JSON（缺少的跑追蹤管線）...")
    segments_with_json = []
    for venue, segs in venue_candidates.items():
        log(f"\n  == {venue} ==")
        for seg in segs:
            json_path = prepare_tracking_json(seg, work_dir)
            if json_path:
                seg["json_path"] = json_path
                segments_with_json.append(seg)
            else:
                log(f"    [SKIP] {seg['seg_name']} (tracking 失敗)")

    log(f"\n  共 {len(segments_with_json)}/{sum(len(v) for v in venue_candidates.values())} 個片段有 JSON")

    # 3. 發球分析
    log("\n[Step 3] 發球分析...")
    analysis_dir = os.path.join(OUTPUT_DIR, "analysis")
    results = run_serve_analysis(segments_with_json, analysis_dir, work_dir)

    # 4. 輸出
    log("\n[Step 4] 輸出結果...")
    csv_path     = os.path.join(OUTPUT_DIR, "pilot_results.csv")
    summary_path = os.path.join(OUTPUT_DIR, "pilot_summary.json")
    export_csv(results, csv_path)
    summary = export_pilot_summary(results, summary_path)
    print_summary(summary)


if __name__ == "__main__":
    main()
