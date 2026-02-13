# batch_court_config.py
# -*- coding: utf-8 -*-
"""
批次 court_config 設定工具

掃描影片目錄 -> 檔名解析分組 -> 檢查已設定 -> 逐一開啟 GUI 設定。
每個場地組 (venue_year_court) 只需設定一次。

支援自動偵測模式（--auto），使用 YOLO keypoint 模型自動偵測場地邊界。

用法：
    # 檢視分組狀態（不啟動 GUI）
    python batch_court_config.py --video-dir input_video/original_video --output-dir court_configs --dry-run

    # 互動式設定（逐一開啟 GUI）
    python batch_court_config.py --video-dir input_video/original_video --output-dir court_configs

    # 自動偵測模式
    python batch_court_config.py --video-dir input_video/original_video --output-dir court_configs --auto

    # 自動偵測 + 人工驗證
    python batch_court_config.py --video-dir input_video/original_video --output-dir court_configs --auto --verify

    # 使用已分割片段目錄
    python batch_court_config.py --segment-dir output_data/video_segments --output-dir court_configs
"""

import os
import sys
import json
import subprocess
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.filename_parser import scan_video_directory, group_videos, parse_filename


def get_reference_video(group_entries, segment_dir=None):
    """
    取得該組的參考影片路徑（用於 GUI 設定）

    優先使用 segment_dir 中的片段（較短，開啟更快），
    否則使用原始影片目錄中的第一部影片。

    Args:
        group_entries: group_videos() 中某組的 entries 列表
        segment_dir: 已分割片段目錄（可選）

    Returns:
        參考影片路徑，找不到時回傳 None
    """
    if segment_dir and os.path.isdir(segment_dir):
        # 嘗試在 segment_dir 中找到該組的片段
        group_key = group_entries[0]['parsed']['group_key'] if group_entries[0].get('parsed') else None
        if group_key:
            # 搜尋 segment_dir 下所有子目錄
            for root, dirs, files in os.walk(segment_dir):
                for f in sorted(files):
                    if not any(f.lower().endswith(ext) for ext in ('.mp4', '.avi', '.mkv', '.mov')):
                        continue
                    parsed = parse_filename(f)
                    if parsed and parsed.get('group_key') == group_key:
                        return os.path.join(root, f)

    # fallback: 使用原始影片目錄中的第一部
    return group_entries[0]['path']


def auto_detect_court(detector, ref_video, config_path, group_key, num_entries,
                      min_confidence, verify, margin_lr, margin_far, margin_near):
    """
    使用模型自動偵測場地邊界並生成 court_config。

    Args:
        detector: AutoCourtDetector instance
        ref_video: Reference video path
        config_path: Output config file path
        group_key: Group key string
        num_entries: Number of videos in this group
        min_confidence: Minimum confidence threshold
        verify: Whether to show visual verification
        margin_lr: Left/right exclusion margin ratio
        margin_far: Far-side exclusion margin ratio
        margin_near: Near-side exclusion margin ratio

    Returns:
        'accepted' | 'rejected' | 'low_confidence' | 'error'
    """
    # Run detection
    detection = detector.detect_from_video(ref_video, sample_frames=5)

    if detection.get('keypoints') is None:
        print(f"  [FAIL] No court detected (0/{5} frames)")
        return 'error'

    conf = detection['confidence']
    n_det = detection['num_detections']
    print(f"  Confidence: {conf:.3f} ({n_det}/5 frames detected)")

    if conf < min_confidence:
        print(f"  [LOW-CONF] Below threshold {min_confidence:.2f}, skipping auto")
        return 'low_confidence'

    # Generate court config
    court_config = detector.generate_court_config(
        detection,
        margin_lr=margin_lr,
        margin_far=margin_far,
        margin_near=margin_near,
    )

    if court_config is None:
        print(f"  [FAIL] Could not generate court config")
        return 'error'

    if verify:
        return _verify_detection(detector, ref_video, detection, court_config,
                                 config_path, group_key)
    else:
        # Auto-accept
        from core.auto_court_detector import save_court_config
        save_court_config(court_config, config_path)
        print(f"  [OK] Auto-saved: {config_path}")
        return 'accepted'


def _verify_detection(detector, ref_video, detection, court_config,
                      config_path, group_key):
    """
    Show detection visualization for manual verification.

    Returns:
        'accepted' | 'rejected'
    """
    import cv2

    # Read a frame for visualization
    cap = cv2.VideoCapture(ref_video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"  [ERROR] Cannot read frame for verification")
        return 'rejected'

    vis = detector.visualize_detection(frame, detection, court_config)

    window_name = f"Verify: {group_key}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.putText(vis, "'a' = Accept, 'r' = Reject (open GUI)", (20, vis.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow(window_name, vis)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('a'):
            cv2.destroyWindow(window_name)
            from core.auto_court_detector import save_court_config
            save_court_config(court_config, config_path)
            print(f"  [OK] Accepted and saved: {config_path}")
            return 'accepted'
        elif key == ord('r'):
            cv2.destroyWindow(window_name)
            print(f"  [REJECT] Manual review needed")
            return 'rejected'
        elif key == 27:  # ESC
            cv2.destroyWindow(window_name)
            print(f"  [REJECT] Cancelled")
            return 'rejected'


def run_batch_court_config(video_dir, output_dir, dry_run=False,
                           skip_configured=True, segment_dir=None,
                           auto=False, auto_model=None, verify=False,
                           min_confidence=0.7, margin_lr=0.15,
                           margin_far=0.30, margin_near=0.25):
    """
    批次設定 court_config

    Args:
        video_dir: 影片目錄
        output_dir: court_config 輸出目錄
        dry_run: 只顯示狀態，不啟動 GUI
        skip_configured: 跳過已有 config 的組
        segment_dir: 已分割片段目錄（用於替代原始影片）
        auto: 使用模型自動偵測
        auto_model: 自動偵測模型路徑
        verify: 自動偵測後顯示視覺化供人工確認
        min_confidence: 自動偵測最低信心度
        margin_lr: 左右邊距比例
        margin_far: 遠端邊距比例
        margin_near: 近端邊距比例
    """
    print("=" * 70)
    print("批次 court_config 設定工具")
    print("=" * 70)
    print(f"影片目錄: {video_dir}")
    print(f"輸出目錄: {output_dir}")
    if segment_dir:
        print(f"片段目錄: {segment_dir}")
    if auto:
        print(f"模式: 自動偵測 (model: {auto_model})")
        print(f"  min_confidence={min_confidence}, verify={verify}")
        print(f"  margins: lr={margin_lr}, far={margin_far}, near={margin_near}")
    if dry_run:
        print("[DRY-RUN] 只顯示狀態，不啟動 GUI")
    print()

    # Load auto detector if needed
    detector = None
    if auto and not dry_run:
        if not auto_model or not os.path.exists(auto_model):
            print(f"[ERROR] Auto model not found: {auto_model}")
            print("  Train with: python tools/train_court_detector.py")
            print("  Falling back to GUI mode.")
            auto = False
        else:
            from core.auto_court_detector import AutoCourtDetector
            detector = AutoCourtDetector(auto_model)
            print(f"[OK] Auto detector loaded: {auto_model}")
            print()

    # 1. 掃描 + 分組
    videos = scan_video_directory(video_dir)
    if not videos:
        print("[ERROR] 在影片目錄中找不到影片檔案")
        return

    groups = group_videos(videos)
    if not groups:
        print("[ERROR] 無法解析任何影片檔名")
        return

    # 統計
    parseable_groups = {k: v for k, v in groups.items() if k != '_unparsed'}
    unparsed = groups.get('_unparsed', [])

    print(f"掃描到 {len(videos)} 部影片, {len(parseable_groups)} 個場地組")
    if unparsed:
        print(f"  ({len(unparsed)} 部無法解析檔名)")
    print()

    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)

    # 2. 檢查各組設定狀態
    configured = 0
    pending = 0
    setup_count = 0
    auto_accepted = 0
    auto_rejected = 0
    auto_low_conf = 0

    print("-" * 70)
    print(f"{'Group Key':<35} {'Videos':>6}  {'Status':<20}")
    print("-" * 70)

    for group_key in sorted(parseable_groups.keys()):
        entries = parseable_groups[group_key]
        config_path = os.path.join(output_dir, f"{group_key}.json")

        if os.path.exists(config_path):
            configured += 1
            status = "[SKIP] configured"
            print(f"{group_key:<35} {len(entries):>6}  {status}")
            continue

        pending += 1
        if dry_run:
            status = "[PENDING] needs setup"
            print(f"{group_key:<35} {len(entries):>6}  {status}")
            continue

        # 取得參考影片
        ref_video = get_reference_video(entries, segment_dir)
        if not ref_video or not os.path.exists(ref_video):
            status = "[ERROR] no video found"
            print(f"{group_key:<35} {len(entries):>6}  {status}")
            continue

        # 自動偵測模式
        if auto and detector is not None:
            print(f"{group_key:<35} {len(entries):>6}  [AUTO] detecting...")
            print(f"  Reference video: {os.path.basename(ref_video)}")

            result = auto_detect_court(
                detector, ref_video, config_path, group_key, len(entries),
                min_confidence, verify, margin_lr, margin_far, margin_near,
            )

            if result == 'accepted':
                setup_count += 1
                auto_accepted += 1
                continue
            elif result == 'low_confidence':
                auto_low_conf += 1
                # Fall through to GUI if not in auto-only mode
                if not verify:
                    print(f"  [SKIP] Use --verify to manually review low-confidence results")
                    continue
                print(f"  Falling back to GUI...")
            elif result == 'rejected':
                auto_rejected += 1
                print(f"  Falling back to GUI...")
            else:
                # error
                print(f"  Falling back to GUI...")

        # GUI 設定 (fallback or primary)
        print(f"{group_key:<35} {len(entries):>6}  [SETUP] launching GUI...")
        print(f"  Reference video: {os.path.basename(ref_video)}")

        gui_script = os.path.join(PROJECT_ROOT, "court_definition", "court_config_generator.py")
        result = subprocess.run([
            sys.executable,
            gui_script,
            "--video_path", ref_video,
            "--output_path", config_path
        ])

        if result.returncode == 0 and os.path.exists(config_path):
            setup_count += 1
            print(f"  [OK] Saved: {config_path}")
        else:
            print(f"  [WARNING] GUI exited without saving (returncode={result.returncode})")

    # 3. 摘要
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total groups:       {len(parseable_groups)}")
    print(f"Already configured: {configured}")
    print(f"Pending setup:      {pending}")
    if not dry_run:
        print(f"Newly configured:   {setup_count}")
        if auto:
            print(f"  Auto accepted:    {auto_accepted}")
            print(f"  Auto rejected:    {auto_rejected}")
            print(f"  Low confidence:   {auto_low_conf}")

    remaining = pending - setup_count if not dry_run else pending
    if remaining > 0:
        print(f"\n[INFO] {remaining} groups still need court_config setup.")
        print(f"  Re-run this script to continue setting up remaining groups.")


def main():
    parser = argparse.ArgumentParser(
        description="批次 court_config 設定工具 - 按場地組逐一設定場地邊界"
    )
    parser.add_argument("--video-dir", type=str, required=True,
                        help="影片目錄路徑")
    parser.add_argument("--output-dir", type=str, default="court_configs",
                        help="court_config 輸出目錄 (預設: court_configs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只顯示分組狀態，不啟動 GUI")
    parser.add_argument("--skip-configured", action="store_true", default=True,
                        help="跳過已有 config 的組 (預設: True)")
    parser.add_argument("--segment-dir", type=str, default=None,
                        help="已分割片段目錄（用較短片段替代原始影片開啟 GUI）")

    # Auto detection arguments
    auto_group = parser.add_argument_group('auto detection',
                                            '自動場地偵測模式參數')
    auto_group.add_argument("--auto", action="store_true",
                            help="使用 YOLO keypoint 模型自動偵測場地邊界")
    auto_group.add_argument("--auto-model", type=str, default="models/court_best.pt",
                            help="自動偵測模型路徑 (預設: models/court_best.pt)")
    auto_group.add_argument("--verify", action="store_true",
                            help="自動偵測後顯示視覺化供人工確認")
    auto_group.add_argument("--min-confidence", type=float, default=0.7,
                            help="自動偵測最低信心度 (預設: 0.7)")
    auto_group.add_argument("--margin-lr", type=float, default=0.15,
                            help="左右邊距比例 (預設: 0.15)")
    auto_group.add_argument("--margin-far", type=float, default=0.30,
                            help="遠端邊距比例 (預設: 0.30)")
    auto_group.add_argument("--margin-near", type=float, default=0.25,
                            help="近端邊距比例 (預設: 0.25)")

    args = parser.parse_args()

    run_batch_court_config(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        skip_configured=args.skip_configured,
        segment_dir=args.segment_dir,
        auto=args.auto,
        auto_model=args.auto_model,
        verify=args.verify,
        min_confidence=args.min_confidence,
        margin_lr=args.margin_lr,
        margin_far=args.margin_far,
        margin_near=args.margin_near,
    )


if __name__ == "__main__":
    main()
