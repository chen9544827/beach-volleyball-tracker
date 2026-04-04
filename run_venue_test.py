# run_venue_test.py - Test one segment from each venue group
import os
import sys
import subprocess
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SEGMENTS_BASE = os.path.join(PROJECT_ROOT, "output_data", "video_segments")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_venue_output")
COURT_CONFIG = os.path.join(PROJECT_ROOT, "court_config.json")

# 5 test segments (one per match)
TEST_SEGMENTS = [
    {
        "name": "Edmonton_WT19_C4",
        "path": os.path.join(SEGMENTS_BASE, "Edmonton_WT19_C4",
            "FIVB_BVB_WT19_Edmonton_3Star_1718_C4_QT_W_007_Strauss_T_Strauss_N_AUT_Krebs_Welsch_GER_2",
            "normal_segments", "segment_010_Team1.mp4"),
    },
    {
        "name": "Chetumal_WT18_C1_match003",
        "path": os.path.join(SEGMENTS_BASE, "Chetumal_WT18_C1",
            "FIVB-BVB-WT18-Chetumal-4Star-251018-C1-MD-W-003-Menegatti-Orsi-Toth-ITA-Herdrick-Ibarra-MEX",
            "normal_segments", "segment_010_Team1.mp4"),
    },
    {
        "name": "Chetumal_WT18_C1_match005",
        "path": os.path.join(SEGMENTS_BASE, "Chetumal_WT18_C1",
            "FIVB-BVB-WT18-Chetumal-4Star-251018-C1-MD-W-005-Walsh-Jennings-Sweat-USA-Huber-Hubscher-SUI",
            "normal_segments", "segment_011_Team2.mp4"),
    },
    {
        "name": "Gstaad_WT19_C1",
        "path": os.path.join(SEGMENTS_BASE, "Gstaad_WT19_C1",
            "FIVB_BVB_WT19_Gstaad_5Star_090718_C1_QT_W_002_Dabizha_Rudykh_RUS_Bell_Ngauamo_AUS_high",
            "normal_segments", "segment_010_Team1.mp4"),
    },
    {
        "name": "Jinjiang_WT19_C1",
        "path": os.path.join(SEGMENTS_BASE, "Jinjiang_WT19_C1",
            "FIVB-BVB-WT19-Jinjiang-4Star-220518-C1-QT-W-009-Megan-Nicole-CAN-LIN-Lingling-J-M-Li-CHN",
            "normal_segments", "segment_010_Team1.mp4"),
    },
]

# Parse CLI arguments
_parser = argparse.ArgumentParser(description="Run venue tests")
_parser.add_argument("--use-sahi", action="store_true",
                     help="Enable SAHI sliced inference for player detection")
_args = _parser.parse_args()

os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, seg in enumerate(TEST_SEGMENTS):
    name = seg["name"]
    video_path = seg["path"]
    
    if not os.path.exists(video_path):
        print(f"\n[{i+1}/5] [SKIP] {name}: File not found: {video_path}")
        continue
    
    print(f"\n{'='*70}")
    print(f"  [{i+1}/5] {name}")
    print(f"  Video: {os.path.basename(video_path)}")
    print(f"{'='*70}")
    
    # Output subdirectory per venue
    venue_output = os.path.join(OUTPUT_DIR, name)
    tracking_output = os.path.join(venue_output, "tracking")
    serve_output = os.path.join(venue_output, "serve_analysis")
    os.makedirs(tracking_output, exist_ok=True)
    os.makedirs(serve_output, exist_ok=True)
    
    # Step 1: Tracking
    print(f"\n  [1/3] Tracking...", flush=True)
    t0 = time.time()
    cmd_track = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "video_processing", "track_ball_and_player_v2.py"),
        "--input", video_path,
        "--output_dir", tracking_output,
    ]
    if _args.use_sahi:
        cmd_track.append("--use_sahi")
    result = subprocess.run(cmd_track, cwd=PROJECT_ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            errors='replace')
    t1 = time.time()

    if result.returncode != 0:
        print(f"  [ERROR] Tracking failed: {result.stderr[-500:] if result.stderr else 'unknown'}")
        continue
    print(f"  [OK] Tracking done ({t1-t0:.1f}s)", flush=True)
    
    # Find the JSON output
    json_files = [f for f in os.listdir(tracking_output) if f.endswith('.json')]
    if not json_files:
        print(f"  [ERROR] No JSON output found")
        continue
    json_path = os.path.join(tracking_output, json_files[0])
    print(f"  JSON: {json_files[0]}")
    
    # Step 2: Serve analysis (without court_config for now - we want to see raw results)
    print(f"\n  [2/3] Serve analysis...", flush=True)
    t0 = time.time()
    # Only analyze the specific segment file, not all files in the directory
    # We use a temp dir with symlink to the specific segment
    serve_video_dir = os.path.join(venue_output, "serve_video_input")
    os.makedirs(serve_video_dir, exist_ok=True)
    seg_basename = os.path.basename(video_path)
    target_link = os.path.join(serve_video_dir, seg_basename)
    if not os.path.exists(target_link):
        import shutil
        shutil.copy2(video_path, target_link)
    cmd_serve = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "batch_test_serve.py"),
        "--video-dir", serve_video_dir,
        "--json-dir", tracking_output,
        "--output", serve_output,
        "--verbose",
    ]
    result = subprocess.run(cmd_serve, cwd=PROJECT_ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            errors='replace')
    t1 = time.time()
    print(f"  [OK] Serve analysis done ({t1-t0:.1f}s)", flush=True)

    # Print serve analysis summary from stdout
    if result.stdout:
        for line in result.stdout.split('\n'):
            if any(kw in line for kw in ['Serve detected', 'serve_detected', 'FOUND', 'NO SERVE',
                                           'Hit speed', 'Jump serve', 'ball_detection_rate',
                                           'Summary', 'Total', 'Detection rate', 'serve events',
                                           'Toss', 'Hit frame', 'Server']):
                print(f"    {line.strip()}")
    
    # Step 3: Visualize tracking as video
    print(f"\n  [3/3] Generating annotated video...", flush=True)
    t0 = time.time()
    video_output_path = os.path.join(venue_output, f"{name}_annotated.mp4")
    cmd_vis = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "visualize_tracking.py"),
        "--video", video_path,
        "--json", json_path,
        "--output", video_output_path,
    ]
    result = subprocess.run(cmd_vis, cwd=PROJECT_ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            errors='replace')
    t1 = time.time()

    if result.returncode == 0 and os.path.exists(video_output_path):
        size_mb = os.path.getsize(video_output_path) / (1024*1024)
        print(f"  [OK] Annotated video: {video_output_path} ({size_mb:.1f} MB)")
    else:
        print(f"  [WARNING] Video generation issue: {result.stderr[-300:] if result.stderr else 'check output'}")

print(f"\n{'='*70}")
print(f"  All done! Results in: {OUTPUT_DIR}")
print(f"{'='*70}")

# List all outputs
for name_dir in sorted(os.listdir(OUTPUT_DIR)):
    dir_path = os.path.join(OUTPUT_DIR, name_dir)
    if os.path.isdir(dir_path):
        print(f"\n  {name_dir}/")
        for root, dirs, files in os.walk(dir_path):
            level = root.replace(dir_path, '').count(os.sep)
            indent = '    ' + '  ' * level
            for f in sorted(files):
                fpath = os.path.join(root, f)
                size = os.path.getsize(fpath) / 1024
                unit = "KB"
                if size > 1024:
                    size /= 1024
                    unit = "MB"
                print(f"{indent}{f} ({size:.1f} {unit})")
