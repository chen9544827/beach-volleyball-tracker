# batch_process_videos.py
import os
import subprocess
import argparse
import sys
from datetime import datetime

def find_video_files(directory):
    """Finds all common video files in the specified directory."""
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                video_files.append(os.path.join(root, file))
    return video_files

def main():
    parser = argparse.ArgumentParser(description="[Final Stable Version] Automated batch processing of volleyball videos.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing video files.")
    parser.add_argument("--output_folder", type=str, default="batch_processing_results", help="Main output directory for all results and logs.")
    
    # --- Analysis parameters that can be tuned ---
    parser.add_argument("--hit_dist", type=float, default=320, help="[Analysis Param] Max contact distance for a hit.")
    parser.add_argument("--wrist_dist", type=float, default=50, help="[Analysis Param] Max distance between wrists for a held pose.")
    parser.add_argument("--toss_vel", type=float, default=5.0, help="[Analysis Param] Min upward velocity for a toss.")
    
    # --- Debug options ---
    parser.add_argument("--save_annotated_frames", action="store_true", help="[Debug] Save each annotated frame as an image file.")
    parser.add_argument("--save_original_frames", action="store_true", help="[Training] Save each original frame as an image file.")
    args = parser.parse_args()

    script_start_time = datetime.now()
    print(f"--- Batch Processing Started at: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    video_files = find_video_files(args.input_folder)
    if not video_files:
        print(f"[ERROR] No supported video files found in '{args.input_folder}'.")
        return

    total_videos = len(video_files)
    print(f"[INFO] Found {total_videos} video files. Starting...")

    base_output_dir = os.path.abspath(args.output_folder)
    os.makedirs(base_output_dir, exist_ok=True)
    
    for idx, video_path in enumerate(video_files):
        video_start_time = datetime.now()
        print(f"\n--- [{idx + 1}/{total_videos}] Processing Video: {os.path.basename(video_path)} ---")
        
        video_base_name = os.path.splitext(os.path.basename(video_path))[0]
        video_specific_output_dir = os.path.join(base_output_dir, video_base_name)
        os.makedirs(video_specific_output_dir, exist_ok=True)
        log_file_path = os.path.join(video_specific_output_dir, f"{video_base_name}_processing.log")
        
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Processing log for: {video_path}\nStarted at: {video_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*50 + "\n\n")

            # --- Stage 1: Tracking ---
            print("  [Step 1/2] Executing Object & Pose Tracking... (logs will be written to file)")
            tracking_output_path = os.path.join(video_specific_output_dir, 'tracking_output')
            
            track_command = [
                sys.executable, os.path.join("video_processing", "track_ball_and_player.py"),
                "--input", video_path, "--output_dir", tracking_output_path
            ]
            if args.save_annotated_frames: track_command.append("--save_annotated_frames")
            if args.save_original_frames: track_command.append("--save_original_frames")
            
            result_track = subprocess.run(track_command, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            log_file.write("--- Stage 1: Tracking (track_ball_and_player.py) ---\n" + result_track.stdout + "\n" + result_track.stderr)
            log_file.flush()

            if result_track.returncode != 0:
                print(f"  [ERROR] Tracking stage failed! See log for details: {log_file_path}"); continue
            print("  [OK] Tracking complete!")

            # --- Stage 2: Analysis ---
            print("  [Step 2/2] Executing Serve Event Analysis... (logs will be written to file)")
            analysis_output_path = os.path.join(video_specific_output_dir, 'analysis_output')
            json_input_path = os.path.join(tracking_output_path, video_base_name, f"{video_base_name}_all_frames_data_with_pose.json")

            if not os.path.exists(json_input_path):
                 print(f"  [ERROR] Cannot find JSON file. Skipping analysis."); continue

            analyze_command = [
                sys.executable, os.path.join("video_processing", "test_serve_analyzer.py"),
                "--video_input", video_path,
                "--json_input", json_input_path,
                "--output_dir", analysis_output_path,
                "--hit_dist", str(args.hit_dist),
                "--wrist_dist", str(args.wrist_dist),
                "--toss_vel", str(args.toss_vel)
            ]
            result_analyze = subprocess.run(analyze_command, capture_output=True, text=True, encoding='utf-8', errors='ignore')

            log_file.write("\n\n--- Stage 2: Serve Analysis (test_serve_analyzer.py) ---\n" + result_analyze.stdout + "\n" + result_analyze.stderr)
            log_file.flush()

            if result_analyze.returncode != 0:
                print(f"  [ERROR] Analysis stage failed! See log for details: {log_file_path}"); continue
            print("  [OK] Analysis complete!")
            
            video_end_time = datetime.now()
            log_file.write(f"\n\nFinished at: {video_end_time.strftime('%Y-%m-%d %H:%M:%S')}\nTotal time: {video_end_time - video_start_time}\n")
            print(f"  [LOG] Full processing log saved to: {log_file_path}")

    print(f"\n--- All videos processed! Total time: {datetime.now() - script_start_time} ---")

if __name__ == '__main__':
    main()