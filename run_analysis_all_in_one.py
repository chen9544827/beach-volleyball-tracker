# -*- coding: utf-8 -*-
# run_analysis_all_in_one.py (最終定義版，包含自動總結報告功能)

import os
import subprocess
import argparse
import sys
import json
import numpy as np
import cv2
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque
import io

# --- 自動安裝並匯入 tqdm ---
try:
    from tqdm import tqdm
except ImportError:
    print("模組 'tqdm' 未找到，正在嘗試自動安裝...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        from tqdm import tqdm; print("tqdm 安裝成功！")
    except Exception as e:
        print(f"自動安裝 tqdm 失敗，請手動執行 'pip install tqdm'。錯誤: {e}"); sys.exit(1)

# ==============================================================================
#  分析邏輯 (Event Analysis Logic) - 內建全新三段式智慧邏輯
# ==============================================================================

def get_ball_center_from_data(frame_data):
    if frame_data and frame_data.get('ball_detections'):
        best_ball = max(frame_data['ball_detections'], key=lambda b: b.get('confidence', 0))
        box = best_ball['box_coords']
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    return None

def analyze_serve_events(all_frames_data, config, log_prefix=""):
    toss_initial_vy_thresh = config.get("toss_initial_vy", 8.0)
    frames_to_validate_toss = config.get("frames_to_validate", 8)
    min_upward_confirms = config.get("min_upward_confirms", 3)
    vertical_ratio_thresh = config.get("vertical_ratio", 1.5)
    hit_v_thresh = config.get("hit_v", 35.0)
    max_frames_between_toss_and_hit = config.get("max_frames_to_hit", 75)
    max_lost_frames = config.get("max_lost_frames", 10)

    serve_events = []
    state = "SEARCHING_TOSS"
    event_candidate = {}
    
    # 不再於此處打印，日誌將由主工作單元統一處理
    # print(f"\n{log_prefix}[分析階段] 使用最終版三段式智慧邏輯進行分析...")

    for i in range(1, len(all_frames_data)):
        prev_ball_pos = get_ball_center_from_data(all_frames_data[i-1])
        curr_ball_pos = get_ball_center_from_data(all_frames_data[i])

        if state == "SEARCHING_TOSS":
            if prev_ball_pos is None or curr_ball_pos is None: continue
            vy = prev_ball_pos[1] - curr_ball_pos[1]
            vx = curr_ball_pos[0] - prev_ball_pos[0]
            if vy > toss_initial_vy_thresh and (abs(vy) / (abs(vx) + 1e-6)) > vertical_ratio_thresh:
                print(f"{log_prefix}  > [第 {i} 幀] 偵測到疑似拋球 (vy={vy:.1f}) -> 進入 確認軌跡 狀態")
                state = "CONFIRMING_TOSS"
                event_candidate = {'confirm_start_frame': i, 'upward_frames_count': 1, 'lost_frames_count': 0, 'last_pos': curr_ball_pos}

        elif state == "CONFIRMING_TOSS":
            if (i - event_candidate['confirm_start_frame']) > frames_to_validate_toss:
                print(f"{log_prefix}  > [第 {i} 幀][重設] 確認期結束，證據不足")
                state = "SEARCHING_TOSS"; continue
            if curr_ball_pos is None:
                event_candidate['lost_frames_count'] += 1
                if event_candidate['lost_frames_count'] > 5:
                    print(f"{log_prefix}  > [第 {i} 幀][重設] 確認期間球消失太久")
                    state = "SEARCHING_TOSS"
                continue
            vy = event_candidate['last_pos'][1] - curr_ball_pos[1]
            if vy > 0: event_candidate['upward_frames_count'] += 1
            event_candidate['last_pos'] = curr_ball_pos
            if event_candidate['upward_frames_count'] >= min_upward_confirms:
                print(f"{log_prefix}  > [第 {i} 幀] 確認拋球成功 (已累積 {event_candidate['upward_frames_count']} 個向上幀) -> 進入 等待擊球 狀態")
                state = "AWAITING_HIT"
                event_candidate['toss_frame'] = event_candidate['confirm_start_frame']
                event_candidate['lost_frames_count'] = 0

        elif state == "AWAITING_HIT":
            if (i - event_candidate['toss_frame']) > max_frames_between_toss_and_hit:
                print(f"{log_prefix}  > [第 {i} 幀][重設] 等待擊球超時")
                state = "SEARCHING_TOSS"; continue
            if curr_ball_pos is None:
                event_candidate['lost_frames_count'] += 1
                if event_candidate['lost_frames_count'] > max_lost_frames:
                    print(f"{log_prefix}  > [第 {i} 幀][重設] 等待期間球消失太久")
                    state = "SEARCHING_TOSS"
                continue
            event_candidate['lost_frames_count'] = 0
            if prev_ball_pos is not None:
                speed = np.linalg.norm(curr_ball_pos - prev_ball_pos)
                if speed > hit_v_thresh:
                    print(f"{log_prefix}  > [第 {i} 幀] 偵測到高速位移 (速度: {speed:.1f}) -> 判定為發球！")
                    server_player = None
                    if all_frames_data[i].get('player_detections'):
                        server_player = min(all_frames_data[i]['player_detections'], key=lambda p: np.linalg.norm(np.array(p['center_point']) - curr_ball_pos))
                    serve_events.append({"frame_id": i, "hit_position": curr_ball_pos.tolist(), "hit_speed": speed, "server_player_data": server_player})
                    state = "SEARCHING_TOSS"
    
    # print(f"{log_prefix}[分析階段] 分析完成，找到 {len(serve_events)} 個發球事件。")
    return serve_events

# ==============================================================================
#  核心處理工作單元 (The Main Worker Unit)
# ==============================================================================
def process_single_video(video_path, base_output_dir, overwrite, frame_args, analysis_args):
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    log_prefix = f"[{video_base_name}] "
    try:
        video_specific_output_dir = os.path.join(base_output_dir, video_base_name)
        tracking_output_path = os.path.join(video_specific_output_dir, 'tracking_output')
        analysis_output_path = os.path.join(video_specific_output_dir, 'analysis_output')
        final_video_output = os.path.join(analysis_output_path, f"{video_base_name}_analysis.mp4")
        json_input_path = os.path.join(tracking_output_path, video_base_name, f"{video_base_name}_all_frames_data_with_pose.json")
        if not overwrite and os.path.exists(final_video_output):
            return {"video": video_base_name, "status": "skipped", "log": "最終輸出檔案已存在。", "events": []}
        os.makedirs(analysis_output_path, exist_ok=True)
        print(f"{log_prefix}開始執行 Stage 1: 物件追蹤...")
        track_command = [sys.executable, os.path.join("video_processing", "track_ball_and_player.py"), "--input", video_path, "--output_dir", tracking_output_path]
        if frame_args.get('save_all_frames'): track_command.append("--save_all_frames")
        result_track = subprocess.run(track_command, capture_output=True)
        stdout_track = result_track.stdout.decode('utf-8', errors='ignore')
        stderr_track = result_track.stderr.decode('utf-8', errors='ignore')
        if result_track.returncode != 0: raise subprocess.CalledProcessError(result_track.returncode, track_command, output=stdout_track, stderr=stderr_track)
        print(f"{log_prefix}物件追蹤完成。")
        if not os.path.exists(json_input_path): raise FileNotFoundError(f"追蹤後所需的JSON檔案不存在: {json_input_path}")
        with open(json_input_path, 'r', encoding='utf-8') as f: all_frames_data = json.load(f)
        analysis_config = {"hit_v_thresh": analysis_args['hit_v'], "vertical_ratio_thresh": analysis_args['vertical_ratio'], "toss_initial_vy": analysis_args['toss_vy']}
        old_stdout = sys.stdout; sys.stdout = analysis_log_buffer = io.StringIO()
        serve_events = analyze_serve_events(all_frames_data, analysis_config, log_prefix)
        sys.stdout = old_stdout; analysis_log_content = analysis_log_buffer.getvalue()
        print(f"{log_prefix}開始產生最終視覺化影片...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise IOError(f"無法開啟影片檔案: {video_path}")
        fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(final_video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        events_by_frame = {e['frame_id']: e for e in serve_events}
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_idx in events_by_frame:
                event = events_by_frame[frame_idx]
                hit_pos = tuple(map(int, event['hit_position']))
                cv2.circle(frame, hit_pos, 40, (0, 255, 0), 4)
                cv2.putText(frame, f"SERVE! (v={event['hit_speed']:.1f})", (hit_pos[0] - 100, hit_pos[1] - 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
            writer.write(frame)
            frame_idx += 1
        cap.release(); writer.release()
        print(f"{log_prefix}影片產生完成: {final_video_output}")
        with open(os.path.join(video_specific_output_dir, "processing.log"), 'w', encoding='utf-8') as f:
            f.write("--- STAGE 1: TRACKING STDOUT ---\n"); f.write(stdout_track)
            f.write("\n--- STAGE 1: TRACKING STDERR ---\n"); f.write(stderr_track)
            f.write("\n\n--- STAGE 2: ANALYSIS LOG ---\n"); f.write(analysis_log_content)
        # ✨ ---【核心修改】回傳 serve_events 以便後續處理 --- ✨
        return {"video": video_base_name, "status": "success", "log": final_video_output, "events": serve_events}
    except Exception as e:
        error_message = f"處理影片 {video_base_name} 時發生嚴重錯誤: {e}"
        if isinstance(e, subprocess.CalledProcessError): error_message += f"\n--- 子程序標準輸出 ---\n{e.output}\n\n--- 子程序標準錯誤（Traceback） ---\n{e.stderr}"
        return {"video": video_base_name, "status": "failed", "log": error_message, "events": []}

# ✨ ---【全新功能】總結報告產生器 --- ✨
def create_first_hit_summary(results, base_output_dir, input_folder):
    """
    在所有影片處理完畢後，建立一個包含首次擊球關鍵幀的總結報告。
    """
    print("\n" + "="*80)
    print("--- 正在產生首次擊球總結報告... ---")
    
    summary_dir = os.path.join(base_output_dir, "first_hit_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_txt_path = os.path.join(summary_dir, "first_hit_frames.txt")
    
    # 預先尋找所有影片的完整路徑
    video_path_map = {os.path.splitext(os.path.basename(p))[0]: p for p in find_video_files(input_folder)}
    
    count = 0
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write("--- 每個影片片段首次偵測到的擊球幀數 ---\n\n")
        
        for result in sorted(results, key=lambda r: r['video']): # 確保順序
            if result['status'] == 'success' and result.get('events'):
                video_base_name = result['video']
                first_event = result['events'][0]
                hit_frame_num = first_event['frame_id']
                
                f.write(f"{video_base_name}: {hit_frame_num}\n")
                
                # --- 擷取關鍵幀 ---
                original_video_path = video_path_map.get(video_base_name)
                if not original_video_path:
                    print(f"[警告] 找不到原始影片檔案: {video_base_name}")
                    continue
                
                try:
                    cap = cv2.VideoCapture(original_video_path)
                    if not cap.isOpened(): continue
                        
                    start_frame = max(0, hit_frame_num - 3)
                    end_frame = hit_frame_num + 3
                    
                    for frame_to_capture in range(start_frame, end_frame + 1):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_capture)
                        ret, frame = cap.read()
                        if ret:
                            tag = ""
                            if frame_to_capture == hit_frame_num:
                                tag = "_HIT"
                            elif frame_to_capture < hit_frame_num:
                                tag = f"_PRE_{hit_frame_num - frame_to_capture}"
                            else: # frame_to_capture > hit_frame_num
                                tag = f"_POST_{frame_to_capture - hit_frame_num}"
                            
                            img_name = f"{video_base_name}_frame_{frame_to_capture:06d}{tag}.jpg"
                            cv2.imwrite(os.path.join(summary_dir, img_name), frame)
                    
                    cap.release()
                    count += 1
                except Exception as e:
                    print(f"[錯誤] 擷取 {video_base_name} 的關鍵幀時失敗: {e}")

    print(f"總結報告產生完畢！共擷取了 {count} 個影片的關鍵幀。")
    print(f"詳細資訊請見: {summary_dir}")
# ✨ ------------------------------------- ✨


# ==============================================================================
#  主程式與批次處理邏輯 (Main Program and Batch Logic)
# ==============================================================================
def find_video_files(directory):
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                video_files.append(os.path.join(root, file))
    return video_files

def main():
    parser = argparse.ArgumentParser(description="[終極定義版] 自動化分析排球影片，並產生總結報告。")
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="volleyball_analysis_results")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    
    analysis_group = parser.add_argument_group('Analysis Parameters')
    analysis_group.add_argument("--hit_v", type=float, default=35.0)
    analysis_group.add_argument("--vertical_ratio", type=float, default=1.5)
    analysis_group.add_argument("--toss_vy", type=float, default=8.0)
    
    frame_group = parser.add_argument_group('Frame Saving Options')
    frame_group.add_argument("--save_all_frames", action="store_true")
    
    args = parser.parse_args()
    frame_args = {'save_all_frames': args.save_all_frames}
    analysis_args = {'hit_v': args.hit_v, 'vertical_ratio': args.vertical_ratio, 'toss_vy': args.toss_vy}
    
    script_start_time = datetime.now()
    print(f"--- 批次處理開始於: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    video_files = find_video_files(args.input_folder)
    if not video_files: print(f"[錯誤] 在 '{args.input_folder}' 中找不到任何支援的影片檔案。"); return
    base_output_dir = os.path.abspath(args.output_folder)
    os.makedirs(base_output_dir, exist_ok=True)
    max_workers = args.workers if args.workers and args.workers > 0 else os.cpu_count()
    print(f"[資訊] 找到 {len(video_files)} 個影片。將使用最多 {max_workers} 個核心進行平行處理。")
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video, video_path, base_output_dir, args.overwrite, frame_args, analysis_args): video_path for video_path in video_files}
        with tqdm(total=len(futures), desc="整體進度", unit="video") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
                try: results.append(future.result())
                except Exception as exc:
                    video_path = futures[future]
                    results.append({"video": os.path.basename(video_path), "status": "failed", "log": f"執行時發生嚴重錯誤: {exc}", "events": []})
    
    # ✨ ---【核心修改】在所有任務完成後，呼叫總結報告產生器 --- ✨
    create_first_hit_summary(results, base_output_dir, args.input_folder)

    print("\n" + "="*80)
    print("--- 所有任務已完成，正在生成摘要報告... ---")
    summary_path = os.path.join(base_output_dir, "summary_report.txt")
    
    success_videos = [r for r in results if r['status'] == 'success']
    failed_videos = [r for r in results if r['status'] == 'failed']
    skipped_videos = [r for r in results if r['status'] == 'skipped']
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"批次處理執行摘要\n"); f.write(f"執行時間: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"總耗時: {datetime.now() - script_start_time}\n"); f.write("="*60 + "\n\n")
        f.write(f"--- ✅ 處理成功 ({len(success_videos)}) ---\n")
        for res in success_videos: f.write(f"- {res['video']} -> {res['log']}\n")
        f.write(f"\n--- ⏭️ 自動跳過 ({len(skipped_videos)}) ---\n")
        for res in skipped_videos: f.write(f"- {res['video']}\n")
        f.write(f"\n--- ❌ 處理失敗 ({len(failed_videos)}) ---\n")
        for res in failed_videos:
             f.write(f"--- Video: {res['video']} ---\n"); f.write(f"{res['log']}\n\n")
    print(f"摘要報告已生成於: {summary_path}")
    print(f"執行結果: {len(success_videos)} 成功, {len(failed_videos)} 失敗, {len(skipped_videos)} 跳過。")
    print("="*80)

if __name__ == '__main__':
    main()