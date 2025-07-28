# -*- coding: utf-8 -*-
# run_analysis_all_in_one.py (釜底抽薪修正版 v5)

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
#  輔助函數 (Helper Function)
# ==============================================================================
def get_ball_center_from_data(frame_data):
    """ 從一幀的資料中安全地獲取球的中心點，會過濾掉背景區的球。 """
    if frame_data and frame_data.get('ball_detections'):
        valid_balls = [b for b in frame_data['ball_detections'] if not b.get('is_in_background_zone', False)]
        if not valid_balls: return None
        best_ball = max(valid_balls, key=lambda b: b.get('confidence', 0))
        box = best_ball['box_coords']
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    return None

# ==============================================================================
#  分析邏輯 (Event Analysis Logic)
# ==============================================================================
def analyze_serve_events(all_frames_data, config, log_prefix=""):
    # (此函數的邏輯現在只專注於找出擊球事件的時間點)
    hit_v_thresh = config.get("hit_v", 40.0)
    # ... 其他參數 ...
    max_plausible_speed = config.get("max_speed", 200.0); toss_initial_vy_thresh = config.get("toss_vy", 8.0); frames_to_validate_toss = config.get("frames_to_validate", 8); min_upward_confirms = config.get("min_upward_confirms", 3); vertical_ratio_thresh = config.get("vertical_ratio", 1.5); hit_horizontal_ratio_thresh = config.get("hit_h_ratio", 2.5); max_frames_to_apex = config.get("max_frames_to_apex", 75); max_frames_to_hit = config.get("max_frames_to_hit", 40); max_lost_frames_apex = config.get("max_lost_frames_apex", 50);

    serve_events = []
    state = "SEARCHING_TOSS"
    event_candidate = {}
    print(f"\n{log_prefix}[分析階段] 正在偵測擊球事件時間點...")
    for i in range(1, len(all_frames_data)):
        prev_ball_pos = get_ball_center_from_data(all_frames_data[i-1])
        curr_ball_pos = get_ball_center_from_data(all_frames_data[i])
        if prev_ball_pos is None or curr_ball_pos is None:
            if state in ["AWAITING_APEX", "AWAITING_HIT", "CONFIRMING_TOSS"]: event_candidate['lost_frames_count'] = event_candidate.get('lost_frames_count', 0) + 1
            if state == "AWAITING_APEX" and event_candidate.get('lost_frames_count', 0) > max_lost_frames_apex: state = "SEARCHING_TOSS"
            elif state == "AWAITING_HIT" and event_candidate.get('lost_frames_count', 0) > 15: state = "SEARCHING_TOSS"
            elif state == "CONFIRMING_TOSS" and event_candidate.get('lost_frames_count', 0) > 5: state = "SEARCHING_TOSS"
            continue
        speed = np.linalg.norm(curr_ball_pos - prev_ball_pos)
        if speed > max_plausible_speed: continue
        if 'lost_frames_count' in event_candidate: event_candidate['lost_frames_count'] = 0
        if state == "SEARCHING_TOSS":
            vy = prev_ball_pos[1] - curr_ball_pos[1]; vx = curr_ball_pos[0] - prev_ball_pos[0]
            if vy > toss_initial_vy_thresh and (abs(vy) / (abs(vx) + 1e-6)) > vertical_ratio_thresh: state = "CONFIRMING_TOSS"; event_candidate = {'confirm_start_frame': i, 'upward_frames_count': 1, 'lost_frames_count': 0, 'last_pos': curr_ball_pos}
        elif state == "CONFIRMING_TOSS":
            if (i - event_candidate['confirm_start_frame']) > frames_to_validate_toss: state = "SEARCHING_TOSS"; continue
            vy = event_candidate['last_pos'][1] - curr_ball_pos[1]
            if vy > 0: event_candidate['upward_frames_count'] += 1
            event_candidate['last_pos'] = curr_ball_pos
            if event_candidate['upward_frames_count'] >= min_upward_confirms: state = "AWAITING_APEX"; event_candidate['toss_start_frame'] = event_candidate['confirm_start_frame']; event_candidate['lost_frames_count'] = 0
        elif state == "AWAITING_APEX":
            if (i - event_candidate['toss_start_frame']) > max_frames_to_apex: state = "SEARCHING_TOSS"; continue
            vy = prev_ball_pos[1] - curr_ball_pos[1]
            if vy < -1: state = "AWAITING_HIT"; event_candidate['apex_frame'] = i; event_candidate['lost_frames_count'] = 0
        elif state == "AWAITING_HIT":
            if (i - event_candidate['apex_frame']) > max_frames_to_hit: state = "SEARCHING_TOSS"; continue
            if speed > hit_v_thresh:
                hit_vx = curr_ball_pos[0] - prev_ball_pos[0]; hit_vy = curr_ball_pos[1] - prev_ball_pos[1]; horizontal_ratio = abs(hit_vx) / (abs(hit_vy) + 1e-6)
                if horizontal_ratio < hit_horizontal_ratio_thresh:
                    print(f"{log_prefix}  > [第 {i} 幀] 偵測到有效擊球事件 (速度: {speed:.1f})")
                    serve_events.append({"frame_id": i, "hit_position": curr_ball_pos.tolist(), "hit_speed": speed})
                    state = "SEARCHING_TOSS"
    return serve_events

# ==============================================================================
#  核心處理工作單元 (The Main Worker Unit)
# ==============================================================================
def process_single_video(video_path, base_output_dir, overwrite, frame_args, analysis_args):
    # (此函數不變)
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    log_prefix = f"[{video_base_name}] "
    try:
        video_specific_output_dir = os.path.join(base_output_dir, video_base_name)
        tracking_output_path = os.path.join(video_specific_output_dir, 'tracking_output')
        analysis_output_path = os.path.join(video_specific_output_dir, 'analysis_output')
        final_video_output = os.path.join(analysis_output_path, f"{video_base_name}_analysis.mp4")
        json_input_path = os.path.join(tracking_output_path, video_base_name, f"{video_base_name}_all_frames_data_with_pose.json")
        if not overwrite and os.path.exists(final_video_output):
            return {"video": video_base_name, "status": "skipped", "log": "最終輸出檔案已存在。", "events": [], "tracking_data_path": None}
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
        analysis_config = {k: v for k, v in analysis_args.items()}
        old_stdout = sys.stdout; sys.stdout = analysis_log_buffer = io.StringIO()
        serve_events = analyze_serve_events(all_frames_data, analysis_config, log_prefix)
        sys.stdout = old_stdout; analysis_log_content = analysis_log_buffer.getvalue()
        # Visualisation part is now mainly handled in create_first_hit_summary
        # But we can still create a basic analysis video if needed. For now, we skip detailed drawing here.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise IOError(f"無法開啟影片檔案: {video_path}")
        fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(final_video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read();
            if not ret: break
            writer.write(frame); frame_idx += 1
        cap.release(); writer.release()
        print(f"{log_prefix}影片產生完成: {final_video_output}")
        with open(os.path.join(video_specific_output_dir, "processing.log"), 'w', encoding='utf-8') as f:
            f.write("--- STAGE 1: TRACKING STDOUT ---\n"); f.write(stdout_track); f.write("\n--- STAGE 1: TRACKING STDERR ---\n"); f.write(stderr_track); f.write("\n\n--- STAGE 2: ANALYSIS LOG ---\n"); f.write(analysis_log_content)
        return {"video": video_base_name, "status": "success", "log": final_video_output, "events": serve_events, "tracking_data_path": json_input_path}
    except Exception as e:
        error_message = f"處理影片 {video_base_name} 時發生嚴重錯誤: {e}"
        if isinstance(e, subprocess.CalledProcessError): error_message += f"\n--- 子程序標準輸出 ---\n{e.output}\n\n--- 子程序標準錯誤（Traceback） ---\n{e.stderr}"
        return {"video": video_base_name, "status": "failed", "log": error_message, "events": []}

# ==============================================================================
#  總結報告產生 (Summary Generation)
# ==============================================================================
def create_first_hit_summary(results, base_output_dir, input_folder, server_search_frames):
    print("\n" + "="*80); print("--- 正在產生首次擊球總結報告 (v5 邏輯)... ---")
    summary_dir = os.path.join(base_output_dir, "first_hit_summary");
    # 清理舊的總結檔案
    if os.path.exists(summary_dir):
        for f in os.listdir(summary_dir):
            os.remove(os.path.join(summary_dir, f))
    else:
        os.makedirs(summary_dir, exist_ok=True)

    summary_txt_path = os.path.join(summary_dir, "first_hit_frames.txt")
    video_path_map = {os.path.splitext(os.path.basename(p))[0]: p for p in find_video_files(input_folder)}
    count = 0

    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write("--- 每個影片片段首次偵測到的擊球資訊 (v5 邏輯) ---\n\n")
        for result in sorted(results, key=lambda r: r['video']):
            if result['status'] != 'success' or not result.get('events'):
                continue
            
            video_base_name = result['video']
            first_event = result['events'][0]
            hit_frame_num = first_event['frame_id']
            tracking_path = result.get('tracking_data_path')

            if not tracking_path or not os.path.exists(tracking_path):
                print(f"[警告] 找不到 {video_base_name} 的追蹤JSON檔案，跳過總結。")
                continue

            with open(tracking_path, 'r', encoding='utf-8') as json_f:
                all_frames_data = json.load(json_f)
            
            # --- ✨【v5 核心邏輯】在這裡重新獨立計算發球員 ---
            server_data = None
            found_at_frame_vis = -1
            print(f"[{video_base_name}] 擊球幀為 {hit_frame_num}，開始回溯查找發球員...")
            for offset in range(server_search_frames, 0, -1):
                lookback_idx = hit_frame_num - offset
                if 0 <= lookback_idx < len(all_frames_data):
                    past_frame = all_frames_data[lookback_idx]
                    ball_pos = get_ball_center_from_data(past_frame)
                    players = past_frame.get('player_detections', [])
                    if ball_pos is not None and players:
                        closest_player = min(players, key=lambda p: np.linalg.norm(np.array(p['center_point']) - ball_pos))
                        server_data = closest_player.copy() # 確保複製資料
                        found_at_frame_vis = lookback_idx
                        print(f"  > 成功在第 {found_at_frame_vis} 幀找到發球員。")
                        break # 找到後立刻停止
            
            # 將計算結果寫入文字檔
            player_id_str = f"位於 ({int(server_data['center_point'][0])}, {int(server_data['center_point'][1])}) (在第 {found_at_frame_vis} 幀找到)" if server_data else "未知"
            f.write(f"{video_base_name}: 擊球幀數={hit_frame_num}, 發球員={player_id_str}\n")
            
            # --- ✨【v5 核心邏輯】在這裡根據重新計算的結果繪製圖片 ---
            original_video_path = video_path_map.get(video_base_name)
            if not original_video_path: continue

            try:
                cap = cv2.VideoCapture(original_video_path)
                if not cap.isOpened(): continue
                start_frame, end_frame = max(0, hit_frame_num - 3), hit_frame_num + 3
                for frame_to_capture in range(start_frame, end_frame + 1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_capture)
                    ret, frame = cap.read()
                    if ret:
                        # 繪製所有偵測到的藍色框
                        if frame_to_capture < len(all_frames_data):
                            frame_data = all_frames_data[frame_to_capture]
                            if frame_data.get('player_detections'):
                                for player in frame_data['player_detections']:
                                    box = player['box_coords']; cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)
                        
                        # 只在擊球幀上繪製特別標記
                        if frame_to_capture == hit_frame_num:
                            hit_pos = tuple(map(int, first_event['hit_position']))
                            cv2.circle(frame, hit_pos, 40, (0, 255, 0), 4) # 綠色圓圈標示球
                            cv2.putText(frame, "SERVE", (hit_pos[0] - 70, hit_pos[1] - 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
                            # 使用我們剛剛獨立計算出來的 server_data 來畫紫色框
                            if server_data:
                                box = server_data['box_coords']
                                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                                cv2.rectangle(frame, p1, p2, (255, 0, 255), 3)
                                cv2.putText(frame, "SERVER", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
                        
                        tag = ""
                        if frame_to_capture == hit_frame_num: tag = "_HIT"
                        elif frame_to_capture < hit_frame_num: tag = f"_PRE_{hit_frame_num - frame_to_capture}"
                        else: tag = f"_POST_{frame_to_capture - hit_frame_num}"
                        img_name = f"{video_base_name}_frame_{frame_to_capture:06d}{tag}.jpg"
                        cv2.imwrite(os.path.join(summary_dir, img_name), frame)
                cap.release(); count += 1
            except Exception as e:
                print(f"[錯誤] 擷取 {video_base_name} 的關鍵幀時失敗: {e}")
    print(f"總結報告產生完畢！共擷取了 {count} 個影片的關鍵幀。"); print(f"詳細資訊請見: {summary_dir}")


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
    parser = argparse.ArgumentParser(description="[釜底抽薪修正版 v5] 自動化分析排球影片，並產生總結報告。")
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="volleyball_analysis_results")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    analysis_group = parser.add_argument_group('Analysis Parameters')
    analysis_group.add_argument("--max_speed", type=float, default=200.0)
    analysis_group.add_argument("--toss_vy", type=float, default=8.0)
    analysis_group.add_argument("--vertical_ratio", type=float, default=1.5)
    analysis_group.add_argument("--hit_v", type=float, default=40.0)
    analysis_group.add_argument("--hit_h_ratio", type=float, default=2.5)
    analysis_group.add_argument("--max_frames_to_apex", type=int, default=75)
    analysis_group.add_argument("--frames_to_validate", type=int, default=8)
    analysis_group.add_argument("--min_upward_confirms", type=int, default=3)
    analysis_group.add_argument("--server_priority_frame", type=int, default=3, help="從擊球偵測幀往前推幾幀作為優先搜尋目標，預設為3。")
    frame_group = parser.add_argument_group('Frame Saving Options')
    frame_group.add_argument("--save_all_frames", action="store_true")
    args = parser.parse_args()
    frame_args = {'save_all_frames': args.save_all_frames}
    analysis_args = {k: v for k, v in vars(args).items() if k in ['max_speed', 'toss_vy', 'vertical_ratio', 'hit_v', 'hit_h_ratio', 'max_frames_to_apex', 'frames_to_validate', 'min_upward_confirms', 'server_priority_frame']}
    
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
        futures = {executor.submit(process_single_video, vp, base_output_dir, args.overwrite, frame_args, analysis_args): vp for vp in video_files}
        with tqdm(total=len(futures), desc="整體進度", unit="video") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
                try: results.append(future.result())
                except Exception as exc: results.append({"video": os.path.basename(futures[future]), "status": "failed", "log": f"執行時發生嚴重錯誤: {exc}", "events": []})

    create_first_hit_summary(results, base_output_dir, args.input_folder, args.server_priority_frame)
    
    print("\n" + "="*80); print("--- 所有任務已完成，正在生成摘要報告... ---")
    summary_path = os.path.join(base_output_dir, "summary_report.txt")
    success_videos = [r for r in results if r['status'] == 'success']; failed_videos = [r for r in results if r['status'] == 'failed']; skipped_videos = [r for r in results if r['status'] == 'skipped']
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"批次處理執行摘要\n執行時間: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n總耗時: {datetime.now() - script_start_time}\n" + "="*60 + "\n\n")
        f.write(f"--- ✅ 處理成功 ({len(success_videos)}) ---\n"); [f.write(f"- {res['video']} -> {res['log']}\n") for res in success_videos]
        f.write(f"\n--- ⏭️ 自動跳過 ({len(skipped_videos)}) ---\n"); [f.write(f"- {res['video']}\n") for res in skipped_videos]
        f.write(f"\n--- ❌ 處理失敗 ({len(failed_videos)}) ---\n"); [f.write(f"--- Video: {res['video']} ---\n{res['log']}\n\n") for res in failed_videos]
    print(f"摘要報告已生成於: {summary_path}"); print(f"執行結果: {len(success_videos)} 成功, {len(failed_videos)} 失敗, {len(skipped_videos)} 跳過。"); print("="*80)

if __name__ == '__main__':
    main()