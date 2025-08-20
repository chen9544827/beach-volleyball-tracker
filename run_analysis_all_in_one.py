# run_analysis_all_in_one.py (v20.3 - v11偵測核心 + 內建優化參數 + 偵錯日誌)

import os
import subprocess
import argparse
import sys
import json
import numpy as np
import cv2
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import locale
import csv
import shutil

# --- 套件與輔助函數 ---
try:
    from tqdm import tqdm
except ImportError:
    print("模組 'tqdm' 未找到，請執行 'pip install tqdm'"); sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("偵錯模式需要 'matplotlib' 模組..."); plt = None

from analysis.jump_serve_analyzer import analyze_jump_serve_by_pose, get_player_center

def get_ball_center_from_data(frame_data):
    if frame_data and frame_data.get('ball_detections'):
        valid_balls = [b for b in frame_data['ball_detections'] if not b.get('is_in_background_zone', False)]
        if not valid_balls: return None
        best_ball = max(valid_balls, key=lambda b: b.get('confidence', 0))
        box = best_ball['box_coords']
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

def get_distance_to_player_box(point, player_box):
    px, py = point; x1, y1, x2, y2 = player_box
    closest_x = max(x1, min(px, x2)); closest_y = max(y1, min(py, y2))
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

def get_serving_zone(server_center, court_polygon):
    if server_center is None or court_polygon is None or len(court_polygon) != 4: return 'Unknown'
    p1 = np.array(court_polygon[1]); p2 = np.array(court_polygon[2])
    zone_ab_divider = p1 + (p2 - p1) / 3; zone_bc_divider = p1 + 2 * (p2 - p1) / 3
    if server_center[0] < zone_ab_divider[0]: return 'A'
    elif server_center[0] < zone_bc_divider[0]: return 'B'
    else: return 'C'

def analyze_serve_events(all_frames_data, config, log_prefix="", debug_log_path=None):
    # (v11 偵測邏輯)
    params = {k: config.get(k, v) for k, v in {
        "hit_v": 40.0, "max_speed": 200.0, "toss_vy": 8.0, "frames_to_validate": 8,
        "min_upward_confirms": 3, "vertical_ratio": 0.8, "hit_h_ratio": 2.5,
        "max_frames_to_apex": 75, 
        # ✨ 核心修改 1: 將您驗證有效的參數直接設為新的預設值
        "max_frames_to_hit": 90, 
        "max_lost_frames_apex": 120
    }.items()}
    serve_events, state, event_candidate = [], "SEARCHING_TOSS", {}
    
    debug_log = []
    
    print(f"\n{log_prefix}[分析階段] 正在使用 v11 邏輯 (優化參數版) 偵測擊球事件...")
    for i in range(1, len(all_frames_data)):
        log_entry = {"frame_id": i, "state": state, "speed": 0, "vy": 0, "vx": 0, "h_ratio": 0, "lost_frames": 0}
        
        prev_ball_pos = get_ball_center_from_data(all_frames_data[i-1])
        curr_ball_pos = get_ball_center_from_data(all_frames_data[i])
        
        if prev_ball_pos is None or curr_ball_pos is None:
            if state in ["AWAITING_APEX", "AWAITING_HIT", "CONFIRMING_TOSS"]:
                event_candidate['lost_frames_count'] = event_candidate.get('lost_frames_count', 0) + 1
                log_entry['lost_frames'] = event_candidate['lost_frames_count']
                if state == "AWAITING_APEX" and event_candidate.get('lost_frames_count', 0) > params['max_lost_frames_apex']: state = "SEARCHING_TOSS"
                elif state == "AWAITING_HIT" and event_candidate.get('lost_frames_count', 0) > 15: state = "SEARCHING_TOSS"
                elif state == "CONFIRMING_TOSS" and event_candidate.get('lost_frames_count', 0) > 5: state = "SEARCHING_TOSS"
            log_entry['state'] = state
            debug_log.append(log_entry)
            continue

        speed = np.linalg.norm(curr_ball_pos - prev_ball_pos)
        vy = prev_ball_pos[1] - curr_ball_pos[1]
        vx = curr_ball_pos[0] - prev_ball_pos[0]
        log_entry.update({"speed": speed, "vy": vy, "vx": vx})

        if speed > params['max_speed']: 
            debug_log.append(log_entry)
            continue
        if 'lost_frames_count' in event_candidate: event_candidate['lost_frames_count'] = 0

        if state == "SEARCHING_TOSS":
            if vy > params['toss_vy']:
                ratio = (abs(vy) / (abs(vx) + 1e-6))
                print(f"DEBUG --- Frame {i}: vy={vy:.2f} (>{params['toss_vy']}). Calculated Ratio = {ratio:.2f}")
                if ratio > params['vertical_ratio']:
                    state = "CONFIRMING_TOSS"; event_candidate = {'confirm_start_frame': i, 'upward_frames_count': 1, 'lost_frames_count': 0, 'last_pos': curr_ball_pos}
        
        elif state == "CONFIRMING_TOSS":
            if (i - event_candidate['confirm_start_frame']) > params['frames_to_validate']: state = "SEARCHING_TOSS"
            if event_candidate['last_pos'][1] - curr_ball_pos[1] > 0: event_candidate['upward_frames_count'] += 1
            event_candidate['last_pos'] = curr_ball_pos
            if event_candidate['upward_frames_count'] >= params['min_upward_confirms']:
                state = "AWAITING_APEX"; event_candidate['toss_start_frame'] = event_candidate['confirm_start_frame']
        
        elif state == "AWAITING_APEX":
            if (i - event_candidate.get('toss_start_frame', i)) > params['max_frames_to_apex']: state = "SEARCHING_TOSS"
            if vy < -1: state = "AWAITING_HIT"; event_candidate['apex_frame'] = i
        
        elif state == "AWAITING_HIT":
            if (i - event_candidate.get('apex_frame', i)) > params['max_frames_to_hit']: state = "SEARCHING_TOSS"
            if speed > params['hit_v']:
                h_ratio = abs(vx) / (abs(vy) + 1e-6)
                log_entry["h_ratio"] = h_ratio
                if h_ratio < params['hit_h_ratio']:
                    serve_events.append({"hit_frame_id": i}); state = "SEARCHING_TOSS"
        
        log_entry["state"] = state
        debug_log.append(log_entry)

    if debug_log_path and debug_log:
        try:
            with open(debug_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=debug_log[0].keys())
                writer.writeheader(); writer.writerows(debug_log)
            print(f"{log_prefix}偵錯日誌已儲存至: {debug_log_path}")
        except Exception as e:
            print(f"{log_prefix}錯誤：寫入偵錯日誌失敗: {e}")

    return serve_events

def process_single_video(video_path, base_output_dir, overwrite, config):
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    log_prefix = f"[{video_base_name}] "
    try:
        video_specific_output_dir = os.path.join(base_output_dir, video_base_name)
        tracking_output_path = os.path.join(video_specific_output_dir, 'tracking_output')
        json_input_path = os.path.join(tracking_output_path, f"{video_base_name}_all_frames_data_with_pose.json")
        debug_log_path = os.path.join(video_specific_output_dir, f"{video_base_name}_serve_debug_log.csv")
        
        if not overwrite and os.path.exists(os.path.join(base_output_dir, "final_summary_refined")): 
            return {"video": video_base_name, "status": "skipped"}
        
        if overwrite or not os.path.exists(json_input_path):
            print(f"{log_prefix}執行 Stage 1: 物件追蹤...")
            track_command = [sys.executable, os.path.join("video_processing", "track_ball_and_player.py"), "--input", video_path, "--output_dir", tracking_output_path]
            result_track = subprocess.run(track_command, capture_output=True, text=True, encoding=locale.getpreferredencoding(), errors='ignore')
            if result_track.returncode != 0: 
                raise subprocess.CalledProcessError(result_track.returncode, track_command, output=result_track.stdout, stderr=result_track.stderr)
        
        with open(json_input_path, 'r', encoding='utf-8') as f: 
            all_frames_data = json.load(f)
        
        serve_events = analyze_serve_events(all_frames_data, config=config, log_prefix=log_prefix, debug_log_path=debug_log_path)
        
        return {"video": video_base_name, "status": "success", "events": serve_events, "original_video_path": video_path, "tracking_data_path": json_input_path}
    except Exception as e:
        error_message = f"處理影片 {video_base_name} 時發生錯誤: {e}"
        return {"video": video_base_name, "status": "failed", "log": error_message}

def create_final_summary(results, base_output_dir, config, reports_archive_folder):
    print("\n" + "="*80 + "\n--- 正在產生最終分析報告... ---")
    summary_dir = os.path.join(base_output_dir, "final_summary_refined"); os.makedirs(summary_dir, exist_ok=True)
    review_frames_dir = os.path.join(base_output_dir, "key_frames_for_review"); os.makedirs(review_frames_dir, exist_ok=True)
    csv_report_path = os.path.join(summary_dir, "analysis_summary.csv")
    csv_data = []

    for result in sorted(results, key=lambda r: r['video']):
        if result['status'] != 'success' or not result.get('events'): continue
        
        video_base_name = result['video']
        initial_hit_event = result['events'][0]
        initial_hit_frame_id = initial_hit_event['hit_frame_id']
        tracking_path = result.get('tracking_data_path')
        
        if not tracking_path or not os.path.exists(tracking_path): continue
        with open(tracking_path, 'r', encoding='utf-8') as json_f: all_frames_data = json.load(json_f)
        
        player_search_frame_id = max(0, initial_hit_frame_id - config.get('search_offset'))
        server_data = None
        if player_search_frame_id < len(all_frames_data):
            search_frame_data = all_frames_data[player_search_frame_id]
            ball_pos = get_ball_center_from_data(search_frame_data)
            players = search_frame_data.get('player_detections', [])
            if ball_pos is not None and players:
                server_data = min(players, key=lambda p: get_distance_to_player_box(ball_pos, p['box_coords']))
        
        if not server_data:
            print(f"[{video_base_name}] 雖偵測到發球，但在搜尋時未能鎖定球員。")
            continue
            
        refined_hit_frame_id = initial_hit_frame_id
        search_window = range(max(0, initial_hit_frame_id - 20), min(len(all_frames_data), initial_hit_frame_id + 10))
        for frame_idx in search_window:
            ball_pos = get_ball_center_from_data(all_frames_data[frame_idx])
            if ball_pos is None: continue
            dist = get_distance_to_player_box(ball_pos, server_data['box_coords'])
            if dist > config.get('ball_leave_threshold', 30):
                prev_ball_pos = get_ball_center_from_data(all_frames_data[frame_idx - 1])
                if prev_ball_pos is not None and np.linalg.norm(ball_pos - prev_ball_pos) > 10:
                    refined_hit_frame_id = frame_idx; break
        print(f"[{video_base_name}] 擊球時間校正: {initial_hit_frame_id} -> {refined_hit_frame_id}")
        
        serving_zone, serve_type, _ = "Unknown", "Unknown", {}
        server_center = get_player_center(server_data)
        serving_zone = get_serving_zone(server_center, config.get('court_polygon'))
        serve_type, _ = analyze_jump_serve_by_pose(all_frames_data, server_data, refined_hit_frame_id, config)
        print(f"[{video_base_name}] 分析完成。區域: {serving_zone}, 類型: {serve_type}")
        csv_data.append([video_base_name, serving_zone, serve_type])
        
        original_video_path = result.get('original_video_path')
        if original_video_path:
            segment_key_frames_dir = os.path.join(base_output_dir, video_base_name, "key_frames"); os.makedirs(segment_key_frames_dir, exist_ok=True)
            try:
                cap = cv2.VideoCapture(original_video_path)
                if cap.isOpened():
                    for frame_offset in range(-3, 4):
                        frame_to_capture = refined_hit_frame_id + frame_offset
                        if 0 <= frame_to_capture < cap.get(cv2.CAP_PROP_FRAME_COUNT):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_capture); ret, frame = cap.read()
                            if ret:
                                tag = "_HIT" if frame_offset == 0 else f"_PRE_{abs(frame_offset)}" if frame_offset < 0 else f"_POST_{frame_offset}"
                                img_name = f"{video_base_name}_frame_{frame_to_capture:06d}{tag}.jpg"
                                cv2.imwrite(os.path.join(segment_key_frames_dir, img_name), frame)
                                cv2.imwrite(os.path.join(review_frames_dir, img_name), frame)
                    cap.release()
            except Exception as e: print(f"[錯誤] 儲存 {video_base_name} 的關鍵幀時失敗: {e}")

    if csv_data:
        try:
            with open(csv_report_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile); writer.writerow(['Segment_Name', 'Serve_Zone', 'Serve_Type']); writer.writerows(csv_data)
            print(f"\nCSV 格式的分析報告已成功儲存至: {os.path.abspath(csv_report_path)}")
            if reports_archive_folder:
                try:
                    os.makedirs(reports_archive_folder, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"); archive_filename = f"summary_{timestamp}.csv"
                    archive_filepath = os.path.join(reports_archive_folder, archive_filename)
                    shutil.copy2(csv_report_path, archive_filepath)
                    print(f"報告已成功存檔至: {os.path.abspath(archive_filepath)}")
                except Exception as e: print(f"\n[錯誤] 存檔 CSV 報告時失敗: {e}")
        except Exception as e: print(f"\n[錯誤] 儲存本次執行的 CSV 報告時失敗: {e}")
    else:
        print("\n[資訊] 本次執行未偵測到任何有效的發球事件，不產生 CSV 報告。")
    print(f"\n總結報告與影像產生完畢！")

def find_video_files(directory):
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv')
    return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.lower().endswith(supported_formats)]

def main():
    parser = argparse.ArgumentParser(description="[v20.3] v11偵測核心 + 內建優化參數")
    parser.add_argument("--input_folder", type=str, required=True, help="包含待分析影片的資料夾路徑。")
    parser.add_argument("--output_folder", type=str, default="volleyball_analysis_results", help="儲存所有分析結果的根目錄。")
    parser.add_argument("--reports_archive_folder", type=str, default="csv_reports_archive", help="儲存所有批次執行結果的CSV存檔資料夾。")
    parser.add_argument("--workers", type=int, default=2, help="同時執行的最大程序數量。")
    parser.add_argument("--overwrite", action="store_true", help="強制重新執行所有影片的分析，覆蓋現有結果。")
    parser.add_argument("--court_config", type=str, help="定義了球場邊界的 court_config.json 檔案路徑 (可選)。")
    
    # ✨ 核心修改 2: 將您驗證有效的參數同步到命令列的預設值
    analysis_group = parser.add_argument_group('Analysis Parameters (v11 穩定版)')
    analysis_group.add_argument("--hit_v", type=float, default=40.0, help="偵測擊球的最小瞬時速度。")
    analysis_group.add_argument("--max_speed", type=float, default=200.0, help="過濾不可能的高速移動。")
    analysis_group.add_argument("--toss_vy", type=float, default=8.0, help="觸發『疑似拋球』的最小初始垂直向上速度。")
    analysis_group.add_argument("--frames_to_validate", type=int, default=8, help="確認拋球動作的幀數窗口。")
    analysis_group.add_argument("--min_upward_confirms", type=int, default=3, help="在確認窗口中，球至少要持續向上幾幀。")
    analysis_group.add_argument("--vertical_ratio", type=float, default=1.5, help="拋球時，垂直速度必須是水平速度的最小倍數。")
    analysis_group.add_argument("--hit_h_ratio", type=float, default=2.5, help="擊球時，水平分量與垂直分量的最大比例(避免將向上的高球誤判為擊球)。")
    analysis_group.add_argument("--max_frames_to_apex", type=int, default=75, help="從拋球到球達頂點的最長幀數。")
    analysis_group.add_argument("--max_frames_to_hit", type=int, default=60, help="從球達頂點到擊球的最長幀數 (已優化)。")
    analysis_group.add_argument("--max_lost_frames_apex", type=int, default=60, help="在等待頂點時，允許球消失的最大幀數 (已優化)。")

    other_group = parser.add_argument_group('Other Analysis Parameters')
    other_group.add_argument("--search_offset", type=int, default=3, help="從粗定位擊球幀往前『回溯』幾幀來尋找發球員。")
    other_group.add_argument("--ball_leave_threshold", type=int, default=30, help="定義球員近身區域的半徑（像素），用於精準校時。")
    other_group.add_argument("--jump_height_threshold", type=int, default=20, help="判定為跳躍的最小臀部垂直位移像素值。")
    
    args = parser.parse_args()
    config = vars(args)

    if args.court_config and os.path.exists(args.court_config):
        try:
            with open(args.court_config, 'r') as f: config['court_polygon'] = json.load(f).get("court_boundary_polygon")
        except Exception as e: print(f"警告：讀取 court_config 檔案 '{args.court_config}' 失敗: {e}")

    script_start_time = datetime.now()
    video_files = find_video_files(args.input_folder)
    if not video_files: print(f"[錯誤] 在 '{args.input_folder}' 中找不到影片。"); return
    
    base_output_dir = os.path.abspath(args.output_folder)
    os.makedirs(base_output_dir, exist_ok=True)
    
    all_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_video, vp, base_output_dir, args.overwrite, config): vp for vp in video_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="整體進度"):
            all_results.append(future.result())

    create_final_summary(all_results, base_output_dir, config, args.reports_archive_folder)
    
    print("\n" + "="*80); print("--- 所有任務已完成，正在生成最終摘要報告... ---")
    final_summary_path = os.path.join(base_output_dir, "master_summary_report.txt")
    with open(final_summary_path, 'w', encoding='utf-8') as f:
        f.write(f"批次處理執行摘要\n執行時間: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n總耗時: {datetime.now() - script_start_time}\n")
    print(f"總摘要報告已生成於: {final_summary_path}"); print("="*80)

if __name__ == '__main__':
    main()