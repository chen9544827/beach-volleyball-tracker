# run_first_hit_analysis.py (最終、最可靠的動作識別版 v12)

import os
import subprocess
import argparse
import sys
import json
import numpy as np
import cv2
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("錯誤：找不到 'tqdm' 模組。請執行 'pip install tqdm' 來安裝。")

# ==============================================================================
#  輔助函數 (與之前版本相同)
# ==============================================================================
def get_ball_center(frame_data):
    if not (frame_data and frame_data.get('ball_detections')): return None
    valid_balls = [b for b in frame_data.get('ball_detections', []) if not b.get('is_in_background_zone', False)]
    if not valid_balls: return None
    best_ball = max(valid_balls, key=lambda b: b.get('confidence', 0))
    box = best_ball['box_coords']
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

def find_video_files(directory):
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv')
    return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.lower().endswith(supported_formats)]

# ==============================================================================
#  核心分析邏輯 (採用最可靠的「動作識別」方案)
# ==============================================================================
def find_first_serve_flow(all_frames_data):
    """
    分析所有幀數據，找出第一個完整的「拋球 -> 擊球」動作流程。
    """
    state, event_candidate = "SEARCHING_TOSS", {}
    params = {"hit_v_thresh": 40.0, "toss_vy_thresh": 8.0, "vertical_ratio": 1.5, "max_frames_to_apex": 75, "max_frames_to_hit": 40}

    for i in range(1, len(all_frames_data)):
        prev_pos, curr_pos = get_ball_center(all_frames_data[i-1]), get_ball_center(all_frames_data[i])
        if prev_pos is None or curr_pos is None: continue

        if state == "SEARCHING_TOSS":
            vy = prev_pos[1] - curr_pos[1]; vx = curr_pos[0] - prev_pos[0]
            if vy > params["toss_vy_thresh"] and (abs(vy) / (abs(vx) + 1e-6)) > params["vertical_ratio"]:
                event_candidate = {'toss_frame': i}; state = "AWAITING_APEX"
        
        elif state == "AWAITING_APEX":
            if (i - event_candidate['toss_frame']) > params["max_frames_to_apex"]: state = "SEARCHING_TOSS"
            vy = prev_pos[1] - curr_pos[1]
            if vy < -1: event_candidate['apex_frame'] = i; state = "AWAITING_HIT"

        elif state == "AWAITING_HIT":
            if (i - event_candidate['apex_frame']) > params["max_frames_to_hit"]: state = "SEARCHING_TOSS"
            speed = np.linalg.norm(curr_pos - prev_pos)
            if speed > params["hit_v_thresh"]:
                return {"toss_frame_id": event_candidate['toss_frame'], "hit_frame_id": i, "hit_position": curr_pos.tolist()}
    return None

def get_server_at_toss(toss_frame_id, all_frames_data):
    """ 在拋球的瞬間，鎖定最近的球員為發球員。 """
    frame_data = all_frames_data[toss_frame_id]
    ball_pos = get_ball_center(frame_data)
    players = frame_data.get('player_detections', [])
    if ball_pos is not None and players:
        closest_player = min(players, key=lambda p: np.linalg.norm(np.array(p['center_point']) - ball_pos))
        return closest_player.copy()
    return None

# ==============================================================================
#  主程式 (Main Execution)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="【v12 最終版】分析首次擊球，並透過識別拋球動作來可靠地標記發球員。")
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="first_hit_analysis_results")
    parser.add_argument("--overwrite_tracking", action="store_true", help="強制重新執行物件追蹤。")
    args = parser.parse_args()

    base_output_dir = os.path.abspath(args.output_folder)
    summary_dir = os.path.join(base_output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_txt_path = os.path.join(summary_dir, "first_hit_frames.txt")

    with open(summary_txt_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write(f"--- 首次擊球分析報告 (v12) ---\n執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        video_files = find_video_files(args.input_folder)
        if not video_files: print(f"[錯誤] 在 '{args.input_folder}' 中找不到影片。"); return

        for video_path in tqdm(video_files, desc="處理所有影片"):
            video_base_name = os.path.splitext(os.path.basename(video_path))[0]
            log_prefix = f"[{video_base_name}] "
            
            tracking_output_dir = os.path.join(base_output_dir, "tracking_data", video_base_name)
            # ✨【v12 修正】這裡的路徑要和追蹤腳本的輸出路徑完全匹配
            json_path = os.path.join(tracking_output_dir, f"{video_base_name}_all_frames_data_with_pose.json")
            
            if args.overwrite_tracking or not os.path.exists(json_path):
                print(f"\n{log_prefix}正在執行物件追蹤...")
                # ✨【v12 修正】傳遞給追蹤腳本的 output_dir 是 `tracking_output_dir`
                track_command = [sys.executable, os.path.join("video_processing", "track_ball_and_player.py"), "--input", video_path, "--output_dir", tracking_output_dir]
                result = subprocess.run(track_command, capture_output=True, text=True, encoding='utf-8')
                if result.returncode != 0:
                    print(f"❌ {log_prefix}物件追蹤失敗！詳細錯誤訊息如下：\n{result.stderr}")
                    continue
            else:
                print(f"\n{log_prefix}找到現有的追蹤資料，跳過追蹤。")

            if not os.path.exists(json_path):
                print(f"🤷 {log_prefix}錯誤：追蹤步驟已執行，但依然找不到必要的JSON檔案：{json_path}")
                continue
            
            print(f"✅ {log_prefix}成功載入JSON檔案，開始分析動作...")
            with open(json_path, 'r', encoding='utf-8') as f: all_frames_data = json.load(f)

            key_moments = find_first_serve_flow(all_frames_data)
            if not key_moments:
                summary_file.write(f"{video_base_name}: 未偵測到完整的發球動作。\n"); continue
            
            toss_frame_id, hit_frame_id, hit_position = key_moments["toss_frame_id"], key_moments["hit_frame_id"], key_moments["hit_position"]
            server_data = get_server_at_toss(toss_frame_id, all_frames_data)
            
            if server_data:
                summary_file.write(f"{video_base_name}: 在第 {toss_frame_id} 幀識別拋球，在第 {hit_frame_id} 幀擊球。\n")
            else:
                summary_file.write(f"{video_base_name}: 雖有發球動作，但在拋球時未能鎖定球員。\n")
            summary_file.flush()

            print(f"{log_prefix}正在產生標記圖片...")
            try:
                cap = cv2.VideoCapture(video_path)
                for offset in range(-3, 4):
                    frame_idx = hit_frame_id + offset
                    if frame_idx < 0: continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx); ret, frame = cap.read()
                    if not ret: continue
                    
                    if frame_idx == hit_frame_id and server_data:
                        box = server_data['box_coords']
                        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                        cv2.rectangle(frame, p1, p2, (255, 0, 255), 3)
                        cv2.putText(frame, f"SERVER (from F{toss_frame_id})", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
                        cv2.circle(frame, tuple(map(int, hit_position)), 40, (0, 255, 0), 4)
                    
                    tag = f"_PRE_{abs(offset)}" if offset < 0 else f"_POST_{offset}" if offset > 0 else "_HIT"
                    cv2.imwrite(os.path.join(summary_dir, f"{video_base_name}_frame_{frame_idx:06d}{tag}.jpg"), frame)
                cap.release()
            except Exception as e:
                print(f"{log_prefix}產生圖片時發生錯誤: {e}")

    print("\n" + "="*80 + "\n--- 【v12 最終版】處理完畢 ---\n" + "="*80)
    print(f"所有分析結果已儲存於: {summary_dir}")

if __name__ == '__main__':
    main()