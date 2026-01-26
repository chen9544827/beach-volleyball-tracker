# test_improvements.py
# -*- coding: utf-8 -*-
"""
改進功能測試腳本

用法：
    # 測試球追蹤器（含遮擋預測）
    python test_improvements.py --test tracker
    
    # 測試發球偵測器
    python test_improvements.py --test serve
    
    # 在實際影片上測試（會自動儲存關鍵幀圖片）
    python test_improvements.py --test video --input your_video.mp4
    
    # 比較新舊版本
    python test_improvements.py --test compare --input your_video.mp4
"""

import os
import sys

# 修復 OpenMP 衝突問題
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import time
import numpy as np

# 添加專案路徑
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def test_ball_tracker():
    """測試 BallTracker 模組"""
    print("\n" + "="*60)
    print("測試 BallTracker - 卡爾曼濾波 + 遮擋預測")
    print("="*60)
    
    from core.ball_tracker import BallTracker
    
    tracker = BallTracker(max_occlusion_frames=10)
    
    # 模擬軌跡：拋物線 + 遮擋
    print("\n模擬場景：發球拋物線，中間有 3 幀遮擋")
    
    test_sequence = [
        {"center_point": [100, 300], "confidence": 0.9},   # 0: 開始
        {"center_point": [120, 275], "confidence": 0.9},   # 1: 上升
        {"center_point": [140, 255], "confidence": 0.9},   # 2: 上升
        {"center_point": [160, 240], "confidence": 0.9},   # 3: 接近頂點
        {"center_point": [180, 235], "confidence": 0.9},   # 4: 頂點
        None,                                               # 5: 遮擋開始
        None,                                               # 6: 遮擋中
        None,                                               # 7: 遮擋中
        {"center_point": [260, 280], "confidence": 0.8},   # 8: 恢復偵測
        {"center_point": [290, 310], "confidence": 0.9},   # 9: 繼續下降
        {"center_point": [320, 345], "confidence": 0.9},   # 10: 擊球後
    ]
    
    print("\n逐幀追蹤結果：")
    print("-" * 70)
    print(f"{'幀號':^6} {'輸入':^20} {'輸出位置':^20} {'狀態':^12} {'預測':^6} {'信心':^6}")
    print("-" * 70)
    
    for i, detection in enumerate(test_sequence):
        result = tracker.update(detection, i)
        
        # 格式化輸入
        if detection:
            input_str = f"({detection['center_point'][0]}, {detection['center_point'][1]})"
        else:
            input_str = "None (遮擋)"
        
        # 格式化輸出
        if result['position']:
            output_str = f"({result['position'][0]:.1f}, {result['position'][1]:.1f})"
        else:
            output_str = "None"
        
        pred_str = "是" if result['is_predicted'] else "否"
        
        print(f"{i:^6} {input_str:^20} {output_str:^20} {result['state']:^12} {pred_str:^6} {result['confidence']:.2f}")
    
    print("-" * 70)
    
    # 計算預測準確度
    trajectory = tracker.get_trajectory()
    predicted_frames = [t for t in trajectory if t[3]]  # is_predicted = True
    
    print(f"\n統計：")
    print(f"  總追蹤幀數: {len(trajectory)}")
    print(f"  預測幀數: {len(predicted_frames)}")
    print(f"  遮擋恢復: {'成功' if len(predicted_frames) > 0 else '失敗'}")
    
    # 驗證預測位置是否合理
    if len(predicted_frames) > 0:
        print(f"\n預測位置驗證：")
        for t in predicted_frames:
            print(f"  幀 {t[0]}: 預測位置 ({t[1]:.1f}, {t[2]:.1f})")
    
    print("\n✅ BallTracker 測試完成！")


def test_serve_detector():
    """測試 ServeDetector 模組"""
    print("\n" + "="*60)
    print("測試 ServeDetector - 發球偵測狀態機")
    print("="*60)
    
    from core.serve_detector import ServeDetector
    
    detector = ServeDetector({
        'hit_v': 35.0,
        'toss_vy': 6.0,
        'hit_h_ratio': 3.0
    })
    
    # 模擬完整的發球軌跡
    print("\n模擬場景：完整發球動作（拋球 → 頂點 → 擊球）")
    
    trajectory = [
        np.array([500, 400]),   # 0: 起始
        np.array([502, 380]),   # 1: 拋球開始（向上）
        np.array([504, 358]),   # 2: 繼續上升
        np.array([506, 340]),   # 3: 繼續上升
        np.array([508, 325]),   # 4: 繼續上升
        np.array([510, 315]),   # 5: 接近頂點
        np.array([512, 310]),   # 6: 頂點附近
        np.array([514, 308]),   # 7: 頂點
        np.array([516, 312]),   # 8: 開始下降
        np.array([518, 320]),   # 9: 下降中
        np.array([522, 332]),   # 10: 下降中
        np.array([528, 348]),   # 11: 下降中
        np.array([580, 380]),   # 12: 擊球！（大位移）
        np.array([640, 400]),   # 13: 擊球後
        np.array([700, 415]),   # 14: 飛行中
    ]
    
    print("\n逐幀處理：")
    print("-" * 80)
    print(f"{'幀號':^6} {'位置':^20} {'速度':^10} {'狀態':^20} {'事件':^10}")
    print("-" * 80)
    
    for i in range(1, len(trajectory)):
        prev_pos = trajectory[i-1]
        curr_pos = trajectory[i]
        
        velocity = curr_pos - prev_pos
        speed = np.linalg.norm(velocity)
        
        event = detector.process_frame(prev_pos, curr_pos, i, use_dynamic_threshold=False)
        
        pos_str = f"({curr_pos[0]:.0f}, {curr_pos[1]:.0f})"
        event_str = "*** 發球! ***" if event else ""
        
        print(f"{i:^6} {pos_str:^20} {speed:^10.1f} {detector.state.value:^20} {event_str:^10}")
        
        if event:
            print(f"\n  🎯 偵測到發球事件！")
            print(f"     擊球幀: {event['hit_frame_id']}")
            print(f"     擊球速度: {event['hit_speed']:.1f}")
            print(f"     拋球幀: {event.get('toss_start_frame')}")
            print(f"     頂點幀: {event.get('apex_frame')}")
            print()
    
    print("-" * 80)
    
    events = detector.get_all_events()
    print(f"\n統計：")
    print(f"  偵測到的發球事件: {len(events)}")
    
    print("\n✅ ServeDetector 測試完成！")


def save_key_frames(video_path: str, events: list, output_dir: str):
    """
    儲存發球事件前後各 3 幀的圖片
    
    Args:
        video_path: 影片路徑
        events: 發球事件列表
        output_dir: 輸出目錄
    
    目錄結構：
        output_dir/
        └── key_frames/
            └── {video_name}/              # 🆕 以影片名稱區分
                └── event_1_hit_frame_XXX/
                    ├── xxx_TOSS.jpg
                    ├── xxx_PRE_3.jpg
                    ├── xxx_PRE_2.jpg
                    ├── xxx_PRE_1.jpg
                    ├── xxx_HIT.jpg
                    ├── xxx_POST_1.jpg
                    ├── xxx_POST_2.jpg
                    └── xxx_POST_3.jpg
    """
    import cv2
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if not events:
        print(f"[{video_name}] 沒有偵測到發球事件，無法儲存關鍵幀。")
        return
    
    # 建立關鍵幀目錄：key_frames/{video_name}/
    video_key_frames_dir = os.path.join(output_dir, "key_frames", video_name)
    os.makedirs(video_key_frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{video_name}] 無法開啟影片")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for event_idx, event in enumerate(events):
        hit_frame = event['hit_frame_id']
        toss_frame = event.get('toss_start_frame', hit_frame - 30)
        
        # 建立此事件的子目錄
        event_dir = os.path.join(video_key_frames_dir, f"event_{event_idx + 1}_hit_frame_{hit_frame}")
        os.makedirs(event_dir, exist_ok=True)
        
        # 儲存擊球前後各 3 幀（共 7 幀）
        frames_to_save = []
        for offset in range(-3, 4):  # -3, -2, -1, 0, 1, 2, 3
            frame_id = hit_frame + offset
            if 0 <= frame_id < total_frames:
                if offset < 0:
                    tag = f"PRE_{abs(offset)}"
                elif offset == 0:
                    tag = "HIT"
                else:
                    tag = f"POST_{offset}"
                frames_to_save.append((frame_id, tag))
        
        # 也儲存拋球幀
        if toss_frame and 0 <= toss_frame < total_frames:
            frames_to_save.append((toss_frame, "TOSS"))
        
        # 儲存圖片
        for frame_id, tag in frames_to_save:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                # 在圖片上標註資訊
                label = f"Frame {frame_id} ({tag})"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # 如果是擊球幀，加上紅色標記
                if tag == "HIT":
                    cv2.putText(frame, f"Speed: {event['hit_speed']:.1f}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                # 儲存圖片
                img_name = f"{video_name}_frame_{frame_id:06d}_{tag}.jpg"
                img_path = os.path.join(event_dir, img_name)
                cv2.imwrite(img_path, frame)
        
        print(f"  [{video_name}] 事件 {event_idx + 1}: 擊球幀={hit_frame}, 已儲存 {len(frames_to_save)} 張圖片")
    
    cap.release()


def test_on_video(video_path: str, court_config_path: str = None):
    """在實際影片上測試"""
    print("\n" + "="*60)
    print(f"在實際影片上測試")
    print(f"影片: {video_path}")
    print("="*60)
    
    if not os.path.exists(video_path):
        print(f"❌ 找不到影片: {video_path}")
        return
    
    # 載入場地配置
    court_config = None
    if court_config_path and os.path.exists(court_config_path):
        with open(court_config_path, 'r') as f:
            court_config = json.load(f)
        print(f"已載入場地配置: {court_config_path}")
    
    # 測試新版追蹤
    from video_processing.track_ball_and_player_v2 import run_tracking_v2
    
    output_dir = os.path.join(PROJECT_ROOT, "test_output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n開始追蹤...")
    start_time = time.time()
    
    json_path = run_tracking_v2(
        video_path=video_path,
        output_dir=output_dir,
        court_config=court_config,
        detection_interval=1,
        use_ball_tracker=True,
        max_occlusion_frames=15,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    print(f"\n追蹤完成！耗時: {elapsed:.1f} 秒")
    
    # 載入結果並測試發球偵測
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames_data = data.get('frames', data)  # 兼容兩種格式
    if isinstance(frames_data, dict):
        frames_data = list(frames_data.values())
    
    print(f"\n分析發球事件...")
    from core.serve_detector import analyze_serve_events_v2
    
    events = analyze_serve_events_v2(
        frames_data,
        config={'hit_v': 40.0, 'toss_vy': 8.0},
        use_dynamic_threshold=True,
        first_only=True  # 每段影片只取第一個發球事件
    )
    
    print(f"\n結果：")
    print(f"  總幀數: {len(frames_data)}")
    print(f"  偵測到發球事件: {len(events)}")
    
    if events:
        print(f"\n發球事件詳情：")
        for i, event in enumerate(events):
            print(f"  {i+1}. 擊球幀={event['hit_frame_id']}, "
                  f"速度={event['hit_speed']:.1f}, "
                  f"拋球幀={event.get('toss_start_frame', 'N/A')}")
    
    # 🆕 自動儲存關鍵幀圖片
    save_key_frames(video_path, events, output_dir)
    
    print(f"\n輸出檔案: {json_path}")
    print("\n✅ 影片測試完成！")
    
    # 提示：如果偵測到多個事件但影片只有一次發球
    if len(events) > 1:
        print("\n⚠️  提示：偵測到多個發球事件，但你說影片只有一次發球。")
        print("   請檢查關鍵幀圖片，確認哪個是真正的發球。")
        print("   如果有誤判，我們需要調整參數。")


def compare_versions(video_path: str):
    """比較新舊版本的追蹤效果"""
    print("\n" + "="*60)
    print("比較新舊版本追蹤效果")
    print("="*60)
    
    if not os.path.exists(video_path):
        print(f"❌ 找不到影片: {video_path}")
        return
    
    output_dir = os.path.join(PROJECT_ROOT, "test_output", "comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # 測試舊版（無追蹤器）
    print("\n--- 測試舊版（基礎偵測）---")
    from video_processing.track_ball_and_player_v2 import run_tracking_v2
    
    start_old = time.time()
    json_old = run_tracking_v2(
        video_path=video_path,
        output_dir=os.path.join(output_dir, "old"),
        use_ball_tracker=False,
        verbose=False
    )
    time_old = time.time() - start_old
    
    # 測試新版（有追蹤器）
    print("\n--- 測試新版（BallTracker）---")
    start_new = time.time()
    json_new = run_tracking_v2(
        video_path=video_path,
        output_dir=os.path.join(output_dir, "new"),
        use_ball_tracker=True,
        verbose=False
    )
    time_new = time.time() - start_new
    
    # 載入結果
    with open(json_old, 'r') as f:
        data_old = json.load(f)
    with open(json_new, 'r') as f:
        data_new = json.load(f)
    
    frames_old = data_old.get('frames', data_old)
    frames_new = data_new.get('frames', data_new)
    
    # 統計
    def count_ball_detections(frames):
        count = 0
        for f in frames:
            balls = f.get('ball_detections', [])
            valid = [b for b in balls if not b.get('is_in_background_zone', False)]
            if valid:
                count += 1
        return count
    
    def count_predicted_frames(frames):
        count = 0
        for f in frames:
            tracking = f.get('ball_tracking', {})
            if tracking.get('is_predicted', False):
                count += 1
        return count
    
    total_frames = len(frames_old)
    det_old = count_ball_detections(frames_old)
    det_new = count_ball_detections(frames_new)
    pred_new = count_predicted_frames(frames_new)
    
    print("\n" + "="*60)
    print("比較結果")
    print("="*60)
    print(f"\n{'指標':<25} {'舊版':>12} {'新版':>12} {'改進':>12}")
    print("-"*60)
    print(f"{'處理時間 (秒)':<25} {time_old:>12.1f} {time_new:>12.1f} {(time_new-time_old)/time_old*100:>+11.1f}%")
    print(f"{'球偵測幀數':<25} {det_old:>12} {det_new:>12} {det_new-det_old:>+12}")
    print(f"{'預測填補幀數':<25} {'N/A':>12} {pred_new:>12} {pred_new:>+12}")
    print(f"{'有效覆蓋率':<25} {100*det_old/total_frames:>11.1f}% {100*det_new/total_frames:>11.1f}% {100*(det_new-det_old)/total_frames:>+11.1f}%")
    print("-"*60)
    
    # 發球偵測比較
    from core.serve_detector import analyze_serve_events_v2
    
    events_old = analyze_serve_events_v2(frames_old, log_prefix="[舊版] ", use_dynamic_threshold=False)
    events_new = analyze_serve_events_v2(frames_new, log_prefix="[新版] ", use_dynamic_threshold=True)
    
    print(f"\n{'發球偵測數量':<25} {len(events_old):>12} {len(events_new):>12}")
    
    print("\n✅ 比較完成！")


def test_on_folder(folder_path: str, court_config_path: str = None):
    """
    批次測試資料夾中的所有影片
    
    Args:
        folder_path: 包含影片的資料夾路徑
        court_config_path: 場地配置檔案路徑
    """
    print("\n" + "="*60)
    print(f"批次測試資料夾")
    print(f"資料夾: {folder_path}")
    print("="*60)
    
    if not os.path.exists(folder_path):
        print(f"❌ 找不到資料夾: {folder_path}")
        return
    
    # 找出所有影片檔案
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = []
    for f in os.listdir(folder_path):
        if f.lower().endswith(video_extensions):
            video_files.append(os.path.join(folder_path, f))
    
    if not video_files:
        print(f"❌ 資料夾中沒有找到影片檔案")
        return
    
    video_files.sort()  # 按名稱排序
    print(f"\n找到 {len(video_files)} 個影片檔案")
    
    # 載入場地配置
    court_config = None
    if court_config_path and os.path.exists(court_config_path):
        with open(court_config_path, 'r') as f:
            court_config = json.load(f)
        print(f"已載入場地配置: {court_config_path}")
    
    # 準備輸出目錄
    output_dir = os.path.join(PROJECT_ROOT, "test_output")
    os.makedirs(output_dir, exist_ok=True)
    
    from video_processing.track_ball_and_player_v2 import run_tracking_v2
    from core.serve_detector import analyze_serve_events_v2
    
    # 統計結果
    results = []
    total_start_time = time.time()
    
    print(f"\n{'='*60}")
    print("開始批次處理...")
    print(f"{'='*60}\n")
    
    for idx, video_path in enumerate(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n[{idx+1}/{len(video_files)}] 處理: {video_name}")
        print("-" * 40)
        
        try:
            # Step 1: 追蹤
            start_time = time.time()
            json_path = run_tracking_v2(
                video_path=video_path,
                output_dir=output_dir,
                court_config=court_config,
                detection_interval=1,
                use_ball_tracker=True,
                max_occlusion_frames=15,
                verbose=False  # 批次模式下減少輸出
            )
            tracking_time = time.time() - start_time
            
            # Step 2: 載入追蹤結果
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            frames_data = data.get('frames', data)
            if isinstance(frames_data, dict):
                frames_data = list(frames_data.values())
            
            # Step 3: 發球偵測
            events = analyze_serve_events_v2(
                frames_data,
                config={'hit_v': 40.0, 'toss_vy': 8.0},
                log_prefix=f"  [{video_name}] ",
                use_dynamic_threshold=True,
                first_only=False  # 批次測試時取所有事件，方便檢查
            )
            
            # Step 4: 儲存關鍵幀
            save_key_frames(video_path, events, output_dir)
            
            # 記錄結果
            result = {
                'video': video_name,
                'status': 'success',
                'frames': len(frames_data),
                'events': len(events),
                'event_details': [(e['hit_frame_id'], e['hit_speed']) for e in events],
                'time': tracking_time
            }
            results.append(result)
            
            print(f"  ✅ 完成！幀數={len(frames_data)}, 事件數={len(events)}, 耗時={tracking_time:.1f}s")
            
        except Exception as e:
            print(f"  ❌ 錯誤: {e}")
            results.append({
                'video': video_name,
                'status': 'failed',
                'error': str(e)
            })
    
    total_time = time.time() - total_start_time
    
    # 顯示總結
    print(f"\n{'='*60}")
    print("批次處理完成！")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\n📊 統計結果：")
    print(f"  總影片數: {len(video_files)}")
    print(f"  成功: {success_count}")
    print(f"  失敗: {failed_count}")
    print(f"  總耗時: {total_time:.1f} 秒")
    print(f"  平均每片: {total_time/len(video_files):.1f} 秒")
    
    print(f"\n📋 詳細結果：")
    print("-" * 70)
    print(f"{'影片名稱':<30} {'狀態':<10} {'事件數':<10} {'擊球幀':<20}")
    print("-" * 70)
    
    for r in results:
        if r['status'] == 'success':
            events_str = ', '.join([str(e[0]) for e in r['event_details']]) if r['event_details'] else 'None'
            print(f"{r['video']:<30} {'✅ 成功':<10} {r['events']:<10} {events_str:<20}")
        else:
            print(f"{r['video']:<30} {'❌ 失敗':<10} {'-':<10} {r.get('error', '')[:20]:<20}")
    
    print("-" * 70)
    
    # 儲存結果到 JSON
    results_path = os.path.join(output_dir, "batch_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📁 結果已儲存: {results_path}")
    
    key_frames_dir = os.path.join(output_dir, "key_frames")
    print(f"📁 關鍵幀目錄: {key_frames_dir}")


def main():
    parser = argparse.ArgumentParser(description="改進功能測試腳本")
    parser.add_argument("--test", type=str, required=True,
                        choices=['tracker', 'serve', 'video', 'batch', 'compare', 'all'],
                        help="測試類型: tracker, serve, video, batch, compare, all")
    parser.add_argument("--input", type=str, help="輸入影片路徑或資料夾路徑")
    parser.add_argument("--court_config", type=str, help="場地配置檔案")
    
    args = parser.parse_args()
    
    if args.test == 'tracker':
        test_ball_tracker()
    elif args.test == 'serve':
        test_serve_detector()
    elif args.test == 'video':
        if not args.input:
            print("❌ 請使用 --input 指定影片路徑")
            return
        test_on_video(args.input, args.court_config)
    elif args.test == 'batch':
        if not args.input:
            print("❌ 請使用 --input 指定資料夾路徑")
            return
        test_on_folder(args.input, args.court_config)
    elif args.test == 'compare':
        if not args.input:
            print("❌ 請使用 --input 指定影片路徑")
            return
        compare_versions(args.input)
    elif args.test == 'all':
        test_ball_tracker()
        test_serve_detector()
        if args.input:
            test_on_video(args.input, args.court_config)
            compare_versions(args.input)
        else:
            print("\n提示：加上 --input 參數可以進行實際影片測試")


if __name__ == "__main__":
    main()