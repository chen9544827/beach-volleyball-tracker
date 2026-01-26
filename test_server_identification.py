# test_server_identification.py
# -*- coding: utf-8 -*-
"""
測試發球員識別

整合發球偵測和發球員識別
"""

import os
import sys
import json
import argparse
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.serve_detector import analyze_serve_events_v2
from core.server_identifier import analyze_serve_player, get_keypoint


def load_tracking_data(json_path: str) -> dict:
    """載入追蹤數據"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_frame_by_id(frames_data: list, frame_id: int) -> dict:
    """根據 frame_id 取得幀資料"""
    for frame in frames_data:
        if frame.get('frame_id') == frame_id:
            return frame
    return None


def draw_serve_analysis(frame, frame_data, ball_position, server_result, frame_label=""):
    """
    在影格上繪製發球分析結果
    """
    # 繪製球位置
    if ball_position:
        x, y = int(ball_position[0]), int(ball_position[1])
        cv2.circle(frame, (x, y), 20, (0, 255, 255), 3)  # 黃色圈
        cv2.putText(frame, "BALL", (x - 20, y - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 繪製所有球員
    players = frame_data.get('player_detections', [])
    server_idx = server_result.get('final_server_index')
    
    for i, player in enumerate(players):
        box = player.get('box_coords')
        if not box:
            continue
        
        # 發球員用綠色，其他用藍色
        if i == server_idx:
            color = (0, 255, 0)  # 綠色
            label = f"SERVER (conf: {server_result.get('final_confidence', 0):.2f})"
        else:
            color = (255, 0, 0)  # 藍色
            label = f"Player {i}"
        
        # 繪製邊框
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 繪製手腕位置
        left_wrist = get_keypoint(player, 9)
        right_wrist = get_keypoint(player, 10)
        
        if left_wrist:
            cv2.circle(frame, (int(left_wrist[0]), int(left_wrist[1])), 8, (0, 0, 255), -1)
        if right_wrist:
            cv2.circle(frame, (int(right_wrist[0]), int(right_wrist[1])), 8, (0, 0, 255), -1)
    
    # 顯示資訊
    info_lines = [
        f"{frame_label}",
        f"Server: Player {server_idx}" if server_idx is not None else "Server: Unknown",
        f"Confidence: {server_result.get('final_confidence', 0):.2f}",
    ]
    
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame


def test_single_video(video_path: str, json_path: str, output_dir: str = "test_output"):
    """
    測試單個影片的發球員識別
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n{'='*60}")
    print(f"測試影片: {video_name}")
    print(f"{'='*60}")
    
    # 載入追蹤數據
    print("載入追蹤數據...")
    data = load_tracking_data(json_path)
    frames_data = data.get('frames', [])
    print(f"  共 {len(frames_data)} 幀")
    
    # 發球偵測
    print("\n發球偵測...")
    serve_events = analyze_serve_events_v2(
        frames_data,
        config={'hit_v': 40.0, 'toss_vy': 8.0},
        use_dynamic_threshold=True,
        first_only=True,
        log_prefix="  "
    )
    
    if not serve_events:
        print("  沒有偵測到發球事件")
        return None
    
    serve_event = serve_events[0]
    toss_frame_id = serve_event.get('toss_start_frame')
    hit_frame_id = serve_event['hit_frame_id']
    
    print(f"\n發球事件:")
    print(f"  拋球幀: {toss_frame_id}")
    print(f"  擊球幀: {hit_frame_id}")
    print(f"  拋球位置: {serve_event.get('toss_position')}")
    print(f"  擊球位置: {serve_event.get('hit_position')}")
    print(f"  擊球速度: {serve_event.get('hit_speed', 0):.1f}")
    
    # 發球員識別（使用拋球幀）
    print("\n發球員識別（使用拋球幀）...")
    server_result = analyze_serve_player(
        frames_data=frames_data,
        serve_event=serve_event,
        method='toss_frame'  # 使用拋球幀判斷
    )
    
    print(f"  發球員索引: {server_result['final_server_index']}")
    print(f"  信心度: {server_result['final_confidence']:.2f}")
    
    if server_result.get('toss_result'):
        toss_r = server_result['toss_result']
        print(f"\n  拋球幀分析:")
        print(f"    手腕位置: {toss_r.get('wrist_position')}")
        print(f"    手腕到球距離: {toss_r.get('wrist_to_ball_distance')}")
        print(f"    所有候選:")
        for c in toss_r['details']['all_candidates']:
            print(f"      球員 {c['index']}: 手腕距離={c['wrist_distance']}, 分數={c['score']}")
    
    # 輸出視覺化圖片
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n正在開啟影片: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        print(f"  影片開啟成功")
        
        # 輸出拋球幀圖片
        if toss_frame_id:
            cap.set(cv2.CAP_PROP_POS_FRAMES, toss_frame_id)
            ret, frame = cap.read()
            if ret:
                toss_frame_data = get_frame_by_id(frames_data, toss_frame_id)
                if toss_frame_data:
                    frame = draw_serve_analysis(
                        frame, toss_frame_data, 
                        serve_event.get('toss_position'),
                        server_result,
                        f"TOSS Frame {toss_frame_id}"
                    )
                    output_path = os.path.join(output_dir, f"{video_name}_server_TOSS.jpg")
                    cv2.imwrite(output_path, frame)
                    print(f"  ✓ 已儲存拋球幀圖片: {output_path}")
        
        # 輸出擊球幀圖片
        cap.set(cv2.CAP_PROP_POS_FRAMES, hit_frame_id)
        ret, frame = cap.read()
        if ret:
            hit_frame_data = get_frame_by_id(frames_data, hit_frame_id)
            if hit_frame_data:
                frame = draw_serve_analysis(
                    frame, hit_frame_data,
                    serve_event.get('hit_position'),
                    server_result,
                    f"HIT Frame {hit_frame_id}"
                )
                output_path = os.path.join(output_dir, f"{video_name}_server_HIT.jpg")
                cv2.imwrite(output_path, frame)
                print(f"  ✓ 已儲存擊球幀圖片: {output_path}")
        
        cap.release()
    else:
        print(f"  ✗ 無法開啟影片，請確認路徑是否正確")
    
    return {
        'video_name': video_name,
        'serve_event': serve_event,
        'server_result': server_result
    }


def main():
    parser = argparse.ArgumentParser(description="發球員識別測試")
    parser.add_argument("--video", type=str, required=True, help="輸入影片路徑")
    parser.add_argument("--json", type=str, required=True, help="追蹤結果 JSON 路徑")
    parser.add_argument("--output", type=str, default="test_output", help="輸出目錄")
    
    args = parser.parse_args()
    
    result = test_single_video(args.video, args.json, args.output)
    
    if result:
        print("\n" + "="*60)
        print("測試完成！")
        print("="*60)


if __name__ == "__main__":
    main()