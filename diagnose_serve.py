# diagnose_serve.py
# -*- coding: utf-8 -*-
"""
發球偵測診斷工具

分析追蹤數據，找出為什麼發球偵測失敗
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def analyze_ball_trajectory(frames_data: list) -> dict:
    """
    分析球的軌跡，找出所有可能的發球模式
    """
    results = {
        'total_frames': len(frames_data),
        'frames_with_ball': 0,
        'ball_positions': [],
        'speeds': [],
        'vertical_speeds': [],  # 正值=向上，負值=向下
        'horizontal_speeds': [],
        'upward_sequences': [],  # 連續向上的片段
        'high_speed_events': [],  # 高速事件
        'potential_serves': []   # 潛在的發球
    }
    
    prev_pos = None
    current_upward_seq = None
    
    for i, frame in enumerate(frames_data):
        frame_id = frame.get('frame_id', i)
        
        # 取得球位置
        ball_pos = None
        if frame.get('ball_detections'):
            valid_balls = [b for b in frame['ball_detections'] 
                          if not b.get('is_in_background_zone', False)]
            if valid_balls:
                best_ball = max(valid_balls, key=lambda b: b.get('confidence', 0))
                box = best_ball.get('box_coords')
                if box:
                    ball_pos = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                elif best_ball.get('center_point'):
                    ball_pos = np.array(best_ball['center_point'])
        
        if ball_pos is not None:
            results['frames_with_ball'] += 1
            results['ball_positions'].append({
                'frame_id': frame_id,
                'x': ball_pos[0],
                'y': ball_pos[1]
            })
            
            if prev_pos is not None:
                velocity = ball_pos - prev_pos
                vx, vy = velocity[0], velocity[1]
                speed = np.linalg.norm(velocity)
                upward_vy = -vy  # 轉換為向上為正
                
                results['speeds'].append({'frame_id': frame_id, 'speed': speed})
                results['vertical_speeds'].append({'frame_id': frame_id, 'vy': upward_vy})
                results['horizontal_speeds'].append({'frame_id': frame_id, 'vx': vx})
                
                # 偵測向上移動序列
                if upward_vy > 5:  # 向上移動閾值
                    if current_upward_seq is None:
                        current_upward_seq = {
                            'start_frame': frame_id,
                            'frames': [frame_id],
                            'start_pos': prev_pos.copy(),
                            'max_vy': upward_vy
                        }
                    else:
                        current_upward_seq['frames'].append(frame_id)
                        current_upward_seq['max_vy'] = max(current_upward_seq['max_vy'], upward_vy)
                else:
                    if current_upward_seq is not None and len(current_upward_seq['frames']) >= 2:
                        current_upward_seq['end_frame'] = frame_id - 1
                        current_upward_seq['end_pos'] = prev_pos.copy()
                        current_upward_seq['duration'] = len(current_upward_seq['frames'])
                        results['upward_sequences'].append(current_upward_seq)
                    current_upward_seq = None
                
                # 偵測高速事件
                if speed > 30:
                    results['high_speed_events'].append({
                        'frame_id': frame_id,
                        'speed': speed,
                        'vx': vx,
                        'vy': vy,
                        'position': ball_pos.tolist()
                    })
            
            prev_pos = ball_pos.copy()
        else:
            prev_pos = None
            if current_upward_seq is not None and len(current_upward_seq['frames']) >= 2:
                current_upward_seq['end_frame'] = frame_id - 1
                current_upward_seq['duration'] = len(current_upward_seq['frames'])
                results['upward_sequences'].append(current_upward_seq)
            current_upward_seq = None
    
    return results


def find_potential_serves(analysis: dict) -> list:
    """
    從分析結果中找出潛在的發球
    
    發球特徵：
    1. 拋球：連續向上移動 3+ 幀
    2. 頂點：向上停止
    3. 擊球：高速向前移動
    """
    potential_serves = []
    
    upward_seqs = analysis['upward_sequences']
    high_speed_events = analysis['high_speed_events']
    
    for seq in upward_seqs:
        if seq['duration'] < 3:
            continue
        
        # 找這個向上序列後面的高速事件
        seq_end = seq.get('end_frame', seq['frames'][-1])
        
        for event in high_speed_events:
            # 高速事件應該在向上序列結束後 5-50 幀內
            frame_diff = event['frame_id'] - seq_end
            if 5 <= frame_diff <= 50:
                # 檢查是否有水平分量（發球特徵）
                h_ratio = abs(event['vx']) / (abs(event['vy']) + 1e-6)
                if 0.3 < h_ratio < 3.0:  # 發球的水平/垂直比例
                    potential_serves.append({
                        'toss_start': seq['frames'][0],
                        'toss_end': seq_end,
                        'toss_duration': seq['duration'],
                        'toss_max_vy': seq['max_vy'],
                        'hit_frame': event['frame_id'],
                        'hit_speed': event['speed'],
                        'hit_vx': event['vx'],
                        'hit_vy': event['vy'],
                        'hit_h_ratio': h_ratio,
                        'frames_toss_to_hit': frame_diff
                    })
                    break
    
    return potential_serves


def diagnose_video(video_name: str, json_path: str):
    """
    診斷單個影片的發球偵測
    """
    print(f"\n{'='*70}")
    print(f"診斷影片: {video_name}")
    print(f"{'='*70}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames_data = data.get('frames', data)
    if isinstance(frames_data, dict):
        frames_data = list(frames_data.values())
    
    # 分析軌跡
    analysis = analyze_ball_trajectory(frames_data)
    
    print(f"\n[統計] 基本統計:")
    print(f"  總幀數: {analysis['total_frames']}")
    print(f"  有球幀數: {analysis['frames_with_ball']} ({100*analysis['frames_with_ball']/analysis['total_frames']:.1f}%)")
    
    # 速度統計
    if analysis['speeds']:
        speeds = [s['speed'] for s in analysis['speeds']]
        print(f"\n[速度] 速度統計:")
        print(f"  平均速度: {np.mean(speeds):.1f}")
        print(f"  最大速度: {np.max(speeds):.1f}")
        print(f"  90百分位: {np.percentile(speeds, 90):.1f}")
    
    # 向上移動序列
    print(f"\n[拋球] 向上移動序列 (可能的拋球):")
    if analysis['upward_sequences']:
        for i, seq in enumerate(analysis['upward_sequences'][:10]):  # 最多顯示10個
            print(f"  {i+1}. 幀 {seq['frames'][0]}-{seq.get('end_frame', seq['frames'][-1])}, "
                  f"持續 {seq['duration']} 幀, 最大向上速度: {seq['max_vy']:.1f}")
    else:
        print("  沒有偵測到向上移動序列！")
    
    # 高速事件
    print(f"\n[擊球] 高速事件 (速度 > 30):")
    if analysis['high_speed_events']:
        for i, event in enumerate(analysis['high_speed_events'][:15]):  # 最多顯示15個
            h_ratio = abs(event['vx']) / (abs(event['vy']) + 1e-6)
            print(f"  {i+1}. 幀 {event['frame_id']}: 速度={event['speed']:.1f}, "
                  f"vx={event['vx']:.1f}, vy={event['vy']:.1f}, "
                  f"水平比={h_ratio:.2f}")
    else:
        print("  沒有高速事件！")
    
    # 潛在發球
    potential_serves = find_potential_serves(analysis)
    print(f"\n[結果] 潛在發球事件:")
    if potential_serves:
        for i, serve in enumerate(potential_serves):
            print(f"  {i+1}. 拋球: 幀 {serve['toss_start']}-{serve['toss_end']} "
                  f"(持續 {serve['toss_duration']} 幀, 最大vy={serve['toss_max_vy']:.1f})")
            print(f"     擊球: 幀 {serve['hit_frame']} "
                  f"(速度={serve['hit_speed']:.1f}, vx={serve['hit_vx']:.1f}, "
                  f"水平比={serve['hit_h_ratio']:.2f})")
            print(f"     拋球到擊球: {serve['frames_toss_to_hit']} 幀")
    else:
        print("  沒有找到潛在發球！這可能表示:")
        print("    - 拋球向上速度太低")
        print("    - 擊球水平/垂直比例不符合")
        print("    - 拋球和擊球之間的時間間隔不對")
    
    # 建議
    print(f"\n[建議] 診斷建議:")
    
    if not analysis['upward_sequences']:
        print("  [!] 沒有向上移動序列 - 可能需要降低 toss_vy 閾值")
    elif all(seq['max_vy'] < 8 for seq in analysis['upward_sequences']):
        print(f"  [!] 所有向上移動的速度都很低 (最大: {max(seq['max_vy'] for seq in analysis['upward_sequences']):.1f})")
        print("     建議降低 toss_vy 閾值到 5.0 或更低")

    if not analysis['high_speed_events']:
        print("  [!] 沒有高速事件 - 可能需要降低 hit_v 閾值")
    elif all(event['speed'] < 40 for event in analysis['high_speed_events']):
        max_speed = max(event['speed'] for event in analysis['high_speed_events'])
        print(f"  [!] 所有高速事件速度都 < 40 (最大: {max_speed:.1f})")
        print(f"     建議降低 hit_v 閾值到 {max_speed * 0.7:.1f}")

    # 檢查水平速度過濾
    high_h_ratio_events = [e for e in analysis['high_speed_events']
                          if abs(e['vx']) / (abs(e['vy']) + 1e-6) > 2.5]
    if high_h_ratio_events and not potential_serves:
        print(f"  [!] 有 {len(high_h_ratio_events)} 個高速事件被水平比例過濾掉")
        print("     可能需要調高 hit_h_ratio 閾值")

    low_h_ratio_events = [e for e in analysis['high_speed_events']
                          if abs(e['vx']) / (abs(e['vy']) + 1e-6) < 0.3]
    if low_h_ratio_events and not potential_serves:
        print(f"  [!] 有 {len(low_h_ratio_events)} 個高速事件水平速度太低")
        print("     可能需要降低 min_hit_horizontal_ratio")
    
    return analysis, potential_serves


def main():
    parser = argparse.ArgumentParser(description="發球偵測診斷工具")
    parser.add_argument("--input", type=str, required=True, 
                        help="JSON 追蹤結果檔案或資料夾")
    parser.add_argument("--video_name", type=str, default=None,
                        help="指定要診斷的影片名稱（當 input 是資料夾時）")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # 單個檔案
        video_name = os.path.splitext(os.path.basename(args.input))[0]
        diagnose_video(video_name, args.input)
    elif os.path.isdir(args.input):
        # 資料夾
        json_files = [f for f in os.listdir(args.input) if f.endswith('.json')]
        
        if args.video_name:
            # 只診斷指定的影片
            matching = [f for f in json_files if args.video_name in f]
            if matching:
                for f in matching:
                    video_name = os.path.splitext(f)[0]
                    diagnose_video(video_name, os.path.join(args.input, f))
            else:
                print(f"找不到包含 '{args.video_name}' 的 JSON 檔案")
        else:
            # 診斷所有
            for f in sorted(json_files)[:5]:  # 最多診斷 5 個
                video_name = os.path.splitext(f)[0]
                diagnose_video(video_name, os.path.join(args.input, f))
    else:
        print(f"找不到: {args.input}")


if __name__ == "__main__":
    main()