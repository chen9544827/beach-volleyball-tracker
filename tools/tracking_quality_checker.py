# tools/tracking_quality_checker.py
# 追蹤品質評估工具 - 分析 JSON 追蹤數據的品質指標

import json
import argparse
import sys
import os
from collections import defaultdict, Counter
import numpy as np


class TrackingQualityChecker:
    """追蹤品質分析器"""
    
    def __init__(self, json_path):
        self.json_path = json_path
        self.frames_data = None
        self.ball_tracks = defaultdict(list)  # {track_id: [frame_ids]}
        self.player_tracks = defaultdict(list)  # {track_id: [frame_ids]}
        
    def load_data(self):
        """載入 JSON 追蹤數據"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.frames_data = json.load(f)
            print(f"✓ 成功載入 {len(self.frames_data)} 幀的追蹤數據")
            return True
        except Exception as e:
            print(f"❌ 載入 JSON 失敗: {e}")
            return False
    
    def extract_tracks(self):
        """提取所有追蹤軌跡"""
        for frame_data in self.frames_data:
            frame_id = frame_data['frame_id']
            
            # 提取球體追蹤
            for ball in frame_data.get('ball_detections', []):
                track_id = ball.get('track_id', -1)
                if track_id >= 0:  # 只記錄有效的 track_id
                    self.ball_tracks[track_id].append(frame_id)
            
            # 提取球員追蹤
            for player in frame_data.get('player_detections', []):
                track_id = player.get('track_id', -1)
                if track_id >= 0:
                    self.player_tracks[track_id].append(frame_id)
        
        print(f"✓ 提取到 {len(self.ball_tracks)} 個球體軌跡")
        print(f"✓ 提取到 {len(self.player_tracks)} 個球員軌跡")
    
    def calculate_metrics(self, tracks_dict, object_type="Object"):
        """計算追蹤品質指標"""
        if not tracks_dict:
            print(f"\n⚠️  {object_type}: 無追蹤數據 (可能未啟用追蹤模式)")
            return None
        
        metrics = {
            "total_tracks": len(tracks_dict),
            "track_lengths": [],
            "track_continuity": [],
            "avg_track_length": 0,
            "max_track_length": 0,
            "min_track_length": 0,
            "avg_continuity": 0,
            "id_switches_estimate": 0
        }
        
        for track_id, frame_ids in tracks_dict.items():
            track_length = len(frame_ids)
            metrics["track_lengths"].append(track_length)
            
            # 計算連續性 (實際幀數 / 預期幀數)
            if track_length > 1:
                expected_frames = frame_ids[-1] - frame_ids[0] + 1
                continuity = track_length / expected_frames
                metrics["track_continuity"].append(continuity)
            else:
                metrics["track_continuity"].append(1.0)
        
        if metrics["track_lengths"]:
            metrics["avg_track_length"] = np.mean(metrics["track_lengths"])
            metrics["max_track_length"] = np.max(metrics["track_lengths"])
            metrics["min_track_length"] = np.min(metrics["track_lengths"])
        
        if metrics["track_continuity"]:
            metrics["avg_continuity"] = np.mean(metrics["track_continuity"])
        
        # 估計 ID 切換次數 (過短的軌跡可能是 ID 切換導致)
        short_tracks = sum(1 for length in metrics["track_lengths"] if length < 10)
        metrics["id_switches_estimate"] = short_tracks
        
        return metrics
    
    def print_report(self):
        """輸出品質報告"""
        print("\n" + "="*80)
        print("追蹤品質分析報告")
        print("="*80)
        print(f"\n影片: {os.path.basename(self.json_path)}")
        print(f"總幀數: {len(self.frames_data)}")
        
        # 球體追蹤指標
        print("\n" + "-"*80)
        print("🏐 球體追蹤指標")
        print("-"*80)
        ball_metrics = self.calculate_metrics(self.ball_tracks, "球體")
        
        if ball_metrics:
            print(f"  總軌跡數: {ball_metrics['total_tracks']}")
            print(f"  平均軌跡長度: {ball_metrics['avg_track_length']:.1f} 幀")
            print(f"  最長軌跡: {ball_metrics['max_track_length']} 幀")
            print(f"  最短軌跡: {ball_metrics['min_track_length']} 幀")
            print(f"  平均連續率: {ball_metrics['avg_continuity']*100:.1f}%")
            print(f"  疑似 ID 切換次數: {ball_metrics['id_switches_estimate']}")
            
            # 品質評估
            if ball_metrics['avg_continuity'] > 0.95:
                quality = "優秀 ✅"
            elif ball_metrics['avg_continuity'] > 0.85:
                quality = "良好 ✓"
            elif ball_metrics['avg_continuity'] > 0.70:
                quality = "中等 ⚠️"
            else:
                quality = "需改善 ❌"
            print(f"\n  整體品質: {quality}")
        
        # 球員追蹤指標
        print("\n" + "-"*80)
        print("👤 球員追蹤指標")
        print("-"*80)
        player_metrics = self.calculate_metrics(self.player_tracks, "球員")
        
        if player_metrics:
            print(f"  總軌跡數: {player_metrics['total_tracks']}")
            print(f"  平均軌跡長度: {player_metrics['avg_track_length']:.1f} 幀")
            print(f"  最長軌跡: {player_metrics['max_track_length']} 幀")
            print(f"  最短軌跡: {player_metrics['min_track_length']} 幀")
            print(f"  平均連續率: {player_metrics['avg_continuity']*100:.1f}%")
            print(f"  疑似 ID 切換次數: {player_metrics['id_switches_estimate']}")
            
            # 品質評估
            if player_metrics['avg_continuity'] > 0.95:
                quality = "優秀 ✅"
            elif player_metrics['avg_continuity'] > 0.85:
                quality = "良好 ✓"
            elif player_metrics['avg_continuity'] > 0.70:
                quality = "中等 ⚠️"
            else:
                quality = "需改善 ❌"
            print(f"\n  整體品質: {quality}")
        
        # 詳細軌跡分佈
        if ball_metrics and ball_metrics['track_lengths']:
            print("\n" + "-"*80)
            print("📊 球體軌跡長度分佈")
            print("-"*80)
            self._print_histogram(ball_metrics['track_lengths'])
        
        if player_metrics and player_metrics['track_lengths']:
            print("\n" + "-"*80)
            print("📊 球員軌跡長度分佈")
            print("-"*80)
            self._print_histogram(player_metrics['track_lengths'])
        
        print("\n" + "="*80)
        
        return ball_metrics, player_metrics
    
    def _print_histogram(self, lengths, bins=5):
        """輸出軌跡長度直方圖"""
        if not lengths:
            print("  無數據")
            return
        
        max_length = max(lengths)
        min_length = min(lengths)
        bin_size = max(1, (max_length - min_length) // bins)
        
        # 建立分組
        hist_bins = [min_length + i * bin_size for i in range(bins + 1)]
        hist_bins[-1] = max_length + 1  # 確保包含最大值
        
        hist = defaultdict(int)
        for length in lengths:
            for i in range(len(hist_bins) - 1):
                if hist_bins[i] <= length < hist_bins[i + 1]:
                    hist[i] += 1
                    break
        
        # 輸出直方圖
        for i in range(len(hist_bins) - 1):
            count = hist[i]
            bar = "█" * int(count * 40 / len(lengths))
            print(f"  {hist_bins[i]:4d}-{hist_bins[i+1]-1:4d} 幀: {bar} ({count})")
    
    def save_report(self, output_path):
        """儲存報告到文字檔"""
        # TODO: 實作報告儲存功能
        pass


def main():
    parser = argparse.ArgumentParser(
        description="追蹤品質評估工具 - 分析 JSON 追蹤數據"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="追蹤 JSON 檔案路徑 (例如: *_all_frames_data_with_pose.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="輸出報告路徑 (可選)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_path):
        print(f"❌ 錯誤: 找不到 JSON 檔案 '{args.json_path}'")
        sys.exit(1)
    
    # 執行分析
    checker = TrackingQualityChecker(args.json_path)
    
    if not checker.load_data():
        sys.exit(1)
    
    checker.extract_tracks()
    ball_metrics, player_metrics = checker.print_report()
    
    if args.output:
        checker.save_report(args.output)
    
    # 根據品質決定退出碼
    if ball_metrics and ball_metrics['avg_continuity'] < 0.70:
        print("\n⚠️  球體追蹤品質較低，建議調整追蹤器參數")
        sys.exit(2)
    
    if player_metrics and player_metrics['avg_continuity'] < 0.70:
        print("\n⚠️  球員追蹤品質較低，建議調整追蹤器參數")
        sys.exit(2)


if __name__ == "__main__":
    main()
