# batch_slice_videos.py
# 批次處理多個長影片，使用記分板 ROI 監控自動切片

import os
import sys
import argparse
import subprocess
import csv
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def find_video_files(directory, extensions=('.mp4', '.avi', '.mov', '.mkv')):
    """搜尋目錄中的所有影片檔案"""
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                video_files.append(os.path.join(root, file))
    return video_files


def process_single_video(video_path, output_base_dir, scoreboard_config, args_dict):
    """處理單一影片的切片作業"""
    video_name = Path(video_path).stem
    log_prefix = f"[{video_name}]"
    
    # 為每個影片建立獨立的輸出目錄
    video_output_dir = os.path.join(output_base_dir, f"{video_name}_segments")
    
    try:
        print(f"{log_prefix} 開始處理...")
        
        # 構建切片命令
        cmd = [
            sys.executable,
            os.path.join("video_processing", "video_slicer_by_score.py"),
            "--input", video_path,
            "--output_dir", video_output_dir,
            "--min_segment_duration", str(args_dict['min_segment_duration']),
            "--long_segment_threshold", str(args_dict['long_segment_threshold']),
            "--roi_check_interval", str(args_dict['roi_check_interval']),
            "--diff_threshold", str(args_dict['diff_threshold'])
        ]
        
        # 如果提供了配置檔，加入參數
        if scoreboard_config:
            cmd.extend(["--scoreboard_config", scoreboard_config])
        
        # 執行切片腳本
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode != 0:
            error_msg = f"切片失敗 (Return Code: {result.returncode})"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr[:500]}"
            return {
                "video": video_name,
                "status": "failed",
                "message": error_msg,
                "output_dir": video_output_dir,
                "segments_count": 0
            }
        
        # 讀取該影片的切片報告
        summary_csv = os.path.join(video_output_dir, "slicing_summary.csv")
        segments_count = 0
        
        if os.path.exists(summary_csv):
            with open(summary_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                segments = list(reader)
                segments_count = len(segments)
        
        print(f"{log_prefix} 完成 - 產生 {segments_count} 個片段")
        
        return {
            "video": video_name,
            "status": "success",
            "message": f"成功產生 {segments_count} 個片段",
            "output_dir": video_output_dir,
            "segments_count": segments_count
        }
        
    except Exception as e:
        return {
            "video": video_name,
            "status": "failed",
            "message": f"執行時發生異常: {str(e)}",
            "output_dir": video_output_dir,
            "segments_count": 0
        }


def generate_master_report(results, output_path, start_time, args):
    """產生總體批次處理報告"""
    end_time = datetime.now()
    duration = end_time - start_time
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    total_segments = sum(r['segments_count'] for r in results)
    
    report_lines = [
        "="*80,
        "沙灘排球影片批次切片處理報告",
        "="*80,
        f"\n執行時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"完成時間: {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"總耗時: {duration}",
        f"\n處理參數:",
        f"  - 最小片段時長: {args.min_segment_duration} 秒",
        f"  - 長片段閾值: {args.long_segment_threshold} 秒",
        f"  - ROI 檢查間隔: {args.roi_check_interval} 秒",
        f"  - SAD 閾值: {args.diff_threshold}",
        f"  - 並行執行數: {args.workers}",
        f"\n處理結果統計:",
        f"  - 總影片數: {len(results)}",
        f"  - 成功處理: {success_count}",
        f"  - 處理失敗: {failed_count}",
        f"  - 總產生片段: {total_segments}",
        "\n" + "="*80,
        "\n詳細結果:\n"
    ]
    
    # 成功的影片
    report_lines.append("✅ 成功處理的影片:")
    for r in sorted(results, key=lambda x: x['video']):
        if r['status'] == 'success':
            report_lines.append(f"  - {r['video']}: {r['segments_count']} 個片段")
            report_lines.append(f"    輸出目錄: {r['output_dir']}")
    
    # 失敗的影片
    if failed_count > 0:
        report_lines.append("\n❌ 處理失敗的影片:")
        for r in sorted(results, key=lambda x: x['video']):
            if r['status'] == 'failed':
                report_lines.append(f"  - {r['video']}")
                report_lines.append(f"    錯誤: {r['message']}")
    
    report_lines.append("\n" + "="*80)
    
    # 寫入文字報告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n總報告已儲存至: {os.path.abspath(output_path)}")
    return report_lines


def generate_master_csv(results, output_path):
    """產生 CSV 格式的總報告"""
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Video_Name', 'Status', 'Segments_Count', 'Output_Directory', 'Message']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in sorted(results, key=lambda x: x['video']):
                writer.writerow({
                    'Video_Name': r['video'],
                    'Status': r['status'],
                    'Segments_Count': r['segments_count'],
                    'Output_Directory': r['output_dir'],
                    'Message': r['message']
                })
        
        print(f"CSV 報告已儲存至: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"⚠️  儲存 CSV 報告時發生錯誤: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="批次處理多個長影片，使用記分板 ROI 監控自動切片"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="包含長影片的輸入目錄"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_data/batch_slicing_results",
        help="儲存所有切片結果的根目錄"
    )
    parser.add_argument(
        "--scoreboard_config",
        type=str,
        help="記分板配置檔路徑 (scoreboard_config.json)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="並行處理的最大執行緒數"
    )
    parser.add_argument(
        "--min_segment_duration",
        type=int,
        default=10,
        help="有效片段的最小持續時間 (秒)"
    )
    parser.add_argument(
        "--long_segment_threshold",
        type=int,
        default=90,
        help="長片段的閾值 (秒)"
    )
    parser.add_argument(
        "--roi_check_interval",
        type=float,
        default=0.5,
        help="ROI 檢查間隔 (秒)"
    )
    parser.add_argument(
        "--diff_threshold",
        type=int,
        default=9000,
        help="SAD 閾值"
    )
    
    args = parser.parse_args()
    
    # 檢查輸入目錄
    if not os.path.exists(args.input_dir):
        print(f"❌ 錯誤: 找不到輸入目錄 '{args.input_dir}'")
        sys.exit(1)
    
    # 搜尋影片檔案
    video_files = find_video_files(args.input_dir)
    
    if not video_files:
        print(f"❌ 錯誤: 在 '{args.input_dir}' 中找不到任何影片檔案")
        sys.exit(1)
    
    print(f"\n找到 {len(video_files)} 個影片檔案")
    print(f"將使用 {args.workers} 個並行執行緒處理\n")
    
    # 建立輸出目錄
    output_base_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 準備參數字典
    args_dict = {
        'min_segment_duration': args.min_segment_duration,
        'long_segment_threshold': args.long_segment_threshold,
        'roi_check_interval': args.roi_check_interval,
        'diff_threshold': args.diff_threshold
    }
    
    start_time = datetime.now()
    results = []
    
    # 使用 ProcessPoolExecutor 進行並行處理
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_video,
                video_path,
                output_base_dir,
                args.scoreboard_config,
                args_dict
            ): video_path
            for video_path in video_files
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                video_path = futures[future]
                video_name = Path(video_path).stem
                print(f"[{video_name}] 處理時發生嚴重錯誤: {e}")
                results.append({
                    "video": video_name,
                    "status": "failed",
                    "message": f"執行時發生嚴重錯誤: {str(e)}",
                    "output_dir": "",
                    "segments_count": 0
                })
    
    # 產生總報告
    print("\n" + "="*80)
    print("批次處理完成，正在產生總報告...")
    print("="*80)
    
    txt_report_path = os.path.join(output_base_dir, "master_slicing_report.txt")
    csv_report_path = os.path.join(output_base_dir, "master_slicing_report.csv")
    
    report_lines = generate_master_report(results, txt_report_path, start_time, args)
    generate_master_csv(results, csv_report_path)
    
    # 顯示摘要
    print("\n" + "="*80)
    print("處理完成！")
    print("="*80)
    for line in report_lines[:20]:  # 顯示前20行
        print(line)


if __name__ == "__main__":
    main()
