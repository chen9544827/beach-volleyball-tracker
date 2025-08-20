# combine_csv_reports.py (修改版)

import os
import csv
import argparse
from datetime import datetime

def find_csv_files_in_archive(archive_directory):
    """
    在指定的存檔目錄下，尋找所有 .csv 檔案。
    
    Args:
        archive_directory (str): CSV 存檔資料夾的路徑。
        
    Returns:
        list: 包含所有找到的 CSV 檔案絕對路徑的列表。
    """
    if not os.path.isdir(archive_directory):
        print(f"[錯誤] 指定的存檔資料夾 '{archive_directory}' 不存在。")
        return []
        
    csv_file_paths = []
    print(f"--- 正在 '{archive_directory}' 目錄下搜尋所有 .csv 檔案 ---")
    
    for filename in sorted(os.listdir(archive_directory)):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(archive_directory, filename)
            print(f"  [找到]: {file_path}")
            csv_file_paths.append(file_path)
            
    return csv_file_paths

def combine_csv_files(csv_paths, output_file):
    """
    將多個 CSV 檔案的內容合併成一個單一的 CSV 檔案。
    (此函數與前一版本完全相同)
    """
    if not csv_paths:
        print("\n[警告] 未找到任何 CSV 檔案，無法進行合併。")
        return

    all_data = []
    header_written = False
    
    for file_path in csv_paths:
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                header = next(reader)
                
                if not header_written:
                    all_data.append(header)
                    header_written = True
                    
                for row in reader:
                    if row: all_data.append(row)
        except Exception as e:
            print(f"[錯誤] 讀取檔案 '{file_path}' 時發生問題: {e}")

    if len(all_data) <= 1:
        print("\n[警告] 所有找到的 CSV 檔案中都沒有包含有效數據，已取消產生合併檔案。")
        return

    try:
        # 將合併報告儲存在存檔資料夾的上一層目錄，以避免下次被自己讀取
        output_dir = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(all_data)
        print(f"\n--- ✅ 合併成功 ---")
        print(f"總共合併了 {len(csv_paths)} 個 CSV 檔案。")
        print(f"包含 {len(all_data) - 1} 筆數據紀錄。")
        print(f"合併後的總報告已儲存至: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"\n[錯誤] 寫入合併檔案 '{output_file}' 時失敗: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="自動從存檔資料夾中，合併所有分散的 CSV 報告。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--reports_archive_folder", 
        type=str, 
        default="csv_reports_archive", 
        help="指定儲存了所有批次執行結果的CSV存檔資料夾。"
    )
    parser.add_argument(
        "--master_report_name", 
        type=str, 
        default="MASTER_REPORT.csv",
        help="指定最終合併完成的總報告檔案名稱。"
    )
    args = parser.parse_args()
        
    # 尋找所有目標 CSV 檔案
    csv_to_combine = find_csv_files_in_archive(args.reports_archive_folder)
    
    # 定義最終輸出檔案的路徑 (存在存檔資料夾的外面一層)
    output_file_path = os.path.join(os.getcwd(), args.master_report_name)
    
    # 執行合併
    combine_csv_files(csv_to_combine, output_file_path)

if __name__ == "__main__":
    main()