# combine_csv_reports.py (Pandas 穩定版)
import os
import pandas as pd

ARCHIVE_FOLDER = 'csv_reports_archive'
MASTER_REPORT_FILENAME = 'MASTER_REPORT.csv'

def combine_reports():
    archive_path = os.path.abspath(ARCHIVE_FOLDER)
    if not os.path.exists(archive_path):
        print(f"錯誤：找不到存檔資料夾 '{archive_path}'。")
        return

    all_csv_files = [os.path.join(root, file) for root, _, files in os.walk(archive_path) for file in files if file == 'analysis_summary.csv']

    if not all_csv_files:
        print(f"在 '{archive_path}' 中找不到任何報告可供合併。")
        return

    print(f"找到 {len(all_csv_files)} 份報告，準備合併...")
    
    master_df = pd.concat([pd.read_csv(f) for f in all_csv_files], ignore_index=True)

    try:
        master_df.to_csv(MASTER_REPORT_FILENAME, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 避免Excel中文亂碼
        print("\n" + "="*50)
        print(f"✅ 成功！所有報告已合併。")
        print(f"總共合併了 {len(master_df)} 筆發球數據。")
        print(f"最終總報告已儲存至：{os.path.abspath(MASTER_REPORT_FILENAME)}")
        print("="*50)
    except Exception as e:
        print(f"\n錯誤：儲存總報告 '{MASTER_REPORT_FILENAME}' 時失敗: {e}")

if __name__ == '__main__':
    combine_reports()