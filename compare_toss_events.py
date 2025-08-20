# compare_toss_events.py

import csv
import matplotlib.pyplot as plt
import argparse
import os

def load_debug_log(filepath):
    """從 CSV 檔案載入偵錯日誌數據"""
    data = {"frame_id": [], "vy": [], "vx": [], "ratio": []}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["frame_id"].append(int(row["frame_id"]))
                vy = float(row["vy"])
                vx = float(row["vx"])
                data["vy"].append(vy)
                data["vx"].append(vx)
                # 計算 ratio，與主程式邏輯保持一致
                ratio = abs(vy) / (abs(vx) + 1e-6) if vy > 0 else 0
                data["ratio"].append(ratio)
        return data
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {filepath}")
        return None
    except Exception as e:
        print(f"讀取檔案 {filepath} 時發生錯誤: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="比較兩個發球事件的偵錯日誌，並視覺化其差異。")
    parser.add_argument("--success_log", type=str, required=True, help="成功偵測案例的 _serve_debug_log.csv 檔案路徑。")
    parser.add_argument("--failure_log", type=str, required=True, help="偵測失敗案例的 _serve_debug_log.csv 檔案路徑。")
    parser.add_argument("--output_image", type=str, default="toss_comparison_plot.png", help="比較的結果圖表儲存路徑。")
    parser.add_argument("--vertical_ratio_thresh", type=float, default=1.5, help="圖表中顯示的垂直比例門檻線。")
    args = parser.parse_args()

    success_data = load_debug_log(args.success_log)
    failure_data = load_debug_log(args.failure_log)

    if not success_data or not failure_data:
        return

    # --- 開始繪圖 ---
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('成功案例 vs. 失敗案例 - 拋球數據比較', fontsize=20)

    # 1. 繪製 VY (垂直速度)
    axs[0].plot(success_data["frame_id"], success_data["vy"], 'g-', label='成功案例 (Success)')
    axs[0].plot(failure_data["frame_id"], failure_data["vy"], 'r--', label='失敗案例 (Failure)')
    axs[0].axhline(y=8.0, color='gray', linestyle=':', label='Toss VY Thresh (8.0)')
    axs[0].set_ylabel('垂直速度 VY (像素/幀)', fontsize=12)
    axs[0].set_title('垂直速度 (VY) 比較', fontsize=14)
    axs[0].legend()
    axs[0].grid(True)

    # 2. 繪製 VX (水平速度)
    axs[1].plot(success_data["frame_id"], success_data["vx"], 'g-', label='成功案例 (Success)')
    axs[1].plot(failure_data["frame_id"], failure_data["vx"], 'r--', label='失敗案例 (Failure)')
    axs[1].set_ylabel('水平速度 VX (像素/幀)', fontsize=12)
    axs[1].set_title('水平速度 (VX) 比較', fontsize=14)
    axs[1].legend()
    axs[1].grid(True)

    # 3. 繪製 Ratio (垂直/水平 速度比)
    axs[2].plot(success_data["frame_id"], success_data["ratio"], 'g-', label='成功案例 (Success)')
    axs[2].plot(failure_data["frame_id"], failure_data["ratio"], 'r--', label='失敗案例 (Failure)')
    axs[2].axhline(y=args.vertical_ratio_thresh, color='b', linestyle='-', linewidth=2, label=f'Vertical Ratio Thresh ({args.vertical_ratio_thresh})')
    axs[2].set_ylabel('速度比 (VY/VX)', fontsize=12)
    axs[2].set_title('關鍵指標：垂直/水平速度比 (Ratio) 比較', fontsize=16)
    axs[2].set_xlabel('幀號 (Frame ID)', fontsize=12)
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    try:
        plt.savefig(args.output_image)
        print(f"\n[成功] 比較圖表已儲存至: {os.path.abspath(args.output_image)}")
    except Exception as e:
        print(f"\n[錯誤] 儲存圖表失敗: {e}")

if __name__ == '__main__':
    main()