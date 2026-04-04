"""
提取影片第一幀並標記 ROI 區域，用於確認分數顯示位置
"""
import cv2
import os
import argparse
import json

# 預設的 ROI 座標（從 video_slicer_by_score.py 複製）
SCORE_ROI_TEAM1 = (280, 29, 59, 51)  # Team1 分數區域 (x, y, w, h)
SCORE_ROI_TEAM2 = (287, 92, 59, 50)  # Team2 分數區域 (x, y, w, h)


def load_roi_config(config_path):
    """載入 ROI 配置檔案

    Args:
        config_path: ROI 配置 JSON 檔案路徑

    Returns:
        tuple: (team1_roi, team2_roi) 若成功，否則 None
    """
    if not config_path or not os.path.exists(config_path):
        if config_path:
            print(f"[WARNING] ROI 配置檔案不存在: {config_path}")
        print(f"[INFO] 使用預設 ROI 座標")
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 驗證必要欄位
        if 'score_roi_team1' not in config or 'score_roi_team2' not in config:
            print(f"[ERROR] ROI 配置格式錯誤: 缺少必要欄位")
            return None

        # 轉換為 tuple 格式 (x, y, w, h)
        roi1 = config['score_roi_team1']
        roi2 = config['score_roi_team2']

        team1_roi = (roi1['x'], roi1['y'], roi1['width'], roi1['height'])
        team2_roi = (roi2['x'], roi2['y'], roi2['width'], roi2['height'])

        print(f"[OK] 已載入 ROI 配置: {config_path}")
        print(f"     Team1 ROI: {team1_roi}")
        print(f"     Team2 ROI: {team2_roi}")

        return team1_roi, team2_roi

    except Exception as e:
        print(f"[ERROR] 載入 ROI 配置失敗: {e}")
        return None


def preview_roi(video_path, output_path, roi_config_path=None):
    """提取影片第一幀並標記 ROI"""
    # 載入 ROI 配置（若指定）
    roi_team1 = SCORE_ROI_TEAM1
    roi_team2 = SCORE_ROI_TEAM2

    if roi_config_path:
        roi_result = load_roi_config(roi_config_path)
        if roi_result:
            roi_team1, roi_team2 = roi_result
        # 若載入失敗，則繼續使用預設值

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤: 無法打開影片 {video_path}")
        return False

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("錯誤: 無法讀取影片第一幀")
        return False

    # 標記 Team1 ROI (綠色)
    x1, y1, w1, h1 = roi_team1
    cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 3)
    cv2.putText(frame, 'Team1 Score ROI', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 標記 Team2 ROI (紅色)
    x2, y2, w2, h2 = roi_team2
    cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 3)
    cv2.putText(frame, 'Team2 Score ROI', (x2, y2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 在畫面上顯示座標資訊（右上角）
    info_text = [
        f"Team1 ROI: x={x1}, y={y1}, w={w1}, h={h1}",
        f"Team2 ROI: x={x2}, y={y2}, w={w2}, h={h2}",
        "Green = Team1, Red = Team2"
    ]

    frame_height, frame_width = frame.shape[:2]

    # 找出最長的文字寬度
    max_width = 0
    for text in info_text:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        max_width = max(max_width, text_size[0])

    # 繪製黑色背景（半透明）
    overlay = frame.copy()
    cv2.rectangle(overlay, (frame_width - max_width - 20, 10),
                 (frame_width - 10, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # 繪製文字（右上角）
    y_offset = 30
    for i, text in enumerate(info_text):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = frame_width - text_size[0] - 15
        text_y = y_offset + i * 30

        # 白色文字
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 儲存圖片
    cv2.imwrite(output_path, frame)
    print(f"[OK] ROI 預覽圖已儲存: {output_path}")
    print(f"     影片解析度: {frame.shape[1]}x{frame.shape[0]}")
    return True

def main():
    parser = argparse.ArgumentParser(description="提取影片第一幀並標記 ROI 區域")
    parser.add_argument("--input", type=str, required=True, help="輸入影片路徑")
    parser.add_argument("--output", type=str, default="roi_preview.jpg", help="輸出圖片路徑")
    parser.add_argument("--roi-config", type=str, default=None,
                        help="ROI 配置 JSON 檔案路徑（若不指定則使用預設值）")
    args = parser.parse_args()

    preview_roi(args.input, args.output, args.roi_config)

if __name__ == "__main__":
    main()
