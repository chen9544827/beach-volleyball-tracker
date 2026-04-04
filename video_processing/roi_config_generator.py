"""
互動式 ROI 配置生成器
基於 court_config_generator.py 的設計模式
允許用戶透過滑鼠拖曳框選分數顯示區域
"""
import cv2
import json
import argparse
import os


class ROIConfigGenerator:
    """互動式 ROI 配置生成器"""

    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.rois = {'team1': None, 'team2': None}
        self.current_team = 'team1'
        self.drawing = False
        self.start_point = None
        self.current_rect = None

    def mouse_callback(self, event, x, y, flags, param):
        """處理滑鼠事件 - 拖曳框選 ROI"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_rect = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # 實時顯示拖曳框
                self.current_rect = (self.start_point[0], self.start_point[1], x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_point = (x, y)

            # 計算 ROI (x, y, w, h)
            x1 = min(self.start_point[0], end_point[0])
            y1 = min(self.start_point[1], end_point[1])
            w = abs(end_point[0] - self.start_point[0])
            h = abs(end_point[1] - self.start_point[1])

            # 驗證 ROI 大小（至少 10x10 像素）
            if w < 10 or h < 10:
                print("[WARNING] ROI 太小，請重新框選")
                self.current_rect = None
                return

            if self.current_team == 'team1':
                self.rois['team1'] = (x1, y1, w, h)
                print(f"[OK] Team1 ROI 已設定: x={x1}, y={y1}, w={w}, h={h}")
                self.current_team = 'team2'
            else:
                self.rois['team2'] = (x1, y1, w, h)
                print(f"[OK] Team2 ROI 已設定: x={x1}, y={y1}, w={w}, h={h}")
                print("\n按 's' 儲存配置，'r' 重置，'q' 退出")

            self.current_rect = None

    def run(self):
        """執行互動式配置流程"""
        # 1. 讀取影片第一幀
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] 無法讀取影片第一幀")
            cap.release()
            return False

        frame_height, frame_width = frame.shape[:2]
        cap.release()

        # 2. 互動式框選
        print("\n=== ROI 配置生成器 ===")
        print(f"影片: {os.path.basename(self.video_path)}")
        print(f"解析度: {frame_width}x{frame_height}")
        print("\n操作說明:")
        print("  步驟 1: 拖曳框選 Team1 分數區域（綠色框）")
        print("  步驟 2: 拖曳框選 Team2 分數區域（紅色框）")
        print("  按 's' 儲存，'r' 重置，'q' 退出\n")

        clone = frame.copy()
        window_name = 'ROI Configuration - ' + os.path.basename(self.video_path)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while True:
            display = clone.copy()

            # 繪製已設定的 ROI
            if self.rois['team1']:
                x, y, w, h = self.rois['team1']
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display, 'Team1', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if self.rois['team2']:
                x, y, w, h = self.rois['team2']
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(display, 'Team2', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 繪製正在拖曳的矩形
            if self.drawing and self.current_rect:
                x1, y1, x2, y2 = self.current_rect
                color = (0, 255, 0) if self.current_team == 'team1' else (0, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # 顯示當前步驟提示（右上角）
            if not self.rois['team2']:
                prompt = f"正在設定: {'Team1 (綠色)' if self.current_team == 'team1' else 'Team2 (紅色)'}"
            else:
                prompt = "按 's' 儲存配置"

            # 計算文字寬度並放置在右上角
            text_size = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = frame_width - text_size[0] - 10
            text_y = 30

            # 繪製背景（黑色半透明）
            overlay = display.copy()
            cv2.rectangle(overlay, (text_x - 5, text_y - 25),
                         (frame_width - 5, text_y + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

            # 繪製文字
            text_color = (0, 255, 0) if self.rois['team2'] else (255, 255, 255)
            cv2.putText(display, prompt, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if self.rois['team1'] and self.rois['team2']:
                    self.save_config(frame_width, frame_height)
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("[WARNING] 請先設定兩個 ROI")

            elif key == ord('r'):
                self.rois = {'team1': None, 'team2': None}
                self.current_team = 'team1'
                self.current_rect = None
                print("[INFO] 已重置，重新開始")
                print("  步驟 1: 拖曳框選 Team1 分數區域（綠色框）")

            elif key == ord('q'):
                print("[INFO] 取消配置")
                cv2.destroyAllWindows()
                return False

    def save_config(self, width, height):
        """儲存 ROI 配置為 JSON"""
        video_name = os.path.basename(self.video_path)

        config = {
            "video_name": video_name,
            "video_resolution": {
                "width": width,
                "height": height
            },
            "score_roi_team1": {
                "x": self.rois['team1'][0],
                "y": self.rois['team1'][1],
                "width": self.rois['team1'][2],
                "height": self.rois['team1'][3],
                "label": "Team1 Score ROI"
            },
            "score_roi_team2": {
                "x": self.rois['team2'][0],
                "y": self.rois['team2'][1],
                "width": self.rois['team2'][2],
                "height": self.rois['team2'][3],
                "label": "Team2 Score ROI"
            },
            "notes": ""
        }

        # 確保輸出目錄存在
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] ROI 配置已儲存: {self.output_path}")
        print("\n配置內容:")
        print(f"  Team1 ROI: x={config['score_roi_team1']['x']}, "
              f"y={config['score_roi_team1']['y']}, "
              f"w={config['score_roi_team1']['width']}, "
              f"h={config['score_roi_team1']['height']}")
        print(f"  Team2 ROI: x={config['score_roi_team2']['x']}, "
              f"y={config['score_roi_team2']['y']}, "
              f"w={config['score_roi_team2']['width']}, "
              f"h={config['score_roi_team2']['height']}")


def main():
    parser = argparse.ArgumentParser(
        description="互動式 ROI 配置生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python roi_config_generator.py --video input.mp4 --output roi_config.json

操作步驟:
  1. 拖曳滑鼠框選 Team1 分數區域（綠色框）
  2. 拖曳滑鼠框選 Team2 分數區域（紅色框）
  3. 按 's' 儲存配置
  4. 按 'r' 重置重新開始
  5. 按 'q' 退出
        """
    )
    parser.add_argument("--video", type=str, required=True,
                        help="輸入影片路徑")
    parser.add_argument("--output", type=str, required=True,
                        help="輸出 JSON 路徑")
    args = parser.parse_args()

    # 驗證影片檔案存在
    if not os.path.exists(args.video):
        print(f"[ERROR] 影片檔案不存在: {args.video}")
        return

    generator = ROIConfigGenerator(args.video, args.output)
    success = generator.run()

    if success:
        print("\n[OK] 配置完成！")
    else:
        print("\n[INFO] 配置已取消")


if __name__ == "__main__":
    main()
