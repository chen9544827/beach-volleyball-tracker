"""
互動式 ROI 配置生成器 v2 - 支援場地模板
允許創建場地模板並重複使用，避免重複設定相同場地
"""
import cv2
import json
import argparse
import os
import glob


class VenueTemplateManager:
    """場地模板管理器"""

    def __init__(self, venues_dir="roi_configs/venues"):
        self.venues_dir = venues_dir
        os.makedirs(self.venues_dir, exist_ok=True)
        self.venues = self.load_venues()

    def load_venues(self):
        """載入所有場地模板"""
        venues = {}
        venue_files = glob.glob(os.path.join(self.venues_dir, "*.json"))

        for venue_file in venue_files:
            try:
                with open(venue_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    venue_name = config.get('venue_name', os.path.basename(venue_file).replace('.json', ''))
                    venues[venue_name] = config
            except Exception as e:
                print(f"[WARNING] 無法載入場地模板 {venue_file}: {e}")

        return venues

    def get_venue_list(self):
        """獲取場地列表"""
        return list(self.venues.keys())

    def get_venue_config(self, venue_name):
        """獲取場地配置"""
        return self.venues.get(venue_name)

    def save_venue_template(self, venue_name, roi_team1, roi_team2, resolution, notes=""):
        """儲存場地模板"""
        config = {
            "venue_name": venue_name,
            "video_resolution": resolution,
            "score_roi_team1": {
                "x": roi_team1[0],
                "y": roi_team1[1],
                "width": roi_team1[2],
                "height": roi_team1[3],
                "label": "Team1 Score ROI"
            },
            "score_roi_team2": {
                "x": roi_team2[0],
                "y": roi_team2[1],
                "width": roi_team2[2],
                "height": roi_team2[3],
                "label": "Team2 Score ROI"
            },
            "notes": notes
        }

        venue_file = os.path.join(self.venues_dir, f"{venue_name}.json")
        with open(venue_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"[OK] 場地模板已儲存: {venue_file}")
        self.venues[venue_name] = config


class ROIConfigGeneratorV2:
    """互動式 ROI 配置生成器 v2"""

    def __init__(self, video_path, output_path, venue_manager):
        self.video_path = video_path
        self.output_path = output_path
        self.venue_manager = venue_manager

        self.current_venue = None
        self.rois = {'team1': None, 'team2': None}
        self.current_team = 'team1'
        self.drawing = False
        self.start_point = None
        self.current_rect = None

        self.frame = None
        self.frame_width = 0
        self.frame_height = 0

        # 選擇模式
        self.mode = 'select'  # 'select' or 'draw'

    def load_video_frame(self):
        """載入影片第一幀"""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] 無法讀取影片第一幀")
            cap.release()
            return False

        self.frame = frame.copy()
        self.frame_height, self.frame_width = frame.shape[:2]
        cap.release()
        return True

    def display_venue_selection(self):
        """顯示場地選擇介面"""
        venues = self.venue_manager.get_venue_list()

        print("\n" + "="*60)
        print("場地模板選擇")
        print("="*60)

        if venues:
            print("\n已存在的場地模板：")
            for i, venue in enumerate(venues, 1):
                print(f"  [{i}] {venue}")
            print(f"\n  [n] 創建新場地模板")
            print(f"  [q] 退出")

            while True:
                choice = input("\n請選擇場地或創建新場地 (1-{}/n/q): ".format(len(venues)))

                if choice.lower() == 'q':
                    return None
                elif choice.lower() == 'n':
                    return self.create_new_venue()
                elif choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(venues):
                        return venues[idx]
                    else:
                        print("[ERROR] 無效的選擇")
                else:
                    print("[ERROR] 無效的輸入")
        else:
            print("\n目前沒有場地模板")
            print("  [n] 創建新場地模板")
            print("  [q] 退出")

            while True:
                choice = input("\n請選擇操作 (n/q): ")
                if choice.lower() == 'q':
                    return None
                elif choice.lower() == 'n':
                    return self.create_new_venue()
                else:
                    print("[ERROR] 無效的輸入")

    def create_new_venue(self):
        """創建新場地模板"""
        print("\n--- 創建新場地模板 ---")
        venue_name = input("請輸入場地名稱（例如：Edmonton, Gstaad）: ").strip()

        if not venue_name:
            print("[ERROR] 場地名稱不能為空")
            return None

        # 檢查是否已存在
        if venue_name in self.venue_manager.venues:
            overwrite = input(f"[WARNING] 場地 '{venue_name}' 已存在，是否覆蓋？(y/n): ")
            if overwrite.lower() != 'y':
                return None

        return venue_name

    def load_venue_rois(self, venue_name):
        """載入場地的 ROI 配置"""
        config = self.venue_manager.get_venue_config(venue_name)
        if config:
            roi1 = config['score_roi_team1']
            roi2 = config['score_roi_team2']

            self.rois['team1'] = (roi1['x'], roi1['y'], roi1['width'], roi1['height'])
            self.rois['team2'] = (roi2['x'], roi2['y'], roi2['width'], roi2['height'])

            print(f"[OK] 已載入場地 '{venue_name}' 的 ROI 配置")
            print(f"     Team1 ROI: {self.rois['team1']}")
            print(f"     Team2 ROI: {self.rois['team2']}")
            return True
        return False

    def mouse_callback(self, event, x, y, flags, param):
        """處理滑鼠事件 - 拖曳框選 ROI"""
        if self.mode != 'draw':
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_rect = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_rect = (self.start_point[0], self.start_point[1], x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_point = (x, y)

            x1 = min(self.start_point[0], end_point[0])
            y1 = min(self.start_point[1], end_point[1])
            w = abs(end_point[0] - self.start_point[0])
            h = abs(end_point[1] - self.start_point[1])

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
                print("\n按 's' 儲存配置，'r' 重置，'e' 編輯，'q' 退出")

            self.current_rect = None

    def draw_display(self):
        """繪製顯示畫面"""
        display = self.frame.copy()

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

        # 顯示場地名稱（右上角）
        if self.current_venue:
            venue_text = f"Venue: {self.current_venue}"
            text_size = cv2.getTextSize(venue_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = self.frame_width - text_size[0] - 20
            text_y = 35

            # 繪製背景（黑色）
            cv2.rectangle(display, (text_x - 10, 10), (self.frame_width - 10, 50), (0, 0, 0), -1)
            # 繪製文字
            cv2.putText(display, venue_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 顯示模式提示
        if self.mode == 'select':
            prompt = "按 'e' 進入編輯模式"
            color = (0, 255, 255)
        else:
            if not self.rois['team2']:
                prompt = f"正在設定: {'Team1 (綠色)' if self.current_team == 'team1' else 'Team2 (紅色)'}"
                color = (255, 255, 255)
            else:
                prompt = "按 's' 儲存配置"
                color = (0, 255, 0)

        cv2.putText(display, prompt, (10, self.frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return display

    def run(self):
        """執行互動式配置流程"""
        # 載入影片
        if not self.load_video_frame():
            return False

        # 選擇或創建場地
        venue_name = self.display_venue_selection()
        if venue_name is None:
            print("[INFO] 取消配置")
            return False

        self.current_venue = venue_name

        # 載入場地 ROI（如果已存在）
        venue_exists = venue_name in self.venue_manager.venues
        if venue_exists:
            self.load_venue_rois(venue_name)
            self.mode = 'select'
        else:
            self.mode = 'draw'

        # 互動式編輯
        print("\n=== ROI 配置生成器 v2 ===")
        print(f"場地: {venue_name}")
        print(f"影片: {os.path.basename(self.video_path)}")
        print(f"解析度: {self.frame_width}x{self.frame_height}")
        print("\n操作說明:")
        if venue_exists:
            print("  's' - 儲存配置")
            print("  'e' - 進入編輯模式（重新設定 ROI）")
        else:
            print("  拖曳滑鼠框選 Team1 分數區域（綠色框）")
            print("  拖曳滑鼠框選 Team2 分數區域（紅色框）")
            print("  's' - 儲存配置")
        print("  'r' - 重置")
        print("  'q' - 退出\n")

        window_name = f'ROI Config - {venue_name}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while True:
            display = self.draw_display()
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                if self.rois['team1'] and self.rois['team2']:
                    self.save_config()
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("[WARNING] 請先設定兩個 ROI")

            elif key == ord('e'):
                if self.mode == 'select':
                    self.mode = 'draw'
                    self.current_team = 'team1'
                    print("[INFO] 進入編輯模式")
                    print("  步驟 1: 拖曳框選 Team1 分數區域（綠色框）")

            elif key == ord('r'):
                self.rois = {'team1': None, 'team2': None}
                self.current_team = 'team1'
                self.mode = 'draw'
                print("[INFO] 已重置，重新開始")

            elif key == ord('q'):
                print("[INFO] 取消配置")
                cv2.destroyAllWindows()
                return False

    def save_config(self):
        """儲存配置"""
        # 儲存場地模板
        resolution = {"width": self.frame_width, "height": self.frame_height}
        self.venue_manager.save_venue_template(
            self.current_venue,
            self.rois['team1'],
            self.rois['team2'],
            resolution
        )

        # 儲存影片配置（連結到場地）
        video_name = os.path.basename(self.video_path)
        config = {
            "video_name": video_name,
            "venue": self.current_venue,
            "video_resolution": resolution,
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
            "notes": f"使用場地模板: {self.current_venue}"
        }

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"[OK] 影片配置已儲存: {self.output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="互動式 ROI 配置生成器 v2 - 支援場地模板",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python roi_config_generator_v2.py \\
      --video input.mp4 \\
      --output roi_configs/videos/input_roi_config.json

場地模板:
  場地模板儲存在 roi_configs/venues/ 目錄
  可以為多個影片重複使用相同的場地配置
        """
    )
    parser.add_argument("--video", type=str, required=True,
                        help="輸入影片路徑")
    parser.add_argument("--output", type=str, required=True,
                        help="輸出 JSON 路徑")
    parser.add_argument("--venues-dir", type=str, default="roi_configs/venues",
                        help="場地模板目錄（預設: roi_configs/venues）")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] 影片檔案不存在: {args.video}")
        return

    venue_manager = VenueTemplateManager(args.venues_dir)
    generator = ROIConfigGeneratorV2(args.video, args.output, venue_manager)
    success = generator.run()

    if success:
        print("\n[OK] 配置完成！")
    else:
        print("\n[INFO] 配置已取消")


if __name__ == "__main__":
    main()
