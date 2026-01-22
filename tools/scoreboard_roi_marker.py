# tools/scoreboard_roi_marker.py
# 互動式記分板 ROI 標定工具 - 用於產生 scoreboard_config.json

import cv2
import json
import argparse
import os
import sys

class ScoreboardROIMarker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"無法開啟影片: {video_path}")
        
        # 讀取第一幀
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError(f"無法讀取影片幀: {video_path}")
        
        self.display_frame = self.frame.copy()
        self.cap.release()
        
        # ROI 點位儲存 (每個 ROI 需要兩個點: 左上、右下)
        self.team1_points = []  # [(x1, y1), (x2, y2)]
        self.team2_points = []  # [(x1, y1), (x2, y2)]
        
        self.current_team = 1  # 目前標記的隊伍 (1 或 2)
        self.window_name = "記分板 ROI 標定工具"
        
    def mouse_callback(self, event, x, y, flags, param):
        """滑鼠事件處理"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_team == 1:
                self.team1_points.append((x, y))
                cv2.circle(self.display_frame, (x, y), 5, (0, 255, 0), -1)
                
                if len(self.team1_points) == 1:
                    print(f"✓ Team1 左上角: ({x}, {y})")
                    print("  請點擊 Team1 記分板的右下角...")
                elif len(self.team1_points) == 2:
                    print(f"✓ Team1 右下角: ({x}, {y})")
                    self._draw_roi(self.team1_points, (0, 255, 0), "Team1")
                    print("\n" + "="*50)
                    print("Team1 標記完成！")
                    print("請點擊 Team2 記分板的左上角...")
                    print("="*50)
                    self.current_team = 2
                    
            elif self.current_team == 2:
                self.team2_points.append((x, y))
                cv2.circle(self.display_frame, (x, y), 5, (0, 0, 255), -1)
                
                if len(self.team2_points) == 1:
                    print(f"✓ Team2 左上角: ({x}, {y})")
                    print("  請點擊 Team2 記分板的右下角...")
                elif len(self.team2_points) == 2:
                    print(f"✓ Team2 右下角: ({x}, {y})")
                    self._draw_roi(self.team2_points, (0, 0, 255), "Team2")
                    print("\n" + "="*50)
                    print("Team2 標記完成！")
                    print("按 's' 儲存配置，按 'r' 重新標記，按 'q' 退出")
                    print("="*50)
                    
            cv2.imshow(self.window_name, self.display_frame)
    
    def _draw_roi(self, points, color, label):
        """繪製 ROI 矩形"""
        if len(points) == 2:
            pt1, pt2 = points
            cv2.rectangle(self.display_frame, pt1, pt2, color, 2)
            cv2.putText(self.display_frame, label, 
                       (pt1[0], pt1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _calculate_roi_coords(self, points):
        """計算 ROI 座標 (x, y, w, h)"""
        if len(points) != 2:
            return None
        
        pt1, pt2 = points
        x = min(pt1[0], pt2[0])
        y = min(pt1[1], pt2[1])
        w = abs(pt2[0] - pt1[0])
        h = abs(pt2[1] - pt1[1])
        
        return (x, y, w, h)
    
    def _save_config(self, output_path):
        """儲存配置到 JSON 檔案"""
        team1_roi = self._calculate_roi_coords(self.team1_points)
        team2_roi = self._calculate_roi_coords(self.team2_points)
        
        if team1_roi is None or team2_roi is None:
            print("錯誤: ROI 座標不完整，無法儲存。")
            return False
        
        config = {
            "source_video": os.path.basename(self.video_path),
            "frame_width": self.frame.shape[1],
            "frame_height": self.frame.shape[0],
            "score_roi_team1": {
                "x": team1_roi[0],
                "y": team1_roi[1],
                "w": team1_roi[2],
                "h": team1_roi[3],
                "description": "Team1 記分板區域 (通常為上方或左方)"
            },
            "score_roi_team2": {
                "x": team2_roi[0],
                "y": team2_roi[1],
                "w": team2_roi[2],
                "h": team2_roi[3],
                "description": "Team2 記分板區域 (通常為下方或右方)"
            },
            "recommended_diff_threshold": 9000,
            "notes": "此配置由 scoreboard_roi_marker.py 自動產生"
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 配置已成功儲存至: {os.path.abspath(output_path)}")
            print(f"\nTeam1 ROI: (x={team1_roi[0]}, y={team1_roi[1]}, w={team1_roi[2]}, h={team1_roi[3]})")
            print(f"Team2 ROI: (x={team2_roi[0]}, y={team2_roi[1]}, w={team2_roi[2]}, h={team2_roi[3]})")
            return True
        except Exception as e:
            print(f"❌ 儲存配置失敗: {e}")
            return False
    
    def _reset(self):
        """重置所有標記"""
        self.team1_points = []
        self.team2_points = []
        self.current_team = 1
        self.display_frame = self.frame.copy()
        cv2.imshow(self.window_name, self.display_frame)
        print("\n已重置標記，請重新開始...")
        print("請點擊 Team1 記分板的左上角...")
    
    def run(self, output_path="scoreboard_config.json"):
        """執行標定流程"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n" + "="*60)
        print("    記分板 ROI 標定工具")
        print("="*60)
        print("\n使用說明:")
        print("  1. 點擊 Team1 記分板的左上角")
        print("  2. 點擊 Team1 記分板的右下角")
        print("  3. 點擊 Team2 記分板的左上角")
        print("  4. 點擊 Team2 記分板的右下角")
        print("\n快捷鍵:")
        print("  's' - 儲存配置")
        print("  'r' - 重新標記")
        print("  'q' - 退出")
        print("="*60)
        print("\n請點擊 Team1 記分板的左上角...")
        
        cv2.imshow(self.window_name, self.display_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n使用者取消操作。")
                break
            elif key == ord('s'):
                if len(self.team1_points) == 2 and len(self.team2_points) == 2:
                    if self._save_config(output_path):
                        break
                else:
                    print("\n⚠️  請先完成所有 ROI 的標記！")
            elif key == ord('r'):
                self._reset()
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="互動式記分板 ROI 標定工具 - 產生 scoreboard_config.json"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="參考影片的路徑 (將從第一幀擷取畫面)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scoreboard_config.json",
        help="輸出的配置檔名 (預設: scoreboard_config.json)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ 錯誤: 找不到影片檔案 '{args.video_path}'")
        sys.exit(1)
    
    try:
        marker = ScoreboardROIMarker(args.video_path)
        marker.run(args.output)
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
