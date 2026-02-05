"""
測試 ROI 配置生成器的文字顯示位置
驗證場地名稱和提示文字是否正確顯示在右上角
"""
import cv2
import numpy as np
import os


def create_test_frame():
    """創建測試用的影片幀"""
    # 創建 1280x720 的測試圖像
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # 填充深灰色背景
    frame[:] = (50, 50, 50)

    # 在左上角繪製紅色區域標示（模擬分數顯示區域）
    cv2.rectangle(frame, (200, 20), (400, 120), (0, 0, 200), 2)
    cv2.putText(frame, "Score Area (Should NOT be covered)", (210, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 在右上角繪製綠色區域標示（應該放置文字的位置）
    cv2.rectangle(frame, (880, 20), (1260, 120), (0, 200, 0), 2)
    cv2.putText(frame, "Text Area (Text should be here)", (900, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame


def test_text_placement():
    """測試文字放置位置"""
    frame = create_test_frame()
    frame_height, frame_width = frame.shape[:2]

    # 測試 1: 場地名稱顯示（模擬 roi_config_generator_v2.py）
    print("\n測試 1: 場地名稱顯示位置")
    venue_text = "Venue: Edmonton"
    text_size = cv2.getTextSize(venue_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = frame_width - text_size[0] - 20
    text_y = 35

    print(f"  文字位置: x={text_x}, y={text_y}")
    print(f"  文字寬度: {text_size[0]} pixels")
    print(f"  文字高度: {text_size[1]} pixels")

    # 繪製場地名稱
    test_frame_1 = frame.copy()
    cv2.rectangle(test_frame_1, (text_x - 10, 10), (frame_width - 10, 50), (0, 0, 0), -1)
    cv2.putText(test_frame_1, venue_text, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 測試 2: 提示文字顯示（模擬 roi_config_generator.py）
    print("\n測試 2: 提示文字顯示位置")
    prompt = "正在設定: Team1 (綠色)"
    text_size_2 = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x_2 = frame_width - text_size_2[0] - 10
    text_y_2 = 30

    print(f"  文字位置: x={text_x_2}, y={text_y_2}")
    print(f"  文字寬度: {text_size_2[0]} pixels")
    print(f"  文字高度: {text_size_2[1]} pixels")

    test_frame_2 = frame.copy()
    overlay = test_frame_2.copy()
    cv2.rectangle(overlay, (text_x_2 - 5, text_y_2 - 25),
                 (frame_width - 5, text_y_2 + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, test_frame_2, 0.3, 0, test_frame_2)
    cv2.putText(test_frame_2, prompt, (text_x_2, text_y_2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 測試 3: 座標資訊顯示（模擬 preview_roi.py）
    print("\n測試 3: 座標資訊顯示位置")
    info_text = [
        "Team1 ROI: x=280, y=29, w=59, h=51",
        "Team2 ROI: x=287, y=92, w=59, h=50",
        "Green = Team1, Red = Team2"
    ]

    max_width = 0
    for text in info_text:
        text_size_3 = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        max_width = max(max_width, text_size_3[0])

    print(f"  最大文字寬度: {max_width} pixels")

    test_frame_3 = frame.copy()
    overlay = test_frame_3.copy()
    cv2.rectangle(overlay, (frame_width - max_width - 20, 10),
                 (frame_width - 10, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, test_frame_3, 0.3, 0, test_frame_3)

    y_offset = 30
    for i, text in enumerate(info_text):
        text_size_4 = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x_3 = frame_width - text_size_4[0] - 15
        text_y_3 = y_offset + i * 30
        cv2.putText(test_frame_3, text, (text_x_3, text_y_3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 儲存測試結果
    output_dir = "test_output/roi_display_position"
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(f"{output_dir}/test_venue_name.jpg", test_frame_1)
    cv2.imwrite(f"{output_dir}/test_prompt.jpg", test_frame_2)
    cv2.imwrite(f"{output_dir}/test_coordinates.jpg", test_frame_3)

    print(f"\n[OK] 測試圖片已儲存至: {output_dir}/")
    print("  - test_venue_name.jpg (場地名稱顯示)")
    print("  - test_prompt.jpg (提示文字顯示)")
    print("  - test_coordinates.jpg (座標資訊顯示)")

    # 驗證位置
    print("\n驗證結果:")

    # 檢查是否在右側
    if text_x > frame_width / 2:
        print("  [OK] 場地名稱在右側")
    else:
        print("  [ERROR] 場地名稱在左側（錯誤）")

    if text_x_2 > frame_width / 2:
        print("  [OK] 提示文字在右側")
    else:
        print("  [ERROR] 提示文字在左側（錯誤）")

    if (frame_width - max_width - 20) > frame_width / 2:
        print("  [OK] 座標資訊在右側")
    else:
        print("  [ERROR] 座標資訊在左側（錯誤）")

    # 檢查是否會遮擋分數區域（假設分數區域在左上角 200-400 範圍）
    score_area_right = 400
    if text_x > score_area_right + 50:
        print("  [OK] 文字不會遮擋分數區域")
    else:
        print("  [ERROR] 文字可能遮擋分數區域")

    print("\n請檢查 test_output/roi_display_position/ 目錄中的圖片")
    print("確認文字是否正確顯示在右上角（綠色區域內）\n")


if __name__ == "__main__":
    print("="*60)
    print("ROI 配置生成器文字顯示位置測試")
    print("="*60)
    test_text_placement()
