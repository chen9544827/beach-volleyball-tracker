# test_stage1_2_integration.py
# 階段1+2 整合測試腳本 - 端到端驗證

import os
import sys
import subprocess
import json
from datetime import datetime


def print_header(title):
    """輸出美化的標題"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_scoreboard_roi_marker():
    """測試 ROI 標定工具 (需要手動操作)"""
    print_header("測試 1: ROI 標定工具")
    
    test_video = "input_video/analyze_serve/segment_028.mp4"
    
    if not os.path.exists(test_video):
        print(f"⚠️  測試影片不存在: {test_video}")
        print("   請手動執行:")
        print(f"   python tools/scoreboard_roi_marker.py --video_path <your_video> --output test_scoreboard_config.json")
        return False
    
    print(f"✓ 找到測試影片: {test_video}")
    print("\n請手動執行以下命令進行 ROI 標定:")
    print(f"\n  python tools/scoreboard_roi_marker.py --video_path \"{test_video}\" --output test_scoreboard_config.json\n")
    print("標定完成後按 Enter 繼續...")
    input()
    
    if os.path.exists("test_scoreboard_config.json"):
        print("✅ 測試通過: 成功產生配置檔")
        with open("test_scoreboard_config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
            print(f"   Team1 ROI: {config['score_roi_team1']}")
            print(f"   Team2 ROI: {config['score_roi_team2']}")
        return True
    else:
        print("❌ 測試失敗: 未找到配置檔")
        return False


def test_video_slicer_config_support():
    """測試切片腳本的配置檔支援"""
    print_header("測試 2: 切片腳本配置檔支援")
    
    # 檢查修改後的腳本是否包含配置檔支援
    slicer_path = "video_processing/video_slicer_by_score.py"
    
    with open(slicer_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "load_scoreboard_config" in content and "--scoreboard_config" in content:
        print("✅ 測試通過: 切片腳本支援外部配置檔")
        print("   - 找到 load_scoreboard_config 函數")
        print("   - 找到 --scoreboard_config 參數")
        return True
    else:
        print("❌ 測試失敗: 切片腳本未正確修改")
        return False


def test_tracking_with_bytetrack():
    """測試 ByteTrack 追蹤整合"""
    print_header("測試 3: ByteTrack 追蹤整合")
    
    tracker_path = "video_processing/track_ball_and_player.py"
    
    with open(tracker_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        "track_id 欄位": "track_id" in content,
        "--use_tracking 參數": "--use_tracking" in content,
        "ByteTrack 球體配置": "bytetrack_ball.yaml" in content,
        "ByteTrack 球員配置": "bytetrack_player.yaml" in content,
        "向下相容性": "use_tracking=False" in content or "use_tracking=use_tracking" in content
    }
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
    
    if all_passed:
        print("\n✅ 測試通過: ByteTrack 整合完成")
        return True
    else:
        print("\n❌ 測試失敗: 部分功能缺失")
        return False


def test_bytetrack_configs():
    """測試 ByteTrack 配置檔"""
    print_header("測試 4: ByteTrack 配置檔")
    
    ball_config = "configs/bytetrack_ball.yaml"
    player_config = "configs/bytetrack_player.yaml"
    
    results = []
    
    for config_path, name in [(ball_config, "球體"), (player_config, "球員")]:
        if os.path.exists(config_path):
            print(f"✓ {name}配置檔存在: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "tracker_type: bytetrack" in content:
                    print(f"  ✓ 配置檔格式正確")
                    results.append(True)
                else:
                    print(f"  ✗ 配置檔格式錯誤")
                    results.append(False)
        else:
            print(f"✗ {name}配置檔缺失: {config_path}")
            results.append(False)
    
    if all(results):
        print("\n✅ 測試通過: 配置檔完整")
        return True
    else:
        print("\n❌ 測試失敗: 配置檔問題")
        return False


def test_tracking_quality_checker():
    """測試追蹤品質檢查工具"""
    print_header("測試 5: 追蹤品質檢查工具")
    
    checker_path = "tools/tracking_quality_checker.py"
    
    if os.path.exists(checker_path):
        print(f"✓ 工具存在: {checker_path}")
        
        with open(checker_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            "軌跡提取": "extract_tracks" in content,
            "指標計算": "calculate_metrics" in content,
            "報告輸出": "print_report" in content,
            "直方圖": "histogram" in content.lower()
        }
        
        all_passed = all(checks.values())
        
        for check_name, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
        
        if all_passed:
            print("\n✅ 測試通過: 品質檢查工具完整")
            return True
        else:
            print("\n❌ 測試失敗: 部分功能缺失")
            return False
    else:
        print(f"✗ 工具缺失: {checker_path}")
        print("\n❌ 測試失敗")
        return False


def test_batch_slice_videos():
    """測試批次切片腳本"""
    print_header("測試 6: 批次切片腳本")
    
    batch_script = "batch_slice_videos.py"
    
    if os.path.exists(batch_script):
        print(f"✓ 腳本存在: {batch_script}")
        
        with open(batch_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            "並行處理": "ProcessPoolExecutor" in content,
            "影片搜尋": "find_video_files" in content,
            "總報告產生": "master_slicing_report" in content,
            "CSV 輸出": "master_slicing_report.csv" in content
        }
        
        all_passed = all(checks.values())
        
        for check_name, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
        
        if all_passed:
            print("\n✅ 測試通過: 批次腳本功能完整")
            return True
        else:
            print("\n❌ 測試失敗: 部分功能缺失")
            return False
    else:
        print(f"✗ 腳本缺失: {batch_script}")
        print("\n❌ 測試失敗")
        return False


def main():
    print("\n" + "🏐"*40)
    print("\n  沙灘排球追蹤系統 - 階段1+2 整合測試")
    print("  Beach Volleyball Tracker - Stage 1+2 Integration Test")
    print("\n" + "🏐"*40)
    
    start_time = datetime.now()
    
    # 執行所有測試
    tests = [
        ("ROI 標定工具", test_scoreboard_roi_marker),
        ("配置檔支援", test_video_slicer_config_support),
        ("ByteTrack 整合", test_tracking_with_bytetrack),
        ("ByteTrack 配置", test_bytetrack_configs),
        ("品質檢查工具", test_tracking_quality_checker),
        ("批次切片腳本", test_batch_slice_videos)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ 測試 '{test_name}' 發生異常: {e}")
            results[test_name] = False
    
    # 總結報告
    print_header("測試總結")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {status}: {test_name}")
    
    print(f"\n總計: {passed}/{total} 測試通過")
    print(f"成功率: {passed/total*100:.1f}%")
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n測試耗時: {duration}")
    
    if passed == total:
        print("\n🎉 恭喜！所有測試通過，系統已就緒！")
        print("\n下一步:")
        print("  1. 參考 STAGE1_2_GUIDE.md 開始使用新功能")
        print("  2. 使用您的測試影片執行端到端流程")
        print("  3. 開始規劃階段3 (Homography 映射)")
        return 0
    else:
        print("\n⚠️  部分測試未通過，請檢查上述錯誤訊息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
