"""
ROI 配置系統驗證測試
測試 ROI 配置的載入和驗證功能
"""
import json
import os
import sys
import tempfile

# 添加父目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_valid_roi_config():
    """測試有效的 ROI 配置載入"""
    print("\n[TEST 1] 測試有效的 ROI 配置載入...")

    # 創建臨時配置檔案
    config = {
        "video_name": "test_video.mp4",
        "video_resolution": {"width": 1280, "height": 720},
        "score_roi_team1": {"x": 280, "y": 29, "width": 59, "height": 51, "label": "Team1 Score ROI"},
        "score_roi_team2": {"x": 287, "y": 92, "width": 59, "height": 50, "label": "Team2 Score ROI"},
        "notes": "Test config"
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False)
        temp_path = f.name

    try:
        # 測試載入
        from video_processing.video_slicer_by_score import load_roi_config

        result = load_roi_config(temp_path)

        if result is None:
            print("  [FAIL] 載入失敗")
            return False

        roi1, roi2 = result

        # 驗證座標
        expected_roi1 = (280, 29, 59, 51)
        expected_roi2 = (287, 92, 59, 50)

        if roi1 == expected_roi1 and roi2 == expected_roi2:
            print("  [PASS] ROI 配置載入正確")
            print(f"    Team1 ROI: {roi1}")
            print(f"    Team2 ROI: {roi2}")
            return True
        else:
            print("  [FAIL] ROI 座標不正確")
            print(f"    預期 Team1: {expected_roi1}, 實際: {roi1}")
            print(f"    預期 Team2: {expected_roi2}, 實際: {roi2}")
            return False

    finally:
        os.unlink(temp_path)


def test_missing_file():
    """測試檔案不存在時的處理"""
    print("\n[TEST 2] 測試檔案不存在時的處理...")

    from video_processing.video_slicer_by_score import load_roi_config

    result = load_roi_config("nonexistent_file.json")

    if result is None:
        print("  [PASS] 正確處理不存在的檔案")
        return True
    else:
        print("  [FAIL] 應該返回 None")
        return False


def test_missing_fields():
    """測試缺少必要欄位時的處理"""
    print("\n[TEST 3] 測試缺少必要欄位時的處理...")

    # 創建不完整的配置檔案
    config = {
        "video_name": "test_video.mp4",
        "score_roi_team1": {"x": 280, "y": 29, "width": 59, "height": 51}
        # 缺少 score_roi_team2
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False)
        temp_path = f.name

    try:
        from video_processing.video_slicer_by_score import load_roi_config

        result = load_roi_config(temp_path)

        if result is None:
            print("  [PASS] 正確處理缺少欄位的配置")
            return True
        else:
            print("  [FAIL] 應該返回 None")
            return False

    finally:
        os.unlink(temp_path)


def test_invalid_json():
    """測試損壞的 JSON 處理"""
    print("\n[TEST 4] 測試損壞的 JSON 處理...")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        f.write("{ invalid json }")
        temp_path = f.name

    try:
        from video_processing.video_slicer_by_score import load_roi_config

        result = load_roi_config(temp_path)

        if result is None:
            print("  [PASS] 正確處理損壞的 JSON")
            return True
        else:
            print("  [FAIL] 應該返回 None")
            return False

    finally:
        os.unlink(temp_path)


def test_preview_roi_integration():
    """測試 preview_roi.py 整合"""
    print("\n[TEST 5] 測試 preview_roi.py 整合...")

    try:
        from video_processing.preview_roi import load_roi_config

        # 創建臨時配置檔案
        config = {
            "video_name": "test_video.mp4",
            "video_resolution": {"width": 1280, "height": 720},
            "score_roi_team1": {"x": 100, "y": 50, "width": 60, "height": 50, "label": "Team1 Score ROI"},
            "score_roi_team2": {"x": 200, "y": 100, "width": 60, "height": 50, "label": "Team2 Score ROI"},
            "notes": "Test"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False)
            temp_path = f.name

        try:
            result = load_roi_config(temp_path)

            if result is None:
                print("  [FAIL] 載入失敗")
                return False

            roi1, roi2 = result

            if roi1 == (100, 50, 60, 50) and roi2 == (200, 100, 60, 50):
                print("  [PASS] preview_roi.py 整合正常")
                return True
            else:
                print("  [FAIL] ROI 座標不正確")
                return False

        finally:
            os.unlink(temp_path)

    except Exception as e:
        print(f"  [FAIL] 發生錯誤: {e}")
        return False


def main():
    """執行所有測試"""
    print("="*60)
    print("ROI 配置系統驗證測試")
    print("="*60)

    tests = [
        test_valid_roi_config,
        test_missing_file,
        test_missing_fields,
        test_invalid_json,
        test_preview_roi_integration
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] 測試執行失敗: {e}")
            failed += 1

    print("\n" + "="*60)
    print("測試結果")
    print("="*60)
    print(f"通過: {passed}/{len(tests)}")
    print(f"失敗: {failed}/{len(tests)}")
    print("="*60)

    if failed == 0:
        print("\n[OK] 所有測試通過！")
        return True
    else:
        print(f"\n[ERROR] {failed} 個測試失敗")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
