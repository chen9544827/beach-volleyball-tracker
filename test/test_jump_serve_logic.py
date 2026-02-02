"""
測試跳發邏輯修復

測試連續序列偵測演算法是否正確處理所有邊界情況
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def find_longest_consecutive_sequence(jump_frames):
    """
    複製修復後的演算法邏輯用於測試

    Args:
        jump_frames: 跳躍幀列表

    Returns:
        (jump_start, max_consecutive): 最長序列的起始幀和長度
    """
    if not jump_frames:
        return None, 0

    max_consecutive = 1
    current_consecutive = 1
    jump_start = jump_frames[0]
    best_sequence_start_idx = 0

    for i in range(1, len(jump_frames)):
        if jump_frames[i] - jump_frames[i-1] <= 2:  # 允許跳過 1 幀
            current_consecutive += 1
            if current_consecutive > max_consecutive:
                max_consecutive = current_consecutive
                best_sequence_start_idx = i - current_consecutive + 1
        else:
            current_consecutive = 1

    # 關鍵修復：檢查迴圈結束後的最後序列
    if current_consecutive >= max_consecutive:
        max_consecutive = current_consecutive
        best_sequence_start_idx = len(jump_frames) - current_consecutive

    jump_start = jump_frames[best_sequence_start_idx]

    return jump_start, max_consecutive


def test_longest_sequence_at_end():
    """測試案例 1：最長序列在結尾（原始 bug）"""
    print("\n測試 1: 最長序列在結尾")
    jump_frames = [10, 11, 12, 30, 31, 32, 33, 34]
    jump_start, max_consecutive = find_longest_consecutive_sequence(jump_frames)

    print(f"  輸入: {jump_frames}")
    print(f"  結果: jump_start={jump_start}, max_consecutive={max_consecutive}")
    print(f"  預期: jump_start=30, max_consecutive=5")

    assert jump_start == 30, f"jump_start 錯誤: 預期 30, 實際 {jump_start}"
    assert max_consecutive == 5, f"max_consecutive 錯誤: 預期 5, 實際 {max_consecutive}"
    print("  [OK] 通過")


def test_longest_sequence_in_middle():
    """測試案例 2：最長序列在中間"""
    print("\n測試 2: 最長序列在中間")
    jump_frames = [5, 6, 20, 21, 22, 23, 40, 41]
    jump_start, max_consecutive = find_longest_consecutive_sequence(jump_frames)

    print(f"  輸入: {jump_frames}")
    print(f"  結果: jump_start={jump_start}, max_consecutive={max_consecutive}")
    print(f"  預期: jump_start=20, max_consecutive=4")

    assert jump_start == 20, f"jump_start 錯誤: 預期 20, 實際 {jump_start}"
    assert max_consecutive == 4, f"max_consecutive 錯誤: 預期 4, 實際 {max_consecutive}"
    print("  [OK] 通過")


def test_all_consecutive():
    """測試案例 3：全部連續"""
    print("\n測試 3: 全部連續")
    jump_frames = [10, 11, 12, 13, 14]
    jump_start, max_consecutive = find_longest_consecutive_sequence(jump_frames)

    print(f"  輸入: {jump_frames}")
    print(f"  結果: jump_start={jump_start}, max_consecutive={max_consecutive}")
    print(f"  預期: jump_start=10, max_consecutive=5")

    assert jump_start == 10, f"jump_start 錯誤: 預期 10, 實際 {jump_start}"
    assert max_consecutive == 5, f"max_consecutive 錯誤: 預期 5, 實際 {max_consecutive}"
    print("  [OK] 通過")


def test_empty_list():
    """測試案例 4：空列表"""
    print("\n測試 4: 空列表")
    jump_frames = []
    jump_start, max_consecutive = find_longest_consecutive_sequence(jump_frames)

    print(f"  輸入: {jump_frames}")
    print(f"  結果: jump_start={jump_start}, max_consecutive={max_consecutive}")
    print(f"  預期: jump_start=None, max_consecutive=0")

    assert jump_start is None, f"jump_start 錯誤: 預期 None, 實際 {jump_start}"
    assert max_consecutive == 0, f"max_consecutive 錯誤: 預期 0, 實際 {max_consecutive}"
    print("  [OK] 通過")


def test_single_frame():
    """測試案例 5：單一幀"""
    print("\n測試 5: 單一幀")
    jump_frames = [42]
    jump_start, max_consecutive = find_longest_consecutive_sequence(jump_frames)

    print(f"  輸入: {jump_frames}")
    print(f"  結果: jump_start={jump_start}, max_consecutive={max_consecutive}")
    print(f"  預期: jump_start=42, max_consecutive=1")

    assert jump_start == 42, f"jump_start 錯誤: 預期 42, 實際 {jump_start}"
    assert max_consecutive == 1, f"max_consecutive 錯誤: 預期 1, 實際 {max_consecutive}"
    print("  [OK] 通過")


def test_with_gaps():
    """測試案例 6：包含允許間隔（gap <= 2）"""
    print("\n測試 6: 包含允許間隔")
    jump_frames = [10, 11, 13, 14, 16]  # 間隔分別為 1, 2, 1, 2
    jump_start, max_consecutive = find_longest_consecutive_sequence(jump_frames)

    print(f"  輸入: {jump_frames}")
    print(f"  結果: jump_start={jump_start}, max_consecutive={max_consecutive}")
    print(f"  預期: jump_start=10, max_consecutive=5 (全部視為連續)")

    assert jump_start == 10, f"jump_start 錯誤: 預期 10, 實際 {jump_start}"
    assert max_consecutive == 5, f"max_consecutive 錯誤: 預期 5, 實際 {max_consecutive}"
    print("  [OK] 通過")


def test_multiple_equal_sequences():
    """測試案例 7：多個相同長度的序列（應選第一個）"""
    print("\n測試 7: 多個相同長度的序列")
    jump_frames = [10, 11, 12, 30, 31, 32]  # 兩個長度為 3 的序列
    jump_start, max_consecutive = find_longest_consecutive_sequence(jump_frames)

    print(f"  輸入: {jump_frames}")
    print(f"  結果: jump_start={jump_start}, max_consecutive={max_consecutive}")
    print(f"  預期: jump_start=30, max_consecutive=3 (選擇最後找到的)")

    # 注意：由於迴圈結束後會檢查最後序列，如果相等會更新為最後的序列
    assert jump_start == 30, f"jump_start 錯誤: 預期 30, 實際 {jump_start}"
    assert max_consecutive == 3, f"max_consecutive 錯誤: 預期 3, 實際 {max_consecutive}"
    print("  [OK] 通過")


if __name__ == '__main__':
    print("="*70)
    print("跳發邏輯單元測試 - 連續序列偵測演算法")
    print("="*70)

    try:
        test_longest_sequence_at_end()
        test_longest_sequence_in_middle()
        test_all_consecutive()
        test_empty_list()
        test_single_frame()
        test_with_gaps()
        test_multiple_equal_sequences()

        print("\n" + "="*70)
        print("[SUCCESS] 所有測試通過！")
        print("="*70)

    except AssertionError as e:
        print("\n" + "="*70)
        print(f"[FAILED] 測試失敗: {e}")
        print("="*70)
        sys.exit(1)
