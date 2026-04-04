"""
inference_onnx_twopass.py
遠側高解析度兩段推理腳本

問題背景：
  VballNet 模型固定在 288x512 推理，攝影機架在某側邊線後方時，
  遠側球在畫面中只有約 2-3px，檢測困難。

解法：兩段推理
  Pass 1 (全畫面)：整幀縮放至 288x512 → 偵測近側球
  Pass 2 (遠側裁切)：裁切畫面上半部 (far_crop_ratio) 後縮放至 288x512
                    → 遠側球有效放大 2-3 倍，提升偵測率
  合併：遠側優先使用 crop 推理結果，近側使用全畫面推理結果

用法 (在 fast-volleyball-tracking-inference/ 目錄執行)：
    conda activate base  # 或含有 onnxruntime 的環境
    cd fast-volleyball-tracking-inference/src
    python inference_onnx_twopass.py \\
        --video_path ../../output_data/test_segment/segment_001_Team1.mp4 \\
        --model_path ../models/VballNetV1b_seq9_grayscale_best.onnx \\
        --output_dir ../../output/vball_twopass \\
        --only_csv \\
        --far_crop_ratio 0.45

輸出 CSV 格式與 inference_onnx_seq_gray_v2.py 完全相同：
    {output_dir}/{video_basename}/ball.csv
    欄位: Frame, Visibility, X, Y  (座標為原始影片解析度)
"""
import argparse
import logging
import os
import queue
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from tqdm import tqdm

try:
    from constants import (
        DEFAULT_HEATMAP_THRESHOLD,
        DEFAULT_INPUT_HEIGHT,
        DEFAULT_INPUT_WIDTH,
    )
    from models import BallTrack
except ImportError:
    # fallback when running from different cwd
    DEFAULT_HEATMAP_THRESHOLD = 0.5
    DEFAULT_INPUT_HEIGHT = 288
    DEFAULT_INPUT_WIDTH = 512

    class BallTrack:
        def __init__(self, maxlen):
            from collections import deque
            self._pts = deque(maxlen=maxlen)
        def update(self, pt):
            if pt is not None:
                self._pts.append(pt)
        def reset(self):
            self._pts.clear()
        def points(self):
            return list(self._pts)

LOG = logging.getLogger(__name__)

# ort.preload_dlls() 視環境決定是否需要；Linux 無此函數
try:
    ort.preload_dlls()
except AttributeError:
    pass


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="VballNet 兩段推理：全畫面 + 遠側裁切，提升遠側球偵測率"
    )
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--track_length", type=int, default=8)
    parser.add_argument(
        "--far_crop_ratio", type=float, default=0.45,
        help="畫面上方（遠側）裁切比例 (預設 0.45 = 上 45%%)"
    )
    parser.add_argument(
        "--far_merge_threshold", type=float, default=0.85,
        help="crop 偵測結果的 Y 座標（歸一化）低於此閾值才採用（預設 0.85 = crop 高度的 85%%）"
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=DEFAULT_HEATMAP_THRESHOLD
    )
    parser.add_argument(
        "--crop_threshold", type=float, default=None,
        help="crop pass 的信心閾值（預設與 confidence_threshold 相同）"
    )
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--only_csv", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--seq_len", type=int, default=None,
        help="手動指定模型序列長度（覆蓋自動偵測：預設從檔名偵測 seq9/seq15，否則 3）"
    )
    return parser.parse_args()


def load_onnx_model(model_path, seq_len_override=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    session = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    has_gru = "h0" in input_names
    h0_shape = None
    if has_gru:
        for inp in session.get_inputs():
            if inp.name == "h0":
                h0_shape = inp.shape
                break
        resolved = []
        for dim in h0_shape:
            if isinstance(dim, str) or dim is None:
                resolved.append(1 if dim in ["batch", "batch_size", None] else 512)
            else:
                resolved.append(dim)
        h0_shape = tuple(resolved)

    # 從 ONNX input/output shape 自動推斷 batch_size (input seq) 和 out_dim (output seq)
    batch_size = None
    out_dim = None
    for inp in session.get_inputs():
        if inp.name != "h0":
            shape = inp.shape
            if len(shape) >= 2 and isinstance(shape[1], int):
                batch_size = shape[1]
            break
    for outp in session.get_outputs():
        shape = outp.shape
        if len(shape) >= 2 and isinstance(shape[1], int):
            out_dim = shape[1]
        break

    # 允許手動覆蓋（用於舊版無法自動偵測的模型）
    if seq_len_override is not None:
        batch_size = seq_len_override

    # Fallback：從檔名推斷
    if batch_size is None:
        if "seq15" in model_path.lower():
            batch_size = 15
        elif "seq9" in model_path.lower():
            batch_size = 9
        else:
            batch_size = 3
    if out_dim is None:
        out_dim = batch_size

    LOG.info("Model: %s | in_seq=%d | out_seq=%d | GRU=%s", model_path, batch_size, out_dim, has_gru)
    return session, has_gru, out_dim, h0_shape, batch_size, input_names, output_names


def initialize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, W, H, fps, total


def setup_csv(video_basename, output_dir):
    if output_dir is None:
        return None
    d = os.path.join(output_dir, video_basename)
    os.makedirs(d, exist_ok=True)
    csv_path = os.path.join(d, "ball.csv")
    pd.DataFrame(columns=["Frame", "Visibility", "X", "Y"]).to_csv(csv_path, index=False)
    return csv_path


def append_csv(result, csv_path):
    if csv_path is None:
        return
    pd.DataFrame([result]).to_csv(csv_path, mode="a", header=False, index=False)


def append_csv_batch(results, csv_path):
    """批次寫入多筆結果，減少檔案開關次數"""
    if csv_path is None or not results:
        return
    pd.DataFrame(results).to_csv(csv_path, mode="a", header=False, index=False)


def preprocess_full(frames, H_model=DEFAULT_INPUT_HEIGHT, W_model=DEFAULT_INPUT_WIDTH):
    """整幀縮放至 H_model x W_model，灰階歸一化"""
    out = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        out.append(cv2.resize(gray, (W_model, H_model)).astype(np.float32) / 255.0)
    return out


def preprocess_crop(frames, crop_ratio, H_model=DEFAULT_INPUT_HEIGHT, W_model=DEFAULT_INPUT_WIDTH):
    """裁切上 crop_ratio 比例（遠側），縮放至 H_model x W_model，灰階歸一化"""
    out = []
    for f in frames:
        h = f.shape[0]
        ch = max(1, int(h * crop_ratio))
        cropped = f[:ch, :]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        out.append(cv2.resize(gray, (W_model, H_model)).astype(np.float32) / 255.0)
    return out


def postprocess(output, threshold, H_model, W_model, out_dim):
    """輸出 [(visibility, cx, cy), ...] 座標在模型空間 (H_model x W_model)"""
    results = []
    for t in range(out_dim):
        heatmap = output[0, t, :, :]
        _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            c = max(contours, key=cv2.contourArea)
            m = cv2.moments(c)
            if m["m00"] != 0:
                results.append((1, int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))
                continue
        results.append((0, 0, 0))
    return results


def run_inference(session, frames_list, has_gru, h0, input_names, output_names,
                  H_model, W_model, batch_size):
    """frames_list: list of (H_model, W_model) float32 arrays, 長度 == batch_size"""
    tensor = np.stack(frames_list, axis=0)          # (batch, H, W)
    tensor = tensor[:, np.newaxis, :, :]             # (batch, 1, H, W)  — channels_first, 1 ch each
    tensor = np.transpose(tensor, (1, 0, 2, 3))      # (1, batch, H, W) → 但原腳本是 (batch, ch, H, W)
    # 實際格式: input shape 為 (1, seq, H, W) = (batch=1, channels=seq, H, W)
    # 根據 inference_onnx_seq_gray_v2.py 的邏輯:
    #   frame_buffer 是 list of (H,W), stack axis=2 → (H,W,seq), expand → (1,H,W,seq), transpose (0,3,1,2) → (1,seq,H,W)
    tensor = np.stack(frames_list, axis=2)            # (H, W, seq)
    tensor = np.expand_dims(tensor, axis=0)           # (1, H, W, seq)
    tensor = np.transpose(tensor, (0, 3, 1, 2))       # (1, seq, H, W)
    tensor = tensor.astype(np.float32)

    inputs = {input_names[0]: tensor}
    if has_gru and h0 is not None:
        inputs[input_names[1]] = h0

    outputs = session.run(output_names, inputs)
    heatmaps = outputs[0]
    new_h0 = outputs[1] if (has_gru and len(outputs) >= 2) else None
    return heatmaps, new_h0


class BallTrackState:
    def __init__(self, maxlen, max_missing):
        self._track = BallTrack(maxlen)
        self._missing = 0
        self._max_missing = max_missing

    def update(self, point):
        self._track.update(point)
        self._missing = 0 if point is not None else self._missing + 1

    def is_lost(self):
        return self._missing >= self._max_missing

    def reset(self):
        self._track.reset()
        self._missing = 0

    def points(self):
        return self._track.points()


def frame_reader(cap, q, batch_size, stop_event, err_q):
    try:
        while not stop_event.is_set():
            batch = []
            for _ in range(batch_size):
                if stop_event.is_set():
                    break
                ret, f = cap.read()
                if not ret:
                    break
                batch.append(f)
            # 使用短輪詢避免 queue.Full 例外誤判為致命錯誤
            # （舊版 timeout=1.0 在 ONNX 熱身 / 批次 I/O 較慢時會誤觸發）
            item = batch if batch else None
            while not stop_event.is_set():
                try:
                    q.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if not batch:
                break
    except Exception as e:
        err_q.put(e)
        stop_event.set()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    H_model = DEFAULT_INPUT_HEIGHT  # 288
    W_model = DEFAULT_INPUT_WIDTH   # 512
    crop_threshold = args.crop_threshold if args.crop_threshold is not None else args.confidence_threshold

    (session, has_gru, out_dim, h0_shape, batch_size,
     input_names, output_names) = load_onnx_model(args.model_path, seq_len_override=args.seq_len)

    cap, vid_W, vid_H, fps, total_frames = initialize_video(args.video_path)
    vname = os.path.splitext(os.path.basename(args.video_path))[0]
    csv_path = setup_csv(vname, args.output_dir)

    # 視訊輸出
    out_writer = None
    if args.output_dir and not args.only_csv:
        d = os.path.join(args.output_dir, vname)
        os.makedirs(d, exist_ok=True)
        out_writer = cv2.VideoWriter(
            os.path.join(d, "predict.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_W, vid_H)
        )

    # 遠側邊界（在原始影片座標中）
    far_y_max = vid_H * args.far_crop_ratio * args.far_merge_threshold

    h0_full = np.zeros(h0_shape, dtype=np.float32) if (has_gru and h0_shape) else None
    h0_crop = np.zeros(h0_shape, dtype=np.float32) if (has_gru and h0_shape) else None

    buf_full = []
    buf_crop = []

    # 每次讀取 out_dim 新幀（滑動步長），緩衝區維持 batch_size 幀
    # 這樣模型始終看到 batch_size 幀的上下文，但每次只前進 out_dim 幀
    # 確保所有幀都被寫入 CSV（不跳幀）
    stride = out_dim  # 每次讀取的新幀數 = 輸出維度
    fq = queue.Queue(maxsize=4)
    eq = queue.Queue()
    stop = threading.Event()
    t = threading.Thread(target=frame_reader, args=(cap, fq, stride, stop, eq), daemon=True)
    t.start()

    pbar = tqdm(total=total_frames, desc=f"TwoPass {vname}", unit="f")
    frame_idx = 0
    stats = {"full": 0, "crop": 0, "none": 0}

    try:
        while not stop.is_set():
            if not eq.empty():
                raise eq.get()
            try:
                frames = fq.get(timeout=0.5)
            except queue.Empty:
                if not t.is_alive():
                    break
                continue
            if not frames:
                break

            # --- 前處理 ---
            pf_full = preprocess_full(frames, H_model, W_model)
            pf_crop = preprocess_crop(frames, args.far_crop_ratio, H_model, W_model)

            # 填充到 batch_size
            pad_full = [pf_full[0] if pf_full else np.zeros((H_model, W_model), np.float32)]
            pad_crop = [pf_crop[0] if pf_crop else np.zeros((H_model, W_model), np.float32)]
            while len(buf_full) < batch_size:
                buf_full.append(pad_full[0])
                buf_crop.append(pad_crop[0])
            for pf, pc in zip(pf_full, pf_crop):
                buf_full.append(pf)
                buf_crop.append(pc)
            buf_full = buf_full[-batch_size:]
            buf_crop = buf_crop[-batch_size:]

            # --- 全畫面推理 ---
            out_full, new_h0_full = run_inference(
                session, buf_full, has_gru, h0_full, input_names, output_names,
                H_model, W_model, batch_size
            )
            if has_gru and new_h0_full is not None:
                h0_full = new_h0_full

            # --- 裁切推理 ---
            out_crop, new_h0_crop = run_inference(
                session, buf_crop, has_gru, h0_crop, input_names, output_names,
                H_model, W_model, batch_size
            )
            if has_gru and new_h0_crop is not None:
                h0_crop = new_h0_crop

            preds_full = postprocess(out_full, args.confidence_threshold, H_model, W_model, out_dim)
            preds_crop = postprocess(out_crop, crop_threshold, H_model, W_model, out_dim)

            # --- 合併 & 輸出 ---
            batch_csv_rows = []  # 批次收集，一次寫入避免頻繁開關檔案
            for i, (frames_i_full, frames_i_crop) in enumerate(
                zip(preds_full[: len(frames)], preds_crop[: len(frames)])
            ):
                vis_full, cx_full, cy_full = frames_i_full
                vis_crop, cx_crop, cy_crop = frames_i_crop

                # 還原到原始影片座標
                if vis_full:
                    x_full_orig = int(cx_full * vid_W / W_model)
                    y_full_orig = int(cy_full * vid_H / H_model)
                else:
                    x_full_orig = y_full_orig = -1

                if vis_crop:
                    crop_h_px = int(vid_H * args.far_crop_ratio)
                    x_crop_orig = int(cx_crop * vid_W / W_model)
                    y_crop_orig = int(cy_crop * crop_h_px / H_model)
                else:
                    x_crop_orig = y_crop_orig = -1

                # 合併規則：
                #   crop 偵測到球且 y < far_y_max (確實在遠側) → 優先用 crop
                #   否則使用全畫面結果
                use_crop = (vis_crop == 1) and (y_crop_orig < far_y_max)

                if use_crop:
                    final_vis, final_x, final_y = 1, x_crop_orig, y_crop_orig
                    stats["crop"] += 1
                elif vis_full:
                    final_vis, final_x, final_y = 1, x_full_orig, y_full_orig
                    stats["full"] += 1
                else:
                    final_vis, final_x, final_y = 0, 0, 0
                    stats["none"] += 1

                batch_csv_rows.append(
                    {"Frame": frame_idx + i, "Visibility": final_vis,
                     "X": final_x, "Y": final_y}
                )

                if args.visualize or out_writer is not None:
                    vis_frame = frames[i].copy()
                    if final_vis:
                        cv2.circle(vis_frame, (final_x, final_y), 8, (0, 255, 255), -1)
                        src = "C" if use_crop else "F"
                        cv2.putText(vis_frame, src, (final_x + 10, final_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    if out_writer is not None:
                        out_writer.write(vis_frame)
                    if args.visualize:
                        cv2.imshow("TwoPass", vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stop.set()
                            break

            # 批次寫入 CSV（一次開關檔案取代每幀一次）
            append_csv_batch(batch_csv_rows, csv_path)
            pbar.update(len(frames))
            frame_idx += len(frames)

    finally:
        stop.set()
        t.join(timeout=2.0)
        pbar.close()
        cap.release()
        if out_writer:
            out_writer.release()
        if args.visualize:
            cv2.destroyAllWindows()

    total = stats["full"] + stats["crop"] + stats["none"]
    det = stats["full"] + stats["crop"]
    LOG.info(
        "完成: 總 %d 幀 | 偵測 %d (%.1f%%) | crop貢獻 %d (%.1f%%) | full貢獻 %d (%.1f%%)",
        total, det, 100 * det / max(total, 1),
        stats["crop"], 100 * stats["crop"] / max(total, 1),
        stats["full"],  100 * stats["full"]  / max(total, 1),
    )
    if csv_path:
        print(f"[OK] CSV: {csv_path}")
        print(f"     偵測率: {100 * det / max(total, 1):.1f}%  "
              f"(crop={100 * stats['crop'] / max(total, 1):.1f}%  "
              f"full={100 * stats['full'] / max(total, 1):.1f}%)")


if __name__ == "__main__":
    main()
