"""compose_video.py

Create a quick before/after preview by overlaying masks on frames and
writing an MP4 to `results/before_after/preview.mp4`.

Usage (from project root):
    python src/compose_video.py --frames data/frames --masks data/masks \
        --out results/before_after/preview.mp4 --fps 10
"""

import os
import argparse
from typing import Tuple

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def overlay_mask_on_frame(frame_bgr: np.ndarray, mask_gray: np.ndarray, color=(0, 255, 0), alpha=0.4) -> np.ndarray:
    """Overlay a single-channel mask onto the frame with given color and alpha."""
    frame = frame_bgr.copy()
    if mask_gray.ndim == 3:
        mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)
    mask_bin = (mask_gray > 0).astype(np.uint8)
    overlay = np.zeros_like(frame)
    overlay[mask_bin.astype(bool)] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def compose_preview(frames_dir: str, masks_dir: str, out_video_path: str, fps: int = 10) -> Tuple[int, Tuple[int, int]]:
    names = sorted([n for n in os.listdir(frames_dir) if n.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not names:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    # Determine size from first frame
    first = cv2.imread(os.path.join(frames_dir, names[0]))
    if first is None:
        raise RuntimeError("Failed to read first frame")
    h, w = first.shape[:2]

    # Ensure even dimensions for H.264
    h_out = (h // 2) * 2
    w_out = (w // 2) * 2

    # Side-by-side: original | overlay
    frame_out_size = (w_out * 2, h_out)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ensure_dir(out_video_path)
    writer = cv2.VideoWriter(out_video_path, fourcc, float(fps), (frame_out_size[0], frame_out_size[1]))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_video_path}")

    count = 0
    for n in names:
        fpath = os.path.join(frames_dir, n)
        base = os.path.splitext(n)[0]
        mpath = os.path.join(masks_dir, f"{base}_mask.png")

        frame = cv2.imread(fpath)
        if frame is None:
            continue

        # Resize to even dims
        frame = cv2.resize(frame, (w_out, h_out), interpolation=cv2.INTER_AREA)

        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            overlay = frame.copy()
        else:
            mask = cv2.resize(mask, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
            overlay = overlay_mask_on_frame(frame, mask, color=(0, 255, 0), alpha=0.5)

        # Concatenate original and overlay side-by-side
        combined = np.concatenate([frame, overlay], axis=1)
        writer.write(combined)
        count += 1

    writer.release()
    return count, (frame_out_size[1], frame_out_size[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose side-by-side frame|overlay preview MP4")
    parser.add_argument("--frames", default=os.path.join("data", "frames"), help="Input frames dir")
    parser.add_argument("--masks", default=os.path.join("data", "masks"), help="Input masks dir")
    parser.add_argument("--out", default=os.path.join("results", "before_after", "preview.mp4"), help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=10, help="Output FPS")
    args = parser.parse_args()

    count, (h, w) = compose_preview(args.frames, args.masks, args.out, fps=args.fps)
    print(f"Wrote preview: {args.out} ({w}x{h}) with {count} frames")


if __name__ == "__main__":
    main()
