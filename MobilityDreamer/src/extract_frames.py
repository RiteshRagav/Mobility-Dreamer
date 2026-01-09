"""extract_frames.py

Extract frames from an MP4 into `data/frames/` and write `metadata.json`.

Requirements:
- opencv-python (`pip install opencv-python`), or install via conda: `conda install -c conda-forge opencv`

Usage (from project root):
    python src/extract_frames.py
"""

import json
import os
from typing import Dict
import argparse

import cv2


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def extract_frames(input_video_path: str, out_dir: str, jpg_quality: int = 90) -> Dict[str, int]:
    """Extract frames from `input_video_path` into `out_dir`.

    Writes frames as JPGs named `frame_XXXX.jpg` and returns metadata:
    {"fps": int, "total_frames": int, "width": int, "height": int}
    """
    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    ensure_dir(out_dir)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out_path = os.path.join(out_dir, f"frame_{count:04d}.jpg")
        # Write JPEG with specified quality
        cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
        count += 1

    cap.release()

    meta = {"fps": fps, "total_frames": count, "width": width, "height": height}
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done! Extracted {count} frames to {out_dir} (fps={fps}, {width}x{height}).")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from a video to data/frames/")
    parser.add_argument(
        "--input",
        dest="input_video",
        default=os.path.join("data", "input_videos", "kitti", "kitti_0048_short.mp4"),
        help="Path to input MP4 video (default: data/input_videos/kitti/kitti_0048_short.mp4)",
    )
    parser.add_argument(
        "--out",
        dest="out_dir",
        default=os.path.join("data", "frames"),
        help="Output directory for extracted frames (default: data/frames)",
    )
    args = parser.parse_args()

    print(f"Using input video: {args.input_video}")
    print(f"Output frames to: {args.out_dir}")
    extract_frames(args.input_video, args.out_dir)


if __name__ == '__main__':
    main()
