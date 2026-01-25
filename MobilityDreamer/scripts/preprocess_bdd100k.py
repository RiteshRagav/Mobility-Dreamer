# -*- coding: utf-8 -*-
"""
Preprocess BDD100K videos into train/val sequence datasets for MobilityDreamer.

Steps (minimal, safe skeleton):
1) Sample videos and extract frames (stride configurable)
2) (Optional) Run segmentation (YOLOv8) to produce class-id maps
3) (Optional) Run MiDaS depth estimation
4) Generate synthetic policy masks (placeholder)
5) Write train/val sequence index JSON files

Run:
    python scripts/preprocess_bdd100k.py --config config/mobility_config.py \
        --max-videos 5 --frames-per-video 40 --val-ratio 0.15

Notes:
- This script is a skeleton; segmentation/depth steps are stubbed with TODOs.
- Keeps everything lightweight and non-destructive.
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np


def load_config(cfg_path: str):
    # Execute cfg file safely
    cfg_scope = {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        code = compile(f.read(), cfg_path, "exec")
        exec(code, cfg_scope)
    return cfg_scope["cfg"]


def list_videos(video_dir: Path) -> List[Path]:
    vids = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mov")))
    return vids


def extract_frames(video_path: Path, out_dir: Path, max_frames: int, stride: int = 2) -> List[int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"WARN: cannot open video {video_path}")
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = []
    idx = 0
    saved = 0
    while saved < max_frames and idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        out_path = out_dir / f"frame_{idx:06d}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame)
        frame_ids.append(idx)
        saved += 1
        idx += stride
    cap.release()
    return frame_ids


def save_dummy_maps(ids: List[int], out_dir: Path, shape=(720, 1280)):
    """Create zero segmentation/policy/depth placeholders so pipeline can run."""
    h, w = shape
    seg_dir = out_dir / "segmentation"
    pol_dir = out_dir / "policy"
    dep_dir = out_dir / "depth"
    for d in [seg_dir, pol_dir, dep_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for fid in ids:
        seg_path = seg_dir / f"seg_{fid:06d}.png"
        pol_path = pol_dir / f"policy_{fid:06d}.png"
        dep_path = dep_dir / f"depth_{fid:06d}.png"
        zero_seg = 0 * np.zeros((h, w), dtype=np.uint8)
        if not seg_path.exists():
            cv2.imwrite(str(seg_path), zero_seg)
        if not pol_path.exists():
            cv2.imwrite(str(pol_path), zero_seg)
        if not dep_path.exists():
            cv2.imwrite(str(dep_path), zero_seg)


def run_segmentation(frame_paths: List[Path], seg_dir: Path):
    """Run YOLOv8 segmentation if available; otherwise, write zeros."""
    seg_dir.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics import YOLO  # type: ignore

        model = YOLO("yolov8n-seg.pt")
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            res = model.predict(source=img, imgsz=img.shape[0], verbose=False)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            if len(res) > 0 and res[0].masks is not None:
                # Use class ids to populate mask; clamp to 255 classes max
                for cls_id, m in zip(res[0].boxes.cls.cpu().numpy().astype(int), res[0].masks.data.cpu().numpy()):
                    m_bin = (m > 0.5).astype(np.uint8)
                    mask[m_bin == 1] = min(int(cls_id), 254) + 1  # reserve 0 for background
            out_path = seg_dir / fp.name.replace("frame_", "seg_").replace(".jpg", ".png")
            cv2.imwrite(str(out_path), mask)
    except Exception as e:
        print(f"WARN: segmentation fallback to zeros (reason: {e})")
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            h, w = (img.shape[0], img.shape[1]) if img is not None else (720, 1280)
            out_path = seg_dir / fp.name.replace("frame_", "seg_").replace(".jpg", ".png")
            cv2.imwrite(str(out_path), np.zeros((h, w), dtype=np.uint8))


def run_depth(frame_paths: List[Path], depth_dir: Path):
    """Run MiDaS depth if available; otherwise, write zeros."""
    depth_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        midas.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = transforms.dpt_transform
        device = next(midas.parameters()).device
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_batch = transform(img_rgb).to(device)
            with torch.no_grad():
                pred = midas(input_batch)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False
                ).squeeze()
            depth = pred.cpu().numpy()
            depth = depth - depth.min()
            depth = depth / (depth.max() + 1e-8)
            depth_u8 = (depth * 255).astype(np.uint8)
            out_path = depth_dir / fp.name.replace("frame_", "depth_").replace(".jpg", ".png")
            cv2.imwrite(str(out_path), depth_u8)
    except Exception as e:
        print(f"WARN: depth fallback to zeros (reason: {e})")
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            h, w = (img.shape[0], img.shape[1]) if img is not None else (720, 1280)
            out_path = depth_dir / fp.name.replace("frame_", "depth_").replace(".jpg", ".png")
            cv2.imwrite(str(out_path), np.zeros((h, w), dtype=np.uint8))


def synth_policy_masks(frame_paths: List[Path], pol_dir: Path, colors=4):
    """Generate simple synthetic policy masks (random rectangles per frame)."""
    pol_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        h, w = (img.shape[0], img.shape[1]) if img is not None else (720, 1280)
        mask = np.zeros((h, w), dtype=np.uint8)
        n_rect = rng.integers(1, 3)
        for _ in range(n_rect):
            x1, y1 = rng.integers(0, w // 2), rng.integers(0, h // 2)
            x2, y2 = rng.integers(w // 2, w), rng.integers(h // 2, h)
            cls = int(rng.integers(1, colors))  # reserve 0 for background
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=cls, thickness=-1)
        out_path = pol_dir / fp.name.replace("frame_", "policy_").replace(".jpg", ".png")
        cv2.imwrite(str(out_path), mask)


def build_sequences(video_id: str, frame_ids: List[int], seq_len: int) -> List[Dict]:
    sequences = []
    for i in range(0, len(frame_ids) - seq_len + 1, seq_len):
        seq = frame_ids[i : i + seq_len]
        sequences.append({
            "sequence_id": f"{video_id}_{seq[0]:06d}",
            "video_id": video_id,
            "frame_indices": seq,
        })
    return sequences


def split_train_val(seq_list: List[Dict], val_ratio: float):
    random.shuffle(seq_list)
    n_val = max(1, int(len(seq_list) * val_ratio))
    val = seq_list[:n_val]
    train = seq_list[n_val:]
    return train, val


def main():
    parser = argparse.ArgumentParser(description="Preprocess BDD100K for MobilityDreamer")
    parser.add_argument("--config", type=str, default="config/mobility_config.py")
    parser.add_argument("--max-videos", type=int, default=5)
    parser.add_argument("--frames-per-video", type=int, default=40)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--enable-seg", action="store_true", help="Run YOLOv8 segmentation if available")
    parser.add_argument("--enable-depth", action="store_true", help="Run MiDaS depth if available")
    parser.add_argument("--enable-policy", action="store_true", help="Generate synthetic policy masks")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg.DATASETS.BDD100K

    video_dir = Path(ds_cfg.VIDEO_DIR)
    processed_dir = Path(ds_cfg.PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(video_dir)
    if not videos:
        print(f"No videos found in {video_dir}. Exiting.")
        return

    videos = videos[: args.max_videos]
    all_sequences = []

    for vid in videos:
        vid_id = vid.stem
        out_vid_dir = processed_dir / "train" / vid_id / "frames"
        out_vid_dir.mkdir(parents=True, exist_ok=True)
        frame_ids = extract_frames(vid, out_vid_dir, max_frames=args.frames_per_video, stride=args.stride)
        if not frame_ids:
            continue
        frame_paths = [out_vid_dir / f"frame_{fid:06d}.jpg" for fid in frame_ids]

        seg_dir = processed_dir / "train" / vid_id / "segmentation"
        pol_dir = processed_dir / "train" / vid_id / "policy"
        dep_dir = processed_dir / "train" / vid_id / "depth"

        if args.enable_seg:
            run_segmentation(frame_paths, seg_dir)
        if args.enable_depth:
            run_depth(frame_paths, dep_dir)
        if args.enable_policy:
            synth_policy_masks(frame_paths, pol_dir, colors=ds_cfg.POLICY_CLASSES)

        # Safety: ensure placeholders exist for any missing modality
        save_dummy_maps(frame_ids, processed_dir / "train" / vid_id, shape=(ds_cfg.IMAGE_SIZE[0], ds_cfg.IMAGE_SIZE[1]))
        seqs = build_sequences(vid_id, frame_ids, ds_cfg.SEQUENCE_LENGTH)
        all_sequences.extend(seqs)

    train_seqs, val_seqs = split_train_val(all_sequences, args.val_ratio)

    with open(processed_dir / "train_sequences.json", "w", encoding="utf-8") as f:
        json.dump(train_seqs, f, indent=2)
    with open(processed_dir / "val_sequences.json", "w", encoding="utf-8") as f:
        json.dump(val_seqs, f, indent=2)

    print(f"Saved {len(train_seqs)} train sequences, {len(val_seqs)} val sequences to {processed_dir}")


if __name__ == "__main__":
    main()
