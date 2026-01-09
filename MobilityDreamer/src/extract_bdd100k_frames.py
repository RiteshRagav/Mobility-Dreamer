"""
Extract Frames from BDD100K Dataset
====================================
Extracts frames from BDD100K videos and prepares them for MobilityDreamer pipeline.

BDD100K is a large-scale driving dataset with diverse urban scenarios.
Perfect for demonstrating how infrastructure changes affect real cities!

Usage:
    python src/extract_bdd100k_frames.py --video-dir PATH/TO/BDD100K/videos/train --out data/frames --num-videos 3 --frames-per-video 10

Example:
    python src/extract_bdd100k_frames.py --video-dir "C:/Users/Udai Ratinam G/Downloads/Minor Project/bdd100k_videos_train_00/bdd100k/videos/train" --out data/frames --num-videos 2 --frames-per-video 5
"""

import cv2
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import random


def get_evenly_spaced_frames(video_path, num_frames):
    """
    Extract evenly spaced frames from a video.
    
    Args:
        video_path: Path to video file
        num_frames: How many frames to extract
        
    Returns:
        List of (frame_index, frame) tuples
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        print(f"[WARNING] Could not read {video_path.name}")
        cap.release()
        return []
    
    # Calculate stride to get evenly spaced frames
    stride = max(1, total_frames // num_frames)
    
    frames = []
    frame_count = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % stride == 0 and frame_count < num_frames:
            frames.append((frame_idx, frame))
            frame_count += 1
        
        frame_idx += 1
    
    cap.release()
    return frames


def extract_from_bdd100k(video_dir, output_dir, num_videos=3, frames_per_video=10):
    """
    Extract frames from BDD100K videos.
    
    Args:
        video_dir: Directory containing .mov files
        output_dir: Where to save extracted frames
        num_videos: How many videos to process
        frames_per_video: Frames to extract per video
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = list(video_dir.glob("*.mov"))
    
    if len(video_files) == 0:
        print(f"[ERROR] No .mov files found in {video_dir}")
        return
    
    # Randomly sample videos (or take first N)
    selected_videos = video_files[:num_videos]
    
    print(f"Extracting frames from BDD100K dataset")
    print(f"   Videos available: {len(video_files)}")
    print(f"   Videos to process: {len(selected_videos)}")
    print(f"   Frames per video: {frames_per_video}")
    print()
    
    frame_counter = 0
    metadata = {
        "dataset": "BDD100K",
        "total_frames": 0,
        "videos_processed": [],
        "frames_per_second": None
    }
    
    for video_idx, video_path in enumerate(selected_videos):
        print(f"Video {video_idx + 1}/{len(selected_videos)}: {video_path.name}")
        
        # Extract frames
        frames = get_evenly_spaced_frames(video_path, frames_per_video)
        
        if not frames:
            print(f"   [WARNING] Skipped (could not read)")
            continue
        
        # Save frames
        for frame_idx, frame in frames:
            frame_name = f"frame_{frame_counter:04d}.jpg"
            frame_path = output_dir / frame_name
            
            # Resize if too large (BDD100K videos are high resolution)
            # Keep aspect ratio, max width 1280
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                new_size = (1280, int(h * scale))
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            
            cv2.imwrite(str(frame_path), frame)
            frame_counter += 1
        
        metadata["videos_processed"].append({
            "video": video_path.name,
            "frames_extracted": len(frames)
        })
        
        print(f"   [OK] Extracted {len(frames)} frames")
    
    metadata["total_frames"] = frame_counter
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[SUCCESS] Done! Extracted {frame_counter} frames from {len(selected_videos)} BDD100K videos")
    print(f"   Saved to: {output_dir}")
    print(f"   Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from BDD100K videos")
    parser.add_argument("--video-dir", type=str, required=True,
                        help="Path to BDD100K videos directory (containing .mov files)")
    parser.add_argument("--out", type=str, default="data/frames",
                        help="Output directory for extracted frames")
    parser.add_argument("--num-videos", type=int, default=3,
                        help="Number of videos to process")
    parser.add_argument("--frames-per-video", type=int, default=10,
                        help="Number of frames to extract from each video")
    args = parser.parse_args()
    
    extract_from_bdd100k(args.video_dir, args.out, args.num_videos, args.frames_per_video)


if __name__ == "__main__":
    main()
