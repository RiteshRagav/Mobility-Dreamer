#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create sequence index from existing processed frames.
This generates the train_sequences.json / val_sequences.json files needed by BDD100KUrbanDataset.
"""
import json
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sequences_from_frames(frames_dir, output_dir, sequence_length=8, stride=2, split="train"):
    """
    Create temporal sequences from existing frames.
    
    Args:
        frames_dir: Directory containing frame_XXXX.jpg files
        output_dir: Output directory for sequence index
        sequence_length: Number of frames per sequence
        stride: Skip frames (stride=2 means use every 2nd frame)
        split: "train" or "val"
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all frames
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.png"))
    
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    
    logger.info(f"Found {len(frame_files)} frames in {frames_dir}")
    
    # Extract frame indices
    frame_indices = []
    for f in frame_files:
        # Extract number from frame_XXXX.jpg
        idx = int(f.stem.split('_')[1])
        frame_indices.append(idx)
    
    frame_indices = sorted(frame_indices)
    logger.info(f"Frame indices range: {min(frame_indices)} to {max(frame_indices)}")
    
    # Create sequences
    sequences = []
    video_id = "default_video"  # Since we have one video worth of frames
    
    for start_idx in range(0, len(frame_indices) - sequence_length * stride + 1, stride):
        seq_indices = []
        for i in range(sequence_length):
            frame_idx = frame_indices[start_idx + i * stride]
            seq_indices.append(frame_idx)
        
        sequence_id = f"{video_id}_seq_{len(sequences):04d}"
        sequences.append({
            "sequence_id": sequence_id,
            "video_id": video_id,
            "frame_indices": seq_indices
        })
    
    logger.info(f"Created {len(sequences)} sequences of length {sequence_length} with stride {stride}")
    
    # Save sequence index
    output_file = output_dir / f"{split}_sequences.json"
    with open(output_file, 'w') as f:
        json.dump(sequences, f, indent=2)
    
    logger.info(f"Saved sequence index to {output_file}")
    
    return sequences


def main():
    parser = argparse.ArgumentParser(description="Create sequence index from existing frames")
    parser.add_argument("--frames", type=str, required=True, help="Directory containing frames")
    parser.add_argument("--output", type=str, required=True, help="Output directory for sequence index")
    parser.add_argument("--sequence-length", type=int, default=8, help="Frames per sequence")
    parser.add_argument("--stride", type=int, default=2, help="Stride between frames")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Split name")
    args = parser.parse_args()
    
    sequences = create_sequences_from_frames(
        args.frames,
        args.output,
        args.sequence_length,
        args.stride,
        args.split
    )
    
    print(f"\n✅ Created {len(sequences)} sequences")
    print(f"   Sequence length: {args.sequence_length}")
    print(f"   Stride: {args.stride}")
    print(f"   Output: {args.output}")
    print("\nExample sequence:")
    print(json.dumps(sequences[0], indent=2))


if __name__ == "__main__":
    main()
