# -*- coding: utf-8 -*-
#
# @File:   bdd100k_dataset.py
# @Author: MobilityDreamer Team
# @Date:   2026-01-23
# @Description: BDD100K dataset loader with temporal sequences and policy conditioning

import os
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BDD100KUrbanDataset(Dataset):
    """
    BDD100K Dataset for temporal urban scene generation with policy conditioning.
    
    Features:
    - Temporal sequences (8 consecutive frames)
    - Semantic segmentation maps
    - Depth maps
    - Policy intervention masks
    - Train/Val split with no video overlap
    
    This is the CORE dataset that replaces the simple frame extraction in original MobilityDreamer.
    Inspired by CityDreamer4D's dataset structure but adapted for street-level dash-cam video.
    """
    
    def __init__(
        self,
        cfg,
        split="train",
        transform=None,
        load_depth=True,
        load_segmentation=True,
        load_policy=True
    ):
        """
        Initialize BDD100K dataset.
        
        Args:
            cfg: Configuration object (EasyDict from mobility_config.py)
            split: "train", "val", or "test"
            transform: Optional data augmentation transforms
            load_depth: Whether to load depth maps
            load_segmentation: Whether to load semantic segmentation
            load_policy: Whether to load policy intervention masks
        """
        super().__init__()
        
        self.cfg = cfg.DATASETS.BDD100K
        self.split = split
        self.transform = transform
        self.load_depth = load_depth
        self.load_segmentation = load_segmentation
        self.load_policy = load_policy
        
        # Paths
        self.processed_dir = Path(self.cfg.PROCESSED_DIR)
        self.sequences_file = self.processed_dir / f"{split}_sequences.json"
        
        # Load sequence index
        if not self.sequences_file.exists():
            raise FileNotFoundError(
                f"Sequence index not found: {self.sequences_file}\n"
                f"Please run preprocessing script first:\n"
                f"python scripts/preprocess_bdd100k.py"
            )
        
        with open(self.sequences_file, 'r') as f:
            self.sequences = json.load(f)
        
        logger.info(f"Loaded {len(self.sequences)} sequences for {split} split")
        
        # Dataset statistics
        self.n_classes = self.cfg.N_CLASSES
        self.n_policy_classes = self.cfg.POLICY_CLASSES
        self.sequence_length = self.cfg.SEQUENCE_LENGTH
        self.image_size = self.cfg.IMAGE_SIZE  # (H, W)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a temporal sequence with all associated data.
        
        Returns:
            data: Dictionary containing:
                - frames: (T, 3, H, W) RGB frames
                - frames_next: (T, 3, H, W) future frames (for temporal prediction)
                - segmentation: (T, C_sem, H, W) semantic segmentation (one-hot)
                - policy: (T, C_policy, H, W) policy intervention masks (one-hot)
                - depth: (T, 1, H, W) depth maps (if load_depth=True)
                - sequence_id: str, unique sequence identifier
                - video_id: str, source video identifier
        """
        sequence_info = self.sequences[idx]
        sequence_id = sequence_info['sequence_id']
        video_id = sequence_info['video_id']
        frame_indices = sequence_info['frame_indices']
        
        # Initialize containers
        frames = []
        frames_next = []  # For temporal prediction (next frame)
        segmentations = []
        policies = []
        depths = []
        
        # Load sequence data
        for i, frame_idx in enumerate(frame_indices):
            # SIMPLIFIED STRUCTURE: Load from flat directories (data/frames, data/masks, etc.)
            # Format: frame_0000.jpg, frame_0000_mask.png, policy_0000.jpg
            
            # Load RGB frame
            frame_path = Path(self.cfg.FRAMES_DIR) / f"frame_{frame_idx:04d}.jpg"
            frame = self._load_image(frame_path)
            frames.append(frame)
            
            # Load next frame (for temporal supervision)
            if i < len(frame_indices) - 1:
                next_frame_path = Path(self.cfg.FRAMES_DIR) / f"frame_{frame_indices[i+1]:04d}.jpg"
                next_frame = self._load_image(next_frame_path)
                frames_next.append(next_frame)
            else:
                # For last frame, repeat itself
                frames_next.append(frame)
            
            # Load segmentation
            if self.load_segmentation:
                seg_path = Path(self.cfg.MASKS_DIR) / f"frame_{frame_idx:04d}_mask.png"
                seg = self._load_segmentation(seg_path)
                segmentations.append(seg)
            
            # Load policy
            if self.load_policy:
                policy_path = Path(self.cfg.POLICY_DIR) / f"policy_{frame_idx:04d}.jpg"
                policy = self._load_policy(policy_path)
                policies.append(policy)
            
            # Load depth (optional)
            if self.load_depth:
                depth_path = Path(self.cfg.DEPTH_DIR) / f"depth_{frame_idx:04d}.png"
                if not depth_path.exists():
                    # Fallback: no depth maps yet
                    depth = torch.zeros(1, *self.image_size)
                else:
                    depth = self._load_depth(depth_path)
                depths.append(depth)
        
        # Stack into tensors
        frames = torch.stack(frames, dim=0)  # (T, 3, H, W)
        frames_next = torch.stack(frames_next, dim=0)  # (T, 3, H, W)
        
        # Build output dictionary
        data = {
            'frames': frames,
            'frames_next': frames_next,
            'sequence_id': sequence_id,
            'video_id': video_id,
            'frame_indices': torch.tensor(frame_indices, dtype=torch.long)
        }
        
        if self.load_segmentation:
            data['segmentation'] = torch.stack(segmentations, dim=0)  # (T, C_sem, H, W)
        
        if self.load_policy:
            data['policy'] = torch.stack(policies, dim=0)  # (T, C_policy, H, W)
        
        if self.load_depth:
            data['depth'] = torch.stack(depths, dim=0)  # (T, 1, H, W)
        
        # Apply transforms (data augmentation)
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    def _load_image(self, path: Path) -> torch.Tensor:
        """
        Load RGB image and convert to tensor.
        
        Args:
            path: Path to image file
            
        Returns:
            tensor: (3, H, W) normalized to [-1, 1]
        """
        if not path.exists():
            logger.warning(f"Image not found: {path}, using black image")
            img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        else:
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize if necessary
            if img.shape[:2] != self.image_size:
                img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        
        # Convert to tensor and normalize to [-1, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # (3, H, W)
        img_tensor = (img_tensor / 255.0) * 2.0 - 1.0  # [-1, 1]
        
        return img_tensor
    
    def _load_segmentation(self, path: Path) -> torch.Tensor:
        """
        Load semantic segmentation and convert to one-hot encoding.
        
        Args:
            path: Path to segmentation map (grayscale PNG with class IDs)
            
        Returns:
            tensor: (C_sem, H, W) one-hot encoded
        """
        if not path.exists():
            logger.warning(f"Segmentation not found: {path}, using zeros")
            seg = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
        else:
            seg = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            
            # Resize if necessary
            if seg.shape[:2] != self.image_size:
                seg = cv2.resize(seg, (self.image_size[1], self.image_size[0]), 
                               interpolation=cv2.INTER_NEAREST)
        
        # Convert to one-hot (C_sem, H, W)
        seg_one_hot = np.zeros((self.n_classes, *self.image_size), dtype=np.float32)
        for c in range(self.n_classes):
            seg_one_hot[c] = (seg == c).astype(np.float32)
        
        return torch.from_numpy(seg_one_hot)
    
    def _load_policy(self, path: Path) -> torch.Tensor:
        """
        Load policy intervention mask and convert to one-hot encoding.
        
        Args:
            path: Path to policy map (grayscale PNG with policy type IDs)
            
        Returns:
            tensor: (C_policy, H, W) one-hot encoded
        """
        if not path.exists():
            # No policy intervention (all zeros except class 0)
            policy_one_hot = np.zeros((self.n_policy_classes, *self.image_size), dtype=np.float32)
            policy_one_hot[0] = 1.0  # "none" class
            return torch.from_numpy(policy_one_hot)
        
        policy = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        
        # Resize if necessary
        if policy.shape[:2] != self.image_size:
            policy = cv2.resize(policy, (self.image_size[1], self.image_size[0]),
                              interpolation=cv2.INTER_NEAREST)
        
        # Convert to one-hot
        policy_one_hot = np.zeros((self.n_policy_classes, *self.image_size), dtype=np.float32)
        for c in range(self.n_policy_classes):
            policy_one_hot[c] = (policy == c).astype(np.float32)
        
        return torch.from_numpy(policy_one_hot)
    
    def _load_depth(self, path: Path) -> torch.Tensor:
        """
        Load depth map and normalize.
        
        Args:
            path: Path to depth map (grayscale PNG, 0-255)
            
        Returns:
            tensor: (1, H, W) normalized to [0, 1]
        """
        if not path.exists():
            logger.warning(f"Depth not found: {path}, using zeros")
            depth = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
        else:
            depth = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            
            # Resize if necessary
            if depth.shape[:2] != self.image_size:
                depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]))
        
        # Normalize to [0, 1]
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float() / 255.0  # (1, H, W)
        
        return depth_tensor
    
    def get_n_classes(self):
        """Return number of semantic classes."""
        return self.n_classes
    
    def get_n_policy_classes(self):
        """Return number of policy intervention types."""
        return self.n_policy_classes


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching temporal sequences.
    
    Args:
        batch: List of data dictionaries from __getitem__
        
    Returns:
        batched_data: Dictionary with batched tensors
    """
    batched_data = {}
    
    # Get all keys from first sample
    keys = batch[0].keys()
    
    for key in keys:
        if key in ['sequence_id', 'video_id']:
            # Keep as list of strings
            batched_data[key] = [sample[key] for sample in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            # Stack tensors
            batched_data[key] = torch.stack([sample[key] for sample in batch], dim=0)
        else:
            # Keep as list for other types
            batched_data[key] = [sample[key] for sample in batch]
    
    return batched_data


if __name__ == "__main__":
    # Test dataset loading
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from config.mobility_config import cfg
    
    print("Testing BDD100K dataset loader...")
    
    # Create dataset
    try:
        dataset = BDD100KUrbanDataset(cfg, split="train")
        print(f"✅ Dataset created successfully")
        print(f"   Number of sequences: {len(dataset)}")
        print(f"   Sequence length: {dataset.sequence_length}")
        print(f"   Image size: {dataset.image_size}")
        print(f"   Semantic classes: {dataset.n_classes}")
        print(f"   Policy classes: {dataset.n_policy_classes}")
        
        # Load one sample
        print("\n📦 Loading sample data...")
        sample = dataset[0]
        
        print("Sample data keys:", list(sample.keys()))
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}, dtype={value.dtype}, "
                      f"range=[{value.min():.3f}, {value.max():.3f}]")
            else:
                print(f"  {key}: {value}")
        
        print("\n✅ Dataset test completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("\nTo create the dataset:")
        print("1. Run preprocessing: python scripts/preprocess_bdd100k.py")
        print("2. This will generate the required sequence index files")
