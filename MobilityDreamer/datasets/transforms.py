# -*- coding: utf-8 -*-
"""
Lightweight transform utilities for MobilityDreamer sequences.
Applies spatial and photometric augmentations consistently across
frames, segmentation masks, policy masks, and depth.
"""
import random
from typing import Dict, Tuple

import torch
import torchvision.transforms.functional as TF


def _random_crop(t: torch.Tensor, top: int, left: int, h: int, w: int) -> torch.Tensor:
    # t is (..., H, W) or (..., C, H, W)
    if t.dim() == 3:
        return t[:, top : top + h, left : left + w]
    if t.dim() == 4:
        return t[:, :, top : top + h, left : left + w]
    return t


def random_crop(data: Dict[str, torch.Tensor], crop_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
    h, w = crop_size
    _, _, H, W = data["frames"].shape  # frames: (B,T,3,H,W) after stacking in collate? Actually dataset returns (T,3,H,W)
    # Handle both (T,3,H,W) and (B,T,3,H,W) by cropping last two dims
    H_src, W_src = data["frames"].shape[-2:]
    if H_src < h or W_src < w:
        return data  # skip if too small
    top = random.randint(0, H_src - h)
    left = random.randint(0, W_src - w)

    def crop_tensor(x):
        if x is None:
            return None
        if x.dim() >= 3:
            return _random_crop(x, top, left, h, w)
        return x

    keys = ["frames", "frames_next", "segmentation", "policy", "depth"]
    for k in keys:
        if k in data:
            data[k] = crop_tensor(data[k])
    return data


def horizontal_flip(data: Dict[str, torch.Tensor], p: float = 0.5) -> Dict[str, torch.Tensor]:
    if random.random() > p:
        return data
    keys = ["frames", "frames_next", "segmentation", "policy", "depth"]
    for k in keys:
        if k in data and data[k] is not None:
            # flip last dimension (width)
            data[k] = torch.flip(data[k], dims=[-1])
    return data


def color_jitter(data: Dict[str, torch.Tensor], p: float = 0.3, brightness=0.1, contrast=0.1, saturation=0.1) -> Dict[str, torch.Tensor]:
    if "frames" not in data or random.random() > p:
        return data

    def jitter_frame(frame: torch.Tensor) -> torch.Tensor:
        # frame shape: (3, H, W), values in [-1, 1]
        f = (frame + 1.0) / 2.0  # to [0,1]
        f = TF.adjust_brightness(f, 1.0 + random.uniform(-brightness, brightness))
        f = TF.adjust_contrast(f, 1.0 + random.uniform(-contrast, contrast))
        f = TF.adjust_saturation(f, 1.0 + random.uniform(-saturation, saturation))
        f = torch.clamp(f, 0.0, 1.0)
        return f * 2.0 - 1.0

    if data["frames"].dim() == 4:  # (T,3,H,W)
        data["frames"] = torch.stack([jitter_frame(f) for f in data["frames"]], dim=0)
    elif data["frames"].dim() == 5:  # (B,T,3,H,W)
        b, t = data["frames"].shape[:2]
        data["frames"] = torch.stack(
            [torch.stack([jitter_frame(data["frames"][b_i, t_i]) for t_i in range(t)], dim=0) for b_i in range(b)],
            dim=0,
        )
    return data


def normalize(data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Frames already in [-1,1]; ensure dtype float
    for k in ["frames", "frames_next"]:
        if k in data:
            data[k] = data[k].float()
    return data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for t in self.transforms:
            data = t(data)
        return data


class CropTransform:
    def __init__(self, crop_size):
        self.crop_size = crop_size
    def __call__(self, data):
        return random_crop(data, self.crop_size)

class FlipTransform:
    def __init__(self, p):
        self.p = p
    def __call__(self, data):
        return horizontal_flip(data, self.p)

class JitterTransform:
    def __init__(self, p):
        self.p = p
    def __call__(self, data):
        return color_jitter(data, self.p)

def build_default_transforms(cfg=None):
    crop_size = (384, 512)
    t_list = [
        CropTransform(crop_size),
        FlipTransform(0.5),
        JitterTransform(0.3),
        normalize,
    ]
    return Compose(t_list)
