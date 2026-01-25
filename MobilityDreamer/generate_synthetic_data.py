#!/usr/bin/env python
"""Generate synthetic frame data for smoke test."""
import numpy as np
import cv2
from pathlib import Path

root = Path('.')
frames_dir = root / 'data' / 'frames'
masks_dir = root / 'data' / 'masks'
policy_dir = root / 'data' / 'policy_maps'
depth_dir = root / 'data' / 'depth_maps'
for d in [frames_dir, masks_dir, policy_dir, depth_dir]:
    d.mkdir(parents=True, exist_ok=True)

H, W = 256, 256
n_frames = 22
rng = np.random.default_rng(0)

for idx in range(n_frames):
    # RGB frame: gradient + noise
    base = np.linspace(0, 255, W, dtype=np.uint8)
    frame = np.tile(base, (H, 1))
    frame = np.stack([frame, np.roll(frame, idx*3, axis=1), np.roll(frame, idx*5, axis=1)], axis=2)
    noise = rng.normal(0, 5, size=frame.shape).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(frames_dir / f'frame_{idx:04d}.jpg'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Segmentation: vertical stripes (classes 0-3)
    seg = np.zeros((H, W), dtype=np.uint8)
    seg[:, :W//3] = 0
    seg[:, W//3:2*W//3] = 1
    seg[:, 2*W//3:] = 2 + (idx % 2)
    cv2.imwrite(str(masks_dir / f'frame_{idx:04d}_mask.png'), seg)
    
    # Policy: small rectangle of intervention
    pol = np.zeros((H, W), dtype=np.uint8)
    x1, y1 = 40 + (idx % 10), 40 + (idx % 7)
    x2, y2 = min(W-1, x1 + 40), min(H-1, y1 + 30)
    pol[y1:y2, x1:x2] = 1
    cv2.imwrite(str(policy_dir / f'policy_{idx:04d}.jpg'), pol)
    
    # Depth: smooth gradient
    depth = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
    cv2.imwrite(str(depth_dir / f'depth_{idx:04d}.png'), depth)

print(f'✅ Synthetic data generated: {n_frames} frames, {H}x{W}')
