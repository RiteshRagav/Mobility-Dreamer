"""
Depth Estimation Module - MiDaS Integration
============================================
Uses MiDaS (Mixed Data Adaptation and Scaling) for monocular depth estimation.

Provides 3D scene understanding for ControlNet conditioning, improving generation quality.

Usage:
    python src/depth_midas.py --frames data/frames --out data/depth
    python src/depth_midas.py --frames data/frames --out data/depth --model DPT_Large
"""

import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def load_midas_model(model_type="DPT_Hybrid"):
    """
    Load MiDaS model from torch hub.
    
    Args:
        model_type: Model variant - "DPT_Large" (best quality, slow), 
                   "DPT_Hybrid" (balanced), or "MiDaS_small" (fast)
    
    Returns:
        model: Loaded MiDaS model
        transform: Input preprocessing transform
        device: CPU or CUDA device
    """
    print(f"📦 Loading MiDaS model: {model_type}")
    
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load model from torch hub
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()
    
    # Load appropriate transform
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    
    print(f"✅ Model loaded successfully")
    return midas, transform, device


def estimate_depth(model, transform, device, image):
    """
    Estimate depth map for a single image.
    
    Args:
        model: MiDaS model
        transform: Preprocessing transform
        device: Computation device
        image: Input image (BGR, uint8)
    
    Returns:
        depth_map: Normalized depth map (0-255, uint8)
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    input_batch = transform(img_rgb).to(device)
    
    # Predict depth
    with torch.no_grad():
        prediction = model(input_batch)
        
        # Resize to original resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convert to numpy
    depth = prediction.cpu().numpy()
    
    # Normalize to 0-255
    depth_min = depth.min()
    depth_max = depth.max()
    
    if depth_max - depth_min > 0:
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth)
    
    depth_map = (depth_normalized * 255).astype(np.uint8)
    
    return depth_map


def process_frames(frames_dir, output_dir, model_type="DPT_Hybrid"):
    """
    Process all frames in directory and save depth maps.
    
    Args:
        frames_dir: Directory containing input frames
        output_dir: Directory to save depth maps
        model_type: MiDaS model variant
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all frames
    frame_files = sorted(frames_dir.glob("frame_*.jpg")) + sorted(frames_dir.glob("frame_*.png"))
    
    if len(frame_files) == 0:
        print(f"❌ No frames found in {frames_dir}")
        return
    
    print(f"🎬 Found {len(frame_files)} frames")
    
    # Load model once
    model, transform, device = load_midas_model(model_type)
    
    # Process each frame
    print(f"\n🔍 Estimating depth maps...")
    
    for frame_path in tqdm(frame_files, desc="Processing frames"):
        # Read frame
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"⚠️  Could not read {frame_path.name}")
            continue
        
        # Estimate depth
        depth_map = estimate_depth(model, transform, device, image)
        
        # Save depth map
        depth_name = frame_path.name.replace("frame_", "depth_")
        depth_path = output_dir / depth_name
        cv2.imwrite(str(depth_path), depth_map)
    
    print(f"\n✅ Done! Depth maps saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate depth maps using MiDaS")
    parser.add_argument("--frames", type=str, required=True, help="Input frames directory")
    parser.add_argument("--out", type=str, required=True, help="Output depth maps directory")
    parser.add_argument("--model", type=str, default="DPT_Hybrid",
                        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
                        help="MiDaS model variant (default: DPT_Hybrid)")
    args = parser.parse_args()
    
    process_frames(args.frames, args.out, args.model)


if __name__ == "__main__":
    main()
