"""
Simplified Future Scenario Generation
======================================
Uses basic image processing and blending to create "future" frames.
This is a DEMO VERSION - not AI-based, but shows the concept quickly!

How it works:
1. Loads original frame
2. Loads policy map
3. Blends them together with color adjustments
4. Applies some image processing to make it look modified
5. Saves as "generated future frame"

Usage:
    python src/generate_simple.py --frames data/frames --policy-maps data/policy_maps --out data/generated_frames
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def apply_policy_to_frame(frame, policy_map):
    """
    Blend the policy map onto the frame to simulate infrastructure changes.
    
    This is a simplified version that uses image processing instead of AI.
    Real version would use Stable Diffusion ControlNet, but this shows the concept.
    
    Args:
        frame: Original BGR frame
        policy_map: Policy intervention map (colored shapes)
        
    Returns:
        modified_frame: Frame with policy interventions "applied"
    """
    # Make a copy to modify
    result = frame.copy().astype(np.float32)
    policy_float = policy_map.astype(np.float32)
    
    # Create a mask of where policy changes exist (non-black pixels)
    policy_mask = np.any(policy_map > 10, axis=2).astype(np.float32)
    policy_mask = cv2.GaussianBlur(policy_mask, (21, 21), 0)  # Smooth edges
    policy_mask = np.stack([policy_mask] * 3, axis=2)  # Make it 3-channel
    
    # Blend the policy colors onto the frame
    # This simulates "adding infrastructure" by colorizing those areas
    blend_strength = 0.4  # How strong the policy overlay appears
    result = result * (1 - policy_mask * blend_strength) + policy_float * policy_mask * blend_strength
    
    # Enhance areas with policy interventions to make them look "new"
    where_policy = policy_mask[:, :, 0] > 0.1
    
    # Increase brightness and saturation in policy areas (makes them "pop")
    hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1][where_policy] = np.clip(hsv[:, :, 1][where_policy] * 1.2, 0, 255)  # Saturation
    hsv[:, :, 2][where_policy] = np.clip(hsv[:, :, 2][where_policy] * 1.1, 0, 255)  # Brightness
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    
    # Add slight sharpening to make changes more visible
    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5,  5.0, -0.5],
                       [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(result.astype(np.uint8), -1, kernel)
    result = cv2.addWeighted(result.astype(np.uint8), 0.7, sharpened, 0.3, 0)
    
    return result.astype(np.uint8)


def create_before_after_comparison(original, generated, policy_map):
    """
    Create a side-by-side comparison image showing original | policy | result.
    
    Args:
        original: Original frame
        generated: Generated future frame
        policy_map: Policy intervention map
        
    Returns:
        comparison: Wide image with all three
    """
    h, w = original.shape[:2]
    
    # Create a canvas for side-by-side display
    comparison = np.zeros((h, w * 3 + 40, 3), dtype=np.uint8)
    
    # Place images with spacing
    comparison[:, :w] = original
    comparison[:, w+10:w*2+10] = policy_map
    comparison[:, w*2+20:w*3+20] = generated
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Policy Map", (w+20, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Future Scenario", (w*2+30, 30), font, 1, (255, 255, 255), 2)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Generate simplified future scenario frames")
    parser.add_argument("--frames", type=str, required=True, help="Path to original frames")
    parser.add_argument("--policy-maps", type=str, required=True, help="Path to policy maps")
    parser.add_argument("--out", type=str, required=True, help="Output directory for generated frames")
    parser.add_argument("--save-comparison", action="store_true", help="Save side-by-side comparison images")
    args = parser.parse_args()
    
    frames_dir = Path(args.frames)
    policy_dir = Path(args.policy_maps)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_comparison:
        comparison_dir = Path("results/comparisons")
        comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all frames
    frame_files = sorted(frames_dir.glob("frame_*.jpg")) + sorted(frames_dir.glob("frame_*.png"))
    
    if len(frame_files) == 0:
        print(f"❌ No frames found in {frames_dir}")
        return
    
    print(f"🎨 Generating future scenarios for {len(frame_files)} frames...")
    print(f"⚠️  NOTE: This is a SIMPLIFIED demo using image processing, not AI generation")
    print(f"   Real version would use Stable Diffusion ControlNet (much slower but higher quality)\n")
    
    for frame_path in tqdm(frame_files, desc="Processing frames"):
        # Load original frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        # Find matching policy map
        policy_path = policy_dir / frame_path.name.replace("frame_", "policy_")
        if not policy_path.exists():
            # Try alternative names
            policy_path = policy_dir / frame_path.name
        
        if not policy_path.exists():
            print(f"⚠️  No policy map found for {frame_path.name}, skipping")
            continue
        
        policy_map = cv2.imread(str(policy_path))
        
        # Generate future frame
        generated = apply_policy_to_frame(frame, policy_map)
        
        # Save generated frame
        out_path = out_dir / frame_path.name.replace("frame_", "generated_")
        cv2.imwrite(str(out_path), generated)
        
        # Optionally save comparison
        if args.save_comparison:
            comparison = create_before_after_comparison(frame, generated, policy_map)
            comp_path = comparison_dir / f"comparison_{frame_path.stem}.jpg"
            cv2.imwrite(str(comp_path), comparison)
    
    print(f"\n✅ Done! Generated frames saved to {out_dir}")
    if args.save_comparison:
        print(f"📊 Comparison images saved to {comparison_dir}")
    
    print(f"\n💡 Next step: Create video with:")
    print(f"   python src/compose_video.py --frames {out_dir} --out results/demo_future.mp4")


if __name__ == "__main__":
    main()
