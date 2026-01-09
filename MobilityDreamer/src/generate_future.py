"""
ControlNet-based Future Scenario Generation
============================================
Uses Stable Diffusion + ControlNet to generate photorealistic future scenarios.

This is the CORE innovation of MobilityDreamer: transforming policy interventions
into realistic visualizations of how cities could look with infrastructure changes.

Requirements:
    pip install diffusers transformers accelerate torch torchvision

Usage:
    # Basic usage with policy maps
    python src/generate_future.py --frames data/frames --policy-maps data/policy_maps --out data/generated_frames
    
    # With depth conditioning for better quality
    python src/generate_future.py --frames data/frames --policy-maps data/policy_maps --depth data/depth --out data/generated_frames
    
    # Advanced: custom prompt and parameters
    python src/generate_future.py --frames data/frames --policy-maps data/policy_maps --out data/generated_frames --prompt "urban street with bike lanes and pedestrian zones" --strength 0.7
"""

import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json

# Check if diffusers is installed
try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    CONTROLNET_AVAILABLE = True
except ImportError:
    CONTROLNET_AVAILABLE = False
    print("⚠️  diffusers not installed. Install with: pip install diffusers transformers accelerate")


def setup_pipeline(use_depth=False):
    """
    Initialize Stable Diffusion + ControlNet pipeline.
    
    Args:
        use_depth: Whether to use depth conditioning
        
    Returns:
        pipe: Configured pipeline
        device: Computation device
    """
    if not CONTROLNET_AVAILABLE:
        raise ImportError("diffusers library required. Install with: pip install diffusers transformers accelerate")
    
    print("📦 Loading ControlNet + Stable Diffusion models...")
    print("   This may take a few minutes on first run (downloading ~5-10GB)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    if device.type == "cpu":
        print("⚠️  Running on CPU - this will be VERY slow (2-5 minutes per frame)")
        print("   Consider using Google Colab with GPU for faster generation")
    
    # Load ControlNet model (Canny edge detection)
    # We use Canny because it works well with structural changes like roads/lanes
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    
    # Load Stable Diffusion pipeline with ControlNet
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None  # Disable for speed
    )
    
    # Optimize for speed
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    # Enable memory optimizations if on GPU
    if device.type == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
    
    print("✅ Pipeline loaded successfully\n")
    return pipe, device


def create_control_image(frame, policy_map):
    """
    Create control image by combining frame structure with policy overlay.
    
    Uses Canny edge detection to preserve scene structure while highlighting
    policy intervention areas.
    
    Args:
        frame: Original frame (BGR)
        policy_map: Policy intervention map (BGR)
        
    Returns:
        control_image: PIL Image for ControlNet conditioning
    """
    # Convert to grayscale for Canny
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect edges with Canny
    edges = cv2.Canny(gray, 50, 150)
    
    # Find policy regions (non-black pixels)
    policy_mask = np.any(policy_map > 10, axis=2).astype(np.uint8) * 255
    
    # Enhance edges in policy regions
    edges_enhanced = np.where(policy_mask > 0, 255, edges)
    
    # Convert to PIL Image (RGB)
    edges_rgb = cv2.cvtColor(edges_enhanced, cv2.COLOR_GRAY2RGB)
    control_pil = Image.fromarray(edges_rgb)
    
    return control_pil


def generate_future_frame(pipe, frame, policy_map, depth_map=None, prompt=None, strength=0.75, num_steps=20):
    """
    Generate future scenario for a single frame.
    
    Args:
        pipe: Stable Diffusion + ControlNet pipeline
        frame: Original frame (BGR numpy array)
        policy_map: Policy intervention map (BGR numpy array)
        depth_map: Optional depth map for better conditioning
        prompt: Text prompt describing desired changes
        strength: How much to modify (0.5-1.0, higher = more change)
        num_steps: Inference steps (more = better quality but slower)
        
    Returns:
        generated: Generated future frame (BGR numpy array)
    """
    # Create control image
    control_image = create_control_image(frame, policy_map)
    
    # Default prompt if not provided
    if prompt is None:
        prompt = "photorealistic urban street scene with new bike lanes, pedestrian zones, and green infrastructure, daytime, clear weather, high quality"
    
    # Negative prompt to avoid unwanted elements
    negative_prompt = "blurry, low quality, distorted, unrealistic, cartoon, anime, painting"
    
    # Generate
    output = pipe(
        prompt=prompt,
        image=control_image,
        num_inference_steps=num_steps,
        controlnet_conditioning_scale=strength,
        negative_prompt=negative_prompt,
        guidance_scale=7.5
    )
    
    # Convert PIL to numpy BGR
    generated_rgb = np.array(output.images[0])
    generated_bgr = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR)
    
    return generated_bgr


def blend_with_original(original, generated, policy_map, blend_ratio=0.6):
    """
    Blend generated frame with original in non-policy regions for smoother result.
    
    Args:
        original: Original frame
        generated: Generated frame
        policy_map: Policy map indicating change areas
        blend_ratio: How much of generated to use in policy regions (0-1)
        
    Returns:
        blended: Blended result
    """
    # Create mask for policy regions
    policy_mask = np.any(policy_map > 10, axis=2).astype(np.float32)
    policy_mask = cv2.GaussianBlur(policy_mask, (21, 21), 0)
    policy_mask = np.stack([policy_mask] * 3, axis=2)
    
    # Blend: use generated in policy areas, original elsewhere
    blended = (
        generated * policy_mask * blend_ratio +
        original * policy_mask * (1 - blend_ratio) +
        original * (1 - policy_mask)
    ).astype(np.uint8)
    
    return blended


def process_all_frames(frames_dir, policy_dir, output_dir, depth_dir=None, 
                       prompt=None, strength=0.75, num_steps=20, save_comparison=False):
    """
    Process all frames with ControlNet generation.
    
    Args:
        frames_dir: Input frames directory
        policy_dir: Policy maps directory
        output_dir: Output directory for generated frames
        depth_dir: Optional depth maps directory
        prompt: Custom text prompt
        strength: ControlNet conditioning strength
        num_steps: Number of inference steps
        save_comparison: Whether to save side-by-side comparisons
    """
    frames_dir = Path(frames_dir)
    policy_dir = Path(policy_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_comparison:
        comparison_dir = Path("results/comparisons_controlnet")
        comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all frames
    frame_files = sorted(frames_dir.glob("frame_*.jpg")) + sorted(frames_dir.glob("frame_*.png"))
    
    if len(frame_files) == 0:
        print(f"❌ No frames found in {frames_dir}")
        return
    
    print(f"🎬 Found {len(frame_files)} frames to process")
    print(f"⚙️  Settings: strength={strength}, steps={num_steps}")
    
    # Setup pipeline
    pipe, device = setup_pipeline(use_depth=depth_dir is not None)
    
    # Process each frame
    print(f"\n🎨 Generating future scenarios...")
    
    metadata = {
        "prompt": prompt,
        "strength": strength,
        "num_steps": num_steps,
        "device": str(device),
        "frames_processed": []
    }
    
    for frame_path in tqdm(frame_files, desc="Generating frames"):
        try:
            # Load frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"⚠️  Could not read {frame_path.name}")
                continue
            
            # Load policy map
            policy_path = policy_dir / frame_path.name.replace("frame_", "policy_")
            if not policy_path.exists():
                policy_path = policy_dir / frame_path.name
            
            if not policy_path.exists():
                print(f"⚠️  No policy map for {frame_path.name}, skipping")
                continue
            
            policy_map = cv2.imread(str(policy_path))
            
            # Load depth if available
            depth_map = None
            if depth_dir:
                depth_path = Path(depth_dir) / frame_path.name.replace("frame_", "depth_")
                if depth_path.exists():
                    depth_map = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
            
            # Generate future frame
            generated = generate_future_frame(
                pipe, frame, policy_map, depth_map,
                prompt=prompt,
                strength=strength,
                num_steps=num_steps
            )
            
            # Blend with original for smoother result
            final = blend_with_original(frame, generated, policy_map)
            
            # Save
            out_path = output_dir / frame_path.name.replace("frame_", "generated_")
            cv2.imwrite(str(out_path), final)
            
            # Save comparison if requested
            if save_comparison:
                h, w = frame.shape[:2]
                comparison = np.zeros((h, w * 3 + 40, 3), dtype=np.uint8)
                comparison[:, :w] = frame
                comparison[:, w+10:w*2+10] = policy_map
                comparison[:, w*2+20:w*3+20] = final
                
                comp_path = comparison_dir / f"controlnet_comparison_{frame_path.stem}.jpg"
                cv2.imwrite(str(comp_path), comparison)
            
            metadata["frames_processed"].append(frame_path.name)
            
        except Exception as e:
            print(f"❌ Error processing {frame_path.name}: {e}")
            continue
    
    # Save metadata
    metadata_path = output_dir / "generation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Done! Generated {len(metadata['frames_processed'])} frames")
    print(f"   Saved to: {output_dir}")
    if save_comparison:
        print(f"   Comparisons: {comparison_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate future scenarios with ControlNet")
    parser.add_argument("--frames", type=str, required=True, help="Input frames directory")
    parser.add_argument("--policy-maps", type=str, required=True, help="Policy maps directory")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--depth", type=str, default=None, help="Optional depth maps directory")
    parser.add_argument("--prompt", type=str, default=None, help="Custom text prompt")
    parser.add_argument("--strength", type=float, default=0.75, help="ControlNet strength (0.5-1.0)")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps (10-50)")
    parser.add_argument("--save-comparison", action="store_true", help="Save side-by-side comparisons")
    args = parser.parse_args()
    
    process_all_frames(
        args.frames,
        args.policy_maps,
        args.out,
        args.depth,
        args.prompt,
        args.strength,
        args.steps,
        args.save_comparison
    )


if __name__ == "__main__":
    main()
