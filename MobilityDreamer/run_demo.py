"""
MobilityDreamer - Quick Demo Runner
====================================
Runs the complete simplified demo pipeline in one command!

This script chains together:
1. Frame extraction (if needed)
2. YOLOv8 segmentation
3. Mask refinement
4. Policy map generation (synthetic)
5. Future frame generation (simplified)
6. Video composition

Usage:
    python run_demo.py
    
    Or with custom paths:
    python run_demo.py --frames data/frames --out results/demo
"""

import subprocess
import argparse
from pathlib import Path
import sys


def run_command(cmd, description):
    """
    Run a shell command and display progress.
    
    Args:
        cmd: List of command arguments
        description: What this step does
    """
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {description} - DONE\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found. Make sure you're in the correct directory and environment.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run MobilityDreamer demo pipeline")
    parser.add_argument("--frames", type=str, default=None, 
                        help="Path to input frames")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for final results")
    parser.add_argument("--skip-segmentation", action="store_true",
                        help="Skip segmentation if already done")
    parser.add_argument("--skip-policy", action="store_true",
                        help="Skip policy map creation if already done")
    parser.add_argument("--extract-bdd100k", type=str, default=None,
                        help="Extract frames from BDD100K (provide path to BDD100K videos dir)")
    parser.add_argument("--num-bdd-videos", type=int, default=2,
                        help="How many BDD100K videos to extract from (default: 2)")
    parser.add_argument("--frames-per-bdd-video", type=int, default=8,
                        help="Frames per BDD100K video (default: 8)")
    args = parser.parse_args()
    
    # Auto-detect if we're in parent dir or MobilityDreamer dir
    cwd = Path.cwd()
    is_in_mobility_dir = cwd.name == "MobilityDreamer"
    
    # Set default paths based on location
    if is_in_mobility_dir:
        default_frames = "data/frames"
        default_out = "results/demo"
        default_masks = "data/masks"
        default_policy = "data/policy_maps"
        default_generated = "data/generated_frames"
    else:
        default_frames = "MobilityDreamer/data/frames"
        default_out = "MobilityDreamer/results/demo"
        default_masks = "MobilityDreamer/data/masks"
        default_policy = "MobilityDreamer/data/policy_maps"
        default_generated = "MobilityDreamer/data/generated_frames"
    
    # Use provided paths or defaults
    frames_path = Path(args.frames or default_frames)
    out_path = Path(args.out or default_out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Define all paths
    masks_path = Path(default_masks)
    masks_refined_path = Path(default_masks.replace("masks", "masks_refined"))
    policy_path = Path(default_policy)
    generated_path = Path(default_generated)
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║          MobilityDreamer - Quick Demo Pipeline              ║
    ║                                                              ║
    ║  Visualizing Sustainable Urban Mobility Futures             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Step 0: Extract from BDD100K if requested
    if args.extract_bdd100k:
        script_path = "src/extract_bdd100k_frames.py" if is_in_mobility_dir else "MobilityDreamer/src/extract_bdd100k_frames.py"
        success = run_command(
            ["python", script_path,
             "--video-dir", args.extract_bdd100k,
             "--out", str(frames_path),
             "--num-videos", str(args.num_bdd_videos),
             "--frames-per-video", str(args.frames_per_bdd_video)],
            "Step 0: Extract Frames from BDD100K Dataset"
        )
        if not success:
            print("⚠️  BDD100K extraction failed. Using existing frames...")
    
    # Check if frames exist
    frame_files = list(frames_path.glob("frame_*.jpg")) + list(frames_path.glob("frame_*.png"))
    if len(frame_files) == 0:
        print(f"❌ No frames found in {frames_path}")
        print(f"Please run: python MobilityDreamer/src/extract_frames.py --input VIDEO.mp4 --out {frames_path}")
        return
    
    print(f"📁 Found {len(frame_files)} frames in {frames_path}")
    
    # Step 1: YOLOv8 Segmentation
    if not args.skip_segmentation:
        seg_script = "src/segmentation_yolo.py" if is_in_mobility_dir else "MobilityDreamer/src/segmentation_yolo.py"
        success = run_command(
            ["python", seg_script, 
             "--frames", str(frames_path),
             "--out", str(masks_path),
             "--save-overlay"],
            "Step 1: YOLOv8 Semantic Segmentation"
        )
        if not success:
            print("⚠️  Segmentation failed. Continuing anyway...")
    else:
        print("⏭️  Skipping segmentation (already done)")
    
    # Step 2: Mask Refinement (optional - can skip for demo)
    # Commenting out to speed up demo
    # run_command(
    #     ["python", "MobilityDreamer/src/segmentation_sam.py",
    #      "--frames", str(frames_path),
    #      "--masks", str(masks_path),
    #      "--out", str(masks_refined_path)],
    #     "Step 2: Mask Refinement (SAM/GrabCut)"
    # )
    
    # Step 3: Generate Policy Maps
    if not args.skip_policy:
        policy_script = "src/create_policy_maps.py" if is_in_mobility_dir else "MobilityDreamer/src/create_policy_maps.py"
        success = run_command(
            ["python", policy_script,
             "--frames", str(frames_path),
             "--out", str(policy_path)],
            "Step 3: Generate Synthetic Policy Maps"
        )
        if not success:
            print("❌ Policy map generation failed. Cannot continue.")
            return
    else:
        print("⏭️  Skipping policy map creation (already done)")
    
    # Step 4: Generate Future Frames
    gen_script = "src/generate_simple.py" if is_in_mobility_dir else "MobilityDreamer/src/generate_simple.py"
    success = run_command(
        ["python", gen_script,
         "--frames", str(frames_path),
         "--policy-maps", str(policy_path),
         "--out", str(generated_path),
         "--save-comparison"],
        "Step 4: Generate Future Scenario Frames"
    )
    if not success:
        print("❌ Frame generation failed. Cannot continue.")
        return
    
    # Step 5: Create Before/After Video
    compose_script = "src/compose_video.py" if is_in_mobility_dir else "MobilityDreamer/src/compose_video.py"
    final_video = out_path / "future_scenario_demo.mp4"
    success = run_command(
        ["python", compose_script,
         "--frames", str(generated_path),
         "--out", str(final_video),
         "--fps", "10"],
        "Step 5: Compose Final Video"
    )
    if not success:
        print("⚠️  Video creation failed, but generated frames are available")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🎉 DEMO COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📊 Results:")
    print(f"   • Input source: {'BDD100K dataset' if args.extract_bdd100k else 'local frames'}")
    print(f"   • Total frames processed: {len(frame_files)}")
    print(f"   • Segmentation masks: {masks_path}")
    print(f"   • Policy maps: {policy_path}")
    print(f"   • Generated frames: {generated_path}")
    print(f"   • Comparison images: MobilityDreamer/results/comparisons")
    print(f"   • Final video: {final_video}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Open {final_video} to see the demo")
    print(f"   2. Check MobilityDreamer/results/comparisons/ for before/after images")
    print(f"   3. For photorealistic results, implement full ControlNet pipeline")
    
    print(f"\n📝 Note: This demo uses simplified image processing.")
    print(f"   For photorealistic results, implement Stable Diffusion ControlNet generation.")


if __name__ == "__main__":
    main()
