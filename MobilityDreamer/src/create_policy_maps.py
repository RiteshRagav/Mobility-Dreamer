"""
Create Synthetic Policy Maps for Demo
======================================
This script generates fake policy intervention maps with colored shapes.
No manual drawing needed - runs automatically!

What it creates:
- Green rectangles = Bike lanes
- Blue circles = Pedestrian zones  
- Orange polygons = EV charging stations
- Pink areas = Green spaces

Usage:
    python src/create_policy_maps.py --frames data/frames --out data/policy_maps
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import random


def create_random_policy_map(frame_shape, frame_number):
    """
    Create a synthetic policy map with random colored shapes.
    
    Args:
        frame_shape: (height, width, channels) of the original frame
        frame_number: Which frame number this is (affects randomness)
        
    Returns:
        policy_map: RGB image with colored policy interventions
    """
    height, width = frame_shape[:2]
    
    # Start with black background (no interventions)
    policy_map = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Set random seed based on frame number for consistency
    random.seed(42 + frame_number)
    
    # 1. Add bike lane (green horizontal strip in lower third)
    # Position changes slightly per frame to simulate movement
    bike_lane_y = int(height * 0.7) + (frame_number * 2) % 20
    bike_lane_height = 40
    cv2.rectangle(
        policy_map,
        (0, bike_lane_y),
        (width, bike_lane_y + bike_lane_height),
        (0, 255, 0),  # Green color (BGR format)
        -1  # Filled rectangle
    )
    
    # 2. Add pedestrian zone (blue rectangle on sidewalk)
    ped_x = int(width * 0.15)
    ped_y = int(height * 0.5)
    ped_width = int(width * 0.2)
    ped_height = int(height * 0.3)
    cv2.rectangle(
        policy_map,
        (ped_x, ped_y),
        (ped_x + ped_width, ped_y + ped_height),
        (255, 100, 0),  # Blue
        -1
    )
    
    # 3. Add EV charging stations (orange circles)
    # Position varies per frame to show "planning multiple locations"
    for i in range(2):
        circle_x = int(width * (0.4 + i * 0.3))
        circle_y = int(height * 0.6) + random.randint(-20, 20)
        circle_radius = 30
        cv2.circle(
            policy_map,
            (circle_x, circle_y),
            circle_radius,
            (0, 165, 255),  # Orange
            -1
        )
    
    # 4. Add green space (pink polygon)
    green_space_points = np.array([
        [int(width * 0.7), int(height * 0.4)],
        [int(width * 0.85), int(height * 0.35)],
        [int(width * 0.9), int(height * 0.6)],
        [int(width * 0.75), int(height * 0.65)]
    ], dtype=np.int32)
    
    cv2.fillPoly(policy_map, [green_space_points], (180, 105, 255))  # Pink
    
    # 5. Add slight transparency effect (optional - makes it look more realistic)
    # Convert to float for blending, then back to uint8
    policy_map = (policy_map * 0.7).astype(np.uint8)
    
    return policy_map


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic policy intervention maps")
    parser.add_argument("--frames", type=str, required=True, help="Path to frames directory")
    parser.add_argument("--out", type=str, required=True, help="Output directory for policy maps")
    args = parser.parse_args()
    
    frames_dir = Path(args.frames)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all frame images
    frame_files = sorted(frames_dir.glob("frame_*.jpg")) + sorted(frames_dir.glob("frame_*.png"))
    
    if len(frame_files) == 0:
        print(f"❌ No frames found in {frames_dir}")
        return
    
    print(f"📊 Creating policy maps for {len(frame_files)} frames...")
    
    for idx, frame_path in enumerate(frame_files):
        # Load frame to get dimensions
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"⚠️  Skipping {frame_path.name} (could not read)")
            continue
        
        # Generate policy map
        policy_map = create_random_policy_map(frame.shape, idx)
        
        # Save with matching name
        out_path = out_dir / frame_path.name.replace("frame_", "policy_")
        cv2.imwrite(str(out_path), policy_map)
        
        if (idx + 1) % 5 == 0:
            print(f"  ✓ Created {idx + 1}/{len(frame_files)} policy maps")
    
    print(f"✅ Done! Policy maps saved to {out_dir}")
    print(f"\n💡 Legend:")
    print(f"   🟢 Green = Bike lanes")
    print(f"   🔵 Blue = Pedestrian zones")
    print(f"   🟠 Orange circles = EV charging stations")
    print(f"   🟣 Pink = Green spaces")


if __name__ == "__main__":
    main()
