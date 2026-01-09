"""
==============================================================================
MobilityDreamer - Infrastructure Metrics Evaluation
==============================================================================

This module quantifies the impact of policy interventions by comparing
original and generated frames. Computes metrics for presentations and reports.

METRICS COMPUTED:
-----------------
1. Infrastructure Area Coverage (%)
   - Total pixel area covered by each intervention type
   - Percentage of frame occupied by bike lanes, pedestrian zones, etc.

2. Visibility Scores
   - Color saliency (how visible interventions are)
   - Contrast with background

3. Density Metrics
   - Number of intervention regions per frame
   - Spatial distribution patterns

4. Change Detection
   - Pixel-level differences between original and generated
   - Semantic change maps

USAGE:
------
Basic:
    python src/evaluate_metrics.py \\
        --original data/frames/ \\
        --generated data/generated_frames/ \\
        --policy-maps data/policy_maps/ \\
        --output results/metrics/

With visualization:
    python src/evaluate_metrics.py \\
        --original data/frames/ \\
        --generated data/generated_frames/ \\
        --policy-maps data/policy_maps/ \\
        --output results/metrics/ \\
        --create-visualizations \\
        --export-csv results/metrics.csv

OUTPUTS:
--------
- metrics_summary.json: Aggregated statistics
- metrics_per_frame.csv: Frame-by-frame breakdown
- visualizations/: Heatmaps and comparison images (optional)
- infrastructure_report.txt: Human-readable summary

For 20% Python knowledge users: Copy-paste the commands above!
==============================================================================
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ==============================================================================
# POLICY COLOR DEFINITIONS
# ==============================================================================

POLICY_COLORS = {
    "bike_lane": np.array([0, 255, 0]),       # Green
    "pedestrian_zone": np.array([0, 100, 255]),  # Blue
    "ev_station": np.array([255, 150, 0]),    # Orange
    "green_space": np.array([255, 100, 200])  # Pink
}


# ==============================================================================
# AREA COVERAGE METRICS
# ==============================================================================

def compute_area_coverage(
    policy_map: np.ndarray,
    intervention_type: str,
    color_tolerance: int = 30
) -> Dict[str, float]:
    """
    Compute pixel area covered by a specific intervention type.
    
    WHAT THIS DOES:
    ---------------
    - Counts how many pixels match the intervention color
    - Calculates percentage of total frame area
    - Returns area in pixels and percentage
    
    PARAMETERS:
    -----------
    policy_map : np.ndarray
        RGB image with colored policy interventions
    intervention_type : str
        Type of intervention ("bike_lane", "pedestrian_zone", etc.)
    color_tolerance : int
        RGB distance tolerance for color matching (0-255)
    
    RETURNS:
    --------
    dict : {
        "area_pixels": int,      # Total pixels covered
        "area_percentage": float,  # % of frame
        "color": list              # RGB color used
    }
    
    EXAMPLE OUTPUT:
    ---------------
    {
        "area_pixels": 152400,
        "area_percentage": 15.24,
        "color": [0, 255, 0]
    }
    """
    
    # Get target color for this intervention type
    target_color = POLICY_COLORS.get(intervention_type, np.array([0, 0, 0]))
    
    # Calculate color distance for each pixel
    # Formula: Euclidean distance in RGB space
    color_diff = np.sqrt(np.sum((policy_map - target_color) ** 2, axis=2))
    
    # Create mask where pixels match target color (within tolerance)
    intervention_mask = color_diff < color_tolerance
    
    # Count matching pixels
    area_pixels = int(np.sum(intervention_mask))
    
    # Calculate percentage
    total_pixels = policy_map.shape[0] * policy_map.shape[1]
    area_percentage = (area_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
    
    return {
        "area_pixels": area_pixels,
        "area_percentage": round(area_percentage, 2),
        "color": target_color.tolist()
    }


# ==============================================================================
# VISIBILITY METRICS
# ==============================================================================

def compute_visibility_score(
    policy_map: np.ndarray,
    original_frame: np.ndarray,
    intervention_type: str,
    color_tolerance: int = 30
) -> Dict[str, float]:
    """
    Compute how visible the intervention is relative to background.
    
    VISIBILITY MEASURES:
    --------------------
    1. Color Saliency: How much intervention color stands out
    2. Contrast: Difference in brightness with surrounding areas
    
    PARAMETERS:
    -----------
    policy_map : np.ndarray
        Policy intervention map (RGB)
    original_frame : np.ndarray
        Original video frame (RGB)
    intervention_type : str
        Type of intervention
    color_tolerance : int
        Color matching tolerance
    
    RETURNS:
    --------
    dict : {
        "saliency_score": float,   # 0-100 (higher = more visible)
        "contrast_score": float,   # 0-100 (higher = more contrast)
        "visibility_score": float  # Average of saliency and contrast
    }
    """
    
    # Get intervention mask
    target_color = POLICY_COLORS.get(intervention_type, np.array([0, 0, 0]))
    color_diff = np.sqrt(np.sum((policy_map - target_color) ** 2, axis=2))
    intervention_mask = color_diff < color_tolerance
    
    if not np.any(intervention_mask):
        # No intervention pixels found
        return {
            "saliency_score": 0.0,
            "contrast_score": 0.0,
            "visibility_score": 0.0
        }
    
    # Compute saliency: difference between intervention color and original frame
    intervention_pixels = policy_map[intervention_mask]
    original_pixels = original_frame[intervention_mask]
    
    # Calculate average color difference (saliency)
    color_distance = np.mean(np.sqrt(np.sum((intervention_pixels - original_pixels) ** 2, axis=1)))
    saliency_score = min(100, (color_distance / 255) * 100)  # Normalize to 0-100
    
    # Compute contrast: brightness difference
    intervention_gray = cv2.cvtColor(policy_map, cv2.COLOR_RGB2GRAY)
    original_gray = cv2.cvtColor(original_frame, cv2.COLOR_RGB2GRAY)
    
    intervention_brightness = np.mean(intervention_gray[intervention_mask])
    original_brightness = np.mean(original_gray[intervention_mask])
    
    contrast = abs(intervention_brightness - original_brightness)
    contrast_score = min(100, (contrast / 255) * 100)  # Normalize to 0-100
    
    # Overall visibility
    visibility_score = (saliency_score + contrast_score) / 2
    
    return {
        "saliency_score": round(saliency_score, 2),
        "contrast_score": round(contrast_score, 2),
        "visibility_score": round(visibility_score, 2)
    }


# ==============================================================================
# DENSITY METRICS
# ==============================================================================

def compute_density_metrics(
    policy_map: np.ndarray,
    intervention_type: str,
    color_tolerance: int = 30
) -> Dict[str, int]:
    """
    Compute spatial distribution metrics (number of regions, clustering).
    
    WHAT THIS DOES:
    ---------------
    - Counts distinct intervention regions (connected components)
    - Calculates average region size
    - Determines spatial distribution
    
    PARAMETERS:
    -----------
    policy_map : np.ndarray
        Policy intervention map (RGB)
    intervention_type : str
        Type of intervention
    color_tolerance : int
        Color matching tolerance
    
    RETURNS:
    --------
    dict : {
        "num_regions": int,         # Count of separate intervention areas
        "avg_region_size": float,   # Average pixels per region
        "largest_region_size": int  # Size of largest intervention area
    }
    """
    
    # Get intervention mask
    target_color = POLICY_COLORS.get(intervention_type, np.array([0, 0, 0]))
    color_diff = np.sqrt(np.sum((policy_map - target_color) ** 2, axis=2))
    intervention_mask = (color_diff < color_tolerance).astype(np.uint8) * 255
    
    # Find connected components (distinct regions)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        intervention_mask, connectivity=8
    )
    
    # Exclude background (label 0)
    num_regions = num_labels - 1
    
    if num_regions == 0:
        return {
            "num_regions": 0,
            "avg_region_size": 0,
            "largest_region_size": 0
        }
    
    # Calculate region sizes (exclude background)
    region_sizes = stats[1:, cv2.CC_STAT_AREA]  # Area column
    
    avg_region_size = float(np.mean(region_sizes))
    largest_region_size = int(np.max(region_sizes))
    
    return {
        "num_regions": int(num_regions),
        "avg_region_size": round(avg_region_size, 2),
        "largest_region_size": largest_region_size
    }


# ==============================================================================
# CHANGE DETECTION
# ==============================================================================

def compute_change_metrics(
    original_frame: np.ndarray,
    generated_frame: np.ndarray
) -> Dict[str, float]:
    """
    Quantify differences between original and generated frames.
    
    CHANGE METRICS:
    ---------------
    1. MSE (Mean Squared Error): Overall pixel difference
    2. PSNR (Peak Signal-to-Noise Ratio): Image quality metric (higher = better)
    3. SSIM (Structural Similarity): Perceptual similarity (0-1, higher = better)
    
    PARAMETERS:
    -----------
    original_frame : np.ndarray
        Original video frame (RGB)
    generated_frame : np.ndarray
        AI-generated future frame (RGB)
    
    RETURNS:
    --------
    dict : {
        "mse": float,           # Mean Squared Error (lower = more similar)
        "psnr": float,          # Peak Signal-to-Noise Ratio (higher = better quality)
        "changed_pixels_pct": float  # % of pixels that changed significantly
    }
    """
    
    # Resize if shapes don't match
    if original_frame.shape != generated_frame.shape:
        generated_frame = cv2.resize(
            generated_frame,
            (original_frame.shape[1], original_frame.shape[0])
        )
    
    # Compute MSE
    mse = float(np.mean((original_frame.astype(float) - generated_frame.astype(float)) ** 2))
    
    # Compute PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Compute percentage of significantly changed pixels (threshold: 30 RGB units)
    pixel_diff = np.sqrt(np.sum((original_frame.astype(float) - generated_frame.astype(float)) ** 2, axis=2))
    changed_pixels = np.sum(pixel_diff > 30)
    total_pixels = original_frame.shape[0] * original_frame.shape[1]
    changed_pixels_pct = (changed_pixels / total_pixels) * 100
    
    return {
        "mse": round(mse, 2),
        "psnr": round(psnr, 2),
        "changed_pixels_pct": round(changed_pixels_pct, 2)
    }


# ==============================================================================
# FRAME-LEVEL EVALUATION
# ==============================================================================

def evaluate_frame(
    original_path: str,
    generated_path: str,
    policy_map_path: str,
    frame_name: str
) -> Dict:
    """
    Compute all metrics for a single frame.
    
    PARAMETERS:
    -----------
    original_path : str
        Path to original frame image
    generated_path : str
        Path to generated frame image
    policy_map_path : str
        Path to policy map image
    frame_name : str
        Name identifier for the frame
    
    RETURNS:
    --------
    dict : Complete metrics for this frame (all intervention types + changes)
    """
    
    # Load images
    original = cv2.imread(original_path)
    generated = cv2.imread(generated_path)
    policy_map = cv2.imread(policy_map_path)
    
    if original is None or generated is None or policy_map is None:
        print(f"⚠️  Warning: Could not load images for {frame_name}")
        return None
    
    # Convert BGR to RGB
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB)
    policy_map = cv2.cvtColor(policy_map, cv2.COLOR_BGR2RGB)
    
    # Initialize results
    frame_metrics = {
        "frame_name": frame_name,
        "interventions": {}
    }
    
    # Compute metrics for each intervention type
    for intervention_type in POLICY_COLORS.keys():
        area_metrics = compute_area_coverage(policy_map, intervention_type)
        
        # Only compute visibility/density if intervention exists
        if area_metrics["area_pixels"] > 0:
            visibility_metrics = compute_visibility_score(
                policy_map, original, intervention_type
            )
            density_metrics = compute_density_metrics(policy_map, intervention_type)
            
            frame_metrics["interventions"][intervention_type] = {
                **area_metrics,
                **visibility_metrics,
                **density_metrics
            }
    
    # Compute change metrics
    frame_metrics["change_metrics"] = compute_change_metrics(original, generated)
    
    return frame_metrics


# ==============================================================================
# DATASET-LEVEL EVALUATION
# ==============================================================================

def evaluate_dataset(
    original_dir: str,
    generated_dir: str,
    policy_maps_dir: str,
    output_dir: str,
    create_visualizations: bool = False,
    export_csv: str = None
) -> Dict:
    """
    Evaluate all frames in dataset and generate summary report.
    
    PARAMETERS:
    -----------
    original_dir : str
        Directory with original frames
    generated_dir : str
        Directory with generated frames
    policy_maps_dir : str
        Directory with policy maps
    output_dir : str
        Directory to save results
    create_visualizations : bool
        Generate heatmaps and comparison images
    export_csv : str
        Path to export CSV file (optional)
    
    RETURNS:
    --------
    dict : Aggregated metrics across all frames
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of frames
    original_frames = sorted([f for f in os.listdir(original_dir) if f.endswith(('.jpg', '.png'))])
    
    print(f"📊 Evaluating {len(original_frames)} frames...")
    
    # Process each frame
    all_frame_metrics = []
    for frame_file in tqdm(original_frames, desc="Computing metrics"):
        original_path = os.path.join(original_dir, frame_file)
        generated_path = os.path.join(generated_dir, frame_file)
        policy_map_path = os.path.join(policy_maps_dir, frame_file)
        
        # Check if all files exist
        if not os.path.exists(generated_path):
            print(f"⚠️  Missing generated frame: {frame_file}")
            continue
        if not os.path.exists(policy_map_path):
            print(f"⚠️  Missing policy map: {frame_file}")
            continue
        
        frame_metrics = evaluate_frame(
            original_path, generated_path, policy_map_path, frame_file
        )
        
        if frame_metrics:
            all_frame_metrics.append(frame_metrics)
    
    # Aggregate metrics
    summary = aggregate_metrics(all_frame_metrics)
    
    # Save results
    save_results(summary, all_frame_metrics, output_dir, export_csv)
    
    # Create visualizations if requested
    if create_visualizations:
        print("🎨 Creating visualizations...")
        create_metric_visualizations(all_frame_metrics, output_dir)
    
    print(f"✅ Evaluation complete! Results saved to: {output_dir}")
    return summary


# ==============================================================================
# AGGREGATION
# ==============================================================================

def aggregate_metrics(frame_metrics_list: List[Dict]) -> Dict:
    """
    Aggregate frame-level metrics into dataset-level summary.
    
    COMPUTES:
    ---------
    - Average area coverage per intervention type
    - Average visibility scores
    - Total infrastructure added
    - Change statistics
    
    RETURNS:
    --------
    dict : Summary statistics for the entire dataset
    """
    
    if not frame_metrics_list:
        return {}
    
    summary = {
        "total_frames": len(frame_metrics_list),
        "interventions_summary": {},
        "change_summary": {}
    }
    
    # Aggregate intervention metrics
    for intervention_type in POLICY_COLORS.keys():
        area_percentages = []
        visibility_scores = []
        num_regions_list = []
        
        for frame_metrics in frame_metrics_list:
            if intervention_type in frame_metrics.get("interventions", {}):
                metrics = frame_metrics["interventions"][intervention_type]
                area_percentages.append(metrics["area_percentage"])
                visibility_scores.append(metrics["visibility_score"])
                num_regions_list.append(metrics["num_regions"])
        
        if area_percentages:
            summary["interventions_summary"][intervention_type] = {
                "avg_area_percentage": round(np.mean(area_percentages), 2),
                "max_area_percentage": round(np.max(area_percentages), 2),
                "avg_visibility_score": round(np.mean(visibility_scores), 2),
                "avg_num_regions": round(np.mean(num_regions_list), 2),
                "frames_with_intervention": len(area_percentages)
            }
    
    # Aggregate change metrics
    mse_list = [fm["change_metrics"]["mse"] for fm in frame_metrics_list]
    psnr_list = [fm["change_metrics"]["psnr"] for fm in frame_metrics_list if fm["change_metrics"]["psnr"] != float('inf')]
    changed_pct_list = [fm["change_metrics"]["changed_pixels_pct"] for fm in frame_metrics_list]
    
    summary["change_summary"] = {
        "avg_mse": round(np.mean(mse_list), 2),
        "avg_psnr": round(np.mean(psnr_list), 2) if psnr_list else float('inf'),
        "avg_changed_pixels_pct": round(np.mean(changed_pct_list), 2)
    }
    
    return summary


# ==============================================================================
# SAVE RESULTS
# ==============================================================================

def save_results(
    summary: Dict,
    frame_metrics: List[Dict],
    output_dir: str,
    csv_path: str = None
):
    """
    Save metrics to JSON, CSV, and human-readable report.
    
    CREATES:
    --------
    1. metrics_summary.json: Aggregated statistics
    2. metrics_per_frame.csv: Frame-by-frame breakdown (if csv_path provided)
    3. infrastructure_report.txt: Human-readable summary
    """
    
    # Save JSON summary
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📄 Saved summary: {summary_path}")
    
    # Save per-frame CSV if requested
    if csv_path or len(frame_metrics) > 0:
        csv_output_path = csv_path or os.path.join(output_dir, "metrics_per_frame.csv")
        
        # Flatten frame metrics into rows
        rows = []
        for fm in frame_metrics:
            row = {"frame_name": fm["frame_name"]}
            
            # Add change metrics
            for key, value in fm["change_metrics"].items():
                row[f"change_{key}"] = value
            
            # Add intervention metrics
            for intervention_type, metrics in fm.get("interventions", {}).items():
                for key, value in metrics.items():
                    row[f"{intervention_type}_{key}"] = value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_output_path, index=False)
        print(f"📊 Saved CSV: {csv_output_path}")
    
    # Save human-readable report
    report_path = os.path.join(output_dir, "infrastructure_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MOBILITYDREAMER - INFRASTRUCTURE IMPACT REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Frames Analyzed: {summary['total_frames']}\n\n")
        
        f.write("INFRASTRUCTURE COVERAGE:\n")
        f.write("-" * 80 + "\n")
        for intervention_type, metrics in summary.get("interventions_summary", {}).items():
            f.write(f"\n{intervention_type.replace('_', ' ').title()}:\n")
            f.write(f"  • Average Area Coverage: {metrics['avg_area_percentage']}%\n")
            f.write(f"  • Maximum Coverage: {metrics['max_area_percentage']}%\n")
            f.write(f"  • Average Visibility Score: {metrics['avg_visibility_score']}/100\n")
            f.write(f"  • Frames with Intervention: {metrics['frames_with_intervention']}/{summary['total_frames']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("VISUAL CHANGE ANALYSIS:\n")
        f.write("-" * 80 + "\n")
        change_summary = summary.get("change_summary", {})
        f.write(f"  • Average MSE: {change_summary.get('avg_mse', 'N/A')}\n")
        f.write(f"  • Average PSNR: {change_summary.get('avg_psnr', 'N/A')} dB\n")
        f.write(f"  • Average Pixels Changed: {change_summary.get('avg_changed_pixels_pct', 'N/A')}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"📝 Saved report: {report_path}")


# ==============================================================================
# VISUALIZATION (OPTIONAL)
# ==============================================================================

def create_metric_visualizations(frame_metrics: List[Dict], output_dir: str):
    """
    Create visual charts and heatmaps (requires matplotlib).
    
    NOTE: This function is optional and requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("⚠️  Matplotlib not installed. Skipping visualizations.")
        print("   Install with: pip install matplotlib")
        return
    
    viz_dir = os.path.join(output_dir, "visualizations")
    Path(viz_dir).mkdir(exist_ok=True)
    
    # Example: Bar chart of average area coverage per intervention
    intervention_types = list(POLICY_COLORS.keys())
    avg_areas = []
    
    for intervention_type in intervention_types:
        areas = [
            fm["interventions"][intervention_type]["area_percentage"]
            for fm in frame_metrics
            if intervention_type in fm.get("interventions", {})
        ]
        avg_areas.append(np.mean(areas) if areas else 0)
    
    plt.figure(figsize=(10, 6))
    colors_hex = ['#00FF00', '#0064FF', '#FF9600', '#FF64C8']  # Matching policy colors
    plt.bar(
        [it.replace('_', ' ').title() for it in intervention_types],
        avg_areas,
        color=colors_hex
    )
    plt.ylabel("Average Area Coverage (%)")
    plt.title("Infrastructure Coverage by Intervention Type")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "area_coverage_chart.png"), dpi=150)
    plt.close()
    
    print(f"📈 Saved visualization: {os.path.join(viz_dir, 'area_coverage_chart.png')}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MobilityDreamer infrastructure metrics"
    )
    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Path to original frames directory"
    )
    parser.add_argument(
        "--generated",
        type=str,
        required=True,
        help="Path to generated frames directory"
    )
    parser.add_argument(
        "--policy-maps",
        type=str,
        required=True,
        help="Path to policy maps directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/metrics/",
        help="Output directory for metrics (default: results/metrics/)"
    )
    parser.add_argument(
        "--create-visualizations",
        action="store_true",
        help="Generate charts and heatmaps (requires matplotlib)"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Export per-frame metrics to CSV file"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    summary = evaluate_dataset(
        original_dir=args.original,
        generated_dir=args.generated,
        policy_maps_dir=args.policy_maps,
        output_dir=args.output,
        create_visualizations=args.create_visualizations,
        export_csv=args.export_csv
    )
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY:")
    print("=" * 80)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
