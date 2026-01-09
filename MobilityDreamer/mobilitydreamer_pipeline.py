"""
MobilityDreamer - Complete Production Pipeline
===============================================
End-to-end orchestration for urban mobility future visualization.

Chains all components:
1. Frame extraction (from video or BDD100K)
2. Semantic segmentation (YOLOv8)
3. Mask refinement (optional SAM/GrabCut)
4. Policy map creation (GUI or synthetic)
5. Depth estimation (MiDaS)
6. Future generation (ControlNet or simplified)
7. Video composition

Usage:
    # Run with default config
    python mobilitydr eamer_pipeline.py --config config/default.yaml
    
    # Run with custom settings
    python mobilitydreamer_pipeline.py --config config/custom.yaml --mode full
    
    # Quick demo mode
    python mobilitydreamer_pipeline.py --mode demo
"""

import yaml
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import logging


class MobilityDreamerPipeline:
    """
    Production pipeline orchestrator.
    """
    
    def __init__(self, config_path=None, mode="full"):
        self.mode = mode
        self.config = self.load_config(config_path)
        self.setup_logging()
        
    def load_config(self, config_path):
        """Load configuration from YAML file and normalize keys."""
        raw_cfg = {}
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                raw_cfg = yaml.safe_load(f) or {}
        
        # Defaults for legacy pipeline keys (backward compatible)
        legacy_defaults = {
            "input": {
                "source": "bdd100k",
                "path": None,
                "num_videos": 2,
                "frames_per_video": 8
            },
            "segmentation": {
                "model": "yolov8n-seg.pt",
                "confidence": 0.25,
                "save_overlay": True
            },
            "policy": {
                "method": "synthetic",
                "gui_port": 7860
            },
            "depth": {
                "enabled": True,
                "model": "DPT_Hybrid"
            },
            "generation": {
                "method": "controlnet",
                "prompt": "photorealistic urban street with sustainable infrastructure",
                "strength": 0.75,
                "num_steps": 20
            },
            "output": {
                "fps": 10,
                "save_comparisons": True
            },
            "paths": {
                "frames": "data/frames",
                "masks": "data/masks",
                "masks_refined": "data/masks_refined",
                "policy_maps": "data/policy_maps",
                "depth": "data/depth",
                "generated": "data/generated_frames",
                "results": "results/pipeline"
            }
        }
        
        # If YAML already provides legacy keys, merge and return
        if "paths" in raw_cfg:
            merged = legacy_defaults
            merged.update(raw_cfg)
            return merged
        
        # Otherwise, map new config schema (config/default.yaml) into legacy shape
        dataset = raw_cfg.get("dataset", {})
        segmentation = raw_cfg.get("segmentation", {})
        policy = raw_cfg.get("policy", {})
        depth = raw_cfg.get("depth", {})
        generation = raw_cfg.get("generation", {})
        composition = raw_cfg.get("composition", {})
        pipeline_cfg = raw_cfg.get("pipeline", {})
        hardware = raw_cfg.get("hardware", {})
        
        # Normalized configuration used by pipeline
        cfg = {
            "input": {
                "source": dataset.get("type", "bdd100k"),
                "path": dataset.get("video_dir"),
                "num_videos": dataset.get("num_videos", 2),
                "frames_per_video": dataset.get("frames_per_video", 8)
            },
            "segmentation": {
                "model": segmentation.get("model_path", "yolov8n-seg.pt"),
                "confidence": segmentation.get("confidence_threshold", 0.25),
                "save_overlay": segmentation.get("save_visualizations", True)
            },
            "policy": {
                "method": policy.get("mode", "auto") if policy else "synthetic",
                "gui_port": policy.get("gui_port", 7860)
            },
            "depth": {
                "enabled": depth.get("enabled", True),
                "model": depth.get("model_type", "DPT_Hybrid")
            },
            "generation": {
                "method": "controlnet" if generation.get("mode", "controlnet") == "controlnet" else "simplified",
                "prompt": generation.get("prompts", {}).get("base", "photorealistic urban street with sustainable infrastructure"),
                "strength": generation.get("blending", {}).get("policy_blend_strength", 0.75),
                "num_steps": generation.get("controlnet", {}).get("num_inference_steps", 20)
            },
            "output": {
                "fps": composition.get("fps", 10),
                "save_comparisons": composition.get("create_comparison", True)
            },
            "paths": {
                "frames": dataset.get("frames_output_dir", "data/frames"),
                "masks": segmentation.get("masks_output_dir", "data/masks"),
                "masks_refined": "data/masks_refined",
                "policy_maps": policy.get("policy_output_dir", "data/policy_maps"),
                "depth": depth.get("depth_output_dir", "data/depth_maps"),
                "generated": generation.get("output_dir", "data/generated_frames"),
                "results": composition.get("output_dir", "results/")
            },
            "pipeline": pipeline_cfg,
            "hardware": hardware
        }
        
        return cfg
    
    def setup_logging(self):
        """Setup logging to file and console with UTF-8 encoding."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"
        
        # Create handlers with UTF-8 encoding for emoji support on Windows
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        console_handler = logging.StreamHandler()
        
        # Set format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Pipeline initialized in {self.mode} mode")
        self.logger.info(f"Log file: {log_file}")
    
    def run_step(self, command, description, critical=True):
        """
        Execute a pipeline step.
        
        Args:
            command: List of command arguments
            description: Step description
            critical: Whether failure should stop pipeline
        
        Returns:
            bool: Success status
        """
        self.logger.info(f"Starting: {description}")
        self.logger.info(f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            self.logger.info(f"[SUCCESS] Completed: {description}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"[FAILED] {description}")
            self.logger.error(f"Error: {e.stderr}")
            
            if critical:
                raise
            return False
    
    def run(self):
        """Execute complete pipeline."""
        self.logger.info("="*60)
        self.logger.info("MobilityDreamer Pipeline Execution Started")
        self.logger.info("="*60)
        
        paths = self.config["paths"]
        
        # Step 1: Extract frames (if needed)
        if self.config["input"]["source"] == "bdd100k" and self.config["input"]["path"]:
            self.run_step(
                ["python", "src/extract_bdd100k_frames.py",
                 "--video-dir", self.config["input"]["path"],
                 "--out", paths["frames"],
                 "--num-videos", str(self.config["input"]["num_videos"]),
                 "--frames-per-video", str(self.config["input"]["frames_per_video"])],
                "Frame Extraction from BDD100K"
            )
        
        # Step 2: Segmentation
        seg_cmd = ["python", "src/segmentation_yolo.py",
                   "--frames", paths["frames"],
                   "--out", paths["masks"]]
        if self.config["segmentation"]["save_overlay"]:
            seg_cmd.append("--save-overlay")
        
        self.run_step(seg_cmd, "Semantic Segmentation (YOLOv8)")
        
        # Step 3: Policy maps
        if self.config["policy"]["method"] == "synthetic":
            self.run_step(
                ["python", "src/create_policy_maps.py",
                 "--frames", paths["frames"],
                 "--out", paths["policy_maps"]],
                "Policy Map Generation (Synthetic)"
            )
        elif self.config["policy"]["method"] == "gui":
            self.logger.info("[GUI] Launching policy GUI - edit policies and press Ctrl+C when done")
            self.run_step(
                ["python", "src/policy_gui.py",
                 "--frames", paths["frames"],
                 "--out", paths["policy_maps"],
                 "--port", str(self.config["policy"]["gui_port"])],
                "Policy Map Creation (GUI)",
                critical=False
            )
        
        # Step 4: Depth estimation (if enabled)
        if self.config["depth"]["enabled"]:
            self.run_step(
                ["python", "src/depth_midas.py",
                 "--frames", paths["frames"],
                 "--out", paths["depth"],
                 "--model", self.config["depth"]["model"]],
                "Depth Estimation (MiDaS)",
                critical=False
            )
        
        # Step 5: Future generation
        gen_config = self.config["generation"]
        
        if gen_config["method"] == "controlnet":
            gen_cmd = ["python", "src/generate_future.py",
                       "--frames", paths["frames"],
                       "--policy-maps", paths["policy_maps"],
                       "--out", paths["generated"],
                       "--strength", str(gen_config["strength"]),
                       "--steps", str(gen_config["num_steps"])]
            
            if self.config["depth"]["enabled"]:
                gen_cmd.extend(["--depth", paths["depth"]])
            
            if gen_config.get("prompt"):
                gen_cmd.extend(["--prompt", gen_config["prompt"]])
            
            if self.config["output"]["save_comparisons"]:
                gen_cmd.append("--save-comparison")
            
            self.run_step(gen_cmd, "Future Scenario Generation (ControlNet)")
        
        else:  # simplified
            self.run_step(
                ["python", "src/generate_simple.py",
                 "--frames", paths["frames"],
                 "--policy-maps", paths["policy_maps"],
                 "--out", paths["generated"],
                 "--save-comparison"],
                "Future Scenario Generation (Simplified)"
            )
        
        # Step 6: Video composition
        results_dir = Path(paths["results"])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        final_video = results_dir / "mobility_future.mp4"
        
        self.run_step(
            ["python", "src/compose_video.py",
             "--frames", paths["generated"],
             "--out", str(final_video),
             "--fps", str(self.config["output"]["fps"])],
            "Video Composition"
        )
        
        # Save pipeline metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode,
            "config": self.config,
            "output_video": str(final_video)
        }
        
        metadata_path = results_dir / "pipeline_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info("="*60)
        self.logger.info("[SUCCESS] Pipeline Execution Completed Successfully")
        self.logger.info(f"Results: {results_dir}")
        self.logger.info(f"Video: {final_video}")
        self.logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="MobilityDreamer Production Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--mode", type=str, default="full", 
                        choices=["full", "demo", "quick"],
                        help="Pipeline mode")
    args = parser.parse_args()
    
    pipeline = MobilityDreamerPipeline(args.config, args.mode)
    pipeline.run()


if __name__ == "__main__":
    main()
