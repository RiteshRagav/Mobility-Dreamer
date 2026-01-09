# MobilityDreamer 🌆🚴♂️🌳

**Production-Ready Framework for Visualizing Sustainable Urban Mobility Futures Using ControlNet and Computer Vision**

[![UN SDG 11](https://img.shields.io/badge/UN_SDG-11_Sustainable_Cities-brightgreen)](https://sdgs.un.org/goals/goal11)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Supporting **UN Sustainable Development Goal 11: Sustainable Cities and Communities**

---

## 🎯 Project Overview

MobilityDreamer translates high-level urban policy interventions into **photorealistic video visualizations** of future city scenarios. Using state-of-the-art ControlNet + Stable Diffusion, policymakers, urban planners, and citizens can see the real-world impact of proposed infrastructure changes (bike lanes, pedestrian zones, EV charging hubs, green spaces) **before implementation**.

### Production Pipeline (7 Stages)

```
📹 Input Video → 🖼️ Frame Extraction → 🎭 Semantic Segmentation → 
🎨 Policy Intervention → 🗺️ Depth Mapping → 🤖 AI Generation → 
🎬 Video Composition
```

**Core Innovation:** Unlike traditional urban visualization tools, MobilityDreamer uses **ControlNet conditioning** to ensure generated scenarios are:
- Structurally coherent with original geometry
- Photorealistic and publicly engaging
- Temporally consistent across video frames

---

## 📁 Project Structure

```
MobilityDreamer/
├── config/
│   └── default.yaml                 # Pipeline configuration
├── data/
│   ├── frames/                      # Extracted video frames
│   ├── masks/                       # YOLOv8 segmentation masks
│   ├── policy_maps/                 # User-defined policy interventions
│   ├── depth_maps/                  # MiDaS depth estimations
│   ├── generated_frames/            # ControlNet-generated futures
│   └── input_videos/
│       ├── bdd100k/                 # BDD100K dataset (18GB, 500+ videos)
│       └── kitti/                   # KITTI dataset
├── models/
│   ├── yolo/                        # YOLOv8 segmentation models
│   ├── sam/                         # Segment Anything (optional)
│   ├── controlnet/                  # ControlNet checkpoints (auto-downloaded)
│   └── midas/                       # MiDaS depth models (auto-downloaded)
├── results/
│   ├── before_after/                # Comparison videos
│   └── overlays/                    # Visualization outputs
├── logs/                            # Pipeline execution logs
├── src/
│   ├── extract_frames.py            # ✅ Frame extraction
│   ├── extract_bdd100k_frames.py    # ✅ BDD100K integration
│   ├── segmentation_yolo.py         # ✅ YOLOv8 segmentation
│   ├── segmentation_sam.py          # ✅ Mask refinement (SAM/GrabCut)
│   ├── policy_gui.py                # ✅ Gradio policy editor (NEW!)
│   ├── create_policy_maps.py        # ✅ Synthetic policy generation
│   ├── depth_midas.py               # ✅ MiDaS depth estimation (NEW!)
│   ├── generate_future.py           # ✅ ControlNet generation (NEW!)
│   ├── generate_simple.py           # ✅ Simplified blending (demo)
│   └── compose_video.py             # ✅ Video composition
├── mobilitydreamer_pipeline.py      # ✅ Production orchestrator (NEW!)
├── run_demo.py                      # ✅ Quick demo script
├── requirements.txt                 # Updated with ControlNet dependencies
├── BDD100K_GUIDE.md                 # Dataset integration guide
├── DEMO_GUIDE.md                    # Quick start guide
└── README.md                        # This file
```

**Legend:**
- ✅ Fully implemented and tested
- 🆕 New production features (ControlNet, MiDaS, Gradio)

---

## 🚀 What's New (Production Release)

### ✨ Production Features

1. **🤖 ControlNet-Based Generation** ([generate_future.py](src/generate_future.py))
    - Stable Diffusion 1.5 + ControlNet Canny conditioning
    - Photorealistic future scenario synthesis
    - Intervention-aware text prompting (bike lanes, EV stations, etc.)
    - Smooth blending between policy regions and original footage

2. **🗺️ MiDaS Depth Estimation** ([depth_midas.py](src/depth_midas.py))
    - DPT_Large/Hybrid/Small model support
    - 3D scene understanding for geometric coherence
    - Depth-conditioned generation (experimental)

3. **🎨 Interactive Policy GUI** ([policy_gui.py](src/policy_gui.py))
    - Gradio web interface (localhost:7860)
    - Draw interventions directly on frames
    - Real-time preview
    - Export policy maps as PNG with metadata

4. **⚙️ Production Pipeline** ([mobilitydreamer_pipeline.py](mobilitydreamer_pipeline.py))
    - YAML-based configuration
    - Step-by-step execution with checkpointing
    - Comprehensive logging
    - Error handling and recovery

5. **📊 BDD100K Dataset Support**
    - 18GB dataset with 500+ diverse urban videos
    - Automated frame extraction at configurable intervals
    - Metadata tracking

### ✅ Core Features (Already Working)
- Frame extraction from multiple video formats
- YOLOv8 semantic segmentation (vehicles, pedestrians, infrastructure)
- SAM/GrabCut mask refinement
- Before/after video composition
- Simplified generation mode (for quick demos)

---

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher (tested on 3.13)
- **RAM**: 16GB minimum (32GB recommended for ControlNet)
- **GPU**: NVIDIA GPU with 6GB+ VRAM recommended (8GB+ ideal)
   - CPU mode supported but 10-50x slower
- **Storage**: 
   - 5GB for models (auto-downloaded)
   - 20GB+ for BDD100K dataset
   - 10GB+ for project outputs

### Python Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.venv\Scripts\activate.bat

# Activate (Linux/Mac)
source .venv/bin/activate
```

---

## 📦 Installation

### Option 1: Full Installation (Production ControlNet)

**Installs everything including ControlNet, MiDaS, Gradio:**

```bash
# For GPU (NVIDIA CUDA 11.8) - RECOMMENDED
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# For CPU (slower but works)
pip install -r requirements.txt
```

**Dependencies installed:**
- PyTorch 2.0+ with CUDA support
- Diffusers 0.21+ (Stable Diffusion, ControlNet)
- Transformers 4.30+ (CLIP, text encoders)
- Ultralytics (YOLOv8)
- OpenCV, NumPy, Pillow
- Gradio 3.50+ (policy GUI)
- PyYAML (config management)
- timm (MiDaS vision transformers)

### Option 2: Minimal Installation (Demo Only)

**For quick testing without ControlNet:**

```bash
pip install opencv-python numpy Pillow ultralytics torch torchvision tqdm
```

This supports:
- Frame extraction
- YOLOv8 segmentation
- Simplified generation (blending, no AI)
- Video composition

---

## 🏃 Quick Start

### 1. Run Simplified Demo (5 minutes)

**Uses image blending instead of ControlNet for fast results:**

```bash
# From MobilityDreamer/ directory
python run_demo.py

# Or from parent directory
python MobilityDreamer/run_demo.py

# With BDD100K dataset
python run_demo.py --extract-bdd100k --num-frames 10
```

**Output:** `results/test_output.mp4` with synthetic policy overlays

### 2. Run Production Pipeline (ControlNet)

**Full AI generation with depth estimation:**

```bash
# Using default configuration
python mobilitydreamer_pipeline.py --config config/default.yaml

# Quick start (creates minimal config)
python mobilitydreamer_pipeline.py
```

**Pipeline stages:**
1. Extract frames from video → `data/frames/`
2. Segment with YOLOv8 → `data/masks/`
3. Create/load policy maps → `data/policy_maps/`
4. Estimate depth with MiDaS → `data/depth_maps/`
5. Generate with ControlNet → `data/generated_frames/`
6. Compose video → `results/`

### 3. Interactive Policy Editor

**Launch Gradio GUI to draw interventions:**

```bash
python src/policy_gui.py --frames data/frames/
```

Then open http://localhost:7860 in your browser:
- Select intervention type (bike lane, pedestrian zone, etc.)
- Draw on frames using sketch canvas
- Navigate with Prev/Next buttons
- Save policy maps with metadata

---

## 📖 Usage Examples

### Example 1: Generate Bike Lane Scenario from BDD100K

```bash
# 1. Extract 20 frames from BDD100K videos
python src/extract_bdd100k_frames.py \
   --video-dir data/input_videos/bdd100k/ \
   --output-dir data/frames/ \
   --num-frames 20

# 2. Segment vehicles and infrastructure
python src/segmentation_yolo.py \
   --frames data/frames/ \
   --output data/masks/ \
   --model yolov8n-seg.pt

# 3. Launch policy GUI to draw bike lanes
python src/policy_gui.py --frames data/frames/
# (Draw green bike lanes on frames, save)

# 4. Estimate depth maps
python src/depth_midas.py \
   --frames data/frames/ \
   --output data/depth_maps/ \
   --model DPT_Hybrid

# 5. Generate future with ControlNet
python src/generate_future.py \
   --frames data/frames/ \
   --policy-maps data/policy_maps/ \
   --depth-maps data/depth_maps/ \
   --output data/generated_frames/ \
   --num-inference-steps 30

# 6. Create before/after video
python src/compose_video.py \
   --original data/frames/ \
   --generated data/generated_frames/ \
   --output results/bike_lane_future.mp4
```

### Example 2: Quick Demo with Synthetic Policies

```bash
python run_demo.py --extract-bdd100k --num-frames 10
```

### Example 3: Production Pipeline with Custom Config

```yaml
# config/my_project.yaml
dataset:
   type: "bdd100k"
   frames_per_video: 15

generation:
   mode: "controlnet"
   controlnet:
      num_inference_steps: 50
      guidance_scale: 8.0

policy:
   mode: "manual"  # Use Gradio GUI
```

```bash
python mobilitydreamer_pipeline.py --config config/my_project.yaml
```

---

## 🔧 Configuration

All settings are defined in [config/default.yaml](config/default.yaml). Key configurations:

### Generation Modes

```yaml
generation:
   mode: "controlnet"  # or "simple" for demo
  
   controlnet:
      num_inference_steps: 20    # 20-50 recommended
      guidance_scale: 7.5         # Higher = more prompt adherence
      seed: 42                    # For reproducibility
```

### Policy Intervention

```yaml
policy:
   mode: "manual"  # Gradio GUI
   # OR
   mode: "auto"    # Synthetic generation
  
   auto:
      bike_lane_probability: 0.6
      pedestrian_zone_probability: 0.4
```

### Depth Estimation

```yaml
depth:
   enabled: true
   model_type: "DPT_Hybrid"  # DPT_Large (best), DPT_Hybrid (balanced), DPT_Small (fast)
```

### Memory Optimization (Low VRAM)

```yaml
generation:
   memory_optimization:
      enable_attention_slicing: true
      enable_vae_slicing: true
      enable_sequential_cpu_offload: true  # For <6GB VRAM
```

---

## 📊 Datasets

### Supported Datasets

1. **BDD100K** (Recommended)
    - 18GB, 500+ videos
    - Diverse US urban scenes (highways, intersections, residential)
    - Download: https://bair.berkeley.edu/blog/2018/05/30/bdd/
    - Integration: [BDD100K_GUIDE.md](BDD100K_GUIDE.md)

2. **KITTI**
    - Classic autonomous driving dataset
    - German urban/suburban scenes
    - Download: http://www.cvlibs.net/datasets/kitti/

3. **Custom Videos**
    - Any .mp4, .mov, .avi format
    - Place in `data/input_videos/custom/`

---

## 🛠️ Troubleshooting

### Issue: Out of Memory (OOM) on GPU

**Solutions:**
1. Enable memory optimizations in config:
    ```yaml
    generation:
       memory_optimization:
          enable_sequential_cpu_offload: true
    ```

2. Use smaller models:
    ```yaml
    depth:
       model_type: "DPT_Small"
   
    generation:
       controlnet:
          num_inference_steps: 15  # Reduce from 20-50
    ```

3. Reduce frame resolution:
    ```yaml
    dataset:
       max_frame_width: 1024  # Default is 1280
    ```

### Issue: ControlNet Generation Too Slow

**For GPU users:**
- Install with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Check GPU is detected: `python -c "import torch; print(torch.cuda.is_available())"`

**For CPU users:**
- Use simplified mode: `generation.mode: "simple"` in config
- OR reduce `num_inference_steps: 10`

### Issue: Policy GUI Not Launching

```bash
# Check Gradio installation
pip install --upgrade gradio

# Ensure frames exist
ls data/frames/

# Try with explicit port
python src/policy_gui.py --frames data/frames/ --port 7860
```

### Issue: ModuleNotFoundError

```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# For diffusers/transformers errors:
pip install --upgrade diffusers transformers accelerate
```

---

## 📚 Documentation

- [Quick Start Demo Guide](DEMO_GUIDE.md) - 5-minute simplified pipeline
- [BDD100K Integration Guide](BDD100K_GUIDE.md) - Dataset setup and usage
- [Configuration Reference](config/default.yaml) - All settings explained
- [QUICKSTART.md](QUICKSTART.md) - Original quick reference

---

## 🔬 Technical Details

### ControlNet Architecture

MobilityDreamer uses **lllyasviel/sd-controlnet-canny** with Stable Diffusion 1.5:

1. **Canny Edge Detection**: Extract structural edges from original frames
2. **Policy Enhancement**: Amplify edges in policy intervention regions
3. **Conditional Generation**: ControlNet preserves structure while Stable Diffusion adds policy-guided details
4. **Blending**: Smooth transition between generated policy regions and original non-policy areas

**Key Parameters:**
- `guidance_scale`: 7.5 (balances prompt adherence vs. diversity)
- `controlnet_conditioning_scale`: 1.0 (full structural preservation)
- `num_inference_steps`: 20-50 (quality vs. speed trade-off)

### MiDaS Depth Estimation

Using **Intel ISL MiDaS** via `torch.hub`:

- **DPT_Large**: Highest quality, slowest (Vision Transformer)
- **DPT_Hybrid**: Balanced (recommended)
- **DPT_Small**: Fastest, lower quality

Depth maps provide:
- 3D scene understanding
- Geometric constraints for ControlNet
- Potential depth-conditioned generation (experimental)

### YOLOv8 Segmentation

**Detected classes (COCO):**
- 0: person
- 2: car
- 3: motorcycle
- 5: bus
- 7: truck
- 9: traffic light
- 11: stop sign
- 13: bench

**Confidence threshold**: 0.5 (configurable)

---

## 🎓 For Beginners (20% Python Knowledge)

### Copy-Paste Commands

**Complete workflow from scratch:**

```bash
# 1. Setup (one-time)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 2. Download BDD100K dataset to: data/input_videos/bdd100k/

# 3. Run simplified demo (5 min)
python run_demo.py --extract-bdd100k --num-frames 10

# 4. Check output
start results\test_output.mp4

# 5. Run production pipeline (30 min - 2 hours depending on GPU)
python mobilitydreamer_pipeline.py --config config/default.yaml

# 6. Check production output
start results\mobility_future_*.mp4
```

**What each command does:**
1. Creates isolated Python environment
2. Activates environment
3. Installs PyTorch with GPU support
4. Installs all other dependencies
5. Dataset should be manually downloaded
6. Runs quick demo with synthetic policies
7. Opens demo video in default player
8. Runs full ControlNet pipeline
9. Opens production video

---

## 🚧 Roadmap

### Completed ✅
- [x] Frame extraction (KITTI, BDD100K, custom)
- [x] YOLOv8 segmentation
- [x] SAM/GrabCut refinement
- [x] Simplified generation (demo)
- [x] Video composition
- [x] ControlNet-based generation
- [x] MiDaS depth estimation
- [x] Gradio policy GUI
- [x] Production pipeline orchestrator
- [x] YAML configuration
- [x] BDD100K integration

### In Progress 🔄
- [ ] Temporal consistency improvements
- [ ] Metrics module (infrastructure area, visibility scores)
- [ ] Unit testing suite
- [ ] Performance benchmarks

### Future Enhancements 🔮
- [ ] Depth-conditioned ControlNet (ControlNet-depth model)
- [ ] Multi-intervention support (combine bike lanes + EV + green spaces)
- [ ] Video-to-video generation (temporal consistency via AnimateDiff)
- [ ] Public web demo (Hugging Face Spaces)
- [ ] Quantitative evaluation metrics
- [ ] Cloud deployment (AWS/GCP)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- **UN SDG 11** for guiding sustainable urban development
- **BDD100K** dataset by UC Berkeley
- **KITTI** dataset by Karlsruhe Institute of Technology
- **Stability AI** for Stable Diffusion
- **lvmin Zhang** for ControlNet
- **Intel ISL** for MiDaS
- **Ultralytics** for YOLOv8
- **Meta AI** for Segment Anything
- **Gradio** team for web UI framework

---

## 📧 Contact & Support

**Repository:** https://github.com/udairatinam-g/Mobility-Dreamer

**Issues:** [GitHub Issues](https://github.com/udairatinam-g/Mobility-Dreamer/issues)

**For questions about:**
- Setup and installation → Check [Troubleshooting](#-troubleshooting)
- Quick demos → See [DEMO_GUIDE.md](DEMO_GUIDE.md)
- BDD100K dataset → See [BDD100K_GUIDE.md](BDD100K_GUIDE.md)
- Advanced configuration → See [config/default.yaml](config/default.yaml)

---

## 🌟 Citation

If you use MobilityDreamer in your research or project, please cite:

```bibtex
@software{mobilitydreamer2024,
   title = {MobilityDreamer: Visualizing Sustainable Urban Futures with ControlNet},
   author = {Your Name},
   year = {2024},
   url = {https://github.com/udairatinam-g/Mobility-Dreamer}
}
```

---

**Made with ❤️ for sustainable cities | UN SDG 11: Sustainable Cities and Communities**
```bash
pip install -r requirements.txt
```

### 2. Download Models

#### YOLOv8 (Required - Automatic)
The `yolov8n-seg.pt` model is already in your project root. If you need other variants:
```bash
# Lightweight (CPU-friendly)
yolo task=segment model=yolov8n-seg.pt

# Higher accuracy
yolo task=segment model=yolov8m-seg.pt
```

#### SAM (Optional - Better Refinement)
Download from [Segment Anything Model](https://github.com/facebookresearch/segment-anything):
```bash
# Place in models/sam/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P models/sam/
```

#### MiDaS (Optional - Depth Estimation)
Will auto-download when running `depth_midas.py` (once implemented)

#### ControlNet (Required for Generation - TO IMPLEMENT)
```bash
# Download from Hugging Face (example)
# pip install huggingface_hub
# huggingface-cli download lllyasviel/ControlNet models/controlnet/
```

---

## 🎮 How to Run

### Stage 1: Extract Frames from Video

```bash
# Option 1: Use KITTI frames (already extracted)
# Your data/frames/ already contains 22 frames

# Option 2: Extract from a new video
python src/extract_frames.py --input path/to/video.mp4 --out data/frames
```

**Outputs:** `data/frames/frame_XXXX.jpg` + `metadata.json`

---

### Stage 2: Semantic Segmentation (YOLOv8)

```bash
# Run segmentation on extracted frames
python src/segmentation_yolo.py --frames data/frames --out data/masks --save-overlay

# Custom model and confidence threshold
python src/segmentation_yolo.py --frames data/frames --out data/masks --model yolov8m-seg.pt --conf 0.3
```

**Outputs:**
- `data/masks/frame_XXXX_mask.png` - Binary segmentation masks
- `results/before_after/overlays/frame_XXXX_overlay.jpg` - Colored class overlays (if --save-overlay)

---

### Stage 3: Refine Masks (SAM or GrabCut)

```bash
# With SAM (better quality, requires checkpoint)
python src/segmentation_sam.py --frames data/frames --masks data/masks --out data/masks_refined --sam-checkpoint models/sam/sam_vit_b_01ec64.pth

# Without SAM (GrabCut fallback)
python src/segmentation_sam.py --frames data/frames --masks data/masks --out data/masks_refined
```

**Outputs:** `data/masks_refined/frame_XXXX_mask.png`

---

### Stage 4: Create Policy Map (GUI) ⚠️ TO IMPLEMENT

```bash
# Launch interactive GUI to draw infrastructure changes
python src/policy_gui.py --frames data/frames --out data/policy_maps
```

**What this should do:**
- Display each frame
- Allow drawing bike lanes, pedestrian zones, green spaces
- Save policy map overlays as conditional inputs
- Export as structured data (JSON + image masks)

---

### Stage 5: Generate Depth Maps (Optional) ⚠️ TO IMPLEMENT

```bash
# Compute depth maps with MiDaS
python src/depth_midas.py --frames data/frames --out data/depth
```

**Why depth?** Helps ControlNet understand 3D scene structure for realistic modifications

---

### Stage 6: Generate Future Scenario ⚠️ TO IMPLEMENT

```bash
# Synthesize future frames using ControlNet
python src/generate_future.py --frames data/frames --policy-maps data/policy_maps --out data/generated_frames

# Optional: Include depth conditioning
python src/generate_future.py --frames data/frames --policy-maps data/policy_maps --depth data/depth --out data/generated_frames
```

**How it works:**
- ControlNet takes: original frame + policy map + (optional) depth
- Generates: photorealistic future scenario frame
- Processes all frames to maintain temporal consistency

---

### Stage 7: Compose Before/After Video

```bash
# Create side-by-side comparison video
python src/compose_video.py --frames data/frames --masks data/masks --out results/before_after/preview.mp4 --fps 10

# For generated frames (once implemented)
python src/compose_video.py --frames data/generated_frames --masks data/masks --out results/future_scenario.mp4 --fps 10
```

**Outputs:** MP4 video with side-by-side original | modified view

---

## 📊 Example Workflow

```bash
# Full pipeline (current capabilities)
python src/extract_frames.py --input data/input_videos/my_city.mp4 --out data/frames
python src/segmentation_yolo.py --frames data/frames --out data/masks --save-overlay
python src/segmentation_sam.py --frames data/frames --masks data/masks --out data/masks_refined
python src/compose_video.py --frames data/frames --masks data/masks_refined --out results/analysis.mp4

# Once implemented, the full future generation pipeline:
# python src/policy_gui.py --frames data/frames --out data/policy_maps
# python src/depth_midas.py --frames data/frames --out data/depth
# python src/generate_future.py --frames data/frames --policy-maps data/policy_maps --depth data/depth --out data/generated_frames
# python src/compose_video.py --frames data/frames --generated data/generated_frames --out results/before_after_full.mp4
```

---

## 🔧 Dependencies

Core packages needed (save as `requirements.txt`):

```
# Computer Vision
opencv-python>=4.8.0
numpy>=1.24.0

# YOLO Segmentation
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0

# Optional: SAM Refinement
# segment-anything @ git+https://github.com/facebookresearch/segment-anything.git

# Optional: Depth Estimation
# timm>=0.9.0

# Optional: ControlNet Generation (TO ADD)
# diffusers>=0.21.0
# transformers>=4.30.0
# accelerate>=0.20.0

# Utilities
Pillow>=10.0.0
```

---

## 🎯 What You Need to Complete the Project

### Critical Components (Priority Order)

#### 1. **Policy GUI** (`policy_gui.py`) - HIGH PRIORITY
**Purpose:** Allow users to define infrastructure changes visually

**Needs:**
- Interactive canvas overlaid on frames
- Drawing tools: lines (bike lanes), polygons (pedestrian zones), markers (EV stations)
- Color coding by policy type
- Export policy maps as PNG masks + JSON metadata

**Suggested libraries:**
- `tkinter` or `PyQt5` for GUI
- `Pillow` or `opencv` for drawing

**Implementation time:** 2-4 days

---

#### 2. **ControlNet Generation** (`generate_future.py`) - HIGH PRIORITY
**Purpose:** Core AI generation - synthesize future scenarios

**Needs:**
- ControlNet model integration (from `diffusers` library)
- Condition on: original frame + policy map + (optional) depth
- Video-to-video translation with temporal consistency
- Batch processing for all frames

**Suggested approach:**
- Use Hugging Face `diffusers` library
- ControlNet with Stable Diffusion backbone
- Consider: ControlNet-Inpaint for targeted modifications
- Temporal consistency: use frame-to-frame prompts or AnimateDiff

**Implementation time:** 1-2 weeks

---

#### 3. **Depth Estimation** (`depth_midas.py`) - MEDIUM PRIORITY
**Purpose:** Provide 3D scene understanding for ControlNet

**Needs:**
- MiDaS model integration
- Batch processing of frames → depth maps
- Normalization for ControlNet input

**Suggested approach:**
- Use `torch.hub.load("intel-isl/MiDaS", "MiDaS_small")` (lightweight)
- Output grayscale depth maps

**Implementation time:** 1-2 days

---

#### 4. **Workflow Orchestration** - MEDIUM PRIORITY
**Purpose:** One-click pipeline execution

**Create:** `run_pipeline.py` that chains all components
```bash
python run_pipeline.py --input video.mp4 --policy-config policy.json --output results/
```

**Implementation time:** 1 day

---

### Optional Enhancements

1. **OpenStreetMap Integration**
   - Overlay map data on frames
   - Suggest infrastructure based on OSM features
   - Libraries: `osmnx`, `folium`

2. **Metrics Dashboard**
   - Quantify changes: % bike lanes added, pedestrian area increased
   - Carbon emission estimates
   - Accessibility scores

3. **Multi-scenario Comparison**
   - Generate 3+ policy variants simultaneously
   - Side-by-side comparison interface

4. **Web Interface**
   - Replace `policy_gui.py` with web app (Streamlit, Gradio)
   - Cloud deployment for broader access

---

## 📝 Development Roadmap

### Phase 1: Core Functionality (Current)
- [x] Frame extraction
- [x] YOLOv8 segmentation
- [x] Mask refinement
- [x] Basic visualization

### Phase 2: Policy Input (1-2 weeks)
- [ ] Implement policy GUI
- [ ] Policy map export system
- [ ] Validation and metadata management

### Phase 3: AI Generation (2-3 weeks)
- [ ] Integrate MiDaS depth estimation
- [ ] Set up ControlNet pipeline
- [ ] Implement conditional generation
- [ ] Optimize temporal consistency

### Phase 4: Integration (1 week)
- [ ] Create end-to-end workflow script
- [ ] Testing and debugging
- [ ] Performance optimization

### Phase 5: Deployment (1 week)
- [ ] Documentation and tutorials
- [ ] Demo video creation
- [ ] Packaging and distribution

---

## 🐛 Troubleshooting

### Common Issues

**1. "ultralytics not found"**
```bash
pip install ultralytics
```

**2. "CUDA out of memory" during generation**
- Use smaller model (SD 1.5 instead of SDXL)
- Reduce batch size
- Enable CPU offloading: `low_cpu_mem_usage=True`

**3. Masks are empty**
- Lower confidence threshold: `--conf 0.15`
- Try detection fallback (automatic if segmentation fails)

**4. Video codec errors**
- Install: `pip install opencv-python-headless`
- Try different codec: modify `compose_video.py` fourcc to `"avc1"`

---

## 📚 Resources

### Datasets
- **KITTI Dataset:** http://www.cvlibs.net/datasets/kitti/
- **Cityscapes:** https://www.cityscapes-dataset.com/
- **BDD100K:** https://bdd-data.berkeley.edu/

### Models
- **YOLOv8:** https://docs.ultralytics.com/
- **SAM:** https://segment-anything.com/
- **ControlNet:** https://huggingface.co/lllyasviel
- **MiDaS:** https://github.com/isl-org/MiDaS

### Research Papers
- CityDreamer4D (reference architecture)
- ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models
- Segment Anything (SAM)

---

## 👥 Contributing

This project supports UN SDG Target 11.2. Contributions welcome!

**Priority needs:**
1. ControlNet integration developer
2. UI/UX designer for policy GUI
3. Urban planning domain expert for validation

---

## 📄 License

[Add your license here - MIT, Apache 2.0, GPL, etc.]

---

## 🎓 Citation

If you use MobilityDreamer in your research, please cite:

```bibtex
@software{mobilitydreamer2026,
  title={MobilityDreamer: Visualizing Sustainable Urban Mobility Futures},
  year={2026},
  url={https://github.com/yourusername/MobilityDreamer}
}
```
