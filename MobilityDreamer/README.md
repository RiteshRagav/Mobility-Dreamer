# MobilityDreamer

**A Framework for Visualizing Sustainable Urban Mobility Futures Using Computer Vision and Controllable Generative AI**

Supporting UN SDG 11: Sustainable Cities and Communities

---

## 🎯 Project Overview

MobilityDreamer translates high-level policy inputs into high-fidelity video visualizations of future urban scenarios. Policymakers, urban planners, and the public can visualize the real-world impact of proposed infrastructure changes (bike lanes, pedestrian zones, EV charging hubs, etc.) before implementation.

### Three-Stage Pipeline

```
1. CV Analysis → Parse real-world footage into semantic layers
2. Policy Input → Define infrastructure changes via interactive GUI
3. AI Generation → Synthesize realistic future scenario videos
```

---

## 📁 Project Structure

```
MobilityDreamer/
├── data/
│   ├── frames/              # Extracted video frames
│   ├── masks/               # YOLOv8 segmentation masks
│   ├── masks_refined/       # SAM/GrabCut refined masks
│   ├── policy_maps/         # User-defined policy intervention maps
│   ├── generated_frames/    # AI-generated future scenario frames
│   └── input_videos/        # Original input videos (e.g., KITTI dataset)
├── models/
│   ├── yolo/                # YOLOv8 models (download from Ultralytics)
│   ├── sam/                 # Segment Anything models (optional)
│   ├── controlnet/          # ControlNet checkpoints (for generation)
│   └── midas/               # MiDaS depth estimation models (optional)
├── results/
│   └── before_after/        # Visualization outputs
├── src/
│   ├── extract_frames.py        # ✅ Extract frames from video
│   ├── segmentation_yolo.py     # ✅ YOLOv8 segmentation
│   ├── segmentation_sam.py      # ✅ Mask refinement (SAM/GrabCut)
│   ├── policy_gui.py            # ⚠️  Policy editor GUI (TO IMPLEMENT)
│   ├── depth_midas.py           # ⚠️  Depth estimation (TO IMPLEMENT)
│   ├── generate_future.py       # ⚠️  ControlNet generation (TO IMPLEMENT)
│   └── compose_video.py         # ✅ Create before/after videos
└── README.md
```

**Legend:**
- ✅ Fully implemented
- ⚠️  Stub/needs implementation

---

## 🚀 Current Status

### ✅ What's Working
- Frame extraction from videos
- YOLOv8-based semantic segmentation (vehicles, roads, sidewalks)
- Mask refinement with SAM or GrabCut fallback
- Before/after visualization compositing

### ⚠️ What's Missing (Critical for Full Pipeline)
1. **Policy GUI** (`policy_gui.py`) - Interactive tool to draw infrastructure changes
2. **Depth Estimation** (`depth_midas.py`) - 3D scene understanding with MiDaS
3. **Future Generation** (`generate_future.py`) - ControlNet-based conditional video synthesis
4. **Workflow Orchestration** - End-to-end pipeline script

---

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB+ recommended for generation models)
- **GPU**: Optional but highly recommended for ControlNet/diffusion models
- **Storage**: ~10GB for models and datasets

### Python Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate
```

---

## 📦 Installation

### 1. Install Core Dependencies

Create `requirements.txt` (see Dependencies section below), then:
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
  author={[Your Name]},
  year={2026},
  url={https://github.com/yourusername/MobilityDreamer}
}
```

---

## 📧 Contact

For questions or collaboration: [your.email@example.com]

---

**Last Updated:** January 7, 2026
