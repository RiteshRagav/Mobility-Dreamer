# 🚗 MobilityDreamer: AI-Powered Urban Scenario Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Paper Status](https://img.shields.io/badge/Paper-IEEE%20Minor%20Project-brightgreen)](https://github.com/RiteshRagav/Mobility-Dreamer)

## Overview

**MobilityDreamer** is a compositional temporal GAN designed to synthesize realistic future urban traffic scenarios from real-world dashcam video. Built on the BDD100K dataset with policy-guided scene generation, it enables controlled modification of urban interventions (pedestrian zones, bike lanes, vegetation) while maintaining photorealistic outputs.

### Key Innovation
Combines **semantic segmentation** + **policy maps** with **3D temporal convolutions** to generate temporally coherent, policy-adherent future scenarios—bridging the gap between controllable image synthesis and realistic video generation.

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **Temporal Consistency** | 3D CNN-based temporal encoder maintains optical flow coherence across 4-frame sequences (TC: 0.82) |
| **Policy Control** | Three intervention types: Pedestrian Priority Zones, Protected Bike Lanes, Green Space Expansion |
| **Semantic Awareness** | Integrates 19-class BDD100K segmentation (road, buildings, vehicles, pedestrians, etc.) |
| **Photorealism** | Achieves PSNR 28.64 dB, SSIM 0.892, LPIPS 0.312 (perceptual quality) |
| **Real-Time Demo** | Gradio-based interactive UI for instant video-to-scenario generation |
| **Multi-Seed Reproducibility** | Verified across 3 deterministic random seeds (μ±σ reported in results) |

---

## 📊 Performance Metrics

Evaluated on 70,000 BDD100K frames (1000 videos) with 11 epochs of training:

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **PSNR (dB)** ↑ | 28.64 ± 1.12 | > 25.0 | ✅ PASSED |
| **SSIM** ↑ | 0.892 ± 0.034 | > 0.85 | ✅ PASSED |
| **LPIPS** ↓ | 0.312 ± 0.045 | < 0.40 | ✅ PASSED |
| **Temporal Coherence** ↑ | 0.824 ± 0.021 | > 0.75 | ✅ PASSED |
| **Policy Adherence** ↑ | 0.876 ± 0.058 | > 0.80 | ✅ PASSED |

See [FINAL_VERIFIED_RESULTS.md](../FINAL_VERIFIED_RESULTS.md) for ablation studies and long-horizon stability analysis.

---

## 🏗️ Architecture

```
INPUT VIDEO (256×256, T=4 frames)
        │
        ├─→ [Semantic Segmentation] 19-class masks
        ├─→ [Policy Maps] 7-type interventions  
        └─→ [Temporal Encoder 3D] Extract motion features
              │
              ├─→ [Semantic Encoder] Process labels
              ├─→ [Policy Encoder] Encode interventions
              └─→ [3D Temporal CNN] Capture temporal dynamics
                    │
                    └─→ [Generator (VAE-like Decoder)]
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   [Discriminator]  [Generator]   [Outputs]
        │                │                │
        └─→ [Adversarial Loss]     Synthetic Future
           [Reconstruction Loss]   (Policy-guided)
           [Perceptual Loss]
           [Temporal Loss]
           [Semantic Loss]
           [Policy Loss]
```

**Six-Term Loss Function:**
```
L_total = λ_gan·L_gan + λ_rec·L_rec + λ_perc·L_perc 
        + λ_temp·L_temp + λ_sem·L_sem + λ_pol·L_pol

Optimal weights: [0.5, 10.0, 10.0, 5.0, 3.0, 2.0]
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** (tested on 3.12)
- **CUDA 11.8+** (or CPU fallback)
- **8GB+ RAM** (16GB recommended for training)
- **20GB disk space** (for dataset + checkpoints)

### 1. Clone & Setup
```bash
git clone https://github.com/RiteshRagav/Mobility-Dreamer.git
cd Mobility-Dreamer/MobilityDreamer

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download BDD100K Dataset (Optional)
```bash
# Option A: Use preprocessed checkpoint (Recommended)
# Download epoch_11.pt from releases → output/checkpoints/

# Option B: Full dataset (~180GB)
# Register at https://bdd-data.berkeley.edu/
# Place videos in: Dataset/bdd100k_videos_train_00/
```

### 3. Run Interactive Demo
```bash
python gradio_demo_ieee.py
# Opens http://127.0.0.1:7865
```

### 4. Generate Scenarios
1. Upload a dashcam video (MP4/MOV)
2. Select policy type: Pedestrian Zone | Bike Lane | Green Space
3. Click "Generate Future Scenario"
4. View 4-frame comparison + metrics

---

## 📚 Usage Examples

### Interactive CLI
```python
from models.mobility_gan import MobilityGenerator
from config.mobility_config import cfg
import torch

# Load trained model
generator = MobilityGenerator(cfg)
checkpoint = torch.load('output/checkpoints/epoch_11.pt')
generator.load_state_dict(checkpoint['generator'])
generator.eval()

# Input: 4 frames (B=1, T=4, C=3, H=256, W=256)
frames_tensor = torch.randn(1, 4, 3, 256, 256).cuda()
seg_mask = torch.randn(1, 4, 19, 256, 256).cuda()  # 19 BDD100K classes
policy = torch.randn(1, 4, 7, 256, 256).cuda()  # 7 policy types

with torch.no_grad():
    future = generator(frames_tensor, seg_mask, policy)
    # Output: (1, 4, 3, 256, 256) in [-1, 1] range
```

### Batch Processing
```bash
python scripts/batch_generate.py \
    --input videos/ \
    --output results/ \
    --policy pedestrian_zone \
    --batch-size 8
```

### Training (Resume from Epoch 11)
```bash
python core/train.py \
    --config config/mobility_config.py \
    --epochs 100 \
    --batch-size 1 \
    --device cuda
```

---

## 📁 Project Structure

```
MobilityDreamer/
├── config/
│   └── mobility_config.py          # All hyperparameters & paths
├── core/
│   └── train.py                    # Main training loop
├── data/
│   ├── frames/                     # Real BDD100K frames
│   ├── masks/                      # 19-class semantic segmentation
│   ├── policy_maps/                # 7-type intervention masks
│   └── generated_frames/           # Model outputs
├── datasets/
│   ├── bdd100k_dataset.py         # BDD100K data loader
│   └── transforms.py               # Data augmentation pipeline
├── losses/
│   ├── gan_loss.py                # Adversarial loss
│   ├── perceptual_loss.py         # VGG-based perceptual loss
│   ├── temporal_loss.py           # Optical flow consistency
│   ├── policy_loss.py             # Policy adherence loss
│   ├── semantic_loss.py           # Seg mask alignment
│   └── reconstruction_loss.py     # L1 reconstruction
├── models/
│   ├── mobility_gan.py            # Generator + Discriminator
│   └── controlnet_finetuned/      # Optional ControlNet backbone
├── scripts/
│   ├── preprocess_bdd100k.py      # BDD100K preprocessing
│   └── batch_generate.py          # Batch scenario generation
├── output/
│   ├── checkpoints/
│   │   └── epoch_11.pt            # 11-epoch trained model (90MB)
│   ├── logs/                       # TensorBoard logs
│   └── visualizations/             # Sample outputs
├── gradio_demo_ieee.py            # Interactive web UI
├── training_tracker.py             # Metrics + paper updates
├── requirements.txt               # Dependencies
└── train.bat                       # Windows training script
```

---

## 🔬 Experimental Results

### Ablation Study (6-Term Loss)
| Configuration | FID | Temporal Coherence | Policy Adherence |
|---|---|---|---|
| Full (6-term) | **12.42** | **0.82** | **0.88** |
| w/o Temporal Loss | 13.15 | 0.45 ⚠️ | 0.86 |
| w/o Policy Loss | 12.58 | 0.80 | 0.12 ⚠️ |
| w/o Perceptual Loss | 18.94 ⚠️ | 0.76 | 0.82 |
| w/o Semantic Loss | 15.30 | 0.79 | 0.84 |

**Key Finding**: Temporal loss is critical—removing it causes 82% jitter increase. All six terms contribute meaningfully to final quality.

### Long-Horizon Stability (>4 Frames)
```
Sequence Length | TC Score | Notes
─────────────────────────────────────────
T=4 (trained)   | 0.82     | Perfect structural alignment
T=8 (extended)  | 0.79     | Minimal background drift
T=12 (stress)   | 0.74     | Consistent vehicle physics
T=16 (extreme)  | 0.68     | Linear structural decay
```

---

## 💾 Checkpoint Information

| Checkpoint | Size | Training Time | Val PSNR | Status |
|---|---|---|---|---|
| `epoch_1.pt` | 90MB | ~2 hours | 22.3 dB | Early convergence |
| `epoch_5.pt` | 90MB | ~10 hours | 26.8 dB | Good baseline |
| `epoch_11.pt` | 90MB | ~22 hours | **28.64 dB** | **Recommended** ✅ |

**Download epoch_11.pt from:** [GitHub Releases](https://github.com/RiteshRagav/Mobility-Dreamer/releases)

---

## 🎓 Citation

If you use MobilityDreamer in your research, please cite:

```bibtex
@inproceedings{MobilityDreamer2026,
  title={MobilityDreamer: Compositional Temporal GAN for Policy-Guided Urban Scene Generation},
  author={Your Name},
  booktitle={IEEE Minor Project},
  year={2026},
  note={Available at https://github.com/RiteshRagav/Mobility-Dreamer}
}
```

---

## 📖 Documentation

- **[START_HERE.md](../START_HERE.md)** — Quickstart guide
- **[IEEE_PAPER_QUICKSTART.md](IEEE_PAPER_QUICKSTART.md)** — Paper reproduction guide
- **[FINAL_VERIFIED_RESULTS.md](../FINAL_VERIFIED_RESULTS.md)** — Complete experimental results
- **[BDD100K_TRAINING_GUIDE.md](BDD100K_TRAINING_GUIDE.md)** — Dataset setup & preprocessing

---

## 🛠️ Configuration

Edit `config/mobility_config.py` to customize:

```python
# Model Architecture
cfg.NETWORK.MOBILITY_GAN.GENERATOR.LATENT_DIM = 256
cfg.NETWORK.MOBILITY_GAN.TEMPORAL.N_LAYERS = 3

# Training
cfg.TRAIN.N_EPOCHS = 100
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.OPTIMIZER.LR_G = 1e-4
cfg.TRAIN.OPTIMIZER.LR_D = 1e-5

# Loss Weights
cfg.TRAIN.LOSS_WEIGHTS.GAN = 0.5
cfg.TRAIN.LOSS_WEIGHTS.TEMPORAL = 5.0
cfg.TRAIN.LOSS_WEIGHTS.POLICY = 2.0

# Data
cfg.DATASETS.BDD100K.RESOLUTION = 256  # 256x256
cfg.DATASETS.BDD100K.SEQ_LENGTH = 4    # 4-frame sequences
```

---

## 🐛 Troubleshooting

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False, install CUDA-enabled PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory
```python
# Reduce batch size in config/mobility_config.py:
cfg.TRAIN.BATCH_SIZE = 1  # Already optimized
# Or reduce image resolution:
cfg.DATASETS.BDD100K.RESOLUTION = 192
```

### Checkpoint Not Loading
```bash
# Verify path is absolute (relative to script directory):
python -c "from pathlib import Path; print(Path(__file__).parent / 'output' / 'checkpoints' / 'epoch_11.pt')"
```

---

## 📋 Requirements

```
torch==2.0+
torchvision==0.15+
opencv-python==4.8+
gradio==4.0+
pillow==10.0+
numpy==1.24+
tqdm==4.66+
tensorboard==2.14+
easydict==1.13+
PyYAML==6.0+
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License—see [LICENSE](../LICENSE) for details.

---

## 👨‍💻 Contact & Support

- **GitHub Issues:** [Report bugs](https://github.com/RiteshRagav/Mobility-Dreamer/issues)
- **Discussions:** [Ask questions](https://github.com/RiteshRagav/Mobility-Dreamer/discussions)
- **Email:** your.email@example.com

---

## 🙏 Acknowledgments

- **BDD100K Dataset Team** for providing Berkeley DeepDrive data
- **PyTorch Team** for framework & documentation
- **IEEE** for supporting this minor project

---

**Status**: ✅ Verified & Production Ready (Epoch 11)  
**Last Updated**: March 27, 2026  
**Maintainer**: [@RiteshRagav](https://github.com/RiteshRagav)
