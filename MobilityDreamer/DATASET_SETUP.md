# Dataset Setup Instructions

## Quick Start (3 Steps)

### Step 1: Download BDD100K Dataset

1. Go to https://bair.berkeley.edu/blog/2018/05/30/bdd/
2. Download **BDD100K 100K Videos** (training set):
   - Size: ~18 GB
   - Format: Multiple .mov video files
   - Contains diverse urban driving scenarios

### Step 2: Place Dataset in Correct Location

After downloading and extracting, structure should be:
```
MobilityDreamer/
├── train.bat                          ← Run this to train
├── bdd100k_videos_train_00/
│   └── bdd100k/
│       └── videos/
│           └── train/
│               ├── 00000.mov
│               ├── 00001.mov
│               ├── ... (500+ videos)
│               └── 0xxxx.mov
├── data/
│   ├── frames/                        ← Auto-created by preprocessing
│   ├── masks/                         ← Auto-created by preprocessing
│   └── policy_maps/                   ← Auto-created by preprocessing
└── ... (other files)
```

**Key Point**: The exact path is:
```
MobilityDreamer/bdd100k_videos_train_00/bdd100k/videos/train/
```

### Step 3: Start Training

Simply double-click:
```
train.bat
```

The script will:
1. ✅ Validate dataset location
2. ✅ Create Python virtual environment
3. ✅ Install all dependencies (PyTorch, OpenCV, etc.)
4. ✅ Generate sequence indices from BDD100K videos
5. ✅ Start 100-epoch training automatically

**On GPU**: ~2-4 hours
**On CPU**: ~4-5 days

---

## What Gets Created During Training

```
output/
├── checkpoints/
│   ├── epoch_5.pt    ← Checkpoint every 5 epochs
│   ├── epoch_10.pt
│   └── epoch_100.pt  ← Final model
└── logs/
    └── tensorboard/  ← View with: tensorboard --logdir output/logs/
```

---

## If Dataset Download is Slow

**Alternative**: Use a subset of 5-10 videos for quick testing:
1. Extract only first 5 videos from BDD100K
2. Place in same folder structure
3. Training will use whatever videos it finds
4. Sequence generation happens automatically

---

## Troubleshooting

**Error: "BDD100K dataset folder not found"**
- Verify path: `bdd100k_videos_train_00/bdd100k/videos/train/` exists
- Check at least one `.mov` file is present

**Error: "Failed to install dependencies"**
- Make sure Python 3.8+ is installed: `python --version`
- Try manual install: `.venv\Scripts\pip install torch torchvision`

**Training is very slow**
- Normal on CPU (4-5 days for 100 epochs)
- Use GPU for 100x speedup
- Can interrupt and resume from last checkpoint

**Out of Memory (OOM)**
- Reduce batch size in `config/mobility_config.py`: `BATCH_SIZE = 2`
- Or reduce image size: `IMAGE_SIZE = (128, 128)` instead of `(256, 256)`

---

## What You're Training

The GAN learns to:
1. **Generate urban video frames** conditioned on semantic layout
2. **Apply policy interventions** (bike lanes, pedestrian zones, green spaces, etc.)
3. **Maintain temporal consistency** across video sequences
4. **Preserve scene realism** using adversarial + perceptual losses

After training, you can generate "future scenario" videos showing how cities would look with policy changes.

---

## Architecture Overview

```
Input: Video frames + semantic segmentation + policy intervention maps
         ↓
    Generator (7M params)
    - Semantic encoder (19 classes)
    - Policy encoder (7 intervention types)  
    - Temporal encoder (3D convolution)
    - Decoder → RGB video output
         ↓
    Discriminator (672K params)
    - Multi-scale spatial critic
    - Temporal consistency critic
         ↓
    Output: Photorealistic future scenario videos
```

Training uses 6 losses:
- GAN loss (adversarial)
- Reconstruction loss (L1 pixel-wise)
- Perceptual loss (VGG19 features)
- Temporal loss (frame smoothness)
- Policy loss (intervention visibility)
- Semantic loss (structure preservation)

---

## Next Steps (After Training)

1. **Inference**: Generate future scenarios from trained model
2. **Evaluation**: Compute FID, LPIPS, temporal consistency metrics
3. **Visualization**: Create before/after comparison videos
4. **Paper**: Document results and contributions

See `SYSTEM_BUILD_SUMMARY.md` for architecture details.
