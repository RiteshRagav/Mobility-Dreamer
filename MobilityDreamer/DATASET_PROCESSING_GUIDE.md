# MobilityDreamer - Full BDD100K Dataset Processing Pipeline

## Overview

This document explains how to preprocess the full BDD100K dataset (700+ videos, 16GB) into frames, semantic masks, and depth-based policy maps for training MobilityDreamer.

## Dataset Structure

### Raw Data
- **Location**: `bdd100k_videos_train_00/`
- **Format**: `.mov` and `.mp4` video files
- **Count**: 700+ videos
- **Size**: ~16 GB
- **Resolution**: 1280×720 @ 30 FPS
- **Total frames available**: ~1,000 frames per video

### Preprocessing Output

After full preprocessing, you'll have:

```
data/
├── frames/
│   ├── frame_000000.jpg        # First frame from first video
│   ├── frame_000001.jpg
│   ├── ...
│   └── frame_069999.jpg        # ~70,000 total frames (100 frames × 700 videos)
├── masks/
│   ├── frame_000000_mask.png   # YOLOv8 segmentation
│   ├── frame_000001_mask.png
│   └── ...
└── policy_maps/
    ├── frame_000000_policy.png # MiDaS depth estimation
    ├── frame_000001_policy.png
    └── ...

datasets/processed/
├── train_sequences.json        # 59,500 frames (85%)
└── val_sequences.json          # 10,500 frames (15%)
```

## Preprocessing Pipeline

### Step 1: Frame Extraction (12 hours)

**Command:**
```bash
python scripts/preprocess_full_bdd100k.py --step extract --max-videos 700 --frames-per-video 100 --stride 10
```

**What it does:**
- Reads all 700+ `.mov` videos from `bdd100k_videos_train_00/`
- Extracts 100 frames from each video with stride=10 (skip 9 frames between extracts)
- Saves ~70,000 JPEG images to `data/frames/`
- Uses JPEG quality 85 to reduce disk space
- Resumes automatically if interrupted

**Output:**
- ~70,000 JPEG frames (1280×720)
- Disk space: ~35 GB

**Estimated time:** 12 hours on RTX 3090

### Step 2: Semantic Segmentation (24 hours)

**Command:**
```bash
python scripts/preprocess_full_bdd100k.py --step segment
```

**What it does:**
- Runs YOLOv8-seg model on all 70,000 frames
- Generates semantic segmentation masks (cars, pedestrians, roads, etc.)
- Saves masks as PNG files to `data/masks/`
- Masks used as conditioning input to generator

**Output:**
- ~70,000 PNG segmentation masks (1280×720, 8-bit)
- Disk space: ~35 GB

**Estimated time:** 24 hours on RTX 3090

**Note:** Requires `yolov8n-seg.pt` model (already included in repo)

### Step 3: Depth Estimation (18 hours)

**Command:**
```bash
python scripts/preprocess_full_bdd100k.py --step depth
```

**What it does:**
- Runs MiDaS depth estimation on all 70,000 frames
- Generates depth maps (used for policy/motion guidance)
- Saves depth as normalized PNG to `data/policy_maps/`
- Provides pixel-level motion guidance to generator

**Output:**
- ~70,000 PNG policy maps (1280×720, 8-bit normalized depth)
- Disk space: ~35 GB

**Estimated time:** 18 hours on RTX 3090

**Note:** Requires PyTorch + MiDaS model (auto-downloaded first run)

### Step 4: Sequence Indexing (< 1 minute)

**Command:**
```bash
python scripts/preprocess_full_bdd100k.py --step indices --val-ratio 0.15
```

**What it does:**
- Creates train/val split: 85% train (59,500 frames), 15% val (10,500 frames)
- Generates JSON index files:
  - `datasets/processed/train_sequences.json` - Training frame indices
  - `datasets/processed/val_sequences.json` - Validation frame indices
- Shuffles frame order to distribute videos evenly

**Output:**
- `train_sequences.json` - 59,500 frame indices
- `val_sequences.json` - 10,500 frame indices

**Estimated time:** < 1 minute

## Running Full Pipeline

### Automatic (Recommended)

The `train.bat` script automatically detects if preprocessing is needed and runs the full pipeline:

```batch
double-click train.bat
```

The script will:
1. Check if 50,000+ frames exist in `data/frames/`
2. If not, run: `python scripts/preprocess_full_bdd100k.py --full`
3. If yes, skip to training

### Manual Processing

To run preprocessing steps individually:

```bash
# Extract frames only
python scripts/preprocess_full_bdd100k.py --step extract --max-videos 700 --frames-per-video 100 --stride 10

# Add segmentation masks
python scripts/preprocess_full_bdd100k.py --step segment

# Add depth policy maps
python scripts/preprocess_full_bdd100k.py --step depth

# Create sequence indices
python scripts/preprocess_full_bdd100k.py --step indices --val-ratio 0.15

# Validate output
python scripts/preprocess_full_bdd100k.py --validate
```

### Resume Interrupted Preprocessing

Preprocessing automatically saves progress to `preprocessing_state.json`:

```bash
# Resume from last step
python scripts/preprocess_full_bdd100k.py --full

# Or resume specific step
python scripts/preprocess_full_bdd100k.py --step segment  # Continues from where it left off
```

## Hardware Requirements

### Minimum
- GPU: 6 GB VRAM (GTX 1060, RTX 2060)
- CPU: 4-core
- RAM: 8 GB
- Disk: 150 GB SSD (for frames + masks + policies)
- **Preprocessing time**: 48-72 hours

### Recommended
- GPU: 24 GB VRAM (RTX 3090, A100)
- CPU: 8+ cores
- RAM: 16 GB+
- Disk: 150 GB NVMe SSD
- **Preprocessing time**: 24-36 hours

### High-Performance
- GPU: 24 GB VRAM (RTX 4090)
- CPU: 16+ cores
- RAM: 32 GB
- Disk: 150 GB NVMe SSD
- **Preprocessing time**: 12-18 hours

## Storage Space

| Stage | Size | Cumulative |
|-------|------|-----------|
| Raw videos (bdd100k_videos_train_00) | 16 GB | 16 GB |
| Extracted frames (JPEG) | 35 GB | 51 GB |
| Segmentation masks (PNG) | 35 GB | 86 GB |
| Depth policy maps (PNG) | 35 GB | 121 GB |
| Sequence indices (JSON) | < 1 MB | 121 GB |
| Training checkpoints | 30 GB | 151 GB |
| **Total** | | **~150 GB** |

**Note:** You can delete raw .mov videos after Step 1 to save 16 GB if needed.

## Validation & Monitoring

### Check Progress
```bash
# See preprocessing state
type preprocessing_state.json

# Count extracted frames
dir /s data\frames\*.jpg | find /c ".jpg"

# Count masks
dir /s data\masks\*_mask.png | find /c ".png"

# Count policy maps
dir /s data\policy_maps\*_policy.png | find /c ".png"
```

### Validate Output
```bash
python scripts/preprocess_full_bdd100k.py --validate
```

Expected output:
```
Frames: 70,000
Masks: 70,000
Policy maps: 70,000

✓ All preprocessing steps completed successfully!
  Total frames: 70,000
  Total disk space: ~105 GB
```

## Troubleshooting

### Issue: "Video directory not found"
**Solution:** Ensure `bdd100k_videos_train_00/` exists in project root with 700+ `.mov` files

### Issue: "CUDA out of memory"
**Solution:** 
- Run preprocessing on CPU: `CUDA_VISIBLE_DEVICES="" python scripts/preprocess_full_bdd100k.py --step segment`
- Or reduce batch size in preprocessing script

### Issue: "YOLOv8 not installed"
**Solution:** 
```bash
pip install ultralytics
```

### Issue: "MiDaS model download fails"
**Solution:**
- Manually download from: https://github.com/intel-isl/MiDaS
- Or script will create synthetic depth maps as fallback

### Issue: Segmentation step stalled
**Solution:** 
- Preprocessing is resume-able - kill script and restart
- State is saved every 50 videos
- Script will continue from last completed video

### Issue: "Not enough disk space"
**Solution:**
- Ensure 150 GB free SSD space
- Delete raw videos after Step 1: `rm -rf bdd100k_videos_train_00/` (saves 16 GB)
- Reduce frame extraction stride (extract fewer frames): `--stride 20` (saves space)

## Data Specifications

### Frame Resolution
- Original: 1280×720 (16:9)
- Saved as: JPEG, quality 85
- Size per frame: ~50-100 KB

### Mask Properties
- Resolution: 1280×720
- Format: PNG grayscale (8-bit)
- Values: 0-255 (semantic class IDs)
- Size per mask: ~35-50 KB

### Policy Map Properties
- Resolution: 1280×720
- Format: PNG grayscale (8-bit)
- Values: 0-255 (normalized depth)
- Size per policy map: ~35-50 KB

## Training with Preprocessed Data

After preprocessing, start training:

```bash
# Run training (automatic resume available)
double-click train.bat

# Or continue training
double-click resume.bat
```

The training script automatically:
1. Loads preprocessed frames from `data/frames/`
2. Loads segmentation masks from `data/masks/`
3. Loads policy maps from `data/policy_maps/`
4. Uses sequence indices from `datasets/processed/`
5. Tracks progress and updates IEEE paper file

## Expected Training Time

With 70,000 frames and batch size 4:
- **Batch time**: 0.1-0.2 seconds/batch (on RTX 3090)
- **Batches per epoch**: 17,500
- **Time per epoch**: 29-58 minutes
- **100 epochs**: 48-97 hours (2-4 days)

**Total pipeline time:**
- Preprocessing: 24-54 hours
- Training (100 epochs): 48-97 hours
- **Total**: 3-7 days on RTX 3090

## References

- **BDD100K**: https://bair.berkeley.edu/blog/2018/05/30/bdd/
- **YOLOv8**: https://github.com/ultralytics/ultralytics
- **MiDaS**: https://github.com/intel-isl/MiDaS
