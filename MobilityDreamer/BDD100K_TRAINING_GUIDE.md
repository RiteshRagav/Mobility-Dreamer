# BDD100K Training System - Complete Guide

## Overview
Your MobilityDreamer project now uses the **real 18GB BDD100K dataset** with 1000 driving videos instead of synthetic data.

---

## Dataset Information

### Raw Dataset
- **Location**: `bdd100k_videos_train_00/`
- **Files**: 1000 `.mov` video files
- **Total Size**: 18 GB
- **Resolution**: 1280×720 HD
- **Content**: Real-world driving scenarios from Berkeley DeepDrive dataset

### Processing Configuration
- **Videos to Process**: 100 out of 1000 (configurable)
- **Frames per Video**: 40 frames
- **Frame Stride**: 10 (every 10th frame extracted)
- **Total Frames**: ~4,000 training frames
- **Train/Val Split**: 85% train, 15% validation

---

## Complete Workflow (For GPU User)

### Step 1: One-Time Preprocessing (5-6 hours)
**What it does:**
1. Extracts 40 frames from 100 BDD100K videos
2. Runs YOLOv8 segmentation on all frames
3. Generates policy maps for mobility guidance
4. Creates train/val sequence index files

**How to run:**
```batch
double-click train.bat
```

**What you'll see:**
```
[5/9] Checking BDD100K dataset preprocessing...

========================================================================
PREPROCESSING REQUIRED
========================================================================

Your BDD100K videos need to be processed into training data.
This is a ONE-TIME process with the following steps:

1. Extract 40 frames from 100 videos (stride 10)
2. Run YOLOv8 segmentation on all frames
3. Generate policy maps for mobility guidance
4. Create train/val sequence indices

Configuration:
  - Videos to process: 100 out of 1000 available
  - Frames per video: 40
  - Frame stride: 10
  - Total frames: ~4,000
  - Estimated time: 5-6 hours
  - Disk space needed: ~15 GB

========================================================================
Press Ctrl+C now to cancel, or any key to begin preprocessing...
========================================================================
```

**Output Structure:**
```
datasets/
  processed/
    train/
      0000f77c-6257be58/          # Video ID
        frames/                    # Extracted frames
          frame_000000.jpg
          frame_000010.jpg
          ...
        segmentation/              # YOLOv8 masks
          seg_000000.png
          seg_000010.png
          ...
        policy/                    # Policy maps
          policy_000000.png
          policy_000010.png
          ...
      0000f77c-62c2a288/          # Next video
        ...
    train_sequences.json           # Training sequence index
    val_sequences.json             # Validation sequence index
```

### Step 2: Training (8-12 hours)
After preprocessing completes automatically, training begins:

```
[9/9] Starting training process...
========================================================================

TRAINING CONFIGURATION:
  - Epochs: 100
  - Batch Size: 4
  - Dataset: BDD100K (100 videos, ~4,000 frames)
  - Train/Val: 85%/15% split
  - GPU: Auto-detected
  - Checkpoints: Saved every epoch
  - Progress: Live updates to IEEE_PAPER_DATA.md

========================================================================

EPOCH 1/100 | Video: 0000f77c-6257be58 | Frame: 000010
  Generator Loss: 2.3456
  Discriminator Loss: 0.8765
  Reconstruction Loss: 0.5432
  Progress: 1.0%
```

---

## File Structure

### Before Preprocessing
```
MobilityDreamer/
├── bdd100k_videos_train_00/    # 18GB raw videos (1000 .mov files)
├── train.bat                    # START HERE
├── resume.bat                   # Resume if interrupted
└── scripts/
    └── preprocess_bdd100k.py    # Preprocessing script
```

### After Preprocessing
```
MobilityDreamer/
├── datasets/
│   └── processed/
│       ├── train/               # 100 video folders
│       │   ├── 0000f77c-6257be58/
│       │   ├── 0000f77c-62c2a288/
│       │   └── ...
│       ├── train_sequences.json # Train index (~85 sequences)
│       └── val_sequences.json   # Val index (~15 sequences)
├── training_state.json          # Current training progress
├── training_metrics.json        # All metrics history
└── output/
    └── checkpoints/             # Model checkpoints (saved every epoch)
```

---

## Configuration Options

### Adjust Processing Scale
Edit `dataset_structure.json` to change:

```json
{
  "dataset_info": {
    "videos_to_process": 100,      // Change to 200, 500, or 1000
    "frames_per_video": 40,         // More frames = more data
    "frame_stride": 10              // Lower stride = more frames
  }
}
```

**Scaling Examples:**
- **Small** (100 videos × 40 frames = 4,000 frames): ~6 hours preprocessing, 15GB disk
- **Medium** (500 videos × 40 frames = 20,000 frames): ~30 hours preprocessing, 75GB disk
- **Full** (1000 videos × 100 frames = 100,000 frames): ~60 hours preprocessing, 300GB disk

### Manual Preprocessing Command
```batch
python scripts/preprocess_bdd100k.py ^
    --config config/mobility_config.py ^
    --max-videos 100 ^
    --frames-per-video 40 ^
    --stride 10 ^
    --val-ratio 0.15 ^
    --enable-seg ^
    --enable-policy
```

---

## Progress Tracking

### Check Preprocessing Progress
```batch
python -c "import json; train=json.load(open('datasets/processed/train_sequences.json')); print(f'Train sequences: {len(train)}'); val=json.load(open('datasets/processed/val_sequences.json')); print(f'Val sequences: {len(val)}')"
```

### Check Training Progress
```batch
python -c "import json; state=json.load(open('training_state.json')); print(f'Epoch: {state[\"current_epoch\"]}/100'); print(f'Progress: {state[\"current_epoch\"]}%')"
```

### Live IEEE Paper Updates
Training automatically updates [IEEE_PAPER_DATA.md](IEEE_PAPER_DATA.md) Section 6.1 with:
- Generator Loss (latest + best)
- Discriminator Loss (latest + best)
- Reconstruction Loss (latest + best)
- Epochs completed
- Training time
- Current video/frame being processed
- Best performance checkpoint

---

## Troubleshooting

### Problem: "BDD100K dataset folder not found"
**Solution**: Ensure `bdd100k_videos_train_00/` exists in project root with 1000 `.mov` files

### Problem: Preprocessing fails
**Solution**: Check GPU memory, reduce `--max-videos` to 50 or 25

### Problem: "No videos found in bdd100k_videos_train_00"
**Solution**: Check folder structure, should be `.mov` files directly in folder (not nested)

### Problem: Out of disk space
**Solution**: Reduce processing scale:
- `--max-videos 50` = ~7.5 GB
- `--max-videos 25` = ~4 GB

### Problem: Preprocessing takes too long
**Solution**: 
- Disable segmentation: remove `--enable-seg`
- Disable policy maps: remove `--enable-policy`
- Use fewer videos: `--max-videos 25`

---

## For Non-Technical GPU User

### What to Do:
1. **Double-click `train.bat`**
2. **Wait for preprocessing** (5-6 hours, shows progress)
3. **Training starts automatically** after preprocessing
4. **If interrupted**: Double-click `resume.bat`

### What NOT to Do:
- Don't close the terminal window
- Don't delete `bdd100k_videos_train_00/` folder
- Don't delete `datasets/processed/` after preprocessing completes
- Don't delete `training_state.json` during training

### When Complete:
Upload these to GitHub:
- `output/checkpoints/` (all `.pt` files)
- `IEEE_PAPER_DATA.md` (auto-updated with metrics)
- `training_metrics.json` (full training history)

---

## Dataset Statistics

### Videos Available
```
Total videos: 1000
Configured to use: 100 (10%)
Frames per video: 40
Frame stride: 10 (sample 1 frame every 10 frames)
```

### Processing Breakdown
```
Input:  100 videos × ~400 frames each = ~40,000 raw frames
Output: 100 videos × 40 frames each = 4,000 training frames

Storage:
- Frames: 4,000 × ~500 KB = ~2 GB
- Segmentation masks: 4,000 × ~100 KB = ~400 MB
- Policy maps: 4,000 × ~100 KB = ~400 MB
- Checkpoints (100 epochs): ~10 GB
Total: ~15 GB
```

---

## Advanced: Full Dataset Training

To train on **all 1000 videos**:

1. Edit `dataset_structure.json`:
```json
"videos_to_process": 1000,
"frames_per_video": 100
```

2. Run preprocessing:
```batch
python scripts/preprocess_bdd100k.py --config config/mobility_config.py --max-videos 1000 --frames-per-video 100 --stride 10 --val-ratio 0.15 --enable-seg --enable-policy
```

**Warning**: This requires:
- ~60 hours preprocessing time
- ~300 GB disk space
- High-end GPU (RTX 3090 or better)

---

## Contact

If you encounter issues:
1. Check terminal output for error messages
2. Look for `logs/training.log`
3. Take screenshot of error
4. Contact project owner with:
   - Error screenshot
   - Last step completed
   - `training_state.json` (if exists)

---

**Remember**: Preprocessing is ONE-TIME. Once `datasets/processed/train_sequences.json` exists, you can train/resume unlimited times without reprocessing!
