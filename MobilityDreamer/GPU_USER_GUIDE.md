# MobilityDreamer - GPU Training Guide
**For Non-Technical Users with GPU Hardware**

---

## Quick Start (3 Simple Steps)

### Step 1: Install Python
1. Go to https://www.python.org/downloads/
2. Download Python 3.13 or newer
3. Run installer and **check "Add Python to PATH"**
4. Click "Install Now"

### Step 2: Start Training
1. Double-click `train.bat`
2. **First time only**: System will preprocess BDD100K videos (5-6 hours)
3. Training starts automatically after preprocessing
4. **DO NOT CLOSE THE WINDOW** (takes 8-12 hours total)

### Step 3: If Training Stops
1. Double-click `resume.bat`
2. Training continues from where it stopped
3. You can repeat this as many times as needed

---

## What You'll See During Training

### First Time Only: Preprocessing (5-6 hours)
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
  - Total frames: ~4,000
  - Estimated time: 5-6 hours
  - Disk space needed: ~15 GB

========================================================================
Pre9] Detecting hardware...
  GPU Available: YES - NVIDIA GeForce RTX 3090
  Device: CUDA

[8/9] Creating output directories...
  Status: Output directories ready

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
  
EPOCH 2/100 | Video: 0000f77c-6257be58 | Frame: 000020
========================================================================
  MOBILITYDREAMER - AUTOMATED TRAINING SYSTEM
========================================================================

[1/8] Checking Python installation...
  Status: OK

[2/8] Setting up virtual environment...
  Status: Virtual environment activated

[3/8] Activating virtual environment...
  Status: Virtual environment activated

[4/8] Installing dependencies...
  Status: All dependencies installed

[5/8] Checking training data...
  Status: Training data exists

[6/8] Validating dataset structure...
  Status: Dataset validation passed

[7/8] Detecting hardware...
  GPU Available: YES - NVIDIA GeForce RTX 3090
  Device: CUDA

[8/8] Starting training process...
========================================================================

EPOCH 1/100 | Processing: data/frames/frame_001.png
  Generator Loss: 2.3456
  Discriminator Loss: 0.8765
  Reconstruction Loss: 0.5432
  Progress: 1.0%
  
EPOCH 2/100 | Processing: data/frames/frame_002.png
  Generator Loss: 2.1234
  Discriminator Loss: 0.8543
  Reconstruction Loss: 0.5123
  Progress: 2.0%Video: 0000f77c-6257be58
  Last Processed Frame: 000450
  Files Processed: 1800
  Best Generator Loss: 0.8234
  Best Discriminator Loss: 0.4321
  Best Reconstruction Loss: 0.2156
  Progress: 45.0%

  Remaining: 55 epochs
  Next Video: 0000f77c-62c2a288
  Next Frame: 000000
[1/4] Loading previous training state...

  Last Training Session:
  =====================
  Epochs Completed: 45
  Last Processed File: data/frames/frame_045.png
  Files Processed: 45
  Best Generator Loss: 0.8234
  Best Discriminator Loss: 0.4321
  Best Reconstruction Loss: 0.2156
  Progress: 45.0%

  Remaining: 55 epochs
  Next File: data/frames/frame_046.png

[2/4] Activating virtual environment...
  Status: Virtual environment activated

[3/4] Detecting hardware...
  GPU Available: YES - NVIDIA GeForce RTX 3090

[4/4] Resuming Video: 0000f77c-62c2a288 | Frame: 000000

EPOCH 46/100 | Processing: data/frames/frame_046.png
  Generator Loss: 0.8123
  Discriminator Loss: 0.4298
  Reconstruction Loss: 0.2134
  Progress: 46.0%
```

---

## Important Information

### Training Time
- **First Time Preprocessing**: 5-6 hours (ONE-TIME ONLY)
- **Training Duration**: 8-12 hours (depends on your GPU)
- **Per Epoch**: 5-10 minutes
- **Total Epochs**: 100
- **Total Time (First Run)**: 13-18 hours
- Detects 18GB BDD100K dataset (1000 videos)  
✅ Preprocesses 100 videos into training data (ONE-TIME)  
✅ Extracts 4,000 frames total  
✅ Runs YOLOv8 segmentation on all frames  
✅ Generates policy maps for mobility guidance  
✅ Creates virtual environment  
✅ Installs all required softwarecally
✅ Creates virtual environment  
✅ Installs all required software  
✅ Vdatasets/processed/` - Preprocessed BDD100K data (~15 GB, ONE-TIME)
- `alidates dataset files  
✅ Detects and uses your GPU  
✅ Saves progress every epoch  
✅ Updates research paper data file  
✅ Creates checkpoints for resume  

### Files Created During Training
- `training_state.json` - Current progress
- `training_metrics.json` - All metrics history
- `output/checkpoints/` - Model checkpoints (every epoch)
- `logs/` - Detailed training logs
- `IEEE_PAPER_DATA.md` - Auto-updated research data

### If SomethinBDD100K dataset folder not found"
**Solution**: Ensure `bdd100k_videos_train_00/` folder exists with 1000 `.mov` files

#### Problem: "Dataset validation failed"
**Solution**: Wait for preprocessing to complete (first run only)

#### Problem: Preprocessing taking too long
**Solution**: This is normal! First run takes 5-6 hours. Subsequent runs skip this step.
#### Problem: "Python not found"
**Solution**: Install Python from https://www.python.org/downloads/

#### Problem: "Dataset validation failed"
**Solution**: Run `train.bat` again - it will generate synthetic data

#### Problem: "GPU not detected"
**Solution**: Training will use CPU (slower but works)

#### Problem: Training window closed accidentally
**Solution**: Double-click `resume.bat` - all progress is saved

#### Problem: Computer restarted
**Solution**: Double-click `resume.bat` - training continues from last epoch

---

## What to Do After Training Completes

### You'll See This Message:
```
========================================================================
  TRAINING COMPLETED SUCCESSFULLY!
========================================================================

Next steps:
  1. Review results in output/checkpoints/
  2. Check updated metrics in IEEE_PAPER_DATA.md
  3. Run tests: python tests/smoke_test.py
  4. Commit to GitHub: git add . && git commit -m "Training complete"
```

### What You Need to Do:
1. **DO NOTHING** - The system is ready
2. Tell the project owner training is complete
3. If requested, upload the following to GitHub:
   - `output/checkpoints/` folder (all .pt files)
   - `IHow long does training take?
**A:** First run: 13-18 hours (5-6 hours preprocessing + 8-12 hours training). After that: only 8-12 hours (no preprocessing needed)

---use my computer during training?
**A:** Yes, but training will be slower. For best results, leave it overnight.

### Q: What is preprocessing?
**A:** Can I close the terminal window?
**A:** NO! This will stop training. Use `resume.bat` to continue.

### Q: Why does first run take so long?
**A:** Preprocessing 100 videos (extracting frames + segmentation) takes 5-6 hours. This only happens ONCE. After that, training starts immediately

### Q: Can I use my computer during training?
**A:** Yes, but training will be slower. For best results, leave it overnight.

### Q: Can I close the terminal window?
**A:** NO! This will stop training. Use `resume.bat` to continue.

### Q: How do I know training is working?
**A:** You'll see "EPOCH X/100" updating every few minutes with new numbers.

### Q: Whatdo I know training is working?
**A:** You'll see "EPOCH X/100" with video names and frame numbers updating every few minutes.

### Q: How much disk space is needed?
**A:** About 25 GB total:
- BDD100K raw videos: 18 GB (already have)
- Preprocessed data: 15 GB (created on first run)
- Checkpoints: ~10 GB (during training) doesn't need internet.

### Q: Can I pause training?
**A:** Press `Ctrl+C` in the window, then run `resume.bat` when ready.

### Q: How much disk space is needed?
**A:** About 10 GB for checkpoints and data.

### Q: What if I see red error messages?
**A:** Take a screenshot and send it to the project owner.

---

## Contact

If you encounter any issues:
1. Take a screenshot of the error
2. Note what step you're on
3. Contact the project owner
4. Include the file `logs/training.log` if it exists

---

**Remember**: You only need to double-click `train.bat` once. If it stops, double-click `resume.bat`. That's it!
