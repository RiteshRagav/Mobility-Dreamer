@echo off
REM ============================================================================
REM MobilityDreamer - Automated GPU Training System
REM ============================================================================
REM This script provides:
REM   - Automatic dataset validation
REM   - GPU detection and utilization
REM   - Progress tracking with live display
REM   - Checkpoint management for resume
REM   - Live IEEE paper data updates
REM   - Detailed logging of all training stages
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo   MOBILITYDREAMER - AUTOMATED TRAINING SYSTEM
echo ========================================================================
echo   Date: %date% %time%
echo   Location: %cd%
echo ========================================================================
echo.

REM Step 1: Check Python installation
echo [1/8] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.13+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo   Status: OK
echo.

REM Step 2: Check/Create Virtual Environment
echo [2/8] Setting up virtual environment...
if not exist ".venv\" (
    echo   Creating new virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo   Status: Virtual environment created
) else (
    echo   Status: Using existing virtual environment
)
echo.

REM Step 3: Activate Virtual Environment
echo [3/8] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo   Status: Virtual environment activated
echo.

REM Step 4: Install/Update Dependencies
echo [4/9] Installing dependencies...
echo   This may take a few minutes on first run...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)
echo   Status: All dependencies installed
echo.

REM Step 5: Check BDD100K Dataset and Preprocessing
echo [5/9] Checking BDD100K dataset preprocessing...
if not exist "bdd100k_videos_train_00\" (
    echo ERROR: BDD100K dataset folder not found!
    echo.
    echo Expected location: %cd%\bdd100k_videos_train_00\
    echo This folder should contain 1000 .mov video files (18GB total)
    echo.
    echo Please ensure the BDD100K dataset exists at this location.
    echo Download from: https://bdd-data.berkeley.edu/
    echo.
    pause
    exit /b 1
)

if not exist "datasets\processed\train_sequences.json" (
    echo.
    echo   ========================================================================
    echo   PREPROCESSING REQUIRED
    echo   ========================================================================
    echo.
    echo   Your BDD100K videos need to be processed into training data.
    echo   This is a ONE-TIME process with the following steps:
    echo.
    echo   1. Extract 40 frames from 100 videos (stride 10)
    echo   2. Run YOLOv8 segmentation on all frames
    echo   3. Generate policy maps for mobility guidance
    echo   4. Create train/val sequence indices
    echo.
    echo   Configuration:
    echo     - Videos to process: 100 out of 1000 available
    echo     - Frames per video: 40
    echo     - Frame stride: 10
    echo     - Total frames: ~4,000
    echo     - Estimated time: 5-6 hours
    echo     - Disk space needed: ~15 GB
    echo.
    echo   ========================================================================
    echo   Press Ctrl+C now to cancel, or any key to begin preprocessing...
    echo   ========================================================================
    pause >nul
    echo.
    echo   Starting BDD100K preprocessing pipeline...
    python scripts/preprocess_bdd100k.py --config config/mobility_config.py --max-videos 100 --frames-per-video 40 --stride 10 --val-ratio 0.15 --enable-seg --enable-policy
    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Preprocessing failed!
        echo Please check the error messages above.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo   ========================================================================
    echo   PREPROCESSING COMPLETE!
    echo   ========================================================================
) else (
    echo   Status: Dataset already preprocessed
    python -c "import json; train=json.load(open('datasets/processed/train_sequences.json')); val=json.load(open('datasets/processed/val_sequences.json')); print(f'  Train sequences: {len(train)}'); print(f'  Val sequences: {len(val)}')" 2>nul
)
echo.
        echo.
        echo To resume preprocessing later, run:
        echo   python scripts/preprocess_full_bdd100k.py --step extract
        echo   python scripts/preprocess_full_bdd100k.py --step segment
        echo   python scripts/preprocess_full_bdd100k.py --step depth
        echo   python scripts/preprocess_full_bdd100k.py --step indices
        pause
        exit /b 1
    )
    echo   Status: Preprocessing complete
) else (
    echo   Status: Dataset already preprocessed
)
echo.

REM Step 7: Validate Dataset Structure
echo [7/9] Validating dataset structure...
python -c "from training_tracker import validate_dataset_structure; import sys; sys.exit(0 if validate_dataset_structure() else 1)"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Dataset validation failed!
    echo Please check the error messages above and ensure all required files exist.
    echo.
    echo Required structure:
    echo   data/frames/
    echo   data/generated_frames/
    echo   data/masks/
    echo   data/policy_maps/
    echo.
    echo Run generate_synthetic_data.py to create missing files:
    echo   python generate_synthetic_data.py
    echo.
    pause
    exit /b 1
)
echo   Status: Dataset validation passed
echo.

REM Step 8: Check GPU Availability
echo [8/9] Detecting hardware...
python -c "import torch; print('  GPU Available: ' + ('YES - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO (using CPU)')); print('  Device:', 'CUDA' if torch.cuda.is_available() else 'CPU'); print('  PyTorch:', torch.__version__)"
echo.

REM Step 8: Create output directories
echo [8/9] Creating output directories...
if not exist "output\" mkdir output
if not exist "output\checkpoints\" mkdir output\checkpoints
if not exist "logs\" mkdir logs
echo   Status: Output directories ready
echo.

REM Step 9: Start Training
echo [9/9] Starting training process...
echo ========================================================================
echo.
echo   TRAINING CONFIGURATION:
echo   - Epochs: 100
echo   - Batch Size: 4
echo   - Dataset: BDD100K (100 videos, ~4,000 frames)
echo   - Train/Val: 85%/15% split
echo   - GPU: Auto-detected
echo   - Checkpoints: Saved every epoch
echo   - Progress: Live updates to IEEE_PAPER_DATA.md
echo.
echo   Press Ctrl+C to pause training (use resume.bat to continue)
echo.
echo ========================================================================
echo.

REM Run training with live output
python core/train.py --config config/mobility_config.py --epochs 100 --batch-size 4 --gpu-auto

REM Check training exit status
if %errorlevel% equ 0 (
    echo.
    echo ========================================================================
    echo   TRAINING COMPLETED SUCCESSFULLY!
    echo ========================================================================
    echo.
    echo   Next steps:
    echo   1. Review results in output/checkpoints/
    echo   2. Check updated metrics in IEEE_PAPER_DATA.md
    echo   3. Run tests: python tests/smoke_test.py
    echo   4. Commit to GitHub: git add . ^&^& git commit -m "Training complete"
    echo.
    echo   Press any key to exit...
    echo ========================================================================
    pause >nul
) else (
    echo.
    echo ========================================================================
    echo   TRAINING INTERRUPTED OR FAILED
    echo ========================================================================
    echo.
    echo   Exit code: %errorlevel%
    echo.
    echo   To resume training, run: resume.bat
    echo   To restart from scratch, run: train.bat
    echo.
    echo   Press any key to exit...
    echo ========================================================================
    pause >nul
    exit /b %errorlevel%
)

endlocal
