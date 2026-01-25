@echo off
REM ============================================================================
REM MobilityDreamer - Resume Interrupted Training
REM ============================================================================
REM This script:
REM   - Loads previous training state
REM   - Shows detailed resume information
REM   - Continues training from last checkpoint
REM   - Maintains all progress tracking
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo   MOBILITYDREAMER - RESUME TRAINING
echo ========================================================================
echo   Date: %date% %time%
echo   Location: %cd%
echo ========================================================================
echo.

REM Check if training state exists
if not exist "training_state.json" (
    echo ERROR: No previous training state found!
    echo.
    echo This means training has not been started yet, or the state file was deleted.
    echo.
    echo To start new training, run: train.bat
    echo.
    pause
    exit /b 1
)

echo [1/4] Loading previous training state...
python -c "from training_tracker import TrainingStateTracker; tracker = TrainingStateTracker(); info = tracker.get_resume_info(); print('\n  Last Training Session:'); print('  ====================='); print(f'  Epochs Completed: {info[\"epochs_completed\"]}'); print(f'  Last Processed File: {info[\"last_file\"]}'); print(f'  Files Processed: {info[\"files_processed\"]}'); print(f'  Best Generator Loss: {info[\"best_g_loss\"]:.4f}'); print(f'  Best Discriminator Loss: {info[\"best_d_loss\"]:.4f}'); print(f'  Best Reconstruction Loss: {info[\"best_rec_loss\"]:.4f}'); print(f'  Progress: {info[\"progress_percentage\"]:.1f}%%'); print(f'\n  Remaining: {info[\"remaining_epochs\"]} epochs'); print(f'  Next File: {info[\"next_file\"]}')"
if %errorlevel% neq 0 (
    echo ERROR: Failed to load training state!
    echo The state file may be corrupted.
    echo.
    pause
    exit /b 1
)
echo.

REM Activate virtual environment
echo [2/4] Activating virtual environment...
if not exist ".venv\" (
    echo ERROR: Virtual environment not found!
    echo Please run train.bat first to set up the environment.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
echo   Status: Virtual environment activated
echo.

REM Check GPU availability
echo [3/4] Detecting hardware...
python -c "import torch; print('  GPU Available: ' + ('YES - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO (using CPU)')); print('  Device:', 'CUDA' if torch.cuda.is_available() else 'CPU')"
echo.

REM Resume training
echo [4/4] Resuming training process...
echo ========================================================================
echo.
echo   RESUME CONFIGURATION:
echo   - Mode: Continue from last checkpoint
echo   - State File: training_state.json
echo   - Checkpoints: Loaded from output/checkpoints/
echo   - Progress: Live updates to IEEE_PAPER_DATA.md
echo.
echo   Press Ctrl+C to pause training (run resume.bat again to continue)
echo.
echo ========================================================================
echo.

REM Run training with resume flag
python core/train.py --config config/mobility_config.py --epochs 100 --batch-size 4 --gpu-auto --resume

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
    echo   To resume training again, run: resume.bat
    echo   To restart from scratch, run: train.bat
    echo.
    echo   Press any key to exit...
    echo ========================================================================
    pause >nul
    exit /b %errorlevel%
)

endlocal
