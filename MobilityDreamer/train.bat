@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo   MOBILITYDREAMER - HIGH-PERFORMANCE TRAINING SYSTEM
echo ========================================================================
echo   Date: %date% %time%
echo ========================================================================

REM ------------------------------------------------------------------------
REM Step 1: Check Python
REM ------------------------------------------------------------------------
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not available in PATH.
    echo Please install Python 3.10+ and restart.
    pause
    exit /b 1
)

REM ------------------------------------------------------------------------
REM Step 2: Virtual Environment
REM ------------------------------------------------------------------------
if not exist ".venv\" (
    echo [1/9] Creating Virtual Environment...
    python -m venv .venv
)

REM ------------------------------------------------------------------------
REM Step 3: Activate Venv
REM ------------------------------------------------------------------------
call .venv\Scripts\activate.bat
echo [2/9] Virtual Environment: ACTIVE

REM ------------------------------------------------------------------------
REM Step 4: Install Dependencies
REM ------------------------------------------------------------------------
echo [3/9] Installing Dependencies...
pip install --upgrade pip >nul
pip install -r requirements.txt

REM ------------------------------------------------------------------------
REM Step 5: Verify PyTorch Installation
REM ------------------------------------------------------------------------
echo [4/9] Verifying PyTorch Installation...

python -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: PyTorch is NOT installed.
    echo Please run:
    echo     pip install torch torchvision torchaudio
    pause
    exit /b 1
)

REM ------------------------------------------------------------------------
REM Step 6: Dataset Discovery
REM ------------------------------------------------------------------------
echo [5/9] Discovering Dataset...
set DATASET_PATH=NOT_FOUND

if exist "Dataset\bdd100k_videos_train_00\bdd100k_videos_train_00" (
    set DATASET_PATH=Dataset\bdd100k_videos_train_00\bdd100k_videos_train_00
)

if "%DATASET_PATH%"=="NOT_FOUND" (
    echo   WARNING: BDD100K Videos not found.
    echo   Using Synthetic Dataset Fallback.
) else (
    echo   SUCCESS: Found BDD100K Dataset at:
    echo   %DATASET_PATH%
)

REM ------------------------------------------------------------------------
REM Step 7: Preprocessing
REM ------------------------------------------------------------------------
if not "%DATASET_PATH%"=="NOT_FOUND" (
    if not exist "datasets\processed\train_sequences.json" (
        echo [6/9] Preprocessing Real BDD100K Videos...
        python scripts/preprocess_bdd100k.py --config config/mobility_config.py --max-videos 20
        if %errorlevel% neq 0 (
            echo WARNING: Preprocessing failed. Falling back to synthetic data.
        )
    ) else (
        echo [6/9] Dataset already preprocessed.
    )
) else (
    if not exist "data\frames\" (
        echo [6/9] Generating Synthetic Dataset...
        python generate_synthetic_data.py
    ) else (
        echo [6/9] Synthetic Dataset Ready.
    )
)

REM ------------------------------------------------------------------------
REM Step 8: GPU Detection (SAFE)
REM ------------------------------------------------------------------------
echo [7/9] Detecting Hardware...
python -c "import torch; print('  >>> GPU ACCELERATION:', 'ENABLED - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'DISABLED (CPU MODE)')"

REM ------------------------------------------------------------------------
REM Step 9: Output Directories
REM ------------------------------------------------------------------------
if not exist "output\checkpoints" mkdir output\checkpoints
if not exist "output\logs" mkdir output\logs
if not exist "output\visualizations" mkdir output\visualizations

REM ------------------------------------------------------------------------
REM Step 10: Training
REM ------------------------------------------------------------------------
echo [8/9] Starting Training...
echo.
echo ========================================================================
echo   MONITORING:
echo   - Images: output\visualizations\
echo   - Logs  : tensorboard --logdir output\logs
echo   - Paper : IEEE_PAPER_DATA.md
echo ========================================================================
echo.

set PYTHONPATH=%CD%
python core/train.py --config config/mobility_config.py %*

echo.
echo ========================================================================
echo   TRAINING FINISHED
echo ========================================================================
pause
