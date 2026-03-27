# MobilityDreamer

Implementation-aligned repository for the paper **“MobilityDreamer: A Practical Policy-Conditioned Framework for Urban Traffic Video Generation from BDD100K.”**

This codebase focuses on the implemented baseline described in the paper:
- BDD100K preprocessing into training-ready temporal sequences
- Policy-conditioned temporal GAN training
- Multi-loss optimization (GAN, reconstruction, perceptual, temporal, policy, semantic)
- Reproducible train/validation split indexing

## Scope

The primary research pipeline in this repository is the **GAN baseline**.  
ControlNet-related assets under `models/controlnet_finetuned/` are retained only as an extension path and are not required for the baseline experiments reported in the paper.

## Project Structure

```text
MobilityDreamer/
├── config/
│   ├── mobility_config.py          # Core training/data configuration
│   └── default.yaml                # Optional production-style config
├── core/
│   └── train.py                    # Baseline training entrypoint
├── datasets/
│   ├── bdd100k_dataset.py          # Temporal dataset loader
│   ├── transforms.py               # Data transforms
│   └── processed/                  # train_sequences.json / val_sequences.json
├── losses/                         # GAN/recon/perceptual/temporal/policy/semantic losses
├── models/
│   ├── mobility_gan.py             # Generator
│   ├── discriminator.py            # Discriminator
│   └── controlnet_finetuned/       # Optional extension assets
├── scripts/
│   ├── preprocess_full_bdd100k.py  # Main preprocessing pipeline
│   └── create_sequence_index.py    # Sequence index utility
├── tests/
│   ├── smoke_test.py
│   └── quick_train_test.py
├── requirements.txt
├── train.bat
└── resume.bat
```

## Data Requirements

Expected BDD100K video source path (default):
`./bdd100k_videos_train_00/bdd100k/videos/train`

Training expects these generated folders/files:
- `data/frames/`
- `data/masks/`
- `data/policy_maps/`
- `datasets/processed/train_sequences.json`
- `datasets/processed/val_sequences.json`

`data/depth_maps/` is optional for baseline training.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Preprocessing

Run full preprocessing (extract → segmentation → depth/policy → indices):

```bash
python scripts/preprocess_full_bdd100k.py --full
```

Or run in stages:

```bash
python scripts/preprocess_full_bdd100k.py --step extract
python scripts/preprocess_full_bdd100k.py --step segment
python scripts/preprocess_full_bdd100k.py --step depth
python scripts/preprocess_full_bdd100k.py --step indices
```

## Training

```bash
python core/train.py
```

Windows helper:

```bash
train.bat
```

## Quick Validation

```bash
python tests/smoke_test.py
python tests/quick_train_test.py
```

## Notes on Reproducibility

- Default baseline settings are defined in `config/mobility_config.py`.
- The paper-reported baseline corresponds to sequence length `T=4`, policy classes `7`, and BDD100K-derived train/val sequence indices.
- Generated artifacts and large datasets are intentionally excluded from versioned source where possible.
