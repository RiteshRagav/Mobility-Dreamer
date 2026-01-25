# -*- coding: utf-8 -*-
#
# @File:   mobility_config.py
# @Author: MobilityDreamer Team
# @Date:   2026-01-23
# @Email:  your@email.com

from easydict import EasyDict

# fmt: off
__C                                             = EasyDict()
cfg                                             = __C

#
# Dataset Config
#
cfg.DATASETS                                     = EasyDict()
cfg.DATASETS.BDD100K                             = EasyDict()
cfg.DATASETS.BDD100K.VIDEO_DIR                   = "./bdd100k_videos_train_00/bdd100k/videos/train"
cfg.DATASETS.BDD100K.FRAMES_DIR                  = "./data/frames"
cfg.DATASETS.BDD100K.MASKS_DIR                   = "./data/masks"
cfg.DATASETS.BDD100K.POLICY_DIR                  = "./data/policy_maps"
cfg.DATASETS.BDD100K.DEPTH_DIR                   = "./data/depth_maps"
cfg.DATASETS.BDD100K.PROCESSED_DIR               = "./datasets/processed"
cfg.DATASETS.BDD100K.IMAGE_SIZE                  = (256, 256)  # H, W (resized for faster training)
cfg.DATASETS.BDD100K.SEQUENCE_LENGTH             = 4           # Number of consecutive frames (reduced for testing)
cfg.DATASETS.BDD100K.FRAME_SKIP                  = 2           # Skip every N frames for diversity
cfg.DATASETS.BDD100K.N_TRAIN_SEQUENCES           = 2000        # Number of training sequences
cfg.DATASETS.BDD100K.N_VAL_SEQUENCES             = 300         # Number of validation sequences
cfg.DATASETS.BDD100K.N_CLASSES                   = 19          # Cityscapes classes
cfg.DATASETS.BDD100K.CLASSES                     = {
    "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4,
    "pole": 5, "traffic_light": 6, "traffic_sign": 7, "vegetation": 8,
    "terrain": 9, "sky": 10, "person": 11, "rider": 12, "car": 13,
    "truck": 14, "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18
}
cfg.DATASETS.BDD100K.POLICY_CLASSES              = 7           # Policy intervention types
cfg.DATASETS.BDD100K.POLICY_TYPES                = {
    "none": 0, "bike_lane": 1, "pedestrian_zone": 2, "ev_station": 3,
    "green_space": 4, "traffic_calming": 5, "bus_lane": 6
}

#
# Constants
#
cfg.CONST                                        = EasyDict()
cfg.CONST.EXP_NAME                               = "mobilitydreamer_baseline"
cfg.CONST.N_WORKERS                              = 4
cfg.CONST.DATASET                                = "BDD100K"
cfg.CONST.DEVICE                                 = "cuda"       # or "cpu"
cfg.CONST.SEED                                   = 42

#
# Directories
#
cfg.DIR                                          = EasyDict()
cfg.DIR.OUTPUT                                   = "./output"
cfg.DIR.CHECKPOINTS                              = "./output/checkpoints"
cfg.DIR.LOGS                                     = "./output/logs"
cfg.DIR.VISUALIZATIONS                           = "./output/visualizations"

#
# WandB / Experiment Tracking
#
cfg.WANDB                                        = EasyDict()
cfg.WANDB.ENABLED                                = True
cfg.WANDB.PROJECT                                = "MobilityDreamer"
cfg.WANDB.ENTITY                                 = "your-wandb-username"  # UPDATE THIS
cfg.WANDB.MODE                                   = "online"    # "online", "offline", "disabled"
cfg.WANDB.RUN_ID                                 = None
cfg.WANDB.LOG_CODE                               = True
cfg.WANDB.SYNC_TENSORBOARD                       = True

#
# Network Architecture
#
cfg.NETWORK                                      = EasyDict()
cfg.NETWORK.MOBILITY_GAN                         = EasyDict()

# Generator
cfg.NETWORK.MOBILITY_GAN.GENERATOR               = EasyDict()
cfg.NETWORK.MOBILITY_GAN.GENERATOR.TYPE          = "TEMPORAL_POLICY"  # Our novel architecture
cfg.NETWORK.MOBILITY_GAN.GENERATOR.STYLE_DIM     = 256        # Style vector dimension

# Semantic Encoder (adapted from CityDreamer4D)
cfg.NETWORK.MOBILITY_GAN.SEMANTIC_ENCODER        = EasyDict()
cfg.NETWORK.MOBILITY_GAN.SEMANTIC_ENCODER.TYPE   = "GLOBAL"   # "GLOBAL" or "LOCAL"
cfg.NETWORK.MOBILITY_GAN.SEMANTIC_ENCODER.OUT_DIM = 64
cfg.NETWORK.MOBILITY_GAN.SEMANTIC_ENCODER.N_BLOCKS = 6

# Policy Encoder (NEW - our contribution)
cfg.NETWORK.MOBILITY_GAN.POLICY_ENCODER          = EasyDict()
cfg.NETWORK.MOBILITY_GAN.POLICY_ENCODER.IN_CHANNELS = 7        # Policy types
cfg.NETWORK.MOBILITY_GAN.POLICY_ENCODER.HIDDEN_DIM = 128
cfg.NETWORK.MOBILITY_GAN.POLICY_ENCODER.OUT_DIM  = 64
cfg.NETWORK.MOBILITY_GAN.POLICY_ENCODER.N_LAYERS = 4

# Temporal Encoder (NEW - our contribution)
cfg.NETWORK.MOBILITY_GAN.TEMPORAL_ENCODER        = EasyDict()
cfg.NETWORK.MOBILITY_GAN.TEMPORAL_ENCODER.TYPE   = "3D_CONV"   # "3D_CONV" or "TRANSFORMER"
cfg.NETWORK.MOBILITY_GAN.TEMPORAL_ENCODER.HIDDEN_DIM = 256
cfg.NETWORK.MOBILITY_GAN.TEMPORAL_ENCODER.N_LAYERS = 4
cfg.NETWORK.MOBILITY_GAN.TEMPORAL_ENCODER.TEMPORAL_KERNEL = 3  # For 3D conv

# Renderer (adapted from CityDreamer4D RenderMLP + RenderCNN)
cfg.NETWORK.MOBILITY_GAN.RENDERER                = EasyDict()
cfg.NETWORK.MOBILITY_GAN.RENDERER.TYPE           = "HYBRID"    # "VOLUME" or "FEATURE" or "HYBRID"
cfg.NETWORK.MOBILITY_GAN.RENDERER.HIDDEN_DIM     = 256
cfg.NETWORK.MOBILITY_GAN.RENDERER.N_LAYERS       = 8
cfg.NETWORK.MOBILITY_GAN.RENDERER.USE_SKIP_CONNECTIONS = True

# Discriminator
cfg.NETWORK.MOBILITY_GAN.DISCRIMINATOR           = EasyDict()
cfg.NETWORK.MOBILITY_GAN.DISCRIMINATOR.TYPE      = "MULTI_SCALE_TEMPORAL"
cfg.NETWORK.MOBILITY_GAN.DISCRIMINATOR.N_SCALES  = 3
cfg.NETWORK.MOBILITY_GAN.DISCRIMINATOR.BASE_CHANNELS = 64
cfg.NETWORK.MOBILITY_GAN.DISCRIMINATOR.TEMPORAL  = True        # Enable temporal discriminator

#
# Training
#
cfg.TRAIN                                        = EasyDict()
cfg.TRAIN.N_EPOCHS                               = 100
cfg.TRAIN.BATCH_SIZE                             = 4           # Per GPU
cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS            = 4           # Effective batch size: 4*4=16
cfg.TRAIN.CKPT_SAVE_FREQ                         = 5           # Save every N epochs
cfg.TRAIN.VAL_FREQ                               = 1           # Validate every N epochs
cfg.TRAIN.VIS_FREQ                               = 500         # Visualize every N iterations
cfg.TRAIN.LOG_FREQ                               = 50          # Log every N iterations

# Optimizer
cfg.TRAIN.OPTIMIZER                              = EasyDict()
cfg.TRAIN.OPTIMIZER.TYPE                         = "Adam"
cfg.TRAIN.OPTIMIZER.LR_G                         = 1e-4        # Generator learning rate
cfg.TRAIN.OPTIMIZER.LR_D                         = 1e-5        # Discriminator learning rate
cfg.TRAIN.OPTIMIZER.BETAS                        = (0.0, 0.999)
cfg.TRAIN.OPTIMIZER.EPS                          = 1e-7
cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY                 = 0

# Learning Rate Scheduler
cfg.TRAIN.SCHEDULER                              = EasyDict()
cfg.TRAIN.SCHEDULER.TYPE                         = "CosineAnnealing"  # "StepLR", "CosineAnnealing"
cfg.TRAIN.SCHEDULER.WARMUP_EPOCHS                = 5
cfg.TRAIN.SCHEDULER.MIN_LR                       = 1e-6

# Loss Weights
cfg.TRAIN.LOSS                                   = EasyDict()
cfg.TRAIN.LOSS.RECONSTRUCTION_WEIGHT             = 10.0        # L1 + L2 pixel loss
cfg.TRAIN.LOSS.PERCEPTUAL_WEIGHT                 = 10.0        # VGG perceptual loss
cfg.TRAIN.LOSS.GAN_WEIGHT                        = 0.5         # Adversarial loss
cfg.TRAIN.LOSS.TEMPORAL_WEIGHT                   = 5.0         # NEW: Temporal consistency
cfg.TRAIN.LOSS.POLICY_WEIGHT                     = 2.0         # NEW: Policy adherence
cfg.TRAIN.LOSS.SEMANTIC_WEIGHT                   = 3.0         # NEW: Semantic consistency

# Perceptual Loss
cfg.TRAIN.LOSS.PERCEPTUAL                        = EasyDict()
cfg.TRAIN.LOSS.PERCEPTUAL.MODEL                  = "vgg19"
cfg.TRAIN.LOSS.PERCEPTUAL.LAYERS                 = ["relu_3_1", "relu_4_1", "relu_5_1"]
cfg.TRAIN.LOSS.PERCEPTUAL.WEIGHTS                = [0.125, 0.25, 1.0]

# Discriminator Warmup
cfg.TRAIN.DISCRIMINATOR                          = EasyDict()
cfg.TRAIN.DISCRIMINATOR.WARMUP_ITERS             = 1000        # Warmup iterations for D
cfg.TRAIN.DISCRIMINATOR.UPDATE_FREQ              = 1           # Update D every N G updates

# EMA (Exponential Moving Average)
cfg.TRAIN.EMA                                    = EasyDict()
cfg.TRAIN.EMA.ENABLED                            = True
cfg.TRAIN.EMA.DECAY                              = 0.999
cfg.TRAIN.EMA.UPDATE_FREQ                        = 10          # Update EMA every N iterations

# Mixed Precision Training
cfg.TRAIN.AMP                                    = EasyDict()
cfg.TRAIN.AMP.ENABLED                            = True        # Use automatic mixed precision
cfg.TRAIN.AMP.OPT_LEVEL                          = "O1"

# Data Augmentation
cfg.TRAIN.AUGMENTATION                           = EasyDict()
cfg.TRAIN.AUGMENTATION.RANDOM_CROP               = True
cfg.TRAIN.AUGMENTATION.CROP_SIZE                 = (384, 512)  # H, W
cfg.TRAIN.AUGMENTATION.RANDOM_FLIP               = True
cfg.TRAIN.AUGMENTATION.COLOR_JITTER              = True
cfg.TRAIN.AUGMENTATION.NORMALIZE                 = True

#
# Testing / Inference
#
cfg.TEST                                         = EasyDict()
cfg.TEST.BATCH_SIZE                              = 1
cfg.TEST.N_SEQUENCES                             = 100          # Number of test sequences
cfg.TEST.SAVE_VISUALIZATIONS                     = True
cfg.TEST.COMPUTE_METRICS                         = True
cfg.TEST.USE_EMA                                 = True         # Use EMA model for inference

# Evaluation Metrics
cfg.TEST.METRICS                                 = EasyDict()
cfg.TEST.METRICS.FID                             = True         # Fréchet Inception Distance
cfg.TEST.METRICS.FVD                             = True         # Fréchet Video Distance
cfg.TEST.METRICS.LPIPS                           = True         # Perceptual similarity
cfg.TEST.METRICS.PSNR                            = True
cfg.TEST.METRICS.SSIM                            = True
cfg.TEST.METRICS.TEMPORAL_CONSISTENCY            = True
cfg.TEST.METRICS.POLICY_ADHERENCE                = True
cfg.TEST.METRICS.SEMANTIC_CONSISTENCY            = True

# fmt: on
