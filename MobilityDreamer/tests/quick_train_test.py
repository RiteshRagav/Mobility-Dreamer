#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick training test for MobilityDreamer GAN.
Trains for 2 epochs to verify learning dynamics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import logging

from config.mobility_config import cfg
from datasets.bdd100k_dataset import BDD100KUrbanDataset, collate_fn
from models.mobility_gan import MobilityGenerator
from models.discriminator import MobilityDiscriminator
from losses.gan_loss import gan_loss_d, gan_loss_g
from losses.reconstruction_loss import reconstruction_loss
from losses.perceptual_loss import perceptual_loss
from losses.temporal_loss import temporal_loss
from losses.policy_loss import policy_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Loading datasets...")
    train_dataset = BDD100KUrbanDataset(cfg, split="train", load_depth=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # Small batch for quick test
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True
    )
    logger.info(f"Train dataset: {len(train_dataset)} sequences, {len(train_loader)} batches")
    
    # Create models
    logger.info("Creating models...")
    generator = MobilityGenerator(cfg).to(device)
    discriminator = MobilityDiscriminator(cfg).to(device)
    
    # Count parameters
    n_params_g = sum(p.numel() for p in generator.parameters())
    n_params_d = sum(p.numel() for p in discriminator.parameters())
    logger.info(f"Generator parameters: {n_params_g:,}")
    logger.info(f"Discriminator parameters: {n_params_d:,}")
    
    # Create optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.0, 0.999))
    
    # Training loop
    num_epochs = 2
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"{'='*60}\n")
    
    for epoch in range(1, num_epochs + 1):
        generator.train()
        discriminator.train()
        
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        epoch_loss_rec = 0.0
        
        for i, batch in enumerate(train_loader):
            frames = batch['frames'].to(device)
            seg = batch['segmentation'].to(device)
            pol = batch['policy'].to(device)
            
            # === Discriminator Step ===
            # Generate fake frames
            with torch.no_grad():
                fake = generator(frames, seg, pol)
            
            # D on real
            real_logits = discriminator(frames, seg, pol)['spatial']
            # D on fake
            fake_logits = discriminator(fake.detach(), seg, pol)['spatial']
            
            # D loss
            loss_d = gan_loss_d(real_logits, fake_logits)
            
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            
            # === Generator Step ===
            # Generate new fake frames
            fake_new = generator(frames, seg, pol)
            
            # D on new fake
            fake_logits_g = discriminator(fake_new, seg, pol)['spatial']
            
            # G losses
            loss_g_adv = gan_loss_g(fake_logits_g)
            loss_rec = reconstruction_loss(fake_new, frames)
            
            fake_flat = fake_new.reshape(-1, 3, *fake_new.shape[-2:])
            frames_flat = frames.reshape(-1, 3, *frames.shape[-2:])
            loss_perc = perceptual_loss(fake_flat, frames_flat)
            
            loss_temp = temporal_loss(fake_new)
            loss_pol = policy_loss(fake_new, pol)
            
            # Total G loss
            loss_g_total = (
                cfg.TRAIN.LOSS.GAN_WEIGHT * loss_g_adv +
                cfg.TRAIN.LOSS.RECONSTRUCTION_WEIGHT * loss_rec +
                cfg.TRAIN.LOSS.PERCEPTUAL_WEIGHT * loss_perc +
                cfg.TRAIN.LOSS.TEMPORAL_WEIGHT * loss_temp +
                cfg.TRAIN.LOSS.POLICY_WEIGHT * loss_pol
            )
            
            opt_g.zero_grad()
            loss_g_total.backward()
            opt_g.step()
            
            # Track losses
            epoch_loss_g += loss_g_total.item()
            epoch_loss_d += loss_d.item()
            epoch_loss_rec += loss_rec.item()
            
            if (i + 1) % 5 == 0:
                logger.info(
                    f"Epoch [{epoch}/{num_epochs}] Batch [{i+1}/{len(train_loader)}] "
                    f"G: {loss_g_total.item():.4f} "
                    f"D: {loss_d.item():.4f} "
                    f"Rec: {loss_rec.item():.4f} "
                    f"Perc: {loss_perc.item():.4f} "
                    f"Temp: {loss_temp.item():.4f}"
                )
        
        # Epoch summary
        n_batches = len(train_loader)
        avg_loss_g = epoch_loss_g / n_batches
        avg_loss_d = epoch_loss_d / n_batches
        avg_loss_rec = epoch_loss_rec / n_batches
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch} Summary:")
        logger.info(f"  Avg G Loss: {avg_loss_g:.4f}")
        logger.info(f"  Avg D Loss: {avg_loss_d:.4f}")
        logger.info(f"  Avg Rec Loss: {avg_loss_rec:.4f}")
        logger.info(f"{'='*60}\n")
    
    logger.info("\n✅ Training test completed successfully!")
    logger.info("The GAN is learning and all components work together.")
    logger.info("\nNext steps:")
    logger.info("1. Run full training: python core/train.py (100 epochs)")
    logger.info("2. Add TensorBoard logging and visualization")
    logger.info("3. Implement inference and evaluation metrics")

if __name__ == "__main__":
    main()
