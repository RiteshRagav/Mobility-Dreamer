# -*- coding: utf-8 -*-
"""
Training loop scaffold for MobilityDreamer.
This is a minimal, end-to-end runnable skeleton intended for rapid bring-up.
"""
import os
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config.mobility_config import cfg
from datasets.bdd100k_dataset import BDD100KUrbanDataset, collate_fn
from datasets.transforms import build_default_transforms
from models.mobility_gan import MobilityGenerator
from models.discriminator import MobilityDiscriminator
from losses.gan_loss import gan_loss_d, gan_loss_g
from losses.reconstruction_loss import reconstruction_loss
from losses.perceptual_loss import perceptual_loss
from losses.temporal_loss import temporal_loss
from losses.policy_loss import policy_loss
from losses.semantic_loss import semantic_loss


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_dataloaders():
    t_train = build_default_transforms(cfg)
    train_set = BDD100KUrbanDataset(cfg, split="train", transform=t_train)
    val_set = BDD100KUrbanDataset(cfg, split="val", transform=None)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.CONST.N_WORKERS,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.CONST.N_WORKERS,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def train_one_epoch(generator, discriminator, opt_g, opt_d, loader, device):
    generator.train()
    discriminator.train()
    total_g = total_d = 0.0
    for batch in loader:
        frames = batch["frames"].to(device)
        seg = batch["segmentation"].to(device) if "segmentation" in batch else None
        pol = batch["policy"].to(device) if "policy" in batch else None
        if seg is None or pol is None:
            continue  # require conditioning
        # Generator forward
        fake = generator(frames, seg, pol)
        # Discriminator real/fake
        real_logits = discriminator(frames, seg, pol)["spatial"]
        fake_logits = discriminator(fake.detach(), seg, pol)["spatial"]

        # D loss
        loss_d = gan_loss_d(real_logits, fake_logits)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # G loss
        fake_logits_g = discriminator(fake, seg, pol)["spatial"]
        loss_g = gan_loss_g(fake_logits_g)
        loss_rec = reconstruction_loss(fake, frames)
        loss_perc = perceptual_loss(fake.view(-1, 3, *fake.shape[-2:]), frames.view(-1, 3, *frames.shape[-2:]))
        loss_temp = temporal_loss(fake)
        loss_pol = policy_loss(fake, pol)
        loss_sem = semantic_loss(fake, seg)

        total_loss_g = (
            cfg.TRAIN.LOSS.GAN_WEIGHT * loss_g
            + cfg.TRAIN.LOSS.RECONSTRUCTION_WEIGHT * loss_rec
            + cfg.TRAIN.LOSS.PERCEPTUAL_WEIGHT * loss_perc
            + cfg.TRAIN.LOSS.TEMPORAL_WEIGHT * loss_temp
            + cfg.TRAIN.LOSS.POLICY_WEIGHT * loss_pol
            + cfg.TRAIN.LOSS.SEMANTIC_WEIGHT * loss_sem
        )

        opt_g.zero_grad()
        total_loss_g.backward()
        opt_g.step()

        total_d += loss_d.item()
        total_g += total_loss_g.item()
    n = max(1, len(loader))
    return total_g / n, total_d / n


def validate(generator, discriminator, loader, device):
    generator.eval()
    discriminator.eval()
    total_rec = 0.0
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            seg = batch["segmentation"].to(device) if "segmentation" in batch else None
            pol = batch["policy"].to(device) if "policy" in batch else None
            if seg is None or pol is None:
                continue
            fake = generator(frames, seg, pol)
            total_rec += reconstruction_loss(fake, frames).item()
    n = max(1, len(loader))
    return total_rec / n


def main():
    device = torch.device(cfg.CONST.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders()

    generator = MobilityGenerator(cfg).to(device)
    discriminator = MobilityDiscriminator(cfg).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.TRAIN.OPTIMIZER.LR_G, betas=cfg.TRAIN.OPTIMIZER.BETAS)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.TRAIN.OPTIMIZER.LR_D, betas=cfg.TRAIN.OPTIMIZER.BETAS)

    for epoch in range(1, 1 + cfg.TRAIN.N_EPOCHS):
        loss_g, loss_d = train_one_epoch(generator, discriminator, opt_g, opt_d, train_loader, device)
        val_rec = validate(generator, discriminator, val_loader, device)
        logger.info(f"Epoch {epoch}: G={loss_g:.4f} D={loss_d:.4f} ValRec={val_rec:.4f}")

        if epoch % cfg.TRAIN.CKPT_SAVE_FREQ == 0:
            ckpt_dir = Path(cfg.DIR.CHECKPOINTS)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
