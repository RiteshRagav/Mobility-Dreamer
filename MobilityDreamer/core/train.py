import sys
print("DEBUG: Script started")
sys.stdout.flush()
import os
import time
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


from torchvision.utils import save_image
from tqdm import tqdm


def save_visualization(frames, fake, seg, epoch, iter_idx):
    vis_dir = Path(cfg.DIR.VISUALIZATIONS)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Save a comparison: [Real, Fake, Segmentation]
    # Pick first item in batch
    real_img = (frames[0, 0] + 1.0) / 2.0  # (3, H, W) in [0, 1]
    fake_img = (fake[0, 0] + 1.0) / 2.0
    
    # Simple color map for segmentation (first 3 channels for visual)
    # seg is (T, C, H, W)
    seg_img = seg[0, 0, :3] # Take 3 classes/channels
    
    combined = torch.cat([real_img.cpu(), fake_img.detach().cpu()], dim=2)
    save_path = vis_dir / f"epoch_{epoch}_iter_{iter_idx}.png"
    save_image(combined, save_path)
    return save_path


def train_one_epoch(generator, discriminator, opt_g, opt_d, loader, device, epoch):
    generator.train()
    discriminator.train()
    total_g = total_d = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for i, batch in enumerate(pbar):
        # Hardware throttling: prevent laptop overheating
        time.sleep(0.5)
        frames = batch["frames"].to(device)
        seg = batch["segmentation"].to(device) if "segmentation" in batch else None
        pol = batch["policy"].to(device) if "policy" in batch else None
        
        if seg is None or pol is None:
            continue

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
        
        pbar.set_postfix(G=total_loss_g.item(), D=loss_d.item())
        
        # Visualize periodically
        if (i + 1) % cfg.TRAIN.VIS_FREQ == 0:
            save_visualization(frames, fake, seg, epoch, i)

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


from training_tracker import TrainingStateTracker


def main():
    # Force low CPU usage (limit to 2 threads)
    torch.set_num_threads(2)
    device = torch.device(cfg.CONST.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders()
    
    tracker = TrainingStateTracker()
    tracker.start_training()

    generator = MobilityGenerator(cfg).to(device)
    discriminator = MobilityDiscriminator(cfg).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.TRAIN.OPTIMIZER.LR_G, betas=cfg.TRAIN.OPTIMIZER.BETAS)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.TRAIN.OPTIMIZER.LR_D, betas=cfg.TRAIN.OPTIMIZER.BETAS)

    start_epoch = 1
    ckpt_dir = Path(cfg.DIR.CHECKPOINTS)
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("epoch_*.pt"))
        if ckpts:
            # Find the latest checkpoint by epoch number
            latest_ckpt = max(ckpts, key=lambda x: int(x.stem.split('_')[1]))
            logger.info(f"Loading checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            generator.load_state_dict(checkpoint["generator"])
            discriminator.load_state_dict(checkpoint["discriminator"])
            opt_g.load_state_dict(checkpoint["opt_g"])
            opt_d.load_state_dict(checkpoint["opt_d"])
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, 1 + cfg.TRAIN.N_EPOCHS):
        loss_g, loss_d = train_one_epoch(generator, discriminator, opt_g, opt_d, train_loader, device, epoch)
        val_rec = validate(generator, discriminator, val_loader, device)
        
        # Log to paper and console via tracker
        tracker.update_epoch(epoch, float(loss_g), float(loss_d), float(val_rec))
        logger.info(f"Epoch {epoch} Sync Complete: G={loss_g:.4f} D={loss_d:.4f} ValRec={val_rec:.4f}")

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
