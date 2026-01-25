#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke test for MobilityDreamer GAN training pipeline.
Tests dataset loading, model forward pass, loss computation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from config.mobility_config import cfg
from datasets.bdd100k_dataset import BDD100KUrbanDataset, collate_fn
from models.mobility_gan import MobilityGenerator
from models.discriminator import MobilityDiscriminator
from losses.gan_loss import gan_loss_d, gan_loss_g
from losses.reconstruction_loss import reconstruction_loss
from losses.perceptual_loss import perceptual_loss
from losses.temporal_loss import temporal_loss
from losses.policy_loss import policy_loss

print("=" * 60)
print("MobilityDreamer GAN Pipeline Smoke Test")
print("=" * 60)

# Test 1: Dataset Loading
print("\n[1/5] Testing Dataset Loading...")
try:
    dataset = BDD100KUrbanDataset(cfg, split="train", load_depth=False)
    print(f"✅ Dataset loaded: {len(dataset)} sequences")
    
    sample = dataset[0]
    print(f"✅ Sample loaded:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"   {key}: {val.shape}")
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(loader))
    print(f"✅ DataLoader works: batch size = {batch['frames'].shape[0]}")
except Exception as e:
    print(f"❌ Dataset test failed: {e}")
    sys.exit(1)

# Test 2: Generator Forward Pass
print("\n[2/5] Testing Generator...")
try:
    generator = MobilityGenerator(cfg)
    
    frames = batch['frames']  # (B, T, 3, H, W)
    seg = batch['segmentation']  # (B, T, C_sem, H, W)
    pol = batch['policy']  # (B, T, C_policy, H, W)
    
    print(f"   Input shapes: frames={frames.shape}, seg={seg.shape}, policy={pol.shape}")
    
    with torch.no_grad():
        fake = generator(frames, seg, pol)
    
    print(f"✅ Generator forward pass: {fake.shape}")
    print(f"   Output range: [{fake.min():.3f}, {fake.max():.3f}]")
except Exception as e:
    print(f"❌ Generator test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Discriminator Forward Pass
print("\n[3/5] Testing Discriminator...")
try:
    discriminator = MobilityDiscriminator(cfg)
    
    with torch.no_grad():
        real_out = discriminator(frames, seg, pol)
        fake_out = discriminator(fake, seg, pol)
    
    print(f"✅ Discriminator forward pass:")
    print(f"   Real logits: {real_out['spatial'].shape}")
    print(f"   Fake logits: {fake_out['spatial'].shape}")
except Exception as e:
    print(f"❌ Discriminator test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Loss Computation
print("\n[4/5] Testing Loss Functions...")
try:
    # GAN losses
    loss_d = gan_loss_d(real_out['spatial'], fake_out['spatial'])
    loss_g = gan_loss_g(fake_out['spatial'])
    
    # Reconstruction loss
    loss_rec = reconstruction_loss(fake, frames)
    
    # Perceptual loss
    fake_flat = fake.reshape(-1, 3, *fake.shape[-2:])
    frames_flat = frames.reshape(-1, 3, *frames.shape[-2:])
    loss_perc = perceptual_loss(fake_flat, frames_flat)
    
    # Temporal loss
    loss_temp = temporal_loss(fake)
    
    # Policy loss
    loss_pol = policy_loss(fake, pol)
    
    print(f"✅ All losses computed:")
    print(f"   D loss: {loss_d:.4f}")
    print(f"   G loss: {loss_g:.4f}")
    print(f"   Reconstruction: {loss_rec:.4f}")
    print(f"   Perceptual: {loss_perc:.4f}")
    print(f"   Temporal: {loss_temp:.4f}")
    print(f"   Policy: {loss_pol:.4f}")
except Exception as e:
    print(f"❌ Loss test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Backward Pass
print("\n[5/5] Testing Backward Pass...")
try:
    # Create fresh models and optimizers
    gen = MobilityGenerator(cfg)
    disc = MobilityDiscriminator(cfg)
    
    opt_g = torch.optim.Adam(gen.parameters(), lr=0.0001)
    opt_d = torch.optim.Adam(disc.parameters(), lr=0.0001)
    
    # D step
    fake_detached = gen(frames, seg, pol).detach()
    real_logits = disc(frames, seg, pol)['spatial']
    fake_logits = disc(fake_detached, seg, pol)['spatial']
    loss_d = gan_loss_d(real_logits, fake_logits)
    
    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()
    print(f"✅ Discriminator backward pass successful")
    
    # G step
    fake_new = gen(frames, seg, pol)
    fake_logits_g = disc(fake_new, seg, pol)['spatial']
    loss_g_total = gan_loss_g(fake_logits_g) + 0.1 * reconstruction_loss(fake_new, frames)
    
    opt_g.zero_grad()
    loss_g_total.backward()
    opt_g.step()
    print(f"✅ Generator backward pass successful")
    
except Exception as e:
    print(f"❌ Backward pass test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nThe GAN training pipeline is ready to run.")
print("Next steps:")
print("1. Run full training: python core/train.py")
print("2. Monitor with TensorBoard (to be implemented)")
print("3. Generate samples with trained model")
