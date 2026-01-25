# -*- coding: utf-8 -*-
"""
Simple GAN loss helpers (BCE with logits) for MobilityDreamer.
This is a lightweight replacement for CityDreamer4D's N+1 loss.
"""
import torch
import torch.nn.functional as F


def gan_loss_d(real_logits, fake_logits):
    """
    Discriminator loss (real vs fake) using BCE with logits.
    real_logits: tensor from discriminator on real samples
    fake_logits: tensor from discriminator on fake samples
    """
    real_tgt = torch.ones_like(real_logits)
    fake_tgt = torch.zeros_like(fake_logits)
    loss_real = F.binary_cross_entropy_with_logits(real_logits, real_tgt)
    loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_tgt)
    return loss_real + loss_fake


def gan_loss_g(fake_logits):
    """
    Generator loss to fool discriminator.
    """
    tgt = torch.ones_like(fake_logits)
    return F.binary_cross_entropy_with_logits(fake_logits, tgt)
