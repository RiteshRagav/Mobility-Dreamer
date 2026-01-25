# -*- coding: utf-8 -*-
"""
Perceptual loss (VGG19). Falls back to L1 if torchvision VGG not available.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


class VGGPerceptual(nn.Module):
    def __init__(self, layers=(3, 8, 17), weights=None):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        last = 0
        for l in layers:
            self.slices.append(vgg[last : l + 1].eval())
            last = l + 1
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        feats = []
        for sl in self.slices:
            x = sl(x)
            feats.append(x)
        return feats


def normalize_img(x):
    # x in [-1,1]
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x + 1.0) / 2.0
    return (x - mean) / std


def perceptual_loss(x, y, weights=(0.125, 0.25, 1.0)):
    if not _HAS_TORCHVISION:
        return F.l1_loss(x, y)
    vgg = VGGPerceptual().to(x.device)
    x_n = normalize_img(x)
    y_n = normalize_img(y)
    fx = vgg(x_n)
    fy = vgg(y_n)
    loss = 0.0
    for fxi, fyi, w in zip(fx, fy, weights):
        loss += w * F.l1_loss(fxi, fyi)
    return loss
