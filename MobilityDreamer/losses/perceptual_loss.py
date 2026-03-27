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


_VGG_MODEL = None

class VGGPerceptual(nn.Module):
    def __init__(self, layers=(3, 8, 17)):
        super().__init__()
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
            self.slices = nn.ModuleList()
            last = 0
            for l in layers:
                self.slices.append(vgg[last : l + 1].eval())
                last = l + 1
            for p in self.parameters():
                p.requires_grad = False
            self.failed = False
        except Exception as e:
            print(f"Warning: Failed to load VGG weights ({e}). Falling back to L1.")
            self.failed = True

    def forward(self, x):
        if self.failed:
            return None
        feats = []
        h = x
        for sl in self.slices:
            h = sl(h)
            feats.append(h)
        return feats


def normalize_img(x):
    # x in [-1,1]
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x + 1.0) / 2.0
    return (x - mean) / std


def perceptual_loss(x, y, weights=(0.125, 0.25, 1.0)):
    global _VGG_MODEL
    if not _HAS_TORCHVISION:
        return F.l1_loss(x, y)
    
    if _VGG_MODEL is None:
        _VGG_MODEL = VGGPerceptual()
    
    # Move VGG to correct device once (not every call)
    if not _VGG_MODEL.failed:
        _VGG_MODEL = _VGG_MODEL.to(x.device)
        
    if _VGG_MODEL.failed:
        return F.l1_loss(x, y)
        
    x_n = normalize_img(x)
    y_n = normalize_img(y)
    with torch.no_grad():
        fx = _VGG_MODEL(x_n)
        fy = _VGG_MODEL(y_n)
    
    if fx is None or fy is None:
        return F.l1_loss(x, y)
        
    loss = 0.0
    for fxi, fyi, w in zip(fx, fy, weights):
        loss += w * F.l1_loss(fxi, fyi)
    return loss
