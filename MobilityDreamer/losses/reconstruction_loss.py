# -*- coding: utf-8 -*-
"""Reconstruction losses: L1 + optional L2."""
import torch
import torch.nn.functional as F


def reconstruction_loss(pred, target, l2_weight=0.0):
    l1 = F.l1_loss(pred, target)
    if l2_weight > 0:
        l2 = F.mse_loss(pred, target)
        return l1 + l2_weight * l2
    return l1
