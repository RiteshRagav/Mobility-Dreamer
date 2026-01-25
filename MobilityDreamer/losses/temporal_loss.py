# -*- coding: utf-8 -*-
"""Temporal consistency loss for video generation."""
import torch
import torch.nn.functional as F

def temporal_loss(pred_seq, target_seq=None, mode='smooth'):
    """
    Temporal consistency loss for generated video sequences.
    
    Args:
        pred_seq: (B, T, C, H, W) generated frames
        target_seq: (B, T, C, H, W) target frames (optional)
        mode: 'smooth' for smoothness, 'match' for target matching
        
    Returns:
        loss: scalar tensor
    """
    if mode == 'match' and target_seq is not None:
        # Direct matching with target sequence
        return F.l1_loss(pred_seq, target_seq)
    
    # Temporal smoothness: minimize difference between consecutive frames
    if pred_seq.shape[1] < 2:
        return torch.tensor(0.0, device=pred_seq.device)
    
    diff = pred_seq[:, 1:] - pred_seq[:, :-1]  # (B, T-1, C, H, W)
    loss = torch.mean(torch.abs(diff))
    
    return loss
