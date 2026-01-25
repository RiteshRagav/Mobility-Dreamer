# -*- coding: utf-8 -*-
"""Policy adherence loss for ensuring generated frames respect policy interventions."""
import torch
import torch.nn.functional as F

def policy_loss(pred_seq, policy_masks, strength=0.5):
    """
    Policy adherence loss: encourage generated frames to show changes in policy regions.
    
    Args:
        pred_seq: (B, T, 3, H, W) generated frames in [-1, 1]
        policy_masks: (B, T, C_policy, H, W) one-hot policy masks
        strength: weight for policy enforcement
        
    Returns:
        loss: scalar tensor
    """
    # Extract policy regions (any non-background class)
    # Assume class 0 is "no intervention"
    if policy_masks.shape[2] <= 1:
        return torch.tensor(0.0, device=pred_seq.device)
    
    policy_active = policy_masks[:, :, 1:].sum(dim=2, keepdim=True)  # (B, T, 1, H, W)
    policy_active = (policy_active > 0.5).float()
    
    # In policy regions, encourage brighter/more saturated output (visible change)
    # This is a simple heuristic - can be improved with learned features
    pred_intensity = torch.mean(torch.abs(pred_seq), dim=2, keepdim=True)  # (B, T, 1, H, W)
    
    # We want higher intensity in policy regions
    target_intensity = torch.ones_like(pred_intensity) * 0.5
    masked_diff = policy_active * (target_intensity - pred_intensity)
    
    loss = strength * torch.mean(F.relu(masked_diff))  # Penalize when intensity is too low
    
    return loss
