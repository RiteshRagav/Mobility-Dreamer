# -*- coding: utf-8 -*-
"""
MobilityDreamer Discriminator (skeleton).
Includes spatial multi-scale branch and optional temporal branch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_c, out_c, k=3, s=1, p=1, norm=True):
    layers = [nn.Conv2d(in_c, out_c, k, s, p), nn.LeakyReLU(0.2, inplace=True)]
    if norm:
        layers.insert(1, nn.BatchNorm2d(out_c))
    return nn.Sequential(*layers)


def conv3d_block(in_c, out_c, k=3, s=1, p=1, norm=True):
    layers = [nn.Conv3d(in_c, out_c, k, s, p), nn.LeakyReLU(0.2, inplace=True)]
    if norm:
        layers.insert(1, nn.BatchNorm3d(out_c))
    return nn.Sequential(*layers)


class SpatialDiscriminator(nn.Module):
    def __init__(self, in_ch=3 + 19 + 7, base=64, n_scales=3):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_scales):
            layers.append(conv_block(ch, base, s=2))
            ch = base
            base *= 2
        self.net = nn.Sequential(*layers)
        self.head = nn.Conv2d(ch, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B*T, C, H, W)
        feat = self.net(x)
        logits = self.head(feat)
        return logits


class TemporalDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base=32, n_layers=3):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_layers):
            layers.append(conv3d_block(ch, base))
            ch = base
            base *= 2
        self.net = nn.Sequential(*layers)
        self.head = nn.Conv3d(ch, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, C, T, H, W)
        feat = self.net(x)
        logits = self.head(feat)
        return logits


class MobilityDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ds = cfg.DATASETS.BDD100K
        self.spatial = SpatialDiscriminator(in_ch=3 + ds.N_CLASSES + ds.POLICY_CLASSES,
                                            base=cfg.NETWORK.MOBILITY_GAN.DISCRIMINATOR.BASE_CHANNELS,
                                            n_scales=cfg.NETWORK.MOBILITY_GAN.DISCRIMINATOR.N_SCALES)
        self.temporal_enabled = cfg.NETWORK.MOBILITY_GAN.DISCRIMINATOR.TEMPORAL
        if self.temporal_enabled:
            self.temporal = TemporalDiscriminator(in_ch=3)

    def forward(self, frames, segmentation, policy):
        """
        Args:
            frames: (B, T, 3, H, W)
            segmentation: (B, T, C_sem, H, W)
            policy: (B, T, C_policy, H, W)
        Returns:
            dict with spatial logits and optional temporal logits
        """
        B, T = frames.shape[:2]
        x_spatial = torch.cat([
            frames.reshape(B * T, *frames.shape[2:]),
            segmentation.reshape(B * T, *segmentation.shape[2:]),
            policy.reshape(B * T, *policy.shape[2:]),
        ], dim=1)
        spatial_logits = self.spatial(x_spatial)
        out = {"spatial": spatial_logits}
        if self.temporal_enabled:
            out["temporal"] = self.temporal(frames.permute(0, 2, 1, 3, 4))
        return out


if __name__ == "__main__":
    from config.mobility_config import cfg
    d = MobilityDiscriminator(cfg)
    x = torch.randn(1, 2, 3, 128, 128)
    seg = torch.zeros(1, 2, cfg.DATASETS.BDD100K.N_CLASSES, 128, 128)
    pol = torch.zeros(1, 2, cfg.DATASETS.BDD100K.POLICY_CLASSES, 128, 128)
    y = d(x, seg, pol)
    print({k: v.shape for k, v in y.items()})
