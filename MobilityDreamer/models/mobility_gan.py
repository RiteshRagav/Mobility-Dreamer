# -*- coding: utf-8 -*-
"""
MobilityDreamer Generator (skeleton).
Components:
- Semantic encoder (2D)
- Policy encoder (2D)
- Temporal encoder (3D)
- Fusion + decoder to RGB frames

This is a lightweight, training-ready scaffold inspired by CityDreamer4D
but simplified for rapid iteration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_c, out_c, k=3, s=1, p=1, norm=True):
    layers = [nn.Conv2d(in_c, out_c, k, s, p), nn.ReLU(inplace=True)]
    if norm:
        layers.insert(1, nn.BatchNorm2d(out_c))
    return nn.Sequential(*layers)


def conv3d_block(in_c, out_c, k=3, s=1, p=1, norm=True):
    layers = [nn.Conv3d(in_c, out_c, k, s, p), nn.ReLU(inplace=True)]
    if norm:
        layers.insert(1, nn.BatchNorm3d(out_c))
    return nn.Sequential(*layers)


class SemanticEncoder(nn.Module):
    def __init__(self, in_ch=19, out_ch=64, n_blocks=4):
        super().__init__()
        blocks = []
        ch = in_ch
        for _ in range(n_blocks):
            blocks.append(conv_block(ch, out_ch))
            ch = out_ch
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        # x: (B*T, C_sem, H, W)
        return self.net(x)


class PolicyEncoder(nn.Module):
    def __init__(self, in_ch=7, out_ch=64, n_blocks=3):
        super().__init__()
        blocks = []
        ch = in_ch
        for _ in range(n_blocks):
            blocks.append(conv_block(ch, out_ch))
            ch = out_ch
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class TemporalEncoder3D(nn.Module):
    def __init__(self, in_ch=128, hidden=256, n_layers=3):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_layers):
            layers.append(conv3d_block(ch, hidden))
            ch = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, T, H, W)
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, in_ch=256, base=128, out_ch=3):
        super().__init__()
        self.up1 = conv_block(in_ch, base)
        self.up2 = conv_block(base, base // 2)
        self.out = nn.Conv2d(base // 2, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B*T, C, H, W)
        x = self.up1(x)
        x = self.up2(x)
        x = torch.tanh(self.out(x))  # keep in [-1, 1]
        return x


class MobilityGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        net_cfg = cfg.NETWORK.MOBILITY_GAN
        self.semantic_enc = SemanticEncoder(
            in_ch=cfg.DATASETS.BDD100K.N_CLASSES,
            out_ch=net_cfg.SEMANTIC_ENCODER.OUT_DIM,
            n_blocks=net_cfg.SEMANTIC_ENCODER.N_BLOCKS,
        )
        self.policy_enc = PolicyEncoder(
            in_ch=cfg.DATASETS.BDD100K.POLICY_CLASSES,
            out_ch=net_cfg.POLICY_ENCODER.OUT_DIM,
            n_blocks=net_cfg.POLICY_ENCODER.N_LAYERS,
        )
        fused_ch = net_cfg.SEMANTIC_ENCODER.OUT_DIM + net_cfg.POLICY_ENCODER.OUT_DIM + 3  # include RGB
        self.fuse = conv_block(fused_ch, 128)
        self.temp_enc = TemporalEncoder3D(in_ch=128, hidden=net_cfg.TEMPORAL_ENCODER.HIDDEN_DIM,
                                          n_layers=net_cfg.TEMPORAL_ENCODER.N_LAYERS)
        self.decoder = Decoder(in_ch=net_cfg.TEMPORAL_ENCODER.HIDDEN_DIM)

    def forward(self, frames, segmentation, policy):
        """
        Args:
            frames: (B, T, 3, H, W) in [-1,1]
            segmentation: (B, T, C_sem, H, W) one-hot
            policy: (B, T, C_policy, H, W) one-hot
        Returns:
            gen_frames: (B, T, 3, H, W)
        """
        B, T, _, H, W = frames.shape
        # flatten time for encoders
        seg_flat = segmentation.reshape(B * T, *segmentation.shape[2:])
        pol_flat = policy.reshape(B * T, *policy.shape[2:])
        frm_flat = frames.reshape(B * T, *frames.shape[2:])

        seg_feat = self.semantic_enc(seg_flat)
        pol_feat = self.policy_enc(pol_flat)
        fused = torch.cat([frm_flat, seg_feat, pol_feat], dim=1)
        fused = self.fuse(fused)  # (B*T, 128, H, W)

        fused_3d = fused.view(B, T, fused.shape[1], H, W).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        temp_feat = self.temp_enc(fused_3d)  # (B, C, T, H, W)

        temp_flat = temp_feat.permute(0, 2, 1, 3, 4).contiguous().view(B * T, temp_feat.shape[1], H, W)
        out_flat = self.decoder(temp_flat)
        out = out_flat.view(B, T, 3, H, W)
        return out


if __name__ == "__main__":
    from config.mobility_config import cfg
    g = MobilityGenerator(cfg)
    x = torch.randn(1, 2, 3, 128, 128)
    seg = torch.zeros(1, 2, cfg.DATASETS.BDD100K.N_CLASSES, 128, 128)
    pol = torch.zeros(1, 2, cfg.DATASETS.BDD100K.POLICY_CLASSES, 128, 128)
    y = g(x, seg, pol)
    print("Generator output:", y.shape, y.min().item(), y.max().item())
