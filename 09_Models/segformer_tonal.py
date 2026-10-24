"""
segformer_tonal.py
====================
A from-scratch, dependency-free (no internet / HuggingFace download needed)
PyTorch implementation of SegFormer-B0 (Xie et al., NeurIPS 2021),
adapted for single-channel spectrogram input and binary tonal segmentation.

Why SegFormer over a CNN/U-Net (see project proposal, Module 5):
    - Overlap patch embeddings + self-attention give every patch a *global*
      receptive field from the very first stage, so a weak but temporally
      continuous tonal line can be distinguished from spatially-isolated
      noise spikes -- something a 3x3/5x5 CNN kernel cannot do directly.
    - Mix-FFN (MLP + 3x3 depthwise conv) injects enough positional bias
      that no explicit (fixed-resolution) positional embedding is needed,
      so the network is resolution-flexible -- important because our
      spectrograms vary in duration.
    - The lightweight "All-MLP" decoder keeps the whole model small enough
      (~3.7M params for B0) for real-time CPU inference on a laptop.

Also includes a lightweight U-Net (see unet_tonal.py) as the baseline
comparison model called for in the proposal.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride,
                               padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                      # (B, C, H, W)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)       # (B, N, C)
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    """Multi-head self-attention with spatial-reduction of K/V (sr_ratio)."""

    def __init__(self, dim, num_heads, sr_ratio=1, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_)
        else:
            kv = self.kv(x)
        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MixFFN(nn.Module):
    """Linear -> 3x3 depthwise conv (positional info) -> GELU -> Linear."""

    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                                 groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, sr_ratio=1, drop=0.0, attn_drop=0.0,
                 drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DropPath(nn.Module):
    """Stochastic depth."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep) * mask


# ---------------------------------------------------------------------------
# Mix Vision Transformer encoder (MiT-B0 configuration by default)
# ---------------------------------------------------------------------------
class MixVisionTransformer(nn.Module):
    def __init__(self, in_ch=1,
                 embed_dims=(32, 64, 160, 256),
                 num_heads=(1, 2, 5, 8),
                 depths=(2, 2, 2, 2),
                 sr_ratios=(8, 4, 2, 1),
                 mlp_ratio=4,
                 patch_sizes=(7, 3, 3, 3),
                 strides=(4, 2, 2, 2),
                 drop_path_rate=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.patch_embeds = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = in_ch
        for i in range(self.num_stages):
            self.patch_embeds.append(
                OverlapPatchEmbed(in_dim, embed_dims[i], patch_sizes[i], strides[i]))
            blocks = nn.ModuleList([
                TransformerBlock(embed_dims[i], num_heads[i], mlp_ratio, sr_ratios[i],
                                  drop_path=dpr[cur + j])
                for j in range(depths[i])
            ])
            self.stages.append(blocks)
            self.norms.append(nn.LayerNorm(embed_dims[i]))
            in_dim = embed_dims[i]
            cur += depths[i]

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.stages[i]:
                x = blk(x, H, W)
            x = self.norms[i](x)
            x_spatial = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x_spatial)
            x = x_spatial  # feed spatial map into next patch-embed
        return outs  # [c1 (1/4), c2 (1/8), c3 (1/16), c4 (1/32)]


# ---------------------------------------------------------------------------
# All-MLP decode head
# ---------------------------------------------------------------------------
class SegformerHead(nn.Module):
    def __init__(self, in_dims=(32, 64, 160, 256), embed_dim=256, num_classes=1, dropout=0.1):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(d, embed_dim) for d in in_dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_dims), embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, feats):
        target_hw = feats[0].shape[2:]
        upsampled = []
        for feat, lin in zip(feats, self.linears):
            B, C, H, W = feat.shape
            x = feat.flatten(2).transpose(1, 2)      # (B, N, C)
            x = lin(x)                                 # (B, N, embed_dim)
            x = x.transpose(1, 2).reshape(B, -1, H, W)
            if (H, W) != target_hw:
                x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            upsampled.append(x)
        x = torch.cat(upsampled, dim=1)
        x = self.fuse(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x  # (B, num_classes, H/4, W/4)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class SegFormerTonal(nn.Module):
    """
    SegFormer-B0 for binary tonal segmentation of spectrograms.
    Input:  (B, 1, F, T) float tensor (e.g. normalized PCEN spectrogram)
    Output: (B, 1, F, T) raw logits (apply sigmoid for probability)
    """

    def __init__(self, in_ch=1, num_classes=1,
                 embed_dims=(32, 64, 160, 256), num_heads=(1, 2, 5, 8),
                 depths=(2, 2, 2, 2), sr_ratios=(8, 4, 2, 1),
                 decoder_dim=256):
        super().__init__()
        self.encoder = MixVisionTransformer(in_ch, embed_dims, num_heads, depths, sr_ratios)
        self.decode_head = SegformerHead(embed_dims, decoder_dim, num_classes)

    @staticmethod
    def _pad_to_multiple(x, multiple=32):
        _, _, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, H, W

    def forward(self, x):
        x_pad, H0, W0 = self._pad_to_multiple(x, 32)
        feats = self.encoder(x_pad)
        logits = self.decode_head(feats)                       # (B, C, H_pad/4, W_pad/4)
        logits = F.interpolate(logits, size=x_pad.shape[2:], mode="bilinear", align_corners=False)
        logits = logits[:, :, :H0, :W0]                         # crop back to original size
        return logits

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def segformer_b0_tonal(in_ch=1, num_classes=1) -> SegFormerTonal:
    """MiT-B0 sized model -- ~3.7M params, suited for laptop CPU deployment."""
    return SegFormerTonal(
        in_ch=in_ch, num_classes=num_classes,
        embed_dims=(32, 64, 160, 256), num_heads=(1, 2, 5, 8),
        depths=(2, 2, 2, 2), sr_ratios=(8, 4, 2, 1), decoder_dim=256,
    )


if __name__ == "__main__":
    model = segformer_b0_tonal()
    print(f"SegFormer-B0-Tonal params: {model.num_parameters() / 1e6:.2f} M")
    x = torch.randn(2, 1, 513, 247)  # matches STFT(n_fft=1024) x ~4s @16kHz
    with torch.no_grad():
        y = model(x)
    print("input", x.shape, "-> output", y.shape)
