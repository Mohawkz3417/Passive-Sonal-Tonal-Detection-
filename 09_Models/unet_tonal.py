"""
unet_tonal.py
==============
Lightweight U-Net baseline (Module: "Baseline Model" in the proposal),
used purely for experimental comparison against SegFormer-B0. This is
intentionally a small U-Net (4 down/up stages, narrow channel widths)
so it trains and runs at a comparable speed/size to the B0 transformer,
keeping the comparison fair.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class UNetTonal(nn.Module):
    def __init__(self, in_ch=1, num_classes=1, base=24):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16

        self.enc1 = conv_block(in_ch, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)
        self.enc4 = conv_block(c3, c4)
        self.bottleneck = conv_block(c4, c5)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec4 = conv_block(c5, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = conv_block(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = conv_block(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = conv_block(c2, c1)

        self.classifier = nn.Conv2d(c1, num_classes, kernel_size=1)

    @staticmethod
    def _pad_to_multiple(x, multiple=16):
        _, _, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, H, W

    def forward(self, x):
        x, H0, W0 = self._pad_to_multiple(x, 16)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.classifier(d1)
        return out[:, :, :H0, :W0]

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = UNetTonal()
    print(f"UNetTonal params: {model.num_parameters() / 1e6:.2f} M")
    x = torch.randn(2, 1, 513, 247)
    with torch.no_grad():
        y = model(x)
    print("input", x.shape, "-> output", y.shape)
