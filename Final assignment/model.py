import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from efficientvit.models.efficientvit import efficientvit_backbone_b0


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU6(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True),
        )
    def forward(self, x):
        return self.pw(self.dw(x))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)


class DAPPM(nn.Module):
    """
    Deep Aggregation Pyramid Pooling Module — from DDRNet (Hong et al. 2021).
    Much cheaper than ASPP: uses avg-pool at 4 scales + cascaded fusion.
    https://arxiv.org/abs/2101.06085
    """
    def __init__(self, in_ch, branch_ch, out_ch):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(1),
                          nn.Conv2d(in_ch, branch_ch, 1, bias=False),
                          nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True)),
            nn.Sequential(nn.AvgPool2d(5, 1, 2),
                          nn.Conv2d(in_ch, branch_ch, 1, bias=False),
                          nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True)),
            nn.Sequential(nn.AvgPool2d(9, 1, 4),
                          nn.Conv2d(in_ch, branch_ch, 1, bias=False),
                          nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True)),
            nn.Sequential(nn.AvgPool2d(17, 1, 8),
                          nn.Conv2d(in_ch, branch_ch, 1, bias=False),
                          nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True)),
        ])
        # identity branch
        self.identity = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True)
        )
        # cascaded fusion convs
        self.fuse = nn.ModuleList([
            nn.Sequential(nn.Conv2d(branch_ch, branch_ch, 3, padding=1, bias=False),
                          nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True))
            for _ in range(len(self.scales))
        ])
        self.project = nn.Sequential(
            nn.Conv2d(branch_ch * (len(self.scales) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        H, W = x.shape[2:]
        branches = [self.identity(x)]
        for scale, fuse in zip(self.scales, self.fuse):
            s = F.interpolate(scale(x), size=(H, W), mode='bilinear', align_corners=False)
            branches.append(fuse(s + branches[-1]))  # cascaded residual fusion
        return self.project(torch.cat(branches, dim=1))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )
        self.se = SEBlock(out_channels, reduction=8)

    def forward(self, x1, x2):
        x1 = F.interpolate(self.up(x1), size=x2.shape[2:],
                           mode='bilinear', align_corners=False)
        return self.se(self.conv(torch.cat([x2, x1], dim=1)))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x):
        return self.conv(x)


class Model(nn.Module):
    """
    Efficient U-Net with EfficientViT-B0 encoder, https://github.com/CVHub520/efficientvit?tab=readme-ov-file 
    DAPPM bottleneck, lightweight decoder.

    Key refs:
      - DDRNet / DAPPM: Hong et al. (2021) https://arxiv.org/abs/2101.06085
      - EfficientViT: Cai et al. (2023) https://arxiv.org/abs/2205.14756
      - MobileNetV3: Howard et al. (2019) https://arxiv.org/abs/1905.02244
    """
    def __init__(self, in_channels=3, n_classes=19, dino_fine_tune=False):
        super().__init__()
        self.in_channels = in_channels
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.backbone = efficientvit_backbone_b0(pretrained=False)

        weights_path = os.path.join(BASE_DIR, "b0.pt")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print("[Model] loaded EfficientViT-B0 from", weights_path)
            print("[Model] missing keys:", missing)
            print("[Model] unexpected keys:", unexpected)
        else:
            print("[Model] WARNING: no weights at", weights_path)

        self.reduce = nn.Sequential(
            nn.Conv2d(128, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
        )
        self.dappm = DAPPM(48, 24, 48)

        self.up1 = Up(48 + 64, 48)
        self.up2 = Up(48 + 32, 32)
        self.up3 = Up(32 + 32, 24)

        self.dropout = nn.Dropout2d(p=0.1)
        self.outc    = OutConv(24, n_classes)

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}")
        H, W = x.shape[2], x.shape[3]

        feats = self.backbone(x)
        x2 = feats["stage2"]
        x3 = feats["stage3"]
        x4 = feats["stage4"]

        bottleneck = self.dappm(self.reduce(x4))
        self._bottleneck = bottleneck

        x = self.up1(bottleneck, x3)
        x = self.up2(x, x2)
        # x = self.up3(x, feats["stage1"])

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return self.outc(self.dropout(x))