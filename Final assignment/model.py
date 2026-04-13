import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision.models import mobilenet_v3_large


class Model(nn.Module):
    """
    Efficient segmentation model for the Cityscapes dataset.
    Uses MobileNetV3-Large as a pretrained backbone (pretrained on COCO segmentation)
    with a custom decoder head using ASPP for multi-scale context and
    Squeeze-and-Excitation attention blocks in the decoder.

    MobileNetV3 paper:
    Howard et al. (2019), "Searching for MobileNetV3"
    https://arxiv.org/abs/1905.02244

    ASPP from DeepLabV3:
    Chen et al. (2017), "Rethinking Atrous Convolution for Semantic Image Segmentation"
    https://arxiv.org/abs/1706.05587

    SE blocks inspo:
    Hu et al. (2018), "Squeeze-and-Excitation Networks"
    https://arxiv.org/abs/1709.01507
    """

    def __init__(self, in_channels=3, n_classes=19, dino_fine_tune=False):

        super().__init__()
        self.in_channels = in_channels
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        backbone = mobilenet_v3_large(weights=None)
        weights_path = os.path.join(BASE_DIR, "lraspp_mobilenetv3_pretrained.pth")
        if os.path.exists(weights_path):
            full_sd = torch.load(weights_path, map_location="cpu", weights_only=True)

            backbone_sd = {
                k[len("backbone."):]: v
                for k, v in full_sd.items()
                if k.startswith("backbone.")
            }
            backbone.features.load_state_dict(backbone_sd, strict=True)
            print("[Model] loaded COCO segmentation backbone from", weights_path)
        else:
            print("[Model] WARNING: no weights found at", weights_path, "- training from scratch")


        # enc1: 16ch  @ H/2
        # enc2: 24ch  @ H/4
        # enc3: 40ch  @ H/8
        # enc4: 112ch @ H/16
        # enc5: 960ch @ H/16  <- bottleneck
        f = backbone.features
        self.enc1 = nn.Sequential(*f[0:2])
        self.enc2 = nn.Sequential(*f[2:4])
        self.enc3 = nn.Sequential(*f[4:7])
        self.enc4 = nn.Sequential(*f[7:13])
        self.enc5 = nn.Sequential(*f[13:17])

        self.reduce = nn.Sequential(
            nn.Conv2d(960, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(64, 64)

        self.up0 = Up(64 + 112, 96) 
        self.up1 = Up(96 + 40,  64)
        self.up2 = Up(64 + 24,  48)
        self.up3 = Up(48 + 16,  32)

        # dropout for regularization
        self.dropout = nn.Dropout2d(p=0.1)

        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        """
        # check if the input tensor has the expected number of channels
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but got {x.shape[1]}"
            )

        H, W = x.shape[2], x.shape[3]

        # Encoder path
        x1 = self.enc1(x)   # 16ch,  H/2
        x2 = self.enc2(x1)  # 24ch,  H/4
        x3 = self.enc3(x2)  # 40ch,  H/8
        x4 = self.enc4(x3)  # 112ch, H/16
        x5 = self.enc5(x4)  # 960ch, H/16

        # Bottleneck: reduce channels then ASPP
        x5 = self.aspp(self.reduce(x5))  # 960 -> 128 -> ASPP -> 128

        # Decoding path
        x = self.up0(x5, x4) 
        x = self.up1(x, x3)  
        x = self.up2(x, x2)   
        x = self.up3(x, x1)  

        # upsample back to full input resolution
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        x = self.dropout(x)

        logits = self.outc(x)

        return logits


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.pw(self.dw(x))

class Up(nn.Module):
    """Upscaling then double conv, followed by Attention!"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),  # replaces DoubleConv
        )
        self.se = SEBlock(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # handle odd input sizes
        x1 = F.interpolate(x1, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        # apply attention before passing to the next layer!
        return self.se(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


##### inspo from https://arxiv.org/abs/1709.01507
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel-wise attention"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, max(1, in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, in_channels // reduction), in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excite
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale the input
        return x * y.expand_as(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1x1 conv, and 3x3 dilated convs
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, 3, padding=2, dilation=2, bias=False
        )
        self.conv3 = nn.Conv2d(
            in_channels, out_channels, 3, padding=4, dilation=4, bias=False
        )
        self.conv4 = nn.Conv2d(
            in_channels, out_channels, 3, padding=8, dilation=8, bias=False
        )

        # Global Average Pooling branch
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

        # project all 5 branches down to out_channels
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        res1 = self.conv1(x)
        res2 = self.conv2(x)
        res3 = self.conv3(x)
        res4 = self.conv4(x)
        res5 = F.interpolate(
            self.pool(x), size=x.shape[2:], mode="bilinear", align_corners=False
        )

        res = torch.cat([res1, res2, res3, res4, res5], dim=1)
        return self.project(res)


if __name__ == "__main__":
    model = Model()