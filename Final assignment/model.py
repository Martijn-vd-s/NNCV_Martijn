import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv, followed by Attention!"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.se = SEBlock(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = F.interpolate(x1, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        return self.se(self.conv(x))


class UpNoSkip(nn.Module):
    """Upscaling then double conv with no skip connection — for the final upsample stages"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        return self.se(self.conv(self.up(x)))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


##### inspo from https://arxiv.org/pdf/2504.05184
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
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=2,  dilation=2,  bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=4,  dilation=4,  bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=8,  dilation=8,  bias=False)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )
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
        return self.project(torch.cat([res1, res2, res3, res4, res5], dim=1))


class Model(nn.Module):
    """
    U-Net with SegFormer-B5 encoder for semantic segmentation on Cityscapes.

    SegFormer-B5 produces 4-scale feature maps natively 
    Decoder mirrors the original DINOv3 U-Net
    depth with 4 upsampling stages + SE attention.

    Key refs:
      - SegFormer: Xie et al. (2021) https://arxiv.org/abs/2105.15203
      - U-Net: Ronneberger et al. (2015) https://arxiv.org/pdf/1505.04597.pdf
      - SE-Net: Hu et al. (2018) https://arxiv.org/abs/1709.01507
      - ASPP: Chen et al. (2017) https://arxiv.org/abs/1706.05587
    """

    def __init__(self, in_channels=3, n_classes=19, dino_fine_tune=False):
        super().__init__()
        self.in_channels = in_channels

        # SegFormer-B5 encoder — pretrained on ImageNet
        # outputs 4 feature maps: H/4, H/8, H/16, H/32
        # channels:               [64,  128,  320,  512]
        self.encoder = SegformerModel.from_pretrained("./mit-b5", output_hidden_states=True)

        # freeze all, then unfreeze last two stages for fine-tuning
        for param in self.encoder.parameters():
            param.requires_grad = False
        for name, param in self.encoder.named_parameters():
            if "encoder.block.2" in name or "encoder.block.3" in name:
                param.requires_grad = True

        # bottleneck
        self.aspp = ASPP(512, 512)
        self.dropout = nn.Dropout2d(p=0.2)

        # Decoding path — 4 stages matching your original depth
        # SegFormer-B5 stage channels: s1=64, s2=128, s3=320, s4=512
        self.up1 = Up(512 + 320, 256)    # s4 + s3  -> 256ch  H/16
        self.up2 = Up(256 + 128, 128)    # up1 + s2 -> 128ch  H/8
        self.up3 = Up(128 + 64,   64)    # up2 + s1 ->  64ch  H/4
        self.up4 = UpNoSkip(64,   64)    # no skip  ->  64ch  H/2
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but got {x.shape[1]}"
            )
        H, W = x.shape[2], x.shape[3]

        # encode
        out = self.encoder(pixel_values=x, output_hidden_states=True)
        s1, s2, s3, s4 = out.hidden_states   # H/4, H/8, H/16, H/32

        # bottleneck
        s4 = self.aspp(s4)
        s4 = self.dropout(s4)

        # decode
        x = self.up1(s4, s3)   # 256ch  H/16
        x = self.up2(x,  s2)   # 128ch  H/8
        x = self.up3(x,  s1)   #  64ch  H/4
        x = self.up4(x)        #  64ch  H/2

        # final upsample to full resolution
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return self.outc(x)


if __name__ == "__main__":
    model = Model()
    dummy = torch.randn(2, 3, 512, 1024)
    out = model(dummy)
    print(out.shape)   # should be (2, 19, 512, 1024)