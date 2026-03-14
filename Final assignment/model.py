import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# https://github.com/facebookresearch/dinov3
# dino citation:
# @misc{simeoni2025dinov3,
#   title={{DINOv3}},
#   author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
#   year={2025},
#   eprint={2508.10104},
#   archivePrefix={arXiv},
#   primaryClass={cs.CV},
#   url={https://arxiv.org/abs/2508.10104},
# }


class Model(nn.Module):
    """
    A simple U-Net architecture for image segmentation.
    Based on the U-Net architecture from the original paper:
    Olaf Ronneberger et al. (2015), "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Adapt this model as needed for your problem-specific requirements. You can make multiple model classes and compare them,
    however, the CodaLab server requires the model class to be named "Model". Also, it will use the default values of the constructor
    to create the model, so make sure to set the default values of the constructor to the ones you want to use for your submission.
    """

    def __init__(self, in_channels=3, n_classes=19, dino_fine_tune=False):
        """
        Args:
            in_channels (int): Number of input channels. Default is 3 for RGB images.
            n_classes (int): Number of output classes. Default is 19 for the Cityscapes dataset.
            dino_fine_tune (bool): Whether to fine-tune the DINO model. Default is False.
        """

        super().__init__()
        self.in_channels = in_channels
        self.dino_fine_tune = dino_fine_tune
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.repo_dir = os.path.join(BASE_DIR, 'facebookresearch_dinov3_main')

        # import 🦖 v3
        # self.dino = torch.hub.load(
        #     "facebookresearch/dinov3", "dinov3_vitb16", pretrained=True
        # )

        self.dino = torch.hub.load(
            repo_or_dir=self.repo_dir,
            model='dinov3_vitb16',
            source='local',
            pretrained=False
        )

        weights_path = os.path.join(BASE_DIR, 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.dino.load_state_dict(state_dict)
        # freeze DINO for now, we only train the decode, maybe later we can compare what the influence would be if we fine tune the model
        for param in self.dino.parameters():
            param.requires_grad = self.dino_fine_tune

        # projection layers to match the CNN
        self.proj5 = nn.Conv2d(768, 512, kernel_size=1)

        # ASPP module for multi-scale context in the bottleneck
        self.aspp = ASPP(512, 512)

        # Encoding path
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # Decoding path
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        """
        # Check if the input tensor has the expected number of channels
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but got {x.shape[1]}"
            )
        
        #CNN Encoder Path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # x5 is now 16x16

        x5 = self.aspp(x5)

        # DINOv3 for feature extraction
        dino_features = self.dino.forward_features(x)["x_norm_patchtokens"]
        x_dino = dino_features.permute(0, 2, 1).reshape(
            x.shape[0], 768, x.shape[2] // 16, x.shape[3] // 16
        )  # x_dino is exactly 16x16!

        # Dino only in bottleneck
        x5 = x5 + self.proj5(x_dino)

        # Decoding path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits


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


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv, followed by Attention!"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        
        # Squeeze-and-Excitation Attention Block
        self.se = SEBlock(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        
        # Apply attention before passing to the next layer!
        return self.se(x)

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
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
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
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=8, dilation=8, bias=False)
        
        # Global Average Pooling branch
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        
        # Project all 5 branches down to out_channels
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res1 = self.conv1(x)
        res2 = self.conv2(x)
        res3 = self.conv3(x)
        res4 = self.conv4(x)
        res5 = F.interpolate(self.pool(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        
        res = torch.cat([res1, res2, res3, res4, res5], dim=1)
        return self.project(res)

if __name__ == "__main__":
    model = Model()
