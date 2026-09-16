import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions.

    Input:
        [B, in_channels, H, W]

    Output:
        [B, out_channels, H, W]
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """
    Downsampling block.

    MaxPool2d reduces spatial dimensions by 2.
    DoubleConv extracts features.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsampling block.

    1. Upsample using transposed convolution.
    2. Concatenate the corresponding encoder feature map.
    3. Apply DoubleConv.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            in_channels=(in_channels // 2) + skip_channels,
            out_channels=out_channels,
        )

    def forward(self, x, skip):
        x = self.up(x)

        # For the current 256x256 input size, spatial dimensions
        # should already match exactly.
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """
    Compact U-Net for binary nuclei segmentation.

    Input:
        [B, 3, 256, 256]

    Output:
        [B, 1, 256, 256]

    The output contains raw logits.
    Apply sigmoid only for prediction/visualization.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_channels=16,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # Encoder
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.down4 = DownBlock(c4, c5)

        # Decoder
        self.up1 = UpBlock(c5, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up3 = UpBlock(c3, c2, c2)
        self.up4 = UpBlock(c2, c1, c1)

        # One output channel for binary segmentation
        self.outc = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Raw logits
        logits = self.outc(x)

        return logits