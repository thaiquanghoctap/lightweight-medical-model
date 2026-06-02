import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss *= self.alpha[targets]

        return focal_loss.mean()


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        avg_pool = torch.mean(x, dim=(2, 3)).view(batch_size, channels)
        max_pool = torch.amax(x, dim=(2, 3)).view(batch_size, channels)
        attention = self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        attention = self.sigmoid(attention).view(batch_size, channels, 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg_pool, max_pool], dim=1))
        return x * self.sigmoid(attention)


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.batch_norm(x)
        return self.relu(x)


class ResidualDSCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=True):
        super().__init__()
        self.convs = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, stride),
            DepthwiseSeparableConv(out_channels, out_channels),
        )
        self.cbam = CBAM(out_channels) if use_cbam else nn.Identity()
        self.shortcut = nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.cbam(self.convs(x)) + self.shortcut(x)


class ResidualCBAMBackbone(nn.Module):
    def __init__(self, in_channels=3, use_cbam=True):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                ResidualDSCBAMBlock(in_channels, 64, stride=1, use_cbam=use_cbam),
                ResidualDSCBAMBlock(64, 128, stride=2, use_cbam=use_cbam),
                ResidualDSCBAMBlock(128, 256, stride=2, use_cbam=use_cbam),
                ResidualDSCBAMBlock(256, 512, stride=2, use_cbam=use_cbam),
                ResidualDSCBAMBlock(512, 1024, stride=2, use_cbam=use_cbam),
            ]
        )

    def forward(self, x):
        activations = []
        for stage in self.stages:
            x = stage(x)
            activations.append(x)
        return x, activations


class SegmentationDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.convs(torch.cat([x, skip], dim=1))


class MedNetSegmentation(nn.Module):
    def __init__(self, num_classes=1, use_cbam=True):
        super().__init__()
        self.backbone = ResidualCBAMBackbone(use_cbam=use_cbam)
        self.decoder4 = SegmentationDecoderBlock(1024, 512, 512)
        self.decoder3 = SegmentationDecoderBlock(512, 256, 256)
        self.decoder2 = SegmentationDecoderBlock(256, 128, 128)
        self.decoder1 = SegmentationDecoderBlock(128, 64, 64)
        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        _, activations = self.backbone(x)
        stage1, stage2, stage3, stage4, stage5 = activations

        x = self.decoder4(stage5, stage4)
        x = self.decoder3(x, stage3)
        x = self.decoder2(x, stage2)
        x = self.decoder1(x, stage1)
        logits = self.segmentation_head(x)

        return F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )


class MedNet(nn.Module):
    def __init__(self, num_classes, use_cbam=True):
        super().__init__()
        self.backbone = ResidualCBAMBackbone(use_cbam=use_cbam)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x, activations = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(self.fc1(x))
        return self.fc2(x), activations
