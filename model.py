import math

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
    def __init__(self, in_channels=3, use_cbam=True, width_mult=1.0):
        super().__init__()
        base_channels = [64, 128, 256, 512, 1024]
        self.channels = [int(round(c * width_mult)) for c in base_channels]
        strides = [1, 2, 2, 2, 2]

        stages = []
        previous = in_channels
        for out_channels, stride in zip(self.channels, strides):
            stages.append(
                ResidualDSCBAMBlock(previous, out_channels, stride=stride, use_cbam=use_cbam)
            )
            previous = out_channels
        self.stages = nn.ModuleList(stages)

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


class MedNetMultiTask(nn.Module):
    def __init__(self, num_classes, num_segmentation_classes=1, use_cbam=True):
        super().__init__()
        self.backbone = ResidualCBAMBackbone(use_cbam=use_cbam)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.decoder4 = SegmentationDecoderBlock(1024, 512, 512)
        self.decoder3 = SegmentationDecoderBlock(512, 256, 256)
        self.decoder2 = SegmentationDecoderBlock(256, 128, 128)
        self.decoder1 = SegmentationDecoderBlock(128, 64, 64)
        self.segmentation_head = nn.Conv2d(
            64, num_segmentation_classes, kernel_size=1
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        encoded, activations = self.backbone(x)
        stage1, stage2, stage3, stage4, stage5 = activations

        classification = self.pool(encoded)
        classification = torch.flatten(classification, 1)
        classification = self.dropout(self.fc1(classification))
        classification_logits = self.fc2(classification)

        segmentation = self.decoder4(stage5, stage4)
        segmentation = self.decoder3(segmentation, stage3)
        segmentation = self.decoder2(segmentation, stage2)
        segmentation = self.decoder1(segmentation, stage1)
        segmentation_logits = self.segmentation_head(segmentation)
        segmentation_logits = F.interpolate(
            segmentation_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return classification_logits, segmentation_logits


def channel_shuffle(x, groups):
    batch_size, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch_size, -1, height, width)


def mk_init_weights(module):
    """MK-UNet 'normal' init scheme (normal std=0.02 for convs, BN to 1/0)."""
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class MKChannelAttention(nn.Module):
    """MK-UNet channel attention (shared MLP over avg and max pooling)."""

    def __init__(self, channels, ratio=16):
        super().__init__()
        if channels < ratio:
            ratio = channels
        reduced = channels // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.apply(mk_init_weights)

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class MKSpatialAttention(nn.Module):
    """MK-UNet spatial attention with a 7x7 convolution."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.apply(mk_init_weights)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(attention)


class GroupedAttentionGate(nn.Module):
    """MK-UNet grouped attention gate fusing a gating signal with a skip."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        intermediate = channels // 2
        groups = intermediate if kernel_size != 1 else 1
        padding = kernel_size // 2
        self.gate = nn.Sequential(
            nn.Conv2d(channels, intermediate, kernel_size, padding=padding, groups=groups, bias=True),
            nn.BatchNorm2d(intermediate),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(channels, intermediate, kernel_size, padding=padding, groups=groups, bias=True),
            nn.BatchNorm2d(intermediate),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(intermediate, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
        self.apply(mk_init_weights)

    def forward(self, gate, skip):
        psi = self.relu(self.gate(gate) + self.skip(skip))
        return skip * self.attention(psi)


class MultiKernelDepthwiseConv(nn.Module):
    """Parallel depthwise convolutions with multiple kernel sizes."""

    def __init__(self, channels, kernel_sizes=(1, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True),
            )
            for kernel_size in kernel_sizes
        )
        self.apply(mk_init_weights)

    def forward(self, x):
        return [branch(x) for branch in self.branches]


class MultiKernelInvertedResidual(nn.Module):
    """MK-UNet MKIR block: expand, multi-kernel depthwise, project."""

    def __init__(self, in_channels, out_channels, expansion_factor=2, kernel_sizes=(1, 3, 5)):
        super().__init__()
        expanded = in_channels * expansion_factor
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_channels, expanded, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.ReLU6(inplace=True),
        )
        self.multi_scale = MultiKernelDepthwiseConv(expanded, kernel_sizes)
        self.shuffle_groups = math.gcd(expanded, out_channels)
        self.pconv2 = nn.Sequential(
            nn.Conv2d(expanded, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.project_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else None
        )
        self.apply(mk_init_weights)

    def forward(self, x):
        out = self.pconv1(x)
        out = sum(self.multi_scale(out))
        out = channel_shuffle(out, self.shuffle_groups)
        out = self.pconv2(out)
        shortcut = x if self.project_shortcut is None else self.project_shortcut(x)
        return out + shortcut


class MKMNetDecoderBlock(nn.Module):
    """One MK-UNet style decoder stage on MedNet encoder features."""

    def __init__(self, in_channels, skip_channels):
        super().__init__()
        self.channel_attention = MKChannelAttention(in_channels)
        self.spatial_attention = MKSpatialAttention()
        self.mkir = MultiKernelInvertedResidual(in_channels, skip_channels)
        self.gate = GroupedAttentionGate(skip_channels)

    def forward(self, x, skip):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.mkir(x)
        x = F.relu(
            F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        )
        return x + self.gate(x, skip)


class MKMNet(nn.Module):
    """MedNet classification backbone fused with an MK-UNet decoder."""

    def __init__(
        self,
        num_classes=3,
        num_segmentation_classes=1,
        deep_supervision=True,
        width_mult=1.0,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.backbone = ResidualCBAMBackbone(use_cbam=True, width_mult=width_mult)
        c1, c2, c3, c4, c5 = self.backbone.channels

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(c5, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.decoder1 = MKMNetDecoderBlock(c5, c4)
        self.decoder2 = MKMNetDecoderBlock(c4, c3)
        self.decoder3 = MKMNetDecoderBlock(c3, c2)
        self.decoder4 = MKMNetDecoderBlock(c2, c1)
        self.segmentation_head = nn.Conv2d(c1, num_segmentation_classes, kernel_size=1)

        self.aux_head1 = nn.Conv2d(c4, num_segmentation_classes, kernel_size=1)
        self.aux_head2 = nn.Conv2d(c3, num_segmentation_classes, kernel_size=1)
        self.aux_head3 = nn.Conv2d(c2, num_segmentation_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        encoded, activations = self.backbone(x)
        stage1, stage2, stage3, stage4, _ = activations

        classification = self.pool(encoded)
        classification = torch.flatten(classification, 1)
        classification = self.dropout(self.fc1(classification))
        classification_logits = self.fc2(classification)

        d1 = self.decoder1(encoded, stage4)
        d2 = self.decoder2(d1, stage3)
        d3 = self.decoder3(d2, stage2)
        d4 = self.decoder4(d3, stage1)

        segmentation_logits = F.interpolate(
            self.segmentation_head(d4), size=input_size, mode="bilinear", align_corners=False
        )

        if not self.deep_supervision:
            return classification_logits, segmentation_logits

        auxiliary_logits = [
            F.interpolate(head(feature), size=input_size, mode="bilinear", align_corners=False)
            for head, feature in (
                (self.aux_head1, d1),
                (self.aux_head2, d2),
                (self.aux_head3, d3),
            )
        ]
        return classification_logits, segmentation_logits, auxiliary_logits


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


class InceptionDepthwiseSeparableConv(nn.Module):
    """Parallel 1x1, 3x3 and 5x5 depthwise separable convolutions."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = [
            out_channels // 3,
            out_channels // 3,
            out_channels - 2 * (out_channels // 3),
        ]

        self.branch1 = self._make_branch(
            in_channels, branch_channels[0], kernel_size=1
        )
        self.branch3 = self._make_branch(
            in_channels, branch_channels[1], kernel_size=3
        )
        self.branch5 = self._make_branch(
            in_channels, branch_channels[2], kernel_size=5
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _make_branch(in_channels, out_channels, kernel_size):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        features = torch.cat(
            [self.branch1(x), self.branch3(x), self.branch5(x)], dim=1
        )
        return self.fusion(features)


class ResidualInceptionDSCBlock(nn.Module):
    """Two inception DSC layers with a projected residual connection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inception1 = InceptionDepthwiseSeparableConv(
            in_channels, out_channels
        )
        self.inception2 = InceptionDepthwiseSeparableConv(
            out_channels, out_channels
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.inception1(x)
        x = self.inception2(x)
        return self.relu(x + residual)


class RCBAMDecoderBlock(nn.Module):
    """Upsample, fuse an encoder output, then refine it."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.refinement = ResidualInceptionDSCBlock(
            out_channels + skip_channels, out_channels
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.channel_reduction(x)
        x = torch.cat([x, skip], dim=1)
        return self.refinement(x)


class RCBAMMNet(nn.Module):
    """MedNet encoder with a residual inception decoder."""

    def __init__(self, num_classes=3, num_segmentation_classes=1):
        super().__init__()
        self.backbone = ResidualCBAMBackbone(use_cbam=True)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.skip_cbam1 = CBAM(64)
        self.skip_cbam2 = CBAM(128)
        self.skip_cbam3 = CBAM(256)
        self.skip_cbam4 = CBAM(512)

        self.decoder4 = RCBAMDecoderBlock(1024, 512, 512)
        self.decoder3 = RCBAMDecoderBlock(512, 256, 256)
        self.decoder2 = RCBAMDecoderBlock(256, 128, 128)
        self.decoder1 = RCBAMDecoderBlock(128, 64, 64)
        self.segmentation_head = nn.Conv2d(
            64, num_segmentation_classes, kernel_size=1
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        encoded, activations = self.backbone(x)
        stage1, stage2, stage3, stage4, _ = activations

        classification = self.pool(encoded)
        classification = torch.flatten(classification, 1)
        classification = self.dropout(self.fc1(classification))
        classification_logits = self.fc2(classification)

        skip1 = self.skip_cbam1(stage1)
        skip2 = self.skip_cbam2(stage2)
        skip3 = self.skip_cbam3(stage3)
        skip4 = self.skip_cbam4(stage4)

        segmentation = self.decoder4(encoded, skip4)
        segmentation = self.decoder3(segmentation, skip3)
        segmentation = self.decoder2(segmentation, skip2)
        segmentation = self.decoder1(segmentation, skip1)
        segmentation_logits = self.segmentation_head(segmentation)
        segmentation_logits = F.interpolate(
            segmentation_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return classification_logits, segmentation_logits
