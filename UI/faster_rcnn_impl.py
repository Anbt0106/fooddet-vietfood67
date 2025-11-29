import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ============ LIGHTWEIGHT RPL MODULES ============
class LightweightChannelAttention(nn.Module):
    """Lightweight Channel Attention"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class LightweightResidualBlock(nn.Module):
    """Lightweight Residual Block"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.channel_att = LightweightChannelAttention(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.channel_att(out)
        return self.relu(out + identity)


class ResidualPatternLearning(nn.Module):
    """Lightweight RPL Module"""

    def __init__(self, in_channels=256, num_blocks=1):
        super().__init__()
        self.domain_adapt = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.ModuleList([
            LightweightResidualBlock(in_channels) for _ in range(num_blocks)
        ])
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        identity = x
        out = self.domain_adapt(x)
        for block in self.residual_blocks:
            out = block(out)
        return identity + self.gate * out


class FasterRCNN_RPL(nn.Module):
    """Faster R-CNN với Lightweight RPL"""

    def __init__(self, num_classes, use_rpl=True, num_rpl_blocks=1, img_size=640):
        super().__init__()

        # Load base model
        self.base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )

        # Config - Set proper image size range
        self.base_model.transform.min_size = (img_size,)
        self.base_model.transform.max_size = img_size

        # Replace ROI head
        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = \
            FastRCNNPredictor(in_features, num_classes)

        self.use_rpl = use_rpl

        if use_rpl:
            self.rpl_modules = nn.ModuleDict({
                f'rpl_layer{i}': ResidualPatternLearning(256, num_rpl_blocks)
                for i in range(3)
            })
            self._inject_rpl_into_fpn()

    def _inject_rpl_into_fpn(self):
        """Inject RPL vào FPN"""
        fpn = self.base_model.backbone.fpn
        original_forward = fpn.forward

        def new_forward(x):
            fpn_outputs = original_forward(x)
            enhanced_outputs = {}
            for idx, (name, feat) in enumerate(fpn_outputs.items()):
                if idx < 3 and f'rpl_layer{idx}' in self.rpl_modules:
                    enhanced_outputs[name] = self.rpl_modules[f'rpl_layer{idx}'](feat)
                else:
                    enhanced_outputs[name] = feat
            return enhanced_outputs

        fpn.forward = new_forward

    def forward(self, images, targets=None):
        return self.base_model(images, targets)

# ============ SMART RESIZE FUNCTION ============
def smart_resize_with_padding(image, target_size=640):
    """
    Resize ảnh về target_size với padding để giữ aspect ratio
    """
    h, w = image.shape[:2]

    # Tính scale factor
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[114, 114, 114]
    )

    return padded, scale, (left, top)

def transform_boxes_inverse(boxes, scale, offset):
    """
    Transform bounding boxes from resized image back to original image
    """
    if len(boxes) == 0:
        return boxes

    left_pad, top_pad = offset
    transformed_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = (x1 - left_pad) / scale
        x2 = (x2 - left_pad) / scale
        y1 = (y1 - top_pad) / scale
        y2 = (y2 - top_pad) / scale
        transformed_boxes.append([x1, y1, x2, y2])

    return transformed_boxes
