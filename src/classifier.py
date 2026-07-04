# models/classifier.py
import torch
import torch.nn as nn
import timm
import math

from SegmentationConfig import ClassificationConfig


class ImageClassifier(nn.Module):
    """Обертка для стандартизации выхода классификатора."""

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            (B, num_classes) logits
        """
        return self.backbone(x)


def build_resnet(config: ClassificationConfig) -> ImageClassifier:
    """ResNet. variant: resnet18, resnet34, resnet50, resnet101..."""
    backbone = timm.create_model(
        config.classifier_variant,
        pretrained=True,
        num_classes=config.num_classes,
    )
    return ImageClassifier(backbone, config.num_classes)


def build_efficientnet(config: ClassificationConfig) -> ImageClassifier:
    """EfficientNet. variant: efficientnet_b0, efficientnet_b1..."""
    backbone = timm.create_model(
        config.classifier_variant,
        pretrained=True,
        num_classes=config.num_classes,
    )
    return ImageClassifier(backbone, config.num_classes)


def build_vit(config: ClassificationConfig) -> ImageClassifier:
    """Vision Transformer. variant: vit_b_16, vit_b_32, vit_l_16..."""
    backbone = timm.create_model(
        config.classifier_variant,
        pretrained=True,
        num_classes=config.num_classes,
    )
    return ImageClassifier(backbone, config.num_classes)


# ─── Новые модели добавляются здесь:
# def build_mymodel(config) → ImageClassifier:
#     backbone = timm.create_model(...)
#     return ImageClassifier(backbone, config.num_classes)