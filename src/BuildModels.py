# models/__init__.py
import torch.nn as nn

from SegmentationConfig import SegmentationConfig, ClassificationConfig


def build_segmentor(config: SegmentationConfig) -> nn.Module:
    """Фабрика сегментатора. Добавить новую модель = 3 строки."""
    if config.segmentor_type == "unet":
        from segmentor import build_unet
        return build_unet(config)
    elif config.segmentor_type == "unet++":
        from segmentor import build_unet_plusplus
        return build_unet_plusplus(config)
    elif config.segmentor_type == "deeplabv3+":
        from segmentor import build_deeplabv3_plus
        return build_deeplabv3_plus(config)
    else:
        raise ValueError(f"Unknown segmentor_type: {config.segmentor_type}")


def build_classifier(config: ClassificationConfig) -> nn.Module:
    """Фабрика классификатора. Добавить новую модель = 3 строки."""
    if config.classifier_type == "resnet":
        from classifier import build_resnet
        return build_resnet(config)
    elif config.classifier_type == "efficientnet":
        from classifier import build_efficientnet
        return build_efficientnet(config)
    elif config.classifier_type == "vit":
        from classifier import build_vit
        return build_vit(config)
    else:
        raise ValueError(f"Unknown classifier_type: {config.classifier_type}")