# models/segmentor.py
import torch.nn as nn
import segmentation_models_pytorch as smp

from SegmentationConfig import SegmentationConfig


def _get_encoder_params(encoder_name: str) -> dict:
    """
    Маппинг encoder_name → параметры предобученной модели в SMP.
    Легко добавить новый encoder.
    """
    return {
        "imagenet": "imagenet",
        "imagenet5k": "imagenet5k",
        "swsl": "swsl",           # semi-supervised
    }.get(encoder_name, "imagenet")


def build_unet(config: SegmentationConfig) -> nn.Module:
    """U-Net с предобученным энкодером."""
    return smp.Unet(
        encoder_name=config.segmentor_encoder,
        encoder_weights=_get_encoder_params(config.segmentor_encoder),
        in_channels=3,
        classes=1,
        activation=None,  # логиты → loss сам применит sigmoid
    )


def build_unet_plusplus(config: SegmentationConfig) -> nn.Module:
    """U-Net++ с предобученным энкодером."""
    return smp.UnetPlusPlus(
        encoder_name=config.segmentor_encoder,
        encoder_weights=_get_encoder_params(config.segmentor_encoder),
        in_channels=3,
        classes=1,
        activation=None,
    )


def build_deeplabv3_plus(config: SegmentationConfig) -> nn.Module:
    """DeepLabV3+ с предобученным энкодером."""
    return smp.DeepLabV3Plus(
        encoder_name=config.segmentor_encoder,
        encoder_weights=_get_encoder_params(config.segmentor_encoder),
        in_channels=3,
        classes=1,
        activation=None,
    )


# ─── Новые модели добавляются здесь:
# def build_mymodel(config) → nn.Module:
#     ...