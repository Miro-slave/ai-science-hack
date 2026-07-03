# config.py
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────
# Сегментация
# ─────────────────────────────────────────

@dataclass
class SegmentationConfig:
    # Пути
    image_dir: str = "data/talc/images"      # тип 1: тальковые руды
    mask_dir: str = "data/talc/masks"
    chunk_dir: str = "data/chunks_talc"

    checkpoint_dir: str = "checkpoints/talc"
    log_dir: str = "logs/talc"

    # Нарезка
    chunk_size: int = 1024
    overlap: int = 128
    min_region_size: int = 500
    skip_empty_ratio: float = 0.9

    # Маска
    hsv_lower_blue: tuple = (100, 100, 50)
    hsv_upper_blue: tuple = (130, 255, 255)
    dilation_iters: int = 0
    padding: int = 20

    # Модель
    segmentor_type: str = "unet"              # "unet" | "deeplabv3+" | "unet++"
    segmentor_encoder: str = "resnet34"       # можно: "efficientnet-b4", "resnet50", ...

    # Обучение
    batch_size: int = 4
    num_epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    test_size: float = 0.2
    num_workers: int = 4
    device: str = "cuda"

    # Loss
    dice_weight: float = 1.0
    bce_weight: float = 1.0

    # Логирование
    log_interval: int = 10
    save_interval: int = 5

    # Инференс
    inference_blend_mode: str = "gaussian"
    segmentation_threshold: float = 0.5

    def __post_init__(self):
        Path(self.mask_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chunk_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────
# Классификация
# ─────────────────────────────────────────

@dataclass
class ClassificationConfig:
    # Пути: папка → класс
    # Структура: data/classification/{ore_class}/images...
    classification_data_dir: str = "data/classification"
    ore_classes: tuple = ("ordinary", "difficult")  # порядок = class_id
    chunk_dir: str = "data/chunks_classification"

    checkpoint_dir: str = "checkpoints/classification"
    log_dir: str = "logs/classification"

    # Нарезка — те же параметры что и для сегментации
    chunk_size: int = 512
    overlap: int = 64

    # Модель
    classifier_type: str = "resnet"            # "resnet" | "efficientnet" | "vit"
    classifier_variant: str = "resnet50"       # resnet50, efficientnet_b0, vit_b_16, ...

    # Обучение
    batch_size: int = 16
    num_epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    test_size: float = 0.2
    num_workers: int = 4
    device: str = "cuda"

    # Loss
    use_class_weights: bool = True

    # Логирование
    log_interval: int = 10
    save_interval: int = 5

    @property
    def num_classes(self) -> int:
        return len(self.ore_classes)

    def class_name(self, class_id: int) -> str:
        return self.ore_classes[class_id]

    def class_id(self, class_name: str) -> int:
        return self.ore_classes.index(class_name)

    def __post_init__(self):
        Path(self.chunk_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────
# Инференс
# ─────────────────────────────────────────

@dataclass
class InferenceConfig:
    seg_checkpoint: str = "checkpoints/talc/best_model.pt"
    cls_checkpoint: str = "checkpoints/classification/best_model.pt"

    seg_config: SegmentationConfig = field(default_factory=SegmentationConfig)
    cls_config: ClassificationConfig = field(default_factory=ClassificationConfig)

    chunk_size: int = 512
    overlap: int = 64
    device: str = "cuda"
    segmentation_threshold: float = 0.5

    # Majority vote
    majority_vote_threshold: float = 0.0  # доля голосов для принятия решения (0 = просто majority)

    def __post_init__(self):
        self.seg_config.chunk_size = self.chunk_size
        self.seg_config.overlap = self.overlap
        self.seg_config.device = self.device
        self.seg_config.inference_blend_mode = "gaussian"
        self.seg_config.num_classes = 1

        self.cls_config.chunk_size = self.chunk_size
        self.cls_config.overlap = self.overlap
        self.cls_config.device = self.device