# dataset.py
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from SegmentationConfig import SegmentationConfig, ClassificationConfig
from cv2_utils import imread_rgb, imread_gray


# ─────────────────────────────────────────
# Аугментации
# ─────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def get_segmentation_transforms(train: bool = True) -> A.Compose:
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_classification_transforms(train: bool = True) -> A.Compose:
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.RandomResizedCrop(
                size=(512, 512),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────
# Dataset: Сегментация
# ─────────────────────────────────────────

class SegmentationDataset(Dataset):
    def __init__(self, chunk_meta: list[dict], transform: A.Compose | None = None):
        self.chunks = chunk_meta
        self.transform = transform

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        item = self.chunks[idx]
        image = imread_rgb(item["image"])
        mask = imread_gray(item["mask"])
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()


def build_segmentation_dataloaders(
    chunk_meta: list[dict],
    config: SegmentationConfig,
) -> tuple[DataLoader, DataLoader]:
    """Split по source-изображениям, чтобы не было утечки."""
    sources = sorted(set(m["source"] for m in chunk_meta))
    train_sources, test_sources = train_test_split(
        sources, test_size=config.test_size, random_state=42,
    )
    train_set = set(train_sources)
    test_set = set(test_sources)

    train_meta = [m for m in chunk_meta if m["source"] in train_set]
    test_meta = [m for m in chunk_meta if m["source"] in test_set]

    print(f"Segmentation — Train: {len(train_meta)}, Test: {len(test_meta)}")

    train_ds = SegmentationDataset(train_meta, get_segmentation_transforms(train=True))
    test_ds = SegmentationDataset(test_meta, get_segmentation_transforms(train=False))

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                          num_workers=config.num_workers, pin_memory=True,
                          persistent_workers=config.num_workers > 0)
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                         num_workers=config.num_workers, pin_memory=True,
                         persistent_workers=config.num_workers > 0)
    return train_dl, test_dl


# ─────────────────────────────────────────
# Dataset: Классификация
# ─────────────────────────────────────────

class ClassificationDataset(Dataset):
    def __init__(self, chunk_meta: list[dict], transform: A.Compose | None = None):
        self.chunks = chunk_meta
        self.transform = transform

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        item = self.chunks[idx]
        image = imread_rgb(item["image"])
        class_id = item["class_id"]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, torch.tensor(class_id, dtype=torch.long)


def build_classification_dataloaders(
    chunk_meta: list[dict],
    config: ClassificationConfig,
) -> tuple[DataLoader, DataLoader]:
    """Split по source-изображениям."""
    sources = sorted(set(m["source"] for m in chunk_meta))
    train_sources, test_sources = train_test_split(
        sources, test_size=config.test_size, random_state=42,
    )
    train_set = set(train_sources)
    test_set = set(test_sources)

    train_meta = [m for m in chunk_meta if m["source"] in train_set]
    test_meta = [m for m in chunk_meta if m["source"] in test_set]

    print(f"Classification — Train: {len(train_meta)}, Test: {len(test_meta)}")

    train_ds = ClassificationDataset(train_meta, get_classification_transforms(train=True))
    test_ds = ClassificationDataset(test_meta, get_classification_transforms(train=False))

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                          num_workers=config.num_workers, pin_memory=True,
                          persistent_workers=config.num_workers > 0)
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                          num_workers=config.num_workers, pin_memory=True,
                         persistent_workers=config.num_workers > 0)
    return train_dl, test_dl