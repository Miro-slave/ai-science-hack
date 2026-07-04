
from SegmentationConfig import SegmentationConfig, InferenceConfig

import logging


logger = logging.getLogger(__name__)

# inference.py

import numpy as np
import torch
import torch.nn as nn
import cv2
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # явно float32
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # явно float32


def preprocess_chunk(chunk: np.ndarray) -> torch.Tensor:
    """
    Нормализация чанка → тензор float32.
    Вынесено в отдельную функцию, используется обоими инференсерами.

    Ключевые детали:
    - chunk приводим к float32 ДО деления
    - IMAGENET_MEAN/STD объявлены как float32 глобально
    - np.ascontiguousarray гарантирует непрерывную память (нужно для torch)
    - явный dtype=np.float32 на выходе страхует от апкаста
    """
    x = chunk.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    x = np.ascontiguousarray(x, dtype=np.float32)
    return torch.from_numpy(x)


class SegmentationInferencer:
    def __init__(self, model: nn.Module, config: InferenceConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()

        self.chunk_size = config.chunk_size
        self.overlap = config.overlap
        self.stride = self.chunk_size - self.overlap
        self.threshold = config.segmentation_threshold
        self.weight_map = self._create_gaussian_weight(self.chunk_size)

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H, W = image.shape[:2]
        accumulator = np.zeros((H, W), dtype=np.float32)
        weight_sum = np.zeros((H, W), dtype=np.float32)

        positions = self._get_positions(H, W)
        batch_size = self.config.seg_config.batch_size

        for i in tqdm(range(0, len(positions), batch_size),
                      desc="Segmentation", leave=False):
            batch_pos = positions[i:i + batch_size]

            chunks = []
            for y, x in batch_pos:
                y_end = min(y + self.chunk_size, H)
                x_end = min(x + self.chunk_size, W)
                chunk = image[y:y_end, x:x_end]
                # Паддинг если чанк у границы
                if chunk.shape[0] < self.chunk_size or chunk.shape[1] < self.chunk_size:
                    padded = np.zeros(
                        (self.chunk_size, self.chunk_size, 3), dtype=np.uint8
                    )
                    padded[:chunk.shape[0], :chunk.shape[1]] = chunk
                    chunk = padded
                chunks.append(preprocess_chunk(chunk))

            batch = torch.stack(chunks).to(self.device)  # (B, 3, H, W) float32

            with torch.no_grad():
                preds = self.model(batch)
                probs = torch.sigmoid(preds).cpu().numpy().squeeze(1)

            for idx, (y, x) in enumerate(batch_pos):
                h_end = min(y + self.chunk_size, H)
                w_end = min(x + self.chunk_size, W)
                h_s = h_end - y
                w_s = w_end - x

                accumulator[y:h_end, x:w_end] += (
                        probs[idx, :h_s, :w_s] * self.weight_map[:h_s, :w_s]
                )
                weight_sum[y:h_end, x:w_end] += self.weight_map[:h_s, :w_s]

        probs = accumulator / np.maximum(weight_sum, 1e-8)
        heatmap = self._prob_to_heatmap(probs)
        return probs.astype(np.float32), heatmap

    def predict_binary(self, image: np.ndarray) -> np.ndarray:
        probs, _ = self.predict(image)
        return ((probs > self.threshold) * 255).astype(np.uint8)

    def _get_positions(self, H: int, W: int) -> list[tuple[int, int]]:
        positions = []
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                positions.append((y, x))
        return positions

    @staticmethod
    def _prob_to_heatmap(probs: np.ndarray) -> np.ndarray:
        probs_uint8 = (probs * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(probs_uint8, cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _create_gaussian_weight(size: int) -> np.ndarray:
        sigma = size / 4.0
        ax = np.arange(size, dtype=np.float32) - size / 2.0
        xx, yy = np.meshgrid(ax, ax)
        w = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        return (w / w.max()).astype(np.float32)


class ClassificationInferencer:
    def __init__(self, model: nn.Module, config: InferenceConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()

        self.chunk_size = config.chunk_size
        self.overlap = config.overlap
        self.stride = self.chunk_size - self.overlap

    def predict(self, image: np.ndarray) -> dict:
        H, W = image.shape[:2]
        positions = self._get_positions(H, W)
        batch_size = self.config.cls_config.batch_size

        votes = {i: 0.0 for i in range(self.config.cls_config.num_classes)}
        chunk_preds = []

        for i in tqdm(range(0, len(positions), batch_size),
                      desc="Classification", leave=False):
            batch_pos = positions[i:i + batch_size]

            chunks = []
            for y, x in batch_pos:
                y_end = min(y + self.chunk_size, H)
                x_end = min(x + self.chunk_size, W)
                chunk = image[y:y_end, x:x_end]
                if chunk.shape[0] < self.chunk_size or chunk.shape[1] < self.chunk_size:
                    padded = np.zeros(
                        (self.chunk_size, self.chunk_size, 3), dtype=np.uint8
                    )
                    padded[:chunk.shape[0], :chunk.shape[1]] = chunk
                    chunk = padded
                chunks.append(preprocess_chunk(chunk))

            batch = torch.stack(chunks).to(self.device)  # float32

            with torch.no_grad():
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # float32

            for idx, (y, x) in enumerate(batch_pos):
                pred_class = int(probs[idx].argmax())
                confidence = float(probs[idx].max())
                votes[pred_class] += confidence
                chunk_preds.append({
                    "y": y, "x": x,
                    "class_id": pred_class,
                    "confidence": confidence,
                })

        total = sum(votes.values()) + 1e-8
        image_class = max(votes, key=votes.get)
        image_conf = votes[image_class] / total
        image_class_name = self.config.cls_config.class_name(image_class)

        return {
            "image_class": image_class,
            "image_class_name": image_class_name,
            "image_confidence": image_conf,
            "chunk_predictions": chunk_preds,
            "vote_breakdown": {
                self.config.cls_config.class_name(k): v / total
                for k, v in votes.items()
            },
        }

    def _get_positions(self, H: int, W: int) -> list[tuple[int, int]]:
        positions = []
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                positions.append((y, x))
        return positions