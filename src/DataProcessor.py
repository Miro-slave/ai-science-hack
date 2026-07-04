# data_processor.py
import numpy as np
import cv2
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Literal


from SegmentationConfig import ClassificationConfig, SegmentationConfig
from cv2_utils import imread_rgb, imread_gray, imwrite

logger = logging.getLogger(__name__)


class SegmentationDataProcessor:
    """Генерация масок и нарезка чанков для задачи сегментации (тальк/не-тальк)."""

    def __init__(self, config: SegmentationConfig):
        self.config = config

    def generate_all_masks(self) -> None:
        image_dir = Path(self.config.image_dir)
        mask_dir = Path(self.config.mask_dir)
        image_paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))

        logger.info(f"Найдено изображений: {len(image_paths)}")

        for img_path in tqdm(image_paths, desc="Генерация масок"):
            image = imread_rgb(img_path)
            mask = self.create_mask(image)
            save_path = mask_dir / f"{img_path.stem}_mask.png"
            imwrite(save_path, mask)

    def create_mask(self, image: np.ndarray) -> np.ndarray:
        blue_mask = self._extract_blue_mask(image)
        talc_mask = self._close_and_fill(blue_mask)
        return talc_mask

    def _extract_blue_mask(self, image: np.ndarray) -> np.ndarray:
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower = np.array(self.config.hsv_lower_blue)
        upper = np.array(self.config.hsv_upper_blue)
        return cv2.inRange(hsv, lower, upper)

    def _close_and_fill(self, blue_mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(blue_mask, kernel, iterations=self.config.dilation_iters)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)

        orig_h, orig_w = closed.shape
        p = self.config.padding
        closed = cv2.copyMakeBorder(closed, p, p, p, p, borderType=cv2.BORDER_CONSTANT, value=0)

        flood_mask = np.zeros((closed.shape[0] + 2, closed.shape[1] + 2), np.uint8)
        fill_image = closed.copy()
        cv2.floodFill(fill_image, flood_mask, (0, 0), 128)

        talc_mask = (fill_image == 0).astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(talc_mask, connectivity=8)
        clean_mask = np.zeros_like(talc_mask)
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] >= self.config.min_region_size:
                clean_mask[labels == label_id] = 255

        return clean_mask[p:p + orig_h, p:p + orig_w]

    def slice_into_chunks(self) -> list[dict]:
        image_dir = Path(self.config.image_dir)
        mask_dir = Path(self.config.mask_dir)
        chunk_dir = Path(self.config.chunk_dir)

        (chunk_dir / "images").mkdir(parents=True, exist_ok=True)
        (chunk_dir / "masks").mkdir(parents=True, exist_ok=True)

        chunk_size = self.config.chunk_size
        stride = chunk_size - self.config.overlap

        image_paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
        chunk_meta = []

        for img_path in tqdm(image_paths, desc="Нарезка (сегментация)"):
            mask_path = mask_dir / f"{img_path.stem}_mask.png"
            if not mask_path.exists():
                logger.warning(f"Маска не найдена: {mask_path}, пропускаем")
                continue

            image = imread_rgb(img_path)
            mask = imread_gray(mask_path)
            H, W = image.shape[:2]

            for y in range(0, H - chunk_size + 1, stride):
                for x in range(0, W - chunk_size + 1, stride):
                    img_chunk = image[y:y + chunk_size, x:x + chunk_size]
                    mask_chunk = mask[y:y + chunk_size, x:x + chunk_size]

                    if mask_chunk.max() == 0 and np.random.random() < self.config.skip_empty_ratio:
                        continue

                    chunk_name = f"{img_path.stem}_y{y}_x{x}"
                    imwrite(chunk_dir / "images" / f"{chunk_name}.png",
                            cv2.cvtColor(img_chunk, cv2.COLOR_RGB2BGR))
                    imwrite(chunk_dir / "masks" / f"{chunk_name}.png", mask_chunk)

                    chunk_meta.append({
                        "image": str(chunk_dir / "images" / f"{chunk_name}.png"),
                        "mask": str(chunk_dir / "masks" / f"{chunk_name}.png"),
                        "source": img_path.stem,
                    })

        logger.info(f"Всего чанков: {len(chunk_meta)}")
        return chunk_meta


class ClassificationDataProcessor:
    """Нарезка чанков для задачи классификации (рядовая/труднообогатимая)."""

    def __init__(self, config: ClassificationConfig):
        self.config = config

    def slice_into_chunks(self) -> list[dict]:
        """
        Структура папок:
        data/classification/
            ordinary/images/...
            difficult/images/...

        Создаём чанки с метаданными: source image + class_id.
        """
        data_dir = Path(self.config.classification_data_dir)
        chunk_dir = Path(self.config.chunk_dir)
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_size = self.config.chunk_size
        stride = chunk_size - self.config.overlap

        chunk_meta = []

        for class_name in self.config.ore_classes:
            class_dir = data_dir / class_name / "images"
            class_id = self.config.class_id(class_name)

            if not class_dir.exists():
                logger.warning(f"Папка не найдена: {class_dir}, пропускаем")
                continue

            image_paths = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
            logger.info(f"Класс '{class_name}': найдено {len(image_paths)} изображений")

            for img_path in tqdm(image_paths, desc=f"Нарезка ({class_name})", leave=False):
                image = imread_rgb(img_path)
                H, W = image.shape[:2]

                for y in range(0, max(H - chunk_size + 1, 1), max(stride, 1)):
                    for x in range(0, max(W - chunk_size + 1, 1), max(stride, 1)):
                        # Обработка маленьких изображений
                        y_end = min(y + chunk_size, H)
                        x_end = min(x + chunk_size, W)
                        img_chunk = image[y:y_end, x:x_end]

                        # Дополняем до chunk_size если нужно
                        if img_chunk.shape[0] < chunk_size or img_chunk.shape[1] < chunk_size:
                            padded = np.zeros((chunk_size, chunk_size, 3), dtype=np.uint8)
                            padded[:img_chunk.shape[0], :img_chunk.shape[1]] = img_chunk
                            img_chunk = padded

                        chunk_name = f"{img_path.stem}_y{y}_x{x}"
                        imwrite(chunk_dir / f"{chunk_name}.png",
                                cv2.cvtColor(img_chunk, cv2.COLOR_RGB2BGR))

                        chunk_meta.append({
                            "image": str(chunk_dir / f"{chunk_name}.png"),
                            "class_id": class_id,
                            "class_name": class_name,
                            "source": img_path.stem,
                        })

        logger.info(f"Всего чанков: {len(chunk_meta)}")
        return chunk_meta