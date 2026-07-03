# cv2_utils.py

# cv2_utils.py
import numpy as np
import cv2
from pathlib import Path

# Расширения которые поддерживает cv2
_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread с поддержкой non-ASCII путей."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return img


def imwrite(path: str | Path, img: np.ndarray, params=None) -> bool:
    """
    cv2.imwrite с поддержкой non-ASCII путей.
    Поддерживаемые форматы: png, jpg, bmp, tif, tiff, webp.
    Для .npy используй np.save().
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Неподдерживаемое расширение '{ext}' для imwrite. "
            f"Поддерживаются: {_SUPPORTED_EXTENSIONS}. "
            f"Для .npy используй np.save()."
        )

    encode_params = params or []
    success, data = cv2.imencode(ext, img, encode_params)

    if not success:
        raise RuntimeError(f"cv2.imencode не смог закодировать изображение в формат {ext}")

    data.tofile(str(path))
    return True


def imread_rgb(path: str | Path) -> np.ndarray:
    """Читает изображение и возвращает RGB."""
    img = imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imread_gray(path: str | Path) -> np.ndarray:
    """Читает изображение в градациях серого."""
    return imread(path, cv2.IMREAD_GRAYSCALE)