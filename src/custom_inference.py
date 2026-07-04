# inference_main.py
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from BuildModels import build_segmentor, build_classifier
from Inferencer import ClassificationInferencer, SegmentationInferencer
from SegmentationConfig import InferenceConfig
from Visualiser import Visualizer
from cv2_utils import imread_rgb, imwrite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_segmentor(checkpoint_path: str, config: InferenceConfig):
    seg_config = config.seg_config
    model = build_segmentor(seg_config)
    state = torch.load(checkpoint_path, map_location=config.device,weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    logger.info(f"Сегментатор загружен из {checkpoint_path}")
    return model


def load_classifier(checkpoint_path: str, config: InferenceConfig):
    cls_config = config.cls_config
    model = build_classifier(cls_config)
    state = torch.load(checkpoint_path, map_location=config.device,weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    logger.info(f"Классификатор загружен из {checkpoint_path}")
    return model


def run_inference(image_path: str, output_dir: str = "results"):
    """
    Полный пайплайн инференса:
    1. Классификация изображения
    2. Сегментация талька
    3. Сохранение heatmap, маски, overlay, визуализации
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = InferenceConfig(
        seg_checkpoint="checkpoints/talc/best_model.pt",
        cls_checkpoint="checkpoints/classification/best_model.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        segmentation_threshold=0.5,
        chunk_size=512,
        overlap=64,
    )

    # ─── Загрузка моделей ───
    seg_model = load_segmentor(config.seg_checkpoint, config)
    cls_model = load_classifier(config.cls_checkpoint, config)

    seg_inferencer = SegmentationInferencer(seg_model, config)
    cls_inferencer = ClassificationInferencer(cls_model, config)

    # ─── Загрузка изображения ───
    logger.info(f"Загрузка: {image_path}")
    image = imread_rgb(image_path)
    logger.info(f"Размер: {image.shape[1]}×{image.shape[0]} px")

    # ─── Классификация ───
    logger.info("Классификация...")
    cls_result = cls_inferencer.predict(image)
    logger.info(
        f"Класс: {cls_result['image_class_name']} "
        f"(уверенность: {cls_result['image_confidence']:.1%})"
    )
    logger.info(f"Распределение голосов: {cls_result['vote_breakdown']}")

    # ─── Сегментация ───
    logger.info("Сегментация...")
    probs, heatmap = seg_inferencer.predict(image)
    binary_mask = seg_inferencer.predict_binary(image)

    talc_pixels = (binary_mask > 0).sum()
    total_pixels = binary_mask.size
    talc_share = talc_pixels / total_pixels
    logger.info(
        f"Тальк: {talc_pixels:,} px / {total_pixels:,} px ({talc_share:.1%})"
    )

    # ─── Сохранение результатов ───
    stem = Path(image_path).stem

    # Бинарная маска
    imwrite(output_dir / f"{stem}_mask.png", binary_mask)

    # Heatmap
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    imwrite(output_dir / f"{stem}_heatmap.png", heatmap_bgr)

    # Вероятности — сохраняем как .npy, НЕ через cv2
    np.save(str(output_dir / f"{stem}_probs.npy"), probs)

    # Overlay
    overlay = Visualizer.create_overlay(image, binary_mask, alpha=0.5)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    imwrite(output_dir / f"{stem}_overlay.png", overlay_bgr)

    # Финальная визуализация
    Visualizer.visualize_inference_results(
        image=image,
        probs=probs,
        binary_mask=binary_mask,
        heatmap=heatmap,
        cls_result=cls_result,
        save_path=str(output_dir / f"{stem}_result.png"),
    )

    # Текстовый отчёт
    talc_share = (binary_mask > 0).sum() / binary_mask.size
    report = (
        f"Image:          {image_path}\n"
        f"Size:           {image.shape[1]}×{image.shape[0]} px\n"
        f"Classification: {cls_result['image_class_name']} "
        f"({cls_result['image_confidence']:.1%})\n"
        f"Votes:          {cls_result['vote_breakdown']}\n"
        f"Talc share:     {talc_share:.1%}\n"
        f"Chunks:         {len(cls_result['chunk_predictions'])}\n"
    )
    with open(output_dir / f"{stem}_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Результаты сохранены в {output_dir}/")
    logger.info(f"  Маска:        {stem}_mask.png")
    logger.info(f"  Heatmap:      {stem}_heatmap.png")
    logger.info(f"  Вероятности:  {stem}_probs.npy  (np.load для чтения)")
    logger.info(f"  Overlay:      {stem}_overlay.png")
    logger.info(f"  Визуализация: {stem}_result.png")
    logger.info(f"  Отчёт:        {stem}_report.txt")


import traceback

def ui_main(q, path_to_image, output_dir):
    try:
        run_inference(path_to_image, output_dir)
    except Exception as err:
        q.put({"success": False, "traceback": traceback.format_exc()})
        raise err
