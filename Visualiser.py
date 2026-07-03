# visualizer.py
import numpy as np
import matplotlib.pyplot as plt
import cv2

from cv2_utils import imwrite


class Visualizer:
    """Визуализация результатов инференса."""

    # Цвета классов в RGB
    CLASS_COLORS = {
        "ordinary": (0, 200, 0),      # зелёный
        "difficult": (255, 165, 0),  # оранжевый
        "talc": (0, 255, 255),       # жёлтый (тальк)
        "not_talc": (128, 128, 128), # серый (не тальк)
    }

    @staticmethod
    def create_overlay(image: np.ndarray, mask: np.ndarray,
                       alpha: float = 0.5) -> np.ndarray:
        """
        Накладывает маску на оригинальное изображение.
        Все пиксели талька окрашиваются в CLASS_COLORS["talc"].
        """
        if mask.ndim == 2:
            mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            talc_mask = mask > 127
            mask_rgb[talc_mask] = Visualizer.CLASS_COLORS["talc"]

        overlay = image.copy()
        talc = mask > 127
        overlay[talc] = (overlay[talc].astype(np.float32) * (1 - alpha) +
                         np.array(Visualizer.CLASS_COLORS["talc"]) * alpha).astype(np.uint8)
        return overlay

    @staticmethod
    def visualize_inference_results(
        image: np.ndarray,
        probs: np.ndarray,
        binary_mask: np.ndarray,
        heatmap: np.ndarray,
        cls_result: dict,
        save_path: str | None = None,
    ):
        """
        Создаёт финальную визуализацию с панелью из 5 элементов:
        1. Оригинальное изображение
        2. Heatmap уверенности
        3. Бинарная маска
        4. Overlay оригинал + маска
        5. Информация о классификации
        """
        fig, axes = plt.subplots(1, 5, figsize=(25, 6))

        # 1. Оригинал
        axes[0].imshow(image)
        axes[0].set_title("Исходное изображение", fontsize=12)
        axes[0].axis("off")

        # 2. Heatmap
        axes[1].imshow(heatmap)
        axes[1].set_title("Heatmap уверенности\n(синий=0%, красный=100%)", fontsize=12)
        axes[1].axis("off")

        # 3. Бинарная маска
        axes[2].imshow(binary_mask, cmap="gray")
        axes[2].set_title("Маска талька", fontsize=12)
        axes[2].axis("off")

        # 4. Overlay
        overlay = Visualizer.create_overlay(image, binary_mask, alpha=0.5)
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay: оригинал + маска", fontsize=12)
        axes[3].axis("off")

        # 5. Информация о классификации
        axes[4].axis("off")
        cls_text = (
            f"═══ Классификация ═══\n\n"
            f"Класс изображения:\n"
            f"  {cls_result['image_class_name']}\n\n"
            f"Уверенность:\n"
            f"  {cls_result['image_confidence']:.1%}\n\n"
            f"Голоса:\n"
        )
        for name, vote in cls_result["vote_breakdown"].items():
            cls_text += f"  {name}: {vote:.1%}\n"

        cls_text += f"\nЧанков: {len(cls_result['chunk_predictions'])}"
        axes[4].text(0.1, 0.5, cls_text, transform=axes[4].transAxes,
                     fontsize=11, verticalalignment="center",
                     fontfamily="monospace",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def visualize_segmentation_only(
        image: np.ndarray,
        probs: np.ndarray,
        binary_mask: np.ndarray,
        heatmap: np.ndarray,
        save_path: str | None = None,
    ):
        """Визуализация только для сегментации."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(image)
        axes[0].set_title("Исходное изображение")
        axes[0].axis("off")

        axes[1].imshow(heatmap)
        axes[1].set_title("Heatmap уверенности")
        axes[1].axis("off")

        axes[2].imshow(binary_mask, cmap="gray")
        axes[2].set_title("Маска талька")
        axes[2].axis("off")

        overlay = Visualizer.create_overlay(image, binary_mask, alpha=0.5)
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay")
        axes[3].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    @staticmethod
    def plot_segmentation_history(history: dict, save_path: str | None = None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(history["train_loss"], label="Train Loss")
        axes[0].plot(history["test_loss"], label="Test Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history["iou"], label="IoU")
        axes[1].plot(history["dice"], label="Dice")
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("Metrics")
        axes[1].legend()
        axes[1].grid(True)

        plt.suptitle("Segmentation training history")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_classification_history(history: dict, save_path: str | None = None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(history["train_loss"], label="Train Loss")
        axes[0].plot(history["test_loss"], label="Test Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history["accuracy"], label="Accuracy")
        axes[1].plot(history["f1"], label="F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("Metrics")
        axes[1].legend()
        axes[1].grid(True)

        plt.suptitle("Classification training history")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()