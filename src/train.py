# train.py
import logging
import torch

from BuildModels import build_segmentor, build_classifier
from CompositeLoss import SegmentationLoss, ClassificationLoss
from DataProcessor import SegmentationDataProcessor, ClassificationDataProcessor
from Dataset import build_segmentation_dataloaders, build_classification_dataloaders
from SegmentationConfig import SegmentationConfig, ClassificationConfig
from Trainer import SegmentationTrainer, ClassificationTrainer
from Visualiser import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def train_segmentation():
    """Обучение сегментатора талька."""
    config = SegmentationConfig(
        image_dir="./data/talc/images",
        mask_dir="./data/talc/masks",
        chunk_dir="./data/chunks_talc",
        checkpoint_dir="./checkpoints/talc",
        num_epochs=20,
        batch_size=4,
        segmentor_type="unet",
        segmentor_encoder="resnet34",
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=0,
    )

    logger.info("═══ Сегментация: подготовка данных ═══")
    processor = SegmentationDataProcessor(config)
    processor.generate_all_masks()
    chunk_meta = processor.slice_into_chunks()

    logger.info("═══ Сегментация: создание DataLoader ═══")
    train_dl, test_dl = build_segmentation_dataloaders(chunk_meta, config)

    logger.info("═══ Сегментация: построение модели ═══")
    model = build_segmentor(config)
    loss_fn = SegmentationLoss(dice_weight=config.dice_weight,
                                 bce_weight=config.bce_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                    weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=config.num_epochs)

    logger.info("═══ Сегментация: обучение ═══")
    trainer = SegmentationTrainer(config, model, loss_fn, optimizer,
                                    scheduler, train_dl, test_dl)
    history = trainer.fit(num_epochs=config.num_epochs)

    Visualizer.plot_segmentation_history(history)


def train_classification():
    """Обучение классификатора руд."""
    config = ClassificationConfig(
        classification_data_dir="data/classification",
        ore_classes=("ordinary", "difficult"),
        chunk_dir="data/chunks_classification",
        checkpoint_dir="checkpoints/classification",
        num_epochs=1,
        batch_size=8,
        classifier_type="resnet",
        classifier_variant="resnet50",
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=0,
    )

    logger.info("═══ Классификация: подготовка данных ═══")
    processor = ClassificationDataProcessor(config)
    chunk_meta = processor.slice_into_chunks()

    logger.info("═══ Классификация: создание DataLoader ═══")
    train_dl, test_dl = build_classification_dataloaders(chunk_meta, config)

    logger.info("═══ Классификация: построение модели ═══")
    model = build_classifier(config)
    loss_fn = ClassificationLoss(num_classes=config.num_classes,
                                   use_weights=config.use_class_weights,
                                   chunk_meta=chunk_meta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                    weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=config.num_epochs)

    logger.info("═══ Классификация: обучение ═══")
    trainer = ClassificationTrainer(config, model, loss_fn, optimizer,
                                     scheduler, train_dl, test_dl)
    history = trainer.fit(num_epochs=config.num_epochs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["segmentation", "classification", "both"],
                        default="both")
    args = parser.parse_args()

    if args.task in ("segmentation", "both"):
        train_segmentation()
    # if args.task in ("classification", "both"):
    #   train_classification()
