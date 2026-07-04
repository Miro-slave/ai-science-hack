# trainer.py
import logging
from tqdm import tqdm

from SegmentationConfig import SegmentationConfig, ClassificationConfig

logger = logging.getLogger(__name__)


# trainer.py
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# ─────────────────────────────────────────
# Базовый Trainer
# ─────────────────────────────────────────

class BaseTrainer:
    """Общий каркас. Наследуемся и переопределяем _compute_metrics."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config,
        log_dir: str,
        checkpoint_dir: str,
        device: str,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device(device)

        self.global_step = 0
        self.best_metric = 0.0
        self.writer = SummaryWriter(log_dir=log_dir)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)

    def fit(self, num_epochs: int, save_interval: int = 5,
            log_interval: int = 10) -> dict:
        raise NotImplementedError

    @torch.no_grad()
    def _evaluate(self) -> dict:
        raise NotImplementedError

    def _save_checkpoint(self, epoch: int, metric_value: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }
        if is_best:
            torch.save(state, self.checkpoint_dir / "best_model.pt")



# ─────────────────────────────────────────
# Segmentation Trainer
# ─────────────────────────────────────────

class SegmentationTrainer(BaseTrainer):
    """Обучение сегментации (тальк/не-тальк)."""

    def __init__(self, config: SegmentationConfig, model: nn.Module,
                 loss_fn: nn.Module, optimizer, scheduler,
                 train_loader: DataLoader, test_loader: DataLoader):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            log_dir=config.log_dir,
            checkpoint_dir=config.checkpoint_dir,
            device=config.device,
        )

    def fit(self, num_epochs: int) -> dict:
        history = {"train_loss": [], "test_loss": [], "iou": [], "dice": []}

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            metrics = self._evaluate()

            self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["test_loss"].append(metrics["loss"])
            history["iou"].append(metrics["iou"])
            history["dice"].append(metrics["dice"])

            self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
            self.writer.add_scalar("epoch/test_loss", metrics["loss"], epoch)
            self.writer.add_scalar("epoch/iou", metrics["iou"], epoch)
            self.writer.add_scalar("epoch/dice", metrics["dice"], epoch)
            self.writer.add_scalar(
                "epoch/lr", self.optimizer.param_groups[0]["lr"], epoch,
            )

            logger.info(
                f"Seg [Epoch {epoch:02d}/{num_epochs}] | "
                f"Loss: {train_loss:.4f}/{metrics['loss']:.4f} | "
                f"IoU: {metrics['iou']:.4f} | Dice: {metrics['dice']:.4f}"
            )

            if metrics["iou"] > self.best_metric:
                self.best_metric = metrics["iou"]
                self._save_checkpoint(epoch, self.best_metric, is_best=True)
                logger.info(f"  ✓ Best saved (IoU={self.best_metric:.4f})")
            elif epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch, metrics["iou"])

        self.writer.close()
        return history

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = self.train_loader
        for images, masks in tqdm(pbar):
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32).unsqueeze(1)

            self.optimizer.zero_grad()
            preds = self.model(images)
            loss_dict = self.loss_fn(preds, masks)
            loss = loss_dict["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            if self.global_step % self.config.log_interval == 0:
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f"step/{k}", v.item(), self.global_step)

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _evaluate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        tp = fp = fn = tn = 0

        for images, masks in self.test_loader:
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32).unsqueeze(1)

            preds = self.model(images)
            loss_dict = self.loss_fn(preds, masks)
            total_loss += loss_dict["total"].item()

            preds_bin = (torch.sigmoid(preds) > 0.5).float()
            tp += (preds_bin * masks).sum().item()
            fp += (preds_bin * (1 - masks)).sum().item()
            fn += ((1 - preds_bin) * masks).sum().item()
            tn += ((1 - preds_bin) * (1 - masks)).sum().item()

        iou = tp / (tp + fp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

        return {
            "loss": total_loss / len(self.test_loader),
            "iou": iou,
            "dice": dice,
        }


# ─────────────────────────────────────────
# Classification Trainer
# ─────────────────────────────────────────

class ClassificationTrainer(BaseTrainer):
    """Обучение классификации (рядовая/труднообогатимая)."""

    def __init__(self, config: ClassificationConfig, model: nn.Module,
                 loss_fn: nn.Module, optimizer, scheduler,
                 train_loader: DataLoader, test_loader: DataLoader):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            log_dir=config.log_dir,
            checkpoint_dir=config.checkpoint_dir,
            device=config.device,
        )

    def fit(self, num_epochs: int) -> dict:
        history = {"train_loss": [], "test_loss": [], "accuracy": [], "f1": []}

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            metrics = self._evaluate()

            self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["test_loss"].append(metrics["loss"])
            history["accuracy"].append(metrics["accuracy"])
            history["f1"].append(metrics["f1"])

            self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
            self.writer.add_scalar("epoch/test_accuracy", metrics["accuracy"], epoch)
            self.writer.add_scalar("epoch/test_f1", metrics["f1"], epoch)
            self.writer.add_scalar(
                "epoch/lr", self.optimizer.param_groups[0]["lr"], epoch,
            )

            logger.info(
                f"Cls  [Epoch {epoch:02d}/{num_epochs}] | "
                f"Loss: {train_loss:.4f}/{metrics['loss']:.4f} | "
                f"Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}"
            )

            if metrics["accuracy"] > self.best_metric:
                self.best_metric = metrics["accuracy"]
                self._save_checkpoint(epoch, self.best_metric, is_best=True)
                logger.info(f"  ✓ Best saved (Acc={self.best_metric:.4f})")
            elif epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch, metrics["accuracy"])

        self.writer.close()
        return history

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0 # TQDM
        for images, labels in tqdm(self.train_loader):
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss_dict = self.loss_fn(logits, labels)
            loss = loss_dict["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _evaluate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        correct = total = 0
        tp = fp = fn = 0

        for images, labels in self.test_loader:
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            logits = self.model(images)
            loss_dict = self.loss_fn(logits, labels)
            total_loss += loss_dict["total"].item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

            # Macro F1
            for c in range(self.config.num_classes):
                tp_c = ((preds == c) & (labels == c)).sum().item()
                fp_c = ((preds == c) & (labels != c)).sum().item()
                fn_c = ((preds != c) & (labels == c)).sum().item()
                tp += tp_c
                fp += fp_c
                fn += fn_c

        accuracy = correct / total
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "loss": total_loss / len(self.test_loader),
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
    