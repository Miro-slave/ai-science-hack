
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SegmentationLoss(nn.Module):
    """Dice + BCE для сегментации."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        targets = targets.float()
        dice_loss = self.dice(preds, targets)
        bce_loss = self.bce(preds, targets)
        total = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return {"total": total, "dice": dice_loss, "bce": bce_loss}


class ClassificationLoss(nn.Module):
    """CrossEntropy с опциональными class weights."""

    def __init__(self, num_classes: int = 2, use_weights: bool = True,
                 chunk_meta: list | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.use_weights = use_weights
        self.chunk_meta = chunk_meta

        weight = self._compute_weights() if use_weights else None
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def _compute_weights(self) -> torch.Tensor | None:
        if self.chunk_meta is None:
            return None
        class_counts = [0] * self.num_classes
        for item in self.chunk_meta:
            class_counts[item["class_id"]] += 1
        total = sum(class_counts)
        if any(c == 0 for c in class_counts):
            return None  # нельзя делить на 0
        weights = [total / (self.num_classes * c) for c in class_counts]
        return torch.tensor(weights, dtype=torch.float32)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        loss = self.ce(preds, targets)
        return {"total": loss, "ce": loss}