import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Input:
        logits: [B, 1, H, W]
        targets: [B, 1, H, W]

    The model outputs logits, so sigmoid is applied internally.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)

        # Flatten each image into one vector
        probabilities = probabilities.view(
            probabilities.shape[0],
            -1,
        )

        targets = targets.view(
            targets.shape[0],
            -1,
        )

        intersection = (probabilities * targets).sum(dim=1)

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        dice_loss = 1.0 - dice_score

        return dice_loss.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCE-with-logits and Dice loss.

    total_loss = bce_weight * BCE + dice_weight * Dice
    """

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )

        return total_loss