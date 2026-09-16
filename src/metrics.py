import torch


def dice_score(logits, targets, threshold=0.5, smooth=1.0):
    """
    Compute Dice score for binary segmentation.

    logits:
        [B, 1, H, W]

    targets:
        [B, 1, H, W]

    Returns:
        scalar Dice score
    """

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= threshold).float()

    predictions = predictions.view(predictions.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    intersection = (predictions * targets).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + smooth
    )

    return dice.mean()


def iou_score(logits, targets, threshold=0.5, smooth=1.0):
    """
    Compute Intersection over Union (IoU).

    IoU = intersection / union
    """

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= threshold).float()

    predictions = predictions.view(predictions.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    intersection = (predictions * targets).sum(dim=1)

    union = (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        - intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean()