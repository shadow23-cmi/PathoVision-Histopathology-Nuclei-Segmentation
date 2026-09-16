# src/evaluate_reconstructed.py

from pathlib import Path

import numpy as np
from PIL import Image

from utils import MASK_DIR, OUTPUT_DIR, VAL_FILE, read_paths

DATA_AUGMENTATION = True

if DATA_AUGMENTATION :
    RECONSTRUCTION_DIR = (OUTPUT_DIR / "reconstructed_augmented")
else:
    RECONSTRUCTION_DIR = (OUTPUT_DIR / "reconstructed")


def binary_dice(prediction, target, smooth=1.0):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    return (
        2.0 * intersection + smooth
    ) / (
        prediction.sum()
        + target.sum()
        + smooth
    )


def binary_iou(prediction, target, smooth=1.0):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    union = np.logical_or(
        prediction,
        target,
    ).sum()

    return (
        intersection + smooth
    ) / (
        union + smooth
    )


def main():
    validation_paths = read_paths(VAL_FILE)

    all_dice = []
    all_iou = []

    print("=" * 60)
    print("Full-image evaluation")
    print("=" * 60)

    for image_path in validation_paths:
        image_name = image_path.stem

        prediction_path = (
            RECONSTRUCTION_DIR
            / f"{image_name}_prediction.png"
        )

        mask_path = MASK_DIR / image_path.name

        prediction = np.asarray(
            Image.open(prediction_path).convert("L"),
            dtype=np.uint8,
        )

        target = np.asarray(
            Image.open(mask_path).convert("L"),
            dtype=np.uint8,
        )

        prediction = prediction > 127
        target = target > 127

        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {image_name}: "
                f"prediction={prediction.shape}, "
                f"target={target.shape}"
            )

        dice = binary_dice(
            prediction,
            target,
        )

        iou = binary_iou(
            prediction,
            target,
        )

        all_dice.append(dice)
        all_iou.append(iou)

        print()
        print(image_name)
        print("  Dice:", dice)
        print("  IoU :", iou)

    print()
    print("=" * 60)
    print("Average full-image metrics")
    print("=" * 60)
    print("Images evaluated:", len(all_dice))
    print("Mean Dice:", np.mean(all_dice))
    print("Mean IoU :", np.mean(all_iou))


if __name__ == "__main__":
    main()