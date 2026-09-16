from pathlib import Path

import numpy as np
from PIL import Image

from utils import MASK_DIR, OUTPUT_DIR, VAL_FILE, read_paths


RECONSTRUCTION_DIR = (
    OUTPUT_DIR / "reconstructed"
)

THRESHOLDS = np.arange(
    0.20,
    0.81,
    0.05,
)


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

    samples = []

    for image_path in validation_paths:
        image_name = image_path.stem

        probability_path = (
            RECONSTRUCTION_DIR
            / f"{image_name}_probability.npy"
        )

        mask_path = MASK_DIR / image_path.name

        probability = np.load(
            probability_path
        )

        target = np.asarray(
            Image.open(mask_path).convert("L"),
            dtype=np.uint8,
        )

        target = target > 127

        if probability.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {image_name}: "
                f"probability={probability.shape}, "
                f"target={target.shape}"
            )

        samples.append(
            (
                image_name,
                probability,
                target,
            )
        )

    results = []

    print("=" * 60)
    print("Threshold Calibration")
    print("=" * 60)
    print("Validation images:", len(samples))
    print()

    for threshold in THRESHOLDS:
        dice_scores = []
        iou_scores = []
        foreground_ratios = []

        for image_name, probability, target in samples:
            prediction = probability >= threshold

            dice = binary_dice(
                prediction,
                target,
            )

            iou = binary_iou(
                prediction,
                target,
            )

            foreground_ratio = prediction.mean()

            dice_scores.append(dice)
            iou_scores.append(iou)
            foreground_ratios.append(
                foreground_ratio
            )

        mean_dice = float(
            np.mean(dice_scores)
        )

        mean_iou = float(
            np.mean(iou_scores)
        )

        mean_foreground_ratio = float(
            np.mean(foreground_ratios)
        )

        results.append(
            {
                "threshold": float(threshold),
                "dice": mean_dice,
                "iou": mean_iou,
                "foreground_ratio": (
                    mean_foreground_ratio
                ),
            }
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"Dice: {mean_dice:.6f} | "
            f"IoU: {mean_iou:.6f} | "
            f"Predicted foreground: "
            f"{mean_foreground_ratio:.4f}"
        )

    best_result = max(
        results,
        key=lambda item: item["dice"],
    )

    print()
    print("=" * 60)
    print("Best threshold")
    print("=" * 60)
    print(
        f"Threshold: "
        f"{best_result['threshold']:.2f}"
    )
    print(
        f"Dice: "
        f"{best_result['dice']:.6f}"
    )
    print(
        f"IoU: "
        f"{best_result['iou']:.6f}"
    )
    print(
        f"Predicted foreground ratio: "
        f"{best_result['foreground_ratio']:.4f}"
    )

    output_path = (
        OUTPUT_DIR
        / "threshold_results.csv"
    )

    with open(output_path, "w") as file:
        file.write(
            "threshold,dice,iou,"
            "foreground_ratio\n"
        )

        for result in results:
            file.write(
                f"{result['threshold']:.2f},"
                f"{result['dice']:.8f},"
                f"{result['iou']:.8f},"
                f"{result['foreground_ratio']:.8f}\n"
            )

    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()