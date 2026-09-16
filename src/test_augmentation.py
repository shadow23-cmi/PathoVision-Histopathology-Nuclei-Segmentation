from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .dataset import MoNuSegPatchDataset
from utils import TRAIN_FILE, read_paths


PATCH_SIZE = 256
STRIDE = 128



def main():
    train_paths = read_paths(TRAIN_FILE)

    dataset = MoNuSegPatchDataset(
        image_paths=train_paths,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        augment=True,
    )

    image1, mask1 = dataset[0]
    image2, mask2 = dataset[0]

    image1 = (
        image1
        .permute(1, 2, 0)
        .numpy()
    )

    image2 = (
        image2
        .permute(1, 2, 0)
        .numpy()
    )

    mask1 = mask1[0].numpy()
    mask2 = mask2[0].numpy()

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(8, 8),
    )

    axes[0, 0].imshow(image1)
    axes[0, 0].set_title("Augmented image 1")

    axes[0, 1].imshow(mask1, cmap="gray")
    axes[0, 1].set_title("Augmented mask 1")

    axes[1, 0].imshow(image2)
    axes[1, 0].set_title("Augmented image 2")

    axes[1, 1].imshow(mask2, cmap="gray")
    axes[1, 1].set_title("Augmented mask 2")

    for axis in axes.flat:
        axis.axis("off")

    plt.tight_layout()

    output_path = "outputs/augmentation_test.png"

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Saved:", output_path)


if __name__ == "__main__":
    main()