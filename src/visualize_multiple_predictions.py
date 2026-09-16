from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import MoNuSegPatchDataset
from .device import get_device
from .model import UNet
from utils import VAL_FILE, read_paths


PATCH_SIZE = 256
STRIDE = 128
BASE_CHANNELS = 32
NUM_SAMPLES = 6

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/best_model.pt"
)




def load_model(device):
    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=BASE_CHANNELS,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def main():
    device = get_device()
    print("Using device:", device)

    val_paths = read_paths(VAL_FILE)

    dataset = MoNuSegPatchDataset(
        image_paths=val_paths,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = load_model(device)

    figure, axes = plt.subplots(
        NUM_SAMPLES,
        4,
        figsize=(12, 3 * NUM_SAMPLES),
    )

    with torch.no_grad():
        for index, (images, masks) in enumerate(loader):
            if index >= NUM_SAMPLES:
                break

            images = images.to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = (
                probabilities >= 0.5
            ).float()

            image = (
                images[0]
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )

            mask = (
                masks[0, 0]
                .numpy()
            )

            prediction = (
                predictions[0, 0]
                .detach()
                .cpu()
                .numpy()
            )

            image = np.clip(image, 0.0, 1.0)

            overlay = image.copy()
            foreground = prediction > 0.5

            overlay[foreground] = (
                0.6 * overlay[foreground]
                + 0.4 * np.array([1.0, 0.0, 0.0])
            )

            row_axes = axes[index]

            row_axes[0].imshow(image)
            row_axes[0].set_title("Image")

            row_axes[1].imshow(
                mask,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            row_axes[1].set_title("Ground truth")

            row_axes[2].imshow(
                prediction,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            row_axes[2].set_title("Prediction")

            row_axes[3].imshow(overlay)
            row_axes[3].set_title("Overlay")

            for axis in row_axes:
                axis.axis("off")

    plt.tight_layout()

    output_path = Path(
        "outputs/multiple_predictions.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print("Saved:", output_path)


if __name__ == "__main__":
    main()