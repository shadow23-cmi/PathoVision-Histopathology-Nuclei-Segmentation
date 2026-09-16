from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .dataset import MoNuSegPatchDataset
from .device import get_device
from .model import UNet
from utils import TRAIN_FILE, VAL_FILE, read_paths


PATCH_SIZE = 256
STRIDE = 128
BATCH_SIZE = 1
BASE_CHANNELS = 32

CHECKPOINT_PATH = Path("outputs/checkpoints/best_model.pt")

def main():
    device = get_device()

    print("Using device:", device)

    val_paths = read_paths(VAL_FILE)

    val_dataset = MoNuSegPatchDataset(
        image_paths=val_paths,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

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

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )
        elif "state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["state_dict"]
            )
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    images, masks = next(iter(val_loader))

    images = images.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        logits = model(images)
        probabilities = torch.sigmoid(logits)
        predictions = (
            probabilities >= 0.5
        ).float()

    image = images[0].detach().cpu()
    mask = masks[0, 0].detach().cpu().numpy()
    prediction = (
        predictions[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0)

    overlay = image.copy()

    # Red where the model predicts foreground.
    overlay[prediction > 0.5] = (
        0.6 * overlay[prediction > 0.5]
        + 0.4 * np.array([1.0, 0.0, 0.0])
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    axes[0].imshow(image)
    axes[0].set_title("Image patch")

    axes[1].imshow(
        mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Ground truth")

    axes[2].imshow(
        prediction,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Prediction")

    axes[3].imshow(overlay)
    axes[3].set_title("Prediction overlay")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    output_path = Path(
        "outputs/prediction_patch.png"
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

    print("Saved visualization to:", output_path)


if __name__ == "__main__":
    main()