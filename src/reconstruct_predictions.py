# src/reconstruct_predictions.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .device import get_device
from .model import UNet
from utils import MASK_DIR, OUTPUT_DIR, VAL_FILE, read_paths


PATCH_SIZE = 256
STRIDE = 128
BASE_CHANNELS = 32
INFERENCE_BATCH_SIZE = 4
THRESHOLD = 0.65
DATA_AUGMENTATION = True

if DATA_AUGMENTATION :
    CHECKPOINT_PATH = Path("outputs/checkpoints/best_model_augmented.pt")
    RECONSTRUCTION_DIR = (OUTPUT_DIR / "reconstructed_augmented")
else:
    CHECKPOINT_PATH = Path("outputs/checkpoints/best_model.pt")
    RECONSTRUCTION_DIR = (OUTPUT_DIR / "reconstructed")


def get_positions(length, patch_size, stride):
    """
    Return patch start positions while ensuring that
    the final patch reaches the image boundary.
    """
    if length < patch_size:
        raise ValueError(
            f"Image dimension {length} is smaller than "
            f"patch size {patch_size}"
        )

    positions = list(
        range(
            0,
            length - patch_size + 1,
            stride,
        )
    )

    final_position = length - patch_size

    if positions[-1] != final_position:
        positions.append(final_position)

    return positions


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

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


def image_to_tensor(image):
    """
    PIL RGB image → torch tensor [3,H,W].
    """
    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    image_array = np.transpose(
        image_array,
        (2, 0, 1),
    )

    image_array /= 255.0

    return torch.from_numpy(image_array)


def reconstruct_one_image(
    image_path,
    model,
    device,
):
    """
    Reconstruct a complete image prediction by
    averaging overlapping patch probabilities.
    """
    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    x_positions = get_positions(
        width,
        PATCH_SIZE,
        STRIDE,
    )

    y_positions = get_positions(
        height,
        PATCH_SIZE,
        STRIDE,
    )

    probability_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    prediction_count = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    patch_tensors = []
    patch_locations = []

    def process_batch():
        if not patch_tensors:
            return

        batch = torch.stack(
            patch_tensors,
            dim=0,
        ).to(device)

        with torch.no_grad():
            logits = model(batch)
            probabilities = torch.sigmoid(logits)

        probabilities = (
            probabilities[:, 0]
            .detach()
            .cpu()
            .numpy()
        )

        for probability, (x, y) in zip(
            probabilities,
            patch_locations,
        ):
            probability_sum[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE,
            ] += probability

            prediction_count[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE,
            ] += 1.0

        patch_tensors.clear()
        patch_locations.clear()

    for y in y_positions:
        for x in x_positions:
            patch = image.crop(
                (
                    x,
                    y,
                    x + PATCH_SIZE,
                    y + PATCH_SIZE,
                )
            )

            patch_tensor = image_to_tensor(patch)

            patch_tensors.append(patch_tensor)
            patch_locations.append((x, y))

            if len(patch_tensors) >= INFERENCE_BATCH_SIZE:
                process_batch()

    process_batch()

    reconstructed_probability = (
        probability_sum
        / np.maximum(prediction_count, 1.0)
    )

    reconstructed_mask = (
        reconstructed_probability >= THRESHOLD
    ).astype(np.uint8)

    return (
        np.asarray(image),
        reconstructed_probability,
        reconstructed_mask,
    )


def save_results(
    image_path,
    image,
    probability,
    prediction,
):
    RECONSTRUCTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_name = image_path.stem

    probability_path = (
        RECONSTRUCTION_DIR
        / f"{image_name}_probability.npy"
    )

    mask_path = (
        RECONSTRUCTION_DIR
        / f"{image_name}_prediction.png"
    )

    figure_path = (
        RECONSTRUCTION_DIR
        / f"{image_name}_reconstruction.png"
    )

    np.save(
        probability_path,
        probability,
    )

    prediction_image = Image.fromarray(
        prediction * 255
    )

    prediction_image.save(mask_path)

    overlay = image.copy()

    foreground = prediction.astype(bool)

    overlay[foreground] = (
        0.6 * overlay[foreground]
        + 0.4 * np.array(
            [255, 0, 0],
            dtype=np.float32,
        )
    ).astype(np.uint8)

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    axes[0].imshow(image)
    axes[0].set_title("Original image")

    axes[1].imshow(
        probability,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("Probability map")

    axes[2].imshow(
        prediction,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Reconstructed mask")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    figure.savefig(
        figure_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Saved:")
    print(" ", probability_path)
    print(" ", mask_path)
    print(" ", figure_path)


def main():
    device = get_device()

    print("=" * 60)
    print("Full-image reconstruction")
    print("=" * 60)
    print("Using device:", device)
    print("Patch size:", PATCH_SIZE)
    print("Stride:", STRIDE)
    print("Checkpoint:", CHECKPOINT_PATH)

    model = load_model(device)

    validation_paths = read_paths(VAL_FILE)

    print("Validation images:", len(validation_paths))

    for index, image_path in enumerate(validation_paths):
        print()
        print(
            f"Processing image {index + 1}/"
            f"{len(validation_paths)}:"
        )
        print(image_path.name)

        image, probability, prediction = (
            reconstruct_one_image(
                image_path=image_path,
                model=model,
                device=device,
            )
        )

        save_results(
            image_path=image_path,
            image=image,
            probability=probability,
            prediction=prediction,
        )

    print()
    print("Full-image reconstruction completed.")


if __name__ == "__main__":
    main()