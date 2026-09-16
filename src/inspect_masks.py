from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import PROJECT_ROOT, RAW_DATA_DIR, IMG_DATA_DIR, XML_DATA_DIR , MASK_DIR


def main():

    image_files = sorted(IMG_DATA_DIR.glob("*.tif"))

    if not image_files:
        print("No processed images found.")
        print("Run the dataset conversion first.")
        return

    # Pick the first image.
    image_path = image_files[10]

    mask_path = MASK_DIR / f"{image_path.stem}.png"

    if not mask_path.exists():
        print(f"Missing mask: {mask_path}")
        return

    image = np.array(Image.open(image_path).convert("RGB"))

    mask = np.array(Image.open(mask_path))

    print("Image information")
    print("-----------------")
    print("Filename:", image_path.name)
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image dtype:", image.dtype)
    print("Mask dtype:", mask.dtype)
    print("Mask unique values:", np.unique(mask))

    # Convert mask to binary.
    binary_mask = mask > 0

    print("Nuclei pixels:", binary_mask.sum())
    print("Background pixels:", (~binary_mask).sum())

    # Plot.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground-truth mask")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(binary_mask, cmap="jet", alpha=0.35)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    output_path = PROJECT_ROOT / "outputs" / "first_sample.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    print()
    print("Saved visualization to:")
    print(output_path)

    plt.show()


if __name__ == "__main__":
    main()