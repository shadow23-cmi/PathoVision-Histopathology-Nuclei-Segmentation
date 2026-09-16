from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from utils import OUTPUT_DIR, VAL_FILE


RECONSTRUCTION_DIR = (
    OUTPUT_DIR / "reconstructed_augmented"
)

INSTANCE_DIR = (
    OUTPUT_DIR / "instances"
)

THRESHOLD = 0.65

# Minimum distance between nucleus centers.
# Start conservatively.
MIN_DISTANCE = 10


def read_paths(file_path):
    with open(file_path, "r") as f:
        return [
            Path(line.strip())
            for line in f
            if line.strip()
        ]


def segment_instances(probability):
    """
    Convert a semantic probability map into
    individual nucleus instances.
    """

    binary_mask = probability >= THRESHOLD

    # Remove extremely small isolated components.
    binary_mask = ndimage.binary_opening(
        binary_mask,
        structure=np.ones((3, 3)),
    )

    # Distance from every foreground pixel
    # to the nearest background pixel.
    distance = ndimage.distance_transform_edt(
        binary_mask
    )

    # Find approximate centers of nuclei.
    coordinates = peak_local_max(
        distance,
        min_distance=MIN_DISTANCE,
        labels=binary_mask,
        exclude_border=False,
    )

    markers = np.zeros(
        binary_mask.shape,
        dtype=np.int32,
    )

    for marker_id, coordinate in enumerate(
        coordinates,
        start=1,
    ):
        y, x = coordinate
        markers[y, x] = marker_id

    # Watershed separates touching nuclei.
    instances = watershed(
        -distance,
        markers,
        mask=binary_mask,
    )

    return (
        binary_mask,
        distance,
        markers,
        instances,
    )


def save_instance_visualization(
    image_path,
    binary_mask,
    distance,
    markers,
    instances,
):
    INSTANCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_name = image_path.stem

    instance_path = (
        INSTANCE_DIR
        / f"{image_name}_instances.npy"
    )

    visualization_path = (
        INSTANCE_DIR
        / f"{image_name}_instances.png"
    )

    np.save(
        instance_path,
        instances,
    )

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    axes[0].imshow(
        binary_mask,
        cmap="gray",
    )
    axes[0].set_title("Semantic mask")

    axes[1].imshow(
        distance,
        cmap="viridis",
    )
    axes[1].set_title("Distance transform")

    axes[2].imshow(
        markers,
        cmap="nipy_spectral",
    )
    axes[2].set_title("Markers")

    axes[3].imshow(
        instances,
        cmap="nipy_spectral",
    )
    axes[3].set_title("Instances")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    figure.savefig(
        visualization_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    return instance_path, visualization_path


def main():
    validation_paths = read_paths(
        VAL_FILE
    )

    INSTANCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("Instance segmentation")
    print("=" * 60)

    for index, image_path in enumerate(
        validation_paths,
        start=1,
    ):
        image_name = image_path.stem

        probability_path = (
            RECONSTRUCTION_DIR
            / f"{image_name}_probability.npy"
        )

        probability = np.load(
            probability_path
        )

        (
            binary_mask,
            distance,
            markers,
            instances,
        ) = segment_instances(
            probability
        )

        instance_count = (
            len(np.unique(instances)) - 1
        )

        print(
            f"{index}/{len(validation_paths)} "
            f"{image_name}: "
            f"{instance_count} nuclei"
        )

        # Visualization will be added after
        # the basic segmentation is verified.
        figure, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4),
        )

        axes[0].imshow(
            binary_mask,
            cmap="gray",
        )
        axes[0].set_title("Semantic mask")

        axes[1].imshow(
            distance,
            cmap="viridis",
        )
        axes[1].set_title("Distance transform")

        axes[2].imshow(
            markers,
            cmap="nipy_spectral",
        )
        axes[2].set_title("Nucleus markers")

        axes[3].imshow(
            instances,
            cmap="nipy_spectral",
        )
        axes[3].set_title(
            f"Instances ({instance_count})"
        )

        for axis in axes:
            axis.axis("off")

        plt.tight_layout()

        output_path = (
            INSTANCE_DIR
            / f"{image_name}_instances.png"
        )

        figure.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)

        print("  Saved:", output_path)

if __name__ == "__main__":
    main()