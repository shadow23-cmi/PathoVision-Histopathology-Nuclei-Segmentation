from pathlib import Path

import torch

from .dataset import MoNuSegDataset, MoNuSegPatchDataset
from utils import TRAIN_FILE, VAL_FILE, read_paths

PATCH_SIZE = 256
STRIDE = 128


def inspect_dataset(name, file_path):

    paths = read_paths(file_path)

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print("Number of images:", len(paths))

    dataset = MoNuSegDataset(
        image_paths=paths,
        image_size=(256, 256),
    )

    print("Dataset length:", len(dataset))

    # Inspect first sample
    image, mask = dataset[0]

    print()
    print("Image:")
    print("  Shape:", image.shape)
    print("  Dtype:", image.dtype)
    print("  Min:", image.min().item())
    print("  Max:", image.max().item())

    print()
    print("Mask:")
    print("  Shape:", mask.shape)
    print("  Dtype:", mask.dtype)
    print("  Unique:", torch.unique(mask))

    assert image.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)

    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32

    assert torch.all(
        (mask == 0) | (mask == 1)
    )

    print()
    print("Dataset test: PASSED")

def inspect_patch_dataset(name, file_path):

    image_paths = read_paths(file_path)

    dataset = MoNuSegPatchDataset(
        image_paths=image_paths,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
    )

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print("Original images:", len(image_paths))
    print("Total patches:", len(dataset))

    image, mask = dataset[0]

    print()
    print("First patch image:")
    print("  Shape:", image.shape)
    print("  Dtype:", image.dtype)
    print("  Min:", image.min().item())
    print("  Max:", image.max().item())

    print()
    print("First patch mask:")
    print("  Shape:", mask.shape)
    print("  Dtype:", mask.dtype)
    print("  Unique:", torch.unique(mask))

    assert image.shape == (
        3,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    assert mask.shape == (
        1,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32

    assert torch.all(
        (mask == 0) | (mask == 1)
    )

    print()
    print("Patch dataset test: PASSED")

def main():

    inspect_dataset("TRAIN DATASET",TRAIN_FILE,)
    inspect_dataset("VALIDATION DATASET",VAL_FILE,)

    print("-"*60)
    print("patch dataset")
    inspect_patch_dataset("TRAIN DATASET",TRAIN_FILE,)
    inspect_patch_dataset("VALIDATION DATASET",VAL_FILE,)

if __name__ == "__main__":
    main()