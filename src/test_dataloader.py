from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import MoNuSegDataset, MoNuSegPatchDataset
from utils import TRAIN_FILE, VAL_FILE, read_paths

PATCH_SIZE = 256
STRIDE = 128
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 2
NUM_WORKERS = 0



def main():

    train_paths = read_paths(TRAIN_FILE)
    val_paths = read_paths(VAL_FILE)

    train_dataset = MoNuSegDataset(
        train_paths,
        image_size=IMAGE_SIZE,
    )

    val_dataset = MoNuSegDataset(
        val_paths,
        image_size=IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("Training dataset:", len(train_dataset))
    print("Validation dataset:", len(val_dataset))

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # --------------------------------------------------
    # Training batch
    # --------------------------------------------------

    images, masks = next(iter(train_loader))

    print()
    print("Training batch:")
    print("  Images:", images.shape)
    print("  Masks :", masks.shape)

    print("  Image dtype:", images.dtype)
    print("  Mask dtype :", masks.dtype)

    print("  Image range:",
          images.min().item(),
          "to",
          images.max().item())

    print("  Mask values:",
          torch.unique(masks))

    # --------------------------------------------------
    # Validation batch
    # --------------------------------------------------

    images, masks = next(iter(val_loader))

    print()
    print("Validation batch:")
    print("  Images:", images.shape)
    print("  Masks :", masks.shape)

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert images.shape == (
        BATCH_SIZE,
        3,
        256,
        256,
    )

    assert masks.shape == (
        BATCH_SIZE,
        1,
        256,
        256,
    )

    print()
    print("DataLoader test: PASSED")

    print("-"*30)
    print("patch dataloader")
    print("-"*30)
    #-------------------------------
    #patch dataloader
    #-------------------------------
    train_paths = read_paths(TRAIN_FILE)
    val_paths = read_paths(VAL_FILE)

    train_dataset = MoNuSegPatchDataset(
        image_paths=train_paths,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
    )

    val_dataset = MoNuSegPatchDataset(
        image_paths=val_paths,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("Training images:", len(train_paths))
    print("Validation images:", len(val_paths))

    print("Training patches:", len(train_dataset))
    print("Validation patches:", len(val_dataset))

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    images, masks = next(iter(train_loader))

    print()
    print("Training batch:")
    print("  Images:", images.shape)
    print("  Masks :", masks.shape)
    print("  Image dtype:", images.dtype)
    print("  Mask dtype :", masks.dtype)
    print(
        "  Image range:",
        images.min().item(),
        "to",
        images.max().item(),
    )
    print("  Mask values:", torch.unique(masks))

    assert images.shape == (
        BATCH_SIZE,
        3,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    assert masks.shape == (
        BATCH_SIZE,
        1,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    print()
    print("Patch DataLoader test: PASSED")

if __name__ == "__main__":
    main()