import os
import random

import numpy as np
import torch
import torch_directml
from torch.utils.data import DataLoader
from pathlib import Path

from .dataset import MoNuSegDataset, MoNuSegPatchDataset
from .model import UNet
from .losses import BCEDiceLoss
from .metrics import dice_score, iou_score
from utils import TRAIN_FILE, VAL_FILE, MASK_DIR, CHECKPOINT_DIR, read_paths
from .device import get_device

# ============================================================
# Configuration
# ============================================================

SEED = 42

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_WORKERS = 6

BASE_CHANNELS = 16*2
DATA_AUGMENTATION = True
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

NUM_EPOCHS = 20


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# One training epoch
# ============================================================

def train_one_epoch(model,loader,optimizer,loss_fn,device,):
    model.train()

    total_loss = 0.0

    for batch_idx, (images, masks) in enumerate(loader):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(
                f"  Batch {batch_idx}/{len(loader)} "
                f"Loss: {loss.item():.4f}", end="\r"
            )

    return total_loss / len(loader)


# ============================================================
# Validation
# ============================================================

def validate(model,loader,loss_fn,device,):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            loss = loss_fn(logits, masks)

            dice = dice_score(logits, masks)
            iou = iou_score(logits, masks)

            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()

    n = len(loader)

    return (
        total_loss / n,
        total_dice / n,
        total_iou / n,
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    device = get_device()

    print("=" * 60)
    print("PathoVision Training")
    print("=" * 60)

    print("Device:", device)
    print("Batch size:", BATCH_SIZE)
    print("Image size:", IMAGE_SIZE)
    print("Base channels:", BASE_CHANNELS)
    print("Learning rate:", LEARNING_RATE)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    train_paths = read_paths(TRAIN_FILE)
    val_paths = read_paths(VAL_FILE)

    print()
    print("Training images:", len(train_paths))
    print("Validation images:", len(val_paths))

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    """train_dataset = MoNuSegDataset(
        image_paths=train_paths,
        #mask_dir=MASK_DIR,
        image_size=IMAGE_SIZE,
    )

    val_dataset = MoNuSegDataset(
        image_paths=val_paths,
        #mask_dir=MASK_DIR,
        image_size=IMAGE_SIZE,
    )"""

    train_dataset = MoNuSegPatchDataset(image_paths=train_paths,patch_size=256,stride=128,augment=DATA_AUGMENTATION)
    val_dataset = MoNuSegPatchDataset(image_paths=val_paths,patch_size=256,stride=128,)

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

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

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=BASE_CHANNELS,
    ).to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_fn = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    best_dice = -1.0

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(1, NUM_EPOCHS + 1):

        print()
        print("=" * 60)
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )
        print("=" * 60)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )

        val_loss, val_dice, val_iou = validate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        print()
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Dice   : {val_dice:.4f}")
        print(f"Val IoU    : {val_iou:.4f}")

        # ----------------------------------------------------
        # Save latest checkpoint
        # ----------------------------------------------------
        if(DATA_AUGMENTATION):
            latest_path = os.path.join(CHECKPOINT_DIR,"latest_augmented.pt",)
        else:
            latest_path = os.path.join(CHECKPOINT_DIR,"latest.pt",)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
            },
            latest_path,
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_dice > best_dice:

            best_dice = val_dice
            if (DATA_AUGMENTATION):
                best_path = os.path.join(CHECKPOINT_DIR,"best_model_augmented.pt",)
            else:
                best_path = os.path.join(CHECKPOINT_DIR,"best_model.pt",)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                },
                best_path,
            )

            print(
                f"Saved new best model "
                f"(Dice = {best_dice:.4f})"
            )


if __name__ == "__main__":
    main()