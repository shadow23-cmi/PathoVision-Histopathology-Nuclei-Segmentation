from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import MoNuSegPatchDataset
from .device import get_device
from .losses import BCEDiceLoss
from .metrics import dice_score, iou_score
from .model import UNet
from utils import VAL_FILE, read_paths


PATCH_SIZE = 256
STRIDE = 128
BATCH_SIZE = 4
BASE_CHANNELS = 32

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/best_model.pt"
)



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
        num_workers=0,
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

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    criterion = BCEDiceLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            loss = criterion(logits, masks)
            dice = dice_score(logits, masks)
            iou = iou_score(logits, masks)

            batch_size = images.shape[0]

            total_loss += loss.item() * batch_size
            total_dice += dice.item() * batch_size
            total_iou += iou.item() * batch_size
            total_samples += batch_size

    print()
    print("=" * 60)
    print("Validation Evaluation")
    print("=" * 60)
    print("Validation patches:", len(val_dataset))
    print("Average loss:", total_loss / total_samples)
    print("Average Dice:", total_dice / total_samples)
    print("Average IoU:", total_iou / total_samples)


if __name__ == "__main__":
    main()