import torch
import torch_directml

from losses import DiceLoss, BCEDiceLoss


def main():
    device = torch_directml.device(1)

    print("Using device:", device)

    logits = torch.randn(
        2,
        1,
        256,
        256,
        dtype=torch.float32,
        device=device,
    )

    targets = torch.randint(
        low=0,
        high=2,
        size=(2, 1, 256, 256),
        dtype=torch.float32,
        device=device,
    )

    dice_loss_fn = DiceLoss()
    combined_loss_fn = BCEDiceLoss()

    dice_loss = dice_loss_fn(logits, targets)
    combined_loss = combined_loss_fn(logits, targets)

    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)

    print("Dice loss:", dice_loss.item())
    print("Combined BCE + Dice loss:", combined_loss.item())

    print("Dice loss device:", dice_loss.device)
    print("Combined loss device:", combined_loss.device)

    assert torch.isfinite(dice_loss).item()
    assert torch.isfinite(combined_loss).item()

    print("Loss computation succeeded.")


if __name__ == "__main__":
    main()