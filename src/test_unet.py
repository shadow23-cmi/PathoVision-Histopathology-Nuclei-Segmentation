import torch
import torch_directml

from model import UNet


def main():
    device = torch_directml.device(1)

    print("Using device:", device)

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=16,
    ).to(device)

    model.eval()

    images = torch.randn(
        2,
        3,
        256,
        256,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        logits = model(images)

    print("Input shape:", images.shape)
    print("Output shape:", logits.shape)
    print("Output device:", logits.device)
    print("Output dtype:", logits.dtype)

    print("Minimum logit:", logits.min().item())
    print("Maximum logit:", logits.max().item())
    print("Mean logit:", logits.mean().item())

    assert logits.shape == (2, 1, 256, 256)

    print("U-Net DirectML forward pass succeeded.")


if __name__ == "__main__":
    main()