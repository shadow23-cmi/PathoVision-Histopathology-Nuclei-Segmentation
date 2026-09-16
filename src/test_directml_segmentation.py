import torch
import torch_directml


def main():
    device = torch_directml.device(1)

    print("Using device:", device)

    # A realistic PathoVision batch:
    # batch size = 2
    # RGB image = 3 channels
    # image size = 256 x 256
    images = torch.randn(
        2,
        3,
        256,
        256,
        dtype=torch.float32,
        device=device,
    )

    # Binary segmentation masks
    masks = torch.randint(
        low=0,
        high=2,
        size=(2, 1, 256, 256),
        dtype=torch.float32,
        device=device,
    )

    print("Images:")
    print("  Shape:", images.shape)
    print("  Device:", images.device)
    print("  Dtype:", images.dtype)

    print("Masks:")
    print("  Shape:", masks.shape)
    print("  Device:", masks.device)
    print("  Dtype:", masks.dtype)

    # A simple operation resembling a model output
    logits = images.mean(dim=1, keepdim=True)

    # Binary cross-entropy requires logits and target
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        masks,
    )

    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())
    print("Loss device:", loss.device)

    print("DirectML segmentation-style test succeeded.")


if __name__ == "__main__":
    main()