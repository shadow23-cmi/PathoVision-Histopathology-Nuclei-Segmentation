from pathlib import Path
import random

from utils import IMG_DIR, PROJECT_ROOT, TRAIN_FILE, VAL_FILE


SEED = 42
TRAIN_RATIO = 0.8


def main():

    random.seed(SEED)

    image_paths = sorted(
        IMG_DIR.glob("*.png")
    )

    if not image_paths:
        raise RuntimeError(
            f"No images found in {IMG_DIR}"
        )

    print("Image directory:", IMG_DIR)
    print("Total images:", len(image_paths))

    random.shuffle(image_paths)

    split_index = int(
        len(image_paths) * TRAIN_RATIO
    )

    train_paths = image_paths[:split_index]
    val_paths = image_paths[split_index:]

    with open(TRAIN_FILE, "w") as f:
        for path in train_paths:
            f.write(str(path.resolve()) + "\n")

    with open(VAL_FILE, "w") as f:
        for path in val_paths:
            f.write(str(path.resolve()) + "\n")

    print("Training images:", len(train_paths))
    print("Validation images:", len(val_paths))

    print("Train file:", TRAIN_FILE)
    print("Validation file:", VAL_FILE)


if __name__ == "__main__":
    main()