from pathlib import Path


# necessary paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "MoNuSeg 2018 Training Data"
IMG_DATA_DIR = RAW_DATA_DIR / "Tissue Images"
XML_DATA_DIR = RAW_DATA_DIR / "Annotations"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

IMG_DIR = PROCESSED_DIR / "images"
MASK_DIR = PROCESSED_DIR / "masks"

# Dataset split files
TRAIN_FILE = PROCESSED_DIR / "train.txt"
VAL_FILE = PROCESSED_DIR / "val.txt"

# Outputs
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


def read_paths(file_path):

    with open(file_path, "r") as f:
        return [
            Path(line.strip())
            for line in f
            if line.strip()
        ]