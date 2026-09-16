import numpy as np
import torch
import random

from PIL import Image
from torch.utils.data import Dataset

from utils import MASK_DIR


class MoNuSegDataset(Dataset):
    """
    PyTorch Dataset for the processed MoNuSeg dataset.

    Images:
        data/processed/images/

    Masks:
        data/processed/masks/

    Image and mask filenames must match.
    """

    def __init__(self, image_paths, image_size=(256, 256),):
        self.image_paths = image_paths
        self.image_size = image_size

        # Verify that every image has a corresponding mask.
        for image_path in self.image_paths:

            image_path = image_path
            mask_path = MASK_DIR / image_path.name

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found:\n"
                    f"Image: {image_path}\n"
                    f"Expected mask: {mask_path}"
                )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        # --------------------------------------------------
        # Paths
        # --------------------------------------------------

        image_path = self.image_paths[index]
        mask_path = MASK_DIR / image_path.name

        # --------------------------------------------------
        # Load image and mask
        # --------------------------------------------------

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # --------------------------------------------------
        # Resize
        # --------------------------------------------------

        image = image.resize(
            self.image_size,
            Image.Resampling.BILINEAR,
        )

        # IMPORTANT:
        # NEAREST is used for masks because masks contain
        # discrete labels: background = 0, nucleus = 255.
        mask = mask.resize(
            self.image_size,
            Image.Resampling.NEAREST,
        )

        # --------------------------------------------------
        # Image → NumPy
        # --------------------------------------------------

        image = np.array(image,dtype=np.float32,)
        # H x W x C → C x H x W
        image = np.transpose(image,(2, 0, 1),)
        # [0, 255] → [0, 1]
        image /= 255.0

        # --------------------------------------------------
        # Mask → NumPy
        # --------------------------------------------------

        mask = np.array(mask,dtype=np.float32,)
        # 0/255 → 0/1
        mask /= 255.0
        # Ensure binary mask
        mask = (mask > 0.5).astype(np.float32)

        # H x W → 1 x H x W
        mask = np.expand_dims(mask,axis=0,)

        # --------------------------------------------------
        # NumPy → PyTorch
        # --------------------------------------------------

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask



class MoNuSegPatchDataset(Dataset):

    def __init__(self,image_paths,patch_size=256, stride=128, augment=False,):
        self.image_paths =  image_paths
        self.patch_size = patch_size
        self.stride = stride
        self.patch_index = []
        self.augment = augment

        for image_path in self.image_paths:
            mask_path = MASK_DIR / image_path.name

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found:\n"
                    f"Image: {image_path}\n"
                    f"Expected mask: {mask_path}"
                )

            with Image.open(image_path) as image:
                width, height = image.size

            if width < patch_size or height < patch_size:
                raise ValueError(
                    f"Image is smaller than patch size:\n"
                    f"Image: {image_path}\n"
                    f"Size: {(width, height)}"
                )

            y_positions = self._get_positions(length=height)
            x_positions = self._get_positions(length=width)

            for y in y_positions:
                for x in x_positions:
                    self.patch_index.append(
                        {
                            "image_path": image_path,
                            "x": x, "y": y,
                        }
                    )

    def _get_positions(self, length):
        positions = list(range(0,length - self.patch_size + 1,self.stride,))

        final_position = length - self.patch_size

        if positions[-1] != final_position:
            positions.append(final_position)

        return positions

    def apply_augmentation(self, image, mask):
        """
        Apply identical geometric transformations to
        the image and its corresponding mask.
        """

        if random.random() < 0.5:
            image = image.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

            mask = mask.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

        if random.random() < 0.5:
            image = image.transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )

            mask = mask.transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )

        rotation = random.randint(0, 3)

        if rotation > 0:
            angle = rotation * 90

            image = image.rotate(
                angle,
                expand=False,
            )

            mask = mask.rotate(
                angle,
                expand=False,
            )

        return image, mask

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, index):
        item = self.patch_index[index]

        image_path = item["image_path"]
        x = item["x"]
        y = item["y"]

        mask_path = MASK_DIR / image_path.name

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.crop(
                (
                    x, y,
                    x + self.patch_size, y + self.patch_size,
                )
            )

        with Image.open(mask_path) as mask:
            mask = mask.convert("L")
            mask = mask.crop(
                (
                    x,y,
                    x + self.patch_size,y + self.patch_size,
                )
            )

        if self.augment:
            image, mask = self.apply_augmentation(
                                                    image,
                                                    mask,
                                                )
        
        image = np.asarray(image,dtype=np.float32,)
        image = np.transpose(image,(2, 0, 1),)

        image /= 255.0

        mask = np.asarray(mask,dtype=np.float32,)

        mask /= 255.0
        mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask