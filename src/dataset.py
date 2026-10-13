from pathlib import Path
import random

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class WHUBuildingDataset(Dataset):
    def __init__(self, image_dir, label_dir, augment=False):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.augment = augment

        self.image_files = sorted(self.image_dir.glob("*.tif"))

        if not self.image_files:
            raise RuntimeError(f"No TIFF images found in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_dir / image_path.name

        if not label_path.exists():
            raise FileNotFoundError(
                f"Label not found for image: {image_path.name}"
            )

        # Read RGB image
        with rasterio.open(image_path) as src:
            image = src.read()

        # Read building mask
        with rasterio.open(label_path) as src:
            mask = src.read(1)

        if image.shape[1:] != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {image_path.name}: "
                f"image={image.shape}, mask={mask.shape}"
            )

        # Convert image from uint8 [0, 255] to float32 [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert mask to binary values
        mask = (mask > 0).astype(np.float32)

        # ---------------------------------------------------------
        # Data augmentation
        # ---------------------------------------------------------

        if self.augment:

            # Random horizontal flip
            if random.random() < 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()

            # Random vertical flip
            if random.random() < 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()

            # Random 90-degree rotation
            k = random.randint(0, 3)

            if k > 0:
                image = np.rot90(image, k=k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

        # Convert NumPy arrays to PyTorch tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask