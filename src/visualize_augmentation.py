from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from dataset import WHUBuildingDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

image_dir = PROJECT_ROOT / "data" / "raw" / "WHU" / "train" / "image"
label_dir = PROJECT_ROOT / "data" / "raw" / "WHU" / "train" / "label"


dataset = WHUBuildingDataset(
    image_dir,
    label_dir,
    augment=True,
)


image, mask = dataset[100]


# Convert image from (C, H, W) to (H, W, C)
image = image.permute(1, 2, 0).numpy()

# Convert mask from (1, H, W) to (H, W)
mask = mask.squeeze(0).numpy()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image)
axes[0].set_title("Augmented Image")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Augmented Building Mask")
axes[1].axis("off")

plt.tight_layout()
plt.show()