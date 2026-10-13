from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parent.parent

image_dir = PROJECT_ROOT / "data" / "raw" / "WHU" / "train" / "image"
label_dir = PROJECT_ROOT / "data" / "raw" / "WHU" / "train" / "label"


image_path = image_dir / "100.tif"
label_path = label_dir / "100.tif"


with rasterio.open(image_path) as src:
    image = src.read()

with rasterio.open(label_path) as src:
    mask = src.read(1)


# Rasterio gives (bands, height, width)
# Matplotlib expects (height, width, bands)
image = np.transpose(image, (1, 2, 0))


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image)
axes[0].set_title("WHU Image")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Building Mask")
axes[1].axis("off")

plt.tight_layout()
plt.show()