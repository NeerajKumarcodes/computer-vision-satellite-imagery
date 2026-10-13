from pathlib import Path

import rasterio


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

image_path = PROJECT_ROOT / "data" / "raw" / "WHU" / "train" / "image" / "0.tif"
label_path = PROJECT_ROOT / "data" / "raw" / "WHU" / "train" / "label" / "0.tif"


# Inspect image
with rasterio.open(image_path) as src:
    print("IMAGE")
    print("------")
    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)
    print("Data type:", src.dtypes)
    print("CRS:", src.crs)
    print("Transform:", src.transform)

    image = src.read()

    print("Array shape:", image.shape)
    print("Minimum:", image.min())
    print("Maximum:", image.max())

# Inspect several labels
print("\nLABEL INSPECTION")
print("----------------")

label_dir = PROJECT_ROOT / "data" / "raw" / "WHU" / "train" / "label"

label_files = sorted(label_dir.glob("*.tif"))[:10]

for label_path in label_files:
    with rasterio.open(label_path) as src:
        label = src.read(1)

        print(
            label_path.name,
            "| shape:", label.shape,
            "| dtype:", label.dtype,
            "| unique values:", set(label.flatten()),
            "| min:", label.min(),
            "| max:", label.max(),
        )