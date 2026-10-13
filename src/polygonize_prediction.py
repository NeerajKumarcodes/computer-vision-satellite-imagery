from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd

from dataset import WHUBuildingDataset
from unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "test" / "image"
TEST_LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "test" / "label"

MODEL_PATH = PROJECT_ROOT / "models" / "unet_best.pth"

OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"

SAMPLE_INDEX = 132
THRESHOLD = 0.5
MIN_AREA = 100


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = WHUBuildingDataset(
    TEST_IMAGE_DIR,
    TEST_LABEL_DIR,
    augment=False,
)

image, mask = dataset[SAMPLE_INDEX]

# --------------------------------------------------
# Ground-truth mask diagnostics
# --------------------------------------------------

gt_mask = mask.squeeze().cpu().numpy()

# Convert ground truth to binary values
gt_binary = (gt_mask > 0.5).astype(np.uint8)

print("\nGround Truth Diagnostics")
print("-" * 40)
print("Ground truth shape:", gt_binary.shape)
print("Ground truth unique values:", np.unique(gt_binary))

gt_foreground = int(gt_binary.sum())
gt_percentage = (gt_foreground / gt_binary.size) * 100

print("Ground truth foreground pixels:", gt_foreground)
print(f"Ground truth foreground percentage: {gt_percentage:.2f}%")

# Save ground-truth mask
import cv2

gt_path = OUTPUT_DIR / f"sample_{SAMPLE_INDEX}_ground_truth.png"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(gt_path), gt_binary * 255)

print("Saved ground truth:", gt_path)

image_batch = image.unsqueeze(0).to(device)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model.eval()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

with torch.no_grad():

    logits = model(image_batch)

    probabilities = torch.sigmoid(logits)

    # Invert the prediction to identify building pixels.
    # 1 = buildings, 0 = background.
    prediction = (
        probabilities[0, 0]
        .cpu()
        .numpy()
        <= THRESHOLD
    ).astype(np.uint8)

# --------------------------------------------------
# Diagnose predicted mask
# --------------------------------------------------

foreground_pixels = int(prediction.sum())
total_pixels = int(prediction.size)
foreground_percentage = (
    foreground_pixels / total_pixels
) * 100

print("\nPrediction Mask Diagnostics")
print("-" * 40)
print("Image shape:", prediction.shape)
print("Foreground pixels:", foreground_pixels)
print("Total pixels:", total_pixels)
print(f"Foreground percentage: {foreground_percentage:.2f}%")
print("Unique values:", np.unique(prediction))

# Save the binary prediction for inspection
mask_path = (
    OUTPUT_DIR
    / f"sample_{SAMPLE_INDEX}_binary_prediction.png"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import cv2

cv2.imwrite(
    str(mask_path),
    prediction * 255,
)

print("Saved binary mask:", mask_path)


# --------------------------------------------------
# Polygonization
# --------------------------------------------------

print("\nPolygonization")
print("-" * 40)

polygons = []

for geometry, value in shapes(
    prediction,
    mask=prediction.astype(bool),
):

    if value != 1:
        continue

    polygon = shape(geometry)

    if polygon.area >= MIN_AREA:
        polygons.append(polygon)


print("Sample index:", SAMPLE_INDEX)
print("Detected polygons:", len(polygons))


# --------------------------------------------------
# Save GeoJSON
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

output_path = (
    OUTPUT_DIR
    / f"sample_{SAMPLE_INDEX}_buildings.geojson"
)


gdf = gpd.GeoDataFrame(
    {
        "id": range(1, len(polygons) + 1),
        "area_px": [
            polygon.area
            for polygon in polygons
        ],
    },
    geometry=polygons,
    crs=None,
)

gdf.to_file(
    output_path,
    driver="GeoJSON",
)


print("Saved:", output_path)