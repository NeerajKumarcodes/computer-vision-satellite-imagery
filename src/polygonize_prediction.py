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

    prediction = (
        probabilities[0, 0]
        .cpu()
        .numpy()
        > THRESHOLD
    ).astype(np.uint8)


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