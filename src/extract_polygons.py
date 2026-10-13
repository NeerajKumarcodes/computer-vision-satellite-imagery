from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import WHUBuildingDataset
from unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "test" / "image"
TEST_LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "test" / "label"

MODEL_PATH = PROJECT_ROOT / "models" / "unet_best.pth"

OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.5

SAMPLE_INDEX = 50

# Ignore extremely small regions
MIN_AREA = 50


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Dataset
# --------------------------------------------------

test_dataset = WHUBuildingDataset(
    TEST_IMAGE_DIR,
    TEST_LABEL_DIR,
    augment=False,
)


# --------------------------------------------------
# Load model
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
# Load image
# --------------------------------------------------

image, ground_truth = test_dataset[SAMPLE_INDEX]

input_tensor = image.unsqueeze(0).to(device)


# --------------------------------------------------
# Model inference
# --------------------------------------------------

with torch.no_grad():

    logits = model(input_tensor)

    probabilities = torch.sigmoid(logits)

    prediction = (
        probabilities > THRESHOLD
    ).float()


# --------------------------------------------------
# Convert prediction to NumPy
# --------------------------------------------------

prediction_mask = (
    prediction
    .squeeze(0)
    .squeeze(0)
    .cpu()
    .numpy()
    .astype(np.uint8)
)


# Convert to 0-255
binary_mask = prediction_mask * 255


# --------------------------------------------------
# Connected components
# --------------------------------------------------

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )
)


print("\nBuilding Extraction")
print("-" * 30)

print("Image index:", SAMPLE_INDEX)
print("Connected components:", num_labels - 1)


# --------------------------------------------------
# Prepare visualization
# --------------------------------------------------

visualization = (
    image.permute(1, 2, 0)
    .numpy()
    * 255
)

visualization = visualization.astype(np.uint8)

visualization = cv2.cvtColor(
    visualization,
    cv2.COLOR_RGB2BGR,
)


# --------------------------------------------------
# Extract each component separately
# --------------------------------------------------

building_count = 0

for component_id in range(1, num_labels):

    area = stats[
        component_id,
        cv2.CC_STAT_AREA,
    ]

    # Ignore tiny regions
    if area < MIN_AREA:
        continue

    # Create mask containing only this component
    component_mask = np.zeros_like(
        binary_mask
    )

    component_mask[
        labels == component_id
    ] = 255

    # Find contour for this individual component
    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        continue

    # Largest contour of this component
    contour = max(
        contours,
        key=cv2.contourArea,
    )

    # Draw contour
    cv2.drawContours(
        visualization,
        [contour],
        -1,
        (0, 255, 0),
        2,
    )

    building_count += 1


print("Extracted building regions:", building_count)


# --------------------------------------------------
# Save prediction mask
# --------------------------------------------------

mask_path = (
    OUTPUT_DIR
    / f"sample_{SAMPLE_INDEX}_prediction_mask.png"
)

cv2.imwrite(
    str(mask_path),
    binary_mask,
)


# --------------------------------------------------
# Save contour visualization
# --------------------------------------------------

visualization_path = (
    OUTPUT_DIR
    / f"sample_{SAMPLE_INDEX}_building_contours.png"
)

cv2.imwrite(
    str(visualization_path),
    visualization,
)


print("\nSaved:")
print(mask_path)
print(visualization_path)