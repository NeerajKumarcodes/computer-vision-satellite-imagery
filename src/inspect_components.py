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

test_dataset = WHUBuildingDataset(
    TEST_IMAGE_DIR,
    TEST_LABEL_DIR,
    augment=False,
)


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
# Load image
# --------------------------------------------------

image, _ = test_dataset[SAMPLE_INDEX]

input_tensor = image.unsqueeze(0).to(device)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

with torch.no_grad():

    logits = model(input_tensor)

    probabilities = torch.sigmoid(logits)

    prediction = (
        probabilities > THRESHOLD
    ).float()


# --------------------------------------------------
# Binary mask
# --------------------------------------------------

prediction_mask = (
    prediction
    .squeeze(0)
    .squeeze(0)
    .cpu()
    .numpy()
    .astype(np.uint8)
)

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


print("\nConnected Component Analysis")
print("-" * 40)

print("Total components:", num_labels - 1)

print("\nComponent areas:")

for component_id in range(1, num_labels):

    area = stats[
        component_id,
        cv2.CC_STAT_AREA,
    ]

    x = stats[
        component_id,
        cv2.CC_STAT_LEFT,
    ]

    y = stats[
        component_id,
        cv2.CC_STAT_TOP,
    ]

    width = stats[
        component_id,
        cv2.CC_STAT_WIDTH,
    ]

    height = stats[
        component_id,
        cv2.CC_STAT_HEIGHT,
    ]

    print(
        f"Component {component_id}: "
        f"area={area} pixels, "
        f"bbox=({x}, {y}, {width}, {height})"
    )


# --------------------------------------------------
# Create component visualization
# --------------------------------------------------

image_np = (
    image.permute(1, 2, 0)
    .numpy()
    * 255
)

image_np = image_np.astype(np.uint8)

visualization = cv2.cvtColor(
    image_np,
    cv2.COLOR_RGB2BGR,
)


# Generate different colors for components
rng = np.random.default_rng(42)

component_colors = rng.integers(
    0,
    256,
    size=(num_labels, 3),
    dtype=np.uint8,
)


for component_id in range(1, num_labels):

    component_mask = (
        labels == component_id
    )

    color = component_colors[
        component_id
    ]

    visualization[
        component_mask
    ] = color


# --------------------------------------------------
# Save visualization
# --------------------------------------------------

output_path = (
    OUTPUT_DIR
    / f"sample_{SAMPLE_INDEX}_components.png"
)

cv2.imwrite(
    str(output_path),
    visualization,
)


print("\nSaved:")
print(output_path)