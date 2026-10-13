from pathlib import Path

import torch

from dataset import WHUBuildingDataset
from unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "test" / "image"
TEST_LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "test" / "label"

MODEL_PATH = PROJECT_ROOT / "models" / "unet_best.pth"

SAMPLE_INDEX = 50
THRESHOLD = 0.5


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# Dataset
dataset = WHUBuildingDataset(
    TEST_IMAGE_DIR,
    TEST_LABEL_DIR,
    augment=False,
)


# Model
model = UNet().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model.eval()


# Load sample
image, mask = dataset[SAMPLE_INDEX]

input_tensor = image.unsqueeze(0).to(device)
target = mask.unsqueeze(0).to(device)


# Prediction
with torch.no_grad():

    logits = model(input_tensor)

    probabilities = torch.sigmoid(logits)

    prediction = (
        probabilities > THRESHOLD
    ).float()


# Pixel counts
prediction_pixels = prediction.sum().item()
target_pixels = target.sum().item()

intersection = (
    prediction * target
).sum().item()

false_positive = (
    prediction * (1 - target)
).sum().item()

false_negative = (
    (1 - prediction) * target
).sum().item()


# Metrics
dice = (
    2 * intersection
    / (
        prediction_pixels
        + target_pixels
        + 1e-7
    )
)

iou = (
    intersection
    / (
        prediction_pixels
        + target_pixels
        - intersection
        + 1e-7
    )
)


print("\nSample 50 Diagnostic")
print("-" * 35)

print("Image shape:", image.shape)
print("Ground truth pixels:", int(target_pixels))
print("Prediction pixels:", int(prediction_pixels))
print("Intersection:", int(intersection))
print("False positives:", int(false_positive))
print("False negatives:", int(false_negative))

print(f"Dice: {dice:.6f}")
print(f"IoU:  {iou:.6f}")

print(
    "\nProbability range:"
)

print(
    "Min:",
    probabilities.min().item()
)

print(
    "Max:",
    probabilities.max().item()
)

print(
    "Mean:",
    probabilities.mean().item()
)