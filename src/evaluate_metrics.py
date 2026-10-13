from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import WHUBuildingDataset
from unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VAL_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "val" / "image"
VAL_LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "val" / "label"

MODEL_PATH = PROJECT_ROOT / "models" / "unet_best.pth"

BATCH_SIZE = 4
THRESHOLD = 0.5


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# Validation dataset
val_dataset = WHUBuildingDataset(
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR,
    augment=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
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


# Global pixel counts
total_tp = 0.0
total_fp = 0.0
total_fn = 0.0


# Evaluation
with torch.no_grad():

    for batch_index, (images, masks) in enumerate(val_loader):

        images = images.to(device)
        masks = masks.to(device)

        # Model prediction
        logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = (probabilities > THRESHOLD).float()

        # True positives
        total_tp += (
            predictions * masks
        ).sum().item()

        # False positives
        total_fp += (
            predictions * (1 - masks)
        ).sum().item()

        # False negatives
        total_fn += (
            (1 - predictions) * masks
        ).sum().item()

        if (batch_index + 1) % 100 == 0:
            print(
                f"Processed "
                f"{batch_index + 1}/{len(val_loader)} batches"
            )


# Metrics
epsilon = 1e-7

dice = (
    2 * total_tp
    / (2 * total_tp + total_fp + total_fn + epsilon)
)

iou = (
    total_tp
    / (total_tp + total_fp + total_fn + epsilon)
)

precision = (
    total_tp
    / (total_tp + total_fp + epsilon)
)

recall = (
    total_tp
    / (total_tp + total_fn + epsilon)
)


# Results
print("\nValidation Metrics")
print("-" * 30)

print("Validation images:", len(val_dataset))

print(f"Dice Score: {dice:.6f}")
print(f"IoU:        {iou:.6f}")
print(f"Precision:  {precision:.6f}")
print(f"Recall:     {recall:.6f}")