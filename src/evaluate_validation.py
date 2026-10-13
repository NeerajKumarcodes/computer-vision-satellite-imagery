from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import WHUBuildingDataset
from unet import UNet


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VAL_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "val" / "image"
VAL_LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "val" / "label"

MODEL_PATH = PROJECT_ROOT / "models" / "unet_epoch_1.pth"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATCH_SIZE = 4


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ---------------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model.eval()


# ---------------------------------------------------------
# Validation Dice
# ---------------------------------------------------------

total_intersection = 0.0
total_prediction = 0.0
total_target = 0.0


with torch.no_grad():

    for batch_index, (images, masks) in enumerate(val_loader):

        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = (probabilities > 0.5).float()

        total_intersection += (
            predictions * masks
        ).sum().item()

        total_prediction += predictions.sum().item()

        total_target += masks.sum().item()

        if (batch_index + 1) % 100 == 0:
            print(
                f"Processed "
                f"{batch_index + 1}/{len(val_loader)} batches"
            )


# ---------------------------------------------------------
# Overall Dice
# ---------------------------------------------------------

dice = (
    (2.0 * total_intersection + 1e-7)
    / (
        total_prediction
        + total_target
        + 1e-7
    )
)


print("\nValidation Results")
print("-" * 30)
print("Validation images:", len(val_dataset))
print("Dice Score:", dice)