from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dataset import WHUBuildingDataset
from unet import UNet

import numpy as np

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VAL_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "val" / "image"
VAL_LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "val" / "label"

MODEL_PATH = PROJECT_ROOT / "models" / "unet_epoch_1.pth"


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

val_dataset = WHUBuildingDataset(
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR,
    augment=False,
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
# Select validation image
# ---------------------------------------------------------

index = 10

image, mask = val_dataset[index]


# Add batch dimension
image_batch = image.unsqueeze(0).to(device)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

with torch.no_grad():

    logits = model(image_batch)

    probabilities = torch.sigmoid(logits)

    prediction = (probabilities > 0.5).float()


# Remove batch dimension
prediction = prediction.squeeze(0).cpu().squeeze(0).numpy()

image = image.permute(1, 2, 0).numpy()
mask = mask.squeeze(0).numpy()


# ---------------------------------------------------------
# Dice score
# ---------------------------------------------------------

intersection = (prediction * mask).sum()

dice = (
    (2.0 * intersection + 1e-7)
    / (
        prediction.sum()
        + mask.sum()
        + 1e-7
    )
)


print("Dice Score:", float(dice))

print("Ground truth unique values:", np.unique(mask))
print("Ground truth building pixels:", int(mask.sum()))

print("Prediction unique values:", np.unique(prediction))
print("Prediction building pixels:", int(prediction.sum()))


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6),
)


axes[0].imshow(image)
axes[0].set_title("Validation Image")
axes[0].axis("off")


axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Ground Truth")
axes[1].axis("off")


axes[2].imshow(prediction, cmap="gray")
axes[2].set_title("U-Net Prediction")
axes[2].axis("off")


plt.tight_layout()
plt.show()