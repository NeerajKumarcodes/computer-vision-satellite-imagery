from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import WHUBuildingDataset
from unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VAL_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "val" / "image"
VAL_LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "val" / "label"

MODEL_PATH = PROJECT_ROOT / "models" / "unet_best.pth"

THRESHOLD = 0.5


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# Dataset
val_dataset = WHUBuildingDataset(
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR,
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


# Samples to inspect
sample_indices = [10, 25, 50, 100]


for index in sample_indices:

    image, mask = val_dataset[index]

    # Add batch dimension
    input_tensor = image.unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        logits = model(input_tensor)

        probabilities = torch.sigmoid(logits)

        prediction = (
            probabilities > THRESHOLD
        ).float()

    # Convert to NumPy
    image_np = image.permute(1, 2, 0).numpy()

    mask_np = mask.squeeze(0).numpy()

    prediction_np = (
        prediction.squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # Dice score for this image
    intersection = (
        prediction_np * mask_np
    ).sum()

    dice = (
        2 * intersection
        / (
            prediction_np.sum()
            + mask_np.sum()
            + 1e-7
        )
    )

    # Display
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask_np)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(prediction_np)
    axes[2].set_title(
        f"Prediction\nDice: {dice:.4f}"
    )
    axes[2].axis("off")

    plt.suptitle(
        f"Validation Sample {index}"
    )

    plt.tight_layout()
    plt.show()

    print(f"\nSample {index}")
    print("-" * 30)
    print(f"Dice: {dice:.6f}")
    print(
        "Ground truth building pixels:",
        int(mask_np.sum()),
    )
    print(
        "Predicted building pixels:",
        int(prediction_np.sum()),
    )