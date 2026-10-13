from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import WHUBuildingDataset
from losses import BCEDiceLoss
from unet import UNet


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_IMAGE_DIR = PROJECT_ROOT / "data/raw/WHU/train/image"
TRAIN_LABEL_DIR = PROJECT_ROOT / "data/raw/WHU/train/label"

VAL_IMAGE_DIR = PROJECT_ROOT / "data/raw/WHU/val/image"
VAL_LABEL_DIR = PROJECT_ROOT / "data/raw/WHU/val/label"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "unet_corrected_best.pth"

BATCH_SIZE = 2
LEARNING_RATE = 1e-4
MAX_EPOCHS = 10
PATIENCE = 3
NUM_WORKERS = 0

SEED = 42


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
# Datasets
# --------------------------------------------------

train_dataset = WHUBuildingDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    augment=True,
)

val_dataset = WHUBuildingDataset(
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR,
    augment=False,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", MAX_EPOCHS)


# --------------------------------------------------
# Verify corrected labels
# --------------------------------------------------

sample_image, sample_mask = train_dataset[0]

print("\nDataset verification")
print("Image shape:", tuple(sample_image.shape))
print("Mask shape:", tuple(sample_mask.shape))
print(
    "Mask values:",
    torch.unique(sample_mask).cpu().numpy(),
)
print(
    "Building pixel percentage:",
    f"{sample_mask.float().mean().item() * 100:.2f}%",
)

if not torch.all(
    (sample_mask == 0) | (sample_mask == 1)
):
    raise ValueError(
        "Expected binary masks containing only 0 and 1."
    )


# --------------------------------------------------
# Model and loss
# --------------------------------------------------

model = UNet().to(device)

criterion = BCEDiceLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_val_dice = -1.0
epochs_without_improvement = 0


# --------------------------------------------------
# Training loop
# --------------------------------------------------

for epoch in range(MAX_EPOCHS):

    print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}")
    print("-" * 50)

    # -------------------------
    # Training
    # -------------------------

    model.train()

    train_loss_sum = 0.0

    for batch_index, (images, masks) in enumerate(
        train_loader
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss = criterion(logits, masks)

        loss.backward()

        optimizer.step()

        train_loss_sum += loss.item()

        if (batch_index + 1) % 100 == 0:
            print(
                f"Training batch "
                f"{batch_index + 1}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    average_train_loss = (
        train_loss_sum / len(train_loader)
    )

    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_loss_sum = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.inference_mode():

        for images, masks in val_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            masks = masks.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            loss = criterion(logits, masks)

            val_loss_sum += loss.item()

            probabilities = torch.sigmoid(logits)

            # Corrected targets: 1 = building.
            predictions = probabilities >= 0.5
            ground_truth = masks >= 0.5

            total_tp += (
                predictions & ground_truth
            ).sum().item()

            total_fp += (
                predictions & ~ground_truth
            ).sum().item()

            total_fn += (
                ~predictions & ground_truth
            ).sum().item()

    average_val_loss = (
        val_loss_sum / len(val_loader)
    )

    epsilon = 1e-7

    val_dice = (
        2 * total_tp
        / (
            2 * total_tp
            + total_fp
            + total_fn
            + epsilon
        )
    )

    val_iou = (
        total_tp
        / (
            total_tp
            + total_fp
            + total_fn
            + epsilon
        )
    )

    # -------------------------
    # Results
    # -------------------------

    print("\nEpoch results")
    print("-" * 30)

    print(f"Training loss:   {average_train_loss:.6f}")
    print(f"Validation loss: {average_val_loss:.6f}")
    print(f"Building Dice:   {val_dice:.6f}")
    print(f"Building IoU:    {val_iou:.6f}")

    # -------------------------
    # Save best checkpoint
    # -------------------------

    if val_dice > best_val_dice:

        best_val_dice = val_dice
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )

        print("\nNew best corrected model saved.")
        print("Path:", MODEL_PATH)
        print(f"Best validation Dice: {best_val_dice:.6f}")

    else:

        epochs_without_improvement += 1

        print(
            "No validation Dice improvement for "
            f"{epochs_without_improvement} epoch(s)."
        )

    # -------------------------
    # Early stopping
    # -------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping: validation Dice "
            "has not improved."
        )

        break

    # -------------------------
    # Thermal check reminder
    # -------------------------

    if device.type == "cuda":
        torch.cuda.empty_cache()


print("\nTraining finished.")
print(f"Best validation Dice: {best_val_dice:.6f}")
print("Best checkpoint:", MODEL_PATH)