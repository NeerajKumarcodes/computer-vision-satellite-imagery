from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import WHUBuildingDataset
from losses import BCEDiceLoss
from unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "train" / "image"
TRAIN_LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "train" / "label"

VAL_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "val" / "image"
VAL_LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "val" / "label"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# Training configuration
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 5


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# Datasets
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


# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# Model
model = UNet().to(device)

criterion = BCEDiceLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# Track best validation Dice
best_val_dice = 0.0


# Training
for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 40)

    # -------------------------
    # Training
    # -------------------------

    model.train()

    running_train_loss = 0.0

    for batch_index, (images, masks) in enumerate(train_loader):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        running_train_loss += loss.item()

        if (batch_index + 1) % 100 == 0:
            print(
                f"Batch {batch_index + 1}/{len(train_loader)} "
                f"- Loss: {loss.item():.4f}"
            )

    average_train_loss = (
        running_train_loss / len(train_loader)
    )


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    running_val_loss = 0.0

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            running_val_loss += loss.item()

            # Convert logits to binary predictions
            probabilities = torch.sigmoid(outputs)

            predictions = (
                probabilities > 0.5
            ).float()

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


    average_val_loss = (
        running_val_loss / len(val_loader)
    )

    # Validation Dice
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

    # Validation IoU
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
    # Print epoch results
    # -------------------------

    print("\nEpoch Results")
    print("-" * 30)

    print(f"Training Loss:   {average_train_loss:.6f}")
    print(f"Validation Loss: {average_val_loss:.6f}")
    print(f"Validation Dice: {val_dice:.6f}")
    print(f"Validation IoU:  {val_iou:.6f}")


    # -------------------------
    # Save best model
    # -------------------------

    if val_dice > best_val_dice:

        best_val_dice = val_dice

        best_model_path = MODEL_DIR / "unet_best.pth"

        torch.save(
            model.state_dict(),
            best_model_path,
        )

        print("\n*** New best model saved! ***")
        print(f"Best Dice: {best_val_dice:.6f}")
        print(f"Saved to: {best_model_path}")


print("\nTraining complete.")
print(f"Best Validation Dice: {best_val_dice:.6f}")