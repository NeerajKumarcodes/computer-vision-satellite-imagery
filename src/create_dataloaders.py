from pathlib import Path

from torch.utils.data import DataLoader

from dataset import WHUBuildingDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

train_image_dir = PROJECT_ROOT / "data" / "raw" / "WHU" / "train" / "image"
train_label_dir = PROJECT_ROOT / "data" / "raw" / "WHU" / "train" / "label"

val_image_dir = PROJECT_ROOT / "data" / "raw" / "WHU" / "val" / "image"
val_label_dir = PROJECT_ROOT / "data" / "raw" / "WHU" / "val" / "label"


# ---------------------------------------------------------
# Datasets
# ---------------------------------------------------------

train_dataset = WHUBuildingDataset(
    train_image_dir,
    train_label_dir,
    augment=True,
)

val_dataset = WHUBuildingDataset(
    val_image_dir,
    val_label_dir,
    augment=False,
)


# ---------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)


# ---------------------------------------------------------
# Check
# ---------------------------------------------------------

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

train_images, train_masks = next(iter(train_loader))

val_images, val_masks = next(iter(val_loader))

print("\nTraining batch:")
print("Images:", train_images.shape)
print("Masks:", train_masks.shape)

print("\nValidation batch:")
print("Images:", val_images.shape)
print("Masks:", val_masks.shape)