from pathlib import Path

from dataset import WHUBuildingDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

image_dir = PROJECT_ROOT / "data" / "raw" / "WHU" / "train" / "image"
label_dir = PROJECT_ROOT / "data" / "raw" / "WHU" / "train" / "label"


dataset = WHUBuildingDataset(
    image_dir,
    label_dir,
    augment=True,
)

print("Dataset size:", len(dataset))

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("Image min:", image.min().item())
print("Image max:", image.max().item())

print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)
print("Mask unique values:", mask.unique())