from pathlib import Path

from torch.utils.data import DataLoader

from dataset import WHUBuildingDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

image_dir = PROJECT_ROOT / "data" / "raw" / "WHU" / "train" / "image"
label_dir = PROJECT_ROOT / "data" / "raw" / "WHU" / "train" / "label"


dataset = WHUBuildingDataset(image_dir, label_dir)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)


images, masks = next(iter(dataloader))

print("Images shape:", images.shape)
print("Masks shape:", masks.shape)
print("Images dtype:", images.dtype)
print("Masks dtype:", masks.dtype)