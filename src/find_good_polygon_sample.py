from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

from dataset import WHUBuildingDataset
from unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "test" / "image"
TEST_LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "WHU" / "test" / "label"

MODEL_PATH = PROJECT_ROOT / "models" / "unet_best.pth"

BATCH_SIZE = 4
THRESHOLD = 0.5
MIN_COMPONENT_AREA = 100


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

test_dataset = WHUBuildingDataset(
    TEST_IMAGE_DIR,
    TEST_LABEL_DIR,
    augment=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model.eval()


# --------------------------------------------------
# Search
# --------------------------------------------------

candidates = []

with torch.no_grad():

    for batch_index, (images, masks) in enumerate(test_loader):

        images = images.to(device)

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities > THRESHOLD
        ).cpu().numpy().astype("uint8")

        for batch_item in range(len(images)):

            sample_index = (
                batch_index * BATCH_SIZE
                + batch_item
            )

            mask = predictions[batch_item, 0]

            binary_mask = mask * 255

            num_labels, labels, stats, centroids = (
                cv2.connectedComponentsWithStats(
                    binary_mask,
                    connectivity=8,
                )
            )

            component_areas = []

            for component_id in range(1, num_labels):

                area = stats[
                    component_id,
                    cv2.CC_STAT_AREA,
                ]

                if area >= MIN_COMPONENT_AREA:
                    component_areas.append(area)

            # We now only require 2 meaningful regions.
            if len(component_areas) >= 2:

                total_pixels = mask.sum()

                largest = max(component_areas)

                largest_ratio = (
                    largest / total_pixels
                    if total_pixels > 0
                    else 1.0
                )

                candidates.append(
                    (
                        sample_index,
                        len(component_areas),
                        largest_ratio,
                        total_pixels,
                        sorted(
                            component_areas,
                            reverse=True,
                        ),
                    )
                )

            # Stop after finding 10 candidates.
            if len(candidates) >= 10:
                break

        if len(candidates) >= 10:
            break


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nCandidate Samples")
print("-" * 70)

if not candidates:

    print("No suitable sample found.")

else:

    for candidate in candidates:

        (
            index,
            components,
            largest_ratio,
            pixels,
            areas,
        ) = candidate

        print(
            f"Index: {index:4d} | "
            f"Components: {components:2d} | "
            f"Largest ratio: {largest_ratio:.3f} | "
            f"Building pixels: {pixels}"
        )

        print(
            f"  Component areas: {areas}"
        )