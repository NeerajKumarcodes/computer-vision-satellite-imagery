
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import WHUBuildingDataset
from unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_IMAGE_DIR = (
    PROJECT_ROOT / "data" / "raw" / "WHU" / "test" / "image"
)

TEST_LABEL_DIR = (
    PROJECT_ROOT / "data" / "raw" / "WHU" / "test" / "label"
)

MODEL_PATH = PROJECT_ROOT / "models" / "unet_best.pth"

OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"

SAMPLE_INDEX = 10
THRESHOLD = 0.5


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = WHUBuildingDataset(
        TEST_IMAGE_DIR,
        TEST_LABEL_DIR,
        augment=False,
    )

    image, mask = dataset[SAMPLE_INDEX]

    model = UNet().to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        probabilities = torch.sigmoid(logits)
        prediction = (
            probabilities[0, 0].cpu().numpy() >= THRESHOLD
        ).astype(np.uint8)

    # Convert image from CHW to HWC for plotting.
    original = image.permute(1, 2, 0).cpu().numpy()
    ground_truth = mask.squeeze(0).cpu().numpy()

    # Calculate Dice for this sample.
    intersection = (prediction * ground_truth).sum()
    dice = (
        2 * intersection
        / (prediction.sum() + ground_truth.sum() + 1e-8)
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(np.clip(original, 0, 1))
    axes[0].set_title("Original Aerial Image")

    axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground-Truth Mask")

    axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"U-Net Prediction | Dice: {dice:.4f}")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        "AI-Based Building Segmentation | WHU Building Dataset",
        fontsize=15,
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "linkedin_segmentation_comparison.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("Saved visual:", output_path)
    print(f"Sample index: {SAMPLE_INDEX}")
    print(f"Dice score: {dice:.4f}")


if __name__ == "__main__":
    main()
