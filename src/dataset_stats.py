from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "WHU"


for split in ["train", "val", "test"]:
    image_dir = DATASET_ROOT / split / "image"
    label_dir = DATASET_ROOT / split / "label"

    image_files = sorted(image_dir.glob("*.tif"))
    label_files = sorted(label_dir.glob("*.tif"))

    image_names = {file.name for file in image_files}
    label_names = {file.name for file in label_files}

    matching_names = image_names & label_names

    print(f"\n{split.upper()}")
    print("-" * len(split))

    print("Images:", len(image_files))
    print("Labels:", len(label_files))
    print("Matching pairs:", len(matching_names))

    print("Images without labels:", len(image_names - label_names))
    print("Labels without images:", len(label_names - image_names))