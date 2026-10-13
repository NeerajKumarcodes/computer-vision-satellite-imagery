# AI-Based Building Extraction from Aerial Imagery

**A Computer Vision and GeoAI project using U-Net semantic segmentation to identify building regions in high-resolution aerial imagery.**

This project explores how deep learning can support geospatial workflows by identifying buildings in aerial images and converting predicted building masks into vector geometries.

The project uses the **WHU Building Dataset**, PyTorch, and a U-Net segmentation architecture.

## Project Overview

Building footprint extraction is an important task in remote sensing and GIS. Accurate building information can support urban mapping, infrastructure planning, and geospatial database updates.

The objective of this project is to develop a deep-learning workflow that identifies building pixels in high-resolution aerial imagery and investigates their conversion into vector polygons.

### Current Progress

- Prepared and explored the WHU Building Dataset.
- Implemented a U-Net segmentation model using PyTorch.
- Trained and evaluated the model.
- Generated building segmentation predictions on unseen test images.
- Implemented an initial raster-to-vector polygonization workflow.
- Investigated challenges associated with merging adjacent buildings during polygon extraction.

## Visual Results

The following comparison shows an aerial image, its ground-truth building mask, and the corresponding U-Net prediction.

![Building segmentation comparison](data/predictions/linkedin_segmentation_comparison.png)

The model predicts building regions at the pixel level. Converting these predictions into separate, accurate building footprints remains an area of development.

## Dataset

This project uses the **WHU Building Dataset**, which contains high-resolution aerial imagery and corresponding building labels.

The cropped image tiles used in this project have the following characteristics:

- Image dimensions: 512 × 512 pixels
- Image format: TIFF
- Image channels: RGB
- Image data type: 8-bit unsigned integer
- Labels: Binary building masks
- Approximate spatial resolution: 0.3 metres per pixel

### Dataset Split

| Split | Number of image-label pairs |
|---|---:|
| Training | 4,736 |
| Validation | 1,036 |
| Test | 2,416 |
| **Total** | **8,188** |

The dataset is not included in this repository because of its size. Obtain it from the dataset's official distribution source and arrange the files as described below.

Expected directory structure:

```text
data/
└── raw/
    └── WHU/
        ├── train/
        │   ├── image/
        │   └── label/
        ├── val/
        │   ├── image/
        │   └── label/
        └── test/
            ├── image/
            └── label/
```

## Methodology

### 1. Data Preparation

- Read aerial imagery and labels using Rasterio.
- Convert RGB imagery into normalized tensors.
- Convert building labels into binary masks.
- Apply data augmentation during training.

### 2. Model Architecture

A U-Net convolutional neural network is used for semantic segmentation.

The architecture consists of an encoder that learns image features and a decoder that reconstructs spatially detailed segmentation maps. Skip connections help preserve spatial information between corresponding encoder and decoder stages.

**Framework:** PyTorch
**Model:** U-Net
**Task:** Binary semantic segmentation

### 3. Training

The model was trained using a combination of binary cross-entropy and Dice loss.

The training workflow includes training and validation datasets, GPU acceleration where available, and saving model checkpoints.

### 4. Evaluation

The model was evaluated on the held-out test set using:

- Dice coefficient
- Intersection over Union (IoU)
- Precision
- Recall

### 5. Polygon Extraction

Predicted binary masks are converted into polygon geometries using raster-to-vector processing.

An initial workflow has been implemented using Rasterio and Shapely. However, adjacent buildings can merge into connected regions, making it difficult to extract individual building footprints reliably.

Improving the separation and geometric accuracy of individual buildings is part of the ongoing work.

## Test Results

The following metrics were obtained from the held-out test set:

| Metric | Test score |
|---|---:|
| Dice coefficient | 0.9883 |
| Intersection over Union (IoU) | 0.9769 |
| Precision | 0.9856 |
| Recall | 0.9910 |

These metrics measure pixel-level segmentation performance. They do **not** establish the accuracy of individual vector building footprints.

## Technology Stack

- **Python** — Core implementation
- **PyTorch** — Deep learning and model training
- **Torchvision** — Computer vision utilities
- **NumPy** — Numerical computation
- **OpenCV** — Image processing
- **Rasterio** — Raster data handling and polygonization
- **Matplotlib** — Visualization
- **Pillow** — Image utilities
- **Scikit-learn** — Machine learning utilities

FastAPI and Uvicorn are included in the project dependencies for potential API development. A functional inference API is not yet documented as completed.

## Repository Structure

```text
Computer-Vision-Satellite-Imagery/
├── src/
│   ├── dataset.py
│   ├── unet.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_test.py
│   ├── evaluate_validation.py
│   ├── extract_polygons.py
│   ├── polygonize_prediction.py
│   ├── save_linkedin_visuals.py
│   └── visualize_predictions.py
├── data/
│   └── predictions/
│       ├── linkedin_segmentation_comparison.png
│       ├── linkedin_polygon_extraction.png
│       └── sample_132_buildings.geojson
├── models/
├── requirements.txt
├── .gitignore
└── README.md
```

This structure highlights the main project components. Additional dataset inspection, diagnostic, evaluation, and visualization scripts are also present in `src/`.

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Computer-Vision-Satellite-Imagery
```

Replace the placeholder with your repository URL after creating it.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build appropriate for your hardware if the default installation does not provide the required GPU support.

### 4. Prepare the dataset

Download the WHU Building Dataset from its official distribution source and arrange the image and label files using the directory structure described above.

### 5. Run the project

The scripts in `src/` cover dataset inspection, training, evaluation, visualization, and polygon extraction.

Review each script's configuration and expected inputs before running it. Training from scratch requires the dataset and can be computationally intensive.

The pretrained checkpoint is not included in this repository, so reproducing inference results requires training the model or obtaining the corresponding checkpoint separately.

## Current Limitations

- The source image tiles do not contain geographic georeferencing information.
- Polygon extraction may merge adjacent buildings into larger regions.
- Pixel-level segmentation metrics do not measure individual-building polygon accuracy.
- A complete, georeferenced building-footprint extraction workflow remains under development.

## Future Work

- Improve separation of adjacent buildings.
- Refine polygon geometry and remove unwanted regions.
- Investigate georeferenced output using suitable source imagery and spatial metadata.
- Develop an inference API.
- Explore integration with a WebGIS interface.

## Acknowledgements

The project uses the WHU Building Dataset for research and experimentation. Please consult the dataset's official documentation for its citation requirements and usage terms.

## Author

**Neeraj Kumar**

M.Tech Geoinformatics | Remote Sensing | GIS | Computer Vision | GeoAI

This is an independent learning and portfolio project exploring the intersection of geospatial technology and deep learning.