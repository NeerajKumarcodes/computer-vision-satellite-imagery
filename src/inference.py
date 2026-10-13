from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine
from rasterio.features import shapes
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

try:
    from .unet import UNet
except ImportError:
    from unet import UNet


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "unet_corrected_best.pth"

TILE_SIZE = 512
DEFAULT_THRESHOLD = 0.40
DEFAULT_MIN_AREA = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

def load_model(model_path=DEFAULT_MODEL_PATH):
    """
    Load the existing U-Net checkpoint.

    The original checkpoint was trained using reversed mask
    semantics, so inference must invert sigmoid probabilities
    to obtain building probabilities.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    model = UNet().to(DEVICE)

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ---------------------------------------------------------
# Tile positioning
# ---------------------------------------------------------

def tile_positions(length, tile_size=TILE_SIZE):
    """
    Generate tile start positions, including the final edge.

    Small images are padded to one tile.
    Larger images use overlapping tiles where possible.
    """

    if length <= tile_size:
        return [0]

    stride = tile_size // 2

    positions = list(range(0, length - tile_size + 1, stride))

    final_position = length - tile_size

    if positions[-1] != final_position:
        positions.append(final_position)

    return positions


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

def prepare_rgb(image):
    """
    Convert an image array into RGB float32 [0, 1].

    Input shape: (height, width, channels)

    Supports common RGB/RGBA and grayscale imagery.
    Assumes that the first three raster bands are RGB when
    reading a conventional three-band raster.
    """

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)

    if image.ndim != 3:
        raise ValueError(
            "Image must have height, width, and channel dimensions."
        )

    channels = image.shape[2]

    if channels == 1:
        image = np.repeat(image, 3, axis=2)

    elif channels >= 3:
        image = image[:, :, :3]

    else:
        raise ValueError(
            f"Unsupported number of image channels: {channels}"
        )

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    else:
        image = image.astype(np.float32)

        if not np.isfinite(image).all():
            raise ValueError(
                "Image contains NaN or infinite values."
            )

        # Normalize common 8-bit-style floating-point imagery.
        if image.min() >= 0 and image.max() > 1.0:
            image = image / 255.0

        # Do not silently rescale arbitrary multispectral data.
        elif image.min() < 0 or image.max() > 255:
            raise ValueError(
                "Unsupported pixel range. "
                "Provide 8-bit RGB imagery or float RGB imagery "
                "with values between 0 and 1."
            )

    return np.clip(image, 0.0, 1.0)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict_buildings(
    image,
    model,
    threshold=DEFAULT_THRESHOLD,
    tile_size=TILE_SIZE,
):
    """
    Predict building probabilities for an RGB image.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image with shape (H, W, 3).
    model : UNet
        Loaded U-Net model.
    threshold : float
        Building probability threshold.
    tile_size : int
        Neural network tile size.

    Returns
    -------
    building_probabilities : numpy.ndarray
        Float32 array of shape (H, W).
    building_mask : numpy.ndarray
        Binary uint8 array of shape (H, W).
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    if tile_size != 512:
        raise ValueError(
            "This checkpoint was trained on 512x512 tiles. "
            "Use tile_size=512."
        )

    image = prepare_rgb(image)

    height, width, _ = image.shape

    probability_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    probability_count = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    y_positions = tile_positions(height, tile_size)
    x_positions = tile_positions(width, tile_size)

    with torch.inference_mode():

        for y in y_positions:
            for x in x_positions:

                tile = image[
                    y:min(y + tile_size, height),
                    x:min(x + tile_size, width),
                    :,
                ]

                tile_height, tile_width, _ = tile.shape

                # Pad smaller edge tiles to 512 x 512.
                padded = np.zeros(
                    (tile_size, tile_size, 3),
                    dtype=np.float32,
                )

                padded[:tile_height, :tile_width, :] = tile

                tensor = torch.from_numpy(
                    padded.transpose(2, 0, 1).copy()
                ).unsqueeze(0).to(DEVICE)

                logits = model(tensor)

                # Original checkpoint predicts reversed semantics.
                probabilities = torch.sigmoid(logits)

                probabilities = (
                    probabilities[0, 0]
                    .cpu()
                    .numpy()
                )

                probabilities = probabilities[
                    :tile_height,
                    :tile_width,
                ]

                probability_sum[
                    y:y + tile_height,
                    x:x + tile_width,
                ] += probabilities

                probability_count[
                    y:y + tile_height,
                    x:x + tile_width,
                ] += 1.0

    building_probabilities = (
        probability_sum / np.maximum(probability_count, 1.0)
    )

    building_mask = (
        building_probabilities >= threshold
    ).astype(np.uint8)

    return building_probabilities, building_mask


# ---------------------------------------------------------
# Polygon extraction
# ---------------------------------------------------------

def extract_building_polygons(
    building_mask,
    transform=Affine.identity(),
    min_area=DEFAULT_MIN_AREA,
    connectivity=8,
):
    """
    Convert a binary building mask into polygon footprints.

    For ordinary uploaded images without georeferencing,
    polygon coordinates are in image-pixel units.

    For georeferenced rasters, supply their affine transform
    to obtain polygons in the raster's coordinate system.

    Returns a list of Shapely polygons.
    """

    if building_mask.ndim != 2:
        raise ValueError(
            "Building mask must be a 2D array."
        )

    binary_mask = building_mask.astype(np.uint8)

    if min_area < 0:
        raise ValueError(
            "min_area cannot be negative."
        )

    # Remove small isolated components.
    if min_area > 0:

        component_labels, component_count = (
            ndimage.label(binary_mask)
        )

        if component_count:
            component_sizes = np.bincount(
                component_labels.ravel()
            )

            keep = component_sizes >= min_area
            keep[0] = False

            binary_mask = keep[
                component_labels
            ].astype(np.uint8)

    polygons = []

    for geometry, value in shapes(
        binary_mask,
        mask=binary_mask.astype(bool),
        transform=transform,
        connectivity=connectivity,
    ):

        if value != 1:
            continue

        polygon = shape(geometry)

        if polygon.is_empty:
            continue

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.is_empty:
            continue

        if polygon.area >= min_area:
            polygons.append(polygon)

    return polygons


# ---------------------------------------------------------
# Raster file processing
# ---------------------------------------------------------

def process_raster(
    raster_path,
    model,
    threshold=DEFAULT_THRESHOLD,
    min_area=DEFAULT_MIN_AREA,
):
    """
    Read a raster, run building inference, and extract polygons.

    Returns a dictionary containing the original image,
    probabilities, mask, polygons, and geospatial metadata.
    """

    raster_path = Path(raster_path)

    with rasterio.open(raster_path) as src:

        data = src.read()
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    if data.shape[0] == 1:
        image = data[0]

    else:
        image = np.moveaxis(data[:3], 0, -1)

    rgb_image = prepare_rgb(image)

    probabilities, mask = predict_buildings(
        rgb_image,
        model,
        threshold=threshold,
    )

    polygons = extract_building_polygons(
        mask,
        transform=transform,
        min_area=min_area,
    )

    return {
        "image": rgb_image,
        "probabilities": probabilities,
        "mask": mask,
        "polygons": polygons,
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
        "profile": profile,
        "source_path": raster_path,
    }