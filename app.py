from pathlib import Path
import json
import tempfile

import numpy as np
import rasterio
import streamlit as st
import torch

from PIL import Image
from affine import Affine
from rasterio.features import shapes
from rasterio.transform import array_bounds
from shapely.geometry import shape, mapping
from shapely.affinity import affine_transform

from src.inference import (
    load_model,
    predict_buildings,
    extract_building_polygons,
    DEFAULT_THRESHOLD,
    DEFAULT_MIN_AREA,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="GeoAI Building Extractor",
    page_icon="🏙️",
    layout="wide",
)

# ============================================================
# MODEL LOADING
# ============================================================


@st.cache_resource
def get_model():
    return load_model()


# ============================================================
# IMAGE PROCESSING
# ============================================================


def normalize_raster_bands(data):
    """
    Convert the first three raster bands into RGB values
    in the [0, 1] range.

    For non-8-bit data, apply a percentile stretch per band.
    This is a visualization-oriented approximation, not a
    substitute for sensor-specific reflectance calibration.
    """

    if data.ndim != 3:
        raise ValueError("Expected raster data with band, height and width.")

    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    elif data.shape[0] >= 3:
        data = data[:3]
    else:
        raise ValueError("The raster must contain one or at least three bands.")

    rgb = np.moveaxis(data, 0, -1).astype(np.float32)

    if not np.isfinite(rgb).all():
        raise ValueError(
            "The raster contains NaN or infinite values. "
            "Please upload a valid RGB image."
        )

    # Match the 8-bit normalization used during model training.
    if data.dtype == np.uint8:
        return rgb / 255.0

    # Stretch other numeric formats into the expected range.
    output = np.zeros_like(rgb, dtype=np.float32)

    for channel in range(3):
        band = rgb[:, :, channel]
        low, high = np.percentile(band, [2, 98])

        if high > low:
            output[:, :, channel] = np.clip(
                (band - low) / (high - low),
                0,
                1,
            )

    return output


def read_uploaded_image(uploaded_file):
    """
    Read an uploaded image.

    Returns:
        rgb_image: normalized RGB image
        transform: raster affine transform or None
        crs: coordinate reference system or None
        profile: original raster profile or None
        is_geotiff: whether the input is a GeoTIFF
    """

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix in {".tif", ".tiff"}:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_file.write(uploaded_file.getvalue())
            temporary_path = Path(temporary_file.name)

        try:
            with rasterio.open(temporary_path) as src:
                data = src.read()
                transform = src.transform
                crs = src.crs
                profile = src.profile.copy()

            rgb_image = normalize_raster_bands(data)

            return rgb_image, transform, crs, profile, True

        finally:
            temporary_path.unlink(missing_ok=True)

    # Ordinary image: no geographic coordinates are available.
    pil_image = Image.open(uploaded_file).convert("RGB")
    rgb_image = np.asarray(pil_image, dtype=np.float32) / 255.0

    return rgb_image, None, None, None, False


def make_overlay(rgb_image, building_mask):
    """
    Highlight predicted building pixels in red.
    """

    overlay = np.clip(rgb_image.copy(), 0, 1)

    building_pixels = building_mask.astype(bool)

    # Blend the original image with a red building highlight.
    overlay[building_pixels] = (
        0.45 * overlay[building_pixels]
        + 0.55 * np.array([1.0, 0.0, 0.0])
    )

    return np.clip(overlay, 0, 1)


def convert_pixel_polygon_to_map(polygon, transform):
    """
    Transform a polygon from pixel coordinates into map coordinates.
    """

    # Shapely affine order:
    # x' = a*x + b*y + xoff
    # y' = d*x + e*y + yoff
    return affine_transform(
        polygon,
        [
            transform.a,
            transform.b,
            transform.d,
            transform.e,
            transform.c,
            transform.f,
        ],
    )


def create_geojson(polygons, transform=None, crs=None):
    """
    Create a standards-compliant GeoJSON FeatureCollection.

    Polygons are initially in pixel coordinates. The raster affine
    transform converts them to source map coordinates. If the source
    CRS is known, the coordinates are then reprojected to EPSG:4326.
    """

    from rasterio.warp import transform_geom

    features = []

    for index, polygon in enumerate(polygons, start=1):
        output_polygon = polygon

        if transform is not None:
            output_polygon = convert_pixel_polygon_to_map(
                output_polygon,
                transform,
            )

        geometry = mapping(output_polygon)

        if crs is not None and transform is not None:
            geometry = transform_geom(
                crs,
                "EPSG:4326",
                geometry,
                precision=8,
            )

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "building_id": index,
                    "area_pixels": float(polygon.area),
                },
                "geometry": geometry,
            }
        )

    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": features,
        },
        indent=2,
    )


# ============================================================
# APPLICATION INTERFACE
# ============================================================

st.title("🏙️ GeoAI Building Footprint Extractor")

st.markdown(
    """
    Upload aerial imagery and use a trained **U-Net segmentation model**
    to detect building regions, visualize predictions, and export
    building footprints.

    **Model:** PyTorch U-Net · **Task:** Building segmentation
    """
)

st.info(
    "Best suited to RGB aerial imagery resembling the WHU Building Dataset. "
    "Results on satellite imagery, different resolutions, or unfamiliar "
    "sensors may require additional training or calibration."
)

with st.sidebar:
    st.header("Inference Settings")

    threshold = st.slider(
        "Building probability threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(DEFAULT_THRESHOLD),
        step=0.05,
        help="Higher thresholds generally reduce detections but may miss buildings.",
    )

    min_area = st.slider(
        "Minimum building size (pixels)",
        min_value=0,
        max_value=500,
        value=int(DEFAULT_MIN_AREA),
        step=5,
        help="Removes small predicted regions based on their pixel area.",
    )

    st.divider()

    device_name = (
        "NVIDIA GPU"
        if torch.cuda.is_available()
        else "CPU"
    )

    st.caption(f"Inference device: {device_name}")

uploaded_file = st.file_uploader(
    "Upload aerial imagery",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    help="Use an RGB image or a GeoTIFF with suitable image bands.",
)

if uploaded_file is not None:

    st.write(f"**File:** {uploaded_file.name}")

    if st.button("Detect Buildings", type="primary"):

        try:
            with st.spinner("Reading image..."):
                (
                    rgb_image,
                    transform,
                    crs,
                    profile,
                    is_geotiff,
                ) = read_uploaded_image(uploaded_file)

            height, width, _ = rgb_image.shape

            st.write(
                f"**Image dimensions:** {width:,} × {height:,} pixels"
            )

            with st.spinner(
                "Running building segmentation on your image..."
            ):
                model = get_model()

                probabilities, building_mask = predict_buildings(
                    rgb_image,
                    model,
                    threshold=threshold,
                )

            with st.spinner("Extracting building polygons..."):
                # Polygonize in pixel coordinates first so that the
                # minimum-area setting always refers to pixel area.
                polygons = extract_building_polygons(
                    building_mask,
                    transform=Affine.identity(),
                    min_area=min_area,
                )

                # A transform is meaningful only when the GeoTIFF
                # actually contains geospatial metadata.
                valid_transform = (
                    is_geotiff
                    and crs is not None
                    and transform is not None
                    and transform != Affine.identity()
                )

                overlay = make_overlay(
                    rgb_image,
                    building_mask,
                )

                total_pixels = height * width
                building_pixels = int(building_mask.sum())
                building_percentage = (
                    100 * building_pixels / total_pixels
                    if total_pixels
                    else 0
                )

                st.session_state["geoai_result"] = {
                    "image": rgb_image,
                    "mask": building_mask,
                    "probabilities": probabilities,
                    "overlay": overlay,
                    "polygons": polygons,
                    "transform": transform if valid_transform else None,
                    "crs": crs if valid_transform else None,
                    "profile": profile,
                    "is_geotiff": is_geotiff,
                    "width": width,
                    "height": height,
                    "building_pixels": building_pixels,
                    "building_percentage": building_percentage,
                    "threshold": threshold,
                    "min_area": min_area,
                }

        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            st.exception(exc)


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.get("geoai_result")

if result is not None:

    st.divider()
    st.header("Detection Results")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Detected Regions",
        f"{len(result['polygons']):,}",
    )

    metric2.metric(
        "Building Pixels",
        f"{result['building_pixels']:,}",
    )

    metric3.metric(
        "Image Coverage",
        f"{result['building_percentage']:.2f}%",
    )

    metric4.metric(
        "Image Size",
        f"{result['width']:,} × {result['height']:,}",
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Original Image",
            "Building Overlay",
            "Segmentation Mask",
        ]
    )

    with tab1:
        st.image(
            result["image"],
            caption="Uploaded aerial imagery",
            use_container_width=True,
        )

    with tab2:
        st.image(
            result["overlay"],
            caption="Predicted buildings highlighted in red",
            use_container_width=True,
        )

    with tab3:
        st.image(
            result["mask"] * 255,
            caption="Binary building mask",
            use_container_width=True,
        )

    st.subheader("Export Results")

    mask_image = Image.fromarray(
        (result["mask"] * 255).astype(np.uint8)
    )

    import io

    mask_buffer = io.BytesIO()
    mask_image.save(mask_buffer, format="PNG")

    geojson_text = create_geojson(
        result["polygons"],
        transform=result["transform"],
        crs=result["crs"],
    )

    download_col1, download_col2 = st.columns(2)

        # --------------------------------------------------------
    # GeoTIFF mask export
    # --------------------------------------------------------

    if (
        result["is_geotiff"]
        and result["profile"] is not None
        and result["transform"] is not None
        and result["crs"] is not None
    ):
        import io

        geotiff_buffer = io.BytesIO()

        export_profile = result["profile"].copy()
        export_profile.update(
            driver="GTiff",
            height=result["mask"].shape[0],
            width=result["mask"].shape[1],
            count=1,
            dtype="uint8",
            nodata=None,
            compress="lzw",
        )

        # Preserve the original spatial reference and transform.
        export_profile["transform"] = result["transform"]
        export_profile["crs"] = result["crs"]

        with rasterio.open(
            geotiff_buffer,
            "w",
            **export_profile,
        ) as dst:
            dst.write(
                result["mask"].astype(np.uint8),
                1,
            )

        st.download_button(
            label="Download Building Mask (GeoTIFF)",
            data=geotiff_buffer.getvalue(),
            file_name="building_mask_georeferenced.tif",
            mime="image/tiff",
            use_container_width=True,
        )

    else:
        st.info(
            "GeoTIFF mask export is available only when the source "
            "image has a valid CRS and geospatial transform."
        )

    with download_col1:
        st.download_button(
            label="Download Building Mask (PNG)",
            data=mask_buffer.getvalue(),
            file_name="building_mask.png",
            mime="image/png",
            use_container_width=True,
        )

    with download_col2:
        st.download_button(
            label="Download Building Footprints (GeoJSON)",
            data=geojson_text.encode("utf-8"),
            file_name="building_footprints.geojson",
            mime="application/geo+json",
            use_container_width=True,
        )

    if result["is_geotiff"] and result["crs"] is not None:
        st.success(
            f"Geospatial metadata detected: {result['crs']}. "
            "Footprint coordinates have been transformed using the "
            "source raster's affine transform."
        )
    else:
        st.warning(
            "This input has no usable geographic reference. "
            "Footprints are expressed in pixel coordinates, not "
            "real-world longitude and latitude."
        )

    st.caption(
        "Image coverage is the proportion of pixels classified as buildings. "
        "It is not a measurement of real-world land area. "
        "The detected region count is the number of extracted polygons."
    )