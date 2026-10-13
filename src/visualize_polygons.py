
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GEOJSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "sample_132_buildings.geojson"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "linkedin_polygon_extraction.png"
)


def main():
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(
            f"GeoJSON file not found: {GEOJSON_PATH}"
        )

    gdf = gpd.read_file(GEOJSON_PATH)

    if gdf.empty:
        raise ValueError(
            "The GeoJSON contains no polygons to visualize."
        )

    fig, ax = plt.subplots(figsize=(8, 8))

    for geometry in gdf.geometry:
        if geometry is None or geometry.is_empty:
            continue

        if isinstance(geometry, Polygon):
            polygons = [geometry]
        elif isinstance(geometry, MultiPolygon):
            polygons = list(geometry.geoms)
        else:
            continue

        for polygon in polygons:
            x, y = polygon.exterior.xy
            ax.fill(
                x,
                y,
                facecolor="royalblue",
                edgecolor="white",
                linewidth=0.6,
                alpha=0.8,
            )

            # Draw interior rings, if any.
            for interior in polygon.interiors:
                ix, iy = interior.xy
                ax.plot(ix, iy, color="black", linewidth=0.4)

    ax.set_title(
        "Building Footprint Polygonization\n"
        "U-Net Prediction | WHU Building Dataset"
    )
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"Saved visualization: {OUTPUT_PATH}")
    print(f"Polygon features: {len(gdf)}")
    print("Note: Coordinates are in pixel space, not geographic space.")


if __name__ == "__main__":
    main()
