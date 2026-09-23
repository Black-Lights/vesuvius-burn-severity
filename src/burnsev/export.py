"""Files for GIS users: rasters as Cloud-Optimised GeoTIFF, vectors as GeoJSON, both checked."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor used below)
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from rasterio import features
from rasterio.io import MemoryFile
from rio_cogeo.cogeo import cog_translate, cog_validate
from rio_cogeo.profiles import cog_profiles
from shapely.geometry import shape

OUTPUT_DIR = Path("outputs")


def write_cog(
    da: xr.DataArray,
    path: Path,
    nodata: float,
    colormap: dict[int, tuple[int, int, int, int]] | None = None,
    overview_resampling: str = "nearest",
    tags: dict[str, str] | None = None,
) -> None:
    """Write one band as a Cloud-Optimised GeoTIFF: tiled, deflate-compressed, with overviews.

    The CRS and transform come from the array itself. A class raster gets a colour table, so
    QGIS shows it in the right colours without a style file. ``nodata`` replaces NaN.
    """
    values = np.where(np.isnan(da.values), nodata, da.values) if da.dtype.kind == "f" else da.values
    profile = {
        "driver": "GTiff",
        "height": da.shape[0],
        "width": da.shape[1],
        "count": 1,
        "dtype": str(values.dtype),
        "crs": da.rio.crs,
        "transform": da.rio.transform(),
        "nodata": nodata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with MemoryFile() as memory:
        with memory.open(**profile) as dataset:
            dataset.write(values, 1)
            if colormap:
                dataset.write_colormap(1, colormap)  # rio-cogeo carries it over to the COG
            if tags:
                dataset.update_tags(**tags)
        cog_profile = cog_profiles.get("deflate")
        cog_profile["predictor"] = 3 if values.dtype.kind == "f" else 2  # difference coding
        with memory.open() as source:
            cog_translate(
                source,
                str(path),
                cog_profile,
                overview_resampling=overview_resampling,
                in_memory=True,
                quiet=True,
            )


def main_fire_perimeter(fire: xr.DataArray) -> gpd.GeoDataFrame:
    """The main fire as polygons along the pixel edges, in the raster's CRS, with its area."""
    mask = fire.values.astype("uint8")
    polygons = [
        shape(geom)
        for geom, value in features.shapes(mask, mask=mask == 1, transform=fire.rio.transform())
        if value == 1
    ]
    out = gpd.GeoDataFrame(geometry=polygons, crs=fire.rio.crs).dissolve()
    out["area_ha"] = (out.geometry.area / 10_000).round(1)
    return out


def write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """GeoJSON as the standard (RFC 7946) defines it: longitude and latitude, polygons wound
    counter-clockwise, coordinates to 6 decimals (about 0.1 m)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_crs("EPSG:4326").to_file(
        path,
        driver="GeoJSON",
        engine="pyogrio",
        layer_options={"RFC7946": "YES", "COORDINATE_PRECISION": 6},
    )


def check_outputs(folder: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Reopen every file written: COG validity, CRS, size, features or pixels."""
    rows = []
    for path in sorted(folder.iterdir()):
        if path.suffix == ".tif":
            valid, errors, _ = cog_validate(str(path), quiet=True)
            with rasterio.open(path) as src:
                crs, content = src.crs.to_string(), f"{src.width} x {src.height} px, {src.dtypes[0]}"
            check = "valid COG" if valid else "; ".join(errors)
        elif path.suffix == ".geojson":
            gdf = gpd.read_file(path)
            crs, content = gdf.crs.to_string(), f"{len(gdf)} features"
            check = "reads back"
        elif path.suffix == ".png":
            with Image.open(path) as image:
                crs, content = "none (picture)", f"{image.width} x {image.height} px"
            check = "reads back"
        else:
            continue
        rows.append(
            {"file": path.name, "size (KB)": round(path.stat().st_size / 1000), "CRS": crs,
             "content": content, "check": check}
        )
    return pd.DataFrame(rows).set_index("file")


def colormap_from_hex(colours: list[str]) -> dict[int, tuple[int, int, int, int]]:
    """Class code to RGBA for a GeoTIFF colour table: code i gets ``colours[i]``."""
    table = {}
    for code, colour in enumerate(colours):
        r, g, b = (int(colour.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        table[code] = (r, g, b, 255)
    return table


def save_png(fig: Figure, path: Path, dpi: int = 100) -> None:
    """Save a figure for reports and close it, so the notebook does not show it a second time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
