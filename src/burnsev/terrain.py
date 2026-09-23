"""Terrain: elevation models and slope in degrees on the cube's 20 m grid.

The slope comes from TINITALY 1.1, the 10 m elevation model of Italy by INGV (Tarquini et al.
2023, https://doi.org/10.13127/tinitaly/1.1, licence CC BY 4.0). It is built from contour lines
and surveyed elevation points, so it describes the ground, and 10 m matches the DEMs the USGS
debris-flow models are applied with. Copernicus DEM GLO-30 (a 30 m radar surface model, canopy
included) is kept as a comparison.
"""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import odc.geo.xr  # registers the .odc accessor used below
import odc.stac
import planetary_computer
import pystac
import requests
import rioxarray
import xarray as xr
from odc.geo.xr import xr_reproject
from rasterio.io import MemoryFile

# TINITALY is cut into 50 km tiles in UTM zone 32N, named after their lower-left corner.
# w45095 (northing 4,500 km, easting 950 km) contains the whole box.
TINITALY_TILE_URL = "https://tinitaly.pi.ingv.it/data_1.1/w45095_s10/w45095_s10.zip"
TINITALY_FILE = Path("data/reference/tinitaly_vesuvius_10m.tif")
TINITALY_CITATION = (
    "Tarquini S., Isola I., Favalli M., Battistini A., Dotta G. (2023). TINITALY, a digital "
    "elevation model of Italy with a 10 meters cell size (Version 1.1). INGV. "
    "https://doi.org/10.13127/tinitaly/1.1. CC BY 4.0"
)


def load_dem(items: list[pystac.Item], like: xr.DataArray) -> xr.DataArray:
    """Elevation in metres on exactly the grid of ``like``.

    Copernicus DEM GLO-30 is stored in latitude and longitude at one arc second, about 30 m.
    odc-stac warps it on this machine onto the cube's own grid (``like.odc.geobox``), so
    every DEM pixel sits on a Sentinel-2 pixel. Bilinear resampling, because elevation is
    continuous: nearest neighbour would leave 30 m steps that read as false breaks in slope.
    """
    ds = odc.stac.load(
        items,
        bands=["data"],
        geobox=like.odc.geobox,
        resampling="bilinear",
        groupby="solar_day",
        patch_url=planetary_computer.sign,
        chunks=None,
    )
    return ds["data"].isel(time=0, drop=True).astype("float32").rename("elevation")


def slope_degrees(dem: xr.DataArray, pixel_m: float) -> xr.DataArray:
    """Slope in degrees with Horn's 3 by 3 method, the one GDAL and QGIS use.

    Each gradient is a weighted difference across the eight neighbours, which smooths noise
    better than a two-pixel difference. The one-pixel border has no full neighbourhood and
    is left NaN.
    """
    z = dem.values.astype("float64")
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]  # row above
    d, f = z[1:-1, :-2], z[1:-1, 2:]  # same row, left and right
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]  # row below
    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * pixel_m)
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * pixel_m)
    out = np.full(z.shape, np.nan, dtype="float32")
    out[1:-1, 1:-1] = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    return dem.copy(data=out).rename("slope")


def clip_tinitaly(like: xr.DataArray, path: Path = TINITALY_FILE, margin_m: float = 200) -> None:
    """Download the TINITALY tile once, cut it to the box plus a margin, and save the cut.

    The tile is 63 MB; the cut is a few MB, small enough for the repository, so the notebook
    never depends on the INGV server. The margin keeps the slope window and the resampling
    whole at the edges of the box. Source, citation and fetch date go into the file's tags.
    """
    response = requests.get(TINITALY_TILE_URL, timeout=600)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".tif"))
        tile_bytes = archive.read(name)
    with MemoryFile(tile_bytes) as memory, memory.open() as source:
        tile = rioxarray.open_rasterio(source, masked=True).squeeze("band", drop=True)
        box = like.odc.geobox.boundingbox.to_crs(tile.rio.crs)
        cut = tile.rio.clip_box(
            box.left - margin_m, box.bottom - margin_m, box.right + margin_m, box.top + margin_m
        ).load()
    cut.rio.write_nodata(-9999.0, encoded=True, inplace=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    cut.rio.to_raster(
        path,
        compress="deflate",
        predictor=3,
        tiled=True,
        tags={
            "source": TINITALY_TILE_URL,
            "citation": TINITALY_CITATION,
            "fetched": datetime.now(UTC).date().isoformat(),
        },
    )


def load_tinitaly(like: xr.DataArray, path: Path = TINITALY_FILE) -> xr.DataArray:
    """TINITALY elevation in metres on its own 10 m grid (UTM 32N), cut to the box.

    Reads the cut saved in the repository; downloads and cuts the tile only if it is missing.
    """
    if not path.exists():
        clip_tinitaly(like, path)
    return rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True).rename("elevation")


def slope_to_grid(dem: xr.DataArray, like: xr.DataArray) -> xr.DataArray:
    """Slope computed on the DEM's own grid, then resampled onto the grid of ``like``.

    Deriving first and resampling after keeps the relief the finer grid sees; resampling the
    elevation first would smooth it (Grohmann 2015). Bilinear, because slope is continuous.
    """
    pixel_m = abs(float(dem.rio.resolution()[0]))
    slope = slope_degrees(dem, pixel_m).rio.write_crs(dem.rio.crs)
    return xr_reproject(slope, like.odc.geobox, resampling="bilinear").rename("slope")
