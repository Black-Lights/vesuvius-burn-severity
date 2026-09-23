"""Terrain: the Copernicus DEM on the cube's 20 m grid, and slope in degrees."""

from __future__ import annotations

import numpy as np
import odc.geo.xr  # registers the .odc accessor used below
import odc.stac
import planetary_computer
import pystac
import xarray as xr


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
