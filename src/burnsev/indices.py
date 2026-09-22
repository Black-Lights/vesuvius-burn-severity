"""Spectral indices and burn severity classes. Pure functions on xarray objects."""

from __future__ import annotations

import numpy as np
import xarray as xr

from . import aoi


def normalised_difference(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """(a - b) / (a + b), NaN where the denominator is zero or either input is missing."""
    denom = a + b
    return ((a - b) / denom).where(denom != 0)


def nbr(ds: xr.Dataset) -> xr.DataArray:
    """Normalised Burn Ratio, (B8A - B12) / (B8A + B12).

    Healthy vegetation is bright in the near infrared (B8A) and dark in the short-wave
    infrared (B12); burned ground is the reverse, so NBR drops sharply after a fire.
    """
    return normalised_difference(ds["B8A"], ds["B12"]).rename("NBR")


def ndvi(ds: xr.Dataset) -> xr.DataArray:
    """Normalised Difference Vegetation Index, (B8A - B04) / (B8A + B04).

    B8A (narrow NIR, 20 m) is used instead of B08 so that NDVI and NBR share the same
    band and the same resolution.
    """
    return normalised_difference(ds["B8A"], ds["B04"]).rename("NDVI")


def window_median(da: xr.DataArray, start: str, end: str) -> xr.DataArray:
    """Per-pixel median over the time steps in [start, end]. NaNs (masked pixels) are
    skipped, so a pixel that is cloudy on one date still gets a value from the others."""
    sel = da.sel(time=slice(start, end))
    if sel.sizes["time"] == 0:
        raise ValueError(f"no scenes between {start} and {end}")
    return sel.median(dim="time", skipna=True)


def dnbr(nbr_da: xr.DataArray, pre: tuple[str, str], post: tuple[str, str]) -> xr.DataArray:
    """dNBR = median NBR before the fire - median NBR after it. Positive means burned."""
    return (window_median(nbr_da, *pre) - window_median(nbr_da, *post)).rename("dNBR")


def severity_class(dnbr_da: xr.DataArray) -> xr.DataArray:
    """Integer class raster from dNBR using the thresholds in ``aoi.SEVERITY_CLASSES``.

    0 = unburned, 1 = low, 2 = moderate-low, 3 = moderate-high, 4 = high; 255 = no data.
    """
    out = xr.full_like(dnbr_da, 255, dtype="uint8")
    for code, (_, lo, hi) in enumerate(aoi.SEVERITY_CLASSES):
        out = out.where(~((dnbr_da >= lo) & (dnbr_da < hi)), code)
    out = out.where(~dnbr_da.isnull(), 255)
    return out.astype("uint8").rename("severity")


def class_names() -> dict[int, str]:
    return {i: name for i, (name, _, _) in enumerate(aoi.SEVERITY_CLASSES)}


def area_by_class(sev: xr.DataArray, pixel_m: float) -> dict[str, float]:
    """Hectares per severity class."""
    names = class_names()
    ha_per_pixel = (pixel_m * pixel_m) / 10_000.0
    vals, counts = np.unique(sev.values, return_counts=True)
    return {names[int(v)]: round(float(c) * ha_per_pixel, 1) for v, c in zip(vals, counts) if int(v) in names}
