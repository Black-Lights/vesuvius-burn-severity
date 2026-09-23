"""Spectral index, composites and burn severity classes. Pure functions on xarray objects."""

from __future__ import annotations

import numpy as np
import xarray as xr

from . import aoi


def normalised_difference(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """(a - b) / (a + b), NaN where the denominator is zero or either input is missing."""
    denom = a + b
    return ((a - b) / denom).where(denom != 0)


def nbr(ds: xr.Dataset) -> xr.DataArray:
    """Normalised Burn Ratio, (B8A - B12) / (B8A + B12), per pixel and date.

    Healthy vegetation is bright in the near infrared (B8A) and dark in the short-wave
    infrared (B12), so NBR is high. Burned ground is the reverse, so NBR drops after a fire.
    """
    return normalised_difference(ds["B8A"], ds["B12"]).rename("NBR")


def window_median(da: xr.DataArray, start: str, end: str) -> xr.DataArray:
    """Per-pixel median over the dates in [start, end], skipping masked (NaN) values.

    A pixel that is cloudy on one date still gets a value from the others, and one odd date
    (thin haze, a missed cloud edge, a small early fire) cannot move the result.
    """
    sel = da.sel(time=slice(start, end))
    if sel.sizes["time"] == 0:
        raise ValueError(f"no dates between {start} and {end}")
    return sel.median(dim="time", skipna=True)


def window_count(da: xr.DataArray, start: str, end: str) -> xr.DataArray:
    """Per-pixel number of usable observations in [start, end]: the evidence behind a median."""
    return da.sel(time=slice(start, end)).count(dim="time")


def dnbr(nbr_da: xr.DataArray, pre: tuple[str, str], post: tuple[str, str]) -> xr.DataArray:
    """dNBR = median NBR before the fire minus median NBR after it. Positive means burned.

    Differencing removes what the two composites share (soil colour, terrain, the sensor)
    and keeps the change, which is what the severity classes are defined on.
    """
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
    return {
        names[int(v)]: round(float(c) * ha_per_pixel, 1)
        for v, c in zip(vals, counts)
        if int(v) in names
    }
