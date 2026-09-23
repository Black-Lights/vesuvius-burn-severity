"""Spectral index, composites and burn severity classes. Pure functions on xarray objects."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import ndimage

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


def ndvi(ds: xr.Dataset) -> xr.DataArray:
    """Normalised Difference Vegetation Index, (B8A - B04) / (B8A + B04), per pixel and date.

    NBR reacts to char and ash through the short-wave infrared; NDVI reacts to green leaf
    through the red band. After a fire it measures the cover left to hold the soil. B8A (the
    narrow near-infrared band) is used instead of the 10 m B08 so both indices share the 20 m grid.
    """
    return normalised_difference(ds["B8A"], ds["B04"]).rename("NDVI")


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
    """Class code per pixel from dNBR, using the breaks in ``aoi.SEVERITY_CLASSES``.

    0 = unburned, 1 = low, 2 = moderate-low, 3 = moderate-high, 4 = high; 255 = no data.
    A pixel exactly on a break goes to the upper class (0.27 is moderate-low).
    """
    breaks = [low for _, low, _ in aoi.SEVERITY_CLASSES[1:]]  # 0.10, 0.27, 0.44, 0.66
    codes = np.digitize(dnbr_da.values, breaks).astype("uint8")
    codes[np.isnan(dnbr_da.values)] = 255
    return dnbr_da.copy(data=codes).rename("severity")


def main_fire(sev: xr.DataArray) -> xr.DataArray:
    """True on the largest connected patch of burned pixels (class low or above).

    Connected includes diagonal neighbours. The small patches elsewhere in the box also pass
    the dNBR breaks, but they are harvested fields, other small fires and cloud edges, not this
    fire; one fire is one perimeter.
    """
    burned = (sev.values >= 1) & (sev.values != 255)
    labels, n = ndimage.label(burned, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return sev.copy(data=np.zeros(sev.shape, dtype=bool)).rename("main_fire")
    sizes = ndimage.sum(burned, labels, range(1, n + 1))
    return sev.copy(data=labels == int(np.argmax(sizes)) + 1).rename("main_fire")


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
