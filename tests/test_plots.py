"""Checks on the display stretch with a small synthetic image. No network, no figure."""

import numpy as np
import xarray as xr

from burnsev import plots


def _image() -> xr.Dataset:
    values = np.linspace(0.0, 1.0, 101).reshape(1, 101)  # 0.00, 0.01, ..., 1.00
    data = {b: xr.DataArray(values.copy(), dims=("y", "x")) for b in ("B04", "B03", "B02")}
    data["B04"][0, 50] = np.nan  # one masked pixel
    return xr.Dataset(data)


def test_bounds_are_the_2nd_and_98th_percentiles():
    bounds = plots.stretch_bounds(_image(), ("B03",))
    lo, hi = bounds["B03"]
    assert lo == 0.02
    assert hi == 0.98


def test_rgb_is_scaled_within_bounds_and_missing_is_grey():
    ds = _image()
    bounds = plots.stretch_bounds(ds, plots.TRUE_COLOUR)
    rgb = plots.to_rgb(ds, plots.TRUE_COLOUR, bounds)
    assert rgb.shape == (1, 101, 3)
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0
    assert rgb[0, 0, 1] == 0.0  # green band at 0.00 is below the black point
    assert rgb[0, 100, 1] == 1.0  # and at 1.00 above the white point
    assert rgb[0, 50, 0] == plots.MISSING_GREY  # the masked red pixel
