"""Slope on synthetic planes of known steepness. No network."""

import numpy as np
import pytest
import xarray as xr

from burnsev import terrain


def _plane(dz_per_pixel_x: float, dz_per_pixel_y: float, n: int = 6) -> xr.DataArray:
    rows, cols = np.mgrid[0:n, 0:n]
    return xr.DataArray(cols * dz_per_pixel_x + rows * dz_per_pixel_y, dims=("y", "x"))


@pytest.mark.parametrize("degrees", [0.0, 10.0, 23.0, 45.0])
def test_slope_of_a_plane_rising_east(degrees):
    rise = 20 * np.tan(np.radians(degrees))  # metres per 20 m pixel
    slope = terrain.slope_degrees(_plane(rise, 0.0), pixel_m=20)
    assert np.allclose(slope.values[1:-1, 1:-1], degrees, atol=1e-4)


def test_slope_ignores_direction_and_leaves_the_border_empty():
    rise = 20 * np.tan(np.radians(30))
    diagonal = rise / np.sqrt(2)  # the same 30 degrees, facing south-east
    slope = terrain.slope_degrees(_plane(diagonal, diagonal), pixel_m=20)
    assert np.allclose(slope.values[1:-1, 1:-1], 30, atol=1e-4)
    assert np.isnan(slope.values[0]).all() and np.isnan(slope.values[:, -1]).all()
