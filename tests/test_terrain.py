"""Slope on synthetic planes of known steepness. No network."""

import numpy as np
import pytest
import rioxarray  # noqa: F401
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


def _utm(values: np.ndarray, pixel_m: float) -> xr.DataArray:
    n_y, n_x = values.shape
    x = 450_000 + pixel_m / 2 + pixel_m * np.arange(n_x)
    y = 4_520_000 - pixel_m / 2 - pixel_m * np.arange(n_y)
    da = xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x})
    return da.rio.write_crs("EPSG:32633")


def test_slope_to_grid_derives_at_10_m_and_keeps_the_angle_on_a_20_m_grid():
    rise = 10 * np.tan(np.radians(30))  # metres per 10 m pixel, 30 degrees facing east
    dem = _utm(_plane(rise, 0.0, n=40).values.astype("float64"), 10)
    like = _utm(np.zeros((18, 18)), 20)
    slope = terrain.slope_to_grid(dem, like)
    assert slope.shape == like.shape
    assert np.allclose(slope.values[2:-2, 2:-2], 30, atol=0.01)


def test_load_tinitaly_reads_the_saved_cut_without_the_network(tmp_path, monkeypatch):
    path = tmp_path / "dem.tif"
    _utm(np.full((4, 4), 100.0, dtype="float32"), 10).rio.to_raster(path)

    def no_network(*args, **kwargs):
        raise AssertionError("the network was called")

    monkeypatch.setattr(terrain, "clip_tinitaly", no_network)
    dem = terrain.load_tinitaly(_utm(np.zeros((2, 2)), 20), path)
    assert float(dem.mean()) == 100.0
