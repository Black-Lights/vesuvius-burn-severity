"""Checks on the display stretch and the web map layers with small synthetic images. No network,
no figure."""

import base64
import io

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from PIL import Image
from rasterio.warp import transform_bounds

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


def _grid(values):
    # pixels of 20 m in UTM 33N near Vesuvius; the first pixel centre 10 m in from the corner
    rows, cols = values.shape
    x = 450_000 + 10 + 20.0 * np.arange(cols)
    y = 4_520_000 - 10 - 20.0 * np.arange(rows)
    grid = xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x})
    return grid.rio.write_crs("EPSG:32633")


def test_to_web_corners_match_the_grid():
    grid = _grid(np.ones((30, 40)))
    _, corners = plots.to_web(grid.values, grid)
    west, south, east, north = transform_bounds("EPSG:32633", "EPSG:4326", *grid.rio.bounds())
    (s, w), (n, e) = corners
    # within about a pixel (20 m is about 0.0002 degrees): the web grid only turns the box a little
    assert abs(s - south) < 3e-4 and abs(n - north) < 3e-4
    assert abs(w - west) < 3e-4 and abs(e - east) < 3e-4


def test_to_web_invents_no_class():
    classes = np.where(np.indices((30, 40)).sum(axis=0) % 2 == 0, 1.0, 3.0)  # a checkerboard of 1 and 3
    grid = _grid(classes)
    moved, _ = plots.to_web(grid.values, grid)
    assert set(np.unique(moved[~np.isnan(moved)])) == {1.0, 3.0}


def test_to_web_keeps_the_bands_of_a_picture():
    grid = _grid(np.zeros((30, 40)))
    rgb = np.dstack([np.full((30, 40), v) for v in (0.1, 0.5, 0.9)])
    moved, _ = plots.to_web(rgb, grid)
    assert moved.shape[-1] == 3
    inside = ~np.isnan(moved[..., 0])
    assert (moved[inside] == np.float32([0.1, 0.5, 0.9])).all()  # nearest: every value kept exactly


def test_data_uri_decodes_back_to_the_image():
    pixels = np.zeros((5, 7, 4), dtype="uint8")
    pixels[2, 3] = (255, 0, 0, 255)
    uri = plots.data_uri(pixels, "PNG")
    assert uri.startswith("data:image/png;base64,")
    back = np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))))
    assert (back == pixels).all()
