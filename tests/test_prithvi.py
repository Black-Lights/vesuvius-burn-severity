"""Checks on the Prithvi input and output handling with small synthetic arrays. No torch, no model,
no network: the model itself runs in the notebook."""

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from burnsev import prithvi


def _refl(first_band: np.ndarray) -> xr.Dataset:
    """Six 20 m bands on one date; the first band is given, the others are 0."""
    image = first_band.reshape(1, *first_band.shape)
    data = {b: (("time", "y", "x"), np.zeros_like(image)) for b in prithvi.BANDS}
    data[prithvi.BANDS[0]] = (("time", "y", "x"), image)
    # A time of day, as in the real cube, so that selecting "2025-08-14" keeps the time axis.
    return xr.Dataset(data, coords={"time": [np.datetime64("2025-08-14T09:50:41")]})


def _like(rows: int, cols: int) -> xr.DataArray:
    """A 20 m grid in UTM 33N, pixel centres 10 m inside a corner at (445140, 4524540)."""
    x = 445140 + 20 * (np.arange(cols) + 0.5)
    y = 4524540 - 20 * (np.arange(rows) + 0.5)
    da = xr.DataArray(np.zeros((rows, cols)), dims=("y", "x"), coords={"y": y, "x": x})
    return da.rio.write_crs("EPSG:32633")


def test_a_30m_pixel_is_the_area_weighted_mean_of_the_20m_pixels_under_it():
    # 3 x 3 pixels of 20 m become 2 x 2 pixels of 30 m. The first 30 m pixel covers all of
    # 20 m pixel (0, 0), half of (0, 1) and of (1, 0), and a quarter of (1, 1): 4/9, 2/9, 2/9, 1/9.
    one_pixel = np.zeros((3, 3))
    one_pixel[0, 0] = 1.0
    assert np.isclose(prithvi.to_30m(_refl(one_pixel), "2025-08-14")[0, 0, 0], 4 / 9)

    on_the_seam = np.zeros((3, 3))
    on_the_seam[0, 1] = 1.0  # split between the first and second 30 m pixel
    out = prithvi.to_30m(_refl(on_the_seam), "2025-08-14")[0]
    assert np.allclose(out, [[2 / 9, 2 / 9], [0, 0]])


def test_the_average_keeps_the_mean_and_a_masked_pixel_spoils_its_30m_pixels():
    values = np.arange(9, dtype=float).reshape(3, 3)
    out = prithvi.to_30m(_refl(values), "2025-08-14")
    assert out.shape == (6, 2, 2)
    assert np.isclose(out[0].mean(), values.mean())  # the same ground, the same mean

    values[2, 2] = np.nan
    out = prithvi.to_30m(_refl(values), "2025-08-14")[0]
    assert np.isnan(out[1, 1])
    assert not np.isnan(out[0, 0])


def test_the_chip_is_512_mirrored_and_standardised():
    image = np.full((6, 300, 400), 0.2)
    image[:, 299, :] = 0.5  # the last row
    image[0, 0, 0] = np.nan
    chip = prithvi.make_chip(image, means=[0.1] * 6, stds=[0.05] * 6)
    assert chip.shape == (6, 512, 512)
    assert chip.dtype == np.float32
    assert np.isclose(chip[1, 10, 10], (0.2 - 0.1) / 0.05)
    assert np.isclose(chip[0, 0, 0], (0.0 - 0.1) / 0.05)  # missing becomes 0, then standardised
    assert np.isclose(chip[1, 299, 5], (0.5 - 0.1) / 0.05)
    assert np.isclose(chip[1, 300, 5], chip[1, 298, 5])  # mirrored below the last row


def test_back_on_20m_each_pixel_takes_the_30m_pixel_under_its_centre():
    values = np.arange(4, dtype=float).reshape(2, 2)  # 30 m
    out = prithvi.to_20m(values, _like(3, 3))
    # 20 m centres at 10, 30 and 50 m fall in 30 m pixels 0, 1 and 1.
    assert out.values.tolist() == [[0, 1, 1], [2, 3, 3], [2, 3, 3]]


def test_a_saved_probability_reads_back_on_the_30m_grid(tmp_path):
    probability = np.array([[0.0, 0.25], [0.4999, np.nan]], dtype="float32")
    path = tmp_path / "p.tif"
    prithvi.save(probability, _like(3, 3), path, "2025-08-14")
    back = prithvi.read(path)
    assert np.allclose(back[0], [0.0, 0.25])
    assert back[1, 0] == np.float32(0.4999)  # exact: still under 0.5, as the model gave it
    assert np.isnan(back[1, 1])

    import rasterio

    with rasterio.open(path) as src:
        assert src.res == (30.0, 30.0)
        assert (src.bounds.left, src.bounds.top) == (445140.0, 4524540.0)  # the 20 m grid's corner


def test_found_by_class_is_the_share_of_each_class_the_model_calls_burned():
    severity = xr.DataArray(np.array([[0, 1, 1, 4], [4, 4, 4, 255]]), dims=("y", "x"))
    burned = xr.DataArray(np.array([[False, True, False, True], [True, True, False, True]]), dims=("y", "x"))
    shares = prithvi.found_by_class(burned, severity)
    assert shares == {"unburned": 0.0, "low": 50.0, "high": 75.0}
