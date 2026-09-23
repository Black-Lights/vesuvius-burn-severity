"""Checks on the index, the window median and dNBR with tiny arrays. No network."""

import numpy as np
import pytest
import xarray as xr

from burnsev import indices


def _series(values, dates) -> xr.DataArray:
    data = np.array(values, dtype="float32").reshape(len(dates), 1, 1)
    coords = {"time": np.array(dates, dtype="datetime64[ns]"), "y": [0.0], "x": [0.0]}
    return xr.DataArray(data, dims=("time", "y", "x"), coords=coords)


def test_nbr_formula_and_zero_denominator():
    ds = xr.Dataset(
        {
            "B8A": xr.DataArray([[0.4, 0.0]], dims=("y", "x")),
            "B12": xr.DataArray([[0.2, 0.0]], dims=("y", "x")),
        }
    )
    out = indices.nbr(ds)
    assert float(out[0, 0]) == pytest.approx((0.4 - 0.2) / (0.4 + 0.2))
    assert np.isnan(out[0, 1])


def test_window_median_skips_masked_dates_and_refuses_an_empty_window():
    da = _series([0.3, np.nan, 0.5, -0.1], ["2025-06-01", "2025-06-05", "2025-06-10", "2025-08-20"])
    assert indices.window_median(da, "2025-06-01", "2025-06-30").item() == pytest.approx(0.4)
    assert indices.window_count(da, "2025-06-01", "2025-06-30").item() == 2
    with pytest.raises(ValueError):
        indices.window_median(da, "2025-07-01", "2025-07-31")


def test_severity_classes_at_the_breaks_and_no_data():
    values = xr.DataArray([[-0.2, 0.09, 0.10, 0.27, 0.44, 0.66, 1.1, np.nan]], dims=("y", "x"))
    codes = indices.severity_class(values)
    assert codes.dtype == "uint8"
    assert codes.values.tolist() == [[0, 0, 1, 2, 3, 4, 4, 255]]


def test_area_by_class_in_hectares():
    codes = xr.DataArray([[0, 0, 1, 4, 255]], dims=("y", "x")).astype("uint8")
    areas = indices.area_by_class(codes, pixel_m=100)  # one 100 m pixel = 1 ha; 255 is no data
    assert areas == {"unburned": 2.0, "low": 1.0, "high": 1.0}


def test_dnbr_is_positive_where_nbr_dropped():
    da = _series([0.3, 0.5, -0.1, -0.1], ["2025-06-01", "2025-06-10", "2025-08-20", "2025-09-01"])
    out = indices.dnbr(da, ("2025-06-01", "2025-06-30"), ("2025-08-13", "2025-09-30"))
    assert out.item() == pytest.approx(0.4 - (-0.1))
