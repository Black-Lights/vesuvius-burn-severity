"""Checks on mask_and_scale with a tiny synthetic cube: two dates, two by two pixels. No network."""

import numpy as np
import pytest
import xarray as xr

from burnsev import ingest

OFFSETS = {"2025-08-06": -1000, "2025-08-17": 0}


def _cube() -> xr.Dataset:
    coords = {
        "time": np.array(["2025-08-06", "2025-08-17"], dtype="datetime64[ns]"),
        "y": [0.0, 1.0],
        "x": [0.0, 1.0],
    }
    b04 = xr.DataArray(
        np.full((2, 2, 2), 2000, dtype="uint16"), dims=("time", "y", "x"), coords=coords
    )
    # SCL: 4 vegetation, 9 high cloud, 3 shadow, 0 no data.
    scl = xr.DataArray(
        np.array([[[4, 4], [4, 9]], [[4, 3], [0, 4]]], dtype="uint8"),
        dims=("time", "y", "x"),
        coords=coords,
    )
    return xr.Dataset({"B04": b04, "SCL": scl})


def test_offset_is_applied_per_date_then_scaled():
    out = ingest.mask_and_scale(_cube(), OFFSETS)
    assert float(out["B04"][0, 0, 0]) == pytest.approx((2000 - 1000) / 10000)
    assert float(out["B04"][1, 0, 0]) == pytest.approx(2000 / 10000)


def test_masked_classes_become_nan_and_are_counted():
    out = ingest.mask_and_scale(_cube(), OFFSETS)
    assert np.isnan(out["B04"][0, 1, 1])  # high cloud
    assert np.isnan(out["B04"][1, 0, 1])  # shadow
    assert np.isnan(out["B04"][1, 1, 0])  # no data
    assert out["valid_fraction"].values.tolist() == [0.75, 0.5]


def test_cache_round_trip_keeps_dtype_and_crs_coordinate(tmp_path):
    ds = _cube().assign_coords(spatial_ref=xr.DataArray(0, attrs={"crs_wkt": "EPSG:32633"}))
    ds["B04"].attrs["nodata"] = 0
    path = tmp_path / "cube.nc"
    ingest.save_cube(ds, path)
    back = ingest.load_cached(path)
    assert list(back.data_vars) == ["B04", "SCL"]
    assert "spatial_ref" in back.coords
    assert back["B04"].dtype == "uint16"
    assert ingest.load_cached(tmp_path / "missing.nc") is None


def test_scl_is_dropped_and_negatives_are_clamped():
    ds = _cube()
    ds["B04"][0, 0, 0] = 500  # below the offset: would be negative reflectance
    out = ingest.mask_and_scale(ds, OFFSETS)
    assert "SCL" not in out
    assert float(out["B04"][0, 0, 0]) == 0.0
