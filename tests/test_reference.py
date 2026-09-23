"""Checks on reference.py with a tiny grid and a fake EFFIS answer. No network."""

import json

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from burnsev import reference


def _fire(fire_id, start, ring):
    return {
        "type": "Feature",
        "properties": {
            "id": fire_id,
            "FIREDATE": f"{start} 00:00:00",
            "FINALDATE": f"{start} 00:00:00",
            "COMMUNE": "Terzigno",
            "AREA_HA": "1",
        },
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _grid():
    # 4 by 4 pixels of 20 m in UTM 33N; pixel centres at 10, 30, 50, 70 m from the corner.
    x = 450_000 + np.arange(10, 80, 20.0)
    y = 4_520_000 - np.arange(10, 80, 20.0)
    grid = xr.DataArray(np.zeros((4, 4)), dims=("y", "x"), coords={"y": y, "x": x})
    return grid.rio.write_crs("EPSG:32633")


def test_fires_between_uses_the_start_date():
    fc = {"features": [_fire("a", "2017-07-11", []), _fire("b", "2025-08-07", [])]}
    assert [f["properties"]["id"] for f in reference.fires_between(fc, "2025-07-01", "2025-08-13")] == ["b"]


def test_rasterise_marks_the_pixel_centres_inside_the_polygon():
    x0, y0 = 450_000, 4_520_000
    square = [[x0, y0], [x0 + 40, y0], [x0 + 40, y0 - 40], [x0, y0 - 40], [x0, y0]]  # top-left 2 by 2
    mask = reference.rasterise([_fire("a", "2025-08-07", square)], _grid(), crs="EPSG:32633")
    assert mask.values.sum() == 4
    assert mask.values[:2, :2].all()


def test_agreement_counts_and_iou():
    ours = xr.DataArray(np.array([[1, 1, 0, 0]], dtype=bool), dims=("y", "x"))
    ref = xr.DataArray(np.array([[0, 1, 1, 0]], dtype=bool), dims=("y", "x"))
    out = reference.agreement(ours, ref, pixel_m=100)  # one pixel = 1 ha
    assert out["both_ha"] == 1.0 and out["only_ours_ha"] == 1.0 and out["only_reference_ha"] == 1.0
    assert out["iou"] == pytest.approx(0.33, abs=0.01)


def test_load_effis_reads_the_saved_file_without_the_network(tmp_path, monkeypatch):
    path = tmp_path / "effis.geojson"
    path.write_text(json.dumps({"features": [], "fetched": "2026-09-23"}), encoding="utf-8")

    def no_network(*args, **kwargs):
        raise AssertionError("the network was called")

    monkeypatch.setattr(reference, "fetch_effis", no_network)
    assert reference.load_effis((14.35, 40.77, 14.50, 40.87), path)["fetched"] == "2026-09-23"
