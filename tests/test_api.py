"""Checks on api.py, the functions behind the MCP tools, with fake EFFIS and STAC answers.
No network."""

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from burnsev import aoi, api, catalog, reference


def test_bbox_is_parsed_in_west_south_east_north_order():
    assert api.parse_bbox("14.35,40.77,14.50,40.87") == (14.35, 40.77, 14.50, 40.87)


@pytest.mark.parametrize(
    ("text", "hint"),
    [
        ("14.35,40.77,14.50", "west,south,east,north"),  # three numbers
        ("a,b,c,d", "west,south,east,north"),  # not numbers
        ("14.50,40.77,14.35,40.87", "west < east"),  # west and east swapped
        ("14.35,40.77,14.50,40.87,1", "west,south,east,north"),  # five numbers
        ("10,40,12,41", "degree a side"),  # 2 degrees wide
    ],
)
def test_a_wrong_bbox_says_how_to_fix_it(text, hint):
    with pytest.raises(api.BadArgument, match=hint):
        api.parse_bbox(text)


def test_period_must_be_iso_dates_in_order():
    assert api.parse_period("2025-08-01", "2025-08-31") == ("2025-08-01", "2025-08-31")
    with pytest.raises(api.BadArgument, match="YYYY-MM-DD"):
        api.parse_period("1 August 2025", "2025-08-31")
    with pytest.raises(api.BadArgument, match="after end"):
        api.parse_period("2025-09-01", "2025-08-31")


def _fire(fire_id, start, hectares, ring):
    return {
        "type": "Feature",
        "properties": {"id": fire_id, "COMMUNE": "Terzigno", "FIREDATE": f"{start} 00:00:00",
                       "FINALDATE": f"{start} 00:00:00", "AREA_HA": str(hectares)},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def test_find_fires_keeps_the_period_sorts_by_area_and_gives_each_fire_its_box(monkeypatch):
    square = [[14.40, 40.80], [14.45, 40.80], [14.45, 40.83], [14.40, 40.83], [14.40, 40.80]]
    collection = {"features": [
        _fire("old", "2017-07-11", 1136, square),
        _fire("small", "2025-08-05", 36, square),
        _fire("big", "2025-08-07", 681, square),
    ]}
    monkeypatch.setattr(reference, "fetch_effis", lambda box: collection)
    out = api.find_fires("14.35,40.77,14.50,40.87", "2025-07-01", "2025-09-15")
    assert out["fires_found"] == 2
    assert [f["effis_id"] for f in out["fires"]] == ["big", "small"]
    assert out["fires"][0]["bbox"] == "14.4000,40.8000,14.4500,40.8300"


def test_list_scenes_counts_products_and_dates(monkeypatch):
    table = pd.DataFrame({
        "id": ["a", "b", "c"], "date": ["2025-08-06", "2025-08-06", "2025-08-14"],
        "satellite": ["Sentinel-2A", "Sentinel-2C", "Sentinel-2B"], "orbit": [79, 79, 122],
        "tile": ["33TVF"] * 3, "cloud_pct": [0.0, 0.1, 3.2], "baseline": ["05.11"] * 3, "boa_add_offset": [-1000] * 3,
    })
    monkeypatch.setattr(catalog, "search_scenes", lambda *args: ["three items"])
    monkeypatch.setattr(catalog, "scene_table", lambda items: table)
    out = api.list_scenes("14.35,40.77,14.50,40.87", "2025-08-01", "2025-08-31")
    assert (out["products_found"], out["dates_found"]) == (3, 2)
    assert set(out["scenes"][0]) == {"date", "satellite", "orbit", "tile", "cloud_pct"}


def test_list_scenes_with_nothing_found(monkeypatch):
    monkeypatch.setattr(catalog, "search_scenes", lambda *args: [])
    monkeypatch.setattr(catalog, "scene_table", lambda items: pd.DataFrame())
    out = api.list_scenes("14.35,40.77,14.50,40.87", "2025-08-01", "2025-08-31")
    assert out["products_found"] == 0 and out["scenes"] == []


def test_utm_zone_comes_from_the_box():
    assert api.utm_crs((14.35, 40.77, 14.50, 40.87)) == "EPSG:32633"  # Vesuvius, as in aoi.CRS
    assert api.utm_crs((-8.6, 40.0, -8.4, 40.2)) == "EPSG:32629"  # central Portugal
    assert api.utm_crs((150.0, -34.0, 150.2, -33.8)) == "EPSG:32756"  # Sydney, southern hemisphere


def test_fire_windows_reproduce_the_notebook():
    pre, post = api.fire_windows(aoi.FIRE_START, aoi.FIRE_END)
    assert (pre, post) == (aoi.PRE_WINDOW, aoi.POST_WINDOW)


def test_pad_box_grows_by_about_a_kilometre():
    west, south, _east, _north = api.pad_box((14.35, 40.77, 14.50, 40.87), 1.0)
    assert round(40.77 - south, 4) == round(1 / 111, 4)
    assert 14.35 - west > 40.77 - south  # a degree of longitude is shorter at 40 N, so more degrees


def test_interval_is_two_iso_dates():
    assert api.parse_interval("2022-06-01/2022-08-31", "period_a") == ("2022-06-01", "2022-08-31")
    with pytest.raises(api.BadArgument, match="period_a"):
        api.parse_interval("summer 2022", "period_a")


def test_tinitaly_covers_vesuvius_only():
    assert api.tinitaly_covers(aoi.BBOX)
    assert not api.tinitaly_covers((-8.6, 40.0, -8.4, 40.2))


def test_heavy_tools_check_their_arguments_before_downloading():
    with pytest.raises(api.BadArgument, match="0.25 degree"):
        api.assess_burn("14.0,40.5,14.5,40.9", "2025-08-08", "2025-08-12")
    with pytest.raises(api.BadArgument, match="slope_threshold_deg"):
        api.assess_burn("14.35,40.77,14.50,40.87", "2025-08-08", "2025-08-12", slope_threshold_deg=60)
    with pytest.raises(api.BadArgument, match="min_drop"):
        api.vegetation_change("14.35,40.77,14.50,40.87", "2022-06-01/2022-08-31", "2024-06-01/2024-08-31", 0.9)


def test_drop_patches_keeps_patches_of_at_least_a_hectare():
    values = np.zeros((30, 40))
    values[5:15, 5:15] = -0.3  # 100 pixels of 20 m: 4 ha
    values[20:22, 30:32] = -0.3  # 4 pixels: 0.16 ha, below 1 ha
    values[25, 5] = -0.05  # a fall smaller than min_drop
    x = 450_000 + 10 + 20.0 * np.arange(40)
    y = 4_520_000 - 10 - 20.0 * np.arange(30)
    change = xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs("EPSG:32633")
    patches = api.drop_patches(change, min_drop=0.1)
    assert len(patches) == 1
    assert patches.iloc[0]["hectares"] == 4.0 and patches.iloc[0]["mean_change"] == -0.3
    assert abs(patches.iloc[0].geometry.area / 10_000 - 4.0) < 1e-9
