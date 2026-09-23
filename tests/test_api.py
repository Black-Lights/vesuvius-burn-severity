"""Checks on api.py, the functions behind the MCP tools, with fake EFFIS and STAC answers.
No network."""

import pandas as pd
import pytest

from burnsev import api, catalog, reference


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
