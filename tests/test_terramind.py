"""Checks on the TerraMind land cover handling with small synthetic arrays. No torch, no model, no
network: the model itself runs in the notebook. At 10 m, 100 pixels are 1 ha."""

import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import wrap_xr

from burnsev import terramind

TREES, BUILT, RANGELAND = terramind.TREES, terramind.BUILT, terramind.CLASSES.index("rangeland")


def _grid_20m(values: np.ndarray, left: float = 0.0, top: float = 1000.0) -> xr.DataArray:
    """A 20 m array in UTM 33N with its top-left corner at (left, top)."""
    rows, cols = values.shape
    box = GeoBox.from_bbox((left, top - 20 * rows, left + 20 * cols, top), "EPSG:32633", tight=True, resolution=20)
    return wrap_xr(values, box)


def test_the_majority_is_the_class_most_samples_give_and_ties_go_to_the_lower_class():
    samples = [np.array([[2, 9]]), np.array([[2, 5]]), np.array([[9, 5]]), np.array([[2, 9]])]
    cover, votes = terramind.majority(samples)
    assert cover.tolist() == [[2, 5]]  # 9 and 5 tie at two votes each: the lower number wins
    assert votes.tolist() == [[3, 2]]


def test_esri_codes_become_the_model_class_numbers():
    codes = np.array([0, 1, 2, 4, 5, 7, 8, 9, 10, 11], dtype="uint8")
    assert terramind.esri_to_classes(codes).tolist() == list(range(10))
    assert terramind.CLASSES[terramind.esri_to_classes(np.array([7]))[0]] == "built area"


def test_the_10m_grid_nests_exactly_in_the_20m_grid():
    like = _grid_20m(np.zeros((3, 4)), left=5.0, top=1005.0)  # a corner off the 10 m multiples
    grid = terramind.grid_10m(like)
    assert grid.shape == (6, 8)
    assert (grid.affine.c, grid.affine.f) == (5.0, 1005.0)  # the same top-left corner
    values = _grid_20m(np.array([[1, 2], [3, 4]]))
    assert terramind.to_10m(values).tolist() == [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]


def test_the_check_against_esri_counts_hectares_and_matched_pixels():
    esri = np.full((10, 40), BUILT)
    esri[:, :20] = TREES  # 2 ha of trees, 2 ha of built area
    cover = esri.copy()
    cover[:, 15:20] = RANGELAND  # a quarter of ESRI's trees called rangeland
    table = terramind.check_against_esri(cover, esri)
    assert table.loc["trees", "ESRI 2023 (ha)"] == 2.0
    assert table.loc["trees", "TerraMind (ha)"] == 1.5
    assert table.loc["trees", "ESRI pixels matched (%)"] == 75.0
    assert table.loc["built area", "ESRI pixels matched (%)"] == 100.0
    assert pd.isna(table.loc["rangeland", "ESRI pixels matched (%)"])  # not in ESRI at all


def test_cover_by_severity_splits_the_fire_and_the_severe_steep_ground():
    severity = np.ones((10, 10), dtype="uint8")  # low on the left half (2 ha) ...
    severity[:, 5:] = 4  # ... high on the right half (2 ha)
    steep = np.zeros((10, 10), dtype=bool)
    steep[:, 5:] = True  # the high half is also steep
    cover = np.full((20, 20), TREES, dtype="uint8")
    cover[:10, 10:] = RANGELAND  # the top of the right half, 1 ha
    table = terramind.cover_by_severity(cover, _grid_20m(severity), _grid_20m(np.ones((10, 10), dtype=bool)),
                                        _grid_20m(steep), {0: "unburned", 1: "low", 4: "high"})
    assert table.loc["trees", "low"] == 2.0
    assert table.loc["trees", "high"] == 1.0
    assert table.loc["rangeland", "high"] == 1.0
    assert table.loc["rangeland", "severe and steep"] == 1.0
    assert "unburned" not in table.columns


def test_cell_exposure_gives_the_tree_share_and_the_distance_to_built_ground():
    # One row of 25 pixels of 20 m (x = 0 to 500 m), burned in the first 12: all in the first
    # 250 m cell. Built ground in the last 10 m column, x = 490 to 500 m.
    fire = _grid_20m(np.array([[True] * 12 + [False] * 13]), left=0.0, top=240.0)
    cover = np.full((2, 50), TREES, dtype="uint8")
    cover[:, 49] = BUILT
    cells = terramind.cell_exposure(cover, fire, fire, cell_m=250)
    assert len(cells) == 1  # only the cell with burned ground
    first = cells.iloc[0]
    assert first["trees_pct"] == 100
    # The cell's last pixel centre is at x = 245 m, the built pixel centre at 495 m.
    assert first["nearest_built_m"] == 250.0
