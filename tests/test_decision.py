"""The grid, the score and the priorities on tiny grids with known answers. No network."""

import numpy as np
import xarray as xr

from burnsev import decision


def _grid(values, pixel_m=100.0):
    values = np.asarray(values, dtype=float)
    n_y, n_x = values.shape
    x = pixel_m / 2 + pixel_m * np.arange(n_x)  # pixel centres from the origin
    y = n_y * pixel_m - pixel_m / 2 - pixel_m * np.arange(n_y)
    return xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x})


def test_grid_index_puts_pixel_centres_in_aligned_cells():
    rows, cols = decision.grid_index(_grid(np.zeros((4, 4))), cell_m=200)
    assert cols[0].tolist() == [0, 0, 1, 1]
    assert rows[:, 0].tolist() == [1, 1, 0, 0]  # the top row of pixels is the northern cell


def test_score_ranks_by_the_severe_and_steep_share():
    # Two cells of 2 by 2 pixels side by side; pixel 100 m = 1 ha.
    fire = _grid([[1, 1, 1, 1], [1, 1, 1, 1]])
    severe = _grid([[1, 1, 1, 0], [1, 1, 0, 0]])
    slope = _grid([[30, 30, 30, 5], [30, 5, 5, 5]])
    green = _grid(np.full((2, 4), 0.2))
    cells = decision.score_cells(fire, severe, slope, green, 23, cell_m=200, shares=(0.5, 0.25))
    assert cells["rank"].tolist() == [1, 2]
    assert cells["severe_steep_share"].tolist() == [0.75, 0.25]
    assert cells["priority"].tolist() == ["1: treat first", "2: treat next"]
    assert cells["severe_steep_ha"].tolist() == [3.0, 1.0]


def test_cells_outside_the_fire_are_dropped_and_polygons_are_squares():
    fire = _grid([[1, 1, 0, 0], [1, 1, 0, 0]])
    ones = _grid(np.ones((2, 4)))
    cells = decision.score_cells(fire, ones, ones * 30, ones, 23, cell_m=200)
    assert len(cells) == 1
    shapes = decision.cells_to_geodataframe(cells, "EPSG:32633", cell_m=200)
    assert shapes.geometry.iloc[0].area == 200 * 200


def test_largest_block_joins_cells_that_share_an_edge():
    cells = decision.pd.DataFrame(
        {
            "row": [0, 0, 1, 5],
            "col": [0, 1, 1, 5],
            "priority": ["1: treat first"] * 4,
        }
    )
    block = decision.largest_block(cells)
    assert sorted(zip(block["row"], block["col"])) == [(0, 0), (0, 1), (1, 1)]
