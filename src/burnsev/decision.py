"""The decision: which burned slopes to treat first against erosion and debris flows.

The rule follows the terrain term of the M1 model of Staley et al. (2017): the share of an area
that is both burned at moderate or high severity and at least 23 degrees steep. Here the area is
a square grid cell instead of a stream catchment, so every cell is a planning unit a crew can be
sent to.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from shapely.geometry import box

from . import aoi

PRIORITY_LABELS = ("1: treat first", "2: treat next", "3: monitor")


def grid_index(like: xr.DataArray, cell_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of the grid cell that each pixel centre falls in.

    Cells are aligned to multiples of ``cell_m`` in the map coordinates, so the same cell has the
    same edges in every run and in any other layer on the same projection. 250 m is not a
    multiple of 20 m, so a cell holds 12 or 13 pixels a side (144 to 169 pixels).
    """
    x, y = np.meshgrid(like.x.values, like.y.values)
    return np.floor(y / cell_m).astype(int), np.floor(x / cell_m).astype(int)


def score_cells(
    fire: xr.DataArray,
    severe: xr.DataArray,
    slope: xr.DataArray,
    green: xr.DataArray,
    threshold_deg: float,
    cell_m: float = aoi.GRID_CELL_M,
    shares: tuple[float, float] = aoi.PRIORITY_SHARES,
) -> pd.DataFrame:
    """One row per grid cell that touches the fire, ranked.

    ``severe_steep_share`` is the share of the cell's pixels that are severe and at least
    ``threshold_deg`` steep, the ranking key (a share, because cells differ in pixel count).
    ``green`` is post-fire NDVI; its median over the burned pixels of a cell says how much
    green cover is left to hold the soil. ``shares`` are the cuts between the three priorities.
    """
    rows, cols = grid_index(fire, cell_m)
    pixel_ha = float(abs(fire.x[1] - fire.x[0])) ** 2 / 10_000
    burned = fire.values.astype(bool)
    hit = burned & severe.values.astype(bool) & (slope.values >= threshold_deg)
    pixels = pd.DataFrame(
        {
            "row": rows.ravel(),
            "col": cols.ravel(),
            "burned": burned.ravel(),
            "severe": (burned & severe.values.astype(bool)).ravel(),
            "severe_steep": hit.ravel(),
            "green": np.where(burned, green.values, np.nan).ravel(),
        }
    )
    groups = pixels.groupby(["row", "col"])
    cells = pd.DataFrame(
        {
            "pixels": groups.size(),
            "burned_ha": groups["burned"].sum() * pixel_ha,
            "severe_ha": groups["severe"].sum() * pixel_ha,
            "severe_steep_ha": groups["severe_steep"].sum() * pixel_ha,
            "green_ndvi": groups["green"].median(),
        }
    )
    cells = cells[cells["burned_ha"] > 0].copy()
    cells["severe_steep_share"] = cells["severe_steep_ha"] / (cells["pixels"] * pixel_ha)
    first, second = shares
    cells["priority"] = np.select(
        [cells["severe_steep_share"] >= first, cells["severe_steep_share"] >= second],
        PRIORITY_LABELS[:2],
        PRIORITY_LABELS[2],
    )
    cells = cells.sort_values(["severe_steep_share", "severe_steep_ha"], ascending=False)
    cells.insert(0, "rank", np.arange(1, len(cells) + 1))
    cells = cells.reset_index()
    cells["x"] = (cells["col"] + 0.5) * cell_m
    cells["y"] = (cells["row"] + 0.5) * cell_m
    return cells


def cells_to_geodataframe(cells: pd.DataFrame, crs: str, cell_m: float = aoi.GRID_CELL_M) -> gpd.GeoDataFrame:
    """The cells as square polygons in ``crs``, with the centre in longitude and latitude."""
    squares = [
        box(c * cell_m, r * cell_m, (c + 1) * cell_m, (r + 1) * cell_m)
        for r, c in zip(cells["row"], cells["col"])
    ]
    out = gpd.GeoDataFrame(cells.copy(), geometry=squares, crs=crs)
    centres = gpd.GeoSeries(gpd.points_from_xy(cells["x"], cells["y"]), crs=crs).to_crs("EPSG:4326")
    out["lon"] = centres.x.round(5).values
    out["lat"] = centres.y.round(5).values
    return out


def sensitivity(
    fire: xr.DataArray,
    severe: xr.DataArray,
    slope: xr.DataArray,
    green: xr.DataArray,
    thresholds: tuple[float, ...],
    chosen: float,
) -> pd.DataFrame:
    """How the answer moves with the slope threshold: area, cells per priority, and how many of
    the first-priority cells at ``chosen`` stay first priority."""
    runs = {t: score_cells(fire, severe, slope, green, t) for t in thresholds}
    reference = set(_first(runs[chosen]))
    rows = {}
    for t, cells in runs.items():
        first = set(_first(cells))
        rows[f"{t:.0f}°"] = {
            "severe and steep (ha)": round(float(cells["severe_steep_ha"].sum())),
            "priority 1 cells": len(first),
            "priority 2 cells": int((cells["priority"] == PRIORITY_LABELS[1]).sum()),
            f"priority 1 cells shared with {chosen:.0f}°": len(first & reference),
        }
    return pd.DataFrame(rows).T.rename_axis("slope threshold")


def _first(cells: pd.DataFrame) -> list[tuple[int, int]]:
    top = cells[cells["priority"] == PRIORITY_LABELS[0]]
    return list(zip(top["row"], top["col"]))


def largest_block(cells: pd.DataFrame, label: str = PRIORITY_LABELS[0]) -> pd.DataFrame:
    """The biggest group of ``label`` cells that touch along an edge: one area a crew can work."""
    chosen = cells[cells["priority"] == label]
    if chosen.empty:
        return chosen
    r0, c0 = chosen["row"].min(), chosen["col"].min()
    grid = np.zeros((chosen["row"].max() - r0 + 1, chosen["col"].max() - c0 + 1), dtype=bool)
    grid[chosen["row"] - r0, chosen["col"] - c0] = True
    groups, _ = ndimage.label(grid)
    member = groups[chosen["row"] - r0, chosen["col"] - c0]
    return chosen[member == np.bincount(member)[1:].argmax() + 1]


def summary(
    grid: pd.DataFrame,
    sens: pd.DataFrame,
    place: str,
    burned_ha: float,
    severe_ha: float,
    threshold_deg: float,
    green_unburned: float,
    cell_m: float = aoi.GRID_CELL_M,
    shares: tuple[float, float] = aoi.PRIORITY_SHARES,
) -> str:
    """The answer in plain words for a reader without remote sensing, built from the numbers so
    it can never disagree with them. ``grid`` is the ranked cells with ``lon`` and ``lat``."""
    first = grid[grid["priority"] == PRIORITY_LABELS[0]]
    second = grid[grid["priority"] == PRIORITY_LABELS[1]]
    block = largest_block(grid)
    steep_ha = grid["severe_steep_ha"].sum()
    kept = sens.iloc[:, -1]
    low, high = sens.index[0], sens.index[-1]
    cover = first["green_ndvi"].median() / green_unburned
    return (
        f"The fire burned {burned_ha:,.0f} ha {place}. On {severe_ha:,.0f} ha it burned at moderate "
        f"or high severity, and {steep_ha:,.0f} ha of that ground is {threshold_deg:.0f}° or "
        f"steeper, where rain washes ash and soil downhill most easily. "
        f"Treat first: {len(first)} squares of {cell_m:.0f} by {cell_m:.0f} m where at least "
        f"{shares[0]:.0%} of the ground is both badly burned and steep, {first['severe_steep_ha'].sum():,.0f} ha. "
        f"{len(block)} of them form one continuous area centred at {block['lat'].mean():.4f} N, "
        f"{block['lon'].mean():.4f} E. Treat next: {len(second)} squares at {shares[1]:.0%} to "
        f"{shares[0]:.0%}, {second['severe_steep_ha'].sum():,.0f} ha. The first squares keep only "
        f"about {cover:.0%} of the green cover of unburned ground, so little holds the soil. The list "
        f"hardly depends on the exact steepness limit: at {low}, {kept.iloc[0]} of the {len(first)} "
        f"stay first; at {high}, {kept.iloc[-1]} do. Every square, with its coordinates, is in "
        f"alert_cells.geojson and in the ranked table."
    )
