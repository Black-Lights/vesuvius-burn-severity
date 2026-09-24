"""The pipeline as plain functions for the MCP server (bonus A).

Small JSON-ready values in and out, so a language model can pass them and read them. Every
argument is checked first; a wrong one raises ``BadArgument`` with a message that says how to
fix it, which the server hands back to the model instead of failing.

Areas are "west,south,east,north" in degrees, the format of the geocode_place tool in
eve-esa/mcp-tool-registry, so the two can be chained.

The two heavy functions run the notebook's steps for any place and date. What was fixed for
Vesuvius in aoi.py is derived from the arguments instead: the UTM zone from the box, the two
windows from the fire dates, the elevation model from what covers the box.
"""

from __future__ import annotations

import hashlib
import math
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio import features
from rasterio.warp import transform_bounds
from scipy import ndimage
from shapely import union_all
from shapely.geometry import shape

from . import aoi, catalog, decision, export, indices, ingest, plots, reference, terrain

MAX_SIDE_DEG = 1.0  # searches only: at most 1 degree a side (about 110 km by 85 km here)
MAX_PIXEL_SIDE_DEG = 0.25  # tools that download pixels: at most 0.25 degree a side (about 28 by 21 km)
MAX_ROWS = 100  # longest list returned; a model reads a short list better than a long one
TOP_CELLS = 10  # cells listed in the answer; every cell is in the GeoJSON file

# The notebook's windows, as lengths: 1 July to 7 August is the 38 days before a fire that started
# on 8 August, and 13 August to 15 September the 34 days after one that ended on 12 August.
PRE_DAYS = 38
POST_DAYS = 34

BURN_BANDS = ["B04", "B8A", "B12", "SCL"]  # NBR needs B8A and B12, NDVI B04 and B8A, the mask SCL
NDVI_BANDS = ["B04", "B8A", "SCL"]
RUNS_DIR = Path("data/cache/runs")  # files written by the tools; not part of the repository
PIXEL_HA = aoi.RESOLUTION**2 / 10_000


class BadArgument(ValueError):
    """An argument the caller can fix. The message says what was wrong and what is expected."""


def parse_bbox(text: str, max_side_deg: float = MAX_SIDE_DEG) -> tuple[float, float, float, float]:
    """A box written "west,south,east,north" in decimal degrees, as a tuple of four floats."""
    try:
        west, south, east, north = (float(v) for v in text.split(","))
    except ValueError:
        raise BadArgument(f"bbox must be 'west,south,east,north' in decimal degrees, got {text!r}") from None
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise BadArgument(
            f"bbox {text!r} must have west < east and south < north, longitudes within -180 to 180 "
            "and latitudes within -90 to 90"
        )
    if east - west > max_side_deg or north - south > max_side_deg:
        raise BadArgument(f"bbox {text!r} is more than {max_side_deg:g} degree a side; give a smaller area")
    return west, south, east, north


def parse_period(start: str, end: str) -> tuple[str, str]:
    """Two ISO dates, YYYY-MM-DD, with start not after end."""
    days = []
    for name, text in (("start", start), ("end", end)):
        try:
            days.append(date.fromisoformat(text).isoformat())
        except ValueError:
            raise BadArgument(f"{name} must be a date written YYYY-MM-DD, got {text!r}") from None
    if days[0] > days[1]:
        raise BadArgument(f"start {days[0]} is after end {days[1]}")
    return days[0], days[1]


def parse_interval(text: str, name: str) -> tuple[str, str]:
    """A period written "YYYY-MM-DD/YYYY-MM-DD", the ISO 8601 interval that STAC searches use."""
    parts = text.split("/")
    if len(parts) != 2:
        raise BadArgument(f"{name} must be 'YYYY-MM-DD/YYYY-MM-DD', got {text!r}")
    try:
        return parse_period(*parts)
    except BadArgument as exc:
        raise BadArgument(f"{name}: {exc}") from None


def _box_text(west: float, south: float, east: float, north: float) -> str:
    return f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}"


def utm_crs(box: tuple[float, float, float, float]) -> str:
    """The UTM zone of the box centre as an EPSG code: metres, so pixels are 20 m squares.

    Zones are 6 degrees wide from 180 W; 326xx in the northern hemisphere, 327xx in the southern.
    The Norway and Svalbard exceptions are ignored.
    """
    lon, lat = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def pad_box(box: tuple[float, float, float, float], km: float) -> tuple[float, float, float, float]:
    """The box grown by ``km`` on every side (a degree of latitude is 111 km, of longitude less)."""
    west, south, east, north = box
    dlat = km / 111.0
    dlon = km / (111.0 * math.cos(math.radians((south + north) / 2)))
    return west - dlon, south - dlat, east + dlon, north + dlat


def fire_windows(fire_start: str, fire_end: str) -> tuple[tuple[str, str], tuple[str, str]]:
    """Pre-fire and post-fire windows from the fire dates, with the notebook's lengths."""
    start, end = date.fromisoformat(fire_start), date.fromisoformat(fire_end)
    pre = ((start - timedelta(days=PRE_DAYS)).isoformat(), (start - timedelta(days=1)).isoformat())
    post = ((end + timedelta(days=1)).isoformat(), (end + timedelta(days=POST_DAYS)).isoformat())
    return pre, post


def tinitaly_covers(box: tuple[float, float, float, float], path: Path = terrain.TINITALY_FILE) -> bool:
    """True when the TINITALY cut saved in the repository covers the whole box."""
    if not path.exists():
        return False
    with rasterio.open(path) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return west <= box[0] and south <= box[1] and east >= box[2] and north >= box[3]


def _run_key(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def _reflectance(items, box, crs: str, bands: list[str], key: str) -> xr.Dataset:
    """Download (or read from the cache), then mask and scale, as in step 3 of the notebook."""
    path = ingest.cache_path(f"run_{key}")
    dn = ingest.load_cached(path)
    if dn is None:
        dn = ingest.load_cube(items, bbox=box, bands=bands, crs=crs)
        ingest.save_cube(dn, path)
    return ingest.mask_and_scale(dn, catalog.offsets_by_day(items))


def _usable_dates(index: xr.DataArray, window: tuple[str, str]) -> int:
    """Median number of usable observations per pixel in a window."""
    return int(indices.window_count(index, *window).median())


def _slope(box, like: xr.DataArray) -> tuple[xr.DataArray, str]:
    """Slope on the cube grid, from TINITALY 10 m where the saved cut covers the box, else from
    Copernicus DEM GLO-30, as in step 8 of the notebook."""
    if tinitaly_covers(box):
        return terrain.slope_to_grid(terrain.load_tinitaly(like), like), "TINITALY 1.1, 10 m (INGV, CC BY 4.0)"
    dem = terrain.load_dem(catalog.search_dem(box), like)
    return terrain.slope_degrees(dem, aoi.RESOLUTION), "Copernicus DEM GLO-30, 30 m"


def _touches_edge(mask: xr.DataArray) -> bool:
    values = mask.values
    return bool(values[0].any() or values[-1].any() or values[:, 0].any() or values[:, -1].any())


def assess_burn(
    bbox: str,
    fire_start: str,
    fire_end: str,
    slope_threshold_deg: float = aoi.SLOPE_THRESHOLD_DEG,
    margin_km: float = 0.0,
) -> dict:
    """Steps 3 to 10 of the notebook for any fire: severity, slope, ranked 250 m cells, files."""
    box = parse_bbox(bbox, MAX_PIXEL_SIDE_DEG)
    fire_start, fire_end = parse_period(fire_start, fire_end)
    if not 10 <= slope_threshold_deg <= 45:
        raise BadArgument(f"slope_threshold_deg must be between 10 and 45 degrees, got {slope_threshold_deg}")
    if not 0 <= margin_km <= 5:
        raise BadArgument(f"margin_km must be between 0 and 5, got {margin_km}")
    box = pad_box(box, margin_km) if margin_km else box
    crs = utm_crs(box)
    pre, post = fire_windows(fire_start, fire_end)
    items = catalog.search_scenes(box, pre[0], post[1], aoi.MAX_CLOUD)
    days = {i.datetime.date().isoformat() for i in items}
    for label, (a, b) in (("pre-fire", pre), ("post-fire", post)):
        if not any(a <= d <= b for d in days):
            raise BadArgument(f"no Sentinel-2 scene under {aoi.MAX_CLOUD:g} % cloud in the {label} window "
                              f"{a} to {b}; check the dates, or look with list_scenes")
    key = _run_key("burn", _box_text(*box), pre[0], post[1], crs)
    refl = _reflectance(items, box, crs, BURN_BANDS, key)
    nbr, ndvi = indices.nbr(refl), indices.ndvi(refl)
    dnbr = indices.dnbr(nbr, pre, post)
    severity_all = indices.severity_class(dnbr)
    fire = indices.main_fire(severity_all)
    result = {
        "bbox": _box_text(*box),
        "crs": crs,
        "pre_fire_window": f"{pre[0]} to {pre[1]}",
        "post_fire_window": f"{post[0]} to {post[1]}",
        "usable_dates_per_pixel": {"pre_fire": _usable_dates(nbr, pre), "post_fire": _usable_dates(nbr, post)},
    }
    if not fire.values.any():
        return result | {"burned_ha": 0, "note": "No burned patch found (dNBR above 0.10). Check the box and the dates."}

    severity = severity_all.where(fire | (severity_all == 255), 0).astype("uint8")
    severe_codes = [code for code, name in indices.class_names().items() if name in aoi.SEVERE_CLASSES]
    severe = fire & severity.isin(severe_codes)
    slope, dem_name = _slope(box, dnbr)
    green = indices.window_median(ndvi, *post)
    threshold = float(slope_threshold_deg)
    cells = decision.score_cells(fire, severe, slope, green, threshold)
    offsets = [t - aoi.SLOPE_THRESHOLD_DEG for t in aoi.SLOPE_SENSITIVITY_DEG]  # the notebook's -3, 0, +3
    sens = decision.sensitivity(fire, severe, slope, green, tuple(threshold + d for d in offsets), threshold)
    grid = decision.cells_to_geodataframe(cells, crs)
    burned_ha = float(fire.sum()) * PIXEL_HA
    severe_ha = float(severe.sum()) * PIXEL_HA
    summary = decision.summary(
        grid, sens, "in the requested area", burned_ha=burned_ha, severe_ha=severe_ha, threshold_deg=threshold,
        green_unburned=float(green.where(abs(dnbr) < 0.1).median()),
    )

    out = RUNS_DIR / key
    export.write_cog(dnbr.astype("float32"), out / "dnbr_20m.tif", nodata=-9999.0, overview_resampling="average")
    export.write_cog(severity, out / "severity_20m.tif", nodata=255,
                     colormap=export.colormap_from_hex(plots.SEVERITY_COLOURS))
    export.write_geojson(export.main_fire_perimeter(fire), out / "main_fire_perimeter.geojson")
    cells_out = grid.round({"severe_steep_share": 3, "severe_steep_ha": 2, "severe_ha": 2, "burned_ha": 2,
                            "green_ndvi": 3})
    alerts = cells_out[cells_out["priority"] != decision.PRIORITY_LABELS[2]]
    export.write_geojson(alerts, out / "alert_cells.geojson")

    top = cells_out.head(TOP_CELLS)
    warnings = []
    if _touches_edge(fire):
        warnings.append("The burned patch touches the edge of the box, so part of the fire may lie outside: "
                        "call again with a larger box or a margin_km.")
    if dem_name.startswith("Copernicus"):
        warnings.append("Slope from a 30 m model finds less steep ground than a 10 m one (in the Vesuvius "
                        "notebook: 128 against 177 ha severe and steep at 23 degrees).")
    finer = "finer" if dem_name.startswith("TINITALY") else "coarser"
    method = {
        "severity": "dNBR = median NBR of the pre-fire window minus that of the post-fire window, clouds "
                    "masked; Key and Benson (2006) classes; the largest connected burned patch is the fire",
        "slope": f"{dem_name}, {finer} than the 20 m pixels; slope computed on the model's own grid "
                 "(Horn's method), then resampled to the 20 m pixels",
        "cells": "250 m squares ranked by the share of ground both burned at moderate or high severity "
                 "(dNBR 0.27 or more) and at least the slope threshold steep: the terrain term of the M1 "
                 "debris-flow model (Staley et al. 2017)",
    }
    limitations = [
        "An early assessment, weeks after the fire: severity can change as vegetation recovers.",
        "The class breaks come from North American forests (Key and Benson 2006); they are not calibrated here.",
        "The ranking is an order of priority, not a probability of debris flow: rainfall is not modelled.",
    ]
    if result["usable_dates_per_pixel"]["post_fire"] < 5:
        limitations.append(f"Only {result['usable_dates_per_pixel']['post_fire']} usable post-fire dates per "
                           "pixel (median), so the post-fire median rests on few images.")
    return result | {
        "burned_ha": round(burned_ha),
        "severity_ha": {k: v for k, v in indices.area_by_class(severity, aoi.RESOLUTION).items() if k != "unburned"},
        "severe_ha": round(severe_ha),
        "severe_and_steep_ha": round(float(cells["severe_steep_ha"].sum())),
        "slope_threshold_deg": threshold,
        "elevation_model": dem_name,
        "priority_1_cells": int((cells["priority"] == decision.PRIORITY_LABELS[0]).sum()),
        "priority_1_ha": round(float(cells.loc[cells["priority"] == decision.PRIORITY_LABELS[0], "severe_steep_ha"].sum())),
        "priority_2_cells": int((cells["priority"] == decision.PRIORITY_LABELS[1]).sum()),
        "top_cells": top[["rank", "priority", "severe_steep_share", "severe_steep_ha", "lat", "lon"]].to_dict("records"),
        "sensitivity": {str(k): {c: int(v) for c, v in row.items()} for k, row in sens.iterrows()},
        "summary": summary,
        "files": [p.as_posix() for p in sorted(out.iterdir())],
        "method": method,
        "limitations": limitations,
        "warnings": warnings,
    }


def _months(start: str, end: str) -> set[int]:
    """The calendar months a period touches."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    return {(first + timedelta(days=k)).month for k in range((last - first).days + 1)}


def drop_patches(change: xr.DataArray, min_drop: float, min_ha: float = 1.0) -> gpd.GeoDataFrame:
    """Connected patches (diagonals included) where NDVI fell by at least ``min_drop``, at least
    ``min_ha`` large, largest first, as polygons in the grid's CRS with area and mean change."""
    dropped = (change <= -min_drop).values
    labels, n = ndimage.label(dropped, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return gpd.GeoDataFrame({"hectares": [], "mean_change": []}, geometry=[], crs=change.rio.crs)
    ids = np.arange(1, n + 1)
    hectares = ndimage.sum(dropped, labels, ids) * PIXEL_HA
    means = ndimage.mean(change.values, labels, ids)
    keep = {int(i): (float(h), float(m)) for i, h, m in zip(ids, hectares, means) if h >= min_ha}
    polygons = {}
    for geom, value in features.shapes(labels.astype("int32"), mask=labels > 0, transform=change.rio.transform()):
        if int(value) in keep:
            polygons.setdefault(int(value), []).append(shape(geom))
    rows = [{"hectares": round(keep[i][0], 1), "mean_change": round(keep[i][1], 3), "geometry": union_all(parts)}
            for i, parts in polygons.items()]
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=change.rio.crs) if rows else gpd.GeoDataFrame(
        {"hectares": [], "mean_change": []}, geometry=[], crs=change.rio.crs)
    return out.sort_values("hectares", ascending=False).reset_index(drop=True)


def vegetation_change(bbox: str, period_a: str, period_b: str, min_drop: float = 0.1) -> dict:
    """Median NDVI in two periods, the change, and where it dropped by at least ``min_drop``."""
    box = parse_bbox(bbox, MAX_PIXEL_SIDE_DEG)
    a, b = parse_interval(period_a, "period_a"), parse_interval(period_b, "period_b")
    if a[1] >= b[0]:  # the change is later minus earlier, so the order decides its sign
        raise BadArgument(f"period_a must end before period_b starts, got {period_a} and {period_b}")
    if not 0.02 <= min_drop <= 0.5:
        raise BadArgument(f"min_drop is an NDVI difference between 0.02 and 0.5, got {min_drop}")
    crs = utm_crs(box)
    items_a = catalog.search_scenes(box, *a, aoi.MAX_CLOUD)
    items_b = catalog.search_scenes(box, *b, aoi.MAX_CLOUD)
    for name, period, items in (("period_a", a, items_a), ("period_b", b, items_b)):
        if not items:
            raise BadArgument(f"no Sentinel-2 scene under {aoi.MAX_CLOUD:g} % cloud in {name}, "
                              f"{period[0]} to {period[1]}; try a longer period")
    key = _run_key("ndvi", _box_text(*box), *a, *b, crs)
    refl = _reflectance(items_a + items_b, box, crs, NDVI_BANDS, key)
    ndvi = indices.ndvi(refl)
    before, after = indices.window_median(ndvi, *a), indices.window_median(ndvi, *b)
    change = (after - before).rename("NDVI_change").rio.write_crs(crs)
    valid = change.notnull()
    patches = drop_patches(change, min_drop)

    out = RUNS_DIR / key
    export.write_cog(change.astype("float32"), out / "ndvi_change_20m.tif", nodata=-9999.0,
                     overview_resampling="average")
    if len(patches):
        export.write_geojson(patches, out / "ndvi_drop_patches.geojson")
    centres = gpd.GeoSeries(patches.geometry.centroid, crs=crs).to_crs("EPSG:4326") if len(patches) else []
    listed = [
        {"hectares": row.hectares, "mean_change": row.mean_change, "lat": round(pt.y, 5), "lon": round(pt.x, 5)}
        for row, pt in zip(patches.head(5).itertuples(), list(centres)[:5])
    ]
    dropped_ha = float((change <= -min_drop).sum()) * PIXEL_HA
    notes = []
    if _months(*a) != _months(*b):
        notes.append("The two periods cover different months, so part of the change can be the season.")
    return {
        "bbox": _box_text(*box),
        "crs": crs,
        "period_a": f"{a[0]} to {a[1]}",
        "period_b": f"{b[0]} to {b[1]}",
        "scenes": {"period_a": len(items_a), "period_b": len(items_b)},
        "median_ndvi": {"period_a": round(float(before.where(valid).median()), 3),
                        "period_b": round(float(after.where(valid).median()), 3)},
        "area_with_data_ha": round(float(valid.sum()) * PIXEL_HA),
        "dropped_ha": round(dropped_ha),
        "dropped_share_pct": round(100 * dropped_ha / (float(valid.sum()) * PIXEL_HA), 1) if valid.any() else 0.0,
        "min_drop": min_drop,
        "largest_drop_patches": listed,
        "files": [p.as_posix() for p in sorted(out.iterdir())],
        "method": "median NDVI of each period per pixel, clouds masked; change = later minus earlier; "
                  "patches are connected pixels, diagonals included, where NDVI fell by at least min_drop, "
                  "1 ha or larger",
        "limitations": [
            ("NDVI measures green cover, not its cause: a harvest, drought, fire or new building "
             "all lower it."),
            *notes,
        ],
    }


def find_fires(bbox: str, start: str, end: str) -> dict:
    """The EFFIS fires that started inside ``bbox`` between ``start`` and ``end``, largest first,
    each with its dates, area and its own box."""
    box = parse_bbox(bbox)
    start, end = parse_period(start, end)
    fires = reference.fires_between(reference.fetch_effis(box), start, end)
    rows = []
    for fire in fires:
        p = fire["properties"]
        rows.append(
            {
                "effis_id": p["id"],
                "municipality": p["COMMUNE"],
                "start": p["FIREDATE"][:10],
                "end": p["FINALDATE"][:10],
                "hectares": round(float(p["AREA_HA"]), 1),
                "bbox": _box_text(*shape(fire["geometry"]).bounds),
            }
        )
    rows.sort(key=lambda r: r["hectares"], reverse=True)
    return {
        "source": "EFFIS burnt areas, Copernicus Emergency Management Service (European Commission, JRC)",
        "bbox": _box_text(*box),
        "start": start,
        "end": end,
        "fires_found": len(rows),
        "fires": rows[:MAX_ROWS],
    }


def list_scenes(bbox: str, start: str, end: str, max_cloud: float = 25.0) -> dict:
    """The Sentinel-2 L2A scenes over ``bbox`` between ``start`` and ``end`` with scene cloud cover
    below ``max_cloud`` percent, oldest first."""
    box = parse_bbox(bbox)
    start, end = parse_period(start, end)
    if not 0 < max_cloud <= 100:
        raise BadArgument(f"max_cloud is a percentage above 0 and at most 100, got {max_cloud}")
    table = catalog.scene_table(catalog.search_scenes(box, start, end, max_cloud))
    columns = ["date", "satellite", "orbit", "tile", "cloud_pct"]
    scenes = [] if table.empty else table[columns].to_dict("records")
    return {
        "collection": "sentinel-2-l2a, Microsoft Planetary Computer",
        "bbox": _box_text(*box),
        "start": start,
        "end": end,
        "max_cloud_pct": max_cloud,
        "products_found": len(scenes),
        "dates_found": len({s["date"] for s in scenes}),
        "scenes": scenes[:MAX_ROWS],
    }
