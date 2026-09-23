"""The EFFIS burnt-area polygons as an independent reference for the burned mask.

EFFIS (European Forest Fire Information System, run by the Joint Research Centre) maps fires
from MODIS and refines the outlines with Sentinel-2 at 20 m. It is a different team and method on
the same satellite family, so agreement tests the processing, not the sensor. It is not ground
truth: it gives an extent, not a severity.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rioxarray  # noqa: F401  (registers the .rio accessor used below)
import xarray as xr
from rasterio import features

EFFIS_WFS = "https://maps.effis.emergency.copernicus.eu/effis"
EFFIS_LAYER = "ms:modis.ba.poly"
REFERENCE_FILE = Path("data/reference/effis_burnt_areas.geojson")


def fetch_effis(bbox: tuple[float, float, float, float], timeout: float = 60) -> dict:
    """Every EFFIS burnt-area polygon that touches ``bbox`` (west, south, east, north), as GeoJSON.

    WFS 1.1.0 with EPSG:4326 expects latitude first, so the box is sent as south, west, north, east.
    """
    west, south, east, north = bbox
    params = {
        "service": "WFS",
        "request": "GetFeature",
        "version": "1.1.0",
        "typename": EFFIS_LAYER,
        "bbox": f"{south},{west},{north},{east},EPSG:4326",
        "outputformat": "geojson",
    }
    response = requests.get(EFFIS_WFS, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_effis(
    bbox: tuple[float, float, float, float], path: Path = REFERENCE_FILE, refresh: bool = False
) -> dict:
    """The EFFIS polygons from the file saved in the repository, fetched live only when missing.

    EFFIS revises its polygons and its service is sometimes down, so the notebook reads a fixed
    copy and records when it was fetched. ``refresh=True`` fetches again and overwrites the file.
    """
    if refresh or not path.exists():
        collection = fetch_effis(bbox)
        collection["fetched"] = datetime.now(UTC).date().isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(collection), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def fires_between(collection: dict, start: str, end: str) -> list[dict]:
    """The fires whose EFFIS start date lies in [start, end], ISO dates."""
    return [
        f for f in collection["features"] if start <= f["properties"]["FIREDATE"][:10] <= end
    ]


def fire_table(fires: list[dict]) -> pd.DataFrame:
    """One row per fire: EFFIS id, municipality, start and end date, area in hectares."""
    rows = [
        {
            "EFFIS id": f["properties"]["id"],
            "municipality": f["properties"]["COMMUNE"],
            "start": f["properties"]["FIREDATE"][:10],
            "end": f["properties"]["FINALDATE"][:10],
            "hectares": float(f["properties"]["AREA_HA"]),
        }
        for f in fires
    ]
    return pd.DataFrame(rows).set_index("EFFIS id")


def rasterise(fires: list[dict], like: xr.DataArray, crs: str = "EPSG:4326") -> xr.DataArray:
    """True on every pixel of ``like``'s grid whose centre falls inside one of the polygons.

    The polygons are reprojected from ``crs`` to the grid's CRS first, so the comparison is
    pixel against pixel on the same 20 m grid.
    """
    shapes = gpd.GeoDataFrame.from_features(fires, crs=crs).to_crs(like.rio.crs).geometry
    if len(shapes) == 0:
        return like.copy(data=np.zeros(like.shape, dtype=bool)).rename("reference")
    burned = features.rasterize(
        ((geom, 1) for geom in shapes),
        out_shape=like.shape,
        transform=like.rio.transform(),
        fill=0,
        dtype="uint8",
    )
    return like.copy(data=burned.astype(bool)).rename("reference")


def agreement(ours: xr.DataArray, ref: xr.DataArray, pixel_m: float) -> dict[str, float]:
    """Hectares burned in each map, in both, in only one, and intersection over union.

    IoU = both / (ours or reference): 1 when the two maps are identical, 0 when they do not touch.
    """
    a, b = ours.values.astype(bool), ref.values.astype(bool)
    ha = pixel_m * pixel_m / 10_000
    union = int((a | b).sum())
    return {
        "ours_ha": round(float(a.sum()) * ha, 1),
        "reference_ha": round(float(b.sum()) * ha, 1),
        "both_ha": round(float((a & b).sum()) * ha, 1),
        "only_ours_ha": round(float((a & ~b).sum()) * ha, 1),
        "only_reference_ha": round(float((~a & b).sum()) * ha, 1),
        "iou": round(float((a & b).sum()) / union, 2) if union else float("nan"),
    }
