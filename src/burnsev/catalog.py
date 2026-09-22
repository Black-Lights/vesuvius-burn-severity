"""STAC search on Microsoft Planetary Computer and scene metadata.

Planetary Computer serves Sentinel-2 L2A as cloud-optimised GeoTIFFs behind a STAC API.
Searching needs no account; reading an asset needs its URL signed, which
``planetary_computer.sign`` does at load time.
"""

from __future__ import annotations

import pandas as pd
import pystac
import pystac_client

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"
DEM_COLLECTION = "cop-dem-glo-30"


def open_catalog() -> pystac_client.Client:
    return pystac_client.Client.open(STAC_URL)


def search_scenes(
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    max_cloud: float,
) -> list[pystac.Item]:
    """Return Sentinel-2 L2A items over ``bbox`` between ``start`` and ``end`` (inclusive
    dates, ISO strings) with scene cloud cover below ``max_cloud`` percent, oldest first.

    The cloud filter is applied here, in Python, rather than in the request. The STAC query
    extension is optional and Planetary Computer does not advertise it (pystac-client warns),
    while the date range alone returns a few dozen items at most, so filtering the returned
    metadata is simpler and works against any STAC API.
    """
    catalog = open_catalog()
    search = catalog.search(
        collections=[S2_COLLECTION],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
    )
    items = [
        item
        for item in search.items()
        if float(item.properties.get("eo:cloud_cover", 100.0)) < max_cloud
    ]
    return newest_per_acquisition(items)


def newest_per_acquisition(items: list[pystac.Item]) -> list[pystac.Item]:
    """Keep one product per acquisition time, sorted oldest first.

    The same acquisition can be published twice: a reprocessed product keeps the acquisition
    time and gets a later generation time. Loading both would count that date twice in the
    median composite. The product id ends with the generation time, so the highest id per
    acquisition is the newest product.
    """
    newest: dict[object, pystac.Item] = {}
    for item in items:
        key = item.datetime
        if key not in newest or item.id > newest[key].id:
            newest[key] = item
    return sorted(newest.values(), key=lambda i: i.datetime)


def scene_table(items: list[pystac.Item]) -> pd.DataFrame:
    """One row per scene with the fields worth checking before trusting the cube."""
    rows = []
    for item in items:
        p = item.properties
        rows.append(
            {
                "id": item.id,
                "date": item.datetime.date().isoformat(),
                "satellite": p.get("platform"),
                "orbit": p.get("sat:relative_orbit"),
                "tile": p.get("s2:mgrs_tile"),
                "cloud_pct": round(float(p.get("eo:cloud_cover", float("nan"))), 1),
                "baseline": p.get("s2:processing_baseline"),
                "boa_offset": boa_offset(item),
            }
        )
    return pd.DataFrame(rows)


def boa_offset(item: pystac.Item) -> int:
    """Additive offset to apply to L2A digital numbers before scaling to reflectance.

    ESA changed the L2A format with processing baseline 04.00 (25 January 2022): every
    reflectance band carries BOA_ADD_OFFSET = -1000, so that reflectance =
    (DN + offset) / 10000. Scenes from older baselines have no offset. Skipping this shifts
    every index by a scene-dependent amount, so it is read here, once, per scene.

    The value is taken from the asset's ``raster:bands`` metadata when the catalogue
    provides it, and otherwise from the processing baseline, which is the documented rule.
    """
    asset = item.assets.get("B04")
    if asset is not None:
        bands = asset.extra_fields.get("raster:bands") or []
        if bands and "offset" in bands[0]:
            # raster:bands gives the offset in reflectance units (e.g. -0.1); convert to DN.
            return round(float(bands[0]["offset"]) * 10000)
    baseline = str(item.properties.get("s2:processing_baseline", "00.00"))
    try:
        major = float(baseline)
    except ValueError:
        major = 0.0
    return -1000 if major >= 4.0 else 0


def search_dem(bbox: tuple[float, float, float, float]) -> list[pystac.Item]:
    """Copernicus DEM GLO-30 tiles covering the bbox (usually one)."""
    catalog = open_catalog()
    return list(catalog.search(collections=[DEM_COLLECTION], bbox=list(bbox)).items())
