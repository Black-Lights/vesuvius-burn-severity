"""STAC search on Microsoft Planetary Computer and scene metadata.

Planetary Computer serves Sentinel-2 L2A as cloud-optimised GeoTIFFs behind a STAC API.
Searching needs no account; reading an asset needs its URL signed, which
``planetary_computer.sign`` does at load time.
"""

from __future__ import annotations

import pandas as pd
import pystac
import pystac_client

from . import aoi

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

    Products are returned as listed, one row per product. The same solar day can hold two
    products (two orbits ten minutes apart, or one acquisition published twice); the load
    step groups by solar day, so a day counts once in the median either way.
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
    items.sort(key=lambda i: i.datetime)
    return items


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
                "boa_add_offset": boa_add_offset(item),
            }
        )
    return pd.DataFrame(rows)


def boa_add_offset(item: pystac.Item) -> int:
    """ESA BOA_ADD_OFFSET in digital numbers, added before dividing by BOA_QUANTIFICATION_VALUE.

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
            # raster:bands states the same offset in reflectance units (-0.1 = -1000 / 10000);
            # convert it to digital numbers.
            return round(float(bands[0]["offset"]) * aoi.BOA_QUANTIFICATION_VALUE)
    baseline = str(item.properties.get("s2:processing_baseline", "00.00"))
    try:
        major = float(baseline)
    except ValueError:
        major = 0.0
    return -1000 if major >= 4.0 else 0


def offsets_by_day(items: list[pystac.Item]) -> dict[str, int]:
    """Map each acquisition date (ISO string) to its BOA offset in DN.

    The cube has one time step per solar day, so the offset must be known per day. At this
    longitude a morning pass has the same date in solar and UTC time. If two products of
    one day disagree (different baselines), the day is refused rather than guessed.
    """
    out: dict[str, int] = {}
    for item in items:
        day = item.datetime.date().isoformat()
        offset = boa_add_offset(item)
        if out.get(day, offset) != offset:
            raise ValueError(f"{day}: products with different offsets {out[day]} and {offset}")
        out[day] = offset
    return out


def orbits_by_day(items: list[pystac.Item]) -> dict[str, set[int]]:
    """Map each acquisition date (ISO string) to the relative orbits that saw the box that day."""
    out: dict[str, set[int]] = {}
    for item in items:
        day = item.datetime.date().isoformat()
        out.setdefault(day, set()).add(int(item.properties["sat:relative_orbit"]))
    return out


def search_dem(bbox: tuple[float, float, float, float]) -> list[pystac.Item]:
    """Copernicus DEM GLO-30 tiles covering the bbox (usually one)."""
    catalog = open_catalog()
    return list(catalog.search(collections=[DEM_COLLECTION], bbox=list(bbox)).items())
