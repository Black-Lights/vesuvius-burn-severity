"""The pipeline as plain functions for the MCP server (bonus A).

Small JSON-ready values in and out, so a language model can pass them and read them. Every
argument is checked first; a wrong one raises ``BadArgument`` with a message that says how to
fix it, which the server hands back to the model instead of failing.

Areas are "west,south,east,north" in degrees, the format of the geocode_place tool in
eve-esa/mcp-tool-registry, so the two can be chained.
"""

from __future__ import annotations

from datetime import date

from shapely.geometry import shape

from . import catalog, reference

MAX_SIDE_DEG = 1.0  # at most 1 degree a side (about 110 km by 85 km here): searches stay quick
MAX_ROWS = 100  # longest list returned; a model reads a short list better than a long one


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


def _box_text(west: float, south: float, east: float, north: float) -> str:
    return f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}"


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
