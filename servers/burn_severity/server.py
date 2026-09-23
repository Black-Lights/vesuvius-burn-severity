"""
Burn Severity MCP Server
========================
The burnsev pipeline (the notebook's code) as MCP tools, in the layout of
eve-esa/mcp-tool-registry. The logic lives in ``burnsev.api``; this file only turns
it into tools and JSON.

Tools:
    find_fires         - EFFIS burnt areas in an area and period: dates, hectares, own bbox
    list_scenes        - Sentinel-2 L2A scenes over an area and period, with cloud cover
    assess_burn        - the notebook for any fire: severity, slope, ranked cells, files
    vegetation_change  - median NDVI in two periods and where it dropped

Usage:
    python server.py --transport stdio                 # local clients, started as a child process
    python server.py --transport http --port 8000      # HTTP transport

Needs no credentials: EFFIS and Microsoft Planetary Computer are open.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Importing burnsev also removes a system-wide PROJ_LIB before rasterio loads (see its __init__).
from burnsev import api

# The pipeline reads data/reference and writes data/cache relative to the repository, whichever
# folder the client starts the server from (an agent, the MCP Inspector, Claude Code).
os.chdir(Path(__file__).resolve().parents[2])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,  # MCP stdio transport: stdout carries the protocol, so logs go to stderr
)
logger = logging.getLogger("burn-severity-mcp")

mcp = FastMCP("Burn Severity Server", host="0.0.0.0", port=8000, stateless_http=True)


async def _run(fn, **kwargs) -> str:
    """Run a pipeline function and return its result as JSON.

    The functions download and compute, so they run in a worker thread and the server stays
    responsive. A wrong argument or a failed service comes back as {"error": ...}, which the
    model can read and act on, instead of breaking the session.
    """
    try:
        result = await asyncio.to_thread(fn, **kwargs)
    except api.BadArgument as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        logger.exception("%s failed", fn.__name__)
        return json.dumps({"error": f"{fn.__name__} failed: {exc}"})
    return json.dumps(result, default=str)


@mcp.tool()
async def find_fires(bbox: str, start: str, end: str) -> str:
    """Find the fires that burned in an area during a period, from the EFFIS burnt-area database.

    Use it when the user names a place and a time but not the exact fire dates. Each fire comes
    with its start and end date, its area and its own bbox.

    Args:
        bbox: Area as "west,south,east,north" in decimal degrees (WGS 84), at most 1 degree a
            side. Example: "14.35,40.77,14.50,40.87" is Mount Vesuvius.
        start: First day of the period, YYYY-MM-DD. A fire is listed if it started in the period.
        end: Last day of the period, YYYY-MM-DD.

    Returns:
        JSON with the fires, largest first: effis_id, municipality, start, end, hectares, bbox.
    """
    return await _run(api.find_fires, bbox=bbox, start=start, end=end)


@mcp.tool()
async def list_scenes(bbox: str, start: str, end: str, max_cloud: float = 25.0) -> str:
    """List the Sentinel-2 L2A images of an area during a period, with their cloud cover.

    Use it to check that clear images exist before and after an event.

    Args:
        bbox: Area as "west,south,east,north" in decimal degrees (WGS 84), at most 1 degree a
            side.
        start: First day, YYYY-MM-DD.
        end: Last day, YYYY-MM-DD.
        max_cloud: Keep scenes with less than this percentage of cloud over the whole scene
            (default 25).

    Returns:
        JSON with the number of products and dates found and, oldest first, each scene's date,
        satellite, orbit, tile and cloud percentage.
    """
    return await _run(api.list_scenes, bbox=bbox, start=start, end=end, max_cloud=max_cloud)


@mcp.tool()
async def assess_burn(
    bbox: str,
    fire_start: str,
    fire_end: str,
    slope_threshold_deg: float = 23.0,
    margin_km: float = 0.0,
) -> str:
    """Map how badly a fire burned and rank where to act first against erosion before the rains.

    Downloads Sentinel-2 L2A over the area and compares the median NBR of the 38 days before the
    fire with that of the 34 days after it (dNBR). Classes severity (Key and Benson 2006), keeps
    the main burned patch, adds slope, and ranks 250 m cells by the share of ground that is both
    burned at moderate or high severity and at least slope_threshold_deg steep (the terrain term
    of the USGS M1 debris-flow model, Staley et al. 2017). Takes one to a few minutes the first
    time for an area; a repeated call is read from the cache.

    Args:
        bbox: Area around the whole fire as "west,south,east,north" in decimal degrees (WGS 84),
            at most 0.25 degree a side. The bbox from find_fires hugs the burn: use it with
            margin_km=1.
        fire_start: First day of the fire, YYYY-MM-DD (find_fires gives it).
        fire_end: Last day of the fire, YYYY-MM-DD.
        slope_threshold_deg: Slope from the horizontal that counts as steep, default 23 degrees
            (a 42 % grade), the value of the M1 model.
        margin_km: Grow the bbox by this many km on every side, default 0.

    Returns:
        JSON with the windows used, burned and severe hectares, hectares per severity class,
        severe and steep hectares, the number of first and second priority cells, the top 10
        cells with latitude and longitude, how the first list changes at 3 degrees less and
        more, a plain-language summary, the files written (Cloud-Optimised GeoTIFF and GeoJSON)
        and warnings.
    """
    return await _run(api.assess_burn, bbox=bbox, fire_start=fire_start, fire_end=fire_end,
                      slope_threshold_deg=slope_threshold_deg, margin_km=margin_km)


@mcp.tool()
async def vegetation_change(bbox: str, period_a: str, period_b: str, min_drop: float = 0.1) -> str:
    """Compare vegetation in an area between two periods and say where it dropped.

    Uses the per-pixel median NDVI of each period from Sentinel-2 L2A, clouds masked. Reports
    the median NDVI of each period, the hectares where NDVI fell by at least min_drop, and the
    largest patches of loss with their centres. Compare the same months of two years, otherwise
    the season changes NDVI too. Takes one to a few minutes the first time for an area.

    Args:
        bbox: Area as "west,south,east,north" in decimal degrees (WGS 84), at most 0.25 degree
            a side.
        period_a: The earlier period as "YYYY-MM-DD/YYYY-MM-DD", for example
            "2022-06-01/2022-08-31" for summer 2022.
        period_b: The later period, same format.
        min_drop: Smallest fall in NDVI that counts as a drop, default 0.1.

    Returns:
        JSON with the scenes used, the median NDVI of each period, the hectares and share that
        dropped, the five largest patches of drop (hectares, mean change, latitude, longitude),
        the files written and notes.
    """
    return await _run(api.vegetation_change, bbox=bbox, period_a=period_a, period_b=period_b,
                      min_drop=min_drop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Burn Severity MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="http",
        help="Transport type (default: http, as in eve-esa/mcp-tool-registry; stdio for local clients)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transport")
    args = parser.parse_args()

    mcp.settings.port = args.port

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http")
