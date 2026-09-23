"""
Burn Severity MCP Server
========================
The burnsev pipeline (the notebook's code) as MCP tools, in the layout of
eve-esa/mcp-tool-registry. The logic lives in ``burnsev.api``; this file only turns
it into tools and JSON.

Tools:
    find_fires   - EFFIS burnt areas in an area and period: dates, hectares, own bbox
    list_scenes  - Sentinel-2 L2A scenes over an area and period, with cloud cover

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
import sys

from mcp.server.fastmcp import FastMCP

# Importing burnsev also removes a system-wide PROJ_LIB before rasterio loads (see its __init__).
from burnsev import api

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
