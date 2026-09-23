"""
Integration test for the Burn Severity MCP Server.

Starts the server as a child process over stdio, connects with an MCP client and calls
every tool through the protocol, exactly as an agent would: list the tools, call each one,
and send one wrong argument to check that the error comes back as JSON.

Needs the network (EFFIS, Planetary Computer), no credentials. Not part of pytest; the
unit tests in tests/test_api.py cover the same logic offline.

Usage:
    python servers/burn_severity/test.py            # all four tools; the first run downloads pixels
    python servers/burn_severity/test.py --quick    # only find_fires and list_scenes
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PY = Path(__file__).resolve().parent / "server.py"
VESUVIUS = "14.35,40.77,14.50,40.87"


async def call(session: ClientSession, tool: str, **arguments) -> dict:
    result = json.loads((await session.call_tool(tool, arguments)).content[0].text)
    if "error" in result:
        print(f"\n{tool} returned an error: {result['error']}")
    return result


async def main() -> int:
    server = StdioServerParameters(command=sys.executable, args=[str(SERVER_PY), "--transport", "stdio"])
    failures = 0
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        tools = (await session.list_tools()).tools
        print("tools/list:")
        for tool in tools:
            arguments = ", ".join(tool.inputSchema["properties"])
            print(f"  {tool.name}({arguments}): {tool.description.strip().splitlines()[0]}")

        fires = await call(session, "find_fires", bbox=VESUVIUS, start="2025-07-01", end="2025-09-15")
        print(f"\nfind_fires: {fires.get('fires_found', 0)} fires")
        for fire in fires.get("fires", []):
            print(f"  {fire['start']} to {fire['end']}  {fire['hectares']:>7} ha  {fire['municipality']}  bbox {fire['bbox']}")
        failures += not fires.get("fires_found")

        scenes = await call(session, "list_scenes", bbox=VESUVIUS, start="2025-08-01", end="2025-08-31")
        print(f"\nlist_scenes: {scenes.get('products_found', 0)} products on {scenes.get('dates_found', 0)} dates")
        for scene in scenes.get("scenes", [])[:5]:
            print(f"  {scene['date']}  {scene['satellite']}  orbit {scene['orbit']}  cloud {scene['cloud_pct']} %")
        failures += not scenes.get("products_found")

        wrong = await call(session, "find_fires", bbox="14.50,40.77,14.35,40.87", start="2025-07-01", end="2025-09-15")
        print(f"\nwrong bbox (west and east swapped): {wrong}")
        failures += "error" not in wrong

        if "--quick" not in sys.argv:
            failures += await heavy(session)

    print("\nOK" if failures == 0 else f"\n{failures} check(s) failed")
    return failures


async def heavy(session: ClientSession) -> int:
    """The two tools that download pixels. On the notebook's box and dates, assess_burn must give
    the notebook's numbers."""
    failures = 0
    burn = await call(session, "assess_burn", bbox=VESUVIUS, fire_start="2025-08-08", fire_end="2025-08-12")
    shown = {k: burn.get(k) for k in ("pre_fire_window", "post_fire_window", "burned_ha", "severe_ha",
                                       "severe_and_steep_ha", "priority_1_cells", "priority_1_ha",
                                       "priority_2_cells", "elevation_model")}
    print("\nassess_burn:", json.dumps(shown, indent=2))
    print("summary:", burn.get("summary"))
    notebook = {"burned_ha": 737, "severe_ha": 576, "severe_and_steep_ha": 177, "priority_1_cells": 24,
                "priority_1_ha": 110, "priority_2_cells": 16}
    for key, value in notebook.items():
        if burn.get(key) != value:
            print(f"  differs from the notebook: {key} = {burn.get(key)}, notebook {value}")
            failures += 1

    change = await call(session, "vegetation_change", bbox=VESUVIUS,
                        period_a="2025-07-01/2025-08-07", period_b="2025-08-13/2025-09-15")
    print("\nvegetation_change:", json.dumps({k: change.get(k) for k in ("median_ndvi", "dropped_ha",
                                                                         "dropped_share_pct", "notes")}, indent=2))
    for patch in change.get("largest_drop_patches", [])[:3]:
        print(f"  {patch['hectares']:>7} ha  mean change {patch['mean_change']}  at {patch['lat']}, {patch['lon']}")
    failures += not change.get("largest_drop_patches")
    return failures


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
