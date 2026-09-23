"""
Integration test for the Burn Severity MCP Server.

Starts the server as a child process over stdio, connects with an MCP client and calls
every tool through the protocol, exactly as an agent would: list the tools, call each one,
and send one wrong argument to check that the error comes back as JSON.

Needs the network (EFFIS, Planetary Computer), no credentials. Not part of pytest; the
unit tests in tests/test_api.py cover the same logic offline.

Usage:
    python servers/burn_severity/test.py
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
    result = await session.call_tool(tool, arguments)
    return json.loads(result.content[0].text)


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
        print(f"\nfind_fires: {fires['fires_found']} fires")
        for fire in fires["fires"]:
            print(f"  {fire['start']} to {fire['end']}  {fire['hectares']:>7} ha  {fire['municipality']}  bbox {fire['bbox']}")
        failures += fires["fires_found"] == 0

        scenes = await call(session, "list_scenes", bbox=VESUVIUS, start="2025-08-01", end="2025-08-31")
        print(f"\nlist_scenes: {scenes['products_found']} products on {scenes['dates_found']} dates")
        for scene in scenes["scenes"][:5]:
            print(f"  {scene['date']}  {scene['satellite']}  orbit {scene['orbit']}  cloud {scene['cloud_pct']} %")
        failures += scenes["products_found"] == 0

        wrong = await call(session, "find_fires", bbox="14.50,40.77,14.35,40.87", start="2025-07-01", end="2025-09-15")
        print(f"\nwrong bbox (west and east swapped): {wrong}")
        failures += "error" not in wrong

    print("\nOK" if failures == 0 else f"\n{failures} check(s) failed")
    return failures


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
