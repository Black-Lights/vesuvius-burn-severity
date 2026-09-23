"""The agent in the notebook: run the conversations when a key is set, otherwise replay the run
saved by the last live execution, and show each turn readably."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

from .settings import why_not_live


async def run_or_replay(conversations: list[tuple[str | None, list[str]]], path: Path,
                        provider: str | None = None) -> dict:
    """Each conversation is (reader, questions). With a key in .env the turns run live and are saved
    to ``path``; without one, the saved turns are read back, so the notebook runs anywhere."""
    reason = why_not_live(provider)
    if reason:
        record = json.loads(path.read_text(encoding="utf-8"))
        return record | {"replayed": True, "reason": reason}
    turns = await _on_agent_loop(_converse(conversations, provider))
    record = {"recorded": datetime.now(UTC).date().isoformat(), "turns": turns}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return record | {"replayed": False}


_LOOP: asyncio.AbstractEventLoop | None = None


async def _on_agent_loop(coroutine) -> list[dict]:
    """Run a live conversation on one background event loop, the same for the whole session.

    The Jupyter kernel's own loop on Windows cannot start the MCP server as a child process, so
    the conversations run on a loop of the platform's own kind in a background thread. It stays
    open between cells because the model clients keep their connections, and a connection
    belongs to the loop that opened it: a fresh loop per cell failed with "Event loop is closed".
    """
    global _LOOP
    if _LOOP is None:
        _LOOP = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        threading.Thread(target=_LOOP.run_forever, name="agent-loop", daemon=True).start()
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coroutine, _LOOP))


async def _converse(conversations, provider) -> list[dict]:
    from .run import Conversation  # the agent packages, needed only for a live run

    turns = []
    for number, (reader, questions) in enumerate(conversations, start=1):
        async with Conversation(provider, reader=reader) as chat:
            for question in questions:
                turns.append({"conversation": number, **(await chat.ask(question))})
    return turns


async def compare_or_replay(question: str, providers: list[str], path: Path, reader: str | None = None) -> dict:
    """The same question to several models, one conversation each. Live when every provider has a
    key and the agent packages are installed, and saved to ``path``; otherwise the saved run."""
    reason = next((r for r in map(why_not_live, providers) if r), None)
    if reason:
        record = json.loads(path.read_text(encoding="utf-8"))
        return record | {"replayed": True, "reason": reason}
    turns = await _on_agent_loop(_each_model(question, providers, reader))
    record = {"recorded": datetime.now(UTC).date().isoformat(), "question": question, "turns": turns}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return record | {"replayed": False}


async def _each_model(question, providers, reader) -> list[dict]:
    from .run import Conversation

    turns = []
    for provider in providers:
        async with Conversation(provider, reader=reader) as chat:
            turns.append(await chat.ask(question))
    return turns


def show_models(record: dict) -> pd.DataFrame:
    """Each model's answer, then a table: calls made, the arguments of assess_burn, what the tool
    found, time, tokens and the number check."""
    source = (f"Replayed: the run recorded on {record['recorded']} ({record['reason']})." if record["replayed"]
              else f"Run live on {record['recorded']}.")
    display(Markdown(f"_{source}_ **Question:** {record['question']}"))
    rows = []
    for t in record["turns"]:
        calls = [c for s in t["steps"] if "tool_calls" in s for c in s["tool_calls"]]
        burn_call = next((c["args"] for c in calls if c["name"] == "assess_burn"), {})
        burn = next((json.loads(s["content"]) for s in t["steps"] if s.get("name") == "assess_burn"), {})
        answer = "\n".join(f"> {line}" for line in t["answer"].splitlines())
        display(Markdown(f"**{t['model']}**\n\n{answer}"))
        rows.append({
            "model": t["model"],
            "tool calls": " > ".join(c["name"] for c in calls),
            "assess_burn dates": f"{burn_call.get('fire_start', '')} to {burn_call.get('fire_end', '')}",
            "margin (km)": burn_call.get("margin_km", 0),
            "burned (ha)": burn.get("burned_ha"),
            "first-priority cells": burn.get("priority_1_cells"),
            "seconds": t["seconds"],
            "tokens": t["tokens"],
            "numbers in the answer": t["check"]["numbers_checked"],
            "not in the tool results": len(t["check"]["not_found"]),
        })
    return pd.DataFrame(rows).set_index("model")


def check_sentence(check: dict) -> str:
    """The number check in words: how many numbers the answer has, and which were not found in
    the question, the tool arguments or the tool results."""
    n, missing = check["numbers_checked"], check["not_found"]
    if not missing:
        return f"Check: all {n} numbers in the answer were found in the tool results."
    return (f"Check: {n - len(missing)} of the {n} numbers in the answer were found in the tool results; "
            f"not found: {', '.join(missing)}.")


def brief(name: str, content: str) -> str:
    """One line on what a tool returned: the numbers that matter, not the whole JSON."""
    r = json.loads(content)
    if "error" in r:
        return f"error: {r['error']}"
    if name == "find_fires":
        fires = "; ".join(f"{f['municipality']} {f['start']} to {f['end']}, {f['hectares']:g} ha" for f in r["fires"][:3])
        return f"{r['fires_found']} fires: {fires}"
    if name == "list_scenes":
        return f"{r['products_found']} products on {r['dates_found']} dates"
    if name == "assess_burn" and "severe_ha" in r:
        return (f"windows {r['pre_fire_window']} and {r['post_fire_window']}; {r['burned_ha']} ha burned, "
                f"{r['severe_ha']} ha severe, {r['severe_and_steep_ha']} ha severe and steep; "
                f"{r['priority_1_cells']} first-priority cells ({r['priority_1_ha']} ha), {r['priority_2_cells']} next; "
                f"slope from {r['elevation_model']}; {len(r['warnings'])} warning(s)")
    if name == "vegetation_change":
        top = r["largest_drop_patches"][0] if r["largest_drop_patches"] else None
        where = f"; largest patch {top['hectares']} ha at {top['lat']} N, {top['lon']} E" if top else ""
        return (f"median NDVI {r['median_ndvi']['period_a']} then {r['median_ndvi']['period_b']}; "
                f"{r['dropped_ha']} ha dropped by {r['min_drop']} or more ({r['dropped_share_pct']} %){where}")
    return ", ".join(f"{k}: {v}" for k, v in list(r.items())[:6])


def show_turns(record: dict) -> pd.DataFrame:
    """Every turn as question, tool calls with a line on each result, answer and check; then a
    table of all turns."""
    source = (f"Replayed: the run recorded on {record['recorded']} ({record['reason']})." if record["replayed"]
              else f"Run live on {record['recorded']}.")
    display(Markdown(f"_{source}_"))
    rows = []
    for number, t in enumerate(record["turns"], start=1):
        lines = [f"**Turn {number}** (conversation {t['conversation']}, reader: {t['reader']})  ",
                 f"**Question:** {t['steps'][0]['content']}", ""]
        calls = [c for s in t["steps"] if "tool_calls" in s for c in s["tool_calls"]]
        results = [s for s in t["steps"] if s["role"] == "tool"]
        for call, result in zip(calls, results):
            args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
            lines.append(f"- `{call['name']}({args})`: {brief(result['name'], result['content'])}")
        if not calls:
            lines.append("- no tool call: answered from the earlier turns")
        lines += [s["content"] for s in t["steps"] if s["role"] == "check"]
        answer = "\n".join(f"> {line}" for line in t["answer"].splitlines())
        stats = f"_{check_sentence(t['check'])} {t['model']}, {t['seconds']} s, {t['tokens']:,} tokens._"
        lines += ["", answer, "", stats]
        display(Markdown("\n".join(lines)))
        rows.append({"turn": number, "conversation": t["conversation"], "reader": t["reader"],
                     "tool calls": len(calls), "seconds": t["seconds"], "tokens": t["tokens"],
                     "numbers in the answer": t["check"]["numbers_checked"],
                     "not in the tool results": len(t["check"]["not_found"])})
    return pd.DataFrame(rows).set_index("turn")
