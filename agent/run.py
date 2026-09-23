"""Ask the agent one question.

Starts the MCP server as a child process over stdio, loads its tools as LangChain tools, runs the
graph and returns a transcript: every message, the tool calls with their arguments, the tool
results, the answer and the number check.

The model is any chat model served in the OpenAI format, chosen in .env (see .env.example):
LLM_PROVIDER picks DeepSeek, Kimi or OpenAI, LLM_MODEL overrides the model. LLM_PROVIDER=custom
uses LLM_BASE_URL, LLM_API_KEY and LLM_MODEL, which is how a self-hosted model such as
EVE-Instruct, served through an OpenAI-compatible endpoint, would be plugged in.

Usage:
    python -m agent.run "Which burned slopes of the Vesuvius fire of August 2025 need work first?"
    python -m agent.run "..." --provider kimi --reader expert --save outputs/agent_run.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .graph import build_graph, is_feedback, sources, text_of
from .grounding import check

SERVER = Path(__file__).resolve().parents[1] / "servers" / "burn_severity" / "server.py"

# name: (base URL, key variable, default model); None is OpenAI's own endpoint
PROVIDERS = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-flash"),  # V4.1 Flash
    "kimi": ("https://api.moonshot.ai/v1", "KIMI_API_KEY", "kimi-k3"),
    "openai": (None, "OPENAI_API_KEY", "gpt-5.4-mini"),
}


def setting(name: str) -> str | None:
    """A value from the environment, without an inline comment; None when empty.

    python-dotenv reads `LLM_MODEL=   # a note` as the note itself, so the comment is cut here.
    """
    value = os.environ.get(name, "").split("#")[0].strip()
    return value or None


def make_llm(provider: str | None = None, model: str | None = None) -> tuple[ChatOpenAI, str, str]:
    """The chat model named in .env or in the arguments, with its provider and model name."""
    load_dotenv(override=False)
    provider = provider or setting("LLM_PROVIDER") or "deepseek"
    if provider == "custom":
        base_url, key, default = setting("LLM_BASE_URL"), setting("LLM_API_KEY"), None
    elif provider in PROVIDERS:
        base_url, key_var, default = PROVIDERS[provider]
        key = setting(key_var)
    else:
        raise ValueError(f"LLM_PROVIDER must be one of {', '.join([*PROVIDERS, 'custom'])}, got {provider!r}")
    if not key:
        raise ValueError(f"no API key for {provider}: set it in .env (see .env.example)")
    name = model or setting("LLM_MODEL") or default
    return ChatOpenAI(model=name, base_url=base_url, api_key=key, timeout=180, max_retries=2), provider, name


def transcript(messages: list, provider: str, model: str, reader: str | None, seconds: float) -> dict:
    """The run as plain data: what was asked, called, returned and answered, and the check."""
    steps = []
    for m in messages:
        if is_feedback(m):
            steps.append({"role": "check", "content": text_of(m)})
        elif isinstance(m, HumanMessage):
            steps.append({"role": "user", "content": text_of(m)})
        elif isinstance(m, AIMessage) and m.tool_calls:
            steps.append({"role": "assistant", "tool_calls": [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]})
        elif isinstance(m, AIMessage):
            steps.append({"role": "assistant", "content": text_of(m)})
        elif isinstance(m, ToolMessage):
            steps.append({"role": "tool", "name": m.name, "content": text_of(m)})
    answer = next(s["content"] for s in reversed(steps) if s["role"] == "assistant" and "content" in s)
    tokens = sum((m.usage_metadata or {}).get("total_tokens", 0) for m in messages if isinstance(m, AIMessage))
    return {
        "provider": provider,
        "model": model,
        "reader": reader or "judged from the question",
        "seconds": round(seconds, 1),
        "tokens": tokens,
        "steps": steps,
        "answer": answer,
        "check": check(answer, sources(messages[:-1])),
    }


async def ask(question: str, provider: str | None = None, model: str | None = None,
              reader: str | None = None) -> dict:
    """Run one question through the agent and the MCP server; return the transcript.

    ``reader`` is "public" (plain words), "expert" (technical terms) or None (judged from the
    question). It changes the words of the answer, never its numbers.
    """
    llm, provider, model = make_llm(provider, model)
    server = StdioServerParameters(command=sys.executable, args=[str(SERVER), "--transport", "stdio"])
    started = time.perf_counter()
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        graph = build_graph(llm, await load_mcp_tools(session), reader)
        state = await graph.ainvoke({"messages": [HumanMessage(question)], "checks": 0, "not_found": []},
                                    {"recursion_limit": 20})  # at most about nine tool rounds
    return transcript(state["messages"], provider, model, reader, time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the burn severity agent one question")
    parser.add_argument("question")
    parser.add_argument("--provider", choices=[*PROVIDERS, "custom"])
    parser.add_argument("--model")
    parser.add_argument("--reader", choices=["public", "expert"], help="default: judged from the question")
    parser.add_argument("--save", type=Path, help="write the transcript as JSON")
    args = parser.parse_args()
    run = asyncio.run(ask(args.question, args.provider, args.model, args.reader))
    for step in run["steps"]:
        body = step.get("content") or json.dumps(step.get("tool_calls"))
        print(f"[{step['role']}{' ' + step['name'] if 'name' in step else ''}] {body[:600]}")
    print(f"\n{run['provider']} {run['model']}: {run['seconds']} s, {run['tokens']} tokens; check: {run['check']}")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
