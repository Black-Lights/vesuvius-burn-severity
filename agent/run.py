"""Ask the agent questions, one or a whole conversation.

``Conversation`` starts the MCP server as a child process over stdio, loads its tools as LangChain
tools and keeps both open for the whole conversation. A LangGraph checkpointer keeps the messages
of every turn, so a follow-up ("and at 20 degrees?") can refer to what came before. Each turn
returns a transcript: the tool calls with their arguments, the tool results, the answer and the
number check.

The model is any chat model served in the OpenAI format, chosen in .env (see .env.example):
LLM_PROVIDER picks DeepSeek, Kimi or OpenAI, LLM_MODEL overrides the model. LLM_PROVIDER=custom
uses LLM_BASE_URL, LLM_API_KEY and LLM_MODEL, which is how a self-hosted model such as
EVE-Instruct, served through an OpenAI-compatible endpoint, would be plugged in.

Usage:
    python -m agent.run "Which burned slopes of the Vesuvius fire of August 2025 need work first?"
    python -m agent.run "..." --provider kimi --reader expert --save outputs/agent_run.json
    python -m agent.chat                       # a conversation, question after question
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Self

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
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


def steps_of(messages: list) -> list[dict]:
    """The messages of a turn as plain data: asked, called, returned, checked, answered."""
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
    return steps


class Conversation:
    """A conversation with the agent: one MCP server and one memory for every turn.

        async with Conversation(reader="public") as chat:
            first = await chat.ask("Which burned slopes of the Vesuvius fire ... first?")
            second = await chat.ask("And if 20 degrees counts as steep?")
    """

    def __init__(self, provider: str | None = None, model: str | None = None, reader: str | None = None):
        self.llm, self.provider, self.model = make_llm(provider, model)
        self.reader = reader
        self.config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 20}
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> Self:
        server = StdioServerParameters(command=sys.executable, args=[str(SERVER), "--transport", "stdio"])
        read, write = await self._stack.enter_async_context(stdio_client(server))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self.graph = build_graph(self.llm, await load_mcp_tools(session), self.reader, InMemorySaver())
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    async def ask(self, question: str) -> dict:
        """One turn: the question, what the agent did for it, the answer and its check."""
        earlier = (await self.graph.aget_state(self.config)).values.get("messages", [])
        started = time.perf_counter()
        state = await self.graph.ainvoke(
            {"messages": [HumanMessage(question)], "checks": 0, "not_found": []}, self.config
        )  # the recursion limit stops a turn after about nine tool rounds
        seconds = time.perf_counter() - started
        messages = state["messages"]
        turn = messages[len(earlier):]
        answer = text_of(messages[-1])
        return {
            "provider": self.provider,
            "model": self.model,
            "reader": self.reader or "judged from the question",
            "seconds": round(seconds, 1),
            "tokens": sum((m.usage_metadata or {}).get("total_tokens", 0) for m in turn if isinstance(m, AIMessage)),
            "steps": steps_of(turn),
            "answer": answer,
            "check": check(answer, sources(messages[:-1])),  # earlier turns count as sources too
        }


async def ask(question: str, provider: str | None = None, model: str | None = None,
              reader: str | None = None) -> dict:
    """One question as a conversation of one turn.

    ``reader`` is "public" (plain words), "expert" (technical terms) or None (judged from the
    question). It changes the words of the answer, never its numbers.
    """
    async with Conversation(provider, model, reader) as chat:
        return await chat.ask(question)


def show(turn: dict) -> None:
    """Print a turn: the tool calls, the answer, time, tokens and the check."""
    for step in turn["steps"]:
        if "tool_calls" in step:
            for call in step["tool_calls"]:
                print(f"  -> {call['name']}({json.dumps(call['args'])})")
        elif step["role"] == "check":
            print(f"  check: {step['content']}")
    print(f"\n{turn['answer']}\n")
    print(f"  [{turn['provider']} {turn['model']}, {turn['seconds']} s, {turn['tokens']} tokens; "
          f"numbers checked {turn['check']['numbers_checked']}, not found {turn['check']['not_found']}]")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # degree signs and dashes on a Windows console
    parser = argparse.ArgumentParser(description="Ask the burn severity agent one question")
    parser.add_argument("question")
    parser.add_argument("--provider", choices=[*PROVIDERS, "custom"])
    parser.add_argument("--model")
    parser.add_argument("--reader", choices=["public", "expert"], help="default: judged from the question")
    parser.add_argument("--save", type=Path, help="write the transcript as JSON")
    args = parser.parse_args()
    turn = asyncio.run(ask(args.question, args.provider, args.model, args.reader))
    show(turn)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(turn, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
