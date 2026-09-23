"""A conversation with the agent in the terminal: type a question, read the answer, ask a
follow-up. An empty line or "exit" ends it.

Usage:
    python -m agent.chat
    python -m agent.chat --reader expert --provider kimi --save outputs/agent_chat.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .run import PROVIDERS, Conversation, show


async def chat(provider: str | None, model: str | None, reader: str | None, save: Path | None) -> None:
    turns = []
    async with Conversation(provider, model, reader) as conversation:
        print(f"{conversation.provider} {conversation.model}, reader: {reader or 'judged from the question'}. "
              "Empty line or 'exit' to stop.\n")
        while True:
            question = (await asyncio.to_thread(input, "you> ")).strip()
            if question.lower() in {"", "exit", "quit"}:
                break
            turn = await conversation.ask(question)
            show(turn)
            turns.append(turn)
    if save and turns:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(json.dumps(turns, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Talk to the burn severity agent")
    parser.add_argument("--provider", choices=[*PROVIDERS, "custom"])
    parser.add_argument("--model")
    parser.add_argument("--reader", choices=["public", "expert"], help="default: judged from each question")
    parser.add_argument("--save", type=Path, help="write every turn as JSON when the chat ends")
    args = parser.parse_args()
    asyncio.run(chat(args.provider, args.model, args.reader, args.save))


if __name__ == "__main__":
    main()
