"""The agent: our own LangGraph graph.

The shape is the ReAct loop of eve-esa/agents (agent -> tools -> agent) with one node added:
``verify`` checks every number in the answer against the question, the tool arguments and the
tool results. A number not found sends the answer back once with the list; if it is still not
right, the run ends with the numbers flagged.

    START -> agent --(tool calls)--> tools --> agent
               |
               +--(answer)--> verify --(all numbers found, or already retried)--> END
                                 |
                                 +--(a number not found)--> agent, once
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .grounding import check

# A long conversation keeps its latest turns within this many tokens (an assess_burn result is about
# 3,000 to 4,000); eve-esa/agents trims the same way, at about 96,000.
MAX_CONTEXT_TOKENS = 60_000

SYSTEM_PROMPT = """\
You answer questions about fires, burn severity and vegetation change with the tools of a
Sentinel-2 pipeline. Every number in your answer must come from a tool result, never from memory.

Using the tools
- Areas are "west,south,east,north" in decimal degrees, longitude first. When the user names a
  place, use its coordinates.
- When the user gives no fire dates, call find_fires first and use the dates and bbox it returns
  (with margin_km=1, since its bbox hugs the burn).

Writing the answer
- Who asks decides the words, never the numbers. Quote numbers exactly as the tools give them,
  in digits, with units, and say which files hold the full result. At most 150 words.
- Add nothing about the place (villages, roads, buildings) that the tools did not return. Take
  the method and its limitations from the tool results, not from memory.
- {reader}
"""

# Who reads the answer. The brief's users are the park authority and civil protection, who do not
# work in remote sensing, so an unknown reader gets plain words.
READERS = {
    "public": "The reader does not work in remote sensing: a park ranger, a civil protection officer, "
    "a mayor. Use plain words (badly burned, steep: 23 degrees or more), name places and actions, "
    "and leave out index names, class names and file formats unless you explain them in a few words.",
    "expert": "The reader is a geospatial or remote sensing scientist. Use the technical terms (dNBR, "
    "the Key and Benson classes, NDVI, the terrain term of the M1 model, the elevation model and its "
    "resolution, the pre- and post-fire windows, COG and GeoJSON) and give the main limitation in one "
    "line.",
    None: "Judge the reader from the question. Technical wording means a specialist, who gets the "
    "technical terms (dNBR, severity classes, the elevation model, the windows); otherwise write in "
    "plain words for someone who does not work in remote sensing, without index names. When unsure, "
    "write plainly.",
}


def system_prompt(reader: str | None = None) -> str:
    """The system prompt for a reader: "public", "expert", or None to judge from the question."""
    if reader not in READERS:
        raise ValueError(f"reader must be one of public, expert or None, got {reader!r}")
    return SYSTEM_PROMPT.format(reader=READERS[reader])


MAX_CHECKS = 1  # times an answer is sent back for numbers not found


class State(MessagesState):
    checks: int
    not_found: list[str]


def text_of(message) -> str:
    """The text of a message whose content is a string or a list of content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)


def is_feedback(message) -> bool:
    return isinstance(message, HumanMessage) and str(message.id or "").startswith("verify")


def sources(messages) -> list[str]:
    """What an answer may quote: the user's words, the arguments sent, the tool results."""
    out = []
    for m in messages:
        if isinstance(m, ToolMessage) or (isinstance(m, HumanMessage) and not is_feedback(m)):
            out.append(text_of(m))
        elif isinstance(m, AIMessage):
            out.extend(str(call["args"]) for call in m.tool_calls)
    return out


def build_graph(llm, tools, reader: str | None = None, checkpointer=None):
    """Compile the graph for a chat model that supports tool calling, a list of tools and a
    reader ("public", "expert", or None to judge from the question).

    With a ``checkpointer`` the graph keeps the messages of every turn of a conversation, so a
    follow-up question can refer to earlier answers and tool results.
    """
    model = llm.bind_tools(tools)
    prompt = system_prompt(reader)

    async def agent(state: State) -> dict:
        # The latest turns that fit, starting at a question, so a tool result never loses its call.
        recent = trim_messages(state["messages"], max_tokens=MAX_CONTEXT_TOKENS, token_counter="approximate",
                               strategy="last", start_on="human")
        reply = await model.ainvoke([SystemMessage(prompt), *recent])
        return {"messages": [reply]}

    def after_agent(state: State) -> Literal["tools", "verify"]:
        return "tools" if state["messages"][-1].tool_calls else "verify"

    def verify(state: State) -> dict:
        messages = state["messages"]
        result = check(text_of(messages[-1]), sources(messages[:-1]))
        checks = state.get("checks", 0)
        if result["not_found"] and checks < MAX_CHECKS:
            note = HumanMessage(
                id=f"verify-{len(messages)}",  # unique in the conversation, so each turn adds its own
                content="These numbers in your answer are not in the tool results: "
                f"{', '.join(result['not_found'])}. Rewrite the answer with numbers from the tool "
                "results only, quoted as the tools give them.",
            )
            return {"messages": [note], "checks": checks + 1, "not_found": result["not_found"]}
        return {"not_found": result["not_found"]}

    def after_verify(state: State) -> Literal["agent", "__end__"]:
        return "agent" if is_feedback(state["messages"][-1]) else END

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))  # a failing tool comes back as a message, not a crash
    graph.add_node("verify", verify)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", "verify"])
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("verify", after_verify, ["agent", END])
    return graph.compile(checkpointer=checkpointer)
