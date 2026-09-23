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

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .grounding import check

SYSTEM_PROMPT = """\
You answer questions about fires, burn severity and vegetation change with the tools of a
Sentinel-2 pipeline. Every number in your answer must come from a tool result; never from memory.

- Areas are "west,south,east,north" in decimal degrees, longitude first. When the user names a
  place, use its coordinates.
- When the user gives no fire dates, call find_fires first and use the dates and bbox it returns
  (with margin_km=1, since its bbox hugs the burn).
- Answer in at most 120 words, in plain language, for a reader who does not work in remote
  sensing. Quote numbers exactly as the tools give them, with units, and say which files hold
  the full result.
"""

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


def build_graph(llm, tools):
    """Compile the graph for a chat model that supports tool calling and a list of tools."""
    model = llm.bind_tools(tools)

    async def agent(state: State) -> dict:
        reply = await model.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    def after_agent(state: State) -> Literal["tools", "verify"]:
        return "tools" if state["messages"][-1].tool_calls else "verify"

    def verify(state: State) -> dict:
        messages = state["messages"]
        result = check(text_of(messages[-1]), sources(messages[:-1]))
        checks = state.get("checks", 0)
        if result["not_found"] and checks < MAX_CHECKS:
            note = HumanMessage(
                id=f"verify-{checks + 1}",
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
    return graph.compile()
