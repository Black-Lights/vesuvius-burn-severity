"""Checks on the agent with a scripted model and a fake tool. No network, no key."""

import asyncio
import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agent.graph import build_graph, system_prompt
from agent.grounding import check, numbers


class ScriptedModel(BaseChatModel):
    """Answers with the next message of a script, whatever it is asked."""

    replies: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


@tool
def assess_burn(bbox: str, fire_start: str, fire_end: str) -> str:
    """Fake burn assessment."""
    return json.dumps({"burned_ha": 737, "severe_ha": 576, "priority_1_share": 0.62})


def _call(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_1", "type": "tool_call"}])


ARGS = {"bbox": "14.35,40.77,14.50,40.87", "fire_start": "2025-08-08", "fire_end": "2025-08-12"}


def _run(replies):
    graph = build_graph(ScriptedModel(replies=replies), [assess_burn])
    question = HumanMessage("How badly did the Vesuvius fire of August 2025 burn?")
    return asyncio.run(graph.ainvoke({"messages": [question], "checks": 0, "not_found": []}))


def test_numbers_are_read_as_written():
    found = numbers("1. Burned: 1,144 ha, a fall of -0.35, 62% of cells, on 2025-08-08.")
    assert [(shown, value) for shown, value, _ in found] == [
        ("1,144", 1144.0), ("0.35", 0.35), ("62", 62.0), ("2025", 2025.0), ("08", 8.0), ("08", 8.0)]


def test_check_accepts_rounding_and_shares_and_rejects_the_rest():
    tool_result = '{"burned_ha": 736.8, "share": 0.62, "lat": 40.81796, "change": -0.349}'
    good = "About 737 ha burned; 62 % of the cell; centre 40.818 N; NDVI fell by 0.35."
    assert check(good, [tool_result])["not_found"] == []
    assert check("750 ha burned", [tool_result])["not_found"] == ["750"]


def test_a_grounded_answer_goes_straight_to_the_end():
    state = _run([_call("assess_burn", ARGS), AIMessage("The fire burned 737 ha, 576 ha of it severely.")])
    assert state["checks"] == 0 and state["not_found"] == []
    assert state["messages"][-1].content.startswith("The fire burned 737 ha")


def test_a_wrong_number_is_sent_back_once_and_corrected():
    state = _run([_call("assess_burn", ARGS), AIMessage("The fire burned 750 ha."),
                  AIMessage("The fire burned 737 ha.")])
    assert state["checks"] == 1 and state["not_found"] == []
    feedback = [m for m in state["messages"] if str(m.id or "").startswith("verify")]
    assert len(feedback) == 1 and "750" in feedback[0].content
    assert state["messages"][-1].content == "The fire burned 737 ha."


def test_a_number_still_wrong_after_the_retry_is_flagged():
    state = _run([_call("assess_burn", ARGS), AIMessage("The fire burned 750 ha."),
                  AIMessage("The fire burned 760 ha.")])
    assert state["checks"] == 1 and state["not_found"] == ["760"]


def test_the_reader_changes_the_words_asked_for_not_the_rules():
    public, expert, judged = system_prompt("public"), system_prompt("expert"), system_prompt(None)
    assert "plain words" in public and "dNBR" in expert and "Judge the reader" in judged
    assert all("never from memory" in p for p in (public, expert, judged))
    with pytest.raises(ValueError, match="reader"):
        system_prompt("child")


def test_numbers_written_as_words_are_checked_too():
    found = [(shown, value) for shown, value, _ in numbers("Twenty more squares; twenty-two of them in one block.")]
    assert found == [("Twenty", 20.0), ("twenty-two", 22.0)]  # "one" is too common to count
    assert check("Twenty more squares come next.", ['{"priority_2_cells": 19}'])["not_found"] == ["Twenty"]


def test_a_follow_up_can_quote_the_first_turn():
    from langgraph.checkpoint.memory import InMemorySaver

    replies = [_call("assess_burn", ARGS), AIMessage("The fire burned 737 ha."),
               AIMessage("Of those, 576 ha burned severely.")]  # no new tool call: 576 is from turn one
    graph = build_graph(ScriptedModel(replies=replies), [assess_burn], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    turn = {"checks": 0, "not_found": []}
    asyncio.run(graph.ainvoke({"messages": [HumanMessage("How much burned?")], **turn}, config))
    state = asyncio.run(graph.ainvoke({"messages": [HumanMessage("And how much of it severely?")], **turn}, config))
    assert state["not_found"] == [] and state["checks"] == 0
    assert [m.content for m in state["messages"] if isinstance(m, HumanMessage)] == [
        "How much burned?", "And how much of it severely?"]
