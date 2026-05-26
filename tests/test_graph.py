from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

import graph


def test_graph_has_only_assistant_and_tools_nodes() -> None:
    compiled = graph.graph.get_graph()
    assert sorted(compiled.nodes.keys()) == ["__end__", "__start__", "assistant", "tools"]


def test_assistant_node_always_uses_same_runnable(monkeypatch) -> None:
    class DummyRunnable:
        def __init__(self) -> None:
            self.calls: list[list[object]] = []

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            self.calls.append(messages)
            return AIMessage(content="ok")

    dummy = DummyRunnable()
    monkeypatch.setattr(graph, "llm_with_tools", dummy)

    state = {
        "messages": [
            SystemMessage(content="system"),
            ToolMessage(
                content='{"ok": true, "path": "D:/tmp/note.md"}',
                tool_call_id="call_1",
                name="save_note_index",
            ),
        ],
    }

    result = asyncio.run(graph.assistant_node(state))

    assert result["messages"][0].content == "ok"
    assert len(dummy.calls) == 1
    assert dummy.calls[0] == state["messages"]
