from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

import main


def test_build_turn_messages_new_thread_includes_system_prompt() -> None:
    messages = main.build_turn_messages(
        "system prompt",
        "current query",
        include_system_prompt=True,
    )

    assert messages == [
        SystemMessage(content="system prompt"),
        HumanMessage(content="current query"),
    ]


def test_build_turn_messages_existing_thread_only_sends_current_query() -> None:
    messages = main.build_turn_messages(
        "system prompt",
        "current query",
        include_system_prompt=False,
    )

    assert messages == [
        HumanMessage(content="current query"),
    ]


def test_resolve_thread_selection_existing_thread() -> None:
    thread_id, include_system_prompt = main.resolve_thread_selection(
        ["thread-1", "thread-2"],
        2,
    )

    assert thread_id == "thread-2"
    assert include_system_prompt is False


def test_resolve_thread_selection_create_new_thread(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "THREADS_PATH", tmp_path / "threads.txt")

    thread_id, include_system_prompt = main.resolve_thread_selection([], 1)

    assert thread_id == "thread-1"
    assert include_system_prompt is True
    assert main.get_all_thread_ids() == ["thread-1"]
