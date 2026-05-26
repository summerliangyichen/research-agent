from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import main


def test_build_session_messages_first_turn_uses_system_prompt() -> None:
    messages = main.build_session_messages("system prompt", [], "这轮问题")

    assert messages == [
        SystemMessage(content="system prompt"),
        HumanMessage(content="这轮问题"),
    ]


def test_build_session_messages_reuses_full_previous_messages() -> None:
    previous_messages = [
        SystemMessage(content="system prompt"),
        HumanMessage(content="上一轮问题"),
        AIMessage(content="上一轮回答"),
    ]

    messages = main.build_session_messages("ignored prompt", previous_messages, "下一轮问题")

    assert messages == [
        *previous_messages,
        HumanMessage(content="下一轮问题"),
    ]
