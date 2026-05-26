from __future__ import annotations

import asyncio
from langchain_deepseek import ChatDeepSeek
from typing import TypedDict, Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
import re
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tool import (
    bash_tool,
    batch_crawl_webpage,
    crawl_webpage,
    fetch_related_urls,
    local_rag_search,
    read_file,
    save_markdown_note,
    save_note_index,
)


WORK_DIR = Path(__file__).parent
load_dotenv(WORK_DIR / ".env")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}},
)

tools = [
    crawl_webpage,
    fetch_related_urls,
    read_file,
    batch_crawl_webpage,
    bash_tool,
    local_rag_search,
    save_markdown_note,
    save_note_index,
]
llm_with_tools = llm.bind_tools(tools)


async def assistant_node(state: AgentState) -> AgentState:
    messages = _sanitize_messages(state["messages"])
    try:
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}
    except Exception as exc:
        if _is_llm_provider_error(exc):
            return {"messages": [AIMessage(content=_llm_provider_error_message(exc))]}
        raise

def route_after_assistant(state: AgentState) -> str:
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "end"

def _sanitize_text(text: str) -> str:
    return "".join("\uFFFD" if 0xD800 <= ord(char) <= 0xDFFF else char for char in text)

def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value)

    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_sanitize_value(item) for item in value)

    if isinstance(value, dict):
        return {_sanitize_value(key): _sanitize_value(item) for key, item in value.items()}

    return value

def _sanitize_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    sanitized_messages: list[BaseMessage] = []

    for message in messages:
        updates: dict[str, Any] = {}
        content = getattr(message, "content", None)
        clean_content = _sanitize_value(content)
        if clean_content != content:
            updates["content"] = clean_content

        additional_kwargs = getattr(message, "additional_kwargs", None)
        clean_additional_kwargs = _sanitize_value(additional_kwargs)
        if clean_additional_kwargs != additional_kwargs:
            updates["additional_kwargs"] = clean_additional_kwargs

        sanitized_messages.append(message.model_copy(update=updates) if updates else message)

    return sanitized_messages

def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return _sanitize_text(content)

    if isinstance(content, list):
        return _sanitize_text("\n".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in content
        ))

    return _sanitize_text(str(content))

def _clean_research_markdown(content: Any) -> str:
    text = _content_to_text(content).strip()
    if not text:
        return text

    heading_match = re.search(r"(?m)^#\s+", text)
    if heading_match:
        return text[heading_match.start():].strip()

    removable_prefixes = [
        r"^Good,\s*I have\b.*?(?:research note\.|compile the research note\.)",
        r"^I have\b.*?(?:research note\.|compile the research note\.)",
        r"^Let me\b.*?(?:research note\.|compile the research note\.)",
        r"^信息足够丰富了[，,].*?研究笔记。",
        r"^我来(?:整合)?生成(?:研究笔记)?。",
        r"^以下是(?:整理好的)?研究笔记[:：]?",
        r"^研究笔记(?:已经)?生成(?:如下)?[:：]?",
        r"^这个问题.*?我来生成。",
    ]
    for pattern in removable_prefixes:
        text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE | re.DOTALL).strip()

    return text

def _is_llm_provider_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {401, 402, 429, 500, 502, 503, 504}:
        return True

    message = str(exc).lower()
    markers = (
        "insufficient balance",
        "invalid api key",
        "rate limit",
        "timeout",
        "connection error",
        "api status error",
    )
    return any(marker in message for marker in markers)


def _llm_provider_error_message(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    detail = str(exc).strip()
    if status_code == 402 or "insufficient balance" in detail.lower():
        return "当前模型提供方余额不足，暂时无法继续调用 LLM。先补余额或切到可用模型，再运行一次。"
    return f"当前模型提供方暂时不可用，无法继续调用 LLM。错误信息：{detail}"

builder = StateGraph(AgentState)

builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START,"assistant")
builder.add_conditional_edges(
    "assistant",
    route_after_assistant,
    {
        "tools": "tools",
        "end": END,
    },
)
builder.add_edge("tools", "assistant")

graph = builder.compile()
