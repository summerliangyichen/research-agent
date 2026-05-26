import sys
import asyncio
import time

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from graph import graph, _clean_research_markdown
from datetime import datetime
from pathlib import Path
import json

WORK_PATH = Path(__file__).parent
OUTPUTS_DIR = Path(__file__).parent / "outputs"
RUNS_LOG = OUTPUTS_DIR / "runs.jsonl"
AGENT_PATH = WORK_PATH / "AGENTS.md"

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(override= True)


def append_run_log(record: dict) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    record["created_at"] = datetime.now().astimezone().isoformat()

    with RUNS_LOG.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

def make_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")

def load_agent() -> str:
    if not AGENT_PATH.exists():
        AGENT_PATH.write_text("", encoding="utf-8")
        return ""
    return AGENT_PATH.read_text(encoding="utf-8")


def build_session_messages(
    agent_prompt: str,
    session_messages: list[BaseMessage],
    query: str,
) -> list[BaseMessage]:
    if session_messages:
        return [
            *session_messages,
            HumanMessage(content=query),
        ]

    return [
        SystemMessage(content=agent_prompt),
        HumanMessage(content=query),
    ]


def get_final_content(result: dict) -> str:
    content = result["messages"][-1].content

    if isinstance(content, str):
        return _clean_research_markdown(content)

    if isinstance(content, list):
        return _clean_research_markdown("\n".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in content
        ))

    return _clean_research_markdown(str(content))


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _parse_tool_content(content: object) -> object | None:
    if isinstance(content, (dict, list)):
        return content
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _tool_message_name(message: BaseMessage) -> str:
    return str(getattr(message, "name", "") or getattr(message, "tool_name", "")).strip()


def _extract_saved_artifacts(messages: list[BaseMessage]) -> tuple[str | None, str | None]:
    saved_path: str | None = None
    note_id: str | None = None

    for message in messages:
        if getattr(message, "type", "") != "tool" and message.__class__.__name__ != "ToolMessage":
            continue

        payload = _parse_tool_content(getattr(message, "content", ""))
        if not isinstance(payload, dict) or not payload.get("ok"):
            continue

        tool_name = _tool_message_name(message)
        if tool_name == "save_markdown_note":
            path = payload.get("path")
            if isinstance(path, str) and path.strip():
                saved_path = path.strip()
        elif tool_name == "save_note_index":
            path = payload.get("path")
            if isinstance(path, str) and path.strip():
                saved_path = path.strip()
            current_note_id = payload.get("note_id")
            if isinstance(current_note_id, str) and current_note_id.strip():
                note_id = current_note_id.strip()

    return saved_path, note_id


def _shorten_text(value: str, limit: int = 400) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n..."


def _summarize_payload(payload: object, limit: int = 400) -> str:
    if payload is None:
        return ""

    if isinstance(payload, BaseMessage):
        return _shorten_text(payload.pretty_repr(), limit)

    if isinstance(payload, str):
        return _shorten_text(payload, limit)

    try:
        text = json.dumps(payload, ensure_ascii=False, default=str, indent=2)
    except TypeError:
        text = str(payload)
    return _shorten_text(text, limit)


def _print_banner(title: str) -> None:
    print(f"\n[{title}]")


def _print_tool_start(name: str, data: dict) -> None:
    tool_input = data.get("input")
    _print_banner(f"tool start: {name}")
    if tool_input is not None:
        print(_summarize_payload(tool_input))


def _print_tool_end(name: str, data: dict) -> None:
    tool_output = data.get("output")
    _print_banner(f"tool end: {name}")
    if tool_output is not None:
        print(_summarize_payload(tool_output))


def _print_message_event(message: BaseMessage) -> None:
    _print_banner("assistant")
    message.pretty_print()


async def run_graph_visualized(payload: dict) -> dict:
    final_result: dict | None = None
    printed_message_ids: set[str] = set()
    seen_node_starts: set[tuple[int | None, str]] = set()
    last_assistant_content = ""

    async for event in graph.astream_events(payload, version="v2"):
        event_name = event.get("event", "")
        name = event.get("name", "")
        metadata = event.get("metadata") or {}
        data = event.get("data") or {}
        node_name = metadata.get("langgraph_node")
        step = metadata.get("langgraph_step")

        if event_name == "on_chain_start" and node_name and name == node_name:
            key = (step, node_name)
            if key not in seen_node_starts:
                seen_node_starts.add(key)
                _print_banner(f"node {step}: {node_name}")
            continue

        if event_name == "on_tool_start":
            _print_tool_start(name, data)
            continue

        if event_name == "on_tool_end":
            _print_tool_end(name, data)
            continue

        if event_name == "on_chain_end" and node_name and name == node_name:
            continue

        if event_name == "on_chat_model_end" and node_name == "assistant":
            output = data.get("output")
            if isinstance(output, BaseMessage):
                message_id = getattr(output, "id", None) or f"{node_name}:{step}"
                if message_id not in printed_message_ids:
                    printed_message_ids.add(message_id)
                    _print_message_event(output)
                content = _clean_research_markdown(_content_to_text(output.content))
                if content:
                    last_assistant_content = content
            continue

        if event_name == "on_chain_end" and name == "LangGraph":
            output = data.get("output")
            if isinstance(output, dict):
                final_result = output

    if final_result is None:
        raise RuntimeError("graph 没有返回最终结果")
    if last_assistant_content:
        final_result["_last_assistant_content"] = last_assistant_content
    return final_result


async def main():
    agent_prompt = load_agent()
    session_messages: list[BaseMessage] = []

    while True:
        start_time = time.time()
        query = input("请输入问题（exit 退出）：").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        run_id = make_run_id()
        payload = {
            "messages": build_session_messages(agent_prompt, session_messages, query)
        }

        _print_banner("user")
        HumanMessage(content=query).pretty_print()

        try:
            result = await run_graph_visualized(payload)
        except Exception as exc:
            append_run_log(
                {
                    "run_id": run_id,
                    "query": query,
                    "status": "error",
                    "saved": False,
                    "elapsed_seconds": round(time.time() - start_time, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise

        final_content = get_final_content(result)
        result_messages = result.get("messages", [])
        if isinstance(result_messages, list):
            session_messages = list(result_messages)

        saved_path, note_id = _extract_saved_artifacts(result_messages)
        elapsed_seconds = round(time.time() - start_time, 3)

        log_record = {
            "run_id": run_id,
            "query": query,
            "status": "success",
            "saved": bool(saved_path),
            "elapsed_seconds": elapsed_seconds,
        }
        if note_id:
            log_record["note_id"] = note_id
        if saved_path:
            log_record["output_file"] = saved_path
    
        append_run_log(
            log_record
        )

        last_assistant_content = result.get("_last_assistant_content", "")
        if final_content and final_content != last_assistant_content:
            _print_banner("result")
            print(final_content)
        if saved_path:
            print(f"\n已保存：{saved_path}")
        print('程序运行时间：%s 秒' % elapsed_seconds)

if __name__ == "__main__":
    asyncio.run(main())
