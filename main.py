import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from graph import _clean_research_markdown, compile_graph

WORK_PATH = Path(__file__).parent
OUTPUTS_DIR = WORK_PATH / "outputs"
RUNS_LOG = OUTPUTS_DIR / "runs.jsonl"
AGENT_PATH = WORK_PATH / "AGENTS.md"
THREADS_PATH = WORK_PATH / "threads.txt"
THREADS_DB_PATH = WORK_PATH / "threads.db"
GRAPH_RECURSION_LIMIT = 100

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(override=True)



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


def build_turn_messages(
    agent_prompt: str,
    query: str,
    *,
    include_system_prompt: bool,
) -> list[BaseMessage]:
    messages: list[BaseMessage] = [HumanMessage(content=query)]
    if include_system_prompt:
        return [SystemMessage(content=agent_prompt), *messages]
    return messages


def get_final_content(result: dict) -> str:
    content = result["messages"][-1].content

    if isinstance(content, str):
        return _clean_research_markdown(content)

    if isinstance(content, list):
        return _clean_research_markdown(
            "\n".join(
                block.get("text", str(block)) if isinstance(block, dict) else str(block)
                for block in content
            )
        )

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


async def run_graph_visualized(payload: dict, thread_id: str, compiled_graph, raw_choice: int) -> dict:
    final_result: dict | None = None
    printed_message_ids: set[str] = set()
    seen_node_starts: set[tuple[int | None, str]] = set()
    last_assistant_content = ""

    async for event in compiled_graph.astream_events(
        payload,
        config={
            "recursion_limit": GRAPH_RECURSION_LIMIT,
            "configurable": {"thread_id": thread_id},
        },
        version="v2",
        
    ):
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


def save_new_thread_id() -> str:
    THREADS_PATH.touch(exist_ok=True)

    with THREADS_PATH.open("a+", encoding="utf-8") as file:
        file.seek(0)
        lines = [line.strip() for line in file if line.strip()]

        if not lines:
            new_thread = "thread-1"
        else:
            prefix, number = lines[-1].rsplit("-", 1)
            new_thread = f"{prefix}-{int(number) + 1}"

        file.write(new_thread + "\n")

    return new_thread


def get_all_thread_ids() -> list[str]:
    THREADS_PATH.touch(exist_ok=True)
    with THREADS_PATH.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def resolve_thread_selection(threads: list[str], selection: int, raw_choice: int) -> tuple[str, bool]:
    create_new_index = len(threads) + 1
    if selection < 1 or selection > create_new_index:
        raise ValueError("invalid choice")

    if selection == create_new_index:
        return save_new_thread_id(), True, raw_choice

    return threads[selection - 1], False, raw_choice


def prompt_for_thread_selection(threads: list[str]) -> tuple[str, bool, int]:
    while True:
        try:
            raw = input("choose a session: ").strip()
        except EOFError:
            raise EOFError("线程选择阶段收到 EOF")
        try:
            selection = int(raw)
            return resolve_thread_selection(threads, selection, raw)
        except (TypeError, ValueError):
            print("invalid choice")

async def main():
    agent_prompt = load_agent()

    threads = get_all_thread_ids()
    for index, thread in enumerate(threads, start=1):
        print(f"{index}: {thread}")
    print(f"{len(threads) + 1}: create_new_session")

    try:
        thread_id, include_system_prompt, raw_choice = prompt_for_thread_selection(threads)
    except EOFError:
        print("\n收到 EOF，已退出。")
        return

    async with AsyncSqliteSaver.from_conn_string(str(THREADS_DB_PATH)) as checkpointer:
        compiled_graph = compile_graph(checkpointer=checkpointer)

        while True:
            try:
                query = input("请输入问题（exit 退出）：").strip()
                start_time = time.time()
            except EOFError:
                print("\n收到 EOF，已退出。")
                break
            if query.lower() == "exit":
                break
            if not query:
                continue

            run_id = make_run_id()
            payload = {
                "messages": build_turn_messages(
                    agent_prompt,
                    query,
                    include_system_prompt=include_system_prompt,
                ),
                "thread_id_choice": raw_choice,
                "user_input": query,
                "title": ""
            }

            _print_banner("user")
            HumanMessage(content=query).pretty_print()

            try:
                result = await run_graph_visualized(payload, thread_id=thread_id, compiled_graph=compiled_graph, raw_choice = raw_choice)
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
            saved_path, note_id = _extract_saved_artifacts(result_messages)
            elapsed_seconds = round(time.time() - start_time, 3)
            include_system_prompt = False

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

            append_run_log(log_record)

            last_assistant_content = result.get("_last_assistant_content", "")
            if final_content and final_content != last_assistant_content:
                _print_banner("result")
                print(final_content)
            if saved_path:
                print(f"\n已保存：{saved_path}")
            print(f"程序运行时间：{elapsed_seconds} 秒")

if __name__ == "__main__":
    asyncio.run(main())
