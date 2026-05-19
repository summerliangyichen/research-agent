from __future__ import annotations

from langchain_deepseek import ChatDeepSeek
from typing import TypedDict, Annotated, Any, NotRequired, Literal
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tool import crawl_webpage, fetch_related_urls, read_file, batch_crawl_webpage
import json


WORK_DIR = Path(__file__).parent
OUTPUTS_DIR = WORK_DIR / "outputs"
INDEX_PATH = OUTPUTS_DIR / "index.jsonl"
load_dotenv(WORK_DIR / ".env")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intention: Literal["research", "chat"]
    filename: str
    run_id: str
    note_id: str
    query: str
    path: str
    indexed: bool
    content: str
    links: list[str]
    sources: list[dict[str, Any]]
    required_path: NotRequired[str | None]
    time_range: NotRequired[str | None]

class search(BaseModel):
    intention: Literal["research", "chat"]

class save_research_note(BaseModel):
    should_save: bool

class structured_filename(BaseModel):
    filename: str

class summary_note(BaseModel):
    summary_format: str

class parse_required_path(BaseModel):
    required_path: str | None = Field(
        default=None,
        description="用户明确指定的保存路径或文件名；如果没有指定则为 None",
    )

llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}},
)

tools = [crawl_webpage, fetch_related_urls, read_file, batch_crawl_webpage]
llm_with_tools = llm.bind_tools(tools)

async def identify_intention(state:AgentState) -> dict:
    structured_llm = llm.with_structured_output(search)
    query = _sanitize_text(state["query"])
    response = await structured_llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "Classify the user's raw input into exactly one label: research or chat.\n\n"
                    "Return research only when the user is asking for a substantive research note, "
                    "multi-source investigation, source-backed explanation, URL/page analysis, news/background research, "
                    "or a complex topic that should become a saved Markdown research artifact.\n\n"
                    "Return chat for greetings, small talk, simple direct questions, date/time questions, "
                    "short factual answers, clarification, commands about the program, or anything that should not be saved "
                    "as a long-term research note.\n\n"
                    "Examples:\n"
                    "- hi -> chat\n"
                    "- 你好 -> chat\n"
                    "- 今天几号 -> chat\n"
                    "- 现在几点 -> chat\n"
                    "- time.sleep 的参数是秒吗 -> chat\n"
                    "- Python 异步编程 -> research\n"
                    "- 帮我研究 CPython 和 Python 的区别 -> research\n"
                    "- 阅读这个 URL 并总结 -> research\n\n"
                    "If unsure, choose chat. Do not classify a query as research merely because it is a question."
                )
            ),
            HumanMessage(content=query),
        ]
    )
    return {"intention": response.intention}

async def determine_required_path(state:AgentState) -> AgentState:
    query = _sanitize_text(state["query"])
    structured_llm = llm.with_structured_output(parse_required_path)
    response = await structured_llm.ainvoke([
        SystemMessage(
            content=(
                "Identify whether the user explicitly mentioned a file path or file name "
                "for saving the generated Markdown note. Return None if no explicit save "
                "path or file name is mentioned. Do not invent a path."
            )
        ),
        HumanMessage(
            content=(
                f"identify the required path in this query: {query}"
            )
        )
    ])

    return {
        "required_path": response.required_path
    }

async def assistant_node(state: AgentState) -> AgentState:
    response = await llm_with_tools.ainvoke(_sanitize_messages(state["messages"]))
    return {"messages": [response]}

async def should_save_research_note(state: AgentState) -> bool:
    if not state["messages"]:
        return False

    content = _clean_research_markdown(state["messages"][-1].content)
    if not content:
        return False

    if _looks_like_research_markdown(content):
        return True

    structured_llm = llm.with_structured_output(save_research_note)
    response = await structured_llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "Decide whether the assistant's final answer should be saved as a long-term Research Markdown note.\n\n"
                    "Return true only when ALL conditions are met:\n"
                    "1. The original user query is a substantive research request, not casual chat or a simple direct question.\n"
                    "2. The assistant final answer is a complete Research Markdown note, not a short answer.\n"
                    "3. The answer starts with a Markdown H1 title such as '# [[...]]'.\n"
                    "4. The answer contains a fixed 核心结论 section and enough prose analysis to be useful as a note.\n"
                    "5. The answer does not need to follow fixed section names such as 背景与上下文, 主要线索, 关联观察, or 仍需确认.\n"
                    "6. The content is worth saving for future retrieval.\n\n"
                    "Return false for greetings, small talk, date/time questions, one-sentence answers, casual replies, "
                    "clarification questions, error messages, tool/status text, or any answer to queries like 'hi', '你好', "
                    "'今天几号', '现在几点'.\n\n"
                    "If unsure, return false."
                )
            ),
            HumanMessage(
                content=(
                    f"Original user query:\n{state.get('query', '')}\n\n"
                    f"Assistant final answer:\n{content}"
                )
            ),
        ]
    )
    return response.should_save

def route_after_assistant(state: AgentState) -> str:
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "decide_generate_research_note"

def decide_generate_research_note(state:AgentState) -> AgentState:
    return {}

async def determine_filename(state: AgentState) -> dict:
    content = _clean_research_markdown(_content_to_text(state["messages"][-1].content))
    structured_llm = llm.with_structured_output(structured_filename)
    response = await structured_llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "你负责给一篇 Research Markdown 笔记生成保存文件名。"
                    "只返回一个适合保存为 Markdown 文件名的短标题。"
                    "文件名必须是 1-2 个核心名词或名词短语，不要解释，不要副标题，"
                    "不要冒号、破折号、括号补充说明、Obsidian 双链符号或 .md 后缀。"
                    "例如用户问“全局解释器锁是什么”，返回“全局解释器锁”；"
                    "用户问“CPython 和 Python”，返回“CPython 与 Python”。"
                )
            ),
            HumanMessage(
                content=(
                    f"用户原始问题：\n{state.get('query', '')}\n\n"
                    f"Research Markdown 笔记：\n{content}"
                )
            ),
        ]
    )
    filename = _normalize_filename_label(response.filename)

    return {"filename": filename, "content": content}

def save_markdown(state:AgentState) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    filename = state["filename"]
    content = _clean_research_markdown(state["content"])
    required_path = state.get("required_path")

    if required_path:
        target_path = safe_path(required_path)
    elif filename:
        safe_name = _safe_markdown_filename(filename)
        target_path = OUTPUTS_DIR / safe_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"output_{timestamp}.md"
        target_path = OUTPUTS_DIR / safe_name

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")

    return {"path": str(target_path), "content": content}

def safe_path(path: str, base_dir: Path = OUTPUTS_DIR) -> Path:
    raw_path = _content_to_text(path).strip().strip("'\"“”` ")
    if not raw_path:
        raise ValueError("path 不能为空")

    candidate = Path(raw_path)
    if any(part in {".", ".."} for part in candidate.parts):
        raise ValueError("path 不能包含 . 或 ..")

    filename = _safe_markdown_filename(candidate.name)

    if candidate.suffix and candidate.suffix.lower() != ".md":
        raise ValueError("只允许保存 Markdown 文件（.md）")

    if candidate.is_absolute():
        target_path = candidate.with_name(filename)
    else:
        target_path = base_dir / candidate.parent / filename

    base = base_dir.resolve()
    resolved = target_path.resolve()

    try:
        relative_path = resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"path 必须位于 {base} 内") from exc

    for part in relative_path.parts:
        _validate_safe_path_part(part)

    return resolved


def _safe_markdown_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
    name = name.rstrip(" .")
    if not name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"output_{timestamp}.md"
    if not name.lower().endswith(".md"):
        name = f"{name}.md"
    if _is_windows_reserved_name(name):
        name = f"_{name}"
    return name

def _is_windows_reserved_name(name: str) -> bool:
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        *{f"COM{i}" for i in range(1, 10)},
        *{f"LPT{i}" for i in range(1, 10)},
    }
    return Path(name).stem.upper() in reserved_names

def _validate_safe_path_part(part: str) -> None:
    if part in {"", ".", ".."}:
        raise ValueError("path 不能包含空路径段、. 或 ..")
    if part != part.rstrip(" ."):
        raise ValueError("path 不能包含以空格或点结尾的路径段")
    if re.search(r'[<>:"|?*\x00-\x1f]', part):
        raise ValueError("path 包含 Windows 非法字符")
    if _is_windows_reserved_name(part):
        raise ValueError("path 包含 Windows 保留名称")

def _normalize_filename_label(filename: str) -> str:
    name = _content_to_text(filename).strip()
    name = name.removeprefix("#").strip()
    name = _wikilinks_to_text(name)
    name = Path(name).name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip()
    if name.lower().endswith(".md"):
        name = name[:-3].strip()
    if not name:
        return datetime.now().strftime("output_%Y%m%d_%H%M%S")
    return name

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

def _wikilinks_to_text(markdown: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw_link = match.group(1).strip()
        target, separator, alias = raw_link.partition("|")
        display_text = alias.strip() if separator else target.strip()
        display_text = display_text.split("#", 1)[0].strip()
        display_text = display_text.split("^", 1)[0].strip()
        return display_text

    return re.sub(r"\[\[([^\[\]\n]+?)\]\]", replace, markdown)

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

def _looks_like_research_markdown(content: str) -> bool:
    text = _clean_research_markdown(content)
    if not re.search(r"(?m)^#\s+\S+", text):
        return False
    if "## 核心结论" not in text:
        return False
    if len(text) < 500:
        return False
    return True

def _extract_wikilinks(markdown: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()

    for match in re.finditer(r"(?<!!)\[\[([^\[\]\n]+?)\]\]", markdown):
        raw_link = match.group(1).strip()
        target = raw_link.split("|", 1)[0].strip()
        target = target.split("#", 1)[0].strip()
        target = target.split("^", 1)[0].strip()

        if target and target not in seen:
            links.append(target)
            seen.add(target)

    return links

def _parse_tool_content(content: Any) -> Any | None:
    if isinstance(content, dict | list):
        return content

    text = _content_to_text(content).strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def _iter_crawl_results(payload: Any):
    if isinstance(payload, dict):
        if "ok" in payload and "url" in payload:
            yield payload
            return

        for value in payload.values():
            if isinstance(value, dict | list):
                yield from _iter_crawl_results(value)

    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_crawl_results(item)

def _extract_sources_from_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for message in messages:
        if getattr(message, "type", "") != "tool" and message.__class__.__name__ != "ToolMessage":
            continue

        payload = _parse_tool_content(getattr(message, "content", ""))
        if payload is None:
            continue

        for result in _iter_crawl_results(payload):
            if result.get("ok") is not True:
                continue

            url = result.get("url")
            if not isinstance(url, str):
                continue

            url = url.strip()
            if not url or url in seen_urls:
                continue

            title = result.get("title")
            site = urlparse(url).netloc
            source: dict[str, Any] = {
                "title": str(title).strip() if title else url,
                "url": url,
            }
            if site:
                source["site"] = site

            sources.append(source)
            seen_urls.add(url)

    return sources

def _make_note_id(title: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = _safe_markdown_filename(title).removesuffix(".md")
    return f"{timestamp}_{name}"

async def save_json(state: AgentState) -> AgentState:
    content = state.get("content") or state["messages"][-1].content
    title = state["filename"]
    content = _content_to_text(content)
    links = _extract_wikilinks(content)
    sources = _extract_sources_from_messages(state["messages"])
    note_id = _make_note_id(title)

    structured_llm = llm.with_structured_output(summary_note)
    response = (
        await structured_llm.ainvoke([
        SystemMessage(content = "you will need to summarize your own research note into 500chars at most"
                                "you will have to contain the '核心结论' and partial of the others."
                                "do not contain urls"),
        HumanMessage(content = f"please summarize the following note:\n\n{content}")
        ])
    ).summary_format

    record = {
        "note_id": note_id,
        "run_id": state.get("run_id", ""),
        "title": title,
        "query": state["query"],
        "filename": state["filename"],
        "path": state["path"],
        "summary": response,
        "links": links,
        "sources": sources,
        "created_at": datetime.now().astimezone().isoformat()
    }

    with INDEX_PATH.open("a", encoding = "utf-8") as file:
        file.write(json.dumps(record,ensure_ascii = False)+"\n")

    return {"indexed": True, "note_id": note_id, "links": links, "sources": sources}

def route_after_intention(state: AgentState) -> str:
    return state["intention"]

builder = StateGraph(AgentState)

builder.add_node("determine_required_path", determine_required_path)
builder.add_node("identify_intention", identify_intention)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("decide_generate_research_note", decide_generate_research_note)
builder.add_node("determine_filename", determine_filename)
builder.add_node("save_markdown", save_markdown)
builder.add_node("save_json", save_json)

builder.add_edge(START,"identify_intention")
builder.add_conditional_edges(
    "identify_intention",
    route_after_intention,
    {
        "research": "determine_required_path",
        "chat": "assistant"
    }
)
builder.add_edge("determine_required_path", "assistant")
builder.add_conditional_edges(
    "assistant",
    route_after_assistant,
    {
        "tools": "tools",
        "decide_generate_research_note": "decide_generate_research_note",
    },
)
builder.add_edge("tools", "assistant")
builder.add_conditional_edges(
    "decide_generate_research_note",
    should_save_research_note,
    {
        True: "determine_filename",
        False: END
    }
)
builder.add_edge("determine_filename", "save_markdown")
builder.add_edge("save_markdown", "save_json")
builder.add_edge("save_json", END)

builder.add_edge("assistant", END)

graph = builder.compile()
