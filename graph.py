from __future__ import annotations

from langchain_deepseek import ChatDeepSeek
from typing import TypedDict, Annotated
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
import re
from datetime import datetime
from pathlib import Path
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from tools import crawl_webpage, fetch_related_urls
import json


WORK_DIR = Path(__file__).parent
OUTPUTS_DIR = WORK_DIR / "outputs"
INDEX_PATH = OUTPUTS_DIR / "index.jsonl"


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    should_search: bool
    filename: str
    query: str
    path: str
    indexed: bool

class search(BaseModel):
    should_search: bool

class save_research_note(BaseModel):
    should_save: bool

class structured_filename(BaseModel):
    filename: str

class summary_note(BaseModel):
    summary_format: str

llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}},
)

tools = [crawl_webpage, fetch_related_urls]
llm_with_tools = llm.bind_tools(tools)

async def should_search(state:AgentState) -> dict:
    structured_llm = llm.with_structured_output(search)
    query = state["query"]
    response = await structured_llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "Decide whether the user's query needs web search. "
                    "Return true for current, factual, research, URL, or source-backed questions. "
                    "Return false for casual chat or tasks that do not need external information."
                )
            ),
            HumanMessage(content=query),
        ]
    )
    return {"should_search": response.should_search}
    
async def assistant_node(state: AgentState) -> AgentState:
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}

async def should_save_research_note(state: AgentState) -> bool:
    if not state["messages"]:
        return False

    content = state["messages"][-1].content
    if not isinstance(content, str):
        return False

    structured_llm = llm.with_structured_output(save_research_note)
    response = await structured_llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "Decide whether the assistant's final answer should be saved as a research note. "
                    "Return true only if it is a complete research-style Markdown note worth saving, "
                    "not a casual reply, short answer, clarification question, error message, or tool/status text."
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
    structured_llm = llm.with_structured_output(structured_filename)
    query = state["query"]
    response = await structured_llm.ainvoke(
        [
            SystemMessage(content="根据用户输入，给出一个适合保存为 Markdown 文件名的中文标题，不要包含路径。"),
            HumanMessage(content=query),
        ]
    )
    return {"filename": response.filename}

def save_markdown(state:AgentState) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    filename = state["filename"]
    content = state["messages"][-1].content

    if filename:
        safe_name = _safe_markdown_filename(filename)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"output_{timestamp}.md"

    target_path = OUTPUTS_DIR / safe_name
    target_path.write_text(content, encoding="utf-8")

    latest_path = OUTPUTS_DIR / "latest.md"
    latest_path.write_text(content, encoding="utf-8")

    return {"path": str(target_path)}


def _safe_markdown_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    if not name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"output_{timestamp}.md"
    if not name.lower().endswith(".md"):
        name = f"{name}.md"
    return name

async def save_json(state: AgentState) -> AgentState:
    content = state["messages"][-1].content
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
        "query":state["query"],
        "filename": state["filename"],
        "path": state["path"],
        "summary": response,
        "created_at": datetime.now().astimezone().isoformat()
    }

    with INDEX_PATH.open("a", encoding = "utf-8") as file:
        file.write(json.dumps(record,ensure_ascii = False)+"\n")

    return {"indexed":True}

builder = StateGraph(AgentState)

builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("decide_generate_research_note", decide_generate_research_note)
builder.add_node("determine_filename", determine_filename)
builder.add_node("save_markdown", save_markdown)
builder.add_node("save_json", save_json)

builder.add_edge(START, "assistant")
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

graph = builder.compile()
