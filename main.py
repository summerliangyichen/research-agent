import sys
import asyncio
import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from graph import graph
from datetime import datetime
from pathlib import Path
import json

WORK_PATH = Path(__file__).parent
OUTPUTS_DIR = Path(__file__).parent / "outputs"
RUNS_LOG = OUTPUTS_DIR / "runs.jsonl"
AGENT_PATH = WORK_PATH / "AGENTS.md"

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


def append_run_log(record: dict) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    record["created_at"] = datetime.now().astimezone().isoformat()

    with RUNS_LOG.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_agent() -> str:
    if not AGENT_PATH.exists():
        AGENT_PATH.write_text("", encoding="utf-8")
        return ""
    return AGENT_PATH.read_text(encoding="utf-8")


def get_final_content(result: dict) -> str:
    content = result["messages"][-1].content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in content
        )

    return str(content)


async def main():
    while True:
        start_time = time.time()
        query = input("enter query： ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        user_request = f"请围绕以下问题或主题进行 research，并生成 Markdown 研究笔记：{query}"
        try:
            result = await graph.ainvoke(
                {
                    "query": query,
                    "messages": [
                        SystemMessage(content=load_agent()),
                        HumanMessage(content=user_request),
                    ]
                }
            )
        except Exception as exc:
            append_run_log(
                {
                    "query": query,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise

        final_content = get_final_content(result)
        saved_path = result.get("path")

        log_record = {
            "query": query,
            "status": "success",
            "saved": bool(saved_path),
        }
        if saved_path:
            log_record["output_file"] = saved_path
    
        append_run_log(
            log_record
        )

        print(final_content)
        if saved_path:
            print(f"\n已保存：{saved_path}")
        end_time = time.time()
        print('程序运行时间：%s 秒' % (end_time - start_time))

if __name__ == "__main__":
    asyncio.run(main())
