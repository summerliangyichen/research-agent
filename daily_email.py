from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from graph import graph, _clean_research_markdown
from main import append_run_log, get_final_content, load_agent
from outlook_mcp import send_outlook_email
from tool import search_related_urls


WORK_DIR = Path(__file__).parent
load_dotenv(WORK_DIR / ".env")

DEFAULT_QUERY = "今日新闻"


def make_daily_run_id() -> str:
    return datetime.now().strftime("daily_%Y%m%d_%H%M%S")


def today_title() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def get_daily_query() -> str:
    return os.getenv("DAILY_EMAIL_QUERY", DEFAULT_QUERY).strip() or DEFAULT_QUERY


def get_daily_recipient() -> str:
    recipient = os.getenv("DAILY_EMAIL_TO", "").strip()
    if not recipient:
        raise RuntimeError("缺少环境变量 DAILY_EMAIL_TO")
    return recipient


def fetch_today_news_urls(query: str, max_results: int = 5) -> list[str]:
    return search_related_urls(
        query=query,
        days=1,
        topic="news",
        max_results=max_results,
        search_depth="basic",
    )


def build_daily_request(query: str, urls: list[str], title: str) -> str:
    urls_text = "\n".join(f"- {url}" for url in urls)
    return (
        f"用户原始输入：{query}\n\n"
        "这是 daily email 定时任务。系统已经用 Tavily 搜索了当天新闻 URL，"
        "搜索参数固定为 days=1, topic=news。请不要再次调用 fetch_related_urls。\n\n"
        "请优先调用 batch_crawl_webpage 一次性读取下面这些 URL，然后生成完整 Research Markdown 研究笔记。\n"
        f"这篇笔记的一级标题必须严格写成：# {title}\n\n"
        f"{urls_text}"
    )


def normalize_daily_markdown_title(markdown: str, title: str) -> str:
    text = _clean_research_markdown(markdown)
    if re.search(r"(?m)^#\s+.+$", text):
        return re.sub(r"(?m)^#\s+.+$", f"# {title}", text, count=1).strip()
    return f"# {title}\n\n{text}".strip()


def markdown_to_email_text(markdown: str) -> str:
    text = _clean_research_markdown(markdown)
    text = re.sub(r"(?m)^#\s+(.+)$", r"\1", text)
    text = re.sub(r"(?m)^##\s+(.+)$", r"\n\1\n", text)
    text = re.sub(r"(?m)^###\s+(.+)$", r"\n\1\n", text)
    text = re.sub(r"\[\[([^\]|#\n]+)(?:#[^\]\n]+)?(?:\|([^\]\n]+))?\]\]", _replace_wikilink, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*\n]+)\*", r"\1", text)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    text = re.sub(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)", r"\1 (\2)", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _replace_wikilink(match: re.Match[str]) -> str:
    target = match.group(1).strip()
    alias = match.group(2)
    return alias.strip() if alias else target


async def daily_email() -> dict:
    start_time = time.time()
    run_id = make_daily_run_id()
    query = get_daily_query()
    recipient = get_daily_recipient()
    title = today_title()

    try:
        urls = fetch_today_news_urls(query=query, max_results=5)
        if not urls:
            raise RuntimeError("Tavily 没有返回当天新闻 URL")

        result = await graph.ainvoke(
            {
                "run_id": run_id,
                "query": query,
                "messages": [
                    SystemMessage(content=load_agent()),
                    HumanMessage(content=build_daily_request(query, urls, title)),
                ],
            }
        )

        saved_path = result.get("path")
        final_content = get_final_content(result)
        markdown_body = Path(saved_path).read_text(encoding="utf-8") if saved_path else final_content
        markdown_body = normalize_daily_markdown_title(markdown_body, title)
        if saved_path:
            Path(saved_path).write_text(markdown_body, encoding="utf-8")
        body = markdown_to_email_text(markdown_body)

        email_result = send_outlook_email(
            to=recipient,
            subject=f"Research Agent Daily News - {datetime.now():%Y-%m-%d}",
            body=body,
        )

        record = {
            "run_id": run_id,
            "query": query,
            "status": "success",
            "saved": bool(saved_path),
            "emailed": True,
            "recipient": recipient,
            "elapsed_seconds": round(time.time() - start_time, 3),
            "searched_days": 1,
            "search_topic": "news",
            "source_urls": urls,
        }
        if saved_path:
            record["output_file"] = saved_path
        if result.get("note_id"):
            record["note_id"] = result["note_id"]
        if email_result.get("subject"):
            record["email_subject"] = email_result["subject"]

        append_run_log(record)
        return record

    except Exception as exc:
        record = {
            "run_id": run_id,
            "query": query,
            "status": "error",
            "saved": False,
            "emailed": False,
            "recipient": recipient,
            "elapsed_seconds": round(time.time() - start_time, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        append_run_log(record)
        raise


if __name__ == "__main__":
    result = asyncio.run(daily_email())
    print(result)
