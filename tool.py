from __future__ import annotations

import json
import os
import re
import asyncio
import time
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from requests.exceptions import RequestException
from tavily import TavilyClient
from dotenv import load_dotenv
from rag import (
    DEFAULT_RAG_MIN_SCORE,
    embedding_provider,
    ensure_embeddings,
    load_chunks,
    load_index_records,
    make_context,
    retrieve,
)

load_dotenv()



DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
)
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR_PATH = Path(WORK_DIR)
OUTPUTS_DIR = WORK_DIR_PATH / "outputs"
BASH_TOOL_MAX_TIMEOUT_SECONDS = 30
BASH_TOOL_MAX_OUTPUT_CHARS = 6000
INDEX_PATH = OUTPUTS_DIR / "index.jsonl"
LOCAL_NOTE_SELECTOR_CATALOG_LIMIT = 40
LOCAL_NOTE_SELECTOR_SUMMARY_CHARS = 280
BASH_TOOL_BLOCKED_PATTERNS = (
    r"(^|[\s;|&])Remove-Item([\s;|&]|$)",
    r"(^|[\s;|&])rm([\s;|&]|$)",
    r"(^|[\s;|&])del([\s;|&]|$)",
    r"(^|[\s;|&])erase([\s;|&]|$)",
    r"(^|[\s;|&])rmdir([\s;|&]|$)",
    r"(^|[\s;|&])rd([\s;|&]|$)",
    r"(^|[\s;|&])Move-Item([\s;|&]|$)",
    r"(^|[\s;|&])mv([\s;|&]|$)",
    r"\bFormat-Volume\b",
    r"\bdiskpart\b",
    r"\bshutdown\b",
    r"\brestart-computer\b",
    r"\bgit\s+reset\b",
    r"\bgit\s+checkout\s+--\b",
)
catalog_selector_llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}},
)


class LocalNoteSelection(BaseModel):
    candidate_ids: list[int] = Field(
        default_factory=list,
        description="最相关的候选文章 catalog_id 列表，按相关性排序",
    )
    reasoning: str = Field(
        default="",
        description="一句话说明为什么选择这些候选文章",
    )


class _ReadableTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_tags: list[str] = []
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_tags.append(tag)
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if self._ignored_tags and self._ignored_tags[-1] == tag:
            self._ignored_tags.pop()
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = _normalize_space(data)
        if not text:
            return

        if self._in_title:
            self.title_parts.append(text)
        elif not self._ignored_tags:
            self.text_parts.append(text)


def is_static_webpage(url: str, timeout: float = 10.0) -> bool:
    """判断一个网页是否更像静态网页。

    返回 True 表示页面更像静态/静态生成页面；返回 False 表示页面更像动态或混合型页面。
    这是基于 HTML、脚本、运行时数据和响应头的启发式判断，不是浏览器级绝对判定。
    """

    page = _fetch_webpage(url, timeout=timeout)
    return _looks_static(page["html"], page["headers"], page["final_url"])

async def _crawl_webpage_once(
    url: str,
    max_chars: int = 3000,
    max_json_chars: int = 20000,
    timeout: float = 10.0,
) -> dict[str, Any]:
    try:
        page = await _fetch_webpage_async(url, timeout=timeout)
        return _build_crawl_result(
            page,
            max_chars=max_chars,
            max_json_chars=max_json_chars,
        )
    except httpx.HTTPStatusError as exc:
        return _crawl_error(
            url,
            "HTTPStatusError",
            str(exc),
            status_code=exc.response.status_code,
        )
    except (httpx.RequestError, ValueError) as exc:
        return _crawl_error(url, type(exc).__name__, str(exc))
    except Exception as exc:
        return _crawl_error(url, type(exc).__name__, str(exc))


@tool
async def crawl_webpage(
    url: str,
    max_chars: int = 3000,
    max_json_chars: int = 20000,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Fetch a webpage and return a unified JSON document.

    The tool first reads the HTTP response. If the response itself is JSON, it parses that
    directly. If it is HTML, it extracts readable text plus embedded JSON blocks such as
    JSON-LD and Next.js __NEXT_DATA__. This keeps callers from deciding static vs dynamic
    before crawling.
    """

    return await _crawl_webpage_once(
        url,
        max_chars=max_chars,
        max_json_chars=max_json_chars,
        timeout=timeout,
    )

async def crawl_webpages(
    urls: list[str],
    max_chars: int = 3000,
    max_json_chars: int = 20000,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    cleaned_urls: list[str] = []
    seen_urls: set[str] = set()

    for url in urls:
        if not isinstance(url, str):
            continue

        cleaned_url = url.strip()
        if not cleaned_url or cleaned_url in seen_urls:
            continue

        cleaned_urls.append(cleaned_url)
        seen_urls.add(cleaned_url)

    return await asyncio.gather(
        *[
            _crawl_webpage_once(
                url,
                max_chars=max_chars,
                max_json_chars=max_json_chars,
                timeout=timeout,
            )
            for url in cleaned_urls
        ]
    )


@tool
async def batch_crawl_webpage(
    urls: list[str],
    max_chars: int = 3000,
    max_json_chars: int = 20000,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Fetch multiple webpages concurrently.

    Each result has the same structure as crawl_webpage. The function keeps input order,
    skips empty or duplicate URLs, and fetches all remaining URLs in one call.
    """

    return await crawl_webpages(
        urls,
        max_chars=max_chars,
        max_json_chars=max_json_chars,
        timeout=timeout,
    )
    


def _build_crawl_result(
    page: dict[str, Any],
    max_chars: int,
    max_json_chars: int,
) -> dict[str, Any]:
    content_type = page["headers"].get("content-type", "")
    body = page["html"]

    json_body = _try_parse_json(body)
    if _is_json_content_type(content_type) and json_body is not None:
        return {
            "ok": True,
            "url": page["final_url"],
            "status_code": page["status_code"],
            "content_type": content_type,
            "response_type": "json",
            "is_static": False,
            "title": _guess_json_title(json_body),
            "text": _json_to_text(json_body, max_chars=max_chars),
            "json": _pack_json_value(json_body, max_chars=max_json_chars),
            "json_blocks": [],
            "json_block_count": 0,
            "needs_browser": False,
        }

    title, text = _extract_title_and_text(body)
    json_blocks = _extract_json_blocks(body, max_total_chars=max_json_chars)

    return {
        "ok": True,
        "url": page["final_url"],
        "status_code": page["status_code"],
        "content_type": content_type,
        "response_type": "html",
        "is_static": _looks_static(body, page["headers"], page["final_url"]),
        "title": title,
        "text": text[:max_chars],
        "json_blocks": json_blocks,
        "json_block_count": len(json_blocks),
        "needs_browser": _needs_browser_rendering(body, text, json_blocks),
    }


def _crawl_error(
    url: str,
    error_type: str,
    message: str,
    status_code: int | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "url": url,
        "status_code": status_code,
        "response_type": "error",
        "title": "",
        "text": "",
        "json_blocks": [],
        "json_block_count": 0,
        "needs_browser": False,
        "error_type": error_type,
        "error": message,
    }


def _fetch_webpage(url: str, timeout: float = 10.0, max_bytes: int = 2_000_000) -> dict[str, Any]:
    normalized_url = _normalize_url(url)
    request = Request(
        normalized_url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/json,application/xml;q=0.9,*/*;q=0.8",
        },
    )

    with urlopen(request, timeout=timeout) as response:
        body = response.read(max_bytes)
        charset = response.headers.get_content_charset() or "utf-8"

        return {
            "final_url": response.geturl(),
            "status_code": getattr(response, "status", response.getcode()),
            "headers": {key.lower(): value for key, value in response.headers.items()},
            "html": body.decode(charset, errors="replace"),
        }


async def _fetch_webpage_async(
    url: str,
    timeout: float = 10.0,
    max_bytes: int = 2_000_000,
) -> dict[str, Any]:
    normalized_url = _normalize_url(url)
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/json,application/xml;q=0.9,*/*;q=0.8",
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(normalized_url, headers=headers)
        response.raise_for_status()

    body = response.content[:max_bytes]
    charset = response.encoding or "utf-8"

    return {
        "final_url": str(response.url),
        "status_code": response.status_code,
        "headers": {key.lower(): value for key, value in response.headers.items()},
        "html": body.decode(charset, errors="replace"),
    }


def _looks_static(html: str, headers: dict[str, str], final_url: str) -> bool:
    lower_html = html.lower()
    dynamic_score = 0
    static_score = 0

    dynamic_markers = (
        "__next_data__",
        "__nuxt__",
        "__apollo_state__",
        "window.__initial_state__",
        "data-reactroot",
        "ng-version",
        "webpack",
        "hydrate",
        "/api/",
        "graphql",
    )
    dynamic_score += sum(2 for marker in dynamic_markers if marker in lower_html)

    script_count = lower_html.count("<script")
    if script_count >= 8:
        dynamic_score += 3
    elif script_count <= 2:
        static_score += 2

    if re.search(r"<script[^>]+type=[\"']application/(json|ld\+json)[\"']", lower_html):
        dynamic_score += 1

    if re.search(r"<(main|article|h1)\b", lower_html):
        static_score += 1

    content_type = headers.get("content-type", "").lower()
    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        static_score += 2

    cache_control = headers.get("cache-control", "").lower()
    vary = headers.get("vary", "").lower()

    if any(token in cache_control for token in ("private", "no-cache", "no-store", "must-revalidate")):
        dynamic_score += 2
    if "set-cookie" in headers:
        dynamic_score += 2
    if any(token in vary for token in ("cookie", "authorization", "user-agent", "accept-language")):
        dynamic_score += 1
    if "public" in cache_control and ("immutable" in cache_control or _max_age_seconds(cache_control) >= 3600):
        static_score += 2

    path = urlparse(final_url).path.lower()
    if path.endswith((".html", ".htm")):
        static_score += 1

    return static_score >= dynamic_score


def _normalize_url(url: str) -> str:
    stripped = url.strip()
    if not stripped:
        raise ValueError("url 不能为空")

    if "://" not in stripped:
        stripped = f"https://{stripped}"

    parsed = urlparse(stripped)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("只支持 http/https 网页 URL")

    return stripped


def _extract_title_and_text(html: str) -> tuple[str, str]:
    parser = _ReadableTextParser()
    parser.feed(html)

    title = _normalize_space(" ".join(parser.title_parts))
    text = _normalize_space(" ".join(parser.text_parts))
    return unescape(title), unescape(text)


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _max_age_seconds(cache_control: str) -> int:
    match = re.search(r"max-age=(\d+)", cache_control)
    return int(match.group(1)) if match else 0


def _is_json_content_type(content_type: str) -> bool:
    media_type = content_type.lower().split(";", maxsplit=1)[0].strip()
    return media_type == "application/json" or media_type.endswith("+json")


def _try_parse_json(value: str) -> Any | None:
    stripped = value.strip()
    if not stripped:
        return None

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _extract_json_blocks(html: str, max_total_chars: int) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    used_chars = 0
    seen_sources: set[str] = set()

    script_pattern = re.compile(
        r"<script\b(?P<attrs>[^>]*)>(?P<body>.*?)</script>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    for index, match in enumerate(script_pattern.finditer(html), start=1):
        attrs = match.group("attrs")
        body = match.group("body").strip()
        script_type = _html_attr(attrs, "type").lower()
        script_id = _html_attr(attrs, "id")
        is_json_script = (
            script_type in {"application/json", "application/ld+json"}
            or script_id in {"__NEXT_DATA__", "__NUXT_DATA__"}
        )

        if not is_json_script:
            continue

        parsed = _try_parse_json(body)
        if parsed is None:
            parsed = _try_parse_json(unescape(body))
        if parsed is None:
            continue

        source = f"script#{script_id}" if script_id else f"script[{index}]"
        block = _json_block(source, script_type or "application/json", parsed, max_total_chars - used_chars)
        used_chars += block["char_count"]
        blocks.append(block)
        seen_sources.add(source)

        if used_chars >= max_total_chars:
            return blocks

    assignment_pattern = re.compile(
        r"(?:window\.)?(?P<name>__INITIAL_STATE__|__APOLLO_STATE__|__NUXT__)\s*=",
        flags=re.IGNORECASE,
    )
    for match in assignment_pattern.finditer(html):
        source = f"assignment:{match.group('name')}"
        if source in seen_sources:
            continue

        raw_json = _extract_balanced_json(html, start=match.end())
        if raw_json is None:
            continue

        parsed = _try_parse_json(raw_json)
        if parsed is None:
            continue

        block = _json_block(source, "application/json", parsed, max_total_chars - used_chars)
        used_chars += block["char_count"]
        blocks.append(block)
        seen_sources.add(source)

        if used_chars >= max_total_chars:
            break

    return blocks


def _html_attr(attrs: str, name: str) -> str:
    match = re.search(rf"\b{name}\s*=\s*([\"'])(.*?)\1", attrs, flags=re.IGNORECASE | re.DOTALL)
    return match.group(2).strip() if match else ""


def _json_block(source: str, json_type: str, value: Any, max_chars: int) -> dict[str, Any]:
    packed = _pack_json_value(value, max_chars=max(0, max_chars))
    return {
        "source": source,
        "type": json_type,
        **packed,
    }


def _pack_json_value(value: Any, max_chars: int) -> dict[str, Any]:
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) <= max_chars:
        return {
            "char_count": len(serialized),
            "truncated": False,
            "data": value,
        }

    return {
        "char_count": len(serialized),
        "truncated": True,
        "data_preview": serialized[:max_chars],
    }


def _extract_balanced_json(source: str, start: int) -> str | None:
    opening_index = -1
    opening_char = ""
    for index in range(start, len(source)):
        if source[index] in "{[":
            opening_index = index
            opening_char = source[index]
            break

        if not source[index].isspace():
            return None

    if opening_index == -1:
        return None

    closing_char = "}" if opening_char == "{" else "]"
    stack = [closing_char]
    in_string = False
    escape_next = False

    for index in range(opening_index + 1, len(source)):
        char = source[index]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return source[opening_index : index + 1]

    return None


def _guess_json_title(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("title", "headline", "name", "description"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return _normalize_space(candidate)

        for child in value.values():
            title = _guess_json_title(child)
            if title:
                return title

    if isinstance(value, list):
        for child in value:
            title = _guess_json_title(child)
            if title:
                return title

    return ""


def _json_to_text(value: Any, max_chars: int) -> str:
    parts: list[str] = []
    _collect_json_strings(value, parts, max_chars=max_chars)
    return _normalize_space(" ".join(parts))[:max_chars]


def _collect_json_strings(value: Any, parts: list[str], max_chars: int) -> None:
    if sum(len(part) for part in parts) >= max_chars:
        return

    if isinstance(value, str):
        text = _normalize_space(value)
        if text and not text.startswith(("http://", "https://")):
            parts.append(text)
        return

    if isinstance(value, dict):
        preferred_keys = ("title", "headline", "name", "description", "summary", "text", "content")
        for key in preferred_keys:
            if key in value:
                _collect_json_strings(value[key], parts, max_chars=max_chars)

        for key, child in value.items():
            if key not in preferred_keys:
                _collect_json_strings(child, parts, max_chars=max_chars)
        return

    if isinstance(value, list):
        for child in value:
            _collect_json_strings(child, parts, max_chars=max_chars)


def _needs_browser_rendering(html: str, text: str, json_blocks: list[dict[str, Any]]) -> bool:
    lower_html = html.lower()
    framework_shell_markers = (
        'id="root"',
        "id='root'",
        'id="app"',
        "id='app'",
        "data-reactroot",
        "ng-version",
        "vite",
    )

    return (
        len(text) < 200
        and not json_blocks
        and any(marker in lower_html for marker in framework_shell_markers)
    )

def search_related_urls(
    query: str,
    days: int | None = None,
    topic: Literal["general", "news", "finance"] = "general",
    max_results: int = 5,
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
) -> list[str]:
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY") or os.getenv("TVLY_API_KEY"))
    last_error = ""
    max_results = max(1, min(max_results, 20))

    for attempt in range(1, 4):
        try:
            response = tavily_client.search(
                query,
                topic=topic,
                search_depth=search_depth,
                max_results=max_results,
                timeout=20,
                days=days,
            )
            break
        except RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 3:
                time.sleep(attempt * 2)
                continue
            raise RuntimeError(
                "SEARCH_ERROR: Tavily 搜索失败，可能是当前网络或 SSL 连接不稳定。"
                f"已重试 {attempt} 次。错误：{last_error}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"SEARCH_ERROR: Tavily 搜索失败。错误：{type(exc).__name__}: {exc}"
            ) from exc

    urls = [result["url"] for result in response.get("results", []) if result.get("url")]
    return urls


@tool
def fetch_related_urls(
    query: str,
    days: int | None = None,
    topic: Literal["general", "news", "finance"] = "general",
    max_results: int = 5,
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
) -> str:
    """
    This function uses the Tavily API to search web pages related to the query.
    You should use this function only if the query does not contain enough URLs.

    Args:
        query: Search query.
        days: Restrict results to recent N days. Use None when the query is not time-sensitive.
        topic: Tavily topic. Use general for most research, news for current events, finance for finance topics.
        max_results: Number of URLs to return, between 1 and 20.
        search_depth: Search depth. Use basic by default; advanced for complex research.
    """
    try:
        urls = search_related_urls(
            query=query,
            days=days,
            topic=topic,
            max_results=max_results,
            search_depth=search_depth,
        )
    except RuntimeError as exc:
        return str(exc)

    if not urls:
        return "NO_RESULTS: Tavily 没有返回可用 URL。"

    urls_text = "\n".join(f"- {url}" for url in urls)

    return urls_text

@tool
def read_file(path:str) -> str:
    """Read a UTF-8 text file from disk."""
    try:
        with open(path, "r", encoding="UTF-8") as file:
            return file.read()
    except Exception as exc:
        return f"error reading file {str(exc)}"


def _coerce_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _clean_markdown_content(content: Any) -> str:
    text = _coerce_text(content).strip()
    if not text:
        return text

    heading_match = re.search(r"(?m)^#\s+", text)
    if heading_match:
        return text[heading_match.start():].strip()
    return text


def _wikilinks_to_text(markdown: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw_link = match.group(1).strip()
        target, separator, alias = raw_link.partition("|")
        display_text = alias.strip() if separator else target.strip()
        display_text = display_text.split("#", 1)[0].strip()
        display_text = display_text.split("^", 1)[0].strip()
        return display_text

    return re.sub(r"\[\[([^\[\]\n]+?)\]\]", replace, markdown)


def _is_windows_reserved_name(name: str) -> bool:
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        *{f"COM{i}" for i in range(1, 10)},
        *{f"LPT{i}" for i in range(1, 10)},
    }
    return Path(name).stem.upper() in reserved_names


def _safe_markdown_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
    name = name.rstrip(" .")
    if not name:
        name = datetime.now().strftime("output_%Y%m%d_%H%M%S")
    if not name.lower().endswith(".md"):
        name = f"{name}.md"
    if _is_windows_reserved_name(name):
        name = f"_{name}"
    return name


def _normalize_filename_label(filename: str) -> str:
    name = _coerce_text(filename).strip()
    name = name.removeprefix("#").strip()
    name = _wikilinks_to_text(name)
    name = Path(name).name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip()
    if name.lower().endswith(".md"):
        name = name[:-3].strip()
    if not name:
        return datetime.now().strftime("output_%Y%m%d_%H%M%S")
    return name


def _validate_safe_path_part(part: str) -> None:
    if part in {"", ".", ".."}:
        raise ValueError("path 不能包含空路径段、. 或 ..")
    if part != part.rstrip(" ."):
        raise ValueError("path 不能包含以空格或点结尾的路径段")
    if re.search(r'[<>:"|?*\x00-\x1f]', part):
        raise ValueError("path 包含 Windows 非法字符")
    if _is_windows_reserved_name(part):
        raise ValueError("path 包含 Windows 保留名称")


def _safe_output_path(path: str | None, title: str) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if path:
        raw_path = _coerce_text(path).strip().strip("'\"“”` ")
        if not raw_path:
            raise ValueError("path 不能为空")

        candidate = Path(raw_path)
        if not candidate.is_absolute() and candidate.parts:
            leading = candidate.parts[0].lower()
            if leading in {OUTPUTS_DIR.name.lower(), "outputs"}:
                remaining_parts = candidate.parts[1:]
                candidate = Path(*remaining_parts) if remaining_parts else Path(candidate.name)
        if any(part in {".", ".."} for part in candidate.parts):
            raise ValueError("path 不能包含 . 或 ..")

        filename = _safe_markdown_filename(candidate.name)
        if candidate.suffix and candidate.suffix.lower() != ".md":
            raise ValueError("只允许保存 Markdown 文件（.md）")

        target_path = candidate.with_name(filename) if candidate.is_absolute() else OUTPUTS_DIR / candidate.parent / filename
        resolved = target_path.resolve()
        base = OUTPUTS_DIR.resolve()
        try:
            relative_path = resolved.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"path 必须位于 {base} 内") from exc

        for part in relative_path.parts:
            _validate_safe_path_part(part)
        return resolved

    safe_name = _safe_markdown_filename(title)
    return (OUTPUTS_DIR / safe_name).resolve()


def _extract_markdown_title(content: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", content)
    if not match:
        return ""
    return _normalize_filename_label(match.group(1))


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


def _extract_sources_from_markdown(markdown: str) -> list[dict[str, Any]]:
    match = re.search(r"(?ms)^##\s+来源\s*\n(?P<body>.+)$", markdown)
    body = match.group("body") if match else markdown
    sources: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        url_match = re.search(r"https?://\S+", line)
        if not url_match:
            continue
        url = url_match.group(0).rstrip(").,")
        if url in seen_urls:
            continue

        label = line.removeprefix("-").strip()
        label = label.replace(url_match.group(0), "").strip(" ：:-")
        site = urlparse(url).netloc
        sources.append(
            {
                "title": label or url,
                "url": url,
                **({"site": site} if site else {}),
            }
        )
        seen_urls.add(url)

    return sources


def _summarize_markdown_for_index(content: str, limit: int = 500) -> str:
    text = re.sub(r"(?ms)^##\s+来源\s*\n.*$", "", content).strip()
    text = _wikilinks_to_text(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _make_note_id(title: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = _safe_markdown_filename(title).removesuffix(".md")
    return f"{timestamp}_{name}"


@tool
def save_markdown_note(
    content: str,
    title: str = "",
    path: str | None = None,
) -> dict[str, Any]:
    """Save a completed Markdown research note under outputs/.

    Pass path only when the user explicitly requested a file name or relative path.
    """

    cleaned_content = _clean_markdown_content(content)
    if not cleaned_content:
        return {"ok": False, "error": "content 不能为空"}

    resolved_title = _normalize_filename_label(title) if title.strip() else _extract_markdown_title(cleaned_content)
    if not resolved_title:
        resolved_title = datetime.now().strftime("output_%Y%m%d_%H%M%S")

    try:
        target_path = _safe_output_path(path, resolved_title)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(cleaned_content, encoding="utf-8")

    return {
        "ok": True,
        "title": resolved_title,
        "path": str(target_path),
        "filename": target_path.name,
    }


@tool
def save_note_index(
    query: str,
    content: str,
    path: str,
    title: str = "",
) -> dict[str, Any]:
    """Append a structured note record to outputs/index.jsonl."""

    cleaned_content = _clean_markdown_content(content)
    if not cleaned_content:
        return {"ok": False, "error": "content 不能为空"}

    resolved_title = _normalize_filename_label(title) if title.strip() else _extract_markdown_title(cleaned_content)
    if not resolved_title:
        resolved_title = Path(path).stem or datetime.now().strftime("output_%Y%m%d_%H%M%S")

    note_path = str(Path(path).resolve())
    links = _extract_wikilinks(cleaned_content)
    sources = _extract_sources_from_markdown(cleaned_content)
    summary = _summarize_markdown_for_index(cleaned_content)
    note_id = _make_note_id(resolved_title)

    record = {
        "note_id": note_id,
        "run_id": "",
        "title": resolved_title,
        "query": query,
        "filename": resolved_title,
        "path": note_path,
        "summary": summary,
        "links": links,
        "sources": sources,
        "created_at": datetime.now().astimezone().isoformat(),
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "ok": True,
        "note_id": note_id,
        "title": resolved_title,
        "path": note_path,
        "links": links,
        "sources": sources,
    }


def _is_blocked_shell_command(command: str) -> str | None:
    normalized = command.strip()
    for pattern in BASH_TOOL_BLOCKED_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return pattern
    return None


@tool
async def bash_tool(command: str, timeout_seconds: int = 10) -> dict[str, Any]:
    """Run a short, non-destructive PowerShell command in the repository root.

    Despite the tool name, this project runs on Windows, so commands are executed through
    PowerShell. Use it for inspection commands such as Get-ChildItem, Get-Content, git
    status, python --version, and small verification commands. Destructive commands are
    blocked.
    """

    command = command.strip()
    if not command:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": "command 不能为空",
        }

    blocked_pattern = _is_blocked_shell_command(command)
    if blocked_pattern:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"command 被安全策略拦截：{blocked_pattern}",
        }

    timeout = max(1, min(int(timeout_seconds), BASH_TOOL_MAX_TIMEOUT_SECONDS))
    try:
        process = await asyncio.create_subprocess_exec(
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
            cwd=WORK_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()
        return {
            "ok": False,
            "returncode": None,
            "stdout": stdout_bytes.decode("utf-8", errors="replace")[:BASH_TOOL_MAX_OUTPUT_CHARS],
            "stderr": f"command 超过 {timeout} 秒未完成",
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc)[:BASH_TOOL_MAX_OUTPUT_CHARS],
        }

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    return {
        "ok": process.returncode == 0,
        "returncode": process.returncode,
        "stdout": stdout[:BASH_TOOL_MAX_OUTPUT_CHARS],
        "stderr": stderr[:BASH_TOOL_MAX_OUTPUT_CHARS],
    }


def _tokenize_index_text(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def _score_index_record(query: str, record: dict[str, Any]) -> float:
    query_tokens = _tokenize_index_text(query)
    haystack = " ".join(
        [
            str(record.get("title", "")),
            str(record.get("query", "")),
            str(record.get("summary", "")),
            " ".join(record.get("links", []) if isinstance(record.get("links"), list) else []),
        ]
    )
    haystack_tokens = set(_tokenize_index_text(haystack))
    return float(sum(1 for token in query_tokens if token in haystack_tokens))


def _prefilter_index_records(query: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(records) <= LOCAL_NOTE_SELECTOR_CATALOG_LIMIT:
        return records

    ranked = sorted(
        records,
        key=lambda record: (_score_index_record(query, record), str(record.get("created_at", ""))),
        reverse=True,
    )
    return ranked[:LOCAL_NOTE_SELECTOR_CATALOG_LIMIT]


def _compact_index_record(record: dict[str, Any], catalog_id: int) -> dict[str, Any]:
    links = record.get("links", [])
    if not isinstance(links, list):
        links = []

    summary = str(record.get("summary", "")).strip()
    if len(summary) > LOCAL_NOTE_SELECTOR_SUMMARY_CHARS:
        summary = summary[:LOCAL_NOTE_SELECTOR_SUMMARY_CHARS].rstrip() + "..."

    return {
        "catalog_id": catalog_id,
        "title": str(record.get("title", "")).strip(),
        "summary": summary,
        "links": links[:8],
        "path": str(record.get("path", "")).strip(),
        "note_id": str(record.get("note_id", "")).strip(),
    }


def _build_selection_result(
    notes: list[dict[str, Any]],
    reasoning: str,
    selection_mode: str,
    ok: bool = True,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "reasoning": reasoning,
        "selection_mode": selection_mode,
        "selected_notes": notes,
        "selected_paths": [item["path"] for item in notes if item.get("path")],
    }


async def _select_candidate_notes_for_query(
    query: str,
    candidate_limit: int,
) -> dict[str, Any]:
    records = load_index_records()
    if not records:
        return _build_selection_result(
            [],
            "index.jsonl 不存在或没有可用记录，已跳过目录候选选择",
            selection_mode="disabled",
            ok=False,
        )

    catalog_records = _prefilter_index_records(query, records)
    catalog = [
        _compact_index_record(record, catalog_id=index)
        for index, record in enumerate(catalog_records, start=1)
    ]
    fallback_notes = catalog[:candidate_limit]

    try:
        selector = catalog_selector_llm.with_structured_output(LocalNoteSelection)
        response = await selector.ainvoke(
            [
                SystemMessage(
                    content=(
                        "你负责从本地知识库目录中挑选最相关的候选文章，供后续 RAG 检索正文片段。\n"
                        "给定用户问题和文章目录，每条目录只包含标题、摘要、links 和 path。\n"
                        "请返回最相关的 catalog_id 列表，最多不要超过用户要求的数量。\n"
                        "优先选择直接相关、主题明确的文章；不要为了凑数量加入弱相关项。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"用户问题：{query}\n\n"
                        f"最多选择：{candidate_limit} 篇\n\n"
                        f"文章目录：\n{json.dumps(catalog, ensure_ascii=False, indent=2)}"
                    )
                ),
            ]
        )
    except Exception as exc:
        return _build_selection_result(
            fallback_notes,
            (
                "目录候选选择调用失败，已回退到基于 title/query/summary/links 的词项匹配。"
                f" 错误：{type(exc).__name__}"
            ),
            selection_mode="fallback",
        )

    selected_ids: list[int] = []
    for value in response.candidate_ids:
        if isinstance(value, int) and 1 <= value <= len(catalog):
            if value not in selected_ids:
                selected_ids.append(value)

    selected_notes = [item for item in catalog if item["catalog_id"] in selected_ids][:candidate_limit]

    if not selected_notes:
        return _build_selection_result(
            fallback_notes,
            "LLM 没有选出候选文章，已回退到词项匹配后的前几条目录记录",
            selection_mode="fallback",
        )

    reasoning = response.reasoning.strip() or "已根据目录摘要挑选候选文章"
    return _build_selection_result(
        selected_notes,
        reasoning,
        selection_mode="llm",
    )


def _local_rag_search_sync(
    query: str,
    top_k: int,
    min_score: float,
    allow_same_file: bool,
    candidate_paths: list[str] | None = None,
) -> dict[str, Any]:
    chunks = load_chunks(markdown_paths=candidate_paths) if candidate_paths else load_chunks()
    if not chunks:
        return {
            "ok": False,
            "query": query,
            "provider": embedding_provider(),
            "results": [],
            "context": "",
            "error": "outputs/ 下没有可检索的 Markdown 笔记",
        }

    embeddings = ensure_embeddings(chunks)
    results = retrieve(
        query,
        chunks,
        embeddings,
        top_k=max(1, top_k),
        min_score=max(0.0, min_score),
        dedupe_by_file=not allow_same_file,
    )
    if not results:
        return {
            "ok": False,
            "query": query,
            "provider": embedding_provider(),
            "results": [],
            "context": "",
            "error": "没有检索到满足条件的本地笔记片段",
        }

    serialized_results = [
        {
            "score": round(score, 6),
            "title": chunk.title,
            "section": chunk.section,
            "path": str(chunk.path),
            "text": chunk.text,
        }
        for score, chunk in results
    ]
    return {
        "ok": True,
        "query": query,
        "provider": embedding_provider(),
        "results": serialized_results,
        "context": make_context(results),
    }


@tool
async def local_rag_search(
    query: str,
    top_k: int = 3,
    min_score: float = DEFAULT_RAG_MIN_SCORE,
    allow_same_file: bool = False,
    filepath: str | None = None,
) -> dict[str, Any]:
    """Search local Markdown notes with embeddings and return retrieved context.

    This tool first uses outputs/index.jsonl summaries to narrow candidate notes, then
    runs embedding retrieval over candidate note bodies from outputs/*.md. It returns
    the retrieved chunks plus a preformatted context string that the calling model can
    use for synthesis.
    """

    query = query.strip()
    if not query:
        return {
            "ok": False,
            "query": query,
            "provider": embedding_provider(),
            "results": [],
            "context": "",
            "error": "query 不能为空",
        }

    forced_paths = [filepath.strip()] if isinstance(filepath, str) and filepath.strip() else None
    selection = (
        {
            "selected_notes": [],
            "reasoning": "用户显式指定了 filepath，已跳过目录候选选择",
            "selection_mode": "filepath",
            "selected_paths": forced_paths or [],
        }
        if forced_paths
        else await _select_candidate_notes_for_query(query, candidate_limit=max(3, min(8, top_k * 2)))
    )
    result = await asyncio.to_thread(
        _local_rag_search_sync,
        query,
        max(1, int(top_k)),
        float(min_score),
        bool(allow_same_file),
        selection.get("selected_paths"),
    )
    result["selected_notes"] = selection.get("selected_notes", [])
    result["selection_reasoning"] = selection.get("reasoning", "")
    result["selection_mode"] = selection.get("selection_mode", "disabled")
    result["selection_used"] = bool(selection.get("selected_notes"))
    if forced_paths:
        result["filepath"] = forced_paths[0]
    return result


__all__ = [
    "is_static_webpage",
    "crawl_webpage",
    "batch_crawl_webpage",
    "crawl_webpages",
    "fetch_related_urls",
    "search_related_urls",
    "read_file",
    "bash_tool",
    "local_rag_search",
    "save_markdown_note",
    "save_note_index",
]
