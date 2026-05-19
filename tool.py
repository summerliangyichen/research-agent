from __future__ import annotations

import json
import os
import re
import asyncio
import time
from html import unescape
from html.parser import HTMLParser
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import httpx
from langchain_core.tools import tool
from requests.exceptions import RequestException
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()



DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
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
    """This tool reads the file with the path path"""
    try:
        with open(path,"r", encoding = "UTF-8") as file:
            return file.read()
    except Exception as e:
        return f"error reading file {str(e)}"




__all__ = [
    "is_static_webpage",
    "crawl_webpage",
    "batch_crawl_webpage",
    "crawl_webpages",
    "fetch_related_urls",
    "search_related_urls",
]
