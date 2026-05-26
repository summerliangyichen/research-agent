from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek


WORK_DIR = Path(__file__).parent
OUTPUTS_DIR = WORK_DIR / "outputs"
CACHE_PATH = OUTPUTS_DIR / ".rag_embedding_cache.json"
LOCAL_MODEL_DIR = WORK_DIR / "models" / "Qwen3-Embedding-0.6B"
IGNORED_SECTIONS = {"来源", "仍需确认", "参考资料", "参考来源"}
DEFAULT_RAG_MIN_SCORE = 0.2
_LOCAL_EMBEDDING_MODEL: Any | None = None
INDEX_PATH = OUTPUTS_DIR / "index.jsonl"



@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    path: Path
    title: str
    section: str
    text: str


def wikilinks_to_text(markdown: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(1).strip()
        target, separator, alias = raw.partition("|")
        display = alias.strip() if separator else target.strip()
        return display.split("#", 1)[0].strip()

    return re.sub(r"\[\[([^\[\]\n]+?)\]\]", replace, markdown)


def note_title(path: Path, text: str) -> str:
    match = re.search(r"(?m)^#\s+(.+)$", text)
    if match:
        return wikilinks_to_text(match.group(1)).strip()
    return path.stem


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_text(text: str, max_chars: int = 1200, overlap: int = 160) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(paragraph), max_chars - overlap):
                chunk = paragraph[start : start + max_chars].strip()
                if chunk:
                    chunks.append(chunk)
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current.strip())
            tail = current[-overlap:].strip()
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def is_useful_chunk(text: str, min_chars: int = 80) -> bool:
    plain = re.sub(r"(?m)^#{1,6}\s+", "", text).strip()
    return len(plain) >= min_chars


def iter_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = "正文"
    current_lines: list[str] = []

    for line in text.splitlines():
        heading = re.match(r"^#{2,6}\s+(.+)$", line)
        if heading:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = heading.group(1).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return [(title, body) for title, body in sections if body]


def _resolve_markdown_paths(
    outputs_dir: Path = OUTPUTS_DIR,
    markdown_paths: str | Path | list[str | Path] | None = None,
) -> list[Path]:
    if markdown_paths is None:
        return sorted(outputs_dir.glob("*.md"))

    raw_paths = markdown_paths if isinstance(markdown_paths, list) else [markdown_paths]
    resolved_paths: list[Path] = []
    seen: set[Path] = set()

    for raw_path in raw_paths:
        candidate = Path(raw_path)
        candidate_paths: list[Path] = []

        if candidate.is_absolute():
            candidate_paths.append(candidate)
        else:
            candidate_paths.append((WORK_DIR / candidate).resolve())
            candidate_paths.append((outputs_dir / candidate).resolve())

        for path in candidate_paths:
            if path.suffix.lower() != ".md":
                continue
            if not path.exists():
                continue
            normalized = path.resolve()
            if normalized in seen:
                continue
            seen.add(normalized)
            resolved_paths.append(normalized)
            break

    return resolved_paths


def load_chunks(
    outputs_dir: Path = OUTPUTS_DIR,
    markdown_paths: str | Path | list[str | Path] | None = None,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    if not outputs_dir.exists():
        return chunks

    for path in _resolve_markdown_paths(outputs_dir=outputs_dir, markdown_paths=markdown_paths):
        raw_text = path.read_text(encoding="utf-8").strip()
        if not raw_text:
            continue

        title = note_title(path, raw_text)
        plain_text = wikilinks_to_text(raw_text)
        index = 0
        for section, section_text in iter_sections(plain_text):
            if section in IGNORED_SECTIONS:
                continue
            for chunk_text in split_text(section_text):
                if not is_useful_chunk(chunk_text):
                    continue
                index += 1
                chunk_id = stable_hash(f"{path.name}:{index}:{chunk_text}")
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        path=path,
                        title=title,
                        section=section,
                        text=chunk_text,
                    )
                )

    return chunks


def embedding_provider() -> str:
    return os.getenv("EMBEDDING_PROVIDER", "openai_compatible").strip().lower()


def embedding_config() -> tuple[str, str, str]:
    api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1").strip()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()

    if not api_key:
        raise RuntimeError(
            "缺少 EMBEDDING_API_KEY。这个示例使用 OpenAI-compatible embeddings API，"
            "请在 .env 中配置 EMBEDDING_API_KEY、EMBEDDING_BASE_URL、EMBEDDING_MODEL。"
        )
    if not base_url:
        raise RuntimeError("缺少 EMBEDDING_BASE_URL")
    if not model:
        raise RuntimeError("缺少 EMBEDDING_MODEL")

    return api_key, base_url.rstrip("/"), model


def embedding_cache_namespace() -> str:
    provider = embedding_provider()
    if provider == "local_qwen":
        model_path = local_embedding_model_path()
        return stable_hash(f"{provider}|{model_path}")[:16]

    _, base_url, model = embedding_config()
    return stable_hash(f"{provider}|{base_url}|{model}")[:16]


def local_embedding_model_path() -> Path:
    raw_path = os.getenv("LOCAL_EMBEDDING_MODEL", "").strip()
    if raw_path:
        path = Path(raw_path)
        return path if path.is_absolute() else WORK_DIR / path
    return LOCAL_MODEL_DIR


def mask_secret(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "<set>"
    return f"{value[:4]}...{value[-4:]}"


def check_config() -> int:
    chunks = load_chunks()
    provider = embedding_provider()
    print(f"notes_dir: {OUTPUTS_DIR}")
    print(f"chunks: {len(chunks)}")
    print(f"EMBEDDING_PROVIDER: {provider}")

    missing: list[str] = []
    if not chunks:
        missing.append("outputs/*.md")

    if provider == "local_qwen":
        model_path = local_embedding_model_path()
        print(f"LOCAL_EMBEDDING_MODEL: {model_path}")
        if not model_path.exists():
            missing.append("LOCAL_EMBEDDING_MODEL")
    elif provider == "openai_compatible":
        api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
        base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1").strip()
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()

        print(f"EMBEDDING_API_KEY: {mask_secret(api_key)}")
        print(f"EMBEDDING_BASE_URL: {base_url or '<missing>'}")
        print(f"EMBEDDING_MODEL: {model or '<missing>'}")

        if not api_key:
            missing.append("EMBEDDING_API_KEY")
        if not base_url:
            missing.append("EMBEDDING_BASE_URL")
        if not model:
            missing.append("EMBEDDING_MODEL")
    else:
        missing.append("EMBEDDING_PROVIDER=openai_compatible|local_qwen")

    if missing:
        print(f"status: not ready ({', '.join(missing)})")
        return 1

    print("status: ready")
    return 0


def embed_texts(texts: list[str]) -> list[list[float]]:
    provider = embedding_provider()
    if provider == "local_qwen":
        return embed_texts_local_qwen(texts)
    if provider != "openai_compatible":
        raise RuntimeError("EMBEDDING_PROVIDER 只支持 openai_compatible 或 local_qwen")

    api_key, base_url, model = embedding_config()
    response = httpx.post(
        f"{base_url}/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": model, "input": texts},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    data = sorted(payload["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in data]


def embed_texts_local_qwen(texts: list[str]) -> list[list[float]]:
    global _LOCAL_EMBEDDING_MODEL

    if _LOCAL_EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer

        model_path = local_embedding_model_path()
        if not model_path.exists():
            raise RuntimeError(f"本地 embedding 模型不存在：{model_path}")
        _LOCAL_EMBEDDING_MODEL = SentenceTransformer(str(model_path), trust_remote_code=True)

    vectors = _LOCAL_EMBEDDING_MODEL.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def load_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {"embeddings": {}}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def save_cache(cache: dict[str, Any]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_embeddings(chunks: list[TextChunk], batch_size: int = 16) -> dict[str, list[float]]:
    cache = load_cache()
    namespace = embedding_cache_namespace()
    namespaced_cache = cache.setdefault("embeddings_by_config", {})
    embeddings: dict[str, list[float]] = namespaced_cache.setdefault(namespace, {})
    missing = [chunk for chunk in chunks if chunk.chunk_id not in embeddings]

    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        vectors = embed_texts([chunk.text for chunk in batch])
        for chunk, vector in zip(batch, vectors, strict=True):
            embeddings[chunk.chunk_id] = vector
        save_cache(cache)

    return embeddings


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def retrieve(
    query: str,
    chunks: list[TextChunk],
    embeddings: dict[str, list[float]],
    top_k: int,
    min_score: float = 0.0,
    dedupe_by_file: bool = True,
) -> list[tuple[float, TextChunk]]:
    query_vector = embed_texts([query])[0]
    scored = [
        (cosine_similarity(query_vector, embeddings[chunk.chunk_id]), chunk)
        for chunk in chunks
        if chunk.chunk_id in embeddings
    ]
    scored.sort(key=lambda item: item[0], reverse=True)

    results: list[tuple[float, TextChunk]] = []
    seen_paths: set[Path] = set()
    for score, chunk in scored:
        if score < min_score:
            continue
        if dedupe_by_file and chunk.path in seen_paths:
            continue
        results.append((score, chunk))
        seen_paths.add(chunk.path)
        if len(results) >= top_k:
            break

    return results


def make_context(results: list[tuple[float, TextChunk]]) -> str:
    blocks: list[str] = []
    for index, (score, chunk) in enumerate(results, start=1):
        blocks.append(
            f"[{index}] {chunk.title}\n"
            f"path: {chunk.path}\n"
            f"section: {chunk.section}\n"
            f"score: {score:.6f}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(blocks)


def answer_with_llm(query: str, context: str) -> str:
    llm = ChatDeepSeek(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        extra_body={"thinking": {"type": "disabled"}},
    )
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是一个独立 RAG 示例程序中的回答器。"
                    "只能基于给定 context 回答；如果 context 不足，明确说资料不足。"
                    "回答要简洁，并引用片段编号，例如 [1]。"
                )
            ),
            HumanMessage(content=f"问题：{query}\n\ncontext:\n{context}"),
        ]
    )
    return str(response.content).strip()


def normalize_note_path(path: str) -> str:
    return str(Path(path).resolve()).lower()


def load_index_records(index_path: Path = INDEX_PATH) -> list[dict[str, Any]]:
    if not index_path.exists():
        return []

    deduped_by_path: dict[str, dict[str, Any]] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue

        path = str(record.get("path", "")).strip()
        if not path:
            continue
        deduped_by_path[normalize_note_path(path)] = record

    records = list(deduped_by_path.values())
    records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return records
