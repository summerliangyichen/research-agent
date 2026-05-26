from __future__ import annotations

import asyncio

import tool


def test_local_rag_search_uses_catalog_selection(monkeypatch) -> None:
    async def fake_select(query: str, candidate_limit: int) -> dict:
        assert query == "python"
        assert candidate_limit == 6
        return {
            "selected_notes": [
                {"title": "Python 历史", "path": r"D:\notes\Python 历史.md"}
            ],
            "selected_paths": [r"D:\notes\Python 历史.md"],
            "reasoning": "命中目录候选",
            "selection_mode": "llm",
        }

    def fake_sync(
        query: str,
        top_k: int,
        min_score: float,
        allow_same_file: bool,
        candidate_paths: list[str] | None = None,
    ) -> dict:
        assert query == "python"
        assert top_k == 3
        assert min_score == tool.DEFAULT_RAG_MIN_SCORE
        assert allow_same_file is False
        assert candidate_paths == [r"D:\notes\Python 历史.md"]
        return {
            "ok": True,
            "query": query,
            "provider": "local_qwen",
            "results": [
                {
                    "score": 0.66,
                    "title": "Python 历史",
                    "section": "核心结论",
                    "path": r"D:\notes\Python 历史.md",
                    "text": "Python 由 Guido van Rossum 创建。",
                }
            ],
            "context": "context",
        }

    monkeypatch.setattr(tool, "_select_candidate_notes_for_query", fake_select)
    monkeypatch.setattr(tool, "_local_rag_search_sync", fake_sync)

    result = asyncio.run(tool.local_rag_search.ainvoke({"query": "python"}))

    assert result["ok"] is True
    assert result["selection_mode"] == "llm"
    assert result["selection_reasoning"] == "命中目录候选"
    assert result["selection_used"] is True
    assert result["selected_notes"][0]["title"] == "Python 历史"


def test_local_rag_search_filepath_bypasses_catalog(monkeypatch) -> None:
    async def should_not_be_called(*args, **kwargs) -> dict:
        raise AssertionError("filepath 模式不应该调用目录候选选择")

    def fake_sync(
        query: str,
        top_k: int,
        min_score: float,
        allow_same_file: bool,
        candidate_paths: list[str] | None = None,
    ) -> dict:
        assert candidate_paths == [r"D:\notes\Python 历史.md"]
        return {
            "ok": True,
            "query": query,
            "provider": "local_qwen",
            "results": [],
            "context": "",
        }

    monkeypatch.setattr(tool, "_select_candidate_notes_for_query", should_not_be_called)
    monkeypatch.setattr(tool, "_local_rag_search_sync", fake_sync)

    result = asyncio.run(
        tool.local_rag_search.ainvoke(
            {
                "query": "python",
                "filepath": r"D:\notes\Python 历史.md",
            }
        )
    )

    assert result["ok"] is True
    assert result["selection_mode"] == "filepath"
    assert result["selection_used"] is False
    assert result["filepath"] == r"D:\notes\Python 历史.md"
