from __future__ import annotations

import json
from pathlib import Path

import tool


def test_save_markdown_note_and_index(tmp_path, monkeypatch) -> None:
    outputs_dir = tmp_path / "outputs"
    index_path = outputs_dir / "index.jsonl"
    monkeypatch.setattr(tool, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(tool, "INDEX_PATH", index_path)

    content = (
        "# [[Python]]\n\n"
        "## 核心结论\n\n"
        "这是一个测试笔记。\n\n"
        "## 来源\n\n"
        "- Python 官网：https://www.python.org/\n"
    )

    saved = tool.save_markdown_note.invoke(
        {"content": content, "title": "Python 测试"}
    )

    assert saved["ok"] is True
    saved_path = Path(saved["path"])
    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8") == content.strip()

    indexed = tool.save_note_index.invoke(
        {
            "query": "python 是什么",
            "content": content,
            "path": saved["path"],
            "title": "Python 测试",
        }
    )

    assert indexed["ok"] is True
    records = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["title"] == "Python 测试"
    assert record["path"] == str(saved_path.resolve())
    assert "Python" in record["links"]
    assert record["sources"][0]["url"] == "https://www.python.org/"


def test_save_markdown_note_rejects_parent_escape(tmp_path, monkeypatch) -> None:
    outputs_dir = tmp_path / "outputs"
    monkeypatch.setattr(tool, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(tool, "INDEX_PATH", outputs_dir / "index.jsonl")

    result = tool.save_markdown_note.invoke(
        {
            "content": "# [[测试]]\n\n## 核心结论\n\n内容",
            "title": "测试",
            "path": "..\\escape.md",
        }
    )

    assert result["ok"] is False
    assert "path 不能包含 . 或 .." in result["error"]
