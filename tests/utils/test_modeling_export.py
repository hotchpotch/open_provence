"""Tests for ``open_provence.utils.modeling_export``."""

from __future__ import annotations

from pathlib import Path

from open_provence.utils.modeling_export import write_modeling_open_provence


def _make_source(tmp_path: Path, content: str) -> Path:
    source = tmp_path / "modeling_open_provence_standalone.py"
    source.write_text(content, encoding="utf-8")
    return source


def test_write_modeling_open_provence_copies_source(tmp_path: Path) -> None:
    content = "DEFAULT_SPLITTER_LANGUAGE = \"auto\"\n"
    source = _make_source(tmp_path, content)
    destination = tmp_path / "out.py"

    write_modeling_open_provence(source, destination)

    assert destination.read_text(encoding="utf-8") == content


def test_write_modeling_open_provence_overwrites_existing(tmp_path: Path) -> None:
    content = "# latest\nDEFAULT_SPLITTER_LANGUAGE = \"auto\"\n"
    source = _make_source(tmp_path, content)
    destination = tmp_path / "out.py"
    destination.write_text("legacy\n", encoding="utf-8")

    write_modeling_open_provence(source, destination)

    assert destination.read_text(encoding="utf-8") == content
