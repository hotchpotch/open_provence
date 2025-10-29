"""Helpers for exporting modeling_open_provence_standalone scripts."""

from __future__ import annotations

from pathlib import Path


def write_modeling_open_provence(
    source: Path,
    destination: Path,
) -> None:
    """Copy modeling_open_provence_standalone.py without mutating its contents."""

    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
