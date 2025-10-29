"""Synchronise modeling_open_provence_standalone.py files in output directories."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


_DEPRECATED_CONFIG_KEYS: tuple[str, ...] = (
    "splitter_default_language",
    "standalone_process_default_language",
    "modeling_open_provence_default_language",
)


@dataclass
class TargetState:
    modeling_path: Path
    config_path: Path | None
    modeling_needs_update: bool
    config_needs_update: bool
    removed_keys: tuple[str, ...]

    def requires_action(self) -> bool:
        return self.modeling_needs_update or self.config_needs_update


def _load_base_content(base_file: Path) -> str:
    if not base_file.exists():
        raise FileNotFoundError(f"Base modeling file not found: {base_file}")
    return base_file.read_text(encoding="utf-8")


def _evaluate_config(modeling_path: Path) -> tuple[Path | None, bool, tuple[str, ...]]:
    config_path = modeling_path.with_name("config.json")
    if not config_path.exists():
        return None, False, ()

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return config_path, False, ()

    if config.get("model_type") != "open_provence":
        return config_path, False, ()

    removed_keys = tuple(key for key in _DEPRECATED_CONFIG_KEYS if key in config)
    return config_path, bool(removed_keys), removed_keys


def _gather_target_states(base_content: str, output_dir: Path) -> list[TargetState]:
    if not output_dir.exists():
        return []

    states: list[TargetState] = []
    for modeling_path in sorted(output_dir.rglob("modeling_open_provence_standalone.py")):
        current_content = modeling_path.read_text(encoding="utf-8")
        modeling_needs_update = current_content != base_content
        config_path, config_needs_update, removed_keys = _evaluate_config(modeling_path)
        states.append(
            TargetState(
                modeling_path=modeling_path,
                config_path=config_path,
                modeling_needs_update=modeling_needs_update,
                config_needs_update=config_needs_update,
                removed_keys=removed_keys,
            )
        )
    return states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the latest modeling_open_provence_standalone.py into every output run (dry run by default)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Apply changes; without this flag the script reports pending updates (dry run).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root directory that contains run outputs (default: ./output).",
    )
    return parser.parse_args()


def plan_sync(base_file: Path, output_dir: Path) -> list[TargetState]:
    base_content = _load_base_content(base_file)
    return _gather_target_states(base_content, output_dir)


def _format_removed_keys(keys: Iterable[str]) -> str:
    formatted = ", ".join(sorted(keys))
    return formatted if formatted else ""


def sync_targets(
    base_file: Path,
    output_dir: Path,
    overwrite: bool,
    *,
    stream: TextIO = sys.stdout,
) -> None:
    if not output_dir.exists():
        print(f"No output directory found at {output_dir}", file=stream)
        return

    base_content = _load_base_content(base_file)
    targets = _gather_target_states(base_content, output_dir)
    if not targets:
        print("No matching modeling_open_provence_standalone.py files found.", file=stream)
        return

    mode = "Applying updates" if overwrite else "Planned updates"
    print(f"{mode} for {len(targets)} target(s):", file=stream)

    any_pending = False

    for state in targets:
        header = f"- {state.modeling_path}"
        if not overwrite:
            if not state.requires_action():
                print(f"{header} → SKIP (already up to date)", file=stream)
                continue

            any_pending = True
            if state.modeling_needs_update:
                print(
                    f"{header} → would copy latest modeling_open_provence_standalone.py",
                    file=stream,
                )
            if state.config_needs_update:
                removed = _format_removed_keys(state.removed_keys)
                print(f"{header} → would remove deprecated config keys: {removed}", file=stream)
            continue

        # overwrite
        if state.modeling_needs_update:
            state.modeling_path.write_text(base_content, encoding="utf-8")
            print(f"{header} → copied modeling_open_provence_standalone.py", file=stream)
        else:
            print(f"{header} → SKIP (already up to date)", file=stream)

        if state.config_needs_update and state.config_path is not None:
            config_path = state.config_path
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for key in state.removed_keys:
                config.pop(key, None)
            config_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            removed = _format_removed_keys(state.removed_keys)
            print(f"{header} → removed deprecated config keys: {removed}", file=stream)

    if not overwrite and any_pending:
        print("Re-run with --overwrite to apply these updates.", file=stream)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    base_file = repo_root / "open_provence" / "modeling_open_provence_standalone.py"
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    sync_targets(base_file, output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
