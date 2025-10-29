from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_sync_module():
    module_path = _repo_root() / "scripts" / "utils" / "sync_output_modeling.py"
    spec = importlib.util.spec_from_file_location("sync_output_modeling", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load sync_output_modeling module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def test_sync_updates_modeling_and_config(tmp_path: Path) -> None:
    repo_root = _repo_root()
    base_file = repo_root / "open_provence" / "modeling_open_provence_standalone.py"
    sync = _load_sync_module()

    output_dir = tmp_path / "output"
    run_dir = output_dir / "toy-open-provence-reranker-japanese-test"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create outdated modeling file
    (run_dir / "modeling_open_provence_standalone.py").write_text(
        "# legacy content\n",
        encoding="utf-8",
    )

    # Config with wrong language and missing legacy field
    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "open_provence",
                "splitter_default_language": "en",
                "standalone_process_default_language": "en",
                "modeling_open_provence_default_language": "en",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    states = sync.plan_sync(base_file, output_dir)
    assert len(states) == 1
    state = states[0]
    assert state.modeling_needs_update is True
    assert state.config_needs_update is True
    assert set(state.removed_keys) == {
        "splitter_default_language",
        "standalone_process_default_language",
        "modeling_open_provence_default_language",
    }

    stream = io.StringIO()
    sync.sync_targets(base_file, output_dir, overwrite=True, stream=stream)

    # modeling file should now match base file
    assert (run_dir / "modeling_open_provence_standalone.py").read_text(
        encoding="utf-8"
    ) == base_file.read_text(encoding="utf-8")

    updated_config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in (
        "splitter_default_language",
        "standalone_process_default_language",
        "modeling_open_provence_default_language",
    ):
        assert key not in updated_config

    output = stream.getvalue()
    assert "copied modeling_open_provence_standalone.py" in output
    assert "removed deprecated config keys" in output


def test_sync_skip_when_up_to_date(tmp_path: Path) -> None:
    repo_root = _repo_root()
    base_file = repo_root / "open_provence" / "modeling_open_provence_standalone.py"
    sync = _load_sync_module()

    output_dir = tmp_path / "output"
    run_dir = output_dir / "toy-open-provence-reranker-test"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Up-to-date modeling file
    run_dir.joinpath("modeling_open_provence_standalone.py").write_text(
        base_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "open_provence",
                "some_other_field": "value",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    stream = io.StringIO()
    sync.sync_targets(base_file, output_dir, overwrite=False, stream=stream)

    assert "SKIP (already up to date)" in stream.getvalue()
