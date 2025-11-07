from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "context-relevance-datasets"
    / "generate_ds_from_sentense_transformer.py"
)


def build_source_dataset(root: Path) -> Path:
    rows = 20
    data = {
        "question": [f"question {i}" for i in range(rows)],
        "answer": [f"answer {i}" for i in range(rows)],
        "neg1": [f"neg1 {i}" for i in range(rows)],
        "neg2": [f"neg2 {i}" for i in range(rows)],
    }
    dataset = Dataset.from_dict(data)
    dataset_dict = DatasetDict({"train": dataset})
    source_path = root / "source_ds"
    dataset_dict.save_to_disk(source_path)
    return source_path


def test_generate_from_local_dataset(tmp_path):
    source_path = build_source_dataset(tmp_path)
    output_root = tmp_path / "converted"

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--dataset",
        str(source_path),
        "--lang",
        "en",
        "--output-root",
        str(output_root),
        "--overwrite",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2])

    output_dirs = list(output_root.iterdir())
    assert len(output_dirs) == 1
    converted = load_from_disk(output_dirs[0])
    assert isinstance(converted, DatasetDict)
    assert set(converted.keys()) == {"train", "validation", "test"}
    first = converted["train"][0]
    assert first["query"].startswith("question")
    assert first["texts"][0].startswith("answer")
    assert first["labels"][0] == 1
    assert all(label in {0, 1} for label in first["labels"])  # sanity check
