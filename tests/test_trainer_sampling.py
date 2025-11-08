"""Tests for dataset sampling logic in ``open_provence.trainer``."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, cast

import pytest
from datasets import Dataset, DatasetDict
from open_provence.trainer import (
    DataArguments,
    _sample_dataset_randomly,
    prepare_dataset,
    sample_items_by_label_priority,
)


def test_sample_dataset_randomly_is_deterministic() -> None:
    dataset = Dataset.from_dict({"value": list(range(10))})

    rnd_first = random.Random(42)
    rnd_second = random.Random(42)

    sampled_first = _sample_dataset_randomly(dataset, 3, rnd_first, "test")
    sampled_second = _sample_dataset_randomly(dataset, 3, rnd_second, "test")

    assert sampled_first["value"] == sampled_second["value"]
    assert len(sampled_first) == 3


def test_sample_dataset_randomly_returns_original_if_large_request() -> None:
    dataset = Dataset.from_dict({"value": [1, 2, 3]})
    rnd = random.Random(42)

    same_dataset = _sample_dataset_randomly(dataset, 5, rnd, "test")
    assert same_dataset is dataset


def test_sample_dataset_randomly_rejects_non_positive_sample_size() -> None:
    dataset = Dataset.from_dict({"value": [1, 2, 3]})
    rnd = random.Random(42)

    with pytest.raises(ValueError):
        _sample_dataset_randomly(dataset, 0, rnd, "test")


def _build_dataset(size: int = 10, validation_size: int = 6) -> DatasetDict:
    data = {
        "query": [f"q{i}" for i in range(size)],
        "positive": [f"pos{i}" for i in range(size)],
        "negative": [f"neg{i}" for i in range(size)],
        "teacher_score": [float(i) for i in range(size)],
    }
    validation = {
        "query": [f"vq{i}" for i in range(validation_size)],
        "positive": [f"vpos{i}" for i in range(validation_size)],
        "negative": [f"vneg{i}" for i in range(validation_size)],
        "teacher_score": [float(i) for i in range(validation_size)],
    }
    return DatasetDict(
        {
            "train": Dataset.from_dict(data),
            "validation": Dataset.from_dict(validation),
        }
    )


def test_prepare_dataset_supports_local_paths(tmp_path: Path) -> None:
    dataset = _build_dataset()
    dataset_path = tmp_path / "local_ds"
    dataset.save_to_disk(dataset_path)

    data_args = DataArguments(
        dataset_name="unused",
        subset="default",
        teacher_column="teacher_score",
        datasets=[
            {
                "dataset_name": str(dataset_path),
                "teacher_column": "teacher_score",
            }
        ],
    )

    train_dataset, eval_dataset = prepare_dataset(data_args, seed=13)

    assert len(train_dataset) == len(dataset["train"])
    assert len(eval_dataset) == len(dataset["validation"])


def test_prepare_dataset_applies_n_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_dataset(name: str, subset: str | None = None) -> DatasetDict:
        return _build_dataset()

    monkeypatch.setattr("open_provence.trainer.load_dataset", fake_load_dataset)

    data_args = DataArguments(
        dataset_name="dummy",
        subset="default",
        teacher_column="teacher_score",
        datasets=[
            {
                "dataset_name": "dummy",
                "subset": "default",
                "teacher_column": "teacher_score",
                "n_samples": 5,
            }
        ],
    )

    train_dataset, eval_dataset = prepare_dataset(data_args, seed=42)

    assert len(train_dataset) == 5
    assert len(eval_dataset) == 3

    # Deterministic sampling: rerunning with the same seed yields identical results
    train_dataset_again, eval_dataset_again = prepare_dataset(data_args, seed=42)
    assert train_dataset_again["query"] == train_dataset["query"]
    assert eval_dataset_again["query"] == eval_dataset["query"]


def test_prepare_dataset_accepts_fractional_n_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_dataset(name: str, subset: str | None = None) -> DatasetDict:
        return _build_dataset()

    monkeypatch.setattr("open_provence.trainer.load_dataset", fake_load_dataset)

    data_args = DataArguments(
        dataset_name="dummy",
        subset="default",
        teacher_column="teacher_score",
        datasets=[
            {
                "dataset_name": "dummy",
                "subset": "default",
                "teacher_column": "teacher_score",
                "n_samples": 0.2,
            }
        ],
    )

    train_dataset, eval_dataset = prepare_dataset(data_args, seed=42)

    assert len(train_dataset) == 2  # ceil(10 * 0.2)
    assert len(eval_dataset) == 2  # ceil(6 * 0.2)

    train_dataset_again, eval_dataset_again = prepare_dataset(data_args, seed=42)
    assert train_dataset_again["query"] == train_dataset["query"]
    assert eval_dataset_again["query"] == eval_dataset["query"]


def test_sample_items_handles_missing_labels() -> None:
    dataset = Dataset.from_dict(
        {
            "texts": [
                ["doc0", "doc1", "doc2", "doc3"],
                ["doc4", "doc5", "doc6"],
            ],
            "teacher_score": [
                [0.9, 0.1, 0.2, 0.3],
                [0.8, 0.6, 0.2],
            ],
            "extra": [
                ["meta0", "meta1", "meta2", "meta3"],
                ["meta4", "meta5", "meta6"],
            ],
        }
    )

    sampled = sample_items_by_label_priority(dataset, max_items=2, seed=7)

    sampled_rows = [cast(dict[str, Any], row) for row in sampled]

    for row in sampled_rows:
        assert len(cast(list[Any], row["texts"])) == 2
        assert len(cast(list[Any], row["teacher_score"])) == 2
        assert len(cast(list[Any], row["extra"])) == 2

    # Deterministic across runs with same seed
    sampled_again = [
        cast(dict[str, Any], row)
        for row in sample_items_by_label_priority(dataset, max_items=2, seed=7)
    ]
    assert sampled_again == sampled_rows
