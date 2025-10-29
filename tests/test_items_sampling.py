from __future__ import annotations

from datasets import Dataset
from open_provence.trainer import sample_items_by_label_priority


def test_items_sampling_keeps_positive_and_samples_negatives():
    dataset = Dataset.from_dict(
        {
            "labels": [[1, 0, 0, 0]],
            "texts": [["pos", "neg-a", "neg-b", "neg-c"]],
            "teacher_scores": [[0.9, 0.2, 0.1, 0.05]],
        }
    )

    filtered = sample_items_by_label_priority(dataset, 3, seed=123, num_proc=1)

    assert len(filtered) == 1
    row = filtered[0]
    assert len(row["labels"]) == 3
    assert row["labels"][0] == 1  # positive entry is retained
    assert row["texts"][0] == "pos"
    # The remaining items originate from the original negatives
    assert set(row["texts"][1:]).issubset({"neg-a", "neg-b", "neg-c"})
    assert len(row["teacher_scores"]) == 3


def test_items_sampling_drops_queries_with_too_few_items():
    dataset = Dataset.from_dict(
        {
            "id": ["short", "long"],
            "labels": [[1, 0], [1, 0, 0]],
            "texts": [["p", "n"], ["p", "n1", "n2"]],
        }
    )

    filtered = sample_items_by_label_priority(dataset, 3, seed=42, num_proc=1)

    assert len(filtered) == 1
    assert filtered[0]["id"] == "long"
    assert len(filtered[0]["labels"]) == 3


def test_items_sampling_handles_rows_without_positive_labels():
    dataset = Dataset.from_dict(
        {
            "labels": [[0, 0, 0, 0]],
            "texts": [["a", "b", "c", "d"]],
        }
    )

    filtered = sample_items_by_label_priority(dataset, 2, seed=7, num_proc=1)

    assert len(filtered) == 1
    row = filtered[0]
    assert len(row["labels"]) == 2
    assert set(row["texts"]).issubset({"a", "b", "c", "d"})


def test_items_sampling_prefers_positive_items_when_exceeding_limit():
    dataset = Dataset.from_dict(
        {
            "labels": [[1, 1, 0, 0]],
            "texts": [["p1", "p2", "n1", "n2"]],
        }
    )

    filtered = sample_items_by_label_priority(dataset, 2, seed=5, num_proc=1)

    assert len(filtered) == 1
    assert filtered[0]["labels"] == [1, 1]
    assert filtered[0]["texts"] == ["p1", "p2"]
