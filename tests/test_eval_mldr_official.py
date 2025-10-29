from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.eval_mldr import (  # noqa: E402
    _should_use_naver_provence_model,
    build_records,
    parse_args,
)


def _build_dummy_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "query_id": "q1",
                "query": "dummy question",
                "positive_passages": [
                    {"text": "positive text", "docid": "doc1", "title": "Title 1"},
                    {"text": "another positive", "docid": "doc2", "title": None},
                ],
                "negative_passages": [],
            }
        ]
    )


def test_official_detector_handles_remote_ids() -> None:
    assert _should_use_naver_provence_model(
        "naver/provence-reranker-debertav3-v1",
        is_local=False,
    )
    assert _should_use_naver_provence_model(
        "NAVER/Provence-multilingual",
        is_local=False,
    )
    assert _should_use_naver_provence_model(
        "naver/xprovence-reranker-bgem3-v1",
        is_local=False,
    )
    assert not _should_use_naver_provence_model(
        "naver/other-model",
        is_local=False,
    )
    assert not _should_use_naver_provence_model(
        "./local/provence",
        is_local=True,
    )


def test_parse_args_auto_adjusts_for_official(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("scripts.eval_mldr.torch.cuda.is_available", lambda: True)
    argv = [
        "prog",
        "--model",
        "naver/xprovence-reranker-bgem3-v1",
        "--lang",
        "en",
        "--output-dir",
        str(tmp_path / "out"),
        "--no-eval",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    args = parse_args()

    assert args.device == "cuda"
    assert args.torch_dtype == "bfloat16"
    assert args.auto_device_cuda
    assert args.auto_torch_dtype


def test_build_records_fills_missing_fields_for_official_results() -> None:
    dataset = _build_dummy_dataset()

    def process_fn(**_: Any) -> dict[str, Any]:
        return {
            "pruned_context": [["positive text", "another positive"]],
            "reranking_score": [[0.8, 0.6]],
            "compression_rate": [[20.0, 30.0]],
        }

    records, stats, num_queries = build_records(
        process_fn,
        dataset,
        threshold=0.1,
        batch_size=2,
        log_timing=False,
        use_best_reranker_score=True,
        show_progress=False,
    )

    assert num_queries == 1
    assert len(records) == 2
    for record in records:
        assert record["kept_sentences"] == []
        assert record["removed_sentences"] == []
    assert stats["pos_scores"] == [0.8, 0.6]


def test_build_records_accepts_scalar_outputs() -> None:
    dataset = Dataset.from_list(
        [
            {
                "query_id": "q1",
                "query": "dummy question",
                "positive_passages": [
                    {"text": "positive text", "docid": "doc1", "title": None},
                ],
                "negative_passages": [],
            }
        ]
    )

    def process_fn(**_: Any) -> dict[str, Any]:
        return {
            "pruned_context": "positive text",
            "reranking_score": 0.9,
            "compression_rate": 25.0,
        }

    records, stats, num_queries = build_records(
        process_fn,
        dataset,
        threshold=0.1,
        batch_size=1,
        log_timing=False,
        use_best_reranker_score=True,
        show_progress=False,
    )

    assert num_queries == 1
    assert len(records) == 1
    assert records[0]["pruned_text"] == "positive text"
    assert records[0]["kept_sentences"] == []
    assert records[0]["removed_sentences"] == []
    assert stats["pos_scores"] == [0.9]
