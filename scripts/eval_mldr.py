"""Unified MLDR processing and LLM評価ツール."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from time import perf_counter
from typing import Any, cast

import litellm
import torch
import yaml
from datasets import Dataset, load_dataset, load_from_disk
from open_provence.modeling_open_provence_standalone import OpenProvenceModel
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoModel, AutoTokenizer

LOGGER = logging.getLogger("eval_mldr")

IGNORES_PATH_DEFAULT = Path(__file__).resolve().parent / "eval_mldr" / "ignored_questions.yaml"

NAVER_PROVENCE_PATTERN = re.compile(r"^naver/.*provence.*", re.IGNORECASE)


def _resolve_model_dir(model_dir: Path) -> Path:
    resolved = model_dir.resolve()
    if resolved.is_dir() and (resolved / "final_model").exists():
        return resolved / "final_model"
    return resolved


def _resolve_torch_dtype(name: str | None) -> torch.dtype | None:
    if not name:
        return None

    normalized = name.strip().lower()
    alias_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "single": torch.float32,
    }

    if normalized in alias_map:
        return alias_map[normalized]

    raise ValueError(
        f"Unsupported torch dtype '{name}'. Use one of: {', '.join(sorted(alias_map))}."
    )


def _should_use_naver_provence_model(model_identifier: str, *, is_local: bool) -> bool:
    if is_local:
        return False
    identifier = model_identifier.strip()
    return bool(NAVER_PROVENCE_PATTERN.search(identifier))


def _prepare_naver_provence_model(
    model: Any,
    *,
    max_length: int | None,
    disable_progress: bool,
) -> Callable[..., dict[str, Any]]:
    original_forward = type(model).forward

    def forward_with_cast(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        output = original_forward(self, *args, **kwargs)
        try:
            ranking_scores = output["ranking_scores"]
        except (KeyError, TypeError):
            ranking_scores = None
        if isinstance(ranking_scores, torch.Tensor) and ranking_scores.dtype == torch.bfloat16:
            output["ranking_scores"] = ranking_scores.to(dtype=torch.float32)

        try:
            compression_logits = output["compression_logits"]
        except (KeyError, TypeError):
            compression_logits = None
        if (
            isinstance(compression_logits, torch.Tensor)
            and compression_logits.dtype == torch.bfloat16
        ):
            output["compression_logits"] = compression_logits.to(dtype=torch.float32)

        return output

    model.forward = forward_with_cast.__get__(model, type(model))

    if max_length is not None:
        if hasattr(model, "max_len"):
            model.max_len = max_length
        if hasattr(model.config, "max_position_embeddings"):
            model.config.max_position_embeddings = max_length

    if disable_progress:
        module = sys.modules.get(model.__class__.__module__)

        def _noop_tqdm(iterable, *args, **kwargs):  # type: ignore[explicit-any]
            return iterable

        if module is not None and hasattr(module, "tqdm"):
            setattr(module, "tqdm", _noop_tqdm)

    process_fn = getattr(model, "process", None)
    if not callable(process_fn):
        raise AttributeError("Loaded Naver Provence model does not expose a callable 'process'.")

    return cast(Callable[..., dict[str, Any]], process_fn)


def _load_process_fn(
    model_source: str | Path,
    *,
    max_length: int | None,
    device: str | None,
    trust_remote_code: bool,
    torch_dtype: torch.dtype | None,
    naver_provence_model: bool,
    disable_progress: bool,
) -> tuple[Callable[..., dict[str, Any]], Any]:
    """Load the process() entrypoint from a standalone modeling script or HF model."""

    source_path = Path(model_source) if isinstance(model_source, Path) else Path(str(model_source))
    is_local = source_path.exists()
    load_identifier = str(source_path if is_local else model_source)

    if not naver_provence_model:
        open_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if torch_dtype is not None:
            open_kwargs["torch_dtype"] = torch_dtype
        try:
            model = OpenProvenceModel.from_pretrained(
                load_identifier,
                device=device,
                max_length=max_length,
                **open_kwargs,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime checkpoints
            print(
                "[eval_mldr] OpenProvenceModel load failed, falling back to AutoModel. "
                f"Reason: {exc}",
            )
            model = None
        else:
            model.eval()
            process_fn = getattr(model, "process", None)
            if callable(process_fn):
                return cast(Callable[..., dict[str, Any]], process_fn), model
            print(
                "[eval_mldr] OpenProvenceModel is missing a callable process(); "
                "falling back to AutoModel.",
            )

    auto_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        auto_kwargs["dtype"] = torch_dtype

    try:
        auto_model = AutoModel.from_pretrained(
            load_identifier,
            **auto_kwargs,
        )
    except (OSError, ValueError):  # pragma: no cover - handled by fallback loader
        auto_model = None

    if auto_model is not None:
        if device:
            auto_model.to(device)
        auto_model.eval()

        if naver_provence_model:
            process_fn = _prepare_naver_provence_model(
                auto_model,
                max_length=max_length,
                disable_progress=disable_progress,
            )
            return process_fn, auto_model

        process_attr = getattr(auto_model, "process", None)
        if callable(process_attr):
            return cast(Callable[..., dict[str, Any]], process_attr), auto_model

    if not is_local:
        raise FileNotFoundError(
            "Could not load process function from OpenProvenceModel or AutoModel, and no local checkpoint directory was found."
        )

    module_path = source_path / "modeling_open_provence_standalone.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"{module_path} not found. Ensure the checkpoint bundles modeling_open_provence_standalone.py."
        )

    spec = importlib.util.spec_from_file_location("standalone_modeling_open_provence", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[call-arg]

    if not hasattr(module, "OpenProvenceModel"):
        raise AttributeError("Standalone module does not expose OpenProvenceModel.")

    model_cls = module.OpenProvenceModel

    model = model_cls.from_pretrained(
        str(source_path),
        device=device,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )

    process_fn = getattr(model, "process")
    if not callable(process_fn):
        raise AttributeError("Loaded model does not expose a callable 'process' method.")
    return cast(Callable[..., dict[str, Any]], process_fn), model


def build_records(
    process_fn: Callable[..., dict[str, Any]],
    dataset: Dataset,
    *,
    threshold: float,
    batch_size: int,
    log_timing: bool,
    use_best_reranker_score: bool,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[float]], int]:
    stats: dict[str, list[float]] = {
        "pos_scores": [],
        "neg_scores": [],
        "pos_compression": [],
        "neg_compression": [],
    }

    def normalize_title(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        if isinstance(value, Sequence):
            parts = []
            for item in value:
                if item is None:
                    continue
                part = str(item).strip()
                if part:
                    parts.append(part)
            if not parts:
                return None
            return " ".join(parts)
        stripped = str(value).strip()
        return stripped if stripped else None

    queries: list[str] = []
    query_ids: list[str] = []
    contexts_per_query: list[list[str]] = []
    titles_per_query: list[list[str | None]] = []
    docids_per_query: list[list[str]] = []
    labels_per_query: list[list[int]] = []

    for raw_row in dataset:
        row = cast(dict[str, Any], raw_row)
        positives = cast(list[dict[str, Any]], row["positive_passages"])
        negatives = cast(list[dict[str, Any]], row["negative_passages"])

        context_texts: list[str] = []
        title_texts: list[str | None] = []
        docids: list[str] = []
        labels: list[int] = []

        for passage in positives:
            context_texts.append(passage["text"])
            title_value = passage.get("title") if isinstance(passage, dict) else None
            title_texts.append(normalize_title(title_value))
            docids.append(passage["docid"])
            labels.append(1)
        for passage in negatives:
            context_texts.append(passage["text"])
            title_value = passage.get("title") if isinstance(passage, dict) else None
            title_texts.append(normalize_title(title_value))
            docids.append(passage["docid"])
            labels.append(0)

        if not context_texts:
            continue

        query_ids.append(cast(str, row["query_id"]))
        queries.append(cast(str, row["query"]))
        contexts_per_query.append(context_texts)
        titles_per_query.append(title_texts)
        docids_per_query.append(docids)
        labels_per_query.append(labels)

    query_lengths = [len(contexts) for contexts in contexts_per_query]
    if not query_lengths:
        return [], stats, 0

    def _convert_nested(obj: Any) -> Any:
        if hasattr(obj, "tolist"):
            try:
                obj = obj.tolist()
            except Exception:  # pragma: no cover - defensive
                pass
        if isinstance(obj, tuple):
            return [_convert_nested(item) for item in obj]
        if isinstance(obj, list):
            return [_convert_nested(item) for item in obj]
        return obj

    def _normalize_nested(
        value: Any,
        *,
        fill_factory: Callable[[], Any],
        name: str,
    ) -> list[list[Any]]:
        if value is None:
            return [
                [fill_factory() for _ in range(expected_docs)] for expected_docs in query_lengths
            ]

        converted = _convert_nested(value)

        if len(query_lengths) == 1 and not isinstance(converted, list):
            expected_docs = query_lengths[0]
            if expected_docs != 1:
                raise ValueError(
                    f"process() returned a scalar for '{name}' but expected {expected_docs} docs."
                )
            return [[converted]]

        if isinstance(converted, list):
            if len(query_lengths) == 1 and (not converted or not isinstance(converted[0], list)):
                expected_docs = query_lengths[0]
                if len(converted) != expected_docs:
                    raise ValueError(
                        f"process() returned {len(converted)} items for '{name}' but expected {expected_docs}."
                    )
                return [converted]

            if len(converted) != len(query_lengths):
                raise ValueError(
                    f"process() returned {len(converted)} query batches for '{name}' but expected {len(query_lengths)}."
                )

            normalized: list[list[Any]] = []
            for idx, expected_docs in enumerate(query_lengths):
                item = converted[idx]
                if isinstance(item, list):
                    if len(item) != expected_docs:
                        raise ValueError(
                            f"process() returned {len(item)} docs for query #{idx} in '{name}' but expected {expected_docs}."
                        )
                    normalized.append(item)
                elif expected_docs == 1:
                    normalized.append([item])
                else:
                    raise ValueError(
                        f"process() returned a scalar for query #{idx} in '{name}' but expected {expected_docs} docs."
                    )
            return normalized

        return [[fill_factory() for _ in range(expected_docs)] for expected_docs in query_lengths]

    kwargs = {
        "question": queries,
        "context": contexts_per_query,
        "title": titles_per_query,
        "threshold": threshold,
        "batch_size": batch_size,
        "log_timing": log_timing,
        "use_best_reranker_score": use_best_reranker_score,
        "show_progress": show_progress,
        "return_sentence_texts": True,
    }

    try:
        signature = inspect.signature(process_fn)
        supported = set(signature.parameters)
    except (ValueError, TypeError):  # pragma: no cover - fallback for C funcs
        supported = set(kwargs)

    if "question" not in supported:
        parent = getattr(process_fn, "__self__", None)
        require_model = getattr(parent, "_require_model", None)
        if callable(require_model):
            try:
                model_obj = cast(Any, require_model())
                base_signature = inspect.signature(model_obj.process)  # type: ignore[attr-defined]
                supported = set(base_signature.parameters)
            except (ValueError, TypeError):  # pragma: no cover - fallback
                supported = set(kwargs)

    for key in list(kwargs):
        if key not in supported:
            kwargs.pop(key)

    process_result = process_fn(**kwargs)

    if "pruned_context" not in process_result:
        raise KeyError("process() result must include 'pruned_context'.")

    pruned_contexts = _normalize_nested(
        process_result.get("pruned_context"),
        fill_factory=lambda: "",
        name="pruned_context",
    )
    reranking_scores = _normalize_nested(
        process_result.get("reranking_score"),
        fill_factory=lambda: None,
        name="reranking_score",
    )
    compression_rates = _normalize_nested(
        process_result.get("compression_rate"),
        fill_factory=lambda: 0.0,
        name="compression_rate",
    )
    kept_sentence_lists = _normalize_nested(
        process_result.get("kept_sentences"),
        fill_factory=lambda: [],
        name="kept_sentences",
    )
    removed_sentence_lists = _normalize_nested(
        process_result.get("removed_sentences"),
        fill_factory=lambda: [],
        name="removed_sentences",
    )
    resolved_titles = process_result.get("title")

    records: list[dict[str, Any]] = []

    for idx, query_text in enumerate(queries):
        pruned_list = pruned_contexts[idx]
        scores_list = reranking_scores[idx]
        compression_list = compression_rates[idx]
        kept_list = kept_sentence_lists[idx]
        removed_list = removed_sentence_lists[idx]
        docids = docids_per_query[idx]
        labels = labels_per_query[idx]
        originals = contexts_per_query[idx]
        qid = query_ids[idx]

        titles_list = titles_per_query[idx]
        model_titles_for_query = (
            resolved_titles[idx]
            if isinstance(resolved_titles, list) and idx < len(resolved_titles)
            else None
        )

        for doc_position, (
            docid,
            label,
            original,
            pruned,
            score,
            compression,
            kept,
            removed,
            title_candidate,
        ) in enumerate(
            zip(
                docids,
                labels,
                originals,
                pruned_list,
                scores_list,
                compression_list,
                kept_list,
                removed_list,
                titles_list,
            )
        ):
            effective_title = normalize_title(title_candidate)
            if (
                effective_title is None
                and isinstance(model_titles_for_query, list)
                and doc_position < len(model_titles_for_query)
            ):
                model_title_value = model_titles_for_query[doc_position]
                effective_title = normalize_title(model_title_value)

            record = {
                "query_id": qid,
                "query": query_text,
                "docid": docid,
                "label": label,
                "title": effective_title,
                "original_text": original,
                "pruned_text": pruned,
                "reranking_score": score,
                "compression_rate": compression,
                "kept_sentences": kept,
                "removed_sentences": removed,
            }
            records.append(record)

            if label == 1:
                stats["pos_scores"].append(score if score is not None else float("nan"))
                stats["pos_compression"].append(compression)
            else:
                stats["neg_scores"].append(score if score is not None else float("nan"))
                stats["neg_compression"].append(compression)

    return records, stats, len(query_ids)


def write_markdown_report(
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    threshold: float,
    limit: int | None,
    language: str | None,
    max_length: int | None,
    max_queries: int = 5,
) -> None:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_query[record["query_id"]].append(record)

    lines: list[str] = []
    limit_label = limit if limit is not None else "all"
    lang_label = language or "unknown"
    lines.append(
        f"# MLDR サマリ (lang={lang_label}, threshold={threshold}, limit={limit_label}, max_len={max_length})"
    )
    lines.append("")
    lines.append(f"- 合計クエリ数: {len(by_query)}")
    lines.append(f"- 合計文書数: {len(records)}")
    lines.append("")

    sorted_queries = sorted(by_query.items())

    for idx, (query_id, rows) in enumerate(sorted_queries[:max_queries], start=1):
        question = rows[0]["query"].replace("\n", " ").strip()
        lines.append(f"## {idx}. {question}")
        lines.append(f"- Query ID: `{query_id}`")
        lines.append("")

        ordered_rows = sorted(
            rows,
            key=lambda r: (1 - r["label"], -(r["reranking_score"] or 0.0)),
        )

        for doc_rank, row in enumerate(ordered_rows, start=1):
            docid = row["docid"]
            label = "POS" if row["label"] == 1 else "NEG"
            score = row["reranking_score"]
            compression = row["compression_rate"]
            kept = row.get("kept_sentences") or []
            removed = row.get("removed_sentences") or []
            title_value = row.get("title")
            pruned = (row.get("pruned_text") or "").replace("\n", " ").strip()

            lines.append(f"### {doc_rank}. `{docid}` ({label})")
            lines.append(f"- スコア: {score:.4f}" if score is not None else "- スコア: N/A")
            lines.append(f"- 圧縮率: {compression:.2f}%")
            if title_value:
                lines.append(f"- タイトル: {title_value}")
            lines.append("")

            lines.append("**保持された文:**")
            if kept:
                for sentence in kept:
                    cleaned_sentence = sentence.replace("\n", " ").strip()
                    lines.append(f"- {cleaned_sentence}")
            else:
                lines.append("- (保持なし)")
            lines.append("")

            lines.append("**削除された文:**")
            if removed:
                for sentence in removed:
                    cleaned_sentence = sentence.replace("\n", " ").strip()
                    lines.append(f"- {cleaned_sentence}")
            else:
                lines.append("- (削除なし)")
            lines.append("")

            lines.append("**削除後テキスト:**")
            lines.append(f"> {pruned if pruned else '(空)'}")
            lines.append("")

        lines.append("---")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_process(args: argparse.Namespace, *, output_dir: Path) -> Path:
    process_dir = output_dir / "process"
    dataset_path = process_dir / "dataset"

    if dataset_path.exists() and not args.force_process:
        LOGGER.info(
            "Process dataset already exists at %s – skipping (use --force-process to rerun).",
            dataset_path,
        )
        return dataset_path

    stage_start = perf_counter()
    process_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = _resolve_torch_dtype(args.torch_dtype)

    lang_override = args.lang_override
    mldr_lang = args.mldr_lang if args.mldr_lang is not None else lang_override
    if mldr_lang is None:
        mldr_lang = "ja"

    LOGGER.info(
        "[process] Loading checkpoint from %s (lang=%s, splitter=auto, limit=%s, threshold=%.3f)",
        args.model,
        mldr_lang,
        args.limit,
        args.threshold,
    )

    AutoTokenizer.from_pretrained(str(args.model_load_source), trust_remote_code=True)

    process_fn, _ = _load_process_fn(
        args.model_load_source,
        max_length=args.max_length,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        naver_provence_model=args.naver_provence_model,
        disable_progress=args.no_progress,
    )

    split_expr = args.split
    if args.limit and "[" not in split_expr:
        split_expr = f"{split_expr}[:{args.limit}]"

    dataset = cast(
        Dataset,
        load_dataset(
            "Shitao/MLDR",
            mldr_lang,
            split=split_expr,
            trust_remote_code=True,
        ),
    )

    LOGGER.info(
        "[process] Loaded MLDR split '%s' with %d examples.",
        split_expr,
        len(dataset),
    )

    records, stats, num_queries = build_records(
        process_fn,
        dataset,
        threshold=args.threshold,
        batch_size=args.batch_size,
        log_timing=args.log_timing,
        use_best_reranker_score=not args.reranker_first_score,
        show_progress=not args.no_progress,
    )

    Dataset.from_list(records).save_to_disk(str(dataset_path))

    elapsed = perf_counter() - stage_start

    summary = {
        "limit": args.limit,
        "threshold": args.threshold,
        "num_records": len(records),
        "num_queries": num_queries,
        "dataset_language": mldr_lang,
        "splitter_language": "auto",
        "max_length": args.max_length,
        "model": args.model,
        "naver_provence_model": args.naver_provence_model,
        "avg_pos_score": fmean([s for s in stats["pos_scores"] if s == s])
        if stats["pos_scores"]
        else None,
        "avg_neg_score": fmean([s for s in stats["neg_scores"] if s == s])
        if stats["neg_scores"]
        else None,
        "avg_pos_compression": fmean(stats["pos_compression"])
        if stats["pos_compression"]
        else None,
        "avg_neg_compression": fmean(stats["neg_compression"])
        if stats["neg_compression"]
        else None,
        "source_text": "pruned",
        "process_time_seconds": elapsed,
    }

    with open(process_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    write_markdown_report(
        records,
        process_dir / "result.md",
        threshold=args.threshold,
        limit=args.limit,
        language="auto",
        max_length=args.max_length,
    )

    LOGGER.info(
        "[process] Saved dataset (%d records, %d queries) to %s in %.1fs",
        len(records),
        num_queries,
        dataset_path,
        elapsed,
    )
    return dataset_path


def run_original_dataset(args: argparse.Namespace, *, output_dir: Path) -> Path:
    process_dir = output_dir / "process_original"
    dataset_path = process_dir / "dataset"

    if dataset_path.exists() and not args.force_process:
        LOGGER.info(
            "[original] Dataset already exists at %s – skipping (use --force-process to rebuild).",
            dataset_path,
        )
        return dataset_path

    stage_start = perf_counter()
    process_dir.mkdir(parents=True, exist_ok=True)

    lang_override = args.lang_override
    mldr_lang = args.mldr_lang if args.mldr_lang is not None else lang_override

    if mldr_lang is None:
        mldr_lang = "ja"

    split_expr = args.split
    if args.limit and "[" not in split_expr:
        split_expr = f"{split_expr}[:{args.limit}]"

    LOGGER.info(
        "[original] Loading MLDR split '%s' (lang=%s, limit=%s) for original-text evaluation.",
        split_expr,
        mldr_lang,
        args.limit,
    )

    dataset = cast(
        Dataset,
        load_dataset(
            "Shitao/MLDR",
            mldr_lang,
            split=split_expr,
            trust_remote_code=True,
        ),
    )

    records, stats, num_queries = build_original_records(dataset)

    if not records:
        raise RuntimeError("No passages were collected for original-text evaluation.")

    Dataset.from_list(records).save_to_disk(str(dataset_path))

    elapsed = perf_counter() - stage_start

    summary = {
        "limit": args.limit,
        "threshold": args.threshold,
        "num_records": len(records),
        "num_queries": num_queries,
        "dataset_language": mldr_lang,
        "splitter_language": "auto",
        "max_length": args.max_length,
        "model": args.model,
        "naver_provence_model": False,
        "avg_pos_score": None,
        "avg_neg_score": None,
        "avg_pos_compression": fmean(stats["pos_compression"])
        if stats["pos_compression"]
        else None,
        "avg_neg_compression": fmean(stats["neg_compression"])
        if stats["neg_compression"]
        else None,
        "source_text": "original",
        "process_time_seconds": elapsed,
    }

    with open(process_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    write_markdown_report(
        records,
        process_dir / "result.md",
        threshold=args.threshold,
        limit=args.limit,
        language=mldr_lang,
        max_length=args.max_length,
    )

    LOGGER.info(
        "[original] Saved dataset (%d records, %d queries) to %s in %.1fs",
        len(records),
        num_queries,
        dataset_path,
        elapsed,
    )
    return dataset_path


def ensure_openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise OSError("OPENAI_API_KEY is not set. Please export it before running evaluation.")
    return api_key


def load_ignore_list(path: Path, lang: str) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get(lang, []) or []
    mapping: dict[str, str] = {}
    for entry in entries:
        qid = entry.get("qid")
        if qid is None:
            continue
        key = str(qid)
        reason = str(entry.get("reason", "")).strip()
        mapping[key] = reason
    return mapping


def build_original_records(
    dataset: Dataset,
) -> tuple[list[dict[str, Any]], dict[str, list[float]], int]:
    stats: dict[str, list[float]] = {
        "pos_scores": [],
        "neg_scores": [],
        "pos_compression": [],
        "neg_compression": [],
    }

    records: list[dict[str, Any]] = []
    query_count = 0

    for raw_row in dataset:
        row = cast(dict[str, Any], raw_row)
        positives = cast(list[dict[str, Any]], row.get("positive_passages") or [])
        negatives = cast(list[dict[str, Any]], row.get("negative_passages") or [])

        if not positives and not negatives:
            continue

        query_id = str(row.get("query_id"))
        query_text = str(row.get("query") or "")

        def _normalize_title(value: Any) -> str | None:
            if value is None:
                return None
            if isinstance(value, str):
                stripped = value.strip()
                return stripped if stripped else None
            if isinstance(value, Sequence):
                parts = []
                for item in value:
                    if item is None:
                        continue
                    part = str(item).strip()
                    if part:
                        parts.append(part)
                if not parts:
                    return None
                return " ".join(parts)
            stripped = str(value).strip()
            return stripped if stripped else None

        def _append_record(passage: dict[str, Any], label: int) -> None:
            text = str(passage.get("text") or "")
            title_value = passage.get("title") if isinstance(passage, dict) else None
            docid = str(passage.get("docid") or "")

            record = {
                "query_id": query_id,
                "query": query_text,
                "docid": docid,
                "label": label,
                "title": _normalize_title(title_value),
                "original_text": text,
                "pruned_text": text,
                "reranking_score": None,
                "compression_rate": 100.0,
                "kept_sentences": [],
                "removed_sentences": [],
            }
            records.append(record)

            if label == 1:
                stats["pos_compression"].append(100.0)
            else:
                stats["neg_compression"].append(100.0)

        for passage in positives:
            _append_record(passage, 1)
        for passage in negatives:
            _append_record(passage, 0)

        query_count += 1

    return records, stats, query_count


class EvalPayload(BaseModel):
    has_answer: int = Field(..., ge=0, le=1)
    answer_score: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str = Field(..., max_length=16000)


@dataclass(slots=True)
class EvaluationConfig:
    model: str
    provider: str | None
    reasoning_effort: str | None
    temperature: float
    max_completion_tokens: int
    retries: int
    retry_delay: float
    concurrency: int
    system_prompt: str
    log_details: bool
    request_timeout: float | None


@dataclass(slots=True)
class ExampleRecord:
    idx: int
    query_id: str
    question: str
    docid: str
    title: str | None
    pruned_text: str
    reranking_score: float | None


@dataclass(slots=True)
class EvaluationResult:
    has_answer: int | None
    answer_score: float | None
    reasoning: str | None
    attempts: int
    error: str | None
    latency: float | None


SYSTEM_PROMPT = (
    "You are an impartial verifier who checks whether a passage answers a question. "
    "Evaluate based on the actual content in <text>, considering <title> as supplementary context.\n"
    'Return strict JSON: {"has_answer": 0 or 1, "answer_score": float between 0 and 1, "reasoning": "brief English explanation (≤3 sentences)"}'
)


def build_user_prompt(question: str, pruned_text: str, title: str | None) -> str:
    question_block = question.strip() or "(empty question)"
    text_block = pruned_text.strip() or "(empty text)"
    title_block = (title or "").strip() or "(none)"

    instruction = """
Determine whether the passage provides sufficient information to answer the question.

Evaluation criteria:
- has_answer=1: The passage explicitly contains facts/statements that directly answer the question OR strongly support a correct answer through clear logical inference.
- has_answer=0: The answer is missing, contradicted, requires external knowledge, or the question is malformed/empty.
- answer_score: A float between 0 and 1 indicating the likelihood/strength that the passage contains the answer (1.0 = definitely contains answer, 0.8 = probably contains answer, 0.5 = ambiguous, 0.2 = probably lacks answer, 0.0 = definitely lacks answer).
- For "why/how" questions: accept answers that explain mechanisms or reasons, even if not exhaustive.
- For factual questions: require explicit mention of the key entity/fact.
- For lengthy passages: scan systematically through the entire text for relevant information before concluding.
- For short passages: be precise about what is explicitly stated.

Examples:
1. Q: "Where were the 2020 Olympics held?" | Text: "The 2020 Summer Olympics took place in Tokyo, Japan."
   ⇒ {"has_answer": 1, "answer_score": 1.0, "reasoning": "Tokyo, Japan is explicitly stated as the location."}

2. Q: "Which temple is the most famous in Kyoto?" | Text: "Kyoto has a humid climate with four distinct seasons."
   ⇒ {"has_answer": 0, "answer_score": 0.0, "reasoning": "The passage discusses climate only; no temple is mentioned."}

3. Q: "asdfkj lkjwer?" | Text: "Paris is the capital of France."
   ⇒ {"has_answer": 0, "answer_score": 0.0, "reasoning": "The question is incoherent gibberish."}

4. Q: "Why did the company's revenue decline?" | Text: "The company faced supply chain disruptions and decreased consumer demand in Q3."
   ⇒ {"has_answer": 1, "answer_score": 0.95, "reasoning": "The passage identifies two clear causes for the decline."}

5. Q: "What is the population of Tokyo?" | Text: "Tokyo is a major metropolitan area. It has significant economic importance."
   ⇒ {"has_answer": 0, "answer_score": 0.05, "reasoning": "Population figure is not provided, only general characteristics."}

6. Q: "Does the report mention climate change?" | Text: "The environmental section discusses rising temperatures and changing weather patterns over the past decade."
   ⇒ {"has_answer": 1, "answer_score": 0.8, "reasoning": "While 'climate change' is not explicitly stated, rising temperatures and changing weather patterns strongly imply it."}
""".strip()

    prompt_parts = [
        f"<instruction>{instruction}</instruction>",
        f"<question>{question_block}</question>",
        f"<title>{title_block}</title>",
        f"<text>{text_block}</text>",
    ]
    return "\n".join(prompt_parts)


async def call_litellm(prompt: str, config: EvaluationConfig) -> EvalPayload:
    completion_kwarg: dict[str, Any] = (
        {"max_completion_tokens": config.max_completion_tokens}
        if "gpt-5" in config.model.lower()
        else {"max_tokens": config.max_completion_tokens}
    )

    if config.reasoning_effort:
        target_tokens = 20000
        if "max_completion_tokens" in completion_kwarg:
            requested = completion_kwarg["max_completion_tokens"]
            if requested is None or requested < target_tokens:
                completion_kwarg["max_completion_tokens"] = target_tokens
        else:
            requested = completion_kwarg["max_tokens"]
            if requested is None or requested < target_tokens:
                completion_kwarg["max_tokens"] = target_tokens

    model_name = (
        config.model
        if "/" in config.model or not config.provider
        else f"{config.provider}/{config.model}"
    )

    temperature = config.temperature
    if "gpt-5" in model_name.lower():
        temperature = 1.0

    response = await litellm.acompletion(
        model=model_name,
        messages=[
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        request_timeout=config.request_timeout,
        response_format={"type": "json_object"},
        **completion_kwarg,
    )

    content = response.choices[0].message.content  # type: ignore[index]
    if content is None:
        raise ValueError("LLM returned empty content.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM response as JSON: {content}") from exc

    if isinstance(parsed, dict) and "reason" in parsed and "reasoning" not in parsed:
        parsed["reasoning"] = parsed.pop("reason")
    if isinstance(parsed, dict) and "has_answer" not in parsed and "contains_answer" in parsed:
        parsed["has_answer"] = parsed.pop("contains_answer")

    try:
        return EvalPayload.model_validate(parsed)
    except ValidationError as exc:  # pragma: no cover
        raise ValueError(f"Invalid payload: {parsed}") from exc


async def evaluate_example(record: ExampleRecord, config: EvaluationConfig) -> EvaluationResult:
    attempts = 0
    start = perf_counter()
    hard_timeout = max(1.0, (config.request_timeout or 0.0)) + 5.0

    while attempts <= config.retries:
        try:
            payload = await asyncio.wait_for(
                call_litellm(
                    prompt=build_user_prompt(record.question, record.pruned_text, record.title),
                    config=config,
                ),
                timeout=hard_timeout,
            )
            latency = perf_counter() - start
            return EvaluationResult(
                has_answer=payload.has_answer,
                answer_score=payload.answer_score,
                reasoning=payload.reasoning,
                attempts=attempts + 1,
                error=None,
                latency=latency,
            )
        except TimeoutError as exc:
            attempts += 1
            if attempts > config.retries:
                latency = perf_counter() - start
                return EvaluationResult(
                    has_answer=None,
                    answer_score=None,
                    reasoning=None,
                    attempts=attempts,
                    error=f"Timeout after {hard_timeout:.1f}s: {exc}",
                    latency=latency,
                )
            await asyncio.sleep(config.retry_delay)
        except Exception as exc:  # pragma: no cover
            attempts += 1
            if attempts > config.retries:
                latency = perf_counter() - start
                return EvaluationResult(
                    has_answer=None,
                    answer_score=None,
                    reasoning=None,
                    attempts=attempts,
                    error=str(exc),
                    latency=latency,
                )
            await asyncio.sleep(config.retry_delay)
    raise RuntimeError("Exceeded maximum retries without producing a result.")


async def evaluate_batch(
    records: list[ExampleRecord], config: EvaluationConfig
) -> list[EvaluationResult]:
    semaphore = asyncio.Semaphore(config.concurrency)

    async def _run(record: ExampleRecord) -> EvaluationResult:
        async with semaphore:
            return await evaluate_example(record, config)

    return await asyncio.gather(*(_run(record) for record in records))


def build_default_config(
    model: str,
    reasoning_effort: str,
    *,
    concurrency: int,
    retries: int,
    retry_delay: float,
    request_timeout: float,
) -> EvaluationConfig:
    return EvaluationConfig(
        model=model,
        provider=None,
        reasoning_effort=reasoning_effort,
        temperature=0.0,
        max_completion_tokens=20000,
        retries=retries,
        retry_delay=retry_delay,
        concurrency=concurrency,
        system_prompt=SYSTEM_PROMPT,
        log_details=False,
        request_timeout=request_timeout,
    )


def run_evaluation(args: argparse.Namespace, *, dataset_path: Path, output_dir: Path) -> None:
    eval_dir = output_dir / "eval_llm"
    dataset_out = eval_dir / "dataset"
    if dataset_out.exists() and not args.force_eval:
        LOGGER.info(
            "LLM evaluation artifacts already exist at %s – skipping (use --force-eval to rerun).",
            eval_dir,
        )
        return

    ensure_openai_api_key()

    dataset = load_from_disk(str(dataset_path))
    records: list[dict[str, Any]] = [cast(dict[str, Any], row) for row in dataset]

    ignore_file = args.ignore_file
    if not ignore_file.exists():
        if args.force_no_ignore:
            LOGGER.warning(
                "Ignore list %s not found. Proceeding without it because --force-no-ignore is set.",
                ignore_file,
            )
            ignore_map: dict[str, str] = {}
        else:
            raise FileNotFoundError(
                f"Ignore list file '{ignore_file}' not found. Create the file or rerun with --force-no-ignore."
            )
    else:
        ignore_map = load_ignore_list(ignore_file, args.lang)

    filtered = []
    skipped_records: list[dict[str, str]] = []
    max_chars = max(0, args.max_text_chars)

    for record in records:
        qid = str(record["query_id"])
        if qid in ignore_map:
            skipped_records.append({"query_id": qid, "reason": ignore_map[qid]})
            continue
        if not args.include_negatives and record.get("label") != 1:
            continue
        truncated_record = dict(record)
        text_value = str(truncated_record.get("pruned_text") or "")
        if max_chars > 0 and len(text_value) > max_chars:
            LOGGER.debug(
                "Truncating qid=%s docid=%s text length from %d to %d characters.",
                qid,
                truncated_record.get("docid"),
                len(text_value),
                max_chars,
            )
            text_value = text_value[:max_chars]
        truncated_record["pruned_text"] = text_value
        filtered.append(truncated_record)

    if not filtered:
        LOGGER.warning("No records eligible for evaluation after filtering. Exiting.")
        return

    examples = [
        ExampleRecord(
            idx=idx,
            query_id=str(record["query_id"]),
            question=str(record["query"]),
            docid=str(record["docid"]),
            title=record.get("title"),
            pruned_text=str(record.get("pruned_text") or ""),
            reranking_score=record.get("reranking_score"),
        )
        for idx, record in enumerate(filtered)
    ]

    LOGGER.info(
        "[eval] Preparing evaluation over %d passages (ignored %d queries, include_negatives=%s)",
        len(examples),
        len(skipped_records),
        args.include_negatives,
    )

    config = build_default_config(
        args.llm_model,
        args.reasoning_effort,
        concurrency=max(1, args.concurrency),
        retries=max(0, args.retries),
        retry_delay=max(0.0, args.retry_delay),
        request_timeout=max(1.0, args.request_timeout),
    )

    eval_start = perf_counter()
    results = asyncio.run(evaluate_batch(examples, config))

    enriched_records = []
    counters = Counter()
    failures = 0
    answer_scores: list[float] = []

    for record, result in zip(filtered, results):
        enriched = dict(record)
        enriched["llm_attempts"] = result.attempts
        enriched["llm_latency"] = result.latency
        enriched["llm_error"] = result.error

        enriched["llm_answer_score"] = result.answer_score

        if result.answer_score is not None:
            answer_scores.append(result.answer_score)

        if result.has_answer is None:
            failures += 1
            enriched["llm_has_answer"] = None
            enriched["llm_reasoning"] = result.error or ""
        else:
            enriched["llm_has_answer"] = result.has_answer
            enriched["llm_reasoning"] = result.reasoning
            counters[result.has_answer] += 1

        enriched_records.append(enriched)

    eval_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(enriched_records).save_to_disk(str(dataset_out))

    answer_score_summary: dict[str, float] | None
    if answer_scores:
        answer_score_summary = {
            "mean": fmean(answer_scores),
            "median": median(answer_scores),
            "min": min(answer_scores),
            "max": max(answer_scores),
        }
    else:
        answer_score_summary = None

    process_time: float | None = None
    process_summary_path = dataset_path.parent / "summary.json"
    if process_summary_path.exists():
        try:
            with open(process_summary_path, encoding="utf-8") as fh:
                process_summary_raw = json.load(fh)
            process_time_value = process_summary_raw.get("process_time_seconds")
            if process_time_value is not None:
                process_time = float(process_time_value)
        except (OSError, json.JSONDecodeError, ValueError):  # pragma: no cover
            process_time = None

    elapsed = perf_counter() - eval_start

    summary = {
        "input_dataset": str(dataset_path),
        "language": args.lang,
        "llm_model": args.llm_model,
        "reasoning_effort": args.reasoning_effort,
        "records_evaluated": len(enriched_records),
        "ignored_count": len(skipped_records),
        "counts": {
            "has_answer_1": counters.get(1, 0),
            "has_answer_0": counters.get(0, 0),
            "failed": failures,
        },
        "answer_score_stats": answer_score_summary,
        "process_time_seconds": process_time,
        "evaluation_time_seconds": elapsed,
    }

    with open(eval_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    summary_lines = [
        "# LLM Evaluation Summary",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Language: {args.lang}",
        f"- LLM model: {args.llm_model}",
        f"- Reasoning effort: {args.reasoning_effort}",
        f"- Records evaluated: {len(enriched_records)}",
        f"- Ignored queries: {len(skipped_records)}",
        f"- Has answer (1): {counters.get(1, 0)}",
        f"- Has answer (0): {counters.get(0, 0)}",
        f"- Failed: {failures}",
    ]

    if process_time is not None:
        summary_lines.append(f"- Process time (s): {process_time:.2f}")
    summary_lines.append(f"- LLM eval time (s): {elapsed:.2f}")

    if answer_score_summary is not None:
        summary_lines.append("")
        summary_lines.append("## Answer Score Statistics")
        summary_lines.append(f"- mean: {answer_score_summary['mean']:.4f}")
        summary_lines.append(f"- median: {answer_score_summary['median']:.4f}")
        summary_lines.append(f"- min: {answer_score_summary['min']:.4f}")
        summary_lines.append(f"- max: {answer_score_summary['max']:.4f}")

    (eval_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    elapsed = perf_counter() - eval_start
    LOGGER.info(
        "[eval] Saved LLM judgments (passages=%d, failed=%d) to %s in %.1fs",
        len(enriched_records),
        failures,
        eval_dir,
        elapsed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process Shitao/MLDR samples and run LLM evaluation with a single entry point."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Hugging Face model ID or local checkpoint path (preferred).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="[deprecated] Legacy model directory argument. Prefer using --model instead.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where evaluation artifacts will be written (e.g., ./output/eval_mldr_output/my_run).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of evaluation samples (default: 200). Use a large value to cover the full split.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Pruning threshold (default: 0.1).",
    )
    parser.add_argument(
        "--text-source",
        choices=["pruned", "original"],
        default="pruned",
        help="Evaluate pruned passages (default) or original MLDR passages without pruning.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for model.process (default: 16)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split expression (default: test)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override maximum token length when loading OpenProvenceModel.",
    )
    parser.add_argument(
        "--log-timing",
        action="store_true",
        help="Log per-stage timing information.",
    )
    parser.add_argument(
        "--reranker-first-score",
        action="store_true",
        help="Use the first block's reranker score instead of the best score across blocks.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars during preprocessing and inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device hint for OpenProvenceModel (default: auto-detect).",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default=None,
        help="Torch dtype override when loading the checkpoint (e.g. float16, bfloat16).",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "jp"],
        required=True,
        help="Language key for ignore list and evaluation.",
    )
    parser.add_argument(
        "--mldr-lang",
        type=str,
        default=None,
        help="Language key for loading the MLDR dataset (default: --lang).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip the LLM evaluation stage (process only).",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Reuse existing process dataset if available (fails if not present).",
    )
    parser.add_argument(
        "--force-process",
        action="store_true",
        help="Re-run processing even when outputs already exist.",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-run LLM evaluation even when outputs already exist.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5-nano",
        help="Model identifier supported by LiteLLM (default: gpt-5-nano).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low",
        help="Reasoning effort parameter for GPT-5 family (default: low).",
    )
    parser.add_argument(
        "--ignore-file",
        type=Path,
        default=IGNORES_PATH_DEFAULT,
        help="YAML file listing query IDs to skip (default: scripts/eval_mldr/ignored_questions.yaml).",
    )
    parser.add_argument(
        "--force-no-ignore",
        action="store_true",
        help="Proceed without an ignore list even if the file is missing.",
    )
    parser.add_argument(
        "--include-negatives",
        action="store_true",
        help="Evaluate negative passages as well (default: positives only).",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=60000,
        help="Clamp each evaluated text to this many characters before invoking the LLM (default: 60000).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent LLM requests (default: 10).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retry attempts per example on failure (default: 2).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.5,
        help="Delay in seconds between retry attempts (default: 1.5).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds for LiteLLM calls (default: 120).",
    )

    args = parser.parse_args()

    if args.model is not None and args.model_dir is not None:
        parser.error("--model and --model-dir cannot be provided together.")
    if args.model is None and args.model_dir is None:
        parser.error("--model is required (legacy --model-dir is temporarily supported).")

    deprecated_model_dir_used = False
    if args.model is None and args.model_dir is not None:
        args.model_dir = args.model_dir.resolve()
        args.model = str(args.model_dir)
        deprecated_model_dir_used = True
    elif args.model_dir is not None:
        args.model_dir = args.model_dir.resolve()
        deprecated_model_dir_used = True

    args.model = args.model.strip()
    if not args.model:
        parser.error("--model must not be empty.")

    model_candidate = Path(args.model).expanduser()
    if model_candidate.exists():
        resolved_dir = _resolve_model_dir(model_candidate)
        args.model_dir = resolved_dir
        args.model_is_local = True
        model_load_source: str | Path = resolved_dir
    else:
        args.model_is_local = False
        args.model_dir = None
        model_load_source = args.model

    args.model_load_source = model_load_source

    args.naver_provence_model = _should_use_naver_provence_model(
        args.model,
        is_local=args.model_is_local,
    )
    args.auto_device_cuda = False
    args.auto_torch_dtype = False

    if args.naver_provence_model:
        if args.device is None and torch.cuda.is_available():
            args.device = "cuda"
            args.auto_device_cuda = True
        if args.torch_dtype is None:
            args.torch_dtype = "bfloat16"
            args.auto_torch_dtype = True

    args.deprecated_model_dir_used = deprecated_model_dir_used

    args.output_dir = args.output_dir.resolve()
    args.ignore_file = args.ignore_file.resolve()
    args.lang_override = args.lang
    args.trust_remote_code = True
    args.use_original_text = args.text_source == "original"
    return args


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.deprecated_model_dir_used:
        LOGGER.warning("--model-dir is deprecated. Please migrate to --model.")
    if args.naver_provence_model:
        LOGGER.info("Detected Naver Provence model: %s", args.model)
        if args.auto_device_cuda:
            LOGGER.info("Auto-selected CUDA device for Naver Provence model.")
        if args.auto_torch_dtype:
            LOGGER.info("Auto-selected bfloat16 dtype for Naver Provence model.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_original_text:
        if args.skip_process:
            dataset_path = output_dir / "process_original" / "dataset"
            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Original-text dataset not found at {dataset_path}. Remove --skip-process or rerun processing."
                )
        else:
            dataset_path = run_original_dataset(args, output_dir=output_dir)
    else:
        if args.skip_process:
            dataset_path = output_dir / "process" / "dataset"
            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Process dataset not found at {dataset_path}. Remove --skip-process or rerun processing."
                )
        else:
            dataset_path = run_process(args, output_dir=output_dir)

    if args.no_eval:
        LOGGER.info("Skipping LLM evaluation because --no-eval was specified.")
        return

    run_evaluation(args, dataset_path=dataset_path, output_dir=output_dir)


if __name__ == "__main__":
    main()
