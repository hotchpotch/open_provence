#!/usr/bin/env python3
"""Evaluate Provence reranker checkpoints on context relevance datasets."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from types import MethodType
from typing import Any, cast

import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from open_provence.modeling_open_provence_standalone import (
    OpenProvenceModel,
    ProcessPerformanceTrace,
)
from transformers import AutoModel


@dataclass
class DatasetSpec:
    dataset_name: str
    subset: str | None = None
    split: str | None = None
    n_samples: int | None = None


@dataclass
class EvalConfig:
    datasets: list[DatasetSpec]
    split: str = "test"


def _parse_dataset_spec(raw: Any) -> DatasetSpec:
    if isinstance(raw, str):
        return DatasetSpec(dataset_name=raw)
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported dataset spec: {raw!r}")
    if "dataset_name" not in raw:
        raise KeyError(f"dataset_name missing in dataset spec: {raw}")
    return DatasetSpec(
        dataset_name=str(raw["dataset_name"]),
        subset=str(raw["subset"]) if "subset" in raw and raw["subset"] is not None else None,
        split=str(raw["split"]) if "split" in raw and raw["split"] is not None else None,
        n_samples=int(raw["n_samples"])
        if "n_samples" in raw and raw["n_samples"] is not None
        else None,
    )


def load_eval_config(path: Path) -> EvalConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Evaluation config must be a mapping (got {type(payload).__name__})")
    datasets_raw = payload.get("datasets")
    if not datasets_raw:
        raise ValueError("Evaluation config has no datasets.")
    datasets = [_parse_dataset_spec(entry) for entry in datasets_raw]
    split = str(payload.get("split", "test"))
    return EvalConfig(datasets=datasets, split=split)


def _patch_manual_span_support(model: Any) -> None:
    """Ensure AutoModel instances preserve manual sentence inputs."""

    require_model = getattr(model, "_require_model", None)
    if require_model is None:
        return

    try:
        standalone = model._require_model()
    except Exception:  # pragma: no cover - defensive safeguard
        return

    if not getattr(standalone, "_manual_span_support_patched", False):
        try:
            patched = MethodType(OpenProvenceModel._normalize_inputs, standalone)
        except AttributeError:  # pragma: no cover - unexpected mismatched implementation
            patched = None
        if patched is not None:
            setattr(standalone, "_normalize_inputs", patched)
            setattr(standalone, "_manual_span_support_patched", True)

    try:
        process_signature = inspect.signature(standalone.process)
    except (AttributeError, ValueError):  # pragma: no cover - defensive safeguard
        process_signature = None
    needs_process_patch = (
        process_signature is None or "return_sentence_metrics" not in process_signature.parameters
    )
    if needs_process_patch or not getattr(standalone, "_sentence_metrics_patched", False):
        try:
            patched_process = MethodType(OpenProvenceModel.process, standalone)
        except AttributeError:  # pragma: no cover - unexpected mismatched implementation
            patched_process = None
        if patched_process is not None:
            setattr(standalone, "process", patched_process)

    helper_names = (
        "_precompute_sentences_and_tokens",
        "_prepare_block_inputs",
        "_run_sequential_fragmentize",
    )
    for helper_name in helper_names:
        helper = getattr(OpenProvenceModel, helper_name, None)
        if helper is not None:
            setattr(standalone, helper_name, MethodType(helper, standalone))
    setattr(standalone, "_sentence_metrics_patched", True)

    if not getattr(model, "_sentence_metrics_patched", False):
        try:
            forwarded = MethodType(
                lambda self, *args, **kwargs: self._require_model().process(*args, **kwargs),
                model,
            )
        except AttributeError:
            forwarded = None
        if forwarded is not None:
            setattr(model, "process", forwarded)
        setattr(model, "_sentence_metrics_patched", True)


def _normalize_relevance(values: Any, span_count: int) -> list[int]:
    if span_count <= 0:
        return []
    if values is None:
        return [0] * span_count
    if not isinstance(values, Sequence):
        raise TypeError(f"context_spans_relevance must be a sequence, got {type(values)}")
    if len(values) == span_count:
        return [1 if int(v) != 0 else 0 for v in values]
    mask = [0] * span_count
    for value in values:
        index = int(value)
        if 0 <= index < span_count:
            mask[index] = 1
    return mask


def _extract_sentences(text: str, spans: Sequence[Sequence[int]]) -> list[str]:
    if not spans:
        return [text] if text else []
    sentences: list[str] = []
    length = len(text)
    for start_raw, end_raw in spans:
        start = max(0, int(start_raw))
        end = min(length, int(end_raw))
        if end <= start:
            sentences.append("")
        else:
            sentences.append(text[start:end])
    return sentences


def _format_threshold_label(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return f"{int(numeric)}"
    return f"{numeric:.6g}"


def _infer_predictions(sentences: Sequence[str], pruned_text: str, span_count: int) -> list[int]:
    if span_count <= 0:
        return []
    predictions: list[int] = []
    cursor = 0
    for sentence in sentences[:span_count]:
        candidate = sentence or ""
        length = len(candidate)
        if length and pruned_text[cursor : cursor + length] == candidate:
            predictions.append(1)
            cursor += length
        else:
            predictions.append(0)
    return predictions


def _resolve_split(
    spec: DatasetSpec,
    cfg: EvalConfig,
    override_split: str | None,
) -> str:
    if spec.split:
        return spec.split
    if override_split:
        return override_split
    return cfg.split


def _load_dataset_split(spec: DatasetSpec, split: str) -> Dataset:
    dataset_dict = load_dataset(spec.dataset_name, spec.subset)
    if not isinstance(dataset_dict, DatasetDict):
        raise TypeError(f"Expected DatasetDict from load_dataset, got {type(dataset_dict)}")
    if split not in dataset_dict:
        available = ", ".join(dataset_dict.keys())
        raise KeyError(f"Split '{split}' not found in dataset ({available})")
    dataset = dataset_dict[split]
    if spec.n_samples is not None:
        dataset = dataset.select(range(min(len(dataset), spec.n_samples)))
    return dataset


def _format_timing_summary(label: str, timing: Mapping[str, float]) -> str:
    preprocess = float(timing.get("preprocess_seconds", 0.0))
    assembly = float(timing.get("assembly_seconds", 0.0))
    inference = float(timing.get("inference_seconds", 0.0))
    postprocess = float(timing.get("postprocess_seconds", 0.0))
    total = float(timing.get("total_seconds", 0.0))
    collect = float(timing.get("sentence_collect_seconds", 0.0))
    normalize = float(timing.get("sentence_normalize_seconds", 0.0))
    tokenize = float(timing.get("tokenize_seconds", 0.0))
    fragment_split = float(timing.get("fragment_split_seconds", 0.0))
    fragment_decode = float(timing.get("fragment_decode_seconds", 0.0))

    return (
        f"[timing] dataset={label} "
        f"preprocess={preprocess:.2f}s "
        f"(collect={collect:.2f}s normalize={normalize:.2f}s "
        f"tokenize={tokenize:.2f}s fragment_split={fragment_split:.2f}s "
        f"fragment_decode={fragment_decode:.2f}s) "
        f"assembly={assembly:.2f}s "
        f"inference={inference:.2f}s "
        f"postprocess={postprocess:.2f}s "
        f"total={total:.2f}s"
    )


def evaluate_dataset(
    model: Any,
    dataset: Dataset,
    *,
    threshold: float,
    batch_size: int,
    dataset_label: str,
    show_progress: bool,
    debug_messages: bool,
    print_timing_summary: bool,
    silent: bool,
) -> dict[str, Any]:
    span_total = 0
    span_correct = 0
    span_skipped = 0
    compression_sum = 0.0
    context_count = 0
    process_time = 0.0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    roc_scores: list[float] = []
    roc_labels: list[int] = []
    roc_predictions: list[int] = []

    questions: list[str] = []
    contexts_nested: list[list[list[str]]] = []
    span_metadata_nested: list[list[int]] = []
    relevance_nested: list[list[Any]] = []

    iterable: Iterable[dict[str, Any]]
    if show_progress and not silent:
        from tqdm import tqdm

        iterable = cast(
            Iterable[dict[str, Any]],
            tqdm(
                dataset,
                desc=f"Collect spans ({dataset_label})",
                unit="ex",
            ),
        )
    else:
        iterable = cast(Iterable[dict[str, Any]], dataset)

    for example in iterable:
        question = example.get("query")
        if question is None:
            continue

        texts = example.get("texts") or []
        spans_list = example.get("context_spans") or []
        relevance_list = example.get("context_spans_relevance") or []

        contexts: list[list[str]] = []
        span_metadata: list[int] = []
        relevance_entries: list[Any] = []

        for idx, text in enumerate(texts):
            spans = spans_list[idx] if idx < len(spans_list) else []
            sentences = _extract_sentences(text, spans)
            contexts.append(sentences)
            span_metadata.append(len(spans))
            relevance_entries.append(relevance_list[idx] if idx < len(relevance_list) else [])

        questions.append(str(question))
        contexts_nested.append(contexts)
        span_metadata_nested.append(span_metadata)
        relevance_nested.append(relevance_entries)

    debug_hook: bool | Callable[[str], None]
    if debug_messages and not silent:

        def _debug_printer(message: str, *, label: str = dataset_label) -> None:
            print(f"[process:{label}] {message}")

        debug_hook = _debug_printer
    else:
        debug_hook = False

    timing_summary: dict[str, float] = {}

    if questions:
        with torch.inference_mode():
            start = perf_counter()
            process_kwargs = {
                "question": questions,
                "context": contexts_nested,
                "title": None,
                "batch_size": batch_size,
                "threshold": threshold,
                "sentence_splitter": None,
                "show_progress": show_progress,
                "debug_messages": debug_hook,
                "return_sentence_metrics": True,
            }
            process_kwargs["show_inference_progress"] = show_progress and not silent
            outputs = model.process(**process_kwargs)
            process_time += perf_counter() - start

        pruned_contexts_all = outputs["pruned_context"]
        compression_rates_all = outputs["compression_rate"]
        sentence_probs_all = outputs.get("sentence_probabilities") or []
        performance_trace = outputs.get("performance_trace")
        timing_payload = outputs.get("timing") or {}
        if isinstance(performance_trace, ProcessPerformanceTrace):
            timing_summary = performance_trace.as_dict()
            process_time = performance_trace.total_seconds
            if not silent:
                print(
                    f"[process:{dataset_label}] total={performance_trace.total_seconds:.2f}s "
                    f"(pre={performance_trace.preprocess_seconds:.2f}s asm={performance_trace.assembly_seconds:.2f}s inf={performance_trace.inference_seconds:.2f}s post={performance_trace.postprocess_seconds:.2f}s)"
                )
        elif isinstance(timing_payload, dict):
            timing_summary = {
                "preprocess_seconds": float(timing_payload.get("preprocess_seconds", 0.0)),
                "assembly_seconds": float(timing_payload.get("assembly_seconds", 0.0)),
                "inference_seconds": float(timing_payload.get("inference_seconds", 0.0)),
                "postprocess_seconds": float(timing_payload.get("postprocess_seconds", 0.0)),
                "total_seconds": float(timing_payload.get("total_seconds", process_time)),
            }
            for extra_key in (
                "sentence_collect_seconds",
                "sentence_normalize_seconds",
                "tokenize_seconds",
                "fragment_split_seconds",
                "fragment_decode_seconds",
            ):
                if extra_key in timing_payload:
                    timing_summary[extra_key] = float(timing_payload.get(extra_key, 0.0))
            process_time = timing_summary["total_seconds"]

        if print_timing_summary and timing_summary:
            print(
                _format_timing_summary(
                    dataset_label,
                    timing_summary,
                )
            )

        for query_idx, sentences_per_query in enumerate(contexts_nested):
            pruned_contexts = (
                pruned_contexts_all[query_idx] if query_idx < len(pruned_contexts_all) else []
            )
            compression_rates = (
                compression_rates_all[query_idx] if query_idx < len(compression_rates_all) else []
            )
            sentence_probs_contexts = (
                sentence_probs_all[query_idx]
                if isinstance(sentence_probs_all, Sequence) and query_idx < len(sentence_probs_all)
                else []
            )
            span_metadata = span_metadata_nested[query_idx]
            relevance_list = relevance_nested[query_idx]

            for ctx_idx, sentences in enumerate(sentences_per_query):
                span_count = span_metadata[ctx_idx] if ctx_idx < len(span_metadata) else 0
                gold = _normalize_relevance(
                    relevance_list[ctx_idx] if ctx_idx < len(relevance_list) else [],
                    span_count,
                )
                pruned_text = pruned_contexts[ctx_idx] if ctx_idx < len(pruned_contexts) else ""
                predicted = _infer_predictions(sentences, pruned_text, span_count)
                sentence_probabilities = (
                    sentence_probs_contexts[ctx_idx]
                    if isinstance(sentence_probs_contexts, Sequence)
                    and ctx_idx < len(sentence_probs_contexts)
                    else []
                )
                probabilities_available = len(sentence_probabilities) >= span_count > 0

                if span_count > 0:
                    if len(gold) != span_count or len(predicted) != span_count:
                        span_skipped += span_count
                    else:
                        span_total += span_count
                        span_correct += sum(1 for a, b in zip(gold, predicted) if a == b)
                        for idx, (gold_label, pred_label) in enumerate(zip(gold, predicted)):
                            if gold_label == 1 and pred_label == 1:
                                true_positive += 1
                            elif gold_label == 1 and pred_label == 0:
                                false_negative += 1
                            elif gold_label == 0 and pred_label == 1:
                                false_positive += 1
                            else:
                                true_negative += 1
                            if probabilities_available:
                                score_value = float(sentence_probabilities[idx])
                                roc_scores.append(score_value)
                                roc_labels.append(int(gold_label))
                                roc_predictions.append(int(pred_label))

                if ctx_idx < len(compression_rates):
                    compression_sum += float(compression_rates[ctx_idx])
                context_count += 1

    accuracy = span_correct / span_total if span_total else None
    compression_mean = compression_sum / context_count if context_count else None
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else None
    )
    if precision is not None and recall is not None and (4 * precision + recall) > 0:
        f2 = (5 * precision * recall) / (4 * precision + recall)
    else:
        f2 = None

    roc_payload = {
        "scores": roc_scores,
        "labels": roc_labels,
        "predictions": roc_predictions,
    }

    return {
        "span_total": span_total,
        "span_correct": span_correct,
        "span_accuracy": accuracy,
        "span_skipped": span_skipped,
        "contexts": context_count,
        "mean_compression": compression_mean,
        "process_time_seconds": process_time,
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "confusion_matrix": {
            "tp": true_positive,
            "fp": false_positive,
            "tn": true_negative,
            "fn": false_negative,
        },
        "roc_data": roc_payload,
        "timing": timing_summary,
    }


def build_markdown(
    metadata: dict[str, Any],
    results_by_threshold: dict[float, dict[str, dict[str, Any]]],
) -> str:
    thresholds: list[float] = []
    if metadata.get("thresholds"):
        thresholds = [float(value) for value in metadata["thresholds"]]
    elif metadata.get("threshold") is not None:
        thresholds = [float(metadata["threshold"])]
    threshold_labels = [_format_threshold_label(value) for value in thresholds]

    lines = [
        f"* Timestamp (UTC): {metadata['timestamp_utc']}",
        f"* Model: `{metadata['model']}`",
        f"* Config: `{metadata['config']}`",
        f"* Batch size: {metadata['batch_size']}",
        f"* Total process time (s): {metadata['total_process_time_seconds']:.2f}",
    ]
    lines.append("* Primary metric: F2 score (β=2).")
    if threshold_labels:
        lines.append(f"* Thresholds: {', '.join(threshold_labels)}")
    if metadata.get("split_override"):
        lines.append(f"* Split override: {metadata['split_override']}")
    if metadata.get("limit_override") is not None:
        lines.append(f"* CLI limit: {metadata['limit_override']}")

    dataset_info = metadata.get("datasets", [])
    if dataset_info:
        lines.append("* Evaluated datasets:")
        for entry in dataset_info:
            lines.append(
                f"  - {entry['key']} (split={entry['split']}, n_samples={entry['n_samples']})"
            )

    per_threshold_times = metadata.get("per_threshold_process_time_seconds") or {}
    if per_threshold_times:
        timing_parts: list[str] = []
        for label in threshold_labels:
            runtime_value = per_threshold_times.get(label)
            if runtime_value is not None:
                timing_parts.append(f"{label}: {runtime_value:.2f}")
        if timing_parts:
            lines.append("* Threshold runtimes (s): " + ", ".join(timing_parts))

    dataset_keys = [entry["key"] for entry in dataset_info]
    if not dataset_keys:
        dataset_keys = sorted(
            {key for metrics in results_by_threshold.values() for key in metrics.keys()}
        )

    for idx, threshold in enumerate(thresholds):
        label = threshold_labels[idx]
        metrics_map = results_by_threshold.get(threshold, {})
        lines.extend(
            [
                "",
                f"### Threshold {label}",
                "",
                "| Dataset | F2 Score | Recall | Precision | FN | TP | FP | TN | Mean Compression (%) | Span Accuracy | Total Spans | Contexts |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|",
            ]
        )
        ordered_keys = [key for key in dataset_keys if key in metrics_map]
        if not ordered_keys:
            ordered_keys = sorted(metrics_map.keys())
        if not ordered_keys:
            lines.append("| (no datasets) | N/A | N/A | N/A | N/A | N/A | 0 | 0 |")
            continue
        for key in ordered_keys:
            metrics = metrics_map[key]
            f2_value = metrics.get("f2")
            recall_value = metrics.get("recall")
            precision_value = metrics.get("precision")
            accuracy = metrics.get("span_accuracy")
            compression = metrics.get("mean_compression")
            f2_str = f"{f2_value:.4f}" if f2_value is not None else "N/A"
            recall_str = f"{recall_value:.4f}" if recall_value is not None else "N/A"
            precision_str = f"{precision_value:.4f}" if precision_value is not None else "N/A"
            accuracy_str = f"{accuracy:.4f}" if accuracy is not None else "N/A"
            compression_str = f"{compression:.2f}" if compression is not None else "N/A"
            confusion = metrics.get("confusion_matrix", {})
            tp_count = confusion.get("tp", 0)
            fp_count = confusion.get("fp", 0)
            tn_count = confusion.get("tn", 0)
            fn_count = confusion.get("fn", 0)
            spans = metrics.get("span_total", 0)
            contexts = metrics.get("contexts", 0)
            lines.append(
                f"| {key} | {f2_str} | {recall_str} | {precision_str} | {fn_count} | {tp_count} | {fp_count} | {tn_count} | "
                f"{compression_str} | {accuracy_str} | {spans} | {contexts} |"
            )
    return "\n".join(lines)


def write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def _load_model(args: argparse.Namespace) -> Any:
    device = getattr(args, "device", None)
    if not getattr(args, "use_automodel", False):
        model = OpenProvenceModel.from_pretrained(args.model)
        if device:
            model = cast(torch.nn.Module, model).to(device)
        model.eval()
        return model

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    _patch_manual_span_support(model)
    if device:
        model = model.to(device)
    model.eval()
    return model


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    eval_config = load_eval_config(config_path)

    silent = getattr(args, "silent", False)
    effective_show_progress = args.show_progress and not silent
    effective_debug = args.timing_details and not silent

    timestamp = datetime.now(UTC).isoformat()
    model = _load_model(args)

    process_attr = getattr(model, "process", None)
    if not callable(process_attr):
        raise TypeError(
            "Loaded model does not expose a callable 'process'. "
            "Ensure the checkpoint was exported with Open Provence tooling."
        )

    threshold_values: list[float] = []
    if args.threshold_list:
        for entry in args.threshold_list:
            for chunk in str(entry).split(","):
                raw_value = chunk.strip()
                if not raw_value:
                    continue
                try:
                    threshold_values.append(float(raw_value))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid threshold value '{raw_value}' supplied via --th/--thresholds."
                    ) from exc
    if not threshold_values:
        threshold_values.append(args.threshold)

    ordered_thresholds: list[float] = []
    seen_threshold_keys: set[str] = set()
    for value in threshold_values:
        key = repr(value)
        if key in seen_threshold_keys:
            continue
        seen_threshold_keys.add(key)
        ordered_thresholds.append(value)
    threshold_values = ordered_thresholds

    metadata: dict[str, Any] = {
        "timestamp_utc": timestamp,
        "model": args.model,
        "config": str(config_path),
        "threshold": threshold_values[0] if len(threshold_values) == 1 else None,
        "thresholds": threshold_values,
        "batch_size": args.batch_size,
        "split_override": args.split,
        "limit_override": args.limit,
        "datasets": [],
        "total_process_time_seconds": 0.0,
        "per_threshold_process_time_seconds": {},
    }

    target_keys = {target.strip() for target in (args.target or []) if target}
    available_keys: list[str] = []
    split_cache: dict[int, str] = {}
    for idx, spec in enumerate(eval_config.datasets):
        spec_split = _resolve_split(spec, eval_config, args.split)
        split_cache[idx] = spec_split
        dataset_key = f"{spec.dataset_name}:{spec.subset or spec_split}"
        available_keys.append(dataset_key)

    if target_keys:
        missing = sorted(target_keys - set(available_keys))
        if missing:
            available_joined = ", ".join(sorted(available_keys))
            raise ValueError(
                "Unknown --target entries: "
                + ", ".join(missing)
                + f". Available keys: {available_joined}"
            )

    dataset_records: list[dict[str, Any]] = []
    for idx, spec in enumerate(eval_config.datasets):
        split = split_cache[idx]
        dataset_key = f"{spec.dataset_name}:{spec.subset or split}"

        if target_keys and dataset_key not in target_keys:
            continue

        dataset = _load_dataset_split(spec, split)

        effective_limit = None
        if args.limit is not None:
            effective_limit = args.limit
        elif spec.n_samples is not None:
            effective_limit = spec.n_samples

        if effective_limit is not None:
            dataset = dataset.select(range(min(len(dataset), effective_limit)))
        dataset_size = len(dataset)
        if not silent:
            print(
                f"[eval_dataset] Prepared {dataset_key} (split={split}, samples={dataset_size:,})"
            )
        dataset_records.append(
            {
                "key": dataset_key,
                "split": split,
                "dataset": dataset,
                "size": dataset_size,
            }
        )
        metadata["datasets"].append(
            {
                "key": dataset_key,
                "split": split,
                "n_samples": dataset_size,
            }
        )

    if target_keys and not dataset_records:
        available_joined = ", ".join(sorted(available_keys))
        raise ValueError(
            "No datasets were evaluated. --target did not match any entries. Available keys: "
            + available_joined
        )

    results_by_threshold: dict[float, dict[str, dict[str, Any]]] = {}
    threshold_runtime_map: dict[str, float] = {}
    total_process_time = 0.0

    for threshold in threshold_values:
        threshold_results: dict[str, dict[str, Any]] = {}
        threshold_runtime = 0.0
        for record in dataset_records:
            metrics = evaluate_dataset(
                model,
                record["dataset"],
                threshold=threshold,
                batch_size=args.batch_size,
                dataset_label=record["key"],
                show_progress=effective_show_progress,
                debug_messages=effective_debug,
                print_timing_summary=(effective_show_progress or effective_debug) and not silent,
                silent=silent,
            )
            threshold_results[record["key"]] = metrics
            threshold_runtime += metrics.get("process_time_seconds", 0.0)
        results_by_threshold[threshold] = threshold_results
        threshold_label = _format_threshold_label(threshold)
        threshold_runtime_map[threshold_label] = threshold_runtime
        total_process_time += threshold_runtime

    metadata["total_process_time_seconds"] = total_process_time
    metadata["per_threshold_process_time_seconds"] = threshold_runtime_map

    markdown_report = build_markdown(metadata, results_by_threshold)
    if args.output_file:
        write_output(Path(args.output_file), markdown_report + "\n")
    else:
        print(markdown_report)

    if args.output_json:
        json_results = {
            _format_threshold_label(threshold): metrics
            for threshold, metrics in results_by_threshold.items()
        }
        json_payload = {"args": metadata, "results": json_results}
        write_output(
            Path(args.output_json), json.dumps(json_payload, indent=2, ensure_ascii=False)
        )

    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Provence reranker checkpoints on context relevance datasets.",
    )
    parser.add_argument("--config", required=True, help="YAML file describing datasets to load.")
    parser.add_argument("--model", required=True, help="Local path or Hugging Face model id.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Pruning probability threshold (default: 0.1).",
    )
    parser.add_argument(
        "--thresholds",
        "--th",
        action="append",
        dest="threshold_list",
        help=(
            "Comma separated list of pruning thresholds to evaluate. "
            "Repeat flag to provide multiple entries (e.g. --th 0.05,0.1 --th 0.2)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for model.process (default: 512).",
    )
    parser.add_argument("--split", help="Override split for every dataset in the config.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N examples of each dataset (useful for smoke tests).",
    )
    parser.add_argument(
        "--target",
        action="append",
        help=(
            "Limit evaluation to specific datasets given as 'dataset_name:subset'. "
            "May be specified multiple times."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="If set, write a Markdown report to this path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="If set, write structured metrics as JSON to this path.",
    )
    parser.add_argument("--device", help="Device to place the model on (e.g. cuda, cpu).")
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="show_progress",
        help="Disable dataset and process() progress bars.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress progress bars and intermediate logs; only emit final results.",
    )
    parser.add_argument(
        "--timing-details",
        action="store_true",
        help=(
            "Print detailed timing diagnostics from model.process and include them in the outputs."
        ),
    )
    parser.add_argument(
        "--use-automodel",
        action="store_true",
        help="Load the checkpoint via transformers.AutoModel instead of OpenProvenceModel.",
    )
    parser.set_defaults(show_progress=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:  # pragma: no cover - surfacing CLI exceptions
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
