#!/usr/bin/env python3
"""
Add cross-encoder reranker teacher scores to Provence-style datasets.

Usage example:

    python scripts/context-relevance-datasets/add_reranker_teacher_scores.py \
        --dataset-path output/context-relevance-datasets/base/tomaarsen_natural-questions-hard-negatives_triplet-5_with_relevance \
        --model hotchpotch/japanese-reranker-xsmall-v2 \
        --column-name japanese-reranker-xsmall-v2 \
        --overwrite

The script loads a DatasetDict created by the provenance converter, scores every
query/passage pair with the specified model, and writes the augmented dataset to
disk. Re-running the command with a different model name appends additional
`teacher_scores.<column_name>` columns to the same dataset directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from sentence_transformers import CrossEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Config:
    dataset_path: Path
    output_path: Path
    model_name: str
    column_name: str
    batch_size: int
    debug_limit: int | None
    validate_samples: int
    overwrite: bool
    dtype: str

    query_column: str = "query"
    texts_column: str = "texts"
    labels_column: str = "labels"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Attach reranker teacher scores to Provence datasets."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        help="Path to the input DatasetDict (output of the relevance pipeline).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Where to save the augmented dataset. Defaults to the input path plus "
            "the suffix '_with_teacher_scores' if not already present."
        ),
    )
    parser.add_argument(
        "--model",
        default="hotchpotch/japanese-reranker-xsmall-v2",
        help="Cross-encoder model identifier on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--column-name",
        default=None,
        help=(
            "Column name suffix to use for the new teacher scores. "
            "Defaults to the sanitized model name (e.g. japanese-reranker-xsmall-v2)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for CrossEncoder.predict (default: 16).",
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=None,
        help="If provided, limit each split to the first N examples for quick iteration.",
    )
    parser.add_argument(
        "--validate-samples",
        type=int,
        default=5,
        help="Number of samples to print for sanity checking.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Computation dtype for the cross-encoder model (default: bfloat16).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    dataset_path = args.dataset_path.resolve()
    if args.output_path is None:
        if str(dataset_path).endswith("_with_teacher_scores"):
            output_path = dataset_path
        else:
            output_path = dataset_path.with_name(f"{dataset_path.name}_with_teacher_scores")
    else:
        output_path = args.output_path.resolve()

    column_name = args.column_name
    if not column_name:
        column_name = args.model.split("/")[-1].replace(".", "_")

    return Config(
        dataset_path=dataset_path,
        output_path=output_path,
        model_name=args.model,
        column_name=column_name,
        batch_size=args.batch_size,
        debug_limit=args.debug_limit,
        validate_samples=args.validate_samples,
        overwrite=args.overwrite,
        dtype=args.dtype,
    )


def _load_split_from_arrow_shards(split_path: Path) -> Dataset:
    arrow_files = sorted(split_path.glob("*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No Arrow shards found under {split_path}")
    logger.info(
        "Falling back to raw Arrow shards for split at %s (%d files).",
        split_path,
        len(arrow_files),
    )
    datasets = [Dataset.from_file(str(arrow_file)) for arrow_file in arrow_files]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def _load_split_with_fallback(split_path: Path) -> Dataset:
    try:
        return Dataset.load_from_disk(str(split_path))
    except FileNotFoundError:
        return _load_split_from_arrow_shards(split_path)


def _load_dataset_dict_from_directory(path: Path) -> DatasetDict:
    dataset_dict_path = path / "dataset_dict.json"
    if dataset_dict_path.exists():
        with dataset_dict_path.open() as fp:
            payload = json.load(fp)
        split_names = payload.get("splits", [])
    else:
        split_names = sorted(entry.name for entry in path.iterdir() if entry.is_dir())
    if not split_names:
        raise FileNotFoundError(f"No dataset splits discovered under {path}")

    dataset_dict = {}
    for split_name in split_names:
        split_path = path / split_name
        if not split_path.exists():
            logger.warning("Split path %s declared but missing on disk.", split_path)
            continue
        try:
            dataset_dict[split_name] = _load_split_with_fallback(split_path)
        except FileNotFoundError as error:
            logger.warning("Skipping split %s: %s", split_name, error)
    if not dataset_dict:
        raise FileNotFoundError(f"Failed to load any dataset splits from {path}")
    return DatasetDict(dataset_dict)


def load_dataset_dict(path: Path) -> DatasetDict:
    logger.info("Loading dataset from %s", path)
    try:
        data = load_from_disk(str(path))
        if isinstance(data, Dataset):
            data = DatasetDict({"train": data})
    except FileNotFoundError as error:
        logger.warning(
            "Standard load_from_disk failed (%s). Attempting Arrow shard fallback.",
            error,
        )
        data = _load_dataset_dict_from_directory(path)
    logger.info(
        "Loaded splits: %s",
        ", ".join(f"{name} ({len(split):,})" for name, split in data.items()),
    )
    return data


def prepare_split(split: Dataset, limit: int | None) -> Dataset:
    if limit is not None and len(split) > limit:
        logger.warning("Debug mode active: trimming split from %d to %d rows", len(split), limit)
        return split.select(range(limit))
    return split


def initialise_model(model_name: str, dtype: str) -> CrossEncoder:
    logger.info("Loading reranker model %s", model_name)
    model = CrossEncoder(model_name, max_length=None)
    base_model = getattr(model, "model", None)
    device = getattr(model, "device", None)
    if isinstance(base_model, nn.Module) and getattr(device, "type", None) in {"cuda", "mps"}:
        target_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }.get(dtype, torch.float32)
        base_model.to(dtype=target_dtype)
        logger.info("Running on %s with dtype=%s.", device, dtype)
    else:
        logger.info("Running on %s.", device if device is not None else "CPU")
    return model


def score_split(
    split: Dataset,
    model: CrossEncoder,
    config: Config,
    split_name: str,
) -> list[list[float]]:
    num_rows = len(split)
    logger.info("Scoring split '%s' with %d rows.", split_name, num_rows)
    scores_per_row: list[list[float]] = []

    pair_buffer: list[tuple[str, str]] = []
    meta_buffer: list[tuple[int, int]] = []

    start_time = time.time()

    def flush_buffer() -> None:
        if not pair_buffer:
            return
        predictions = model.predict(
            pair_buffer,
            show_progress_bar=False,
            batch_size=config.batch_size,
        )
        for (row_idx, text_idx), score in zip(meta_buffer, predictions):
            scores_per_row[row_idx][text_idx] = float(score)
        pair_buffer.clear()
        meta_buffer.clear()

    for row_idx in tqdm(range(num_rows), desc=f"{split_name} rows"):
        record = split[row_idx]
        texts: Sequence[str] = record[config.texts_column]
        scores_per_row.append([0.0] * len(texts))
        query = record[config.query_column]

        for text_idx, text in enumerate(texts):
            pair_buffer.append((query, text))
            meta_buffer.append((row_idx, text_idx))
            if len(pair_buffer) >= config.batch_size:
                flush_buffer()

    flush_buffer()

    elapsed = time.time() - start_time
    total_pairs = sum(len(row) for row in scores_per_row)
    if elapsed > 0:
        logger.info(
            "Split '%s' scored %d pairs in %.1f seconds (%.1f pairs/sec).",
            split_name,
            total_pairs,
            elapsed,
            total_pairs / elapsed,
        )
    else:
        logger.info("Split '%s' scored %d pairs.", split_name, total_pairs)

    return scores_per_row


def add_column(
    split: Dataset,
    column_name: str,
    values: Iterable[list[float]],
    overwrite: bool,
) -> Dataset:
    target_column = f"teacher_scores.{column_name}"
    if target_column in split.column_names:
        if not overwrite:
            raise ValueError(
                f"Column '{target_column}' already exists. Use --overwrite to replace it."
            )
        split = split.remove_columns(target_column)
        logger.info("Overwriting existing column %s", target_column)
    split = split.add_column(
        target_column,
        list(values),
        new_fingerprint=f"{target_column}_updated",
    )
    return split


def validate_samples(split: Dataset, column_name: str, count: int) -> None:
    if count <= 0 or len(split) == 0:
        return
    target_column = f"teacher_scores.{column_name}"
    logger.info("Validating first %d samples for column %s", min(count, len(split)), target_column)
    for idx in range(min(count, len(split))):
        rec = split[idx]
        scores = rec[target_column]
        logger.info("Sample %d id=%s scores=%s", idx, rec.get("id", "N/A"), scores)

    # Basic stats
    sample_size = min(1000, len(split))
    all_scores = np.array(sum((split[i][target_column] for i in range(sample_size)), []))
    if all_scores.size:
        logger.info(
            "Score stats -> mean: %.4f, std: %.4f, min: %.4f, max: %.4f",
            float(np.mean(all_scores)),
            float(np.std(all_scores)),
            float(np.min(all_scores)),
            float(np.max(all_scores)),
        )
        if len(split) > sample_size:
            logger.info("(Stats based on first %d rows)", sample_size)


def _remove_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            _remove_directory(child)
        else:
            child.unlink()
    path.rmdir()


def save_dataset(dataset: DatasetDict, path: Path, overwrite: bool) -> None:
    tmp_path = path
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path {path} already exists. Use --overwrite to replace."
            )
        tmp_path = path.parent / f".{path.name}.tmpwrite"
    if tmp_path.exists():
        _remove_directory(tmp_path)

    logger.info("Saving dataset to %s", tmp_path)
    dataset.save_to_disk(str(tmp_path))

    if tmp_path != path:
        if path.exists():
            _remove_directory(path)
        tmp_path.replace(path)
    logger.info("Dataset saved to %s", path)


def main() -> None:
    config = parse_args()

    dataset = load_dataset_dict(config.dataset_path)
    model = initialise_model(config.model_name, config.dtype)

    augmented_splits = {}
    for split_name, split in dataset.items():
        prepared = prepare_split(split, config.debug_limit)
        scores = score_split(prepared, model, config, split_name)
        augmented = add_column(prepared, config.column_name, scores, config.overwrite)
        validate_samples(augmented, config.column_name, config.validate_samples)
        augmented_splits[split_name] = augmented

    augmented_dataset = DatasetDict(augmented_splits)
    save_dataset(
        augmented_dataset,
        config.output_path,
        config.overwrite or config.output_path == config.dataset_path,
    )
    logger.info("Teacher score column '%s' added successfully.", config.column_name)


if __name__ == "__main__":
    main()
