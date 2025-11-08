#!/usr/bin/env python3
"""Upload Provence-style context relevance datasets to the Hugging Face Hub.

Usage example:

    python scripts/context-relevance-datasets/upload_context_relevance_to_hf.py \
        --dataset-path output/context-relevance-datasets/base/tomaarsen_natural-questions-hard-negatives_triplet-5_with_relevance_with_teacher_scores \
        --repo-id hotchpotch/natural-questions-context-relevance \
        --subset default \
        --commit-message "add nq context relevance with teacher scores"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a context relevance dataset to the Hugging Face Hub."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        help="Path to the dataset directory created by the conversion scripts.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face Hub repository in the form <namespace>/<dataset_name>.",
    )
    parser.add_argument(
        "--subset",
        default="default",
        help="Configuration name for the dataset (HF subset). Defaults to 'default'.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="If set, only push a single split (train/validation/test). Otherwise pushes all splits.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="500MB",
        help="Maximum shard size when uploading (default: 500MB).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of shards to write. Defaults to automatic selection based on max_shard_size.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes for dataset preparation before upload.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload dataset",
        help="Optional commit message for the upload.",
    )
    parser.add_argument(
        "--commit-description",
        default=None,
        help="Optional commit description accompanying the upload.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch name to push to (defaults to main).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face access token. Defaults to the logged-in user token.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the uploaded repository public. Private by default.",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Disable embedding external files in the Parquet shards (sets embed_external_files=False).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print upload parameters without performing the push.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def ensure_dataset_dict(data) -> DatasetDict:
    if isinstance(data, DatasetDict):
        return data
    if isinstance(data, Dataset):
        return DatasetDict({"train": data})
    raise TypeError(f"Unsupported dataset type: {type(data)}")


def estimate_nbytes(ds: DatasetDict, split: str | None) -> int | None:
    try:
        if split:
            return ds[split]._estimate_nbytes()  # type: ignore[attr-defined]
        total = 0
        for _, subset in ds.items():
            total += subset._estimate_nbytes()  # type: ignore[attr-defined]
        return total
    except Exception:
        return None


def format_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f}{units[unit_index]}"


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    dataset_path = args.dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    logger.info("Loading dataset from %s", dataset_path)
    data = load_from_disk(str(dataset_path))
    dataset = ensure_dataset_dict(data)

    if args.split and args.split not in dataset:
        raise ValueError(
            f"Split '{args.split}' not found in dataset. Available: {list(dataset.keys())}"
        )

    total_rows = (
        sum(len(split) for split in dataset.values())
        if not args.split
        else len(dataset[args.split])
    )
    total_size = estimate_nbytes(dataset, args.split)
    logger.info(
        "Prepared dataset with %d rows (estimated size: %s)", total_rows, format_size(total_size)
    )

    if args.split:
        logger.info("Uploading single split '%s'", args.split)
        target_dataset = dataset[args.split]
        logger.info(
            "Upload parameters: repo_id=%s config=%s split=%s max_shard_size=%s num_shards=%s",
            args.repo_id,
            args.subset,
            args.split,
            args.max_shard_size,
            args.num_shards,
        )
        if args.dry_run:
            logger.info("Dry run enabled; skipping push_to_hub call.")
            return
        commit_info = target_dataset.push_to_hub(
            repo_id=args.repo_id,
            config_name=args.subset,
            split=args.split,
            commit_message=args.commit_message,
            commit_description=args.commit_description,
            private=None if args.public else True,
            token=args.token,
            revision=args.revision,
            max_shard_size=args.max_shard_size,
            num_shards=args.num_shards,
            embed_external_files=not args.no_embed,
        )
    else:
        logger.info("Uploading all splits for DatasetDict")
        logger.info(
            "Upload parameters: repo_id=%s config=%s max_shard_size=%s",
            args.repo_id,
            args.subset,
            args.max_shard_size,
        )
        if args.dry_run:
            logger.info("Dry run enabled; skipping push_to_hub call.")
            return
        if args.num_shards is not None:
            logger.warning(
                "DatasetDict.push_to_hub expects a dict for num_shards; ignoring provided --num-shards"
            )
        commit_info = dataset.push_to_hub(
            repo_id=args.repo_id,
            config_name=args.subset,
            commit_message=args.commit_message,
            commit_description=args.commit_description,
            private=None if args.public else True,
            token=args.token,
            revision=args.revision,
            max_shard_size=args.max_shard_size,
            embed_external_files=not args.no_embed,
        )

    logger.info("Upload complete: %s", commit_info)


if __name__ == "__main__":
    main()
