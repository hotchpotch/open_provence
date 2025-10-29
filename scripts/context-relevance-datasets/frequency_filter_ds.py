from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import datasets
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
    load_from_disk,
)


@dataclass
class DuplicateStats:
    rows_total: int
    rows_kept: int
    rows_removed: int
    texts_total: int
    texts_unique: int
    texts_duplicates: int
    texts_dup_ratio: float
    texts_total_filtered: int
    texts_unique_filtered: int
    texts_duplicates_filtered: int
    texts_dup_ratio_filtered: float
    duplicate_buckets_total: Counter[int]
    duplicate_buckets_kept: Counter[int]
    duplicate_buckets_removed: Counter[int]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter context-relevance datasets by limiting duplicate texts per row "
            "based on MD5 fingerprints."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name registered in HuggingFace Datasets.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Optional dataset subset / configuration name.",
    )
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        help="Splits to include (default: train). May be provided multiple times.",
    )
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        type=int,
        action="append",
        help="Duplicate allowance per row. May be provided multiple times (default: 1,2,3).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the filtered DatasetDict and summary will be stored.",
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=None,
        help="Optional cap on the number of train rows to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="Column name containing unique identifiers per row (default: id).",
    )
    parser.add_argument(
        "--texts-column",
        default="texts",
        help="Column name containing the list of texts to deduplicate (default: texts).",
    )
    return parser.parse_args(argv)


def canonical_command(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def limit_dataset(ds: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return ds
    if limit <= 0:
        raise ValueError("--debug-limit must be positive.")
    limit = min(limit, len(ds))
    return ds.select(range(limit))


def frequency_filter_train(
    ds: Dataset,
    threshold: int,
    id_column: str,
    texts_column: str,
) -> tuple[Dataset, DuplicateStats, set[str]]:
    if threshold < 0:
        raise ValueError("Threshold must be non-negative.")

    global_counter: Counter[str] = Counter()
    seen_counter: Counter[str] = Counter()
    duplicate_bucket_total: Counter[int] = Counter()
    duplicate_bucket_kept: Counter[int] = Counter()
    duplicate_bucket_removed: Counter[int] = Counter()
    drop_ids: set[str] = set()

    rows_total = len(ds)
    for idx, row in enumerate(ds, start=1):
        if not isinstance(row, Mapping):
            raise TypeError("Dataset row is not a mapping; cannot access columns by name.")

        texts_value = row.get(texts_column)
        if not isinstance(texts_value, Iterable):
            raise TypeError(f"Row {row.get(id_column, 'unknown')} has non-iterable texts column.")

        texts = list(texts_value)

        md5_list: list[str] = []
        for text in texts:
            md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
            md5_list.append(md5)
            global_counter[md5] += 1

        dup_count = sum(1 for md5 in md5_list if seen_counter[md5] > 0)
        duplicate_bucket_total[dup_count] += 1

        if dup_count > threshold:
            row_id = row.get(id_column)
            if not isinstance(row_id, str):
                raise TypeError("Row id must be a string when identifying duplicates.")
            drop_ids.add(row_id)
            duplicate_bucket_removed[dup_count] += 1
            continue

        duplicate_bucket_kept[dup_count] += 1
        for md5 in md5_list:
            seen_counter[md5] += 1

        if idx % 50000 == 0:
            print(
                f"[threshold={threshold}] processed {idx:,} rows "
                f"(kept={sum(duplicate_bucket_kept.values()):,}, "
                f"removed={sum(duplicate_bucket_removed.values()):,})"
            )

    total_texts = sum(global_counter.values())
    unique_texts = len(global_counter)
    duplicate_texts = total_texts - unique_texts
    duplicate_ratio = duplicate_texts / total_texts if total_texts else 0.0

    filtered_total_texts = sum(seen_counter.values())
    filtered_unique_texts = len(seen_counter)
    filtered_duplicate_texts = filtered_total_texts - filtered_unique_texts
    filtered_duplicate_ratio = (
        filtered_duplicate_texts / filtered_total_texts if filtered_total_texts else 0.0
    )

    stats = DuplicateStats(
        rows_total=rows_total,
        rows_kept=sum(duplicate_bucket_kept.values()),
        rows_removed=sum(duplicate_bucket_removed.values()),
        texts_total=total_texts,
        texts_unique=unique_texts,
        texts_duplicates=duplicate_texts,
        texts_dup_ratio=duplicate_ratio,
        texts_total_filtered=filtered_total_texts,
        texts_unique_filtered=filtered_unique_texts,
        texts_duplicates_filtered=filtered_duplicate_texts,
        texts_dup_ratio_filtered=filtered_duplicate_ratio,
        duplicate_buckets_total=duplicate_bucket_total,
        duplicate_buckets_kept=duplicate_bucket_kept,
        duplicate_buckets_removed=duplicate_bucket_removed,
    )

    print(
        f"[threshold={threshold}] finished scanning rows_total={rows_total:,} "
        f"rows_kept={stats.rows_kept:,} rows_removed={stats.rows_removed:,}"
    )

    filtered_dataset = ds.filter(lambda example: example[id_column] not in drop_ids)
    return filtered_dataset, stats, drop_ids


def write_summary(
    summary_path: Path,
    *,
    dataset: str,
    subset: str | None,
    splits: Sequence[str],
    threshold: int,
    debug_limit: int | None,
    stats: DuplicateStats,
    drop_ids_count: int,
    command: str,
) -> None:
    timestamp = datetime.now(UTC).isoformat()

    lines: list[str] = []
    lines.append("# Frequency Filter Summary")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Dataset: `{dataset}`")
    lines.append(f"- Subset: `{subset}`" if subset else "- Subset: (none)")
    lines.append(f"- Threshold (N): {threshold}")
    lines.append(f"- Splits requested: {', '.join(splits) if splits else 'train'}")
    lines.append("- Filtered split: train")
    lines.append(f"- Debug limit: {debug_limit if debug_limit else 'None'}")
    lines.append(f"- Command: `{command}`")
    lines.append(f"- Timestamp (UTC): {timestamp}")
    lines.append("")

    lines.append("## Train Split Statistics")
    lines.append(f"- Rows (input): {stats.rows_total:,}")
    lines.append(f"- Rows kept: {stats.rows_kept:,}")
    lines.append(f"- Rows removed: {stats.rows_removed:,}")
    lines.append(f"- Drop id count: {drop_ids_count:,}")
    lines.append("")
    lines.append(f"- Texts total (input): {stats.texts_total:,}")
    lines.append(f"- Texts unique (input): {stats.texts_unique:,}")
    lines.append(f"- Texts duplicates (input): {stats.texts_duplicates:,}")
    lines.append(f"- Text duplicate ratio (input): {stats.texts_dup_ratio:.4%}")
    lines.append("")
    lines.append(f"- Texts total (filtered): {stats.texts_total_filtered:,}")
    lines.append(f"- Texts unique (filtered): {stats.texts_unique_filtered:,}")
    lines.append(f"- Texts duplicates (filtered): {stats.texts_duplicates_filtered:,}")
    lines.append(f"- Text duplicate ratio (filtered): {stats.texts_dup_ratio_filtered:.4%}")
    lines.append("")

    lines.append("## Duplicate Distribution (by duplicate count seen before row)")
    lines.append("| Duplicate texts | Total rows | Kept rows | Removed rows |")
    lines.append("| --- | ---: | ---: | ---: |")
    max_bucket = max(stats.duplicate_buckets_total.keys(), default=0)
    for dup_count in range(0, max_bucket + 1):
        total_rows = stats.duplicate_buckets_total.get(dup_count, 0)
        kept_rows = stats.duplicate_buckets_kept.get(dup_count, 0)
        removed_rows = stats.duplicate_buckets_removed.get(dup_count, 0)
        if total_rows == 0 and kept_rows == 0 and removed_rows == 0:
            continue
        lines.append(f"| {dup_count} | {total_rows:,} | {kept_rows:,} | {removed_rows:,} |")
    lines.append("")

    lines.append("## Metadata")
    lines.append(
        f"- Environment: Python {sys.version.split()[0]} | datasets {datasets.__version__}"
    )
    lines.append(f"- Summary generated at: {timestamp}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def prepare_dataset_dict(
    original: DatasetDict,
    filtered_train: Dataset,
) -> DatasetDict:
    new_dict = DatasetDict()
    for split_name, ds in original.items():
        if split_name == "train":
            new_dict[split_name] = filtered_train
        else:
            new_dict[split_name] = ds
    return new_dict


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    thresholds = args.thresholds or [1, 2, 3]
    thresholds = sorted(set(thresholds))

    splits = args.splits or ["train"]
    if "train" not in splits:
        print("Warning: --split did not include 'train'; proceeding to filter train split anyway.")

    output_root = Path(args.output_dir)
    ensure_output_dir(output_root)

    print("Loading dataset...")
    dataset_dict: datasets.DatasetDict
    try:
        loaded_dataset = (
            load_dataset(args.dataset, name=args.subset)
            if args.subset
            else load_dataset(args.dataset)
        )
    except Exception as load_err:
        dataset_path = Path(args.dataset)
        if dataset_path.is_dir():
            print(
                "load_dataset failed; attempting load_from_disk on",
                dataset_path,
            )
            loaded_dataset = load_from_disk(str(dataset_path))
        else:
            raise load_err

    if isinstance(loaded_dataset, (IterableDatasetDict, IterableDataset)):
        raise TypeError("Streaming datasets are not supported by the frequency filter.")

    if isinstance(loaded_dataset, Dataset):
        dataset_dict = DatasetDict({"train": loaded_dataset})
    else:
        dataset_dict = loaded_dataset

    if "train" not in dataset_dict:
        raise ValueError("Dataset must contain a 'train' split.")

    base_train = dataset_dict["train"]
    train_to_process = limit_dataset(base_train, args.debug_limit)

    multi_threshold = len(thresholds) > 1

    for threshold in thresholds:
        print(f"\n=== Processing threshold {threshold} ===")
        filtered_train, stats, drop_ids = frequency_filter_train(
            train_to_process,
            threshold=threshold,
            id_column=args.id_column,
            texts_column=args.texts_column,
        )

        command = canonical_command(sys.argv)
        if multi_threshold:
            save_dir = output_root / f"freq{threshold}"
        else:
            save_dir = output_root

        print(f"Saving filtered dataset to {save_dir} ...")
        ensure_output_dir(save_dir)

        summary_path = save_dir / "frequency_filter_ds_summary.md"
        write_summary(
            summary_path,
            dataset=args.dataset,
            subset=args.subset,
            splits=splits,
            threshold=threshold,
            debug_limit=args.debug_limit,
            stats=stats,
            drop_ids_count=len(drop_ids),
            command=command,
        )

        if drop_ids:
            drop_ids_path = save_dir / "drop_ids.txt"
            drop_ids_path.write_text("\n".join(sorted(drop_ids)), encoding="utf-8")

        final_dataset_dict = prepare_dataset_dict(dataset_dict, filtered_train)
        final_dataset_dict.save_to_disk(str(save_dir))
        print(f"Saved dataset for threshold={threshold} at {save_dir}")

    metadata = {
        "dataset": args.dataset,
        "subset": args.subset,
        "thresholds": thresholds,
        "splits_request": splits,
        "debug_limit": args.debug_limit,
        "output_root": str(output_root),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    metadata_path = output_root / "frequency_filter_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
