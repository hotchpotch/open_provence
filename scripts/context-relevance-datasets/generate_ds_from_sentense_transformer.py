"""Example:
    $ python scripts/context-relevance-datasets/generate_ds_from_sentense_transformer.py \
        --dataset tomaarsen/natural-questions-hard-negatives \
        --subset triplet-5 \
        --lang en

The command converts the source dataset into a Provence-style DatasetDict with `train`,
`validation`, and `test` splits. If the upstream dataset lacks validation/test splits, the script
samples 1% (or 5k examples, whichever is smaller) from `train` to create new splits, mirroring the
schema used by `hotchpotch/wip-msmarco-context-relevance`.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import cast

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

try:
    from fast_bunkai import FastBunkai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    FastBunkai = None

try:
    from nltk import sent_tokenize  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sent_tokenize = None


SentenceSplitter = Callable[[str], list[str]]

_NLTK_PUNKT_LANG_CODES = {
    "da": "danish",
    "de": "german",
    "en": "english",
    "es": "spanish",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "el": "greek",
    "it": "italian",
    "nb": "norwegian",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "sl": "slovene",
    "sv": "swedish",
    "tr": "turkish",
}

_NLTK_LANGUAGE_ALIAS_MAP: dict[str, str] = {}
for _code, _name in _NLTK_PUNKT_LANG_CODES.items():
    _NLTK_LANGUAGE_ALIAS_MAP[_code] = _name
    _NLTK_LANGUAGE_ALIAS_MAP[_name] = _name
    _NLTK_LANGUAGE_ALIAS_MAP[_name.replace("_", "-")] = _name

_NLTK_LANGUAGE_ALIAS_MAP.update({"no": "norwegian", "nn": "norwegian"})

_FAST_BUNKAI = None
if FastBunkai is not None:  # pragma: no branch
    try:
        _FAST_BUNKAI = FastBunkai()
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError("Failed to initialize FastBunkai sentence splitter") from exc


def _build_nltk_sentence_splitter(language_name: str) -> SentenceSplitter:
    if sent_tokenize is None:
        raise RuntimeError(
            "nltk is not installed. Install `nltk` and download punkt or choose another splitter."
        )

    def _splitter(text: str) -> list[str]:
        if not text:
            return []
        sentences = [
            sentence.strip()
            for sentence in sent_tokenize(text, language=language_name)
            if sentence.strip()
        ]
        if sentences:
            return sentences
        stripped = text.strip()
        return [stripped] if stripped else []

    return _splitter


def _get_nltk_sentence_splitter(language_key: str) -> SentenceSplitter:
    canonical = _NLTK_LANGUAGE_ALIAS_MAP.get(language_key.lower())
    if canonical is None:
        raise ValueError(f"No punkt sentence splitter available for language '{language_key}'.")
    return _build_nltk_sentence_splitter(canonical)


def fast_bunkai_sentence_splitter(text: str) -> list[str]:
    if _FAST_BUNKAI is None:
        raise RuntimeError(
            "fast-bunkai is not installed. Install `fast-bunkai` or use the regex splitter."
        )
    sentences = [sentence.strip() for sentence in _FAST_BUNKAI(text) if sentence.strip()]
    if sentences:
        return sentences
    stripped = text.strip()
    return [stripped] if stripped else []


_REGEX_SPLIT_PATTERN = re.compile(r".+?(?:。|！|？|!|\?|\n|$)", re.S)
_GENERIC_REGEX_SPLIT_PATTERN = re.compile(r".+?(?:[.!?]|\n|$)", re.S)


def regex_sentence_splitter(text: str) -> list[str]:
    if not text:
        return []
    sentences = [match.strip() for match in _REGEX_SPLIT_PATTERN.findall(text) if match.strip()]
    if sentences:
        return sentences
    stripped = text.strip()
    return [stripped] if stripped else []


def generic_sentence_splitter(text: str) -> list[str]:
    if not text:
        return []
    sentences = [
        match.strip() for match in _GENERIC_REGEX_SPLIT_PATTERN.findall(text) if match.strip()
    ]
    if sentences:
        return sentences
    stripped = text.strip()
    return [stripped] if stripped else []


def resolve_sentence_splitter(language: str) -> SentenceSplitter:
    canonical = language.lower()
    if canonical == "ja":
        if _FAST_BUNKAI is None:
            raise RuntimeError(
                "fast-bunkai is required for --lang ja. Install `fast-bunkai` or choose another "
                "language."
            )
        return fast_bunkai_sentence_splitter
    if canonical in _NLTK_LANGUAGE_ALIAS_MAP:
        return _get_nltk_sentence_splitter(canonical)
    logging.warning(
        "No punkt-based splitter registered for language '%s'; falling back to a generic regex "
        "splitter.",
        canonical,
    )
    return generic_sentence_splitter


def text_to_spans(text: str, splitter: SentenceSplitter) -> list[list[int]]:
    sentences = splitter(text)
    spans: list[list[int]] = []
    cursor = 0
    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue
        start = text.find(stripped, cursor)
        if start == -1:
            start = text.find(stripped)
        if start == -1:
            continue
        end = start + len(stripped)
        spans.append([start, end])
        cursor = end
    if spans:
        return spans
    stripped = text.strip()
    if not stripped:
        return [[0, 0]]
    start = text.find(stripped)
    if start == -1:
        start = 0
    end = start + len(stripped)
    return [[start, end]]


def _normalise_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(part) for part in value if part)
    if isinstance(value, str):
        return value
    return str(value)


def extract_fields(example: dict[str, object]) -> tuple[str, str, list[str]]:
    query = _normalise_text(
        example.get("query") or example.get("question") or example.get("prompt")
    )
    if not query:
        raise ValueError("Example does not contain a query/question field.")
    positive = _normalise_text(
        example.get("answer")
        or example.get("positive")
        or example.get("pos")
        or example.get("target")
    )
    if not positive:
        raise ValueError("Example does not contain an answer/positive field.")

    negative_fields = [
        key
        for key in example.keys()
        if isinstance(key, str) and (key.startswith("negative") or key.startswith("neg"))
    ]
    negative_fields.sort()
    negatives: list[str] = []
    for field in negative_fields:
        value = _normalise_text(example.get(field))
        if value:
            negatives.append(value)
    return query, positive, negatives


def build_record(
    example: dict[str, object],
    *,
    splitter: SentenceSplitter,
    idx: int,
    dataset_slug: str,
    split: str,
) -> dict[str, object]:
    query, positive, negatives = extract_fields(example)
    texts = [positive, *negatives]
    context_spans = [text_to_spans(text, splitter) for text in texts]
    labels = [1] + [0] * (len(texts) - 1)
    record_id = f"{dataset_slug}:{split}:{idx}"
    return {
        "id": record_id,
        "query": query,
        "texts": texts,
        "context_spans": context_spans,
        "labels": labels,
    }


def slugify_dataset_name(name: str, subset: str | None) -> str:
    base = name.replace("/", "_")
    if subset:
        base = f"{base}_{subset}"
    return base


def convert_split(
    dataset: Dataset,
    *,
    splitter: SentenceSplitter,
    dataset_slug: str,
    split: str,
) -> Dataset:
    records: list[dict[str, object]] = []
    for idx, example_obj in enumerate(dataset):
        example = cast(dict[str, object], example_obj)
        record = build_record(
            example,
            splitter=splitter,
            idx=idx,
            dataset_slug=dataset_slug,
            split=split,
        )
        records.append(record)
    if not records:
        empty = {
            "id": [],
            "query": [],
            "texts": [],
            "context_spans": [],
            "labels": [],
        }
        return Dataset.from_dict(empty)

    return Dataset.from_list(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert sentence-transformer style datasets to Open Provence format."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Hugging Face dataset identifier, e.g. tomaarsen/natural-questions-hard-negatives.",
    )
    parser.add_argument("--subset", default=None, help="Optional dataset subset name.")
    parser.add_argument("--lang", default="en", help="Language key for the sentence splitter.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="If provided, shuffle each split and keep only N examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling before sampling.",
    )
    parser.add_argument(
        "--output-root",
        default="output/context-relevance-datasets/base",
        help="Root directory for converted datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing dataset directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    splitter = resolve_sentence_splitter(args.lang)
    dataset_slug = slugify_dataset_name(args.dataset, args.subset)

    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        logging.info("Loading local dataset from %s", dataset_path)
        dataset_dict = load_from_disk(str(dataset_path))
    else:
        logging.info("Loading dataset %s (subset=%s)", args.dataset, args.subset)
        dataset_dict = load_dataset(args.dataset, args.subset)
    if isinstance(dataset_dict, Dataset):
        dataset_dict = DatasetDict({"train": dataset_dict})
    elif not isinstance(dataset_dict, DatasetDict):
        raise TypeError("load_dataset must return a Dataset or DatasetDict.")

    if "train" not in dataset_dict:
        raise ValueError("Source dataset must expose a 'train' split.")

    processed_splits: dict[str, Dataset] = {}

    for split_name, split_dataset in dataset_dict.items():
        logging.info("Preparing split '%s' (%d examples)", split_name, len(split_dataset))
        if args.sample_size is not None:
            if args.sample_size <= 0:
                raise ValueError("--sample-size must be positive if provided.")
            split_dataset = split_dataset.shuffle(seed=args.seed)
            split_dataset = split_dataset.select(range(min(args.sample_size, len(split_dataset))))
        processed_splits[split_name] = split_dataset

    if "validation" not in processed_splits or "test" not in processed_splits:
        logging.info("Creating validation/test splits from train (1%% or 5k examples each).")
        train_split = processed_splits["train"].shuffle(seed=args.seed)
        total = len(train_split)
        if total == 0:
            raise ValueError("Train split is empty; cannot create validation/test splits.")

        def compute_split_size(total_rows: int) -> int:
            if total_rows <= 1:
                return 0
            desired = min(max(1, int(round(total_rows * 0.01))), 5000)
            max_for_split = max(total_rows - 1, 0)
            if max_for_split <= 1:
                return min(desired, max_for_split)
            return min(desired, max_for_split // 2 if max_for_split >= 2 else max_for_split)

        val_size = compute_split_size(total)
        if val_size == 0 and total > 1:
            val_size = min(1, total - 1)

        remaining = total - val_size
        test_size = compute_split_size(remaining)
        if test_size == 0 and remaining > 1:
            test_size = 1

        if total - val_size - test_size <= 0:
            shortfall = 1 - (total - val_size - test_size)
            if test_size > shortfall:
                test_size -= shortfall
            elif val_size > shortfall:
                val_size -= shortfall
            else:
                raise ValueError(
                    "Unable to allocate train/validation/test splits with positive size."
                )

        val_indices_end = val_size
        test_indices_end = val_size + test_size
        validation = train_split.select(range(val_indices_end))
        test = train_split.select(range(val_indices_end, test_indices_end))
        train = train_split.select(range(test_indices_end, total))

        processed_splits["train"] = train
        processed_splits["validation"] = validation
        processed_splits["test"] = test

        logging.info(
            "Split sizes -> train: %d, validation: %d, test: %d",
            len(train),
            len(validation),
            len(test),
        )

    converted: dict[str, Dataset] = {}
    for split_name, split_dataset in processed_splits.items():
        logging.info("Converting split '%s' to Provence format", split_name)
        converted_split = convert_split(
            split_dataset,
            splitter=splitter,
            dataset_slug=dataset_slug,
            split=split_name,
        )
        converted[split_name] = converted_split

    output_root = Path(args.output_root)
    output_dir = output_root / dataset_slug
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} already exists. Use --overwrite to replace it."
            )
        logging.warning("Overwriting existing directory at %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Saving converted dataset to %s", output_dir)
    dataset_dict = DatasetDict(converted)
    dataset_dict.save_to_disk(output_dir)
    logging.info(
        "Finished conversion. Saved splits: %s",
        ", ".join(f"{k}={len(v)}" for k, v in dataset_dict.items()),
    )


if __name__ == "__main__":
    main()
