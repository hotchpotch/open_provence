#!/usr/bin/env python3
"""Add ``context_spans_relevance`` labels to Provence-style datasets.

The script loads a dataset produced by
``scripts/context-relevance-datasets/generate_ds_from_sentense_transformer.py``,
runs a pruner LLM via vLLM to score each chunk across *all* available splits,
and writes the augmented dataset back to disk. Relevance inferences are cached
under ``./cache/context_spans_relevance/{dataset_name}/`` so interrupted runs
can be resumed safely.

Note:
    Execute this script with the Python interpreter from the pruning environment
    (e.g., ``/path/to/lm-trainers/pruning/.venv/bin/python``)
    so that vLLM dependencies are available.

Example:
    python scripts/context-relevance-datasets/add_context_spans_relevance.py \
        --dataset-path output/context-relevance-datasets/base/tomaarsen_natural-questions-hard-negatives_triplet-5 \
        --model hotchpotch/query-context-pruner-multilingual-Qwen3-4B \
        --overwrite
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import time
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

try:
    _vllm = importlib.import_module("vllm")
except ImportError:  # pragma: no cover - optional dependency
    class _MissingVLLM:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - helper
            raise ImportError(
                "vLLM is required to run add_context_spans_relevance.py. Install vllm before "
                "invoking this script."
            )

    class _MissingSamplingParams:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "vLLM is required to run add_context_spans_relevance.py. Install vllm before "
                "invoking this script."
            )

    LLM = _MissingVLLM  # type: ignore[assignment]
    SamplingParams = _MissingSamplingParams  # type: ignore[assignment]
else:
    LLM = cast(type[Any], getattr(_vllm, "LLM"))
    SamplingParams = cast(type[Any], getattr(_vllm, "SamplingParams"))

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    dataset_path: Path
    dataset_name: str
    query_column: str = "query"
    texts_column: str = "texts"
    context_spans_column: str = "context_spans"
    relevance_column: str = "context_spans_relevance"
    model_name_or_path: str = "hotchpotch/query-context-pruner-multilingual-Qwen3-4B"
    max_model_len: int = 1280
    gpu_memory_utilization: float = 0.9
    temperature: float = 0.1
    top_p: float = 0.95
    stop_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])
    batch_size: int = 200
    group_size: int = 1000
    cache_root: Path = field(default_factory=lambda: Path("./cache/context_spans_relevance"))
    output_path: Path | None = None
    overwrite: bool = False
    verbose: bool = False
    force_reprocess: bool = False
    debug: bool = False
    debug_limit: int = 100

    def cache_dir(self, split: str) -> Path:
        base = self.cache_root / self.dataset_name
        if self.debug:
            base = base / "debug"
        return base / split

    def resolved_output_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path
        suffix = "_with_relevance_debug" if self.debug else "_with_relevance"
        return self.dataset_path.parent / f"{self.dataset_path.name}{suffix}"


# --------------------------------------------------------------------------- #
# Dataset utilities
# --------------------------------------------------------------------------- #


class DatasetProcessor:
    def __init__(self, config: Config):
        self.config = config

    def load(self) -> DatasetDict:
        logger.info("Loading dataset from %s", self.config.dataset_path)
        data = load_from_disk(str(self.config.dataset_path))
        if isinstance(data, Dataset):
            data = DatasetDict({"train": data})
        return data

    def add_relevance(self, dataset: Dataset, relevance: Mapping[str, list[list[int]]]) -> Dataset:
        relevance_column = self.config.relevance_column
        spans_column = self.config.context_spans_column

        def inject(example: MutableMapping[str, Any], idx: int) -> MutableMapping[str, Any]:
            sample_id = example.get("id", f"idx_{idx}")
            spans = example.get(spans_column, [])
            example[relevance_column] = relevance.get(sample_id) or [[] for _ in spans]
            return example

        return dataset.map(inject, with_indices=True, num_proc=None)

    def save(self, dataset: DatasetDict) -> Path:
        output_path = self.config.resolved_output_path()
        temp_path = output_path
        if output_path.exists():
            if not self.config.overwrite:
                raise FileExistsError(
                    f"Output directory {output_path} already exists. Pass --overwrite to replace it."
                )
            temp_path = output_path.parent / f".{output_path.name}.tmpwrite"

        if temp_path.exists():
            _remove_tree(temp_path)

        logger.info("Writing dataset to %s", temp_path)
        dataset.save_to_disk(str(temp_path))

        if temp_path != output_path:
            if output_path.exists():
                logger.info("Removing previous dataset at %s", output_path)
                _remove_tree(output_path)
            temp_path.replace(output_path)

        return output_path


def _remove_tree(path: Path) -> None:
    if path.is_dir():
        for child in path.iterdir():
            _remove_tree(child)
        path.rmdir()
    else:
        path.unlink()


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


class CacheManager:
    def __init__(self, config: Config):
        self.config = config

    def path(self, split: str, start: int, end: int) -> Path:
        return self.config.cache_dir(split) / f"{start}-{end}.json"

    def load(self, split: str, start: int, end: int) -> dict[str, Any] | None:
        path = self.path(split, start, end)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def save(self, split: str, start: int, end: int, data: Mapping[str, Any]) -> None:
        path = self.path(split, start, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def exists(self, split: str, start: int, end: int) -> bool:
        return not self.config.force_reprocess and self.path(split, start, end).exists()


class ProgressTracker:
    """Track prompt-level progress within a split and estimate ETA."""

    def __init__(self, split_name: str, total_prompts: int):
        self.split_name = split_name
        self.total_prompts = max(total_prompts, 1)
        self.completed_prompts = 0
        self.start_time = time.time()
        from collections import deque

        self._per_prompt_times = deque(maxlen=10)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(0.0, seconds)
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def update(self, processed_prompts: int, batch_time: float | None) -> None:
        if processed_prompts <= 0:
            return
        self.completed_prompts += processed_prompts
        elapsed = time.time() - self.start_time
        percent = (self.completed_prompts / self.total_prompts) * 100

        if batch_time is None:
            logging.info(
                "[%s] Cache hit (+%d prompts) -> %d/%d (%.2f%%), elapsed=%s",
                self.split_name,
                processed_prompts,
                self.completed_prompts,
                self.total_prompts,
                percent,
                self._format_seconds(elapsed),
            )
            return

        per_prompt = batch_time / processed_prompts
        self._per_prompt_times.append(per_prompt)
        average = sum(self._per_prompt_times) / len(self._per_prompt_times)
        remaining = max(self.total_prompts - self.completed_prompts, 0)
        eta_str = self._format_seconds(remaining * average) if average > 0 else "--:--:--"
        logging.info(
            "[%s] Processed prompts: %d/%d (%.2f%%) | batch=%d prompts | "
            "batch_time=%.2fs | avg_last10=%.4fs/prompt | elapsed=%s | eta≈%s",
            self.split_name,
            self.completed_prompts,
            self.total_prompts,
            percent,
            processed_prompts,
            batch_time,
            average,
            self._format_seconds(elapsed),
            eta_str,
        )


# --------------------------------------------------------------------------- #
# LLM labeler
# --------------------------------------------------------------------------- #


class RelevanceLabeler:
    MAX_OUTPUT_TOKENS = 64

    def __init__(self, config: Config):
        self.config = config
        self.tokenizer: AutoTokenizer | None = None
        self.llm: Any | None = None
        self.sampling_params: Any | None = None

    def initialize(self) -> None:
        if self.llm is not None:
            return

        logger.info("Loading tokenizer %s", self.config.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
        )

        logger.info("Loading vLLM model %s", self.config.model_name_or_path)
        self.llm = LLM(
            model=self.config.model_name_or_path,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.MAX_OUTPUT_TOKENS,
            stop=self.config.stop_tokens,
        )

    # ------------------------------------------------------------------ #
    # Prompt building
    # ------------------------------------------------------------------ #

    def create_prompt(self, query: str, chunks: list[str]) -> str:
        assert self.tokenizer is not None
        safe_margin = 10
        token_limit = self.config.max_model_len - self.MAX_OUTPUT_TOKENS - safe_margin
        original_query = query

        for iteration in range(5):
            if len(query) > 256:
                query = query[:256]
            if iteration > 0:
                query, chunks = self._truncate_content(query, chunks, iteration)

            messages = self._build_messages(query, chunks)
            tokens = self.tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if len(tokens) <= token_limit:
                return self.tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            if self.config.verbose:
                logger.warning(
                    "Prompt truncation attempt %d failed (tokens=%d limit=%d)",
                    iteration,
                    len(tokens),
                    token_limit,
                )

        raise ValueError(
            "Failed to create prompt within token limit. "
            f"Original query length={len(original_query)}, chunks={len(chunks)}"
        )

    def _build_messages(self, query: str, chunks: list[str]) -> list[dict[str, str]]:
        chunk_block = "\n".join(f"[{idx + 1}] {text}" for idx, text in enumerate(chunks))
        return [{"role": "user", "content": f"{query}\n---\n{chunk_block}"}]

    def _truncate_content(
        self, query: str, chunks: list[str], iteration: int
    ) -> tuple[str, list[str]]:
        if len(query) > 100:
            query = query[:100]

        if iteration == 1:
            chunks = [chunk[:200] + ("..." if len(chunk) > 200 else "") for chunk in chunks[:10]]
        elif iteration == 2:
            chunks = [chunk[:100] + ("..." if len(chunk) > 100 else "") for chunk in chunks[:10]]
        elif iteration == 3:
            chunks = [chunk[:50] + ("..." if len(chunk) > 50 else "") for chunk in chunks[:5]]
        else:
            chunks = ["none"]
        return query, chunks

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #

    def parse_indices(self, response: str, num_chunks: int) -> list[int]:
        indices = []
        for number in re.findall(r"\d+", response):
            idx = int(number) - 1
            if 0 <= idx < num_chunks:
                indices.append(idx)
            elif self.config.verbose:
                logger.warning("Index out of range from model: %s (chunks=%d)", number, num_chunks)
        return sorted(set(indices))

    def to_flags(self, indices: Iterable[int], num_chunks: int) -> list[int]:
        flags = [0] * num_chunks
        for idx in indices:
            if 0 <= idx < num_chunks:
                flags[idx] = 1
        return flags

    # ------------------------------------------------------------------ #
    # Batch processing
    # ------------------------------------------------------------------ #

    def process_batch(
        self, batch: list[Mapping[str, Any]]
    ) -> tuple[dict[str, list[list[int]]], int]:
        prompts: list[str] = []
        metadata: list[dict[str, Any]] = []

        for sample_idx, sample in enumerate(batch):
            sample_id = sample.get("id", f"idx_{sample_idx}")
            query = sample[self.config.query_column]
            texts = sample[self.config.texts_column]
            spans = sample[self.config.context_spans_column]

            for text_index, (text, text_spans) in enumerate(zip(texts, spans)):
                chunks = [text[start:end].strip() for start, end in text_spans if end > start]
                if not chunks:
                    continue
                prompt = self.create_prompt(query, chunks)
                prompts.append(prompt)
                metadata.append(
                    {
                        "sample_id": sample_id,
                        "text_idx": text_index,
                        "num_chunks": len(chunks),
                    }
                )

        results: dict[str, list[list[int]]] = {}
        if not prompts:
            return results, 0

        assert self.llm is not None and self.sampling_params is not None
        outputs = self.llm.generate(prompts, self.sampling_params)

        for generated, meta in zip(outputs, metadata):
            response = generated.outputs[0].text
            indices = self.parse_indices(response, meta["num_chunks"])
            flags = self.to_flags(indices, meta["num_chunks"])

            sample_id = meta["sample_id"]
            if sample_id not in results:
                results[sample_id] = []
            while len(results[sample_id]) <= meta["text_idx"]:
                results[sample_id].append([])
            results[sample_id][meta["text_idx"]] = flags

        for sample_idx, sample in enumerate(batch):
            sample_id = sample.get("id", f"idx_{sample_idx}")
            spans = sample[self.config.context_spans_column]
            existing = results.setdefault(sample_id, [])
            while len(existing) < len(spans):
                existing.append([])

        return results, len(prompts)


# --------------------------------------------------------------------------- #
# Processing pipeline
# --------------------------------------------------------------------------- #


def process_split(
    config: Config,
    split_name: str,
    dataset: Dataset,
    cache: CacheManager,
    labeler: RelevanceLabeler,
) -> Dataset:
    required_columns = {
        config.query_column,
        config.texts_column,
        config.context_spans_column,
    }
    missing = required_columns - set(dataset.column_names)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    def count_prompts_range(start_idx: int, end_idx: int) -> int:
        count = 0
        for i in range(start_idx, end_idx):
            count += len(dataset[i][config.context_spans_column])
        return count

    total_prompts = count_prompts_range(0, len(dataset))
    progress_tracker = ProgressTracker(split_name, total_prompts)

    all_results: dict[str, list[list[int]]] = {}

    for start in range(0, len(dataset), config.group_size):
        end = min(start + config.group_size, len(dataset))

        if cache.exists(split_name, start, end):
            cached = cache.load(split_name, start, end)
            if cached:
                all_results.update(cached)
                processed = count_prompts_range(start, end)
                progress_tracker.update(processed, None)
                continue

        if labeler.llm is None:
            labeler.initialize()

        process_group = getattr(labeler, "process_group", None)
        if process_group is None:
            raise AttributeError("RelevanceLabeler.process_group is not available.")
        batch_results = cast(
            dict[str, list[list[int]]],
            process_group(dataset, start, end, progress_tracker),  # type: ignore[misc]
        )
        cache.save(split_name, start, end, batch_results)
        all_results.update(batch_results)

    return DatasetProcessor(config).add_relevance(dataset, all_results)


def _process_group(
    self,
    dataset: Dataset,
    start: int,
    end: int,
    progress_tracker: ProgressTracker,
) -> dict[str, list[list[int]]]:
    results: dict[str, list[list[int]]] = {}
    for idx in range(start, end, self.config.batch_size):
        batch_end = min(idx + self.config.batch_size, end)
        batch = [dataset[i] for i in range(idx, batch_end)]
        batch_start_time = time.time()
        batch_results, prompt_count = self.process_batch(batch)
        batch_time = time.time() - batch_start_time
        progress_tracker.update(prompt_count, batch_time)
        results.update(batch_results)
    return results


# Bind helper as method of RelevanceLabeler dynamically
setattr(RelevanceLabeler, "process_group", _process_group)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add context_spans_relevance to Provence datasets using a vLLM model."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        help="Path to the dataset directory produced by the converter script.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Name used for caching. Defaults to the dataset directory name.",
    )
    parser.add_argument(
        "--model",
        default="hotchpotch/query-context-pruner-multilingual-Qwen3-4B",
        help="vLLM-compatible model name or path.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1280,
        help="Maximum sequence length for the model context window.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to allocate to vLLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--stop-tokens",
        default="<|endoftext|>",
        help="Stop tokens separated by spaces. Use '\\n' for newline if needed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of samples to process per inference batch.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=1000,
        help="Number of samples per cache shard.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("./cache/context_spans_relevance"),
        help="Directory for caching inference outputs.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Destination for the augmented dataset. Defaults to `<dataset-path>_with_relevance`.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Ignore existing cache files and recompute.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Process only the first N examples per split (default 100) and save to debug paths.",
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=100,
        help="Number of examples per split when --debug is enabled (default: 100).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    dataset_name = args.dataset_name or args.dataset_path.name
    stop_tokens = [token for token in args.stop_tokens.replace("\\n", "\n").split(" ") if token]

    config = Config(
        dataset_path=args.dataset_path,
        dataset_name=dataset_name.replace("/", "_"),
        model_name_or_path=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_tokens=stop_tokens,
        batch_size=args.batch_size,
        group_size=args.group_size,
        cache_root=args.cache_root,
        output_path=args.output_path,
        overwrite=args.overwrite,
        verbose=args.verbose,
        force_reprocess=args.force_reprocess,
        debug=args.debug,
        debug_limit=args.debug_limit,
    )

    processor = DatasetProcessor(config)
    dataset_dict = processor.load()

    cache = CacheManager(config)
    labeler = RelevanceLabeler(config)

    augmented_splits = {}
    for split_name, split in dataset_dict.items():
        original_len = len(split)
        if config.debug:
            limit = min(config.debug_limit, original_len)
            split = split.select(range(limit))
            logger.info(
                "Debug mode active; limiting split '%s' from %d to %d rows",
                split_name,
                original_len,
                len(split),
            )
        logger.info("Processing split '%s' with %d rows", split_name, len(split))
        augmented_splits[split_name] = process_split(config, split_name, split, cache, labeler)

    augmented = DatasetDict(augmented_splits)
    output_path = processor.save(augmented)
    logger.info("Finished. Augmented dataset saved to %s", output_path)


if __name__ == "__main__":
    main()
