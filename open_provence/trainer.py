"""
OpenProvenceTrainer: Trainer for OpenProvenceEncoder models.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoConfig
from transformers.trainer import Trainer
from transformers.trainer_utils import set_seed
from transformers.training_args import TrainingArguments
from transformers.utils import logging as hf_logging

from . import (
    OpenProvenceDataCollator,
    OpenProvenceEncoder,
    OpenProvenceLoss,
)

try:
    import wandb

    _wandb_available = True
except ImportError:
    _wandb_available = False

from .modeling_open_provence_standalone import OpenProvenceConfig
from .utils.modeling_export import write_modeling_open_provence

logger = logging.getLogger(__name__)


def _load_dataset_dict(dataset_name: str | None, subset: str | None) -> DatasetDict:
    """Load a dataset from Hugging Face or a local ``datasets.save_to_disk`` directory."""

    if dataset_name:
        dataset_path = Path(dataset_name).expanduser()
        if dataset_path.exists():
            logger.info("Loading local dataset from %s", dataset_path)
            return cast(DatasetDict, load_from_disk(str(dataset_path)))

    return cast(DatasetDict, load_dataset(dataset_name or "", subset or None))


def _sample_dataset_randomly(
    dataset: Dataset,
    sample_size: int,
    rnd: random.Random,
    dataset_label: str,
) -> Dataset:
    """Return a deterministic random subset of ``dataset`` of length ``sample_size``."""

    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")

    dataset_length = len(dataset)
    if dataset_length <= sample_size:
        logger.info(
            "Skipping sampling for %s: requested %s samples but dataset has %s rows",
            dataset_label,
            sample_size,
            dataset_length,
        )
        return dataset

    indices = sorted(rnd.sample(range(dataset_length), sample_size))
    logger.info(
        "Sampled %s/%s rows from %s using deterministic random generator",
        sample_size,
        dataset_length,
        dataset_label,
    )
    return cast(Dataset, dataset.select(indices))


def run_eval_datasets_for_model(
    model_path: str | Path, eval_settings: dict[str, Any] | None
) -> None:
    """Run eval_datasets script for a saved model if configuration is provided."""
    if not eval_settings:
        logger.info("eval_datasets settings not provided; skipping dataset evaluation.")
        return

    config_path = eval_settings.get("config")
    if not config_path:
        logger.warning("eval_datasets config not specified; skipping dataset evaluation.")
        return

    threshold = eval_settings.get("threshold")
    if threshold is None:
        threshold = eval_settings.get("threadshold")  # Backwards compatibility typo
    if threshold is None:
        threshold = 0.1

    batch_size = eval_settings.get("batch_size", 256)
    timing_details = eval_settings.get("timing_details", True)

    model_path = Path(model_path)
    output_dir = model_path / "eval_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "results.json"
    output_md = output_dir / "results.md"

    uv_executable = shutil.which("uv")
    if uv_executable:
        command = [
            uv_executable,
            "run",
            "python",
            "scripts/eval_datasets.py",
        ]
    else:
        logger.info("Could not find 'uv' command; falling back to current Python interpreter.")
        command = [sys.executable, "scripts/eval_datasets.py"]

    command.extend(
        [
            "--config",
            str(config_path),
            "--model",
            str(model_path),
            "--threshold",
            str(threshold),
            "--batch-size",
            str(batch_size),
            "--output-json",
            str(output_json),
            "--output-file",
            str(output_md),
        ]
    )
    if timing_details:
        command.append("--timing-details")

    logger.info("Running eval_datasets: %s", " ".join(command))
    subprocess.run(command, check=True)

    if output_md.exists():
        print(f"\n{'=' * 80}")
        print("Eval Datasets Results")
        print(f"{'=' * 80}")
        print(output_md.read_text(encoding="utf-8"))
        print(f"{'=' * 80}\n")


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""

    model_name_or_path: str = field(
        default="hotchpotch/japanese-reranker-xsmall-v2",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    num_labels: int | None = field(
        default=None,
        metadata={
            "help": "Number of labels for ranking head. If None, will auto-detect from model or use 2 (Provence default)"
        },
    )
    classifier_dropout: float = field(
        default=0.1, metadata={"help": "Dropout rate for classifier heads"}
    )
    max_length: int = field(default=512, metadata={"help": "Maximum sequence length"})
    config_name: str | None = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: str | None = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    dataset_name: str = field(
        default="hotchpotch/wip-msmarco-context-relevance",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    subset: str = field(default="msmarco-ja-minimal", metadata={"help": "Dataset subset to use"})
    teacher_column: str | None = field(
        default=None,
        metadata={
            "help": "Column name containing teacher scores (e.g., 'teacher_scores.model-name')"
        },
    )
    datasets: list[dict[str, Any]] | None = field(
        default=None,
        metadata={
            "help": "List of datasets with their teacher columns for multi-dataset training"
        },
    )
    items: int | None = field(
        default=None,
        metadata={
            "help": "If set, limit each query to this many items by keeping positives first and sampling the rest"
        },
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of training examples"},
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of evaluation examples"},
    )
    validation_split: float | None = field(
        default=None,
        metadata={"help": "Validation split ratio (0-1). If None, use existing validation set"},
    )
    validation_split_samples: int | None = field(
        default=None,
        metadata={
            "help": "Number of validation samples to split. Takes precedence over validation_split ratio"
        },
    )
    validation_split_name: str = field(
        default="validation",
        metadata={
            "help": "Name of the validation split to use (e.g., 'validation', 'test', 'dev')"
        },
    )
    preprocessing_num_workers: int | None = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    filter_zero_relevance_max_items: int | None = field(
        default=None,
        metadata={
            "help": "If set, filters rows with all-zero relevance and limits items per row to this number. Default: None (no filtering)"
        },
    )
    filter_zero_relevance_max_items_reverse: bool = field(
        default=False,
        metadata={
            "help": "If True, sort by relevance average in ascending order (keep low relevance items). Default: False (keep high relevance items)"
        },
    )
    filter_keep_first_item: bool = field(
        default=False,
        metadata={
            "help": "If True, always keep the first item regardless of its relevance score. Default: False"
        },
    )
    upsample_factor: float | None = field(
        default=None,
        metadata={
            "help": "Optional multiplier ≥ 1.0 to duplicate training samples for the primary dataset."
        },
    )


@dataclass
class PruningTrainingArguments(TrainingArguments):
    """Training arguments specific to OpenProvenceEncoder training."""

    output_dir: str | None = field(
        default=None,
        metadata={
            "help": "Output directory for model and checkpoints. Format example: ./output/japanese-reranker-xsmall-v2_reranking-pruning_minimal_20250111_123456"
        },
    )
    ranking_weight: float = field(
        default=0.05, metadata={"help": "Weight for ranking loss (Provence paper default: 0.05)"}
    )
    pruning_weight: float = field(
        default=1.0, metadata={"help": "Weight for pruning loss (Provence paper default: 1.0)"}
    )
    use_teacher_scores: bool = field(
        default=True, metadata={"help": "Whether to use teacher scores for distillation"}
    )
    sentence_level_pruning: bool = field(
        default=True, metadata={"help": "Whether to perform sentence-level pruning"}
    )
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Remove columns not required by the model"}
    )
    # Override some defaults
    per_device_train_batch_size: int = field(
        default=32, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=2,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    num_train_epochs: float = field(
        default=1.0, metadata={"help": "Total number of training epochs to perform."}
    )
    warmup_ratio: float = field(
        default=0.1, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    optim: str = field(default="adafactor", metadata={"help": "The optimizer to use."})
    bf16: bool = field(
        default=True,
        metadata={
            "help": "Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training."
        },
    )
    lr_scheduler_type: str = field(
        default="cosine", metadata={"help": "The scheduler type to use."}
    )
    eval_datasets: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Optional configuration dict describing eval_datasets run after training."
        },
    )


class OpenProvenceTrainer(Trainer):
    """Custom Trainer that uses OpenProvenceTrainer internally for compatibility."""

    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self._eval_loss_components = {}
        # Track loss components similar to yast
        self._accumulated_loss_components = {}
        self._loss_component_counts = {}

    def _save(self, output_dir: str | None = None, state_dict=None):
        """Save model checkpoint, ensuring config.json is saved."""
        # Call parent's _save method first
        super()._save(output_dir, state_dict)

        # Additionally save config.json
        if output_dir is not None and hasattr(self.model, "mode"):
            model = cast(OpenProvenceEncoder, self.model)
            # Create and save OpenProvenceConfig
            # OpenProvenceConfig imported at top of file

            # Get the hidden size from the actual model
            hidden_size = model.ranking_model.config.hidden_size

            # Get classifier_dropout from pruning_head config or use default
            classifier_dropout = 0.1  # default
            if hasattr(model, "pruning_head") and hasattr(model.pruning_head, "config"):
                classifier_dropout = getattr(model.pruning_head.config, "classifier_dropout", 0.1)

            base_config_dict = model.ranking_model.config.to_dict()
            config = OpenProvenceConfig(
                model_type="open_provence",
                base_model_name_or_path=model.model_name_or_path,
                base_model_config=base_config_dict,
                tokenizer_name_or_path=None,
                mode="reranking_pruning",
                num_labels=model.num_labels,
                num_pruning_labels=2,  # Always 2 for binary pruning (keep/prune)
                classifier_dropout=classifier_dropout,
                max_length=model.max_length,
                pruning_config={"hidden_size": hidden_size},
                encoder_architecture=getattr(model.ranking_model.config, "model_type", None),
                auto_map={
                    "AutoConfig": "modeling_open_provence_standalone.OpenProvenceConfig",
                    "AutoModel": "modeling_open_provence_standalone.OpenProvenceForSequenceClassification",
                    "AutoModelForSequenceClassification": "modeling_open_provence_standalone.OpenProvenceForSequenceClassification",
                    "AutoModelForTokenClassification": "modeling_open_provence_standalone.OpenProvenceForTokenClassification",
                },
            )
            config.save_pretrained(output_dir)

            standalone_file = Path(__file__).parent / "modeling_open_provence_standalone.py"
            if standalone_file.exists():
                write_modeling_open_provence(
                    standalone_file,
                    Path(output_dir) / "modeling_open_provence_standalone.py",
                )

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute loss using OpenProvenceLoss."""
        sentence_features = inputs["sentence_features"]
        labels = inputs["labels"]

        # Move tensors to device
        for key in sentence_features[0]:
            if isinstance(sentence_features[0][key], torch.Tensor):
                sentence_features[0][key] = sentence_features[0][key].to(model.device)

        for key in labels:
            if isinstance(labels[key], torch.Tensor):
                labels[key] = labels[key].to(model.device)

        # Compute loss
        loss = self.loss_fn(sentence_features, labels)

        # Track loss components for aggregation (similar to yast)
        if hasattr(self.loss_fn, "last_loss_components") and self.loss_fn.last_loss_components:
            for name, value in self.loss_fn.last_loss_components.items():
                if name not in self._accumulated_loss_components:
                    self._accumulated_loss_components[name] = 0.0
                    self._loss_component_counts[name] = 0

                if isinstance(value, torch.Tensor):
                    self._accumulated_loss_components[name] += value.item()
                else:
                    self._accumulated_loss_components[name] += value
                self._loss_component_counts[name] += 1

        if return_outputs:
            return loss, None
        return loss

    def log(self, logs: dict[str, Any], start_time: float | None = None, **kwargs: Any) -> None:
        """Override log method to include accumulated loss components (inspired by yast)."""
        # Add step and epoch
        logs["step"] = self.state.global_step
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        # Calculate and add mean loss components
        if self._accumulated_loss_components:
            mean_components = {}
            for name, total in self._accumulated_loss_components.items():
                count = self._loss_component_counts.get(name, 1)
                if count > 0:
                    mean_components[name] = total / count

            # Update logs with mean components
            logs.update(mean_components)

            # Clear accumulators
            self._accumulated_loss_components.clear()
            self._loss_component_counts.clear()

        # Append to log history
        output = {**logs, "step": self.state.global_step}
        self.state.log_history.append(output)

        # Call parent's callback handler
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Custom prediction step that handles OpenProvenceDataCollator format."""
        has_labels = "labels" in inputs

        with torch.no_grad():
            loss_tensor: torch.Tensor | None = None
            if has_labels:
                computed_loss = self.compute_loss(model, inputs, return_outputs=False)
                if isinstance(computed_loss, tuple):
                    loss_tensor = computed_loss[0]
                else:
                    loss_tensor = computed_loss

                # Track eval loss components
                if (
                    hasattr(self.loss_fn, "last_loss_components")
                    and self.loss_fn.last_loss_components
                ):
                    for name, value in self.loss_fn.last_loss_components.items():
                        if name not in self._eval_loss_components:
                            self._eval_loss_components[name] = []
                        if isinstance(value, torch.Tensor):
                            self._eval_loss_components[name].append(value.item())
                        else:
                            self._eval_loss_components[name].append(value)
            else:
                loss_tensor = None

        # For evaluation, we mainly care about the loss
        if prediction_loss_only:
            return (loss_tensor, None, None)

        # We don't need logits for evaluation in this case
        return (loss_tensor, None, None)

    def evaluation_loop(self, *args, **kwargs):
        """Override evaluation loop to log loss components."""
        # Reset eval loss components
        self._eval_loss_components = {}

        # Run parent evaluation loop
        output = super().evaluation_loop(*args, **kwargs)

        # Calculate average loss components and add to metrics
        if self._eval_loss_components:
            for name, values in self._eval_loss_components.items():
                avg_value = sum(values) / len(values)
                output.metrics[f"eval_{name}"] = avg_value

        return output


def filter_pruning_dataset(
    dataset: Dataset,
    max_items: int,
    num_proc: int = 11,
    reverse_sort: bool = False,
    keep_first: bool = False,
) -> Dataset:
    """
    Filter dataset for pruning training:
    1. Remove items within each row where context_spans_relevance is all zeros
    2. Limit each row to first max_items non-zero items
    3. Remove rows with less than max_items items after filtering

    Args:
        dataset: HuggingFace dataset to filter
        max_items: Maximum number of items per row
        num_proc: Number of processes for parallel processing
        reverse_sort: If True, sort by relevance in ascending order (keep low relevance items)
        keep_first: If True, always keep the first item regardless of relevance

    Returns:
        Filtered dataset
    """
    logger = logging.getLogger(__name__)
    initial_size = len(dataset)

    # Step 1 & 2: Remove zero-relevance items and limit to max_items
    first_run = True

    def filter_and_limit_items(example: dict[str, Any]) -> dict[str, Any]:
        nonlocal first_run
        relevance = example.get("context_spans_relevance", [])
        if not relevance:
            return example

        # Get the original length
        original_length = len(relevance)

        # Initialize indices to keep
        indices_to_keep = []

        # If keep_first is True, always include the first item
        if keep_first and len(relevance) > 0:
            indices_to_keep.append(0)
            start_idx = 1  # Start processing from the second item
            remaining_slots = max_items - 1
        else:
            start_idx = 0
            remaining_slots = max_items

        # Calculate average relevance for each item and collect non-zero items
        items_with_avg = []
        for i in range(start_idx, len(relevance)):
            item = relevance[i]
            if isinstance(item, list):
                # Calculate average relevance
                avg_relevance = sum(item) / len(item) if len(item) > 0 else 0
                # Only include items with at least one non-zero element
                if any(r != 0 for r in item):
                    items_with_avg.append((i, avg_relevance))
            else:
                # Single value case
                if item != 0:
                    items_with_avg.append((i, item))

        # Sort by average relevance and take remaining slots
        # If reverse_sort is True, sort in ascending order (keep low relevance items)
        if remaining_slots > 0:
            items_with_avg.sort(key=lambda x: x[1], reverse=not reverse_sort)
            indices_to_keep.extend([idx for idx, _ in items_with_avg[:remaining_slots]])

        # Keep the indices in their original order
        indices_to_keep.sort()

        # Find all fields that are lists with the same length as context_spans_relevance
        fields_to_filter = []
        for field_name, value in example.items():
            if isinstance(value, list) and len(value) == original_length:
                fields_to_filter.append(field_name)

        # Log fields to be filtered (only on first example to avoid spam)
        if first_run:
            first_run = False
            logger.debug(
                f"Fields to filter (same length as context_spans_relevance): {fields_to_filter}"
            )

        # Filter all identified fields by the indices to keep
        for field_name in fields_to_filter:
            example[field_name] = [
                example[field_name][i] for i in indices_to_keep if i < len(example[field_name])
            ]

        return example

    logger.info(f"Filtering zero-relevance items and limiting to {max_items} items per row...")
    dataset = cast(Dataset, dataset.map(filter_and_limit_items, num_proc=num_proc))

    # Step 3: Remove rows with less than max_items items
    def has_at_least_n_items(example: dict[str, Any]) -> bool:
        relevance = example.get("context_spans_relevance", [])
        return len(relevance) >= max_items

    logger.info(f"Removing rows with less than {max_items} items...")
    dataset = cast(Dataset, dataset.filter(has_at_least_n_items, num_proc=num_proc))

    final_size = len(dataset)
    logger.info(
        f"Final dataset size: {final_size} rows ({final_size / initial_size * 100:.1f}% of original)"
    )
    logger.info(f"Removed {initial_size - final_size} rows total")

    return dataset


def sample_items_by_label_priority(
    dataset: Dataset,
    max_items: int,
    seed: int,
    *,
    label_column: str = "labels",
    num_proc: int = 11,
) -> Dataset:
    """
    Downsample query candidates by keeping positives first and randomly sampling the rest.

    Args:
        dataset: HuggingFace dataset to filter.
        max_items: Maximum number of items per query.
        seed: Integer seed to produce deterministic sampling.
        label_column: Column containing binary labels.
        num_proc: Number of processes for parallel processing.

    Returns:
        Filtered dataset with at most max_items items per row.
    """
    if max_items <= 0:
        raise ValueError("items must be a positive integer")

    logger = logging.getLogger(__name__)
    initial_size = len(dataset)

    label_column_present = label_column in dataset.column_names

    sample_reference_column: str | None = None
    fallback_warning_logged = False

    if not label_column_present:
        # Heuristically pick a column that mirrors candidate lists so we can still cap items
        preferred_candidates = [
            "texts",
            "context_spans",
            "context",
            "passages",
        ]

        for candidate in preferred_candidates:
            if candidate in dataset.column_names:
                sample_reference_column = candidate
                break

        if sample_reference_column is None:
            # Fallback: choose the first list-like column from the first row
            first_row = dataset[0] if len(dataset) else {}
            for name, value in first_row.items():
                if isinstance(value, list):
                    sample_reference_column = name
                    break

        if sample_reference_column is None:
            logger.warning(
                "Could not find a list column to apply 'items' sampling without '%s'. Skipping.",
                label_column,
            )
            return dataset

        logger.info(
            "Column '%s' not found; falling back to uniform sampling using '%s'.",
            label_column,
            sample_reference_column,
        )

    first_run = True

    def sample_and_limit(example: dict[str, Any], idx: int) -> dict[str, Any]:
        nonlocal first_run, fallback_warning_logged
        reference_column = label_column if label_column_present else sample_reference_column
        labels = example.get(label_column) if label_column_present else None

        if label_column_present and isinstance(labels, list):
            original_length = len(labels)
        else:
            reference_values = example.get(reference_column) if reference_column else None
            if not isinstance(reference_values, list):
                if not fallback_warning_logged:
                    fallback_warning_logged = True
                    logger.warning(
                        "Row is missing usable list data for sampling (index %s); leaving unchanged.",
                        idx,
                    )
                return example
            original_length = len(reference_values)

        if original_length == 0:
            return example

        selected_indices: list[int]

        if label_column_present and isinstance(labels, list):
            positive_indices = [i for i, value in enumerate(labels) if value == 1]
            negative_indices = [i for i, value in enumerate(labels) if value != 1]

            selected_indices = []

            if positive_indices:
                selected_indices.extend(positive_indices[:max_items])

            remaining_slots = max_items - len(selected_indices)

            if remaining_slots > 0:
                candidates = negative_indices if positive_indices else list(range(original_length))
                rng = random.Random(seed + idx)
                rng.shuffle(candidates)
                selected_indices.extend(candidates[:remaining_slots])
        else:
            rng = random.Random(seed + idx)
            candidates = list(range(original_length))
            rng.shuffle(candidates)
            selected_indices = candidates[:max_items]

        selected_indices = sorted(set(i for i in selected_indices if i < original_length))

        # Identify fields that align with the candidate dimension
        fields_to_filter = []
        for field_name, value in example.items():
            if isinstance(value, list) and len(value) == original_length:
                fields_to_filter.append(field_name)

        if first_run:
            first_run = False
            logger.debug(
                f"[items filter] Fields trimmed alongside '{reference_column}': {fields_to_filter}"
            )

        for field_name in fields_to_filter:
            example[field_name] = [example[field_name][i] for i in selected_indices]

        return example

    label_desc = label_column if label_column_present else f"uniform({sample_reference_column})"
    logger.info(f"Sampling items by '{label_desc}' priority with max_items={max_items}...")
    dataset = cast(Dataset, dataset.map(sample_and_limit, with_indices=True, num_proc=num_proc))

    def has_required_items(example):
        if label_column_present:
            labels = example.get(label_column, [])
            return isinstance(labels, list) and len(labels) >= max_items
        reference_values = example.get(sample_reference_column, [])
        return isinstance(reference_values, list) and len(reference_values) >= max_items

    logger.info(f"Removing rows with fewer than {max_items} items after sampling...")
    dataset = cast(Dataset, dataset.filter(has_required_items, num_proc=num_proc))

    final_size = len(dataset)
    removed_rows = initial_size - final_size
    if initial_size:
        retention_pct = final_size / initial_size * 100
    else:
        retention_pct = 0.0
    logger.info(
        f"Final dataset size after 'items' sampling: {final_size} rows ({retention_pct:.1f}% of original)"
    )
    logger.info(
        f"Removed {removed_rows} rows that did not have at least {max_items} items after sampling"
    )

    return dataset


def upsample_dataset(
    dataset: Dataset,
    multiplier: float,
    *,
    seed: int,
    dataset_label: str | None = None,
) -> Dataset:
    """
    Duplicate dataset rows to upsample small domains quickly using Hugging Face primitives.

    Args:
        dataset: Hugging Face Dataset to upsample.
        multiplier: Factor >= 1.0 determining how many copies (with a fractional tail) to add.
        seed: Random seed to sample the fractional tail deterministically.

    Returns:
        Concatenated dataset with approximately multiplier * len(dataset) rows.
    """
    if multiplier < 1.0:
        raise ValueError("upsample_factor must be >= 1.0")

    base_size = len(dataset)
    if base_size == 0:
        return dataset
    if multiplier <= 1.0:
        logger = logging.getLogger(__name__)
        target = dataset_label or "dataset"
        logger.info(
            f"Upsample factor {multiplier:.3f} specified for {target}, but ≤ 1.0. Skipping duplication."
        )
        return dataset

    logger = logging.getLogger(__name__)

    whole_copies = int(multiplier)
    fractional = multiplier - whole_copies

    pieces: list[Dataset] = []
    if whole_copies > 0:
        pieces.extend([dataset] * whole_copies)

    take = 0
    if fractional > 1e-6:
        take = int(round(fractional * base_size))
        take = max(1, min(take, base_size))
        tail = dataset.shuffle(seed=seed).select(range(take))
        pieces.append(tail)

    if not pieces:
        return dataset

    upsampled = concatenate_datasets(pieces)
    target = dataset_label or "dataset"
    added = len(upsampled) - base_size
    detail = f"{whole_copies} full copy/copies" if whole_copies else ""
    if fractional > 1e-6:
        fractional_pct = take / base_size * 100
        fractional_info = f"{take} samples ({fractional_pct:.1f}%) from shuffled tail"
        detail = f"{detail}, {fractional_info}" if detail else fractional_info
    if not detail:
        detail = "exact duplicate"
    logger.info(
        f"Upsampled {target} from {base_size:,} to {len(upsampled):,} rows "
        f"(factor={multiplier:.3f}, +{added:,}; {detail})."
    )
    return upsampled


def prepare_dataset(data_args: DataArguments, seed: int = 42) -> tuple[Any, Any]:
    """
    Load dataset with filtering - let OpenProvenceDataCollator handle the processing.

    Args:
        data_args: Data arguments containing dataset info
        seed: Random seed for splitting
    """
    # Convert single dataset to datasets list format for unified processing
    if data_args.datasets:
        datasets_to_load = data_args.datasets
        logger.info(f"Loading {len(datasets_to_load)} datasets for concatenation")
    else:
        # Convert single dataset to list format
        # Use teacher_column from data_args if specified, otherwise default to "teacher_score"
        teacher_column = data_args.teacher_column if data_args.teacher_column else "teacher_score"

        datasets_to_load = [
            {
                "dataset_name": data_args.dataset_name,
                "subset": data_args.subset,
                "teacher_column": teacher_column,
                **({"items": data_args.items} if data_args.items is not None else {}),
                **(
                    {"upsample_factor": data_args.upsample_factor}
                    if data_args.upsample_factor is not None
                    else {}
                ),
            }
        ]
        logger.info(f"Loading dataset: {data_args.dataset_name}:{data_args.subset}")

    train_datasets = []
    eval_datasets = []

    rnd = random.Random(seed)

    num_proc = data_args.preprocessing_num_workers or 11

    # Process each dataset
    for dataset_config in datasets_to_load:
        dataset_name = dataset_config.get("dataset_name")
        subset = dataset_config.get("subset")
        teacher_column = dataset_config.get("teacher_column", "teacher_score")
        items_per_query = dataset_config.get("items", data_args.items)
        upsample_factor = dataset_config.get("upsample_factor", data_args.upsample_factor)
        sample_size = dataset_config.get("n_samples")
        dataset_id = f"{dataset_name}:{subset}" if dataset_name else subset or "train"
        train_sampling_ratio: float | None = None

        dataset = _load_dataset_dict(dataset_name, subset)

        # Process train dataset
        train_ds = cast(Dataset, dataset["train"])  # type: ignore[index]
        original_train_size = len(train_ds)

        # Apply filtering if specified
        if data_args.filter_zero_relevance_max_items is not None:
            logger.info(
                f"Applying filtering to {dataset_name}:{subset} train set (removing zero-relevance items, max_items={data_args.filter_zero_relevance_max_items})"
            )
            train_ds = filter_pruning_dataset(
                train_ds,
                data_args.filter_zero_relevance_max_items,
                num_proc=num_proc,
                reverse_sort=data_args.filter_zero_relevance_max_items_reverse,
                keep_first=data_args.filter_keep_first_item,
            )
            filtered_train_size = len(train_ds)
            logger.info(
                f"  → {dataset_name}:{subset} train: {original_train_size:,} → {filtered_train_size:,} samples ({filtered_train_size / original_train_size * 100:.1f}% retained)"
            )

        if items_per_query is not None:
            logger.info(
                f"Applying 'items' sampling to {dataset_name}:{subset} train set (items={items_per_query})"
            )
            train_ds = sample_items_by_label_priority(
                train_ds,
                items_per_query,
                seed=seed,
                label_column="labels",
                num_proc=num_proc,
            )

        # Rename teacher column to unified name
        if teacher_column != "teacher_score" and teacher_column in train_ds.column_names:  # type: ignore[attr-defined]
            logger.info(f"Renaming {teacher_column} to teacher_score")
            train_ds = cast(Dataset, train_ds.rename_column(teacher_column, "teacher_score"))  # type: ignore[attr-defined]

        if upsample_factor is not None:
            train_ds = upsample_dataset(
                train_ds,
                float(upsample_factor),
                seed=seed,
                dataset_label=f"{dataset_id} train",
            )

        if sample_size is not None:
            try:
                sample_size_value = float(sample_size)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Invalid n_samples value {sample_size!r} for {dataset_name}:{subset}"
                ) from exc

            if sample_size_value <= 0:
                raise ValueError("n_samples must be greater than 0")

            pre_sample_size = len(train_ds)

            if sample_size_value <= 1:
                target_count = max(1, math.ceil(pre_sample_size * sample_size_value))
            else:
                target_count = int(sample_size_value)

            target_count = min(pre_sample_size, target_count)

            train_ds = _sample_dataset_randomly(
                train_ds,
                target_count,
                rnd,
                dataset_label=f"{dataset_id} train",
            )

            if pre_sample_size > 0:
                train_sampling_ratio = len(train_ds) / pre_sample_size
            else:
                train_sampling_ratio = 1.0
        else:
            train_sampling_ratio = None

        train_datasets.append(train_ds)

        # Process eval dataset - check multiple possible splits
        eval_split = None
        if data_args.validation_split_name in dataset:
            eval_split = data_args.validation_split_name
        elif "validation" in dataset:
            eval_split = "validation"
        elif "test" in dataset:
            eval_split = "test"

        if eval_split:
            logger.info(f"Using existing {eval_split} set for {dataset_name}:{subset}")
            eval_ds = cast(Dataset, dataset[eval_split])  # type: ignore[index]
            original_eval_size = len(eval_ds)

            # Apply filtering if specified
            if data_args.filter_zero_relevance_max_items is not None:
                logger.info(
                    f"Applying filtering to {dataset_name}:{subset} {eval_split} set (removing zero-relevance items, max_items={data_args.filter_zero_relevance_max_items})"
                )
                eval_ds = filter_pruning_dataset(
                    eval_ds,
                    data_args.filter_zero_relevance_max_items,
                    num_proc=num_proc,
                    reverse_sort=data_args.filter_zero_relevance_max_items_reverse,
                    keep_first=data_args.filter_keep_first_item,
                )
                filtered_eval_size = len(eval_ds)
                logger.info(
                    f"  → {dataset_name}:{subset} {eval_split}: {original_eval_size:,} → {filtered_eval_size:,} samples ({filtered_eval_size / original_eval_size * 100:.1f}% retained)"
                )

            if items_per_query is not None:
                logger.info(
                    f"Applying 'items' sampling to {dataset_name}:{subset} {eval_split} set (items={items_per_query})"
                )
                eval_ds = sample_items_by_label_priority(
                    eval_ds,
                    items_per_query,
                    seed=seed,
                    label_column="labels",
                    num_proc=num_proc,
                )

            if teacher_column != "teacher_score" and teacher_column in eval_ds.column_names:  # type: ignore[attr-defined]
                eval_ds = eval_ds.rename_column(teacher_column, "teacher_score")  # type: ignore[attr-defined]

            if sample_size is not None and train_sampling_ratio is not None and len(eval_ds) > 0:
                eval_sample_size = min(
                    len(eval_ds),
                    max(1, math.ceil(len(eval_ds) * train_sampling_ratio)),
                )

                eval_ds = _sample_dataset_randomly(
                    eval_ds,
                    eval_sample_size,
                    rnd,
                    dataset_label=f"{dataset_id} {eval_split}",
                )

            eval_datasets.append(eval_ds)

    # Combine datasets
    if len(train_datasets) > 1:
        # Multiple datasets - need to find common columns
        common_columns = set(train_datasets[0].column_names)
        for ds in train_datasets[1:]:
            common_columns = common_columns.intersection(set(ds.column_names))

        # Prioritize essential columns
        essential_columns = ["query", "positive", "negative", "teacher_score"]
        context_columns = ["context_spans", "context_spans_relevance"]

        # Build column list with priority
        existing_columns = []

        # Add essential columns first
        for col in essential_columns:
            if col in common_columns:
                existing_columns.append(col)

        # Add context columns if available
        for col in context_columns:
            if col in common_columns:
                existing_columns.append(col)

        # Add remaining common columns
        for col in sorted(common_columns):
            if col not in existing_columns:
                existing_columns.append(col)

        logger.info(f"Using columns: {existing_columns}")

        # Select columns and concatenate
        train_datasets = [ds.select_columns(existing_columns) for ds in train_datasets]
        train_dataset = concatenate_datasets(train_datasets)
        logger.info(f"Concatenated train dataset size: {len(train_dataset):,}")

        if eval_datasets:
            eval_datasets = [
                ds.select_columns(existing_columns)
                for ds in eval_datasets
                if all(col in ds.column_names for col in existing_columns)
            ]
            eval_dataset = concatenate_datasets(eval_datasets) if eval_datasets else None
            if eval_dataset:
                logger.info(f"Concatenated eval dataset size: {len(eval_dataset):,}")
        else:
            eval_dataset = None
    else:
        # Single dataset
        train_dataset = train_datasets[0]
        eval_dataset = eval_datasets[0] if eval_datasets else None

    # Handle validation split if no eval dataset exists
    if eval_dataset is None and (
        data_args.validation_split is not None or data_args.validation_split_samples is not None
    ):
        if data_args.validation_split_samples is not None:
            # Use absolute number of samples
            if (
                data_args.validation_split_samples <= 0
                or data_args.validation_split_samples >= len(train_dataset)
            ):
                raise ValueError(
                    f"validation_split_samples must be between 1 and {len(train_dataset) - 1}"
                )
            logger.info(
                f"Creating validation split with {data_args.validation_split_samples} samples"
            )
            ratio = data_args.validation_split_samples / len(train_dataset)
        else:
            # Use ratio
            if data_args.validation_split is None or not (0 < data_args.validation_split < 1):
                raise ValueError("validation_split must be between 0 and 1")
            logger.info(
                f"Creating validation split with {data_args.validation_split:.0%} of training data"
            )
            ratio = data_args.validation_split

        split_dataset = train_dataset.train_test_split(test_size=ratio, seed=seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    # Apply sampling if specified
    if data_args.max_train_samples and len(train_dataset) > data_args.max_train_samples:
        logger.info(
            f"Sampling {data_args.max_train_samples} training examples from {len(train_dataset):,}"
        )
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if (
        eval_dataset is not None
        and data_args.max_eval_samples
        and len(eval_dataset) > data_args.max_eval_samples
    ):
        logger.info(
            f"Sampling {data_args.max_eval_samples} evaluation examples from {len(eval_dataset):,}"
        )
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Log final sizes
    logger.info("Final dataset sizes:")
    logger.info(f"  Train samples: {len(train_dataset):,}")
    logger.info(f"  Validation samples: {len(eval_dataset) if eval_dataset is not None else 0:,}")

    return train_dataset, eval_dataset


def calculate_dynamic_steps(
    dataset_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: float,
    num_devices: int = 1,
    target_eval_points: int = 20,
    target_log_points: int = 100,
) -> tuple[int, int, int]:
    """
    Calculate dynamic eval_steps and logging_steps based on dataset size and training config.

    Args:
        dataset_size: Number of training samples
        per_device_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        num_epochs: Number of training epochs
        num_devices: Number of devices (for distributed training)
        target_eval_points: Target number of evaluation points
        target_log_points: Target number of logging points

    Returns:
        Tuple of (eval_steps, logging_steps, total_steps)
    """
    # Calculate total steps
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps * num_devices
    steps_per_epoch = dataset_size // effective_batch_size
    total_steps = int(steps_per_epoch * num_epochs)

    # Calculate dynamic steps
    eval_steps = max(1, total_steps // target_eval_points)
    logging_steps = max(1, total_steps // target_log_points)

    # Ensure logging is more frequent than eval
    if logging_steps > eval_steps:
        logging_steps = max(1, eval_steps // 2)

    return eval_steps, logging_steps, total_steps


def parse_config_file(
    config_file: str,
) -> tuple[ModelArguments, DataArguments, PruningTrainingArguments]:
    """Parse YAML configuration file and convert to dataclass arguments."""
    hf_logging.set_verbosity(logging.WARNING)
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Extract model arguments
    model_config = config.get("model_args", {})
    model_args = ModelArguments(
        model_name_or_path=model_config.get(
            "model_name_or_path", "hotchpotch/japanese-reranker-xsmall-v2"
        ),
        classifier_dropout=model_config.get("classifier_dropout", 0.1),
        max_length=model_config.get("max_length", 512),
        config_name=model_config.get("config_name", None),
        tokenizer_name=model_config.get("tokenizer_name", None),
        cache_dir=model_config.get("cache_dir", None),
    )

    # Extract data arguments
    data_config = config.get("data_args", {})
    data_args = DataArguments(
        dataset_name=data_config.get("dataset_name", "hotchpotch/wip-msmarco-context-relevance"),
        subset=data_config.get("subset", "msmarco-ja-minimal"),
        teacher_column=data_config.get("teacher_column", None),
        max_train_samples=data_config.get("max_train_samples", None),
        max_eval_samples=data_config.get("max_eval_samples", None),
        validation_split=data_config.get("validation_split", None),
        validation_split_samples=data_config.get("validation_split_samples", None),
        validation_split_name=data_config.get("validation_split_name", "validation"),
        preprocessing_num_workers=data_config.get("preprocessing_num_workers", None),
        datasets=data_config.get("datasets", None),
        items=data_config.get("items", None),
        filter_zero_relevance_max_items=data_config.get("filter_zero_relevance_max_items", None),
        filter_zero_relevance_max_items_reverse=data_config.get(
            "filter_zero_relevance_max_items_reverse", False
        ),
        filter_keep_first_item=data_config.get("filter_keep_first_item", False),
        upsample_factor=data_config.get("upsample_factor", None),
    )

    # Extract training arguments
    training_config = config.get("training_args", {})
    # Ensure evaluation strategy matches save strategy when load_best_model_at_end is True
    load_best_model = training_config.get("load_best_model_at_end", True)

    # Note: eval_steps and logging_steps will be calculated dynamically
    # Remove them from config to avoid confusion
    eval_steps = training_config.get("eval_steps", None)
    logging_steps = training_config.get("logging_steps", None)
    save_steps = training_config.get("save_steps", None)

    training_args = PruningTrainingArguments(
        output_dir=training_config.get(
            "output_dir", None
        ),  # Optional, will be auto-generated if not provided
        overwrite_output_dir=training_config.get("overwrite_output_dir", True),
        do_train=training_config.get("do_train", True),
        do_eval=training_config.get("do_eval", True),
        num_train_epochs=training_config.get("num_train_epochs", 1),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 32),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 2),
        learning_rate=training_config.get("learning_rate", 5e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        # Dynamic steps will be set later
        logging_steps=logging_steps or 100,  # Temporary default
        save_steps=save_steps or 500,  # Temporary default
        eval_steps=eval_steps or 500,  # Temporary default
        eval_strategy="steps" if load_best_model else "no",  # Enable evaluation if needed
        save_total_limit=training_config.get("save_total_limit", 5),
        load_best_model_at_end=load_best_model,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", True),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 8),
        optim=training_config.get("optimizer", training_config.get("optim", "adafactor")),
        report_to=training_config.get("report_to", ["wandb"]),
    )

    # Store original config values for reference
    training_args._original_eval_steps = eval_steps  # type: ignore[attr-defined]
    training_args._original_logging_steps = logging_steps  # type: ignore[attr-defined]
    training_args._original_save_steps = save_steps  # type: ignore[attr-defined]

    eval_datasets_config = training_config.get("eval_datasets")
    if eval_datasets_config is not None:
        training_args.eval_datasets = eval_datasets_config  # type: ignore[attr-defined]

    return model_args, data_args, training_args


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: PruningTrainingArguments,
    run_name: str | None = None,
    timestamp: str | None = None,
) -> str:
    """
    Train OpenProvenceEncoder model.

    Args:
        model_args: Model configuration arguments
        data_args: Data configuration arguments
        training_args: Training configuration arguments
        run_name: Optional run name for wandb/logging
        timestamp: Optional timestamp for output directory

    Returns:
        Path to the saved final model
    """
    hf_logging.set_verbosity(logging.WARNING)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Output directory: {training_args.output_dir}")

    # Set seed
    set_seed(training_args.seed)

    # Initialize WandB if available
    if _wandb_available and training_args.report_to and "wandb" in training_args.report_to:
        # Set WandB project name
        os.environ["WANDB_PROJECT"] = "open-provence"

        wandb.init(  # type: ignore[attr-defined]
            project="open-provence",
            name=run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "mode": "reranking_pruning",
                "dataset": data_args.dataset_name,
                "subset": data_args.subset,
                "num_epochs": training_args.num_train_epochs,
                "batch_size": training_args.per_device_train_batch_size,
                "learning_rate": training_args.learning_rate,
                "optim": training_args.optim,
                "ranking_weight": training_args.ranking_weight,
                "pruning_weight": training_args.pruning_weight,
                "timestamp": timestamp,
            },
        )

    # Log teacher column usage
    if data_args.teacher_column:
        logger.info(f"Using teacher column: {data_args.teacher_column}")
    else:
        logger.info("No teacher column specified, using default teacher_score column")

    # Set TrainingArguments run_name to match WandB
    training_args.run_name = run_name

    # Show warnings for filtering options
    if data_args.filter_zero_relevance_max_items is not None:
        if data_args.filter_zero_relevance_max_items_reverse:
            logger.warning(
                "⚠️  filter_zero_relevance_max_items_reverse is enabled: Keeping items with LOWER relevance scores (reverse sort)"
            )
        if data_args.filter_keep_first_item:
            logger.warning(
                "⚠️  filter_keep_first_item is enabled: Always keeping the first item regardless of relevance"
            )

    # Load dataset with teacher scores
    train_dataset, eval_dataset = prepare_dataset(data_args=data_args, seed=training_args.seed)

    # Calculate dynamic steps based on dataset size
    eval_steps, logging_steps, total_steps = calculate_dynamic_steps(
        dataset_size=len(train_dataset),
        per_device_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        num_epochs=training_args.num_train_epochs,
        num_devices=training_args.n_gpu if training_args.n_gpu > 0 else 1,
    )

    # Check if we're overriding original config values and warn
    original_eval_steps = getattr(training_args, "_original_eval_steps", None)
    original_logging_steps = getattr(training_args, "_original_logging_steps", None)
    original_save_steps = getattr(training_args, "_original_save_steps", None)

    if original_eval_steps and original_eval_steps != eval_steps:
        logger.warning(
            f"Overriding eval_steps from config ({original_eval_steps}) with dynamic value ({eval_steps})"
        )
    if original_logging_steps and original_logging_steps != logging_steps:
        logger.warning(
            f"Overriding logging_steps from config ({original_logging_steps}) with dynamic value ({logging_steps})"
        )

    # Update training arguments with dynamic values
    training_args.eval_steps = eval_steps
    training_args.logging_steps = logging_steps
    training_args.save_steps = (
        original_save_steps or eval_steps
    )  # Use same as eval_steps if not specified

    # Enable evaluation if we have eval dataset
    if eval_dataset is not None:
        training_args.eval_strategy = "steps"
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False
    else:
        # Disable evaluation if no eval dataset
        training_args.eval_strategy = "no"
        training_args.load_best_model_at_end = False

    logger.info("Dynamic step calculation:")
    logger.info(f"  Dataset size: {len(train_dataset):,}")
    logger.info(f"  Total steps: {total_steps:,}")
    logger.info(f"  Eval steps: {eval_steps} (20 evaluations)")
    logger.info(f"  Logging steps: {logging_steps} (100 logs)")
    logger.info(f"  Save steps: {training_args.save_steps}")

    # Initialize OpenProvenceEncoder
    # Determine num_labels
    if model_args.num_labels is not None:
        num_labels = model_args.num_labels
        logger.info(f"Using specified num_labels={num_labels}")
    else:
        # Auto-detect from model or use default
        try:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            existing_num_labels = getattr(config, "num_labels", None)
            if existing_num_labels is not None:
                num_labels = existing_num_labels
                logger.info(f"Auto-detected num_labels={num_labels} from model")
            else:
                num_labels = 2  # Default for 2-class classification
                logger.info(f"Using default num_labels={num_labels} (2-class classification)")
        except Exception:
            num_labels = 2  # Default for 2-class classification
            logger.info(f"Could not detect num_labels, using default={num_labels}")

    logger.info(
        "Initializing OpenProvenceEncoder with %s",
        model_args.model_name_or_path,
    )
    model = OpenProvenceEncoder(
        model_name_or_path=model_args.model_name_or_path,
        num_labels=num_labels,
        max_length=model_args.max_length,
        pruning_config={
            "dropout": model_args.classifier_dropout,
            "sentence_pooling": "mean",
            "use_weighted_pooling": False,
        },
    )

    # Create data collator with teacher score column and correct column names
    # Always use 'teacher_score' since we rename it in prepare_dataset
    teacher_score_column = "teacher_score"
    logger.info(f"Using teacher score column: {teacher_score_column}")
    data_collator = OpenProvenceDataCollator(
        tokenizer=model.tokenizer,
        max_length=model.max_length,
        scores_column=teacher_score_column,  # Use specific teacher score column
        chunks_pos_column="context_spans",  # Use dataset's column name
        relevant_chunks_column="context_spans_relevance",  # Use dataset's column name
        # mini_batch_size=16  # Disabled for debugging
    )

    # Create loss function
    loss_fn = OpenProvenceLoss(
        model=model,
        ranking_weight=training_args.ranking_weight,
        pruning_weight=training_args.pruning_weight,
        is_regression=True,  # Regression task for teacher score distillation
    )

    # Use OpenProvenceTrainer (now based on HuggingFace Trainer)
    logger.info("Using OpenProvenceTrainer")
    trainer = OpenProvenceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        loss_fn=loss_fn,
    )

    # Enable evaluation loss calculation
    training_args.prediction_loss_only = False

    # Train
    logger.info("Starting training...")
    logger.info("Mode: reranking_pruning")
    if data_args.datasets:
        dataset_specs = []
        for spec in data_args.datasets:
            dataset_name = spec.get("dataset_name")
            subset = spec.get("subset")
            if dataset_name and subset:
                dataset_specs.append(f"{dataset_name}:{subset}")
            elif dataset_name:
                dataset_specs.append(str(dataset_name))
            elif subset:
                dataset_specs.append(str(subset))
        if dataset_specs:
            logger.info("Datasets: %s", ", ".join(dataset_specs))
        else:
            logger.info("Datasets: (custom list with missing metadata)")
    else:
        logger.info("Dataset: %s:%s", data_args.dataset_name, data_args.subset)
    logger.info(f"Output: {training_args.output_dir}")

    if training_args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {training_args.resume_from_checkpoint}")

    if training_args.do_train:
        logger.info("Starting trainer.train()...")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        logger.info("trainer.train() completed successfully")
    else:
        logger.info("Skipping training as do_train=False")

    # Save final model
    if training_args.output_dir is None:
        raise ValueError("training_args.output_dir cannot be None")
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    logger.info(f"Saving final model to {final_model_path}")
    model.save_pretrained(final_model_path)

    # Save training arguments
    with open(os.path.join(final_model_path, "training_args.json"), "w") as f:
        args_dict = {
            "model_args": model_args.__dict__,
            "data_args": data_args.__dict__,
            "training_args": {
                k: v for k, v in training_args.__dict__.items() if not k.startswith("_")
            },
        }
        json.dump(args_dict, f, indent=2, default=str)

    # Push to hub if requested
    if training_args.push_to_hub:
        logger.info(f"Pushing model to hub: {training_args.hub_model_id}")
        model.push_to_hub(training_args.hub_model_id)  # type: ignore[misc]

    # Simple model loading test
    logger.info("\n" + "=" * 50)
    logger.info("Verifying model can be loaded...")
    logger.info("=" * 50)

    try:
        # Load the saved model
        # The saved encoder repo includes custom modeling files (auto_map entries),
        # so we need to opt-in to loading that code when validating the artifact.
        loaded_model = OpenProvenceEncoder.from_pretrained(
            final_model_path,
            trust_remote_code=True,
        )
        logger.info("✓ Model loaded successfully")
        logger.info(f"  Mode: {loaded_model.mode}")
        logger.info(f"  Max length: {loaded_model.max_length}")

        # Simple inference test
        test_text = "これはテストです。"
        encoding = loaded_model.tokenizer(test_text, return_tensors="pt")
        logger.info(f"✓ Tokenizer works (test text has {encoding['input_ids'].shape[1]} tokens)")

        # Clean up model to free GPU memory
        del loaded_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✓ GPU memory cleared")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")

    # Release trainer/model references before optional eval-datasets run
    if "trainer" in locals():
        del trainer
    if "model" in locals():
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Optional eval_datasets execution
    eval_settings = getattr(training_args, "eval_datasets", None)
    if eval_settings:
        run_eval_datasets_for_model(final_model_path, eval_settings)

    # Print final summary
    print(f"\n{'=' * 80}")
    print("🎉 Training completed successfully!")
    print(f"📁 Model saved to: {final_model_path}")
    print(f"{'=' * 80}\n")

    logger.info("\n" + "=" * 50)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {final_model_path}")

    return final_model_path
