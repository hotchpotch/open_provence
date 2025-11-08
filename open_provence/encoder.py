"""
OpenProvenceEncoder: A query-dependent text pruning encoder with reranking capabilities.

This module implements the unified reranking+pruning pipeline. Historical
``pruning_only`` support has been removed in favour of the multi-task encoder,
which always exposes ranking logits alongside pruning signals.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PretrainedConfig,
)

from .data_structures import (
    OpenProvenceConfig as DataOpenProvenceConfig,
)
from .data_structures import (
    OpenProvenceOutput,
    RerankingOpenProvenceOutput,
)

# Import the correct config class for saving
from .modeling_open_provence_standalone import OpenProvenceConfig
from .models.open_provence_head import (
    OpenProvenceHead,
    OpenProvenceHeadConfig,
)
from .utils.modeling_export import write_modeling_open_provence

logger = logging.getLogger(__name__)


class OpenProvenceEncoder(nn.Module):
    model_name_or_path: str
    mode: Literal["reranking_pruning"]
    num_labels: int
    max_length: int
    device: str | torch.device
    cache_dir: str | None
    tokenizer: PreTrainedTokenizerBase
    config: PretrainedConfig
    ranking_model: PreTrainedModel
    _original_num_labels: int
    pruning_head: OpenProvenceHead
    text_chunker: Any | None
    use_raw_logits: bool
    pruning_config: DataOpenProvenceConfig
    """
    OpenProvenceEncoder performs query-dependent text pruning with reranking.

    Args:
        model_name_or_path (str): HuggingFace model name or path
        mode (str): Deprecated. Retained for backwards compatibility; must be "reranking_pruning".
        num_labels (int): Number of labels for ranking (default: 2 via regression head)
        max_length (int): Maximum sequence length
        device (str): Device to use (cuda/cpu)
        pruning_config (Dict): Configuration for the pruning head
        cache_dir (str): Cache directory for models
        tokenizer_args (Dict): Additional tokenizer arguments
        model_args (Dict): Additional model arguments
    """

    def __init__(
        self,
        model_name_or_path: str,
        mode: str = "reranking_pruning",
        num_labels: int = 2,  # Number of labels for ranking head
        max_length: int = 512,
        device: str | None = None,
        pruning_config: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        tokenizer_args: dict[str, Any] | None = None,
        model_args: dict[str, Any] | None = None,
    ):
        super().__init__()

        # transformers>=4.57 expects this attribute on all models when loading checkpoints
        self._keys_to_ignore_on_save: list[str] = []

        # Validate mode
        if mode != "reranking_pruning":
            raise ValueError(
                f"Unsupported mode: {mode}. Only 'reranking_pruning' is supported after the pruning-only deprecation."
            )

        # Initialize config
        self.model_name_or_path = model_name_or_path
        self.mode = "reranking_pruning"
        self.num_labels = num_labels
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.device = resolved_device
        self.cache_dir = cache_dir

        # Default configs
        tokenizer_args = tokenizer_args or {}
        model_args = model_args or {}
        pruning_config = pruning_config or {}

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, **tokenizer_args
        )

        # Load the original config to check existing num_labels
        original_config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)

        # Check if we need to adjust num_labels
        original_num_labels = getattr(original_config, "num_labels", None)

        if original_num_labels is not None and original_num_labels != num_labels:
            logger.info(
                f"Model was trained with num_labels={original_num_labels}, but requested num_labels={num_labels}"
            )
            logger.info(
                f"Loading with original num_labels={original_num_labels} and will adapt as needed"
            )

            # Load with original num_labels to avoid size mismatch
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                num_labels=original_num_labels,
                cache_dir=cache_dir,
                **model_args,
            )

            # Load ranking model with original config
            self.ranking_model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, config=self.config, cache_dir=cache_dir, **model_args
            )

            # Store both original and target num_labels
            self._original_num_labels = original_num_labels
            self.num_labels = num_labels

        else:
            # Load normally if num_labels matches
            self.config = AutoConfig.from_pretrained(
                model_name_or_path, num_labels=num_labels, cache_dir=cache_dir, **model_args
            )

            self.ranking_model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, config=self.config, cache_dir=cache_dir, **model_args
            )

            self._original_num_labels = num_labels
            self.num_labels = num_labels

        # Initialize pruning head (common for both modes)
        hidden_size = self.config.hidden_size
        pruning_head_config = OpenProvenceHeadConfig(
            hidden_size=pruning_config.get("hidden_size", hidden_size),
            num_labels=2,  # Binary: keep/prune
            classifier_dropout=pruning_config.get("dropout", 0.1),
            sentence_pooling=pruning_config.get("sentence_pooling", "mean"),
            use_weighted_pooling=pruning_config.get("use_weighted_pooling", False),
        )
        self.pruning_head = OpenProvenceHead(pruning_head_config)

        # Text chunker for sentence segmentation (only needed for raw text API)
        self.text_chunker = None

        # Activation function for ranking scores
        # Note: For training, we use raw logits (no activation)
        # For inference, we may apply sigmoid/softmax
        self.use_raw_logits = True  # Use raw logits for MSE loss

        # Move to device
        self.to(self.device)

        # Default Pruning config
        self.pruning_config = DataOpenProvenceConfig()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sentence_boundaries: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs,  # Accept additional kwargs like token_type_ids
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Forward pass for ranking and/or pruning based on mode.

        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            sentence_boundaries: Token boundaries for each sentence
            return_dict: Whether to return a dictionary

        Returns:
            Dictionary with mode-appropriate outputs
        """
        # Handle OpenProvenceDataCollator output format
        if input_ids is None and "sentence_features" in kwargs:
            # This is called from OpenProvenceDataCollator format
            sentence_features = kwargs.pop("sentence_features")
            if sentence_features and len(sentence_features) > 0:
                input_ids = sentence_features[0].get("input_ids")
                attention_mask = sentence_features[0].get("attention_mask")

        # Handle standard HuggingFace format
        if input_ids is None and "input_ids" in kwargs:
            input_ids = kwargs.pop("input_ids")
        if attention_mask is None and "attention_mask" in kwargs:
            attention_mask = kwargs.pop("attention_mask")

        if input_ids is None:
            logger.error(
                f"Forward called without input_ids. Available kwargs: {list(kwargs.keys())}"
            )
            raise ValueError("input_ids must be provided")
        if attention_mask is None:
            logger.error(
                f"Forward called without attention_mask. Available kwargs: {list(kwargs.keys())}"
            )
            raise ValueError("attention_mask must be provided")
        model_kwargs = dict(kwargs)
        model_kwargs.pop("output_hidden_states", None)
        model_kwargs["output_hidden_states"] = True

        outputs = self.ranking_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **model_kwargs,
        )

        ranking_logits: torch.Tensor = outputs.logits
        hidden_states: torch.Tensor = outputs.hidden_states[-1]

        # Get pruning predictions (common for both modes)
        pruning_outputs = self.pruning_head(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            sentence_boundaries=sentence_boundaries,
        )

        if return_dict:
            return {
                "ranking_logits": ranking_logits,
                "pruning_logits": pruning_outputs.logits,
                "hidden_states": hidden_states,
            }

        return ranking_logits, pruning_outputs.logits

    def predict(
        self,
        sentences: list[tuple[str, str]] | tuple[str, str] | list[list[tuple[str, str]]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        apply_pruning: bool = False,
        pruning_threshold: float = 0.5,
        return_documents: bool = False,
    ) -> list[float] | np.ndarray | torch.Tensor | list[OpenProvenceOutput]:
        """
        Predict ranking scores and optionally apply pruning.

        Args:
            sentences: Query-document pairs
            batch_size: Batch size for prediction
            show_progress_bar: Show progress bar
            convert_to_numpy: Convert to numpy array (only for scores)
            convert_to_tensor: Convert to tensor (only for scores)
            apply_pruning: Whether to apply token-level pruning
            pruning_threshold: Threshold for pruning decisions
            return_documents: Whether to return pruned documents

        Returns:
            If apply_pruning is False: Ranking scores
            If apply_pruning is True: List of RerankingOpenProvenceOutput objects
        """
        if apply_pruning:
            # Use predict_with_pruning for full functionality
            return self.predict_with_pruning(  # type: ignore[return-value]
                sentences=sentences,  # type: ignore[arg-type]
                batch_size=batch_size,
                pruning_threshold=pruning_threshold,
                return_documents=return_documents,
                show_progress_bar=show_progress_bar,
            )

        # Original predict behavior for ranking only
        self.eval()

        single_input = isinstance(sentences[0], str)
        if single_input:
            sentences_list = [sentences]  # type: ignore[list-item]
        else:
            sentences_list = sentences  # type: ignore[assignment]

        all_scores = []

        for start_idx in tqdm(
            range(0, len(sentences_list), batch_size),
            desc="Batches",
            disable=not show_progress_bar,
        ):
            batch = sentences_list[start_idx : start_idx + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.forward(**encoded)
                logits = outputs["ranking_logits"]  # type: ignore[index]

                # Apply activation for inference
                # Handle different num_labels configurations
                if logits.shape[-1] == 2:
                    # 2-class classification: use first class score
                    scores = logits[:, 0]
                elif logits.shape[-1] == 1:
                    # Single output regression
                    scores = logits.squeeze(-1)
                else:
                    # Multi-class: use first class
                    scores = logits[:, 0]

                all_scores.extend(scores.cpu().tolist())

        if single_input:
            if convert_to_tensor:
                return torch.tensor(all_scores)
            elif convert_to_numpy:
                return np.array(all_scores)
            else:
                return all_scores
        else:
            if convert_to_tensor:
                return torch.tensor(all_scores)
            elif convert_to_numpy:
                return np.array(all_scores)
            else:
                return all_scores

    def predict_with_pruning(
        self,
        sentences: list[tuple[str, str]] | tuple[str, str],
        batch_size: int = 32,
        pruning_threshold: float = 0.5,
        return_documents: bool = False,
        show_progress_bar: bool = False,
    ) -> RerankingOpenProvenceOutput | list[RerankingOpenProvenceOutput]:
        """
        Predict with token-level pruning.

        Args:
            sentences: Query-document pairs
            batch_size: Batch size
            pruning_threshold: Threshold for pruning decisions
            return_documents: Whether to return pruned documents
            show_progress_bar: Show progress bar

        Returns:
            RerankingOpenProvenceOutput or list of them based on input cardinality
        """
        self.eval()

        single_input = isinstance(sentences[0], str)
        if single_input:
            sentences_list = [sentences]  # type: ignore[list-item]
        else:
            sentences_list = sentences  # type: ignore[assignment]

        all_outputs = []

        for start_idx in tqdm(
            range(0, len(sentences_list), batch_size),
            desc="Batches",
            disable=not show_progress_bar,
        ):
            batch = sentences_list[start_idx : start_idx + batch_size]

            # Tokenize with offset mapping and auxiliary masks for span resolution
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
                return_token_type_ids=True,
                return_special_tokens_mask=True,
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            offset_mapping = encoded["offset_mapping"]
            token_type_ids = encoded.get("token_type_ids")
            special_tokens_mask = encoded.get("special_tokens_mask")

            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            if special_tokens_mask is not None:
                special_tokens_mask = special_tokens_mask.to(self.device)

            with torch.no_grad():
                forward_inputs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                if token_type_ids is not None:
                    forward_inputs["token_type_ids"] = token_type_ids
                outputs = self.forward(**forward_inputs)

                ranking_logits = outputs["ranking_logits"]  # type: ignore[index]
                # Handle different num_labels configurations
                if ranking_logits.shape[-1] == 2:
                    # 2-class classification: use first class score
                    ranking_scores = ranking_logits[:, 0]
                elif ranking_logits.shape[-1] == 1:
                    # Single output regression
                    ranking_scores = ranking_logits.squeeze(-1)
                else:
                    # Multi-class: use first class
                    ranking_scores = ranking_logits[:, 0]

                # Get token-level pruning predictions
                pruning_logits = outputs["pruning_logits"]  # type: ignore[index]
                pruning_probs = torch.nn.functional.softmax(pruning_logits, dim=-1)
                keep_probs = pruning_probs[:, :, 1]  # Probability of keeping each token

                # Process each example
                for i in range(len(batch)):
                    _, document = batch[i]  # query not used here
                    doc_text = str(document)
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                    offsets = offset_mapping[i]
                    token_type_row = token_type_ids[i] if token_type_ids is not None else None
                    special_mask_row = (
                        special_tokens_mask[i] if special_tokens_mask is not None else None
                    )
                    doc_span = self._resolve_document_span(
                        input_ids[i],
                        offsets,
                        token_type_row,
                        special_mask_row,
                    )

                    if doc_span is None:
                        # Failed to resolve document span, return empty output to avoid crashes
                        output = RerankingOpenProvenceOutput(
                            ranking_scores=np.array([ranking_scores[i].cpu().item()]),
                            pruning_masks=np.array([[]]),
                            sentences=[[]],
                            compression_ratio=0.0,
                            num_pruned_sentences=0,
                        )
                        if return_documents:
                            output.pruned_documents = [""]
                        all_outputs.append(output)
                        continue

                    doc_start, doc_end = doc_span

                    # Get document tokens and their keep probabilities
                    doc_keep_probs = keep_probs[i, doc_start:doc_end]
                    doc_tokens = tokens[doc_start:doc_end]
                    doc_offsets = offsets[doc_start:doc_end]

                    # Apply threshold
                    keep_mask = doc_keep_probs > pruning_threshold

                    # Calculate metrics
                    num_kept = keep_mask.sum().item()
                    num_total = len(doc_tokens)
                    compression_ratio = 1.0 - (num_kept / num_total) if num_total > 0 else 0.0

                    # Reconstruct pruned document
                    pruned_doc = ""
                    if return_documents:
                        kept_ranges = []
                        for keep_flag, offset in zip(keep_mask, doc_offsets):
                            start_val: int
                            end_val: int
                            if isinstance(offset, torch.Tensor):
                                start_val = int(offset[0].item())
                                end_val = int(offset[1].item())
                            else:
                                start_raw, end_raw = offset
                                start_val = int(start_raw)
                                end_val = int(end_raw)
                            if bool(keep_flag) and not (start_val == 0 and end_val == 0):
                                kept_ranges.append((start_val, end_val))

                        # Merge overlapping ranges
                        if kept_ranges:
                            kept_ranges.sort()
                            merged_ranges = [kept_ranges[0]]
                            for start_val, end_val in kept_ranges[1:]:
                                if start_val <= merged_ranges[-1][1]:
                                    merged_ranges[-1] = (
                                        merged_ranges[-1][0],
                                        max(merged_ranges[-1][1], end_val),
                                    )
                                else:
                                    merged_ranges.append((start_val, end_val))

                            # Extract text
                            pruned_doc_parts: list[str] = []
                            for start_val, end_val in merged_ranges:
                                pruned_doc_parts.append(doc_text[start_val:end_val])
                            pruned_doc = " ".join(pruned_doc_parts)

                    output = RerankingOpenProvenceOutput(
                        ranking_scores=np.array([ranking_scores[i].cpu().item()]),
                        pruning_masks=np.array([keep_mask.cpu().numpy()]),
                        sentences=[doc_tokens],  # Store tokens instead of sentences
                        compression_ratio=compression_ratio,
                        num_pruned_sentences=int(
                            num_total - num_kept
                        ),  # Actually num pruned tokens
                    )

                    if return_documents:
                        output.pruned_documents = [pruned_doc]

                    all_outputs.append(output)

        return all_outputs[0] if single_input else all_outputs

    @staticmethod
    def _normalize_offsets(
        offsets: Sequence[Sequence[int]] | torch.Tensor,
    ) -> list[tuple[int, int]]:
        """Convert offset structures to a list of integer tuples."""
        if isinstance(offsets, torch.Tensor):
            return [(int(start), int(end)) for start, end in offsets.tolist()]

        normalized: list[tuple[int, int]] = []
        for entry in offsets:
            if isinstance(entry, torch.Tensor):
                start, end = entry.tolist()
                normalized.append((int(start), int(end)))
            else:
                start, end = entry
                normalized.append((int(start), int(end)))
        return normalized

    @staticmethod
    def _normalize_mask(mask: torch.Tensor | Sequence[int] | None) -> list[int] | None:
        """Convert mask tensors/sequences into a list of integers."""
        if mask is None:
            return None
        if isinstance(mask, torch.Tensor):
            return [int(value) for value in mask.tolist()]
        return [int(value) for value in mask]

    @staticmethod
    def _is_special_token(mask_value: int | torch.Tensor | None, offset: tuple[int, int]) -> bool:
        """Return True when the token is a special token."""
        if mask_value is not None:
            if isinstance(mask_value, torch.Tensor):
                if int(mask_value.item()) == 1:
                    return True
            elif int(mask_value) == 1:
                return True
        start, end = offset
        return start == 0 and end == 0

    @staticmethod
    def _trim_span(
        start: int,
        end: int,
        offsets: list[tuple[int, int]],
        special_tokens_mask: list[int] | None,
    ) -> tuple[int, int] | None:
        """Trim special tokens from the candidate document span."""
        length = len(offsets)
        start = max(0, min(start, length))
        end = max(0, min(end, length))
        if end <= start:
            return None

        while start < end and OpenProvenceEncoder._is_special_token(
            (special_tokens_mask[start] if special_tokens_mask is not None else None),
            offsets[start],
        ):
            start += 1

        while end > start and OpenProvenceEncoder._is_special_token(
            (special_tokens_mask[end - 1] if special_tokens_mask is not None else None),
            offsets[end - 1],
        ):
            end -= 1

        if end <= start:
            return None
        return start, end

    def _resolve_document_span(
        self,
        token_ids: torch.Tensor,
        offsets: Sequence[Sequence[int]] | torch.Tensor,
        token_type_ids: torch.Tensor | None,
        special_tokens_mask: torch.Tensor | None,
    ) -> tuple[int, int] | None:
        """
        Determine the start/end token indices (exclusive) for the document portion.
        """
        normalized_offsets = self._normalize_offsets(offsets)
        normalized_mask = self._normalize_mask(special_tokens_mask)

        # 1) Prefer token_type_ids when available (Sentence A = 0, Sentence B = 1)
        if token_type_ids is not None:
            doc_positions = torch.nonzero(token_type_ids == 1, as_tuple=True)[0]
            if doc_positions.numel() > 0:
                doc_start = int(doc_positions[0].item())
                doc_end = int(doc_positions[-1].item()) + 1
                trimmed = self._trim_span(doc_start, doc_end, normalized_offsets, normalized_mask)
                if trimmed is not None:
                    return trimmed

        # 2) Fallback to separator token positions (eos/sep)
        separator_ids: list[int] = []
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        sep_token_id = getattr(self.tokenizer, "sep_token_id", None)
        if eos_token_id is not None:
            separator_ids.append(int(eos_token_id))
        if sep_token_id is not None:
            separator_ids.append(int(sep_token_id))

        if separator_ids:
            positions: list[int] = []
            for sep_id in separator_ids:
                matched = torch.nonzero(token_ids == sep_id, as_tuple=True)[0]
                if matched.numel() > 0:
                    positions.extend(int(index) for index in matched.tolist())
            positions = sorted(set(positions))
            if len(positions) >= 2:
                first_sep = positions[0]
                last_sep = positions[-1]
                trimmed = self._trim_span(
                    first_sep + 1, last_sep, normalized_offsets, normalized_mask
                )
                if trimmed is not None:
                    return trimmed
            elif positions:
                first_sep = positions[0]
                trimmed = self._trim_span(
                    first_sep + 1, len(normalized_offsets), normalized_offsets, normalized_mask
                )
                if trimmed is not None:
                    return trimmed

        # 3) Final fallback: attempt to use offset heuristic (skip leading specials, keep remaining)
        first_non_special = None
        for idx, offset in enumerate(normalized_offsets):
            if not self._is_special_token(
                (normalized_mask[idx] if normalized_mask is not None else None),
                offset,
            ):
                first_non_special = idx
                break

        if first_non_special is None:
            return None

        last_non_special = first_non_special
        for idx in range(len(normalized_offsets) - 1, first_non_special - 1, -1):
            if not self._is_special_token(
                (normalized_mask[idx] if normalized_mask is not None else None),
                normalized_offsets[idx],
            ):
                last_non_special = idx + 1
                break

        if last_non_special <= first_non_special:
            return None

        return first_non_special, last_non_special

    def predict_context(
        self,
        sentences: list[tuple[str, str]] | tuple[str, str],
        chunk_positions: list[list[list[tuple[int, int]]]] | list[list[tuple[int, int]]],
        batch_size: int = 32,
        token_threshold: float = 0.5,
        chunk_threshold: float = 0.5,
        show_progress_bar: bool = False,
    ) -> OpenProvenceOutput | list[OpenProvenceOutput]:
        """
        Predict with chunk-based evaluation.

        Args:
            sentences: Query-document pairs
            chunk_positions: Chunk positions for each document [[start, end], ...]
            batch_size: Batch size
            token_threshold: Threshold for token-level predictions
            chunk_threshold: Minimum ratio of tokens to consider chunk as relevant
            show_progress_bar: Show progress bar

        Returns:
            OpenProvenceOutput or list of OpenProvenceOutput
        """
        self.eval()

        single_input = isinstance(sentences[0], str)
        if single_input:
            sentences_list = [sentences]  # type: ignore[list-item]
            chunk_positions_list = [chunk_positions]  # type: ignore[list-item]
        else:
            sentences_list = sentences  # type: ignore[assignment]
            chunk_positions_list = chunk_positions  # type: ignore[assignment]

        all_outputs = []

        for start_idx in tqdm(
            range(0, len(sentences_list), batch_size),
            desc="Batches",
            disable=not show_progress_bar,
        ):
            batch = sentences_list[start_idx : start_idx + batch_size]
            batch_chunks = chunk_positions_list[start_idx : start_idx + batch_size]

            # Tokenize with offset mapping and auxiliary masks for span resolution
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
                return_token_type_ids=True,
                return_special_tokens_mask=True,
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            offset_mapping = encoded["offset_mapping"]
            token_type_ids = encoded.get("token_type_ids")
            special_tokens_mask = encoded.get("special_tokens_mask")

            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            if special_tokens_mask is not None:
                special_tokens_mask = special_tokens_mask.to(self.device)

            with torch.no_grad():
                forward_inputs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                if token_type_ids is not None:
                    forward_inputs["token_type_ids"] = token_type_ids
                outputs = self.forward(**forward_inputs)

                ranking_logits = outputs["ranking_logits"]  # type: ignore[index]
                # Handle different num_labels configurations
                if ranking_logits.shape[-1] == 2:
                    # 2-class classification: use first class score
                    ranking_scores = ranking_logits[:, 0]
                elif ranking_logits.shape[-1] == 1:
                    # Single output regression
                    ranking_scores = ranking_logits.squeeze(-1)
                else:
                    # Multi-class: use first class
                    ranking_scores = ranking_logits[:, 0]

                pruning_logits = outputs["pruning_logits"]  # type: ignore[index]
                pruning_probs = torch.nn.functional.softmax(pruning_logits, dim=-1)
                keep_probs = pruning_probs[:, :, 1]  # Probability of keeping each token

                # Process each example in the batch
                for i in range(len(batch)):
                    _ = batch[i]  # query, document not used here
                    chunks = batch_chunks[i]
                    offsets = offset_mapping[i]
                    token_type_row = token_type_ids[i] if token_type_ids is not None else None
                    special_mask_row = (
                        special_tokens_mask[i] if special_tokens_mask is not None else None
                    )
                    doc_span = self._resolve_document_span(
                        input_ids[i],
                        offsets,
                        token_type_row,
                        special_mask_row,
                    )

                    if doc_span is None:
                        # Failed to find document boundaries
                        output = OpenProvenceOutput(
                            ranking_scores=ranking_scores[i].cpu().item(),
                            chunk_predictions=np.array([]),
                            chunk_scores=np.array([]),
                            token_scores=np.array([]),
                            chunk_positions=chunks,  # type: ignore[arg-type]
                            compression_ratio=0.0,
                        )
                        all_outputs.append(output)
                        continue

                    doc_start, doc_end = doc_span

                    # Get document tokens and their keep probabilities
                    doc_keep_probs = keep_probs[i, doc_start:doc_end]
                    doc_offsets = offsets[doc_start:doc_end]

                    # Map chunks to token predictions
                    chunk_scores, chunk_predictions = self._evaluate_chunks(
                        chunks
                        if isinstance(chunks, list)
                        and len(chunks) > 0
                        and isinstance(chunks[0], tuple)
                        else chunks[0],  # type: ignore[arg-type]
                        doc_keep_probs,
                        doc_offsets,
                        token_threshold,
                        chunk_threshold,
                    )

                    # Calculate compression ratio
                    num_kept_chunks = chunk_predictions.sum()
                    num_total_chunks = len(chunks)
                    compression_ratio = (
                        1.0 - (num_kept_chunks / num_total_chunks) if num_total_chunks > 0 else 0.0
                    )

                    # Create output
                    output = OpenProvenceOutput(
                        ranking_scores=ranking_scores[i].cpu().item(),
                        chunk_predictions=chunk_predictions,
                        chunk_scores=chunk_scores,
                        token_scores=doc_keep_probs.cpu().numpy(),
                        chunk_positions=chunks,  # type: ignore[arg-type]
                        compression_ratio=compression_ratio,
                    )
                    all_outputs.append(output)

        return all_outputs[0] if single_input else all_outputs

    def _evaluate_chunks(
        self,
        chunks: list[tuple[int, int]],
        token_probs: torch.Tensor,
        token_offsets: torch.Tensor,
        token_threshold: float,
        chunk_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate chunks based on token predictions.

        Args:
            chunks: List of chunk positions [(start, end), ...]
            token_probs: Token-level keep probabilities
            token_offsets: Token offset mapping
            token_threshold: Threshold for binary token classification
            chunk_threshold: Minimum ratio for chunk classification

        Returns:
            chunk_scores: Average probability for each chunk
            chunk_predictions: Binary predictions for each chunk
        """
        chunk_scores = []
        chunk_predictions = []

        for chunk_start, chunk_end in chunks:
            # Find tokens that overlap with this chunk
            overlapping_tokens = []
            overlapping_probs = []

            for j, (token_start, token_end) in enumerate(token_offsets):
                if token_start != 0 and token_end != 0:  # Skip special tokens
                    # Check if token overlaps with chunk
                    if token_start < chunk_end and token_end > chunk_start:
                        overlapping_tokens.append(j)
                        overlapping_probs.append(token_probs[j].item())

            if overlapping_probs:
                # Calculate chunk-level score
                chunk_score = np.mean(overlapping_probs)

                # Apply chunk-level threshold
                # Count tokens above token_threshold
                tokens_above_threshold = sum(
                    1 for prob in overlapping_probs if prob > token_threshold
                )
                ratio_above_threshold = tokens_above_threshold / len(overlapping_probs)

                # Chunk is predicted as relevant if enough tokens are above threshold
                chunk_pred = 1 if ratio_above_threshold >= chunk_threshold else 0
            else:
                # No overlapping tokens found
                chunk_score = 0.0
                chunk_pred = 0

            chunk_scores.append(chunk_score)
            chunk_predictions.append(chunk_pred)

        return np.array(chunk_scores), np.array(chunk_predictions)

    def prune(
        self,
        query: str,
        document: str,
        threshold: float = 0.5,
        min_sentences: int = 1,
        return_sentences: bool = False,
    ) -> str | dict[str, Any]:
        """
        Prune a single document based on a query using token-level pruning.

        Args:
            query: Query text
            document: Document text to prune
            threshold: Pruning threshold
            min_sentences: Ignored (kept for compatibility)
            return_sentences: Return detailed info

        Returns:
            Pruned document or detailed results
        """
        output = self.predict_with_pruning(
            (query, document), pruning_threshold=threshold, return_documents=True
        )

        if return_sentences:
            return {
                "pruned_document": output.pruned_documents[0],  # type: ignore[attr-defined]
                "sentences": [],  # Not applicable for token-level pruning
                "pruning_masks": [],  # Not applicable
                "ranking_score": float(output.ranking_scores)  # type: ignore[attr-defined]
                if hasattr(output, "ranking_scores") and output.ranking_scores is not None  # type: ignore[attr-defined]
                else None,
                "compression_ratio": output.compression_ratio,  # type: ignore[attr-defined]
                "num_pruned_sentences": 0,  # Not applicable
            }
        else:
            return output.pruned_documents[0]  # type: ignore[attr-defined, no-any-return]

    def prune_texts(
        self,
        queries: list[str],
        texts: list[str],
        threshold: float = 0.5,
        batch_size: int = 32,
        return_tokens: bool = False,
        show_progress_bar: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Simple API for text pruning without manual ranking handling.

        Args:
            queries: List of queries
            texts: List of texts to prune
            threshold: Pruning threshold (0-1)
            batch_size: Batch size for processing
            return_tokens: Whether to return token-level masks
            show_progress_bar: Show progress bar

        Returns:
            List of dicts with:
            - pruned_text: Pruned text
            - kept_ratio: Percentage of text kept
            - pruning_mask: Token-level mask (if return_tokens=True)
        """
        # Convert to pairs format
        sentences = [(q, t) for q, t in zip(queries, texts)]

        # Use predict_with_pruning internally
        outputs = self.predict_with_pruning(
            sentences=sentences,
            batch_size=batch_size,
            pruning_threshold=threshold,
            return_documents=True,
            show_progress_bar=show_progress_bar,
        )

        # Format results
        results = []
        for i, output in enumerate(outputs):  # type: ignore[arg-type]
            result = {
                "pruned_text": output.pruned_documents[0] if output.pruned_documents else texts[i],
                "kept_ratio": 1.0 - output.compression_ratio,
            }

            if return_tokens:
                result["pruning_mask"] = output.pruning_mask

            results.append(result)

        return results

    def _copy_model_files_for_transformers(self, save_directory: Path) -> None:
        """Copy the standalone modeling file for Transformers compatibility."""
        standalone_file = Path(__file__).parent / "modeling_open_provence_standalone.py"
        if standalone_file.exists():
            write_modeling_open_provence(
                standalone_file,
                save_directory / "modeling_open_provence_standalone.py",
            )

    def state_dict(
        self,
        *args: Any,
        destination: Any = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> OrderedDict[str, torch.Tensor]:
        """
        Override state_dict to return the correct format for AutoModel compatibility.
        Remove the ranking_model/encoder prefix to match the expected structure.
        """
        # Both modes now use ranking_model
        remaining_args = args
        actual_destination = cast(OrderedDict[str, torch.Tensor] | None, destination)
        if args:
            actual_destination = cast(OrderedDict[str, torch.Tensor] | None, args[0])
            remaining_args = args[1:]

        state_dict_kwargs: dict[str, Any] = {"prefix": prefix, "keep_vars": keep_vars}
        if actual_destination is not None:
            state_dict_kwargs["destination"] = actual_destination

        base_state_dict = cast(
            OrderedDict[str, torch.Tensor],
            self.ranking_model.state_dict(
                *remaining_args,
                **state_dict_kwargs,
            ),
        )

        # Add pruning head with prefix
        pruning_state_dict = cast(
            OrderedDict[str, torch.Tensor],
            self.pruning_head.state_dict(prefix=prefix, keep_vars=keep_vars),
        )
        for key, value in pruning_state_dict.items():
            base_state_dict[f"{prefix}pruning_head.{key}"] = value

        return base_state_dict

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save the model to a directory."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        saved_num_labels = getattr(self, "_original_num_labels", None)
        if saved_num_labels is None:
            saved_num_labels = getattr(self.ranking_model.config, "num_labels", self.num_labels)
        saved_num_labels = int(saved_num_labels)

        # Merge ranking model and pruning head weights with explicit prefixes
        merged_state: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, value in self.ranking_model.state_dict().items():
            merged_state[f"ranking_model.{key}"] = value

        for key, value in self.pruning_head.state_dict().items():
            merged_state[f"pruning_head.{key}"] = value

        base_config_dict = self.ranking_model.config.to_dict()
        encoder_architecture = getattr(self.ranking_model.config, "model_type", None)

        config = OpenProvenceConfig(
            mode="reranking_pruning",
            base_model_name_or_path=self.model_name_or_path,
            base_model_config=base_config_dict,
            tokenizer_name_or_path=None,
            pruning_config=self.pruning_head.config.to_dict(),
            max_length=self.max_length,
            num_labels=saved_num_labels,
            num_pruning_labels=2,
            encoder_architecture=encoder_architecture,
        )

        config.vocab_size = getattr(self.ranking_model.config, "vocab_size", None)
        hidden_size = getattr(self.ranking_model.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.ranking_model.config, "dim_model"):
            hidden_size = getattr(self.ranking_model.config, "dim_model")
        config.hidden_size = hidden_size

        config.architectures = ["OpenProvenceForSequenceClassification"]
        config.auto_map = {
            "AutoConfig": "modeling_open_provence_standalone.OpenProvenceConfig",
            "AutoModel": "modeling_open_provence_standalone.OpenProvenceForSequenceClassification",
            "AutoModelForSequenceClassification": "modeling_open_provence_standalone.OpenProvenceForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_open_provence_standalone.OpenProvenceForTokenClassification",
        }

        config.save_pretrained(save_directory)
        save_file(merged_state, save_directory / "model.safetensors")

        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

        # Copy modeling file for AutoModel support
        self._copy_model_files_for_transformers(save_directory)

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str | Path, device: str | None = None, **kwargs
    ) -> OpenProvenceEncoder:
        """Load a pretrained OpenProvenceEncoder."""

        # Load config (preserve trust_remote_code if passed)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        )

        # Get OpenProvence metadata from config
        mode = getattr(config, "mode", getattr(config, "open_provence_mode", "reranking_pruning"))
        if mode != "reranking_pruning":
            raise ValueError(
                "Checkpoints saved in 'pruning_only' mode are no longer supported. "
                "Please export a reranking+pruning checkpoint."
            )
        max_length = getattr(
            config, "max_length", getattr(config, "open_provence_max_length", 512)
        )
        pruning_config = getattr(config, "pruning_config", {})

        # First load the full state dict to check for pruning head
        model_path = Path(model_name_or_path)
        state_dict_path = model_path / "pytorch_model.bin"
        if not state_dict_path.exists():
            # Try safetensors
            state_dict_path = model_path / "model.safetensors"
            if state_dict_path.exists():
                full_state_dict = {}
                with safe_open(state_dict_path, framework="pt") as f:
                    for key in f.keys():
                        full_state_dict[key] = f.get_tensor(key)
            else:
                raise ValueError(f"No model file found in {model_path}")
        else:
            full_state_dict = torch.load(state_dict_path, map_location="cpu")

        # Extract pruning head state dict
        pruning_head_state_dict = {}
        base_model_state_dict = {}

        for key, value in full_state_dict.items():
            if key.startswith("ranking_model."):
                base_model_state_dict[key.replace("ranking_model.", "", 1)] = value
            elif key.startswith("pruning_head."):
                pruning_head_state_dict[key.replace("pruning_head.", "", 1)] = value
            else:
                # Legacy checkpoints without explicit prefixes store base weights at root level
                base_model_state_dict[key] = value

        if not pruning_head_state_dict:
            raise ValueError("No pruning head found in the model")

        # Initialize pruning head
        pruning_head = OpenProvenceHead(OpenProvenceHeadConfig(**pruning_config))
        pruning_head.load_state_dict(pruning_head_state_dict)

        # Load base model first without state dict
        # Use the original base model name if available in config
        base_model_name = getattr(config, "base_model_name_or_path", None)

        # Load model based on mode
        if base_model_name is None:
            base_model_name = model_name_or_path

        # Both modes now use AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, trust_remote_code=trust_remote_code, **kwargs
        )

        # Now update with correct state dict (without pruning head)
        model.load_state_dict(base_model_state_dict, strict=True)

        # Create encoder instance
        encoder = cls.__new__(cls)
        nn.Module.__init__(encoder)

        # Use the original base model name if available in config
        if hasattr(config, "base_model_name_or_path") and config.base_model_name_or_path:
            encoder.model_name_or_path = config.base_model_name_or_path
        else:
            encoder.model_name_or_path = str(model_name_or_path)
        encoder.mode = "reranking_pruning"
        encoder.num_labels = config.num_labels
        encoder.max_length = max_length
        encoder.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        encoder.cache_dir = kwargs.get("cache_dir")
        encoder.config = config
        encoder.use_raw_logits = True
        encoder.pruning_config = OpenProvenceConfig()
        encoder.text_chunker = None
        encoder._original_num_labels = config.num_labels
        encoder._keys_to_ignore_on_save = []

        # Set model components - both modes now use ranking_model
        encoder.ranking_model = model

        encoder.pruning_head = pruning_head
        encoder.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Move to device
        encoder.to(encoder.device)

        return encoder

    def export_ranking_model(self, save_directory: str | Path) -> None:
        """Export only the base ranking model for standard Transformers usage.

        This creates a standalone model that can be loaded with:
        AutoModelForSequenceClassification.from_pretrained(save_directory)

        Args:
            save_directory: Directory to save the model
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save the ranking model and tokenizer
        logger.info(f"Exporting ranking model to {save_directory}")
        self.ranking_model.save_pretrained(str(save_directory))
        self.tokenizer.save_pretrained(str(save_directory))

        logger.info("✓ Ranking model exported successfully!")
        logger.info(
            f"  Load with: AutoModelForSequenceClassification.from_pretrained('{save_directory}')"
        )

    def to(self, *args: Any, **kwargs: Any) -> OpenProvenceEncoder:
        """Move model components to the specified device and dtype."""
        super().to(*args, **kwargs)
        self.ranking_model.to(*args, **kwargs)
        self.pruning_head.to(*args, **kwargs)  # type: ignore[misc]
        candidate = args[0] if args else kwargs.get("device")
        if isinstance(candidate, (str, torch.device)):
            self.device = candidate
        return self
