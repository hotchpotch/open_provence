"""
Data collator for pruning training that generates pruning labels based on relevant chunks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class OpenProvenceDataCollator:
    """
    Data collator for pruning training that dynamically generates pruning labels based on chunk relevance.

    This collator uses the relevant_chunks information to determine which tokens to keep:
    - Tokens in relevant chunks (marked in relevant_chunks) get label 1
    - All other tokens get label 0
    - [CLS] and query tokens always get label 0

    This collator works directly with HuggingFace datasets without requiring conversion.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        padding: bool | str = True,
        truncation: bool = True,
        query_column: str = "query",
        texts_column: str = "texts",
        labels_column: str = "labels",
        scores_column: str | None = None,
        chunks_pos_column: str = "chunks_pos",
        relevant_chunks_column: str = "relevant_chunks",
        dataset_name_column: str | None = "dataset_name",
        id_column: str | None = "id",
        mini_batch_size: int | None = None,
    ):  # Deprecated, kept for compatibility
        """
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
            query_column: Name of the query column in the dataset
            texts_column: Name of the texts column in the dataset
            labels_column: Name of the labels column in the dataset
            scores_column: Name of the teacher scores column containing continuous reranker scores
                          for knowledge distillation. If None, falls back to binary labels_column.
            chunks_pos_column: Name of the chunks positions column
            relevant_chunks_column: Name of the relevant chunks column
            dataset_name_column: Name of the dataset name column (optional)
            id_column: Name of the ID column (optional)
            mini_batch_size: Size for processing mini-batches (if None, process all at once)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.mini_batch_size = None  # Disabled for performance - mini batching is slower

        # Column names
        self.query_column = query_column
        self.texts_column = texts_column
        self.labels_column = labels_column
        self.scores_column = scores_column
        self.chunks_pos_column = chunks_pos_column
        self.relevant_chunks_column = relevant_chunks_column
        self.dataset_name_column = dataset_name_column
        self.id_column = id_column

        # Required columns for training
        self._required_columns = [
            self.query_column,
            self.texts_column,
            self.labels_column,
            self.chunks_pos_column,
            self.relevant_chunks_column,
        ]

        # Column validation will be done when first batch is processed
        self._validated = False

        # Cache tokenizer properties for performance
        self._has_sep_token = "[SEP]" in self.tokenizer.get_vocab()
        self._eos_token_id = self.tokenizer.eos_token_id or 2
        self._sep_token_id = (
            self.tokenizer.sep_token_id if hasattr(self.tokenizer, "sep_token_id") else None
        )

    def _validate_columns(self, dataset_or_batch):
        """Validate that required columns exist in the dataset."""
        if self._validated:
            return

        # Get column names from either a Dataset or a batch dict
        if isinstance(dataset_or_batch, Dataset):
            columns = dataset_or_batch.column_names
        elif isinstance(dataset_or_batch, dict):
            columns = dataset_or_batch.keys()
        elif isinstance(dataset_or_batch, list) and len(dataset_or_batch) > 0:
            columns = dataset_or_batch[0].keys()
        else:
            return  # Can't validate yet

        missing_columns = []
        for col in self._required_columns:
            if col not in columns:
                missing_columns.append(col)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {list(columns)}\n"
                f"Required columns: {self._required_columns}"
            )

        # Scores column is optional
        if self.scores_column and self.scores_column not in columns:
            logger.warning(
                f"Teacher scores column '{self.scores_column}' not found. "
                f"Using '{self.labels_column}' for ranking targets."
            )
            self.scores_column = None

        self._validated = True

    def __call__(self, features: list[dict[str, Any]] | Dataset) -> dict[str, Any]:
        """
        Collate batch of examples for training.

        Args:
            features: Either a list of dicts or a Dataset batch containing required columns

        Returns:
            Batch dictionary compatible with OpenProvenceLoss
        """
        # Validate columns on first call
        self._validate_columns(features)

        # Handle both list of dicts and Dataset batch
        if isinstance(features, Dataset):
            # Convert Dataset batch to list of dicts
            batch_size = len(features)
            features_list = []
            for i in range(batch_size):
                features_list.append({col: features[col][i] for col in features.column_names})  # type: ignore[attr-defined]
            features = features_list

        batch_size = len(features)

        # Create all query-text pairs using column names
        pairs = []
        batch_indices = []
        doc_indices = []
        pair_ranking_labels = []
        pair_ranking_targets = []  # Can be labels or teacher scores
        pair_chunks_pos = []
        pair_relevant_chunks = []

        for batch_idx, feature in enumerate(features):
            query = feature[self.query_column]
            texts = feature[self.texts_column]
            chunks_pos = feature[self.chunks_pos_column]
            relevant_chunks_raw = feature[self.relevant_chunks_column]

            # Convert binary labels to indices if necessary
            # If relevant_chunks contains binary labels [1, 0, 1], convert to indices [0, 2]
            relevant_chunks = []
            for text_idx, chunk_labels in enumerate(relevant_chunks_raw):
                if isinstance(chunk_labels, list) and len(chunk_labels) > 0:
                    # Check if it's binary labels (all values are 0 or 1)
                    # AND the length matches the number of chunks (indicating it's a binary mask)
                    if len(chunk_labels) == len(chunks_pos[text_idx]) and all(
                        label in [0, 1] for label in chunk_labels
                    ):
                        # Convert binary labels to indices
                        indices = [idx for idx, label in enumerate(chunk_labels) if label == 1]
                        relevant_chunks.append(indices)
                    else:
                        # Already indices (or doesn't match expected format)
                        relevant_chunks.append(chunk_labels)
                else:
                    relevant_chunks.append(chunk_labels)

            # Get ranking labels/targets for distillation (always reranking mode)
            labels = feature[self.labels_column]
            if self.scores_column and self.scores_column in feature:
                ranking_targets = feature[self.scores_column]
            else:
                ranking_targets = labels

            for doc_idx, (text, label, target, chunk_pos, rel_chunks) in enumerate(
                zip(texts, labels, ranking_targets, chunks_pos, relevant_chunks)
            ):
                pairs.append([query, text])
                batch_indices.append(batch_idx)
                doc_indices.append(doc_idx)
                pair_ranking_labels.append(label)
                pair_ranking_targets.append(target)
                pair_chunks_pos.append(chunk_pos)
                pair_relevant_chunks.append(rel_chunks)

        # Tokenize all pairs at once (no mini-batching for performance)
        encoded_inputs = self.tokenizer(
            pairs,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        # Extract offset mappings (keep on same device as inputs for speed)
        offset_mappings = encoded_inputs.pop("offset_mapping")

        # Generate chunk-based pruning labels
        # Try to use the improved method if possible
        try:
            logger.debug("Attempting to use v2 label generation method")
            # Extract chunk texts from positions
            pair_chunks_text = []
            for i, (pair, chunk_positions) in enumerate(zip(pairs, pair_chunks_pos)):
                query, document = pair
                chunk_texts = []
                for start, end in chunk_positions:
                    # Extract chunk text from document using character positions
                    chunk_text = document[start:end]
                    chunk_texts.append(chunk_text)
                pair_chunks_text.append(chunk_texts)

            # Use improved method
            logger.debug(f"Using v2 with chunks_text: {pair_chunks_text}")
            logger.debug(f"Relevant chunks: {pair_relevant_chunks}")
            pruning_labels = self._generate_chunk_based_labels_v2(
                encoded_inputs, pairs, pair_chunks_text, pair_relevant_chunks
            )
        except Exception as e:
            # Fallback to original method if there's any issue
            logger.warning(f"Falling back to v1 label generation: {e}")
            pruning_labels = self._generate_chunk_based_labels(
                encoded_inputs, offset_mappings, pairs, pair_chunks_pos, pair_relevant_chunks
            )

        # Prepare labels in the format expected by OpenProvenceLoss
        max_docs = max(len(feature[self.texts_column]) for feature in features)

        # Ranking targets matrix for knowledge distillation
        # Uses continuous teacher scores when available for regression loss
        # Falls back to binary labels for classification if no teacher scores
        ranking_targets_matrix = torch.full(
            (batch_size, max_docs),
            fill_value=-100,  # Padding value
            dtype=torch.float32,
        )

        # Fill matrix
        for i, feature in enumerate(features):
            texts = feature[self.texts_column]
            num_docs = len(texts)

            # Get targets for distillation
            # Prefer teacher scores (float values from teacher reranker)
            # over binary labels for better knowledge transfer
            if self.scores_column and self.scores_column in feature:
                targets = feature[self.scores_column]  # Continuous teacher scores
            else:
                targets = feature[self.labels_column]  # Binary labels as fallback

            # Fill ranking targets
            ranking_targets_matrix[i, :num_docs] = torch.tensor(targets, dtype=torch.float32)

        # Prepare output format compatible with OpenProvenceLoss
        labels = {
            "ranking_targets": ranking_targets_matrix,
            "pruning_labels": pruning_labels,
            "batch_indices": torch.tensor(batch_indices),
            "doc_indices": torch.tensor(doc_indices),
            "docs_per_query": [len(feature[self.texts_column]) for feature in features],
        }

        # Return in the format expected by OpenProvenceLoss
        return {"sentence_features": [encoded_inputs], "labels": labels}

    def _generate_chunk_based_labels_v2(
        self,
        encoded_inputs: dict[str, torch.Tensor],
        pairs: list[list[str]],
        chunks_text: list[list[str]],
        relevant_chunks: list[list[int]],
    ) -> torch.Tensor:
        """
        Generate token-level pruning labels using improved span position calculation.

        This version uses progressive encoding to accurately determine token positions
        for each span, handling tokenizer-specific behaviors correctly.

        Args:
            encoded_inputs: Tokenized inputs
            pairs: Original text pairs [query, document]
            chunks_text: Text content of each chunk (list of spans for each pair)
            relevant_chunks: Indices of relevant chunks

        Returns:
            Token-level pruning labels tensor
        """
        batch_size = encoded_inputs["input_ids"].shape[0]
        seq_length = encoded_inputs["input_ids"].shape[1]

        # Initialize with -100 (ignore in loss)
        pruning_labels = torch.full((batch_size, seq_length), -100, dtype=torch.long)

        for idx in range(len(pairs)):
            query, _ = pairs[idx]  # document not used in simple strategy
            spans = chunks_text[idx]
            rel_chunk_indices = relevant_chunks[idx]

            # Compute token positions for each span
            span_positions = compute_span_token_positions(self.tokenizer, query, spans)

            # Validate the positions (optional, can be disabled in production)
            if logger.isEnabledFor(logging.DEBUG):
                is_valid = validate_span_tokenization(self.tokenizer, query, spans, span_positions)
                if not is_valid:
                    logger.debug(f"Span tokenization validation failed for pair {idx}")

            # Set labels based on relevant chunks
            for chunk_idx in rel_chunk_indices:
                if chunk_idx < len(span_positions):
                    start_pos, end_pos = span_positions[chunk_idx]
                    # Ensure we don't exceed sequence length
                    start_pos = min(start_pos, seq_length)
                    end_pos = min(end_pos, seq_length)
                    # Set tokens in relevant chunks to 1 (keep)
                    pruning_labels[idx, start_pos:end_pos] = 1

            # Set tokens in non-relevant chunks to 0 (prune)
            for chunk_idx in range(len(span_positions)):
                if chunk_idx not in rel_chunk_indices:
                    start_pos, end_pos = span_positions[chunk_idx]
                    # Ensure we don't exceed sequence length
                    start_pos = min(start_pos, seq_length)
                    end_pos = min(end_pos, seq_length)
                    pruning_labels[idx, start_pos:end_pos] = 0

        return pruning_labels

    def _generate_chunk_based_labels(
        self,
        encoded_inputs: dict[str, torch.Tensor],
        offset_mappings: torch.Tensor,
        pairs: list[list[str]],
        chunks_pos: list[list[list[int]]],
        relevant_chunks: list[list[int]],
    ) -> torch.Tensor:
        """
        Generate token-level pruning labels based on chunk relevance.

        Rules:
        - Special tokens: -100 (ignored in loss)
        - Query tokens: -100 (ignored in loss - we only prune document content)
        - Tokens in relevant chunks: 1 (keep)
        - Tokens in non-relevant chunks: 0 (prune)

        Args:
            encoded_inputs: Tokenized inputs
            offset_mappings: Character offset mappings for each token
            pairs: Original text pairs [query, document]
            chunks_pos: Character boundaries for each chunk [[start, end], ...]
            relevant_chunks: Indices of relevant chunks

        Returns:
            Token-level pruning labels tensor
        """
        batch_size = encoded_inputs["input_ids"].shape[0]
        seq_length = encoded_inputs["input_ids"].shape[1]

        # Initialize with zeros (following original working implementation)
        pruning_labels = torch.zeros((batch_size, seq_length), dtype=torch.long)

        # offset_mappings is a tensor with shape [num_pairs, seq_length, 2]
        # We need to iterate over the first dimension properly
        for idx in range(len(pairs)):
            # pair = pairs[idx]  # Not used in this method
            chunk_positions = chunks_pos[idx]
            rel_chunk_indices = relevant_chunks[idx]
            offsets = offset_mappings[idx]
            # query, document = pair  # Not used in this method

            # Find where the document starts in the tokenized sequence
            # Different tokenizers use different formats:
            # - XLMRoberta: <s> query </s> </s> document </s>
            # - ModernBERT: [CLS] query [SEP] document [SEP]
            token_ids = encoded_inputs["input_ids"][idx]

            # Use cached tokenizer properties for performance
            if self._has_sep_token:
                # ModernBERT style with [SEP] tokens
                sep_positions = (token_ids == self._sep_token_id).nonzero(as_tuple=True)[0]

                if len(sep_positions) >= 2:  # Need at least 2 [SEP] tokens
                    # Document starts after first [SEP]
                    doc_start_token = int(sep_positions[0].item()) + 1
                    doc_end_token = int(sep_positions[1].item())
                else:
                    # Skip if we can't find proper boundaries
                    continue
            else:
                # XLMRoberta style with </s> tokens
                sep_positions = (token_ids == self._eos_token_id).nonzero(as_tuple=True)[0]

                if len(sep_positions) >= 2:  # Need at least 2 </s> tokens
                    # Document starts after first </s> + <s>
                    doc_start_token = int(sep_positions[0].item()) + 2  # Skip </s> and <s>
                    doc_end_token = int(
                        sep_positions[1].item()
                    )  # Use the second </s> as document end
                else:
                    # Skip if we can't find proper boundaries
                    continue

            # Find document offset for adjustment (make offsets relative to document)
            doc_offset = 0
            for i in range(
                int(doc_start_token), min(int(doc_start_token) + 5, int(doc_end_token))
            ):
                if offsets[i][0] != 0 or offsets[i][1] != 0:
                    doc_offset = (
                        offsets[i][0].item() if torch.is_tensor(offsets[i][0]) else offsets[i][0]
                    )
                    break

            # Mask query and special tokens with -100 (exclude from loss)
            for token_idx in range(0, int(doc_start_token)):
                pruning_labels[idx, token_idx] = -100

            # For each token in the document range (original simple logic)
            for token_idx in range(int(doc_start_token), int(doc_end_token)):
                # Get character position of this token
                token_start, token_end = offsets[token_idx]

                # Skip special tokens
                if token_start == 0 and token_end == 0:
                    continue

                # Adjust to document-relative offsets (as in original version)
                token_start_rel = token_start - doc_offset
                token_end_rel = token_end - doc_offset

                # Check if this token belongs to any relevant chunk
                for chunk_idx in rel_chunk_indices:
                    if chunk_idx < len(chunk_positions):
                        chunk_start, chunk_end = chunk_positions[chunk_idx]

                        # Check if token overlaps with this chunk (using document-relative offsets)
                        if token_start_rel < chunk_end and token_end_rel > chunk_start:
                            pruning_labels[idx, token_idx] = 1  # Keep
                            break

            # Mask tokens after document end with -100
            for token_idx in range(int(doc_end_token), seq_length):
                pruning_labels[idx, token_idx] = -100

        return pruning_labels


def compute_span_token_positions(tokenizer, query: str, spans: list[str]) -> list[tuple[int, int]]:
    """
    Compute the token positions for each span by progressively encoding the text.

    This method is more accurate than using character offsets because it handles
    tokenizer-specific behaviors (like subword tokenization) correctly.

    Args:
        tokenizer: HuggingFace tokenizer
        query: Query text
        spans: List of document spans/chunks

    Returns:
        List of (start_token_idx, end_token_idx) tuples for each span within the full sequence

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        >>> query = "What is machine learning?"
        >>> spans = ["Machine learning is AI.", "It uses algorithms."]
        >>> positions = compute_span_token_positions(tokenizer, query, spans)
        >>> # positions = [(8, 15), (15, 21)]  # Token positions for each span
    """
    if not spans:
        return []

    # Progressively build the text and encode each version
    span_positions = []

    # Build progressive texts: query + span1, query + span1 + span2, etc.
    progressive_texts = []
    accumulated_text = ""
    for i, span in enumerate(spans):
        if i > 0:
            # Add a space between spans to prevent tokenization issues
            # This ensures "first span" + "second span" doesn't become "first spanssecond span"
            accumulated_text += " "
        accumulated_text += span
        # Create the pair as the tokenizer would see it
        progressive_texts.append([query, accumulated_text])

    # Batch encode all progressive texts
    encodings = tokenizer(
        progressive_texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_offsets_mapping=False,
        return_attention_mask=False,
    )

    # Find where document starts in the first encoding
    # This handles different tokenizer formats (BERT, RoBERTa, etc.)
    encodings["input_ids"][0]

    # Encode just the query to find where it ends
    query_only = tokenizer(
        [query],
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_offsets_mapping=False,
        return_attention_mask=False,
    )
    query_length = len(query_only["input_ids"][0])

    # Find the document start position (after query + separator tokens)
    # The exact position depends on the tokenizer format
    # Try encoding a query-document pair to find the pattern
    test_pair = tokenizer(
        [[query, "test"]],
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_offsets_mapping=False,
        return_attention_mask=False,
    )
    test_tokens = test_pair["input_ids"][0]

    # Find where "test" tokens start
    test_only = tokenizer(
        ["test"],
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_offsets_mapping=False,
        return_attention_mask=False,
    )
    test_token_ids = test_only["input_ids"][0]

    # Search for the test tokens in the full sequence
    doc_start_offset = None
    for i in range(query_length, len(test_tokens) - len(test_token_ids) + 1):
        if test_tokens[i : i + len(test_token_ids)] == test_token_ids:
            doc_start_offset = i
            break

    if doc_start_offset is None:
        # Fallback: assume document starts after query + 1 separator
        doc_start_offset = query_length

    # Now compute span positions
    prev_doc_length = 0
    for i, _ in enumerate(encodings["input_ids"]):
        # Get the current document text (with spaces between spans)
        current_doc = ""
        for j in range(i + 1):
            if j > 0:
                current_doc += " "
            current_doc += spans[j]

        # Encode just the document to get its token length
        doc_only = tokenizer(
            [current_doc],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_offsets_mapping=False,
            return_attention_mask=False,
        )
        current_doc_length = len(doc_only["input_ids"][0])

        # The span starts where the previous document ended
        span_start = doc_start_offset + prev_doc_length
        span_end = doc_start_offset + current_doc_length

        span_positions.append((span_start, span_end))
        prev_doc_length = current_doc_length

    return span_positions


def validate_span_tokenization(
    tokenizer, query: str, spans: list[str], span_positions: list[tuple[int, int]]
) -> bool:
    """
    Validate that the computed span positions correctly map to the original spans.

    Args:
        tokenizer: HuggingFace tokenizer
        query: Query text
        spans: List of document spans
        span_positions: Computed token positions from compute_span_token_positions

    Returns:
        True if the positions correctly decode to the original spans
    """
    # Encode the full text (with spaces between spans)
    doc_text = ""
    for i, span in enumerate(spans):
        if i > 0:
            doc_text += " "
        doc_text += span
    full_text = [query, doc_text]
    encoding = tokenizer(
        [full_text],
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_offsets_mapping=False,
        return_attention_mask=False,
    )

    tokens = encoding["input_ids"][0]

    # Validate each span
    for i, (span_text, (start_pos, end_pos)) in enumerate(zip(spans, span_positions)):
        # Decode the tokens for this span
        span_tokens = tokens[start_pos:end_pos]
        decoded_text = tokenizer.decode(span_tokens, skip_special_tokens=True)

        # Normalize both texts (remove extra spaces, etc.)
        normalized_original = " ".join(span_text.split())
        normalized_decoded = " ".join(decoded_text.split())

        # For more flexible validation, also check lowercase versions
        # This handles tokenizers like BERT that lowercase inputs
        if normalized_original != normalized_decoded:
            # Try case-insensitive comparison
            if normalized_original.lower() == normalized_decoded.lower():
                continue  # Accept case differences

            # For even more flexibility, check if the decoded text contains
            # the essential parts of the original (handling subword tokenization)
            original_words = normalized_original.lower().split()
            normalized_decoded.lower().split()

            # Check if all original words appear in decoded (allowing for subword splits)
            decoded_text_lower = normalized_decoded.lower().replace(" ", "")
            all_found = True
            for word in original_words:
                if word.lower() not in decoded_text_lower:
                    all_found = False
                    break

            if not all_found:
                logger.warning(
                    f"Span {i} mismatch:\n"
                    f"  Original: '{normalized_original}'\n"
                    f"  Decoded:  '{normalized_decoded}'\n"
                    f"  Positions: {start_pos}-{end_pos}"
                )
                return False

    return True
