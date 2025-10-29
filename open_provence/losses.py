"""
OpenProvenceLoss for training with dynamic label generation based on chunk relevance.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class OpenProvenceLoss(nn.Module):
    """
    Loss function designed for OpenProvenceDataCollator outputs.

    This loss function handles:
    - Flattened query-text pairs from the data collator
    - Reconstruction of batch structure for ranking loss
    - Token-level pruning loss

    The DataCollator determines whether to use teacher scores or hard labels
    by setting the appropriate scores_column parameter.
    """

    def __init__(
        self,
        model,
        ranking_loss_fn: nn.Module | None = None,
        pruning_loss_fn: nn.Module | None = None,
        ranking_weight: float = 0.05,
        pruning_weight: float = 1.0,
        is_regression: bool = True,
        use_raw_logits: bool = True,
    ):
        """
        Args:
            model: OpenProvenceEncoder model
            ranking_loss_fn: Loss function for ranking (default: MSELoss for regression, BCEWithLogitsLoss for classification)
            pruning_loss_fn: Loss function for pruning (default: CrossEntropyLoss)
            ranking_weight: Weight for ranking loss (default: 0.05)
            pruning_weight: Weight for pruning loss (default: 1.0)
            is_regression: Whether the ranking task is regression (True) or classification (False)
        """
        super().__init__()

        self.model = model
        self.ranking_weight = ranking_weight
        self.pruning_weight = pruning_weight
        self.is_regression = is_regression
        self.use_raw_logits = use_raw_logits  # Follow Provence: use raw logits

        # Loss functions
        if is_regression:
            self.ranking_loss_fn = ranking_loss_fn or nn.MSELoss()
        else:
            self.ranking_loss_fn = ranking_loss_fn or nn.BCEWithLogitsLoss()

        self.pruning_loss_fn = pruning_loss_fn or nn.CrossEntropyLoss(ignore_index=-100)

        # Initialize last loss components tracker
        self.last_loss_components = {}

    def forward(
        self, sentence_features: list[dict[str, torch.Tensor]], labels: dict[str, torch.Tensor]
    ):
        """
        Compute joint loss for ranking and pruning.

        Args:
            sentence_features: List with single dict containing flattened tokenized inputs
            labels: Dictionary containing:
                - 'ranking_targets': [batch_size, max_docs] matrix of targets (teacher scores or labels)
                - 'pruning_labels': [num_pairs, seq_len] flattened format
                - 'batch_indices': [num_pairs] mapping to original batch
                - 'doc_indices': [num_pairs] mapping to document index
                - 'docs_per_query': List[int] number of docs per query

        Returns:
            Total loss value
        """
        # Get model inputs
        inputs = sentence_features[0]

        # Get model outputs
        outputs = self.model.forward(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        total_loss = 0.0
        losses = {}

        # 1. Ranking loss
        if "ranking_logits" in outputs and "ranking_targets" in labels:
            ranking_loss = self._compute_ranking_loss(outputs, labels)
            if ranking_loss is not None:
                total_loss += self.ranking_weight * ranking_loss
                losses["ranking_loss"] = ranking_loss
        elif "ranking_targets" in labels:
            raise ValueError(
                "Model did not output ranking_logits but ranking_targets were provided."
            )

        # 2. Pruning loss (for both modes)
        if "pruning_logits" in outputs and "pruning_labels" in labels:
            pruning_loss = self._compute_pruning_loss(outputs, labels)
            if pruning_loss is not None:
                total_loss += self.pruning_weight * pruning_loss
                losses["pruning_loss"] = pruning_loss
        else:
            raise ValueError(
                "Pruning logits or labels missing. pruning_logits in outputs: {} pruning_labels in labels: {}".format(
                    "pruning_logits" in outputs, "pruning_labels" in labels
                )
            )

        # Store loss components for logging
        self.last_loss_components = losses

        # Log loss components
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Loss components: {losses}")

        # Return total loss as tensor for HF Trainer compatibility
        return total_loss

    def _compute_ranking_loss(
        self, outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]
    ):
        """Compute ranking loss from flattened outputs and matrix labels."""
        ranking_logits = outputs["ranking_logits"]  # [num_pairs]

        # Ensure ranking_logits is 1D
        if ranking_logits.dim() > 1:
            ranking_logits = ranking_logits.squeeze(-1)

        # Get reconstruction info
        batch_indices = labels["batch_indices"]  # [num_pairs]
        doc_indices = labels["doc_indices"]  # [num_pairs]

        # Target values (either teacher scores or hard labels, determined by DataCollator)
        target_matrix = labels["ranking_targets"]  # [batch_size, max_docs]

        # Extract target values for each pair
        target_values = []
        for batch_idx, doc_idx in zip(batch_indices, doc_indices):
            if doc_idx < target_matrix.shape[1]:
                target_val = target_matrix[batch_idx, doc_idx]
                # Skip padding values
                if target_val != -100:
                    target_values.append(target_val)
                else:
                    target_values.append(0.0)  # Fallback
            else:
                target_values.append(0.0)  # Fallback

        target_tensor = torch.tensor(
            target_values, dtype=torch.float32, device=ranking_logits.device
        )

        # Ensure same length
        min_len = min(len(ranking_logits), len(target_tensor))
        ranking_logits = ranking_logits[:min_len]
        target_tensor = target_tensor[:min_len]

        if len(ranking_logits) == 0:
            return None

        # Compute loss - Provence paper uses raw MSE without sigmoid
        if self.is_regression and self.use_raw_logits:
            # Direct MSE loss as per Provence paper: (s_n - z_{n,0})^2
            # Handle different output dimensions
            if ranking_logits.dim() > 1:
                if ranking_logits.shape[-1] == 2:
                    # 2-class model: use first dimension (Provence style)
                    ranking_scores = ranking_logits[:, 0]
                elif ranking_logits.shape[-1] == 1:
                    # 1-class model: squeeze last dimension
                    ranking_scores = ranking_logits.squeeze(-1)
                else:
                    # Multi-class: use first dimension
                    ranking_scores = ranking_logits[:, 0]
            else:
                ranking_scores = ranking_logits

            loss = self.ranking_loss_fn(ranking_scores, target_tensor)
        else:
            # Standard loss computation
            loss = self.ranking_loss_fn(ranking_logits, target_tensor)

        return loss

    def _compute_pruning_loss(
        self, outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]
    ):
        """Compute token-level pruning loss."""
        pruning_logits = outputs["pruning_logits"]  # [num_pairs, seq_len, 2]
        pruning_labels = labels["pruning_labels"]  # [num_pairs, seq_len]

        # Ensure compatible shapes
        if pruning_logits.shape[:2] != pruning_labels.shape:
            min_batch = min(pruning_logits.shape[0], pruning_labels.shape[0])
            min_seq = min(pruning_logits.shape[1], pruning_labels.shape[1])

            pruning_logits = pruning_logits[:min_batch, :min_seq]  # [min_batch, min_seq, 2]
            pruning_labels = pruning_labels[:min_batch, :min_seq]  # [min_batch, min_seq]

        # Flatten for cross entropy loss
        _, _ = pruning_labels.shape
        pruning_logits_flat = pruning_logits.view(-1, 2)  # [batch*seq, 2]
        pruning_labels_flat = pruning_labels.view(-1)  # [batch*seq]

        # Check for valid labels before computing loss
        valid_mask = pruning_labels_flat != -100
        num_valid = valid_mask.sum().item()

        if num_valid == 0:
            # No valid tokens to compute loss on
            logger.warning(
                f"No valid pruning labels found (all are -100). "
                f"Batch shape: {pruning_labels.shape}, "
                f"Total tokens: {pruning_labels.numel()}. Returning zero loss."
            )
            return torch.tensor(0.0, device=pruning_logits.device, requires_grad=True)

        # Compute loss (ignore_index=-100 will skip padding)
        loss = self.pruning_loss_fn(pruning_logits_flat, pruning_labels_flat)

        # Check for NaN
        if torch.isnan(loss):
            logger.error(
                f"NaN detected in pruning loss! "
                f"Valid tokens: {num_valid}, "
                f"Logits min/max: {pruning_logits_flat.min():.4f}/{pruning_logits_flat.max():.4f}"
            )
            # Return a small positive loss instead of NaN
            return torch.tensor(0.001, device=pruning_logits.device, requires_grad=True)

        return loss
