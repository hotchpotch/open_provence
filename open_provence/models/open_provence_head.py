"""
Sentence Compression Head module for query-dependent text pruning.
Compatible with AutoModelForTokenClassification.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel  # type: ignore[attr-defined]
from transformers.modeling_outputs import TokenClassifierOutput

logger = logging.getLogger(__name__)


class OpenProvenceHeadConfig(PretrainedConfig):
    """Configuration class for OpenProvenceHead."""

    model_type = "open_provence_head"

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        sentence_pooling: str = "mean",
        use_weighted_pooling: bool = False,
        **kwargs,
    ):
        """
        Args:
            hidden_size: Hidden size of the input features
            num_labels: Number of labels (2 for binary: keep/prune)
            classifier_dropout: Dropout probability
            sentence_pooling: Pooling strategy for sentence-level predictions
                             ("mean", "max", "first", "last")
            use_weighted_pooling: Whether to use attention weights for pooling
        """
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        self.sentence_pooling = sentence_pooling
        self.use_weighted_pooling = use_weighted_pooling


class OpenProvenceHead(PreTrainedModel):
    """
    Pruning head for query-dependent text pruning.
    Can be used standalone or integrated with reranking models.
    Compatible with AutoModelForTokenClassification.
    """

    config_class = OpenProvenceHeadConfig

    def __init__(self, config: OpenProvenceHeadConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.sentence_pooling = config.sentence_pooling
        self.use_weighted_pooling = config.use_weighted_pooling

        # Dropout layer
        self.dropout = nn.Dropout(config.classifier_dropout)

        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Optional: Weighted pooling layer
        if self.use_weighted_pooling:
            self.pooling_weights = nn.Linear(config.hidden_size, 1)

        # Initialize weights
        self.init_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sentence_boundaries: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> TokenClassifierOutput | tuple[torch.Tensor, ...]:
        """
        Forward pass for pruning classification.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            sentence_boundaries: [batch_size, max_sentences, 2] - start/end token indices
            labels: [batch_size, seq_len] or [batch_size, max_sentences]
            return_dict: Whether to return a TokenClassifierOutput object

        Returns:
            TokenClassifierOutput or tuple of tensors
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Apply dropout
        hidden_states = self.dropout(hidden_states)

        # Token-level classification
        logits = self.classifier(hidden_states)  # [batch_size, seq_len, num_labels]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            if sentence_boundaries is not None:
                # Sentence-level loss
                loss = self._compute_sentence_loss(
                    logits, labels, sentence_boundaries, attention_mask
                )
            else:
                # Token-level loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = labels.view(-1)

                    if active_loss.sum() > 0:
                        active_logits = active_logits[active_loss]
                        active_labels = active_labels[active_loss]
                        loss = loss_fct(active_logits, active_labels)
                    else:
                        loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + kwargs.get("hidden_states", ())
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,  # type: ignore[arg-type]
            logits=logits,
            hidden_states=kwargs.get("hidden_states"),
            attentions=kwargs.get("attentions"),
        )

    def _compute_sentence_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        boundaries: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute sentence-level loss by pooling token predictions within each sentence.

        Args:
            logits: [batch_size, seq_len, num_labels]
            labels: [batch_size, max_sentences] - sentence-level labels
            boundaries: [batch_size, max_sentences, 2] - token boundaries
            attention_mask: [batch_size, seq_len]
        """
        batch_size, _, num_labels = logits.shape
        max_sentences = boundaries.shape[1]
        device = logits.device

        # Initialize sentence logits
        sentence_logits = torch.zeros(
            batch_size, max_sentences, num_labels, device=device, dtype=logits.dtype
        )

        # Pool logits for each sentence
        for b in range(batch_size):
            for s in range(max_sentences):
                start, end = boundaries[b, s]

                # Skip padding (-1 boundaries)
                if start == -1 or end == -1:
                    continue

                # Get token logits for this sentence
                sentence_tokens = logits[b, start:end]  # [num_tokens, num_labels]

                if sentence_tokens.shape[0] == 0:
                    continue

                # Apply pooling
                if self.sentence_pooling == "mean":
                    # Note: weighted pooling not available in this context since we only have logits
                    pooled = sentence_tokens.mean(dim=0)
                elif self.sentence_pooling == "max":
                    pooled = sentence_tokens.max(dim=0)[0]
                elif self.sentence_pooling == "first":
                    pooled = sentence_tokens[0]
                elif self.sentence_pooling == "last":
                    pooled = sentence_tokens[-1]
                else:
                    # Default to mean
                    pooled = sentence_tokens.mean(dim=0)

                sentence_logits[b, s] = pooled

        # Flatten for loss computation
        valid_mask = boundaries[:, :, 0] != -1  # [batch_size, max_sentences]
        valid_logits = sentence_logits[valid_mask]  # [num_valid, num_labels]
        valid_labels = labels[valid_mask]  # [num_valid]

        # Compute loss
        if valid_logits.shape[0] > 0:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(valid_logits, valid_labels)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return loss

    def predict_sentences(
        self,
        hidden_states: torch.Tensor,
        sentence_boundaries: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get sentence-level predictions by pooling token predictions.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            sentence_boundaries: [batch_size, max_sentences, 2]
            attention_mask: [batch_size, seq_len]

        Returns:
            Sentence probabilities [batch_size, max_sentences, num_labels]
        """
        # Get token-level logits
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        batch_size = logits.shape[0]
        max_sentences = sentence_boundaries.shape[1]
        device = logits.device

        # Initialize sentence predictions
        sentence_probs = torch.zeros(
            batch_size, max_sentences, self.num_labels, device=device, dtype=logits.dtype
        )

        # Pool predictions for each sentence
        for b in range(batch_size):
            for s in range(max_sentences):
                start, end = sentence_boundaries[b, s]

                if start == -1 or end == -1:
                    # Invalid boundary - set to neutral prediction
                    sentence_probs[b, s] = torch.tensor([0.5, 0.5], device=device)
                    continue

                # Get token probabilities for this sentence
                sentence_logits = logits[b, start:end]

                if sentence_logits.shape[0] == 0:
                    sentence_probs[b, s] = torch.tensor([0.5, 0.5], device=device)
                    continue

                # Convert to probabilities
                sentence_token_probs = torch.softmax(sentence_logits, dim=-1)

                # Pool probabilities
                if self.sentence_pooling == "mean":
                    pooled_probs = sentence_token_probs.mean(dim=0)
                elif self.sentence_pooling == "max":
                    pooled_probs = sentence_token_probs.max(dim=0)[0]
                elif self.sentence_pooling == "first":
                    pooled_probs = sentence_token_probs[0]
                elif self.sentence_pooling == "last":
                    pooled_probs = sentence_token_probs[-1]
                else:
                    pooled_probs = sentence_token_probs.mean(dim=0)

                sentence_probs[b, s] = pooled_probs

        return sentence_probs

    def save_pretrained(
        self,
        save_directory: str | os.PathLike[str],
        is_main_process: bool = True,
        state_dict: dict[str, torch.Tensor] | None = None,
        save_function: Callable[..., Any] | None = None,
        push_to_hub: bool = False,
        max_shard_size: int | str = "5GB",
        safe_serialization: bool = True,
        variant: str | None = None,
        token: str | bool | None = None,
        save_peft_format: bool = True,
        **kwargs: Any,
    ) -> None:
        """Delegate to the base implementation while defaulting to `torch.save` when needed."""
        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function or torch.save,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load the model from a directory or HuggingFace Hub."""
        # Load config
        config = OpenProvenceHeadConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Initialize model
        if not isinstance(config, OpenProvenceHeadConfig):
            raise TypeError(f"config must be OpenProvenceHeadConfig, got {type(config)}")
        model = cls(config)

        # Load state dict
        if os.path.isdir(pretrained_model_name_or_path):
            state_dict_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            if os.path.exists(state_dict_path):
                state_dict = torch.load(state_dict_path, map_location="cpu")
                model.load_state_dict(state_dict)

        return model


# Register with AutoModel (this would typically be done in transformers library)
# For now, we'll just document how it would be used:
"""
Usage with AutoModelForTokenClassification:

from transformers import AutoConfig, AutoModelForTokenClassification

# Register the config and model
AutoConfig.register("open_provence_head", OpenProvenceHeadConfig)
AutoModelForTokenClassification.register(OpenProvenceHeadConfig, OpenProvenceHead)

# Then use it
model = AutoModelForTokenClassification.from_pretrained(
    "your-username/provence-pruner-deberta-v3",
    trust_remote_code=True
)
"""
