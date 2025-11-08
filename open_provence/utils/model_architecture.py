"""
Model architecture detection and state_dict conversion utilities.

This module provides utilities for detecting model architectures and converting
state_dicts between different model formats, particularly for handling ModernBert
and BERT-like models.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelArchitectureUtils:
    """Utilities for model architecture detection and state_dict conversion."""

    # Known architecture patterns
    ARCHITECTURE_PATTERNS = {
        "modernbert": {
            "identifiers": ["tok_embeddings", "attn.Wqkv", "mlp_norm"],
            "structure": "flat",  # No model prefix in base form
            "requires_prefix": "model.",  # Prefix needed for classification models
        },
        "bert": {
            "identifiers": ["word_embeddings", "encoder.layer", "LayerNorm"],
            "structure": "nested",  # Has bert. prefix
            "requires_prefix": None,
        },
        "roberta": {
            "identifiers": ["roberta.embeddings", "roberta.encoder"],
            "structure": "nested",
            "requires_prefix": None,
        },
    }

    @staticmethod
    def detect_architecture(state_dict_keys: list[str]) -> str:
        """
        Detect model architecture from state_dict keys.

        Args:
            state_dict_keys: List of keys from model state_dict

        Returns:
            Architecture name: "modernbert", "bert", "roberta", or "unknown"
        """
        # key_set = set(state_dict_keys)  # Not used
        keys_str = " ".join(state_dict_keys)

        # Check for specific patterns
        for arch_name, patterns in ModelArchitectureUtils.ARCHITECTURE_PATTERNS.items():
            identifiers_raw = patterns.get("identifiers")
            if identifiers_raw is None:
                continue
            if isinstance(identifiers_raw, str):
                identifiers_iterable = [identifiers_raw]
            elif isinstance(identifiers_raw, Iterable):
                identifiers_iterable = list(identifiers_raw)
            else:
                continue

            if all(
                any(identifier in key for key in state_dict_keys)
                for identifier in identifiers_iterable
            ):
                logger.info(f"Detected {arch_name} architecture")
                return arch_name

        # Additional checks for ModernBert
        if "tok_embeddings" in keys_str and "Wqkv" in keys_str:
            return "modernbert"

        # Check by prefix
        if any(k.startswith("bert.") for k in state_dict_keys):
            return "bert"
        elif any(k.startswith("roberta.") for k in state_dict_keys):
            return "roberta"

        logger.warning("Could not detect model architecture")
        return "unknown"

    @staticmethod
    def needs_prefix_conversion(
        saved_keys: list[str], target_architecture: str
    ) -> tuple[bool, Optional[str]]:
        """
        Check if state_dict keys need prefix conversion.

        Args:
            saved_keys: Keys from saved state_dict
            target_architecture: Target model architecture

        Returns:
            Tuple of (needs_conversion, prefix_to_add)
        """
        # Check if keys already have expected prefix
        if target_architecture == "modernbert":
            # ModernBert in classification form needs "model." prefix
            has_model_prefix = any(k.startswith("model.") for k in saved_keys)
            has_flat_structure = any(
                k.startswith("embeddings.") or k.startswith("layers.") for k in saved_keys
            )

            if has_flat_structure and not has_model_prefix:
                return True, "model."

        return False, None

    @staticmethod
    def convert_state_dict_keys(
        state_dict: dict[str, Any],
        add_prefix: Optional[str] = None,
        remove_prefix: Optional[str] = None,
        skip_keys: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Convert state_dict keys by adding or removing prefixes.

        Args:
            state_dict: Original state dict
            add_prefix: Prefix to add to keys
            remove_prefix: Prefix to remove from keys
            skip_keys: List of key patterns to skip conversion

        Returns:
            Converted state dict
        """
        if skip_keys is None:
            skip_keys = ["pruning_head"]

        converted = {}

        for key, value in state_dict.items():
            # Check if key should be skipped
            if any(skip_pattern in key for skip_pattern in skip_keys):
                converted[key] = value
                continue

            new_key = key

            # Remove prefix if specified
            if remove_prefix and key.startswith(remove_prefix):
                new_key = key[len(remove_prefix) :]

            # Add prefix if specified
            if add_prefix:
                new_key = f"{add_prefix}{new_key}"

            converted[new_key] = value

        return converted

    @staticmethod
    def auto_fix_state_dict(
        state_dict: dict[str, Any],
        target_model_keys: list[str],
        architecture: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Automatically fix state_dict to match target model structure.

        Args:
            state_dict: State dict to fix
            target_model_keys: Expected keys from target model
            architecture: Model architecture (will be detected if not provided)

        Returns:
            Fixed state dict
        """
        saved_keys = list(state_dict.keys())

        if architecture is None:
            architecture = ModelArchitectureUtils.detect_architecture(saved_keys)

        # For ModernBert, check if we need to add model. prefix
        if architecture == "modernbert":
            needs_prefix, prefix = ModelArchitectureUtils.needs_prefix_conversion(
                saved_keys, architecture
            )

            if needs_prefix:
                logger.info(f"Adding '{prefix}' prefix to ModernBert state_dict keys")
                return ModelArchitectureUtils.convert_state_dict_keys(
                    state_dict, add_prefix=prefix, skip_keys=["pruning_head"]
                )

        # If no conversion needed, return original
        return state_dict

    @staticmethod
    def normalize_state_dict_for_saving(
        state_dict: dict[str, Any], architecture: str
    ) -> dict[str, Any]:
        """
        Normalize state_dict for consistent saving format.

        Args:
            state_dict: State dict to normalize
            architecture: Model architecture

        Returns:
            Normalized state dict
        """
        # For ModernBert, we want to save without model. prefix
        if architecture == "modernbert" and any(k.startswith("model.") for k in state_dict.keys()):
            logger.info("Removing 'model.' prefix from ModernBert state_dict for saving")
            return ModelArchitectureUtils.convert_state_dict_keys(
                state_dict, remove_prefix="model.", skip_keys=["pruning_head"]
            )

        return state_dict
