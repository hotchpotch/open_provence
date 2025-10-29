"""Tests for ``open_provence.utils.model_architecture``."""

from __future__ import annotations

from open_provence.utils.model_architecture import ModelArchitectureUtils


def test_detect_architecture_modernbert() -> None:
    keys = [
        "tok_embeddings.weight",
        "layers.0.attn.Wqkv.weight",
        "layers.0.mlp_norm.weight",
    ]
    assert ModelArchitectureUtils.detect_architecture(keys) == "modernbert"


def test_detect_architecture_prefers_known_prefixes() -> None:
    keys = [
        "bert.embeddings.word_embeddings.weight",
        "bert.encoder.layer.0.attention.self.query.weight",
        "bert.pooler.dense.weight",
    ]
    assert ModelArchitectureUtils.detect_architecture(keys) == "bert"


def test_detect_architecture_unknown_when_no_patterns() -> None:
    keys = ["linear.weight", "classifier.bias"]
    assert ModelArchitectureUtils.detect_architecture(keys) == "unknown"


def test_needs_prefix_conversion_identifies_flat_modernbert_keys() -> None:
    keys = [
        "embeddings.word_embeddings.weight",
        "layers.0.attn.Wqkv.weight",
    ]
    needs_conversion, prefix = ModelArchitectureUtils.needs_prefix_conversion(keys, "modernbert")
    assert needs_conversion is True
    assert prefix == "model."


def test_needs_prefix_conversion_no_action_when_prefixed() -> None:
    keys = [
        "model.embeddings.word_embeddings.weight",
        "model.layers.0.attn.Wqkv.weight",
    ]
    needs_conversion, prefix = ModelArchitectureUtils.needs_prefix_conversion(keys, "modernbert")
    assert needs_conversion is False
    assert prefix is None


def test_convert_state_dict_keys_adds_and_skips() -> None:
    state_dict = {
        "embeddings.word_embeddings.weight": "weights",
        "layers.0.attn.Wqkv.weight": "attn",
        "pruning_head.linear.weight": "head",
    }

    converted = ModelArchitectureUtils.convert_state_dict_keys(
        state_dict,
        add_prefix="model.",
        skip_keys=["pruning_head"],
    )

    assert converted["model.embeddings.word_embeddings.weight"] == "weights"
    assert converted["model.layers.0.attn.Wqkv.weight"] == "attn"
    assert converted["pruning_head.linear.weight"] == "head"


def test_auto_fix_state_dict_adds_model_prefix_for_modernbert() -> None:
    state_dict = {
        "embeddings.word_embeddings.weight": "weights",
        "layers.0.attn.Wqkv.weight": "attn",
    }

    fixed = ModelArchitectureUtils.auto_fix_state_dict(state_dict, list(state_dict.keys()), "modernbert")

    assert "model.embeddings.word_embeddings.weight" in fixed
    assert "model.layers.0.attn.Wqkv.weight" in fixed


def test_normalize_state_dict_for_saving_removes_model_prefix() -> None:
    state_dict = {
        "model.embeddings.word_embeddings.weight": "weights",
        "model.layers.0.attn.Wqkv.weight": "attn",
    }

    normalized = ModelArchitectureUtils.normalize_state_dict_for_saving(state_dict, "modernbert")

    assert "embeddings.word_embeddings.weight" in normalized
    assert "layers.0.attn.Wqkv.weight" in normalized
