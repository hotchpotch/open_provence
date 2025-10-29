"""Tests for ``open_provence.data_structures`` helpers."""

from __future__ import annotations

import numpy as np
import torch
from open_provence.data_structures import (
    OpenProvenceOnlyOutput,
    OpenProvenceOutput,
    RerankingOpenProvenceOutput,
)


def test_open_provence_output_to_dict_serializes_numpy() -> None:
    output = OpenProvenceOutput(
        ranking_scores=np.array([0.1, 0.2]),
        chunk_predictions=np.array([[1, 0], [0, 1]]),
        chunk_positions=[[(0, 1)]],
        compression_ratio=0.5,
    )

    result = output.to_dict()

    assert result["ranking_scores"] == [0.1, 0.2]
    assert result["chunk_predictions"] == [[1, 0], [0, 1]]
    assert result["chunk_positions"] == [[(0, 1)]]
    assert result["compression_ratio"] == 0.5
    assert "token_scores" not in result


def test_open_provence_only_output_to_dict_handles_torch() -> None:
    logits = torch.tensor([[[0.2, 0.8], [0.7, 0.3]]])
    output = OpenProvenceOnlyOutput(
        pruning_logits=logits,
        pruning_masks=np.array([[1, 0]]),
        num_pruned_tokens=5,
    )

    result = output.to_dict()

    np.testing.assert_allclose(
        result["pruning_logits"],
        [[[0.2, 0.8], [0.7, 0.3]]],
    )
    assert result["pruning_masks"] == [[1, 0]]
    assert result["num_pruned_tokens"] == 5
    assert "pruning_probs" not in result


def test_reranking_output_repr_includes_shapes() -> None:
    output = RerankingOpenProvenceOutput(
        ranking_scores=np.ones(2),
        pruning_masks=np.ones((1, 2)),
        compression_ratio=0.75,
        pruning_logits=torch.zeros(1, 2, 2),
    )

    result = output.to_dict()
    assert result["pruning_logits"] == [[[0.0, 0.0], [0.0, 0.0]]]

    representation = repr(output)
    assert "ranking_scores=(2,)" in representation
    assert "pruning_masks=(1, 2)" in representation
    assert "compression_ratio=0.75" in representation
