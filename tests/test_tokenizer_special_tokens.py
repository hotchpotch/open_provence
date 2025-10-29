from __future__ import annotations

import pytest
from transformers import AutoTokenizer

ENGLISH_MODEL_NAME = "Alibaba-NLP/gte-reranker-modernbert-base"
JAPANESE_MODEL_NAME = "hotchpotch/japanese-reranker-base-v2"


@pytest.mark.parametrize(
    ("model_name", "query", "document"),
    [
        (
            ENGLISH_MODEL_NAME,
            "What is artificial intelligence?",
            "Artificial intelligence studies intelligent behaviour in machines.",
        ),
        (
            JAPANESE_MODEL_NAME,
            "AIとは何ですか？",
            "AIは人工知能の略称で、人間の知能を機械で再現することを指します。",
        ),
    ],
)
def test_encode_plus_inserts_special_tokens(model_name: str, query: str, document: str) -> None:
    """Ensure encode_plus inserts special tokens for both English and Japanese checkpoints."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    encoding = tokenizer.encode_plus(
        query,
        document,
        add_special_tokens=True,
        return_token_type_ids=True,
    )

    input_ids = encoding["input_ids"]
    assert input_ids, "Tokenizer returned empty input ids."

    start_candidates = [
        tokenizer.cls_token_id,
        tokenizer.bos_token_id,
        tokenizer.special_tokens_map.get("cls_token_id"),
        tokenizer.special_tokens_map.get("bos_token_id"),
    ]
    start_candidates = [tok_id for tok_id in start_candidates if isinstance(tok_id, int)]
    assert start_candidates, "Tokenizer has no CLS/BOS token id defined."
    assert input_ids[0] in start_candidates, (
        f"Expected one of {start_candidates} at start, but got {input_ids[0]}."
    )

    boundary_candidates = [
        tokenizer.sep_token_id,
        tokenizer.eos_token_id,
        tokenizer.special_tokens_map.get("sep_token_id"),
        tokenizer.special_tokens_map.get("eos_token_id"),
    ]
    boundary_candidates = [tok_id for tok_id in boundary_candidates if isinstance(tok_id, int)]
    assert boundary_candidates, "Tokenizer has no SEP/EOS token id defined."

    boundary_indices = [
        idx for idx, tok in enumerate(input_ids[1:], start=1) if tok in boundary_candidates
    ]
    assert boundary_indices, (
        "No boundary token found between query and document "
        f"(candidates={boundary_candidates}, tokens={input_ids})."
    )
    assert boundary_indices[0] < len(input_ids) - 1, (
        "Boundary token should not be the final token."
    )

    # Confirm that removing special tokens changes the sequence start.
    encoding_no_special = tokenizer.encode_plus(
        query,
        document,
        add_special_tokens=False,
        return_token_type_ids=True,
    )
    assert encoding_no_special["input_ids"], "encode_plus without specials returned no tokens."
    assert encoding_no_special["input_ids"][0] not in start_candidates, (
        "encode_plus(add_special_tokens=False) unexpectedly kept the start special token; "
        "this would invalidate the special-token check."
    )
