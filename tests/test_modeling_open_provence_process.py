from __future__ import annotations

from pathlib import Path

import pytest
from open_provence.modeling_open_provence_standalone import (
    OpenProvenceModel,
    _FragmentRecord,
    english_sentence_splitter,
)

ENGLISH_MODEL_PATH = Path("output/open-provence-reranker-v1-gte-modernbert-base")
JAPANESE_MODEL_PATH = Path("output/open-provence-reranker-japanese-v20251021-bs256")
ENGLISH_RELEASE_MODEL_PATH = Path(
    "output/release_models/open-provence-reranker-v1-gte-modernbert-base"
)


def _requires_checkpoint(path: Path) -> bool:
    if not path.exists():
        # These integration tests are intended to run against real checkpoints.
        # The repository omits large weights, so missing artifacts are skipped here,
        # but CI/release environments should provide the checkpoints to exercise the
        # full process pipeline.
        pytest.skip(f"checkpoint directory {path} is missing")
        return False
    return True


def _build_single_fragment(model: OpenProvenceModel, document: str) -> list[_FragmentRecord]:
    token_ids = model.tokenizer.encode(document, add_special_tokens=False)
    return [
        _FragmentRecord(
            text=document,
            sentence_index=0,
            fragment_index=0,
            global_index=0,
            token_length=len(token_ids),
            token_ids=token_ids,
        )
    ]


@pytest.mark.parametrize(
    ("checkpoint", "question", "document"),
    [
        (
            ENGLISH_MODEL_PATH,
            "What is artificial intelligence?",
            "Artificial intelligence studies intelligent behaviour in machines.",
        ),
        (
            JAPANESE_MODEL_PATH,
            "AIとは何ですか？",
            "AIは人工知能の略称で、人間の知能を機械で再現することを指します。",
        ),
    ],
)
def test_prepare_block_inputs_inserts_special_tokens(
    checkpoint: Path, question: str, document: str
) -> None:
    if not _requires_checkpoint(checkpoint):
        return

    model = OpenProvenceModel.from_pretrained(str(checkpoint))

    query_tokens = model.tokenizer.encode(question, add_special_tokens=False)
    fragments = _build_single_fragment(model, document)
    sep_token_ids = model.tokenizer.encode(
        model.tokenizer.sep_token or "", add_special_tokens=False
    )
    blocks = model._assemble_blocks_from_fragments(
        len(query_tokens), len(sep_token_ids), fragments
    )
    input_ids, attention_mask, token_type_ids, ranges = model._prepare_block_inputs(
        query_tokens, blocks[0]
    )

    assert input_ids, "input ids should not be empty"
    cls_candidates = [model.tokenizer.cls_token_id, model.tokenizer.bos_token_id]
    cls_candidates = [candidate for candidate in cls_candidates if isinstance(candidate, int)]
    if cls_candidates:
        assert input_ids[0] in cls_candidates, (
            f"expected CLS/BOS token at start, got {input_ids[0]} (candidates={cls_candidates})"
        )

    sep_candidates = [model.tokenizer.sep_token_id, model.tokenizer.eos_token_id]
    sep_candidates = [candidate for candidate in sep_candidates if isinstance(candidate, int)]
    if sep_candidates:
        assert any(token in sep_candidates for token in input_ids[1:]), (
            "no SEP/EOS token found in prepared inputs"
        )

    assert ranges, "context ranges must be populated"
    assert attention_mask == [1] * len(input_ids), "attention mask should align with inputs"
    if token_type_ids is not None:
        assert len(token_type_ids) == len(input_ids)


@pytest.mark.parametrize(
    ("checkpoint", "expected_flag"),
    [
        (ENGLISH_MODEL_PATH, True),
        (JAPANESE_MODEL_PATH, False),
    ],
)
def test_manual_special_token_detection(checkpoint: Path, expected_flag: bool) -> None:
    if not _requires_checkpoint(checkpoint):
        return

    model = OpenProvenceModel.from_pretrained(str(checkpoint))
    actual_flag = getattr(model, "_manual_special_tokens_required", False)
    assert actual_flag is expected_flag


@pytest.mark.parametrize(
    ("checkpoint", "question", "base_text"),
    [
        (
            ENGLISH_MODEL_PATH,
            "What is artificial intelligence?",
            "Artificial intelligence allows machines to learn from data, adapt to new situations, and support people in solving complex problems across industry, science, and daily life. ",
        ),
        (
            JAPANESE_MODEL_PATH,
            "AIとは何ですか？",
            "人工知能は大量のデータから学習し、人間が扱いにくい複雑な課題に対して柔軟に適応し、日常生活や産業のさまざまな場面で意思決定を支援する技術です。",
        ),
    ],
)
def test_process_handles_long_document(checkpoint: Path, question: str, base_text: str) -> None:
    if not _requires_checkpoint(checkpoint):
        return

    model = OpenProvenceModel.from_pretrained(str(checkpoint))

    multiplier = (2000 // len(base_text)) + 2
    long_document = (base_text * multiplier)[:2000]

    result = model.process(
        question=question,
        context=[long_document],
        threshold=0.1,
        show_progress=False,
        return_sentence_metrics=True,
        return_sentence_texts=True,
    )

    assert "kept_sentences" in result
    assert "removed_sentences" in result
    kept_sentences = result.get("kept_sentences", [[]])
    assert kept_sentences, "process should return kept sentences for long document"
    pruned_contexts = result.get("pruned_context", [])
    assert len(pruned_contexts) == 1
    assert isinstance(pruned_contexts[0], str)


@pytest.mark.parametrize(
    ("checkpoint", "question", "base_text"),
    [
        (
            ENGLISH_MODEL_PATH,
            "What is artificial intelligence?",
            "Artificial intelligence allows machines to learn from data, adapt to new situations, and support people in solving complex problems across industry, science, and daily life. ",
        ),
        (
            JAPANESE_MODEL_PATH,
            "AIとは何ですか？",
            "人工知能は大量のデータから学習し、人間が扱いにくい複雑な課題に対して柔軟に適応し、日常生活や産業のさまざまな場面で意思決定を支援する技術です。",
        ),
    ],
)
def test_process_omits_sentence_texts_by_default(
    checkpoint: Path, question: str, base_text: str
) -> None:
    if not _requires_checkpoint(checkpoint):
        return

    model = OpenProvenceModel.from_pretrained(str(checkpoint))

    result = model.process(
        question=question,
        context=[base_text],
        threshold=0.1,
        show_progress=False,
    )

    assert "kept_sentences" not in result
    assert "removed_sentences" not in result


def test_english_sentence_splitter_preserves_newlines() -> None:
    context = (
        "Work deadlines piled up today, and I kept rambling about budget spreadsheets to my roommate.\n"
        "Next spring I'm planning a trip to Japan so I can wander Kyoto's markets and taste every regional dish I find.\n"
        "Sushi is honestly my favourite—I want to grab a counter seat and let the chef serve endless nigiri until I'm smiling through soy sauce.\n"
        "Later I remembered to water the plants and pay the electricity bill before finally getting some sleep.\n"
    )

    sentences = english_sentence_splitter(context)

    assert len(sentences) == 4
    assert sentences[0].endswith("\n")
    assert sentences[2].startswith("Sushi is honestly my favourite"), (
        "expected sushi sentence to be preserved"
    )


def test_process_filters_irrelevant_sentences_with_english_splitter() -> None:
    if not _requires_checkpoint(ENGLISH_RELEASE_MODEL_PATH):
        return

    model = OpenProvenceModel.from_pretrained(str(ENGLISH_RELEASE_MODEL_PATH))

    question = "What's your favorite Japanese food?"
    context = (
        "Work deadlines piled up today, and I kept rambling about budget spreadsheets to my roommate.\n"
        "Next spring I'm planning a trip to Japan so I can wander Kyoto's markets and taste every regional dish I find.\n"
        "Sushi is honestly my favourite—I want to grab a counter seat and let the chef serve endless nigiri until I'm smiling through soy sauce.\n"
        "Later I remembered to water the plants and pay the electricity bill before finally getting some sleep.\n"
    )

    result = model.process(
        question=question,
        context=context,
        threshold=0.1,
        show_progress=False,
        return_sentence_metrics=True,
        return_sentence_texts=True,
    )

    kept = result.get("kept_sentences", [])
    removed = result.get("removed_sentences", [])
    probs = result.get("sentence_probabilities", [])

    assert kept and kept[0].startswith("Sushi is honestly my favourite"), (
        "relevant sentence should be kept"
    )
    assert removed, "irrelevant sentences should be removed"
    assert all(sentence.endswith("\n") for sentence in removed), "splitter should retain newlines"
    assert probs, "sentence probabilities should be returned"
    assert probs[2] > 0.9
    assert probs[0] < 0.1 and probs[1] < 0.1 and probs[3] < 0.1
