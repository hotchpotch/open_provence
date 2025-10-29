from __future__ import annotations

from typing import Any

from open_provence.modeling_open_provence_standalone import (
    OpenProvenceModel,
    SentenceSplitter,
)


class _StubTokenizer:
    """Minimal tokenizer stub that operates on Unicode codepoints."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def __call__(
        self,
        sentences: list[str],
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
    ) -> dict[str, Any]:
        return {"input_ids": [[ord(ch) for ch in sentence] for sentence in sentences]}

    def batch_decode(
        self,
        sequences: list[list[int]],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> list[str]:
        return ["".join(chr(ch) for ch in seq) for seq in sequences]

    def decode(
        self,
        sequence: list[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return "".join(chr(ch) for ch in sequence)


def _split_sentences(text: str) -> list[str]:
    return [segment for segment in text.split("。") if segment] or [text]


def test_run_sequential_fragmentize_produces_fragments() -> None:
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    model.tokenizer = _StubTokenizer()

    job = {
        "query_idx": 0,
        "context_idx": 0,
        "context_text": "吾輩は猫である。名前はまだない。",
        "prefix_sentences": [],
        "manual_sentences": None,
        "cached_sentences": None,
        "cached_token_lists": None,
    }

    splitter: SentenceSplitter = _split_sentences

    results = model._run_sequential_fragmentize(
        [job],
        max_fragment_tokens=16,
        splitter=splitter,
        show_progress=False,
        strip_sentences=True,
        respect_sentence_boundaries=False,
    )

    assert len(results) == 1
    entry = results[0]

    assert entry["sentences"] == ["吾輩は猫である", "名前はまだない"]
    assert entry["fragment_texts"] == ["吾輩は猫である", "名前はまだない"]
    assert entry["fragment_token_ids"]
    assert entry["timing_sentence_collect"] >= 0.0
    assert entry["timing_fragment_decode"] >= 0.0
