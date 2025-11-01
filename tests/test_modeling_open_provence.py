from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import nltk
import numpy as np
import pytest
import torch
import torch.nn as nn
from open_provence import modeling_open_provence_standalone as standalone
from open_provence.modeling_open_provence_standalone import (
    DEFAULT_ENGLISH_SENTENCE_MAX_CHARS,
    DEFAULT_PROCESS_THRESHOLD,
    OpenProvenceConfig,
    OpenProvenceModel,
    OpenProvenceRawPrediction,
    _collect_candidate_sentences,
    _fragmentize_example,
    _FragmentRecord,
    _normalize_sentences,
    _split_token_lists,
    _tokenize_sentences_with_context,
    create_auto_sentence_splitter,
    create_english_sentence_splitter,
    english_sentence_splitter,
    fast_bunkai_sentence_splitter,
    simple_sentence_splitter,
)


@pytest.fixture(scope="module", autouse=True)
def ensure_punkt_model() -> None:
    try:
        nltk.data.find("tokenizers/punkt/english.pickle")
    except LookupError:
        nltk.download("punkt", quiet=True)


class DummyTokenizer:
    sep_token = "|"
    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2

    def __call__(
        self,
        sentences,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
        return_offsets_mapping: bool = False,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ):
        if isinstance(sentences, str):
            sentences_list = [sentences]
        else:
            sentences_list = list(sentences)

        input_ids = [[ord(ch) for ch in sentence] for sentence in sentences_list]

        if add_special_tokens:
            input_ids = [[self.cls_token_id, *ids, self.sep_token_id] for ids in input_ids]

        result: dict[str, Any] = {"input_ids": input_ids}

        if return_attention_mask:
            result["attention_mask"] = [[1] * len(ids) for ids in input_ids]

        if return_offsets_mapping:
            offsets = []
            for sentence in sentences_list:
                offsets.append([(idx, idx + 1) for idx, _ in enumerate(sentence)])
            result["offset_mapping"] = offsets

        if return_tensors == "pt":
            import torch

            result = {key: torch.tensor(value) for key, value in result.items()}

        return result

    def encode(self, text, add_special_tokens: bool = False):
        tokens = [ord(ch) for ch in text]
        if add_special_tokens:
            return [self.cls_token_id, *tokens, self.sep_token_id]
        return tokens

    def batch_decode(
        self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ):
        return ["".join(chr(i) for i in tokens) for tokens in token_ids]

    def decode(self, tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(i) for i in tokens)

    def build_inputs_with_special_tokens(self, tokens_a, tokens_b=None):
        tokens_b = tokens_b or []
        if tokens_b:
            return [self.cls_token_id, *tokens_a, self.sep_token_id, *tokens_b, self.sep_token_id]
        return [self.cls_token_id, *tokens_a, self.sep_token_id]

    def create_token_type_ids_from_sequences(self, tokens_a, tokens_b):
        tokens_b = tokens_b or []
        if tokens_b:
            return [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        return [0] * (len(tokens_a) + 2)


class WhitespaceTokenizer(DummyTokenizer):
    def batch_decode(
        self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ):
        return ["   " for _ in token_ids]

    def decode(self, tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "fallback"


class DoubleSepTokenizer(DummyTokenizer):
    def build_inputs_with_special_tokens(self, tokens_a, tokens_b=None):
        tokens_b = tokens_b or []
        if tokens_b:
            return [
                self.cls_token_id,
                *tokens_a,
                self.sep_token_id,
                self.sep_token_id,
                *tokens_b,
                self.sep_token_id,
            ]
        return [self.cls_token_id, *tokens_a, self.sep_token_id, self.sep_token_id]

    def create_token_type_ids_from_sequences(self, tokens_a, tokens_b=None):
        tokens_b = tokens_b or []
        if tokens_b:
            return [0] * (len(tokens_a) + 3) + [1] * (len(tokens_b) + 1)
        return [0] * (len(tokens_a) + 3)


@pytest.fixture
def minimal_model_config(monkeypatch):
    class TinyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.Embedding(32, 4)
            self.config = SimpleNamespace(hidden_size=4)

        def forward(self, *args: Any, **kwargs: Any):  # pragma: no cover - not exercised here
            raise NotImplementedError

        def get_input_embeddings(self):
            return self.embeddings

        def set_input_embeddings(self, value):
            self.embeddings = value

    monkeypatch.setattr(
        standalone.AutoModelForSequenceClassification,
        "from_config",
        lambda config: TinyBackbone(),
    )
    monkeypatch.setattr(
        standalone.AutoTokenizer,
        "from_pretrained",
        lambda reference: DummyTokenizer(),
    )

    return OpenProvenceConfig(
        base_model_config={
            "model_type": "bert",
            "vocab_size": 32,
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "intermediate_size": 4,
        },
        tokenizer_name_or_path="dummy-tokenizer",
        pruning_config={"hidden_size": 4},
        max_length=16,
    )


def test_model_init_accepts_device_kwarg(minimal_model_config):
    model = OpenProvenceModel(minimal_model_config, device="cpu")

    assert model._runtime_device.type == "cpu"
    assert next(model.parameters()).device.type == "cpu"


def test_model_init_rejects_unknown_device(minimal_model_config):
    with pytest.raises(ValueError, match="Invalid device specification"):
        OpenProvenceModel(minimal_model_config, device="accelerator-42")


def test_config_parses_default_threadshold() -> None:
    config = OpenProvenceConfig(default_threadshold=0.25)

    assert config.default_threadshold == pytest.approx(0.25)
    assert config.default_threshold == pytest.approx(0.25)


def test_config_warns_when_default_threshold_used() -> None:
    with pytest.warns(RuntimeWarning, match="default_threshold"):
        config = OpenProvenceConfig(default_threshold=0.3)

    assert config.default_threadshold == pytest.approx(0.3)
    assert config.default_threshold == pytest.approx(0.3)


def test_resolve_process_threshold_prefers_model_default() -> None:
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    model.default_threshold = 0.45

    resolved = OpenProvenceModel._resolve_process_threshold(model, None)

    assert resolved == pytest.approx(0.45)


def test_resolve_process_threshold_falls_back_to_constant() -> None:
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    if hasattr(model, "default_threshold"):
        delattr(model, "default_threshold")

    resolved = OpenProvenceModel._resolve_process_threshold(model, None)

    assert resolved == pytest.approx(DEFAULT_PROCESS_THRESHOLD)


def test_process_uses_resolved_threshold(monkeypatch):
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    model.tokenizer = DummyTokenizer()
    model.max_length = 32
    model._runtime_device = torch.device("cpu")

    sentinel_threshold = 0.37

    def fake_resolve(self, value):
        assert value is None
        return sentinel_threshold

    monkeypatch.setattr(OpenProvenceModel, "_resolve_process_threshold", fake_resolve)

    def fake_splitter(text: str) -> list[str]:
        return [text] if text else []

    monkeypatch.setattr(
        OpenProvenceModel,
        "_resolve_sentence_splitter",
        lambda self, sentence_splitter, language: fake_splitter,
    )
    monkeypatch.setattr(
        OpenProvenceModel,
        "_normalize_inputs",
        lambda self, question, context: (["q"], [["ctx"]], "str"),
    )
    monkeypatch.setattr(
        OpenProvenceModel,
        "_resolve_titles",
        lambda self, queries, contexts, title, first_line_as_title: (contexts, [None]),
    )
    monkeypatch.setattr(
        OpenProvenceModel,
        "_build_preprocess_jobs",
        lambda self, queries, contexts, titles, splitter, strip_sentences, show_progress: ([], []),
    )
    monkeypatch.setattr(OpenProvenceModel, "_resolve_preprocess_workers", lambda self, workers: 0)

    def fake_auto_tune(
        self,
        *,
        total_jobs,
        inference_batch_size,
        current_workers,
        current_preprocess_batch,
        current_prefetch,
        workers_explicit,
        batch_explicit,
        prefetch_explicit,
    ):
        return current_workers, current_preprocess_batch, current_prefetch

    monkeypatch.setattr(OpenProvenceModel, "_auto_tune_preprocess_loader", fake_auto_tune)

    captured_threshold: dict[str, float] = {}

    def fake_postprocess(
        self,
        queries,
        contexts,
        contexts_info,
        *,
        threshold,
        always_select_title,
        use_best_reranker_score,
        sentence_probability_groups_requested,
        collect_sentence_texts,
        first_line_as_title,
        zero_score_when_empty,
    ):
        _ = zero_score_when_empty
        captured_threshold["value"] = threshold
        return (
            [["ctx"]],
            [[None]],
            [[0.0]],
            None,
            None,
            [[None]],
            None,
            0.0,
        )

    monkeypatch.setattr(OpenProvenceModel, "_postprocess_contexts", fake_postprocess)

    class EmptyLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset

        def __iter__(self):
            class _Iter:
                def __iter__(self):
                    return self

                def __next__(self):
                    raise StopIteration

                def _shutdown_workers(self):
                    return None

            return _Iter()

    monkeypatch.setattr(standalone, "DataLoader", EmptyLoader)

    result = OpenProvenceModel.process(
        model,
        question="What is provenance?",
        context="Long context",
        show_progress=False,
    )

    assert captured_threshold["value"] == pytest.approx(sentinel_threshold)
    assert result["pruned_context"] == "ctx"


def test_normalize_inputs_preserves_manual_sentences_single_query():
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    queries, contexts, structure = OpenProvenceModel._normalize_inputs(  # type: ignore[misc]
        model,
        "question",
        [["s1", "s2"], ["s3"]],
    )

    assert queries == ["question"]
    assert structure == "list"
    assert contexts == [[["s1", "s2"], ["s3"]]]


def test_normalize_inputs_preserves_manual_sentences_multiple_queries():
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    queries, contexts, structure = OpenProvenceModel._normalize_inputs(  # type: ignore[misc]
        model,
        ["q1", "q2"],
        [
            [["a1", "a2"], ["b1"]],
            [["c1"], ["d1", "d2"]],
        ],
    )

    assert queries == ["q1", "q2"]
    assert structure == "nested"
    assert contexts == [
        [["a1", "a2"], ["b1"]],
        [["c1"], ["d1", "d2"]],
    ]


def test_extract_first_line_titles_handles_mixed_inputs():
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    contexts = [
        [
            "Title line\nBody line one\nBody line two",
            ["", "List Title", "Item A", "Item B"],
        ]
    ]

    updated_contexts, titles = model._extract_first_line_titles(contexts)  # type: ignore[misc]

    assert updated_contexts == [
        [
            "Body line one\nBody line two",
            ["Item A", "Item B"],
        ]
    ]
    assert titles == [["Title line", "List Title"]]


def test_apply_reordering_sorts_and_limits():
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    pruned_contexts = [["docA", "docB", "docC"]]
    reranking_scores: list[list[float | None]] = [[0.1, None, 0.9]]
    compression_rates = [[10.0, 5.0, 1.0]]
    kept_sentences = [[["ka"], ["kb"], ["kc"]]]
    removed_sentences = [[["ra"], ["rb"], ["rc"]]]
    titles = [[None, "Title B", "Title C"]]
    sentence_probs = [[[0.1], [0.2], [0.3]]]

    (
        pruned_out,
        scores_out,
        compression_out,
        kept_out,
        removed_out,
        titles_out,
        probs_out,
    ) = model._apply_reordering(  # type: ignore[misc]
        pruned_contexts,
        reranking_scores,
        compression_rates,
        kept_sentences,
        removed_sentences,
        titles,
        sentence_probs,
        top_k=2,
    )

    assert pruned_out == [["docC", "docA"]]
    assert scores_out == [[0.9, 0.1]]
    assert compression_out == [[1.0, 10.0]]
    assert kept_out == [[["kc"], ["ka"]]]
    assert removed_out == [[["rc"], ["ra"]]]
    assert titles_out == [["Title C", None]]
    assert probs_out == [[[0.3], [0.1]]]


def test_apply_reordering_allows_empty_results():
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    pruned_contexts = [["docA", "docB"]]
    reranking_scores: list[list[float | None]] = [[0.4, 0.2]]
    compression_rates = [[3.0, 1.0]]
    kept_sentences = [[["ka"], ["kb"]]]
    removed_sentences = [[["ra"], ["rb"]]]
    titles = [[None, None]]

    (
        pruned_out,
        scores_out,
        compression_out,
        kept_out,
        removed_out,
        titles_out,
        probs_out,
    ) = model._apply_reordering(  # type: ignore[misc]
        pruned_contexts,
        reranking_scores,
        compression_rates,
        kept_sentences,
        removed_sentences,
        titles,
        sentence_probability_groups=None,
        top_k=0,
    )

    assert pruned_out == [[]]
    assert scores_out == [[]]
    assert compression_out == [[]]
    assert kept_out == [[]]
    assert removed_out == [[]]
    assert titles_out == [[]]
    assert probs_out is None


def test_apply_reordering_handles_missing_sentence_texts():
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    pruned_contexts = [["docA", "docB"]]
    reranking_scores: list[list[float | None]] = [[0.9, 0.8]]
    compression_rates = [[5.0, 2.0]]
    titles = [[None, None]]

    (
        pruned_out,
        scores_out,
        compression_out,
        kept_out,
        removed_out,
        titles_out,
        probs_out,
    ) = model._apply_reordering(  # type: ignore[misc]
        pruned_contexts,
        reranking_scores,
        compression_rates,
        kept_sentences=None,
        removed_sentences=None,
        title_values=titles,
        sentence_probability_groups=None,
        top_k=None,
    )

    assert pruned_out == pruned_contexts
    assert scores_out == reranking_scores
    assert compression_out == compression_rates
    assert kept_out is None
    assert removed_out is None
    assert titles_out == titles
    assert probs_out is None


def _build_postprocess_inputs(
    pruning_prob: float, ranking_score: float
) -> tuple[list[str], list[list[str]], dict[tuple[int, int], dict[str, Any]]]:
    fragment = _FragmentRecord(
        text="Sentence 1",
        sentence_index=0,
        fragment_index=0,
        global_index=0,
        token_length=1,
        token_ids=[1],
    )
    raw_prediction = OpenProvenceRawPrediction(
        query="query",
        contexts=["Sentence 1"],
        ranking_score=ranking_score,
        pruning_probs=np.array([pruning_prob], dtype=np.float32),
        context_ranges=[(0, 1)],
    )
    contexts_info = {
        (0, 0): {
            "sentences": ["Sentence 1"],
            "fragments": [fragment],
            "blocks": [[fragment]],
            "prefix_length": 0,
            "prefix_sentences": [],
            "prefix_token_counts": [],
            "title_is_first_sentence": False,
            "original_text": "Sentence 1",
            "raw_blocks": [(0, raw_prediction)],
        }
    }
    queries = ["query"]
    contexts = [["Sentence 1"]]
    return queries, contexts, contexts_info


def test_postprocess_sets_zero_score_when_pruned_empty():
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    queries, contexts, contexts_info = _build_postprocess_inputs(
        pruning_prob=0.0,
        ranking_score=0.87,
    )

    (
        pruned_contexts,
        reranking_scores,
        compression_rates,
        kept_sentences,
        removed_sentences,
        title_values,
        sentence_probability_groups,
        _,
    ) = model._postprocess_contexts(
        queries,
        contexts,
        contexts_info,
        threshold=0.5,
        always_select_title=False,
        use_best_reranker_score=True,
        sentence_probability_groups_requested=False,
        collect_sentence_texts=True,
        first_line_as_title=False,
        zero_score_when_empty=True,
    )

    assert pruned_contexts == [[""]]
    assert reranking_scores == [[0.0]]
    assert compression_rates[0][0] == pytest.approx(100.0)
    assert kept_sentences == [[[]]]
    assert removed_sentences == [[["Sentence 1"]]]
    assert title_values == [[None]]
    assert sentence_probability_groups is None


def test_postprocess_retains_ranking_score_when_empty_if_disabled():
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    queries, contexts, contexts_info = _build_postprocess_inputs(
        pruning_prob=0.0,
        ranking_score=0.73,
    )

    (
        pruned_contexts,
        reranking_scores,
        compression_rates,
        kept_sentences,
        removed_sentences,
        title_values,
        sentence_probability_groups,
        _,
    ) = model._postprocess_contexts(
        queries,
        contexts,
        contexts_info,
        threshold=0.5,
        always_select_title=False,
        use_best_reranker_score=True,
        sentence_probability_groups_requested=False,
        collect_sentence_texts=True,
        first_line_as_title=False,
        zero_score_when_empty=False,
    )

    assert pruned_contexts == [[""]]
    assert reranking_scores == [[0.73]]
    assert compression_rates[0][0] == pytest.approx(100.0)
    assert kept_sentences == [[[]]]
    assert removed_sentences == [[["Sentence 1"]]]
    assert title_values == [[None]]
    assert sentence_probability_groups is None


def test_postprocess_preserves_reranker_score_when_text_remains():
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    queries, contexts, contexts_info = _build_postprocess_inputs(
        pruning_prob=1.0,
        ranking_score=0.42,
    )

    (
        pruned_contexts,
        reranking_scores,
        compression_rates,
        kept_sentences,
        removed_sentences,
        title_values,
        sentence_probability_groups,
        _,
    ) = model._postprocess_contexts(
        queries,
        contexts,
        contexts_info,
        threshold=0.5,
        always_select_title=False,
        use_best_reranker_score=True,
        sentence_probability_groups_requested=False,
        collect_sentence_texts=True,
        first_line_as_title=False,
        zero_score_when_empty=True,
    )

    assert pruned_contexts == [["Sentence 1"]]
    assert reranking_scores == [[0.42]]
    assert compression_rates[0][0] == pytest.approx(0.0)
    assert kept_sentences == [[["Sentence 1"]]]
    assert removed_sentences == [[[]]]
    assert title_values == [[None]]
    assert sentence_probability_groups is None


def test_collect_candidate_sentences_prefers_manual():
    example = {
        "context_text": "ignored",
        "prefix_sentences": ["prefix"],
        "manual_sentences": ["manual", None],
    }

    def splitter(text: str) -> list[str]:
        return ["split-1", "split-2"]

    sentences = _collect_candidate_sentences(example, splitter)
    assert sentences == ["prefix", "manual"]


def test_normalize_sentences_strip_and_fallback():
    sentences = _normalize_sentences(["  hello  ", "", "\n"], " context ", True)
    assert sentences == ["hello"]

    fallback = _normalize_sentences([], " context ", True)
    assert fallback == ["context"]


def test_split_token_lists_respects_fragment_size():
    fragments = _split_token_lists([[1, 2, 3, 4, 5]], max_fragment_tokens=2)
    assert fragments == [
        ([1, 2], 0, 0, 0),
        ([3, 4], 0, 1, 1),
        ([5], 0, 2, 2),
    ]


def test_fragmentize_example_chunks_and_strips():
    tokenizer = DummyTokenizer()

    def splitter(text: str) -> list[str]:
        return [" foo ", "bar", " ", "baz"]

    example = {"context_text": " foo bar baz ", "prefix_sentences": ["  prefix  "]}

    result = _fragmentize_example(
        example,
        tokenizer,
        max_fragment_tokens=3,
        splitter=splitter,
        strip_sentences=True,
    )

    assert result["sentences"] == ["prefix", "foo", "bar", "baz"]
    assert result["fragment_texts"][0] == "pre"
    assert result["fragment_sentence_index"][0] == 0
    assert result["fragment_fragment_index"][0] == 0


def test_fragmentize_example_falls_back_when_decoded_empty():
    tokenizer = WhitespaceTokenizer()

    def splitter(text: str) -> list[str]:
        return ["context"]

    example = {"context_text": "context"}

    result = _fragmentize_example(
        example,
        tokenizer,
        max_fragment_tokens=5,
        splitter=splitter,
        strip_sentences=True,
    )

    assert result["fragment_texts"] == ["fallback"]
    assert result["fragment_token_ids"] == [[99, 111, 110, 116, 101]]


def test_fragmentize_example_sentence_boundary_mode_keeps_sentences():
    tokenizer = DummyTokenizer()

    def splitter(text: str) -> list[str]:
        return ["こんにちは、", "可愛いですね"]

    example = {"context_text": "こんにちは、可愛いですね"}

    result = _fragmentize_example(
        example,
        tokenizer,
        max_fragment_tokens=50,
        splitter=splitter,
        strip_sentences=False,
        respect_sentence_boundaries=True,
    )

    assert result["fragment_texts"] == ["こんにちは、", "可愛いですね"]
    assert result["fragment_fragment_index"] == [0, 0]


def test_fragmentize_example_sentence_boundary_mode_splits_when_required():
    tokenizer = DummyTokenizer()

    def splitter(text: str) -> list[str]:
        return ["ABCDEFG"]

    example = {"context_text": "ABCDEFG"}

    result = _fragmentize_example(
        example,
        tokenizer,
        max_fragment_tokens=3,
        splitter=splitter,
        strip_sentences=False,
        respect_sentence_boundaries=True,
    )

    assert result["fragment_texts"] == ["ABC", "DEF", "G"]
    assert result["fragment_fragment_index"] == [0, 1, 2]


def test_fast_bunkai_sentence_splitter_handles_japanese():
    text = "寿司が好きです。ラーメンも好きです。"
    sentences = fast_bunkai_sentence_splitter(text)
    assert sentences == ["寿司が好きです。", "ラーメンも好きです。"]


def test_tokenize_sentences_with_context_matches_offsets():
    tokenizer = DummyTokenizer()
    sentences = ["abc", "def"]
    tokens = _tokenize_sentences_with_context(
        tokenizer,
        sentences,
        prefix_count=0,
        context_text="abcdef",
        strip_sentences=False,
    )
    assert tokens == [tokenizer.encode("abc"), tokenizer.encode("def")]


def test_prepare_block_inputs_matches_build_inputs():
    tokenizer = DummyTokenizer()
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    model.tokenizer = tokenizer
    model.max_length = 128

    query_tokens = tokenizer.encode("Q")
    fragments = [
        _FragmentRecord("abc", 0, 0, 0, 3, tokenizer.encode("abc")),
        _FragmentRecord("def", 1, 0, 1, 3, tokenizer.encode("def")),
    ]

    input_ids, attention_mask, token_type_ids, ranges = model._prepare_block_inputs(
        query_tokens,
        fragments,
    )

    context_tokens = tokenizer.encode("abcdef")
    expected_input_ids = tokenizer.build_inputs_with_special_tokens(query_tokens, context_tokens)
    assert input_ids == expected_input_ids
    assert attention_mask == [1] * len(expected_input_ids)
    assert ranges == [(3, 6), (6, 9)]
    expected_type_ids = tokenizer.create_token_type_ids_from_sequences(
        query_tokens, context_tokens
    )
    assert token_type_ids == expected_type_ids


def test_prepare_block_inputs_handles_additional_special_tokens():
    tokenizer = DoubleSepTokenizer()
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    model.tokenizer = tokenizer
    model.max_length = 128

    query_tokens = tokenizer.encode("Q")
    fragments = [
        _FragmentRecord("abc", 0, 0, 0, 3, tokenizer.encode("abc")),
        _FragmentRecord("def", 1, 0, 1, 3, tokenizer.encode("def")),
    ]

    input_ids, attention_mask, token_type_ids, ranges = model._prepare_block_inputs(
        query_tokens,
        fragments,
    )

    context_tokens = tokenizer.encode("abcdef")
    expected_input_ids = tokenizer.build_inputs_with_special_tokens(query_tokens, context_tokens)
    assert input_ids == expected_input_ids
    assert attention_mask == [1] * len(expected_input_ids)
    assert ranges == [(4, 7), (7, 10)]
    expected_type_ids = tokenizer.create_token_type_ids_from_sequences(
        query_tokens, context_tokens
    )
    assert token_type_ids == expected_type_ids


def test_english_sentence_splitter_handles_lists_with_nltk():
    text = "1: First item\n2: Second item. Third bullet? Fourth!"
    sentences = english_sentence_splitter(text)
    assert [sentence.rstrip() for sentence in sentences] == [
        "1: First item",
        "2: Second item.",
        "Third bullet?",
        "Fourth!",
    ]


def test_english_sentence_splitter_breaks_long_sentences():
    clause = (
        "This clause keeps extending to ensure the sentence remains extremely long "
        "without obvious punctuation boundaries other than the delimiters we inject;"
    )
    long_text = "Intro sentence. " + clause * 30 + " final segment that should still be coherent."
    sentences = english_sentence_splitter(long_text)

    assert sentences[0].rstrip() == "Intro sentence."
    assert len(sentences) > 2
    assert max(len(sentence) for sentence in sentences) <= DEFAULT_ENGLISH_SENTENCE_MAX_CHARS


def test_custom_english_sentence_splitter_respects_limit():
    splitter = create_english_sentence_splitter(max_chars=80)
    long_sentence = "This clause " * 30  # ~360 chars
    sentences = splitter(long_sentence)
    assert all(len(sentence) <= 80 for sentence in sentences)
    assert len(sentences) >= 4


@pytest.mark.parametrize(
    ("text", "expected_with", "expected_without"),
    [
        ("- item one\n- item two\n- item three\n", 3, 1),
        ("1. First entry\n2. Second entry\n3. Third entry\n", 6, 4),
        ("A) Alpha\nB) Bravo\nC) Charlie\n", 3, 1),
    ],
)
def test_english_sentence_splitter_bullet_grouping(
    text: str, expected_with: int, expected_without: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    grouped = create_english_sentence_splitter()(text)
    assert len(grouped) == expected_with

    # Disable bullet heuristics and confirm that the segmentation changes.
    monkeypatch.setattr(standalone, "_looks_like_bullet_line", lambda line: False)
    flattened = standalone.create_english_sentence_splitter()(text)
    assert len(flattened) == expected_without


def test_auto_sentence_splitter_detects_language() -> None:
    auto_splitter = create_auto_sentence_splitter()
    ja_text = "寿司が好きです。ラーメンも好きです。"
    en_text = "Sushi tastes great. I enjoy ramen too."

    assert auto_splitter(ja_text) == fast_bunkai_sentence_splitter(ja_text)
    assert auto_splitter(en_text) == english_sentence_splitter(en_text)


def test_resolve_sentence_splitter_language_routing() -> None:
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    splitter_auto = model._resolve_sentence_splitter(None, "auto")
    splitter_ja = model._resolve_sentence_splitter(None, "ja")
    splitter_en = model._resolve_sentence_splitter(None, "en")

    ja_text = "寿司が好きです。"
    en_text = "Sushi is tasty."

    assert splitter_auto(ja_text) == fast_bunkai_sentence_splitter(ja_text)
    assert splitter_ja(ja_text) == fast_bunkai_sentence_splitter(ja_text)
    assert splitter_en(en_text) == english_sentence_splitter(en_text)

    with pytest.raises(ValueError):
        model._resolve_sentence_splitter(None, "es")


def test_english_sentence_splitter_handles_numeric_headings():
    text = "1001: Item A\n1002: Item B\n"
    sentences = english_sentence_splitter(text)
    assert [s.strip() for s in sentences[:2]] == ["1001: Item A", "1002: Item B"]


def test_process_streaming_pipeline_matches_sequential(monkeypatch):
    model = OpenProvenceModel.__new__(OpenProvenceModel)
    model.tokenizer = DummyTokenizer()
    model.max_length = 32
    model._runtime_device = torch.device("cpu")

    def fake_forward(self, return_dict=True, **kwargs):
        batch = len(kwargs["input_ids"])
        logits = torch.zeros((batch, 1))
        pruning = torch.zeros((batch, 2))
        return {
            "ranking_logits": logits,
            "pruning_logits": pruning,
        }

    monkeypatch.setattr(OpenProvenceModel, "forward", fake_forward)

    question = "What is AI?"
    context = "Artificial intelligence enables machines to learn from experience."

    sequential = OpenProvenceModel.process(
        model,
        question=question,
        context=context,
        batch_size=2,
        show_progress=False,
    )

    streaming = OpenProvenceModel.process(
        model,
        question=question,
        context=context,
        batch_size=2,
        preprocess_workers=0,
        preprocess_batch_size=1,
        show_progress=False,
    )

    assert sequential["pruned_context"] == streaming["pruned_context"]
    assert sequential["compression_rate"] == streaming["compression_rate"]


def test_process_strips_multiprocessing_context_and_shuts_down(monkeypatch):
    import torch
    from open_provence import modeling_open_provence_standalone as standalone

    recorded_kwargs: dict[str, Any] = {}

    class RecordingLoaderIter:
        def __init__(self, dataset, batch_size, collate_fn):
            self._dataset = dataset
            self._batch_size = max(1, int(batch_size))
            self._collate_fn = collate_fn
            self._index = 0
            self.shutdown_called = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._index >= len(self._dataset):
                raise StopIteration
            batch = []
            for _ in range(self._batch_size):
                if self._index >= len(self._dataset):
                    break
                batch.append(self._dataset[self._index])
                self._index += 1
            if not batch:
                raise StopIteration
            if self._collate_fn is not None:
                return self._collate_fn(batch)
            return batch

        def _shutdown_workers(self):
            self.shutdown_called = True

    class RecordingLoader:
        last_iterator: RecordingLoaderIter | None = None

        def __init__(self, dataset, **kwargs):
            nonlocal recorded_kwargs
            recorded_kwargs = dict(kwargs)
            self._dataset = dataset
            self._kwargs = kwargs

        def __iter__(self):
            iterator = RecordingLoaderIter(
                self._dataset,
                self._kwargs.get("batch_size", 1),
                self._kwargs.get("collate_fn"),
            )
            RecordingLoader.last_iterator = iterator
            return iterator

    monkeypatch.setattr(standalone, "DataLoader", RecordingLoader)
    monkeypatch.delenv("OPEN_PROVENCE_PREPROCESS_WORKERS", raising=False)

    model = OpenProvenceModel.__new__(OpenProvenceModel)
    model.tokenizer = DummyTokenizer()
    model.max_length = 32
    model._runtime_device = torch.device("cpu")

    def fake_forward(self, return_dict=True, **kwargs):
        batch = len(kwargs.get("input_ids", []))
        logits = torch.zeros((batch, 1))
        pruning = torch.zeros((batch, 2))
        return {"ranking_logits": logits, "pruning_logits": pruning}

    monkeypatch.setattr(OpenProvenceModel, "forward", fake_forward)

    result = OpenProvenceModel.process(
        model,
        question="What is AI?",
        context="Artificial intelligence enables machines to learn from experience.",
        batch_size=2,
        preprocess_workers=0,
        preprocess_batch_size=1,
        torch_dataloader_kwargs={"multiprocessing_context": "spawn"},
        show_progress=False,
        debug_messages=False,
        sentence_splitter=simple_sentence_splitter,
    )

    assert result["pruned_context"], "process should return non-empty output"
    assert "multiprocessing_context" not in recorded_kwargs
    assert recorded_kwargs.get("num_workers") == 0
    assert recorded_kwargs.get("persistent_workers") is False
    assert RecordingLoader.last_iterator is not None
    assert RecordingLoader.last_iterator.shutdown_called is True


def test_auto_tune_preprocess_loader_gpu_defaults(monkeypatch):
    import torch
    from open_provence import modeling_open_provence_standalone as standalone

    model = OpenProvenceModel.__new__(OpenProvenceModel)
    model._runtime_device = torch.device("cuda:0")

    monkeypatch.setattr(standalone, "_default_preprocess_workers", lambda: 8)
    monkeypatch.setattr(
        OpenProvenceModel, "_estimate_device_memory_bytes", lambda self: 16 * (1024**3)
    )

    workers, batch, prefetch = OpenProvenceModel._auto_tune_preprocess_loader(
        model,
        total_jobs=5000,
        inference_batch_size=192,
        current_workers=8,
        current_preprocess_batch=192,
        current_prefetch=None,
        workers_explicit=False,
        batch_explicit=False,
        prefetch_explicit=False,
    )

    assert workers == 8
    assert batch == 128
    assert prefetch == 8


def test_auto_tune_preprocess_loader_respects_overrides(monkeypatch):
    model = OpenProvenceModel.__new__(OpenProvenceModel)

    workers, batch, prefetch = OpenProvenceModel._auto_tune_preprocess_loader(
        model,
        total_jobs=100,
        inference_batch_size=64,
        current_workers=6,
        current_preprocess_batch=200,
        current_prefetch=5,
        workers_explicit=True,
        batch_explicit=True,
        prefetch_explicit=True,
    )

    assert workers == 6
    assert batch == 200
    assert prefetch == 5


def test_auto_tune_preprocess_loader_clamps_small_corpora(monkeypatch):
    """Small job counts should fall back to single-process execution.

    The heuristic was introduced to avoid the overhead of spinning up
    worker processes when the dataset finishes almost instantly.  We
    pin the defaults here so future adjustments keep that behaviour
    intact.
    """

    model = OpenProvenceModel.__new__(OpenProvenceModel)

    workers, batch, prefetch = OpenProvenceModel._auto_tune_preprocess_loader(
        model,
        total_jobs=100,  # < 2000 ⇒ expect workers=0 regardless of defaults
        inference_batch_size=64,
        current_workers=4,
        current_preprocess_batch=64,
        current_prefetch=None,
        workers_explicit=False,
        batch_explicit=False,
        prefetch_explicit=False,
    )

    assert workers == 0
    assert batch == 64
    assert prefetch is None


def test_auto_tune_preprocess_loader_prefetch_balances_batch(monkeypatch):
    """Default prefetch should scale with batch size and worker count.

    Keeping this check in tests prevents regressions where future
    refactors forget to update the prefetch heuristic alongside the
    worker logic.
    """

    model = OpenProvenceModel.__new__(OpenProvenceModel)

    # The heuristic depends on the CPU budget exposed via
    # `_default_preprocess_workers`.  Pin it here so test results remain
    # deterministic across environments (CI runners expose different
    # core counts).
    monkeypatch.setattr(standalone, "_default_preprocess_workers", lambda: 4)

    workers, batch, prefetch = OpenProvenceModel._auto_tune_preprocess_loader(
        model,
        total_jobs=10_000,
        inference_batch_size=128,
        current_workers=0,
        current_preprocess_batch=128,
        current_prefetch=None,
        workers_explicit=False,
        batch_explicit=True,  # keep batch fixed so only prefetch is tuned
        prefetch_explicit=False,
    )

    assert workers == 4
    assert prefetch == max(2, min(8, (batch + workers - 1) // workers))
