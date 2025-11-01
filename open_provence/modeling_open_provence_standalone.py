"""
Standalone utilities and lightweight Hugging Face integrations for running
Provence reranker checkpoints.

`OpenProvenceModel` provides a self-contained wrapper that can be copied next
to a checkpoint and executed without installing the full ``open_provence``
package.  In addition, this module now exposes `OpenProvenceConfig`,
`OpenProvenceForSequenceClassification`, and
`OpenProvenceForTokenClassification` so that checkpoints can be loaded via
``transformers.AutoModel`` without shipping extra modeling files.

Keep this module self-contained—avoid intra-package imports—so exported
checkpoints remain portable.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import platform
import re
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, TypeAlias, cast

import numpy as np
import torch
import transformers.utils.logging as hf_logging
from torch import FloatTensor, Tensor, nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import ModelOutput

try:
    import nltk
    from nltk.tokenize import PunktSentenceTokenizer
except ImportError as exc:  # pragma: no cover - mandatory dependency
    raise ImportError(
        "modeling_open_provence_standalone.py requires `nltk`. Install via `uv add nltk`."
    ) from exc

LOGGER = logging.getLogger(__name__)


DEFAULT_SPLITTER_LANGUAGE = "auto"  # Updated during export; keep marker for tooling

DEFAULT_PROCESS_THRESHOLD = 0.1  # Default pruning threshold when config does not specify one

_PROGRESS_BAR_ENABLED = True


def enable_progress_bar() -> None:
    """Enable progress output for preprocessing and inference helpers."""

    global _PROGRESS_BAR_ENABLED
    _PROGRESS_BAR_ENABLED = True


def disable_progress_bar() -> None:
    """Disable progress output for preprocessing and inference helpers."""

    global _PROGRESS_BAR_ENABLED
    _PROGRESS_BAR_ENABLED = False


def is_progress_bar_enabled() -> bool:
    """Return True when progress output should be shown."""

    return _PROGRESS_BAR_ENABLED


def _default_preprocess_workers() -> int:
    """Infer a reasonable default number of preprocessing workers."""

    cpu_total: int | None = None
    try:  # pragma: no cover - optional dependency
        import psutil

        cpu_total = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
    except Exception:
        cpu_total = os.cpu_count()

    if cpu_total is None:
        return 0

    return max(0, int(cpu_total) - 1)


_ENGLISH_SENTENCE_TOKENIZER: PunktSentenceTokenizer | None = None
DEFAULT_ENGLISH_SENTENCE_MAX_CHARS = 1200
_ENGLISH_LANGUAGE_ALIASES = {
    "en",
    "english",
    "en-us",
    "en_gb",
    "en-gb",
    "en_us",
}
_BULLET_PREFIX_RE = re.compile(
    r"""^\s*(?:[\-\*\u2022•]+|\d{1,4}[:.)]|[A-Za-z]{1}[:.)])\s+""",
    re.UNICODE,
)

_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_TABLE_ROW_RE = re.compile(r"^\s*\|")
_NUMERIC_HEADING_RE = re.compile(r"^\s*\d{3,}[:\-]")

SUPPORTED_SPLITTER_LANGUAGES = {"ja", "en", "auto"}


def _is_kana_letter_cp(cp: int) -> bool:
    """Return True when code point corresponds to a kana letter."""

    if 0x3041 <= cp <= 0x3096:  # Hiragana letters (ぁ-ゖ)
        return True
    if 0x30A1 <= cp <= 0x30FA:  # Katakana letters (ァ-ヺ)
        return True
    if 0x31F0 <= cp <= 0x31FF:  # Katakana phonetic extensions (ㇰ-ㇿ)
        return True
    if 0xFF71 <= cp <= 0xFF9D:  # Half-width katakana letters (ｱ-ﾝ)
        return True
    return False


def is_japanese_fast(text: str, window: int = 500, min_kana_per_window: int = 1) -> bool:
    """Heuristic that quickly classifies text as Japanese when kana density is high."""

    if not text:
        return False

    if text.isascii():
        return False

    required = math.ceil(len(text) / window) * min_kana_per_window
    if required <= 0:
        return False

    count = 0
    for ch in text:
        cp = ord(ch)
        if cp > 0x7F and _is_kana_letter_cp(cp):
            count += 1
            if count >= required:
                return True
    return False


warnings.filterwarnings("ignore", message="Flash Attention 2 only supports")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_transformers_logger = logging.getLogger("transformers.modeling_utils")
_dynamic_module_logger = logging.getLogger("transformers.dynamic_module_utils")


class _SuppressTransformersWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - log hygiene
        message = record.getMessage()
        if "Flash Attention 2 only supports" in message:
            return False
        if "`torch_dtype` is deprecated" in message:
            return False
        return True


_transformers_logger.addFilter(_SuppressTransformersWarnings())


class _SuppressDynamicModuleWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - log hygiene
        message = record.getMessage()
        if "The module name" in message and "is not a valid Python identifier" in message:
            return False
        if "The module name" in message and "is a reserved keyword" in message:
            return False
        return True


_dynamic_module_logger.addFilter(_SuppressDynamicModuleWarnings())

_LOGGING_CONFIGURED = False


def _ensure_transformers_logging_configured() -> None:
    """Configure transformers logging once to suppress noisy warnings in standalone mode."""

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    hf_logging.set_verbosity_error()
    _LOGGING_CONFIGURED = True


def _supports_flash_attention() -> bool:
    """Return True when CUDA is available and we optimistically enable FlashAttention v2."""

    if not torch.cuda.is_available():
        return False

    try:
        pass  # type: ignore[import-not-found]
    except Exception:
        return False

    return True


def _select_default_torch_dtype(device: str | None) -> torch.dtype | str | None:
    """Select a sensible default dtype based on the target device."""

    if not device:
        return None

    normalized = str(device).lower()
    if normalized == "cuda" and torch.cuda.is_available():
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", None)
        try:
            if callable(supports_bf16) and supports_bf16():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    if normalized == "mps":
        return "auto"

    if normalized == "cpu":
        system = platform.system()
        machine = platform.machine().lower()
        if system == "Darwin" and machine in {"arm64", "aarch64"}:
            return "auto"

    return None


def _coerce_dtype_for_torch_to(value: torch.dtype | str | None) -> torch.dtype | None:
    """Convert user/config provided dtype hints into torch.dtype for Module.to."""

    if value is None or isinstance(value, torch.dtype):
        return value

    normalized = str(value).strip().lower()
    if normalized == "auto":
        return None

    # Accept common dtype aliases used by Transformers configs/CLI flags.
    alias_map: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    resolved = alias_map.get(normalized)
    if resolved is None:
        raise TypeError(f"Unsupported dtype value for torch.to(): {value!r}")

    return resolved


def _mps_is_available() -> bool:
    backend = getattr(torch, "backends", None)
    if backend is None:
        return False
    mps = getattr(backend, "mps", None)
    if mps is None:
        return False
    try:
        return bool(mps.is_available())
    except Exception:
        return False


def auto_detect_device() -> torch.device:
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin" and machine in {"arm64", "aarch64"} and _mps_is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if _mps_is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _validate_device(candidate: torch.device) -> None:
    if candidate.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is not available.")
        if candidate.index is not None:
            total = torch.cuda.device_count()
            if candidate.index < 0 or candidate.index >= total:
                raise ValueError(
                    f"CUDA device index {candidate.index} out of range (count={total})."
                )
    elif candidate.type == "mps":
        if not _mps_is_available():
            raise ValueError("MPS device requested but MPS backend is not available.")


def resolve_inference_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        candidate = device
    elif device is None:
        return auto_detect_device()
    else:
        normalized = str(device).strip().lower()
        if not normalized or normalized == "auto":
            return auto_detect_device()
        if normalized == "cpu":
            candidate = torch.device("cpu")
        elif normalized.startswith("cuda"):
            candidate = torch.device(normalized)
        elif normalized.startswith("mps"):
            candidate = torch.device("mps")
        else:
            raise ValueError(f"Unsupported device specification: {device!r}")

    _validate_device(candidate)
    return candidate


try:
    from fast_bunkai import FastBunkai
except ImportError:  # pragma: no cover - optional dependency
    FastBunkai = None


_FAST_BUNKAI = None
if FastBunkai is not None:  # pragma: no branch
    try:
        _FAST_BUNKAI = FastBunkai()
    except Exception as exc:  # pragma: no cover - runtime safety
        raise RuntimeError("Failed to initialize FastBunkai sentence splitter") from exc


@dataclass
class OpenProvenceHeadConfig:
    """Lightweight configuration for the pruning head."""

    hidden_size: int = 768
    num_labels: int = 2
    classifier_dropout: float = 0.1
    sentence_pooling: str = "mean"
    use_weighted_pooling: bool = False

    def __init__(self, **kwargs: Any) -> None:
        self.hidden_size = int(kwargs.pop("hidden_size", 768))
        self.num_labels = int(kwargs.pop("num_labels", 2))
        self.classifier_dropout = float(kwargs.pop("classifier_dropout", 0.1))
        self.sentence_pooling = kwargs.pop("sentence_pooling", "mean")
        self.use_weighted_pooling = bool(kwargs.pop("use_weighted_pooling", False))
        # Store any additional fields for completeness
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass(frozen=True)
class ProcessPerformanceTrace:
    """Structured runtime telemetry for `OpenProvenceModel.process` calls."""

    preprocess_seconds: float = 0.0
    assembly_seconds: float = 0.0
    inference_seconds: float = 0.0
    postprocess_seconds: float = 0.0
    total_seconds: float = 0.0
    sentence_collect_seconds: float = 0.0
    sentence_normalize_seconds: float = 0.0
    tokenize_seconds: float = 0.0
    fragment_split_seconds: float = 0.0
    fragment_decode_seconds: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "preprocess_seconds": float(self.preprocess_seconds),
            "assembly_seconds": float(self.assembly_seconds),
            "inference_seconds": float(self.inference_seconds),
            "postprocess_seconds": float(self.postprocess_seconds),
            "total_seconds": float(self.total_seconds),
            "sentence_collect_seconds": float(self.sentence_collect_seconds),
            "sentence_normalize_seconds": float(self.sentence_normalize_seconds),
            "tokenize_seconds": float(self.tokenize_seconds),
            "fragment_split_seconds": float(self.fragment_split_seconds),
            "fragment_decode_seconds": float(self.fragment_decode_seconds),
        }


class OpenProvenceHead(nn.Module):
    """Minimal pruning head used by Provence pruning checkpoints."""

    def __init__(self, config: OpenProvenceHeadConfig):
        super().__init__()
        self.config = config
        self.num_labels = getattr(config, "num_labels", 2)
        self.sentence_pooling = getattr(config, "sentence_pooling", "mean")
        self.use_weighted_pooling = getattr(config, "use_weighted_pooling", False)

        dropout_prob = float(getattr(config, "classifier_dropout", 0.1))
        self.dropout = nn.Dropout(dropout_prob)
        hidden_size = int(getattr(config, "hidden_size", 768))
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        if self.use_weighted_pooling:
            self.pooling_weights = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        if hasattr(self, "pooling_weights"):
            nn.init.xavier_uniform_(self.pooling_weights.weight)
            nn.init.zeros_(self.pooling_weights.bias)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sentence_boundaries: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Produce token-level pruning logits."""

        _ = attention_mask  # not required for current inference path
        _ = sentence_boundaries

        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return {"logits": logits}


@dataclass
class OpenProvenceRawPrediction:
    """Container for raw pruning outputs."""

    query: str
    contexts: list[str]
    ranking_score: float | None
    pruning_probs: np.ndarray
    context_ranges: list[tuple[int, int]]


# Type alias for sentence splitter functions
SentenceSplitter = Callable[[str], list[str]]


def _get_english_sentence_tokenizer() -> PunktSentenceTokenizer:
    global _ENGLISH_SENTENCE_TOKENIZER
    if _ENGLISH_SENTENCE_TOKENIZER is None:
        try:
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        except LookupError as exc:  # pragma: no cover - requires punkt download
            raise LookupError(
                "Missing NLTK punkt tokenizer data. Run `python -m nltk.downloader punkt`."
            ) from exc
        if not isinstance(tokenizer, PunktSentenceTokenizer):
            raise TypeError(f"Expected PunktSentenceTokenizer, got {type(tokenizer).__name__}.")
        _ENGLISH_SENTENCE_TOKENIZER = tokenizer
    return _ENGLISH_SENTENCE_TOKENIZER


def _looks_like_bullet_line(line: str) -> bool:
    return bool(_BULLET_PREFIX_RE.match(line))


def _iter_english_blocks(text: str) -> Iterable[tuple[str, int, int]]:
    """Yield text blocks with their span indices for English sentence segmentation."""

    if not text:
        return

    total_len = len(text)
    lines = text.splitlines(keepends=True)
    if not lines:
        block = text
        if block:
            yield block, 0, total_len
        return

    accumulated = 0
    current_parts: list[str] = []
    current_start = 0

    for line in lines:
        line_start = accumulated
        accumulated += len(line)
        plain_line = line.rstrip("\r\n")

        if _looks_like_bullet_line(plain_line) and current_parts:
            block_text = "".join(current_parts)
            if block_text:
                block_end = current_start + len(block_text)
                yield block_text, current_start, block_end
            current_parts = [line]
            current_start = line_start
        else:
            if not current_parts:
                current_start = line_start
            current_parts.append(line)

    if current_parts:
        block_text = "".join(current_parts)
        if block_text:
            block_end = current_start + len(block_text)
            yield block_text, current_start, block_end

    if accumulated < total_len:
        remainder = text[accumulated:]
        if remainder:
            yield remainder, accumulated, total_len


def _split_overlong_sentence(
    sentence: str,
    max_chars: int = DEFAULT_ENGLISH_SENTENCE_MAX_CHARS,
    *,
    preserve_whitespace: bool = False,
) -> list[str]:
    if preserve_whitespace:
        working = sentence
    else:
        working = sentence.strip()

    if not working:
        return []

    if len(working) <= max_chars:
        return [working if preserve_whitespace else working.strip()]

    chunks: list[str] = []
    start = 0
    length = len(working)
    punctuation = ".?!;:\n"

    while start < length:
        target = min(start + max_chars, length)

        # Prefer a newline boundary when available within the window to keep list items concise.
        newline_idx = working.rfind("\n", start + 1, target)
        boundary = None
        if newline_idx != -1 and newline_idx >= start + 1:
            boundary = newline_idx + 1

        if boundary is None or boundary <= start:
            for idx in range(target, start, -1):
                if working[idx - 1] in punctuation:
                    boundary = idx
                    break

        if boundary is None or boundary <= start:
            boundary = target

        chunk = working[start:boundary]
        if not preserve_whitespace:
            chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        start = boundary

    return chunks or ([working] if preserve_whitespace else [working.strip()])


def _split_multiline_sentence(text: str, strip_sentences: bool) -> list[str]:
    if "\n" not in text:
        return [text.strip() if strip_sentences else text]

    segments = text.splitlines(keepends=not strip_sentences)
    meaningful = [segment for segment in segments if segment.strip()]
    if len(meaningful) <= 1:
        return [text.strip() if strip_sentences else text]

    # Skip splitting when the sentence already contains clear punctuation across lines.
    punctuation_count = sum(1 for ch in text if ch in ".?!")
    if punctuation_count >= len(meaningful):
        return [text.strip() if strip_sentences else text]

    # Avoid splitting when any line is excessively long (likely already handled elsewhere).
    if any(len(seg.strip()) > DEFAULT_ENGLISH_SENTENCE_MAX_CHARS for seg in meaningful):
        return [text.strip() if strip_sentences else text]

    processed: list[str] = []
    for segment in meaningful:
        if strip_sentences:
            value = segment.strip()
            if value:
                processed.append(value)
        else:
            processed.append(segment)

    if processed:
        return processed

    return [text.strip() if strip_sentences else text]


def _collect_candidate_sentences(
    example: Mapping[str, Any], splitter: SentenceSplitter
) -> list[str]:
    """Collect sentences from prefixes, manual overrides, or by splitting the context text."""

    prefix_sentences = example.get("prefix_sentences") or []
    manual_sentences = example.get("manual_sentences")
    context_text = str(example.get("context_text", ""))

    sentences: list[str] = [str(s) for s in prefix_sentences if s is not None]
    if manual_sentences is not None:
        sentences.extend(str(s) for s in manual_sentences if s is not None)
    else:
        sentences.extend(str(s) for s in splitter(context_text) if s is not None)

    return sentences


def _fallback_sentence(context_text: str, strip_sentences: bool) -> str:
    if not strip_sentences:
        return context_text
    stripped = context_text.strip()
    return stripped or context_text


def _normalize_sentences(
    raw_sentences: Sequence[str], context_text: str, strip_sentences: bool
) -> list[str]:
    sentences: list[str] = []
    for entry in raw_sentences:
        text = str(entry)
        if not text:
            continue

        segmented = _split_multiline_sentence(text, strip_sentences)
        for segment in segmented:
            if strip_sentences:
                if segment:
                    sentences.append(segment)
            else:
                if segment:
                    sentences.append(segment)

    if sentences:
        return sentences

    return [_fallback_sentence(context_text, strip_sentences)]


def _tokenize_sentences(tokenizer: Any, sentences: Sequence[str]) -> list[list[int]]:
    if not sentences:
        return []
    tokenized = tokenizer(
        list(sentences),
        add_special_tokens=False,
        return_attention_mask=False,
    )
    return tokenized.get("input_ids", []) if isinstance(tokenized, Mapping) else []


def _tokenize_sentences_with_context(
    tokenizer: Any,
    sentences: Sequence[str],
    prefix_count: int,
    context_text: str,
    *,
    strip_sentences: bool,
) -> list[list[int]]:
    return _tokenize_sentences(tokenizer, sentences)


def _split_token_lists(
    token_lists: Sequence[Sequence[int]],
    max_fragment_tokens: int,
    *,
    keep_sentence_boundaries: bool = False,
) -> list[tuple[list[int], int, int, int]]:
    fragments: list[tuple[list[int], int, int, int]] = []
    global_index = 0
    step = max(1, int(max_fragment_tokens))

    for sentence_index, token_ids in enumerate(token_lists):
        tokens = list(token_ids)
        if not tokens:
            continue
        if keep_sentence_boundaries and len(tokens) <= max_fragment_tokens:
            fragments.append((tokens, int(sentence_index), 0, global_index))
            global_index += 1
            continue
        for fragment_index, start in enumerate(range(0, len(tokens), step)):
            fragment_tokens = tokens[start : start + step]
            if not fragment_tokens:
                continue
            fragments.append(
                (fragment_tokens, int(sentence_index), int(fragment_index), global_index)
            )
            global_index += 1

    return fragments


def _collect_sentences_for_job(
    example: Mapping[str, Any],
    splitter: SentenceSplitter,
    strip_sentences: bool,
) -> tuple[list[str], float, float]:
    context_text = str(example.get("context_text", ""))
    cached_sentences = example.get("cached_sentences")

    if cached_sentences is not None:
        sentences = [str(sentence) for sentence in cached_sentences]
        return sentences, 0.0, 0.0

    start = perf_counter()
    raw_sentences = _collect_candidate_sentences(example, splitter)
    sentence_collect_time = perf_counter() - start
    start = perf_counter()
    sentences = _normalize_sentences(raw_sentences, context_text, strip_sentences)
    sentence_normalize_time = perf_counter() - start
    return sentences, sentence_collect_time, sentence_normalize_time


def _tokenize_sentences_for_examples(
    tokenizer: Any,
    sentences_nested: Sequence[Sequence[str]],
    cached_token_lists: Sequence[Any] | None,
) -> tuple[list[list[list[int]]], list[float]]:
    result_token_ids: list[list[list[int]] | None] = []
    timings: list[float | None] = []
    sentences_to_tokenize: list[str] = []
    mapping: list[tuple[int, int]] = []

    total_examples = len(sentences_nested)
    cached_token_lists = cached_token_lists or [None] * total_examples

    for example_index, (sentences, cached_tokens) in enumerate(
        zip(sentences_nested, cached_token_lists)
    ):
        if cached_tokens is not None:
            token_lists = [[int(token) for token in tokens] for tokens in cached_tokens]
            result_token_ids.append(token_lists)
            timings.append(0.0)
            continue

        if sentences:
            mapping.append((example_index, len(sentences)))
            sentences_to_tokenize.extend(sentences)
        result_token_ids.append(None)
        timings.append(None)

    if sentences_to_tokenize:
        start = perf_counter()
        tokenized = tokenizer(
            sentences_to_tokenize,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        tokenize_time = perf_counter() - start
        input_ids = tokenized.get("input_ids", [])
        pointer = 0
        total_sentences = len(sentences_to_tokenize)
        time_per_sentence = tokenize_time / total_sentences if total_sentences else 0.0

        for example_index, sentence_count in mapping:
            slice_ids = input_ids[pointer : pointer + sentence_count]
            pointer += sentence_count
            result_token_ids[example_index] = [
                [int(token) for token in tokens] for tokens in slice_ids
            ]
            timings[example_index] = time_per_sentence * sentence_count

    finalized_token_ids: list[list[list[int]]] = []
    finalized_timings: list[float] = []
    for tokens, timing in zip(result_token_ids, timings):
        finalized_token_ids.append(tokens or [])
        finalized_timings.append(float(timing or 0.0))

    return finalized_token_ids, finalized_timings


def _build_fragment_payload(
    tokenizer: Any,
    sentences: Sequence[str],
    token_lists: Sequence[Sequence[int]],
    context_text: str,
    max_fragment_tokens: int,
    strip_sentences: bool,
    respect_sentence_boundaries: bool,
) -> tuple[dict[str, Any], float, float]:
    normalized_tokens = [[int(token) for token in tokens] for tokens in token_lists]

    start = perf_counter()
    fragments = _split_token_lists(
        normalized_tokens,
        max_fragment_tokens,
        keep_sentence_boundaries=respect_sentence_boundaries,
    )
    fragment_split_time = perf_counter() - start

    if not fragments:
        fallback_source = _fallback_sentence(context_text, strip_sentences)
        fallback_tokens = tokenizer.encode(fallback_source, add_special_tokens=False)
        fragments = [(list(fallback_tokens), 0, 0, 0)]

    start = perf_counter()
    fragment_payload = _decode_and_filter_fragments(
        tokenizer,
        fragments,
        strip_sentences=strip_sentences,
    )
    fragment_decode_time = perf_counter() - start

    if not fragment_payload["fragment_token_ids"]:
        tokens, sentence_idx, fragment_idx, global_idx = fragments[0]
        decoded_text = tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        processed_text = decoded_text.strip() if strip_sentences else decoded_text
        fragment_payload = {
            "fragment_texts": [processed_text],
            "fragment_token_ids": [list(tokens)],
            "fragment_sentence_index": [sentence_idx],
            "fragment_fragment_index": [fragment_idx],
            "fragment_global_index": [global_idx],
        }

    return fragment_payload, fragment_split_time, fragment_decode_time


def _decode_and_filter_fragments(
    tokenizer: Any,
    fragments: Sequence[tuple[list[int], int, int, int]],
    *,
    strip_sentences: bool,
) -> dict[str, list[Any]]:
    if not fragments:
        return {
            "fragment_texts": [],
            "fragment_token_ids": [],
            "fragment_sentence_index": [],
            "fragment_fragment_index": [],
            "fragment_global_index": [],
        }

    token_sequences = [tokens for tokens, _, _, _ in fragments]
    fragment_texts = tokenizer.batch_decode(
        token_sequences,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    filtered_tokens: list[list[int]] = []
    filtered_texts: list[str] = []
    sentence_indices: list[int] = []
    fragment_indices: list[int] = []
    global_indices: list[int] = []

    for text, (tokens, sentence_idx, fragment_idx, global_idx) in zip(fragment_texts, fragments):
        processed_text = text.strip() if strip_sentences else text
        if strip_sentences:
            if not processed_text:
                continue
        else:
            if not text:
                continue
        filtered_tokens.append(list(tokens))
        filtered_texts.append(processed_text)
        sentence_indices.append(sentence_idx)
        fragment_indices.append(fragment_idx)
        global_indices.append(global_idx)

    return {
        "fragment_texts": filtered_texts,
        "fragment_token_ids": filtered_tokens,
        "fragment_sentence_index": sentence_indices,
        "fragment_fragment_index": fragment_indices,
        "fragment_global_index": global_indices,
    }


def _fragmentize_single_job(
    tokenizer: Any,
    job: dict[str, Any],
    *,
    max_fragment_tokens: int,
    splitter: SentenceSplitter,
    strip_sentences: bool,
    respect_sentence_boundaries: bool,
) -> dict[str, Any]:
    sentences, collect_time, normalize_time = _collect_sentences_for_job(
        job,
        splitter,
        strip_sentences,
    )

    token_ids_nested, tokenize_timings = _tokenize_sentences_for_examples(
        tokenizer,
        [sentences],
        [job.get("cached_token_lists")],
    )
    token_lists = token_ids_nested[0]
    if not token_lists:
        cached_lists = job.get("cached_token_lists")
        token_lists = (
            [[int(token) for token in tokens] for tokens in cached_lists] if cached_lists else []
        )

    fragment_payload, fragment_split_time, fragment_decode_time = _build_fragment_payload(
        tokenizer=tokenizer,
        sentences=sentences,
        token_lists=token_lists,
        context_text=str(job.get("context_text", "")),
        max_fragment_tokens=max_fragment_tokens,
        strip_sentences=strip_sentences,
        respect_sentence_boundaries=respect_sentence_boundaries,
    )

    entry = {
        "sentences": sentences,
        "timing_sentence_collect": collect_time,
        "timing_sentence_normalize": normalize_time,
        "timing_tokenize": tokenize_timings[0],
        "timing_fragment_split": fragment_split_time,
        "timing_fragment_decode": fragment_decode_time,
    }
    entry.update(fragment_payload)
    return entry


def _preprocess_collate_fn(
    batch: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not batch:
        return [], []
    jobs, entries = zip(*batch)
    return list(jobs), list(entries)


class _PreprocessDataset(Dataset):
    """Map-style dataset that fragmentizes preprocessing jobs."""

    def __init__(
        self,
        jobs: Sequence[dict[str, Any]],
        tokenizer: Any,
        splitter: SentenceSplitter,
        max_fragment_tokens: int,
        strip_sentences: bool,
        respect_sentence_boundaries: bool,
    ) -> None:
        self._jobs = list(jobs)
        self._tokenizer = tokenizer
        self._splitter = splitter
        self._max_fragment_tokens = max_fragment_tokens
        self._strip_sentences = strip_sentences
        self._respect_sentence_boundaries = respect_sentence_boundaries

    def __len__(self) -> int:
        return len(self._jobs)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        job = self._jobs[index]
        entry = _fragmentize_single_job(
            self._tokenizer,
            job,
            max_fragment_tokens=self._max_fragment_tokens,
            splitter=self._splitter,
            strip_sentences=self._strip_sentences,
            respect_sentence_boundaries=self._respect_sentence_boundaries,
        )
        return job, entry


@dataclass
class _FragmentRecord:
    """Metadata for a context fragment produced during long-context splitting."""

    text: str
    sentence_index: int
    fragment_index: int
    global_index: int
    token_length: int
    token_ids: list[int]


def fast_bunkai_sentence_splitter(text: str) -> list[str]:
    """Split sentences with fast-bunkai. Raises if the library is unavailable."""

    if _FAST_BUNKAI is None:
        raise RuntimeError(
            "fast-bunkai is not installed. Install `fast-bunkai` or provide a custom sentence_splitter "
            "(e.g. `simple_sentence_splitter`)."
        )

    sentences = [sentence for sentence in _FAST_BUNKAI(text) if sentence]
    if sentences:
        return sentences

    return [text] if text else []


def simple_sentence_splitter(text: str) -> list[str]:
    """Lightweight regex-based sentence splitter for Japanese text."""

    if not text:
        return []

    pattern = re.compile(r".+?(?:。|！|？|!|\?|\n|$)", re.S)
    sentences = [match for match in pattern.findall(text) if match]
    if sentences:
        return sentences

    return [text] if text else []


def create_english_sentence_splitter(
    max_chars: int = DEFAULT_ENGLISH_SENTENCE_MAX_CHARS,
) -> SentenceSplitter:
    """Factory for English sentence splitters that preserve whitespace and newlines.

    Processing pipeline (executed for every call of the returned splitter):
    1. `_iter_english_blocks` walks the source text line-by-line, grouping adjacent
       lines while respecting bullet-style headings. This yields blocks together with
       their start/end byte offsets so we always know where we are in the original
       string.
    2. Each block is tokenised with NLTK's Punkt model (`span_tokenize`). The spans
       are mapped back to absolute offsets (`global_start`/`global_end`). We stretch
       the end offset across trailing whitespace so that paragraph boundaries keep
       their newline markers.
    3. Every raw segment is routed through `_split_overlong_sentence`, which trims
       *nothing* but ensures no fragment exceeds ``max_chars``. When Punkt does not
       emit any spans (e.g., extremely long strings without punctuation), the whole
       block is handed directly to this fallback splitter so we still return
       manageable chunks.
    4. Empty segments and whitespace-only fragments are skipped. If the whole text
       reduces to whitespace we fall back to returning the stripped source.

    This design guarantees that:
      * sentence boundaries preserve the original whitespace/newline layout,
      * sections and lists stay intact because block slicing mirrors the input, and
      * even pathological long sentences are clipped deterministically at
        ``max_chars`` before downstream tokenisation.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    def _split_text(text: str) -> list[str]:
        if not text:
            return []

        tokenizer = _get_english_sentence_tokenizer()
        sentences: list[str] = []

        for block_text, block_start, block_end in _iter_english_blocks(text):
            if not block_text:
                continue
            try:
                spans = list(tokenizer.span_tokenize(block_text))
            except LookupError as exc:  # pragma: no cover - requires punkt download
                raise LookupError(
                    "Missing NLTK punkt tokenizer. Run `python -m nltk.downloader punkt`."
                ) from exc

            if not spans:
                segment = text[block_start:block_end]
                if segment.strip():
                    sentences.extend(
                        _split_overlong_sentence(
                            segment,
                            max_chars=max_chars,
                            preserve_whitespace=True,
                        )
                    )
                continue

            for span_start, span_end in spans:
                global_start = block_start + span_start
                global_end = block_start + span_end

                extended_end = global_end
                while extended_end < block_end and text[extended_end].isspace():
                    extended_end += 1

                segment = text[global_start:extended_end]
                if segment and segment.strip():
                    sentences.extend(
                        _split_overlong_sentence(
                            segment,
                            max_chars=max_chars,
                            preserve_whitespace=True,
                        )
                    )

        if sentences:
            return sentences

        fallback = text.strip()
        return [fallback] if fallback else []

    return _split_text


_DEFAULT_ENGLISH_SENTENCE_SPLITTER = create_english_sentence_splitter()


def english_sentence_splitter(text: str) -> list[str]:
    """Default English sentence splitter using the module's configured limit."""

    return _DEFAULT_ENGLISH_SENTENCE_SPLITTER(text)


def create_auto_sentence_splitter(
    *,
    japanese_splitter: SentenceSplitter = fast_bunkai_sentence_splitter,
    english_splitter: SentenceSplitter = english_sentence_splitter,
    kana_window: int = 500,
    min_kana_per_window: int = 1,
) -> SentenceSplitter:
    """Return a splitter that detects Japanese text via kana density before splitting."""

    def _split_text(text: str) -> list[str]:
        if is_japanese_fast(text, window=kana_window, min_kana_per_window=min_kana_per_window):
            return japanese_splitter(text)
        return english_splitter(text)

    return _split_text


def _fragmentize_example(  # pyright: ignore[reportUnusedFunction]
    example: dict[str, Any],
    tokenizer,
    max_fragment_tokens: int,
    splitter: SentenceSplitter,
    strip_sentences: bool,
    *,
    respect_sentence_boundaries: bool = False,
) -> dict[str, Any]:
    """Fragmentize a single context example for parallel preprocessing."""

    context_text = str(example.get("context_text", ""))
    cached_sentences = example.get("cached_sentences")
    cached_token_lists = example.get("cached_token_lists")

    timer_start = perf_counter()

    if cached_sentences is not None:
        sentences = [str(sentence) for sentence in cached_sentences]
        sentence_collect_time = 0.0
        sentence_normalize_time = 0.0
    else:
        raw_sentences = _collect_candidate_sentences(example, splitter)
        sentence_collect_time = perf_counter() - timer_start
        timer_start = perf_counter()
        sentences = _normalize_sentences(raw_sentences, context_text, strip_sentences)
        sentence_normalize_time = perf_counter() - timer_start

    prefix_sentences = example.get("prefix_sentences") or []

    if cached_token_lists is not None:
        token_lists = [[int(token) for token in tokens] for tokens in cached_token_lists]
        tokenize_time = 0.0
    else:
        timer_start = perf_counter()
        token_lists = _tokenize_sentences_with_context(
            tokenizer,
            sentences,
            len(prefix_sentences),
            context_text,
            strip_sentences=strip_sentences,
        )
        tokenize_time = perf_counter() - timer_start
    timer_start = perf_counter()
    fragments = _split_token_lists(
        token_lists,
        max_fragment_tokens,
        keep_sentence_boundaries=respect_sentence_boundaries,
    )
    fragment_split_time = perf_counter() - timer_start

    if not fragments:
        timer_start = perf_counter()
        fallback_source = _fallback_sentence(context_text, strip_sentences)
        fallback_tokens = tokenizer.encode(fallback_source, add_special_tokens=False)
        tokenize_time += perf_counter() - timer_start
        fragments = [(list(fallback_tokens), 0, 0, 0)]
        sentences = [fallback_source]

    timer_start = perf_counter()
    fragment_payload = _decode_and_filter_fragments(
        tokenizer,
        fragments,
        strip_sentences=strip_sentences,
    )
    decode_time = perf_counter() - timer_start

    if not fragment_payload["fragment_token_ids"]:
        tokens, sentence_idx, fragment_idx, global_idx = fragments[0]
        timer_start = perf_counter()
        decoded_text = tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decode_time += perf_counter() - timer_start
        processed_text = decoded_text.strip() if strip_sentences else decoded_text
        fragment_payload = {
            "fragment_texts": [processed_text],
            "fragment_token_ids": [list(tokens)],
            "fragment_sentence_index": [sentence_idx],
            "fragment_fragment_index": [fragment_idx],
            "fragment_global_index": [global_idx],
        }

    return {
        "sentences": sentences,
        "fragment_texts": fragment_payload["fragment_texts"],
        "fragment_sentence_index": fragment_payload["fragment_sentence_index"],
        "fragment_fragment_index": fragment_payload["fragment_fragment_index"],
        "fragment_global_index": fragment_payload["fragment_global_index"],
        "fragment_token_ids": fragment_payload["fragment_token_ids"],
        "timing_sentence_collect": sentence_collect_time,
        "timing_sentence_normalize": sentence_normalize_time,
        "timing_tokenize": tokenize_time,
        "timing_fragment_split": fragment_split_time,
        "timing_fragment_decode": decode_time,
    }


class OpenProvenceConfig(PretrainedConfig):
    """Configuration metadata for OpenProvence checkpoints."""

    model_type = "open_provence"

    def __init__(
        self,
        mode: str = "reranking_pruning",
        base_model_name_or_path: str | None = None,
        base_model_config: dict[str, Any] | PretrainedConfig | None = None,
        tokenizer_name_or_path: str | None = None,
        pruning_config: dict | None = None,
        max_length: int = 512,
        num_labels: int | None = None,
        num_pruning_labels: int | None = None,
        encoder_architecture: str | None = None,
        **kwargs: Any,
    ) -> None:
        raw_default_threadshold = kwargs.pop("default_threadshold", None)
        alt_default_threshold = kwargs.pop("default_threshold", None)
        # Backwards compatibility: drop deprecated language hints from historical configs.
        kwargs.pop("splitter_default_language", None)
        kwargs.pop("standalone_process_default_language", None)
        super().__init__(**kwargs)
        self.mode = mode
        if isinstance(base_model_config, PretrainedConfig):
            base_model_config = base_model_config.to_dict()
        self.base_model_name_or_path = base_model_name_or_path
        self.base_model_config = dict(base_model_config) if base_model_config is not None else None
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.pruning_config = pruning_config or {}
        self.max_length = max_length
        self.encoder_architecture = encoder_architecture
        self.num_labels = 1 if num_labels is None else num_labels
        self.num_pruning_labels = 2 if num_pruning_labels is None else num_pruning_labels
        self.default_threadshold = None
        if raw_default_threadshold is not None:
            try:
                self.default_threadshold = float(raw_default_threadshold)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Config value 'default_threadshold' must be a numeric type convertible to float."
                ) from exc
        elif alt_default_threshold is not None:
            warnings.warn(
                "Config key 'default_threshold' detected. Did you intend 'default_threadshold'? "
                "Using the provided value for backwards compatibility.",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                self.default_threadshold = float(alt_default_threshold)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Config value 'default_threshold' must be a numeric type convertible to float."
                ) from exc
        self.default_threshold = self.default_threadshold


class OpenProvencePreTrainedModel(PreTrainedModel):
    """Base class implementing the shared Provence reranker backbone."""

    config_class = OpenProvenceConfig
    base_model_prefix = "open_provence"

    def __init__(
        self,
        config: OpenProvenceConfig,
        *model_args: Any,
        device: str | torch.device | None = None,
        **model_kwargs: Any,
    ) -> None:
        _ensure_transformers_logging_configured()

        cleaned_kwargs = dict(model_kwargs)
        cleaned_kwargs.pop("device", None)

        resolved_device: torch.device | None = None
        if device is not None:
            try:
                resolved_device = resolve_inference_device(device)
            except ValueError as exc:
                class_name = self.__class__.__name__
                raise ValueError(
                    f"Invalid device specification for {class_name}: {device!r}"
                ) from exc

        super().__init__(config, *model_args, **cleaned_kwargs)
        self.max_length = config.max_length
        self.num_labels = config.num_labels
        self.num_pruning_labels = config.num_pruning_labels
        self.default_splitter_language = DEFAULT_SPLITTER_LANGUAGE
        self._runtime_device = torch.device("cpu")

        self.base_model_config = self._build_base_model_config(config)
        self.ranking_model = AutoModelForSequenceClassification.from_config(self.base_model_config)
        self.pruning_head = OpenProvenceHead(OpenProvenceHeadConfig(**config.pruning_config))
        self.tokenizer = self._init_tokenizer(config)
        self._manual_special_tokens_required = False
        self._manual_cls_token_id: int | None = None
        self._manual_sep_token_id: int | None = None
        self._update_tokenizer_runtime()
        self.default_threshold = self._resolve_default_threshold(config)
        self.eval()

        if resolved_device is not None:
            self.to(device=resolved_device)

    def _build_base_model_config(self, config: OpenProvenceConfig) -> PretrainedConfig:
        if config.base_model_config:
            config_dict = deepcopy(config.base_model_config)
            model_type = config_dict.pop("model_type", None)
            if model_type is None:
                raise ValueError(
                    "base_model_config must include 'model_type' to rebuild the backbone."
                )
            base_config = AutoConfig.for_model(model_type, **config_dict)
        else:
            base_reference = (
                config.base_model_name_or_path
                or config._name_or_path
                or config.encoder_architecture
            )
            if not base_reference:
                raise ValueError(
                    "OpenProvenceConfig must define base_model_config or base_model_name_or_path."
                )
            base_config = AutoConfig.from_pretrained(base_reference, trust_remote_code=True)
        base_config.num_labels = config.num_labels
        return base_config

    def _init_tokenizer(self, config: OpenProvenceConfig):
        tokenizer_reference = (
            config.tokenizer_name_or_path or config._name_or_path or config.base_model_name_or_path
        )
        if not tokenizer_reference:
            raise ValueError("Unable to determine tokenizer reference for OpenProvence model.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_reference)
        except Exception as exc:  # pragma: no cover - surface failure to caller
            raise RuntimeError(
                f"Failed to initialize tokenizer from '{tokenizer_reference}'."
            ) from exc
        return tokenizer

    def _update_tokenizer_runtime(self, max_length_override: int | None = None) -> None:
        if self.tokenizer is None:
            return
        upper_bound = max(getattr(self.tokenizer, "model_max_length", 0) or 0, 1_000_000)
        if max_length_override is not None and max_length_override > 0:
            upper_bound = max(upper_bound, int(max_length_override))
        elif self.max_length and self.max_length > 0:
            upper_bound = max(upper_bound, int(self.max_length))
        self.tokenizer.model_max_length = upper_bound

    def _update_runtime_defaults(self) -> None:
        tokenizer = cast(Any, self.tokenizer)
        special_map = cast(Mapping[str, Any], getattr(tokenizer, "special_tokens_map", {}))
        self._manual_special_tokens_required = self._requires_manual_special_tokens()  # type: ignore[reportCallIssue]
        if self._manual_special_tokens_required:
            self._manual_cls_token_id = self._resolve_special_token_id(
                getattr(tokenizer, "cls_token_id", None),
                special_map.get("cls_token_id"),
                getattr(tokenizer, "bos_token_id", None),
                special_map.get("bos_token_id"),
            )  # type: ignore[reportCallIssue]
            self._manual_sep_token_id = self._resolve_special_token_id(
                getattr(tokenizer, "sep_token_id", None),
                special_map.get("sep_token_id"),
                getattr(tokenizer, "eos_token_id", None),
                special_map.get("eos_token_id"),
            )  # type: ignore[reportCallIssue]
        else:
            self._manual_cls_token_id = None
            self._manual_sep_token_id = None

    def _resolve_default_threshold(self, config: OpenProvenceConfig) -> float:
        value = getattr(config, "default_threadshold", None)
        if value is None:
            return DEFAULT_PROCESS_THRESHOLD
        try:
            return float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - config validation
            raise TypeError(
                "OpenProvenceConfig.default_threadshold must be numeric when provided."
            ) from exc

    def to(self, *args: Any, **kwargs: Any) -> OpenProvencePreTrainedModel:  # type: ignore[override]
        result = super().to(*args, **kwargs)
        candidate = kwargs.get("device") if kwargs else None
        if candidate is None and args:
            candidate = args[0]
        if candidate is not None:
            self._runtime_device = torch.device(candidate)
        return cast("OpenProvencePreTrainedModel", result)

    def get_input_embeddings(self):
        return self.ranking_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.ranking_model.set_input_embeddings(value)

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True):  # type: ignore[override]
        converted = self._convert_legacy_state_dict(state_dict)
        return super().load_state_dict(converted, strict=strict)

    @staticmethod
    def _convert_legacy_state_dict(
        state_dict: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        if any(key.startswith("ranking_model.") for key in state_dict):
            return state_dict
        converted: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("pruning_head."):
                converted[key] = value
            else:
                converted[f"ranking_model.{key}"] = value
        return converted


class OpenProvenceModel(OpenProvencePreTrainedModel):
    """Lightweight wrapper around the Provence reranker checkpoint."""

    def __init__(
        self,
        config: OpenProvenceConfig,
        *model_args: Any,
        device: str | torch.device | None = None,
        **model_kwargs: Any,
    ) -> None:
        super().__init__(config, *model_args, device=device, **model_kwargs)
        self.default_splitter_language = DEFAULT_SPLITTER_LANGUAGE
        self._update_tokenizer_runtime()
        self._update_runtime_defaults()

    def _resolve_process_threshold(self, threshold: float | None) -> float:
        if threshold is None:
            resolved = getattr(self, "default_threshold", DEFAULT_PROCESS_THRESHOLD)
            if resolved is None:
                resolved = DEFAULT_PROCESS_THRESHOLD
        else:
            resolved = threshold

        try:
            return float(resolved)
        except (TypeError, ValueError) as exc:
            raise TypeError("Resolved threshold must be numeric.") from exc

    def _resolve_special_token_id(self, *candidates: int | None) -> int | None:
        for candidate in candidates:
            if isinstance(candidate, int):
                return candidate
        return None

    def _requires_manual_special_tokens(self) -> bool:
        """Detect tokenizers (e.g., ModernBERT) that omit special tokens in build_inputs."""

        tokenizer = cast(Any, self.tokenizer)
        try:
            query_tokens = tokenizer.encode("open provence query", add_special_tokens=False)
            context_tokens = tokenizer.encode("open provence document", add_special_tokens=False)
        except Exception:  # pragma: no cover - tokenizer specific errors
            return False

        if not query_tokens or not context_tokens:
            return False

        built = tokenizer.build_inputs_with_special_tokens(query_tokens, context_tokens)
        built = [int(token) for token in built]

        special_map = cast(Mapping[str, Any], getattr(tokenizer, "special_tokens_map", {}))

        cls_candidates = [
            getattr(tokenizer, "cls_token_id", None),
            special_map.get("cls_token_id"),
            getattr(tokenizer, "bos_token_id", None),
            special_map.get("bos_token_id"),
        ]
        cls_candidates = [value for value in cls_candidates if isinstance(value, int)]

        sep_candidates = [
            getattr(tokenizer, "sep_token_id", None),
            special_map.get("sep_token_id"),
            getattr(tokenizer, "eos_token_id", None),
            special_map.get("eos_token_id"),
        ]
        sep_candidates = [value for value in sep_candidates if isinstance(value, int)]

        missing_cls = bool(cls_candidates) and not any(token in cls_candidates for token in built)
        missing_sep = bool(sep_candidates) and not any(token in sep_candidates for token in built)

        return missing_cls or missing_sep

    @staticmethod
    def _extract_model_output(outputs: Any, key: str) -> torch.Tensor:
        candidate: torch.Tensor | None = None
        if isinstance(outputs, Mapping):
            candidate = outputs.get(key)
            if candidate is None and key == "ranking_logits":
                candidate = outputs.get("logits")
        if candidate is None:
            candidate = getattr(outputs, key, None)
            if candidate is None and key == "ranking_logits":
                candidate = getattr(outputs, "logits", None)

        if candidate is None:
            raise KeyError(f"{key} not found in model outputs")

        return candidate

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        device: str | torch.device | None = None,
        trust_remote_code: bool = True,
        max_length: int | None = None,
        torch_dtype: torch.dtype | str | None = None,
        **kwargs: Any,
    ) -> OpenProvenceModel:
        """Load a finetuned Provence reranker with pruning head."""

        _ensure_transformers_logging_configured()

        try:
            resolved_device = resolve_inference_device(device)
        except ValueError as exc:
            raise ValueError(
                f"Invalid device specification for OpenProvenceModel: {device!r}"
            ) from exc

        resolved_device_str = str(resolved_device).lower()

        if "torch_dtype" in kwargs and "dtype" not in kwargs:
            kwargs["dtype"] = kwargs.pop("torch_dtype")

        target_dtype = kwargs.get("dtype")

        if target_dtype is None and torch_dtype is not None:
            target_dtype = torch_dtype

        if target_dtype is None:
            dtype_hint = _select_default_torch_dtype(resolved_device_str)
            if dtype_hint is not None:
                target_dtype = dtype_hint

        attn_impl = kwargs.get("attn_implementation")
        want_flash_attention = False

        if resolved_device_str.startswith("cuda"):
            if _supports_flash_attention():
                want_flash_attention = True
                if target_dtype is None:
                    bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                    target_dtype = torch.bfloat16 if bf16_supported else torch.float16
                if attn_impl is None:
                    attn_impl = "flash_attention_2"
            else:
                if attn_impl is None:
                    attn_impl = "eager"
        elif resolved_device_str.startswith("mps"):
            if attn_impl is None:
                attn_impl = "eager"

        if target_dtype is not None:
            kwargs["dtype"] = target_dtype
        if attn_impl is not None:
            kwargs["attn_implementation"] = attn_impl

        def _apply_config_overrides(target: Any) -> None:
            attn_impl = kwargs.get("attn_implementation")
            if attn_impl is not None and hasattr(target, "config"):
                setattr(target.config, "attn_implementation", attn_impl)
            dtype_value = kwargs.get("dtype")
            if dtype_value is not None and hasattr(target, "config"):
                setattr(target.config, "torch_dtype", dtype_value)

        try:
            model = super().from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        except Exception:
            if not want_flash_attention:
                raise

            kwargs["attn_implementation"] = "eager"
            kwargs["dtype"] = torch.float32

            model = super().from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        requested_dtype = kwargs.get("dtype")
        _apply_config_overrides(model)
        if hasattr(model, "ranking_model"):
            _apply_config_overrides(getattr(model, "ranking_model"))

        dtype_for_to = _coerce_dtype_for_torch_to(requested_dtype)
        if dtype_for_to is not None:
            model.to(device=resolved_device, dtype=dtype_for_to)
        else:
            model.to(resolved_device)

        if max_length is not None:
            model.max_length = int(max_length)
            if hasattr(model.config, "max_length"):
                model.config.max_length = int(max_length)

        model._update_tokenizer_runtime(max_length_override=max_length)
        model._update_runtime_defaults()

        model.eval()
        return model

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> ModelOutput | tuple[torch.Tensor, ...]:
        """Run the ranking backbone and pruning head."""

        if input_ids is None:
            raise ValueError("input_ids must be provided")

        effective_return_dict = return_dict if return_dict is not None else True

        attention_mask = (
            attention_mask.to(self._runtime_device) if attention_mask is not None else None
        )
        input_ids = input_ids.to(self._runtime_device)

        outputs = self.ranking_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        ranking_logits = cast(FloatTensor, outputs.logits)
        hidden_states = outputs.hidden_states[-1]
        pruning_inputs = hidden_states
        head_param = next(self.pruning_head.parameters(), None)
        if head_param is not None and pruning_inputs.dtype != head_param.dtype:
            pruning_inputs = pruning_inputs.to(head_param.dtype)

        pruning_outputs = self.pruning_head(
            hidden_states=pruning_inputs,
            attention_mask=attention_mask,
        )
        pruning_logits = cast(Tensor, pruning_outputs["logits"])

        loss_tensor: torch.Tensor | None = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss_fct = nn.BCEWithLogitsLoss()
                loss_tensor = loss_fct(ranking_logits.view(-1), labels.float())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss_tensor = loss_fct(
                    ranking_logits.view(-1, self.config.num_labels), labels.view(-1)
                )

        loss_output: FloatTensor | None
        if loss_tensor is None:
            loss_output = None
        else:
            loss_output = cast(FloatTensor, loss_tensor.to(dtype=ranking_logits.dtype))

        result = SequenceClassifierOutput(
            loss=loss_output,
            logits=ranking_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        setattr(result, "pruning_logits", pruning_logits)
        setattr(result, "ranking_logits", ranking_logits)

        if not effective_return_dict:
            output: tuple[torch.Tensor, ...] = (ranking_logits, pruning_logits)
            if loss_output is not None:
                return (loss_output,) + output
            return output

        return result

    @torch.no_grad()
    def get_raw_predictions(
        self,
        query: str,
        contexts: Iterable[str],
    ) -> OpenProvenceRawPrediction:
        """Compute token-level keep probabilities for a single context list."""

        batch_result = self.get_raw_predictions_batch(query, [list(contexts)])
        return batch_result[0]

    def get_raw_predictions_batch(
        self,
        query: str | Sequence[str],
        contexts_batch: Sequence[Sequence[str]],
        batch_size: int | None = None,
    ) -> list[OpenProvenceRawPrediction]:
        """Compute raw predictions for multiple context lists.

        Supports either a single query string shared across the batch or a sequence of
        per-sample queries matching ``contexts_batch``.
        """

        if not contexts_batch:
            return []

        sep_token = self.tokenizer.sep_token or ""
        if batch_size is None or batch_size <= 0:
            batch_size = len(contexts_batch)

        if isinstance(query, Sequence) and not isinstance(query, str):
            query_list = [str(entry) for entry in query]
            if len(query_list) != len(contexts_batch):
                raise ValueError(
                    "When providing multiple queries, their count must match contexts_batch."
                )
        else:
            query_list = [str(query)] * len(contexts_batch)

        results: list[OpenProvenceRawPrediction] = []

        for start in range(0, len(contexts_batch), batch_size):
            chunk = contexts_batch[start : start + batch_size]
            chunk_queries = query_list[start : start + batch_size]

            chunk_combined = [
                chunk_queries[idx] + sep_token + "".join(contexts)
                for idx, contexts in enumerate(chunk)
            ]
            encoding = self.tokenizer(
                chunk_combined,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoding = {key: value.to(self._runtime_device) for key, value in encoding.items()}

            model_outputs = self.forward(return_dict=True, **encoding)
            ranking_logits = self._extract_model_output(model_outputs, "ranking_logits")
            pruning_logits = self._extract_model_output(model_outputs, "pruning_logits")
            ranking_logits = ranking_logits.detach().cpu()
            pruning_logits = pruning_logits.detach().cpu()

            if ranking_logits.dtype != torch.float32:
                ranking_logits = ranking_logits.to(dtype=torch.float32)
            if pruning_logits.dtype != torch.float32:
                pruning_logits = pruning_logits.to(dtype=torch.float32)

            for idx, contexts in enumerate(chunk):
                if len(contexts) == 0:
                    continue

                logits = ranking_logits[idx]
                if logits.ndim == 0 or logits.numel() == 1:
                    ranking_score = torch.sigmoid(logits.flatten())[0].item()
                else:
                    ranking_score = torch.sigmoid(logits[..., 0]).item()

                pruning_logit = pruning_logits[idx]
                pruning_probs = torch.softmax(pruning_logit, dim=-1).numpy()
                if pruning_probs.ndim == 2 and pruning_probs.shape[1] == 2:
                    pruning_probs = pruning_probs[:, 1]
                elif pruning_probs.ndim == 1:
                    pruning_probs = pruning_probs
                else:
                    pruning_probs = pruning_probs.reshape(-1)

                context_ranges = self._context_ranges_from_contexts(chunk_queries[idx], contexts)

                results.append(
                    OpenProvenceRawPrediction(
                        query=chunk_queries[idx],
                        contexts=list(contexts),
                        ranking_score=ranking_score,
                        pruning_probs=pruning_probs,
                        context_ranges=context_ranges,
                    )
                )

        return results

    def predict_with_thresholds(
        self,
        query: str,
        contexts: Iterable[str],
        thresholds: Iterable[float],
        *,
        use_majority: bool = False,
    ) -> dict[str, Any]:
        """Return keep/delete decisions for each context under the thresholds."""

        raw = self.get_raw_predictions(query, contexts)
        predictions: dict[float, list[int]] = {}

        for threshold in thresholds:
            context_predictions: list[int] = []

            for start, end in raw.context_ranges:
                segment = raw.pruning_probs[start:end]
                if segment.size == 0:
                    context_predictions.append(1)
                    continue

                if use_majority:
                    kept_tokens = np.count_nonzero(segment > threshold)
                    context_predictions.append(1 if kept_tokens >= (segment.size / 2) else 0)
                else:
                    mean_prob = float(segment.mean())
                    context_predictions.append(1 if mean_prob > threshold else 0)

            predictions[threshold] = context_predictions

        return {
            "query": raw.query,
            "contexts": raw.contexts,
            "ranking_score": raw.ranking_score,
            "predictions": predictions,
            "context_ranges": raw.context_ranges,
            "pruning_probs": raw.pruning_probs,
        }

    def _compute_context_ranges(
        self,
        query: str,
        contexts: list[str],
        pruning_probs: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Reconstruct token spans for each context string."""

        sep_token = self.tokenizer.sep_token or ""
        prefix = query + sep_token
        context_boundaries: list[int] = []

        for idx in range(len(contexts)):
            cumulative_text = prefix + "".join(contexts[: idx + 1])
            cumulative_encoding = self.tokenizer(
                cumulative_text,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = cast(Tensor, cumulative_encoding["input_ids"])
            context_boundaries.append(int(input_ids.shape[1]))

        prefix_encoding = self.tokenizer(
            prefix,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        prefix_len = int(cast(Tensor, prefix_encoding["input_ids"]).shape[1])

        context_ranges: list[tuple[int, int]] = []
        prev = prefix_len
        total = pruning_probs.shape[0]

        for boundary in context_boundaries:
            end = min(boundary, total)
            context_ranges.append((prev, end))
            prev = end

        return context_ranges

    def _context_ranges_from_contexts(
        self,
        query: str,
        contexts: Sequence[str],
    ) -> list[tuple[int, int]]:
        """Compute token index ranges for a list of contexts given a query."""

        if not contexts:
            return []

        sep_token = self.tokenizer.sep_token or ""
        prefix = query + sep_token

        cumulative_texts = []
        for idx in range(len(contexts)):
            cumulative_texts.append(prefix + "".join(contexts[: idx + 1]))

        boundaries: list[int] = []
        for text in cumulative_texts:
            encoding = self.tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = cast(Tensor, encoding["input_ids"])
            boundaries.append(int(input_ids.shape[1]))

        prefix_encoding = self.tokenizer(
            prefix,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        prefix_len = int(cast(Tensor, prefix_encoding["input_ids"]).shape[1])

        ranges: list[tuple[int, int]] = []
        prev = prefix_len
        for boundary in boundaries:
            ranges.append((prev, boundary))
            prev = boundary

        return ranges

    def _resolve_prefix_sentences(
        self,
        title_spec: None | str | list[str] | list[list[str]],
        context_idx: int,
    ) -> tuple[list[str], bool]:
        """Determine prefix sentences and whether the first context sentence is a title."""

        prefix_sentences: list[str] = []
        title_is_first_sentence = False

        if title_spec == "first_sentence":
            title_is_first_sentence = True
        elif isinstance(title_spec, list):
            if title_spec and isinstance(title_spec[0], list):
                raw_title = title_spec[context_idx] if context_idx < len(title_spec) else None
                if raw_title:
                    prefix_sentences.extend(
                        [
                            title.strip()
                            for title in raw_title
                            if isinstance(title, str) and title.strip()
                        ]
                    )
            else:
                raw_title = title_spec[context_idx] if context_idx < len(title_spec) else None
                if isinstance(raw_title, str) and raw_title.strip():
                    prefix_sentences.append(raw_title.strip())
        elif isinstance(title_spec, str) and title_spec.strip():
            prefix_sentences.append(title_spec.strip())

        if prefix_sentences:
            last_idx = len(prefix_sentences) - 1
            prefix_sentences[last_idx] = prefix_sentences[last_idx].rstrip("\n") + "\n"

        return prefix_sentences, title_is_first_sentence

    def _resolve_sentence_splitter(
        self,
        splitter: SentenceSplitter | Mapping[str, SentenceSplitter] | None,
        language: str | None,
    ) -> SentenceSplitter:
        if isinstance(splitter, Mapping):
            if language is None:
                raise ValueError("language must be provided when sentence_splitter is a mapping")
            if language in splitter:
                return splitter[language]
            raise ValueError(f"No sentence splitter registered for language '{language}'")

        if callable(splitter):
            return splitter

        default_language = getattr(self, "default_splitter_language", None)
        lang = language if language is not None else default_language
        if lang is None:
            lang = "auto"

        lang_normalized = str(lang).lower()
        if lang_normalized == "auto":
            return create_auto_sentence_splitter()

        if lang_normalized == "ja":
            return fast_bunkai_sentence_splitter

        if lang_normalized == "en":
            return english_sentence_splitter

        raise ValueError(
            f"Unsupported language code for sentence splitting: '{lang}'. Supported values are 'auto', 'en', and 'ja'."
        )

    def _run_sequential_fragmentize(
        self,
        jobs: list[dict[str, Any]],
        *,
        max_fragment_tokens: int,
        splitter: SentenceSplitter,
        show_progress: bool,
        strip_sentences: bool,
        respect_sentence_boundaries: bool,
    ) -> list[dict[str, Any]]:
        processed_entries: list[dict[str, Any]] = []
        if not jobs:
            return processed_entries

        progress = None
        if show_progress and is_progress_bar_enabled():
            try:
                from tqdm import tqdm  # pragma: no cover - optional dependency
            except Exception:  # pragma: no cover - tqdm may be unavailable
                progress = None
            else:
                progress = tqdm(total=len(jobs), desc="Preprocess")

        for job in jobs:
            entry = _fragmentize_single_job(
                self.tokenizer,
                job,
                max_fragment_tokens=max_fragment_tokens,
                splitter=splitter,
                strip_sentences=strip_sentences,
                respect_sentence_boundaries=respect_sentence_boundaries,
            )
            processed_entries.append(entry)
            if progress is not None:
                progress.update(1)

        if progress is not None:
            progress.close()

        return processed_entries

    def _truncate_fragment(self, fragment: _FragmentRecord, max_tokens: int) -> _FragmentRecord:
        if max_tokens <= 0:
            max_tokens = 1
        if fragment.token_length <= max_tokens:
            return fragment

        new_tokens = fragment.token_ids[:max_tokens]
        new_text = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return _FragmentRecord(
            text=new_text,
            sentence_index=fragment.sentence_index,
            fragment_index=fragment.fragment_index,
            global_index=fragment.global_index,
            token_length=len(new_tokens),
            token_ids=list(new_tokens),
        )

    def _prepare_block_inputs(
        self,
        query_tokens: Sequence[int],
        fragments: Sequence[_FragmentRecord],
    ) -> tuple[list[int], list[int], list[int] | None, list[tuple[int, int]]]:
        query_list = [int(token) for token in query_tokens]
        context_tokens: list[int] = []
        for fragment in fragments:
            context_tokens.extend(int(token) for token in fragment.token_ids)

        built_with_specials = self.tokenizer.build_inputs_with_special_tokens(
            query_list, context_tokens
        )
        built_with_specials = [int(token) for token in built_with_specials]

        manual_override = getattr(self, "_manual_special_tokens_required", False)
        manual_cls_token = getattr(self, "_manual_cls_token_id", None)
        manual_sep_token = getattr(self, "_manual_sep_token_id", None)

        if manual_override:
            # Some tokenizers, notably ModernBERT, omit CLS/SEP when provided with pre-tokenised
            # input. We rebuild the sequence manually so that downstream code sees consistent
            # boundaries without ever converting back to strings.
            input_ids: list[int] = []
            if manual_cls_token is not None:
                input_ids.append(manual_cls_token)
            input_ids.extend(int(token) for token in query_list)
            if manual_sep_token is not None:
                input_ids.append(manual_sep_token)
            input_ids.extend(int(token) for token in context_tokens)
            if manual_sep_token is not None and context_tokens:
                input_ids.append(manual_sep_token)
        else:
            # Most tokenizers already handle special tokens correctly, so we can reuse the
            # sequence they produce directly.
            if built_with_specials:
                input_ids = built_with_specials
            else:
                input_ids = [int(token) for token in query_list]
                input_ids.extend(int(token) for token in context_tokens)

        attention_mask = [1] * len(input_ids)

        token_type_ids: list[int] | None
        try:
            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(
                query_list,
                context_tokens,
            )
        except Exception:
            token_type_ids = None
        else:
            if token_type_ids is not None:
                token_type_ids = [int(token) for token in token_type_ids]

        def _find_subsequence_start(
            haystack: Sequence[int],
            needle: Sequence[int],
        ) -> int:
            if not needle:
                return -1
            needle_list = list(needle)
            limit = len(haystack) - len(needle_list) + 1
            for idx in range(max(limit, 0)):
                if haystack[idx : idx + len(needle_list)] == needle_list:
                    return idx
            return -1

        ranges: list[tuple[int, int]] = []
        if context_tokens:
            context_start = _find_subsequence_start(input_ids, context_tokens)
            if context_start < 0:
                prefix_ids = self.tokenizer.build_inputs_with_special_tokens(query_list, [])
                context_start = len(prefix_ids)
            cursor = context_start
            for fragment in fragments:
                start = cursor
                cursor += len(fragment.token_ids)
                ranges.append((start, cursor))
        else:
            ranges = []

        if token_type_ids is not None and len(token_type_ids) < len(input_ids):
            pad_value = token_type_ids[-1] if token_type_ids else 0
            token_type_ids = token_type_ids + [pad_value] * (len(input_ids) - len(token_type_ids))

        if token_type_ids is None:
            token_type_ids = [0] * len(input_ids)
            context_start = ranges[0][0] if context_tokens else len(input_ids)
            for idx in range(context_start, len(input_ids)):
                token_type_ids[idx] = 1

        return input_ids, attention_mask, token_type_ids, ranges

    def _precompute_sentences_and_tokens(
        self,
        context_text: str,
        prefix_sentences: list[str],
        manual_sentences: list[str] | None,
        splitter: SentenceSplitter,
        strip_sentences: bool,
    ) -> tuple[list[str], list[list[int]]]:
        example_payload = {
            "context_text": context_text,
            "prefix_sentences": prefix_sentences,
            "manual_sentences": manual_sentences,
        }
        raw_sentences = _collect_candidate_sentences(example_payload, splitter)
        sentences = _normalize_sentences(raw_sentences, context_text, strip_sentences)
        token_lists = _tokenize_sentences_with_context(
            self.tokenizer,
            sentences,
            len(prefix_sentences),
            context_text,
            strip_sentences=strip_sentences,
        )
        return sentences, token_lists

    def _assemble_blocks_from_fragments(
        self,
        query_token_length: int,
        sep_token_length: int,
        fragments: list[_FragmentRecord],
    ) -> list[list[_FragmentRecord]]:
        if not fragments:
            return []

        available_len = self.max_length - 2  # [CLS], [SEP]
        base_len = query_token_length + sep_token_length
        max_fragment_capacity = max(1, available_len - base_len)

        blocks: list[list[_FragmentRecord]] = []
        current_block: list[_FragmentRecord] = []
        current_len = base_len

        for fragment in fragments:
            fragment_len = fragment.token_length

            if current_len + fragment_len <= available_len:
                current_block.append(fragment)
                current_len += fragment_len
                continue

            if current_block:
                blocks.append(current_block)
                current_block = []
                current_len = base_len

            truncated_fragment = self._truncate_fragment(fragment, max_fragment_capacity)
            current_block.append(truncated_fragment)
            current_len = base_len + truncated_fragment.token_length

        if current_block:
            blocks.append(current_block)

        return blocks

    def _normalize_inputs(
        self,
        question: str | Sequence[str],
        context: ContextInput,
    ) -> tuple[list[str], list[list[Any]], str]:
        """Normalize input structures for process()."""

        if isinstance(question, str):
            queries = [question]
        else:
            queries = [str(q) for q in question]

        def _is_sequence(value: Any) -> bool:
            return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))

        def _normalize_context_collection(values: Sequence[Any]) -> list[Any]:
            normalized: list[Any] = []
            for item in values:
                if _is_sequence(item):
                    normalized.append([str(element) for element in item])
                else:
                    normalized.append(str(item))
            return normalized

        if isinstance(context, str):
            context_structure = "str"
            contexts: list[list[Any]] = [[context]]
        elif not _is_sequence(context):
            raise ValueError("Unsupported context format")
        elif len(queries) == 1:
            normalized_contexts = _normalize_context_collection(context)
            context_structure = "list"
            contexts = [normalized_contexts]
        else:
            context_structure = "nested"
            normalized_nested: list[list[Any]] = []
            for entry in context:
                if not _is_sequence(entry):
                    raise ValueError("Number of context lists must match number of queries")
                normalized_nested.append(_normalize_context_collection(entry))
            contexts = normalized_nested

        if context_structure == "list" and len(queries) != 1:
            raise ValueError("Single list of contexts requires a single query")
        if context_structure == "nested" and len(contexts) != len(queries):
            raise ValueError("Number of context lists must match number of queries")
        if context_structure == "str" and len(queries) != 1:
            raise ValueError("Single context string requires a single query")

        if context_structure in {"str", "list"}:
            contexts = [contexts[0]]

        return queries, contexts, context_structure

    def _prepare_titles(
        self,
        title: None | str | Sequence[str] | Sequence[Sequence[str]],
        queries: list[str],
        contexts: list[list[str]],
    ) -> list[Any]:
        """Normalize title inputs for process()."""

        n_queries = len(queries)

        if title is None:
            return [None] * n_queries

        if isinstance(title, str):
            if title == "first_sentence":
                return ["first_sentence"] * n_queries
            return [[title for _ in ctxs] for ctxs in contexts]

        if isinstance(title, Sequence):
            normalized: list[Any] = []
            for entry in title:
                if isinstance(entry, Sequence) and not isinstance(entry, str):
                    normalized.append([str(value) for value in entry])
                else:
                    normalized.append(str(entry))

            if n_queries == 1 and all(isinstance(item, str) for item in normalized):
                return [[str(item) for item in normalized]]

            if len(normalized) == n_queries and all(isinstance(item, list) for item in normalized):
                return [list(map(str, item)) for item in normalized]  # type: ignore[list-item]

            if len(normalized) == n_queries and all(isinstance(item, str) for item in normalized):
                return [[value for _ in contexts[idx]] for idx, value in enumerate(normalized)]

        raise ValueError("Unsupported title format")

    def _extract_first_line_titles(
        self,
        contexts: list[list[Any]],
    ) -> tuple[list[list[Any]], list[list[str]]]:
        """Split the first non-empty line from each context as a title candidate."""

        updated_contexts: list[list[Any]] = []
        extracted_titles: list[list[str]] = []

        for context_group in contexts:
            group_titles: list[str] = []
            updated_group: list[Any] = []

            for entry in context_group:
                if isinstance(entry, list):
                    normalized = [str(value) for value in entry]
                    title_candidate = ""
                    remainder: list[str] = []
                    for idx, segment in enumerate(normalized):
                        if segment.strip():
                            title_candidate = segment.rstrip("\r\n")
                            remainder = normalized[idx + 1 :]
                            break
                    else:
                        remainder = normalized
                    group_titles.append(title_candidate)
                    updated_group.append(remainder)
                else:
                    text_entry = str(entry)
                    title_candidate = ""
                    remainder_text = ""
                    if text_entry:
                        lines = text_entry.splitlines(keepends=True)
                        remainder_segments: list[str] = []
                        for idx, line in enumerate(lines):
                            if line.strip():
                                title_candidate = line.rstrip("\r\n")
                                remainder_segments = lines[idx + 1 :]
                                break
                        else:
                            remainder_segments = lines
                        remainder_text = "".join(remainder_segments)
                    group_titles.append(title_candidate)
                    updated_group.append(remainder_text)

            extracted_titles.append(group_titles)
            updated_contexts.append(updated_group)

        return updated_contexts, extracted_titles

    def _resolve_titles(
        self,
        queries: list[str],
        contexts: list[list[Any]],
        title: None | str | Sequence[str] | Sequence[Sequence[str]],
        *,
        first_line_as_title: bool,
    ) -> tuple[list[list[Any]], list[Any]]:
        """Resolve title inputs, optionally extracting first lines from contexts."""

        title_payload: None | str | Sequence[str] | Sequence[Sequence[str]]
        if first_line_as_title:
            if title not in (None, "first_sentence"):
                raise ValueError(
                    "first_line_as_title=True cannot be combined with an explicit title override."
                )
            contexts, extracted_titles = self._extract_first_line_titles(contexts)
            title_payload = extracted_titles
        else:
            title_payload = title

        titles = self._prepare_titles(title_payload, queries, contexts)
        return contexts, titles

    def _build_preprocess_jobs(
        self,
        queries: list[str],
        contexts: list[list[Any]],
        titles: list[Any],
        splitter: SentenceSplitter,
        *,
        strip_sentences: bool,
        show_progress: bool,
    ) -> tuple[list[dict[str, Any]], list[list[int]]]:
        """Construct preprocessing jobs and cache query token ids."""

        preprocess_jobs: list[dict[str, Any]] = []
        query_token_ids: list[list[int]] = []

        total_contexts = sum(len(context_collection) for context_collection in contexts)
        progress = None
        if show_progress and is_progress_bar_enabled() and total_contexts:
            try:
                from tqdm import tqdm  # pragma: no cover - optional dependency
            except Exception:  # pragma: no cover - tqdm may be unavailable
                progress = None
            else:
                progress = tqdm(total=total_contexts, desc="Prepare contexts")

        for query_idx, query_text in enumerate(queries):
            query_tokens = self.tokenizer.encode(query_text, add_special_tokens=False)
            query_token_ids.append(query_tokens)
            title_spec = titles[query_idx]

            for context_idx, context_entry in enumerate(contexts[query_idx]):
                if isinstance(context_entry, list):
                    manual_sentences = [str(s) for s in context_entry if str(s).strip()]
                    context_text = "".join(manual_sentences)
                else:
                    manual_sentences = None
                    context_text = context_entry

                prefix_sentences, title_is_first_sentence = self._resolve_prefix_sentences(
                    title_spec,
                    context_idx,
                )
                cached_sentences, cached_token_lists = self._precompute_sentences_and_tokens(
                    context_text,
                    prefix_sentences,
                    manual_sentences,
                    splitter,
                    strip_sentences,
                )

                prefix_count = len(prefix_sentences)
                if cached_token_lists is not None:
                    prefix_token_counts = [
                        len(tokens) for tokens in cached_token_lists[:prefix_count]
                    ]
                else:
                    prefix_token_counts = [
                        len(self.tokenizer.encode(sentence, add_special_tokens=False))
                        if sentence
                        else 0
                        for sentence in prefix_sentences
                    ]

                preprocess_jobs.append(
                    {
                        "query_idx": query_idx,
                        "context_idx": context_idx,
                        "context_text": context_text,
                        "prefix_sentences": prefix_sentences,
                        "title_is_first_sentence": title_is_first_sentence,
                        "prefix_token_counts": prefix_token_counts,
                        "manual_sentences": manual_sentences,
                        "cached_sentences": cached_sentences,
                        "cached_token_lists": cached_token_lists,
                    }
                )

                if progress is not None:
                    progress.update(1)

        if progress is not None:
            progress.close()

        return preprocess_jobs, query_token_ids

    def _resolve_preprocess_workers(self, override: int | None) -> int:
        if override is not None:
            return max(0, int(override))

        env_value = os.getenv("OPEN_PROVENCE_PREPROCESS_WORKERS")
        if env_value:
            try:
                parsed = int(env_value)
            except ValueError:
                parsed = 0
            if parsed > 0:
                return parsed

        return _default_preprocess_workers()

    def _estimate_device_memory_bytes(self) -> int | None:
        override_gb = os.getenv("OPEN_PROVENCE_DEVICE_MEMORY_GB")
        if override_gb:
            try:
                parsed = float(override_gb)
            except ValueError:
                parsed = None
            else:
                if parsed > 0:
                    return int(parsed * (1024**3))

        device = getattr(self, "_runtime_device", None)
        if not isinstance(device, torch.device):
            return None

        if device.type == "cuda":
            try:
                index = device.index if device.index is not None else torch.cuda.current_device()
            except Exception:
                index = None
            if index is None:
                return None
            try:
                props = torch.cuda.get_device_properties(index)
            except Exception:
                return None
            total = getattr(props, "total_memory", None)
            return int(total) if total is not None else None

        return None

    def _auto_tune_preprocess_loader(
        self,
        *,
        total_jobs: int,
        inference_batch_size: int,
        current_workers: int,
        current_preprocess_batch: int,
        current_prefetch: int | None,
        workers_explicit: bool,
        batch_explicit: bool,
        prefetch_explicit: bool,
    ) -> tuple[int, int, int | None]:
        # NOTE: This helper encapsulates several heuristics that evolved from
        # manual benchmarking.  Adding a comment here keeps the expectations
        # close to the code, so future refactors know which behaviours must
        # stay stable (and are covered by tests).
        jobs_count = max(0, int(total_jobs))
        workers = max(0, int(current_workers))
        preprocess_batch = max(1, int(current_preprocess_batch))
        prefetch_factor = current_prefetch if prefetch_explicit else None

        if not workers_explicit:
            cpu_limit = max(0, _default_preprocess_workers())
            workers = min(workers or cpu_limit, cpu_limit)
            if jobs_count < 2_000:
                workers = 0
            elif workers == 0 and cpu_limit > 0:
                workers = min(cpu_limit, 4)
            if jobs_count:
                workers = min(workers, jobs_count)

        if not batch_explicit:
            device_bytes = self._estimate_device_memory_bytes()
            cap_from_device: int | None = None
            if device_bytes:
                device_gb = device_bytes / float(1024**3)
                if device_gb < 12:
                    cap_from_device = 64
                elif device_gb < 20:
                    cap_from_device = 128
                else:
                    cap_from_device = 192
            fallback_cap = min(96, max(32, inference_batch_size))
            target_cap = cap_from_device or fallback_cap
            preprocess_batch = min(preprocess_batch, target_cap)
            preprocess_batch = min(preprocess_batch, max(1, inference_batch_size))
            if jobs_count:
                preprocess_batch = min(preprocess_batch, jobs_count)

        if workers <= 0:
            workers = 0
        if workers == 0 and not prefetch_explicit:
            prefetch_factor = None
        elif workers > 0 and not prefetch_explicit:
            prefetch_factor = max(2, min(8, math.ceil(preprocess_batch / workers)))

        return workers, preprocess_batch, prefetch_factor

    def _run_preprocess_pipeline(
        self,
        jobs: list[dict[str, Any]],
        max_fragment_tokens: int,
        splitter: SentenceSplitter,
        show_progress: bool,
        strip_sentences: bool,
        *,
        respect_sentence_boundaries: bool,
    ) -> tuple[list[dict[str, Any]], float]:
        """Execute the preprocessing pipeline and return processed entries with timing."""

        preprocess_start = perf_counter()
        processed_entries = self._run_sequential_fragmentize(
            jobs,
            max_fragment_tokens=max_fragment_tokens,
            splitter=splitter,
            show_progress=show_progress,
            strip_sentences=strip_sentences,
            respect_sentence_boundaries=respect_sentence_boundaries,
        )
        preprocess_time = perf_counter() - preprocess_start
        return processed_entries, preprocess_time

    def _assemble_inference_inputs(
        self,
        preprocess_jobs: list[dict[str, Any]],
        processed_entries: list[dict[str, Any]],
        query_token_ids: list[list[int]],
        sep_token_ids: list[int],
    ) -> tuple[
        dict[tuple[int, int], dict[str, Any]],
        list[dict[str, Any]],
        dict[str, float],
        float,
    ]:
        """Convert processed entries into inference jobs and aggregate timing metrics."""

        contexts_info: dict[tuple[int, int], dict[str, Any]] = {}
        inference_jobs: list[dict[str, Any]] = []
        timing_totals = {
            "sentence_collect_seconds": 0.0,
            "sentence_normalize_seconds": 0.0,
            "tokenize_seconds": 0.0,
            "fragment_split_seconds": 0.0,
            "fragment_decode_seconds": 0.0,
        }

        def _consume_timing(payload: dict[str, Any], key: str) -> float:
            value = payload.pop(key, 0.0)
            if isinstance(value, (list, tuple)):
                value = sum(value)
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        assembly_start = perf_counter()
        for job, processed in zip(preprocess_jobs, processed_entries):
            job.pop("cached_sentences", None)
            job.pop("cached_token_lists", None)
            timing_totals["sentence_collect_seconds"] += _consume_timing(
                processed, "timing_sentence_collect"
            )
            timing_totals["sentence_normalize_seconds"] += _consume_timing(
                processed, "timing_sentence_normalize"
            )
            timing_totals["tokenize_seconds"] += _consume_timing(processed, "timing_tokenize")
            timing_totals["fragment_split_seconds"] += _consume_timing(
                processed, "timing_fragment_split"
            )
            timing_totals["fragment_decode_seconds"] += _consume_timing(
                processed, "timing_fragment_decode"
            )

            fragment_texts = processed.get("fragment_texts", [])
            sentence_indices = processed.get("fragment_sentence_index", [])
            fragment_indices = processed.get("fragment_fragment_index", [])
            global_indices = processed.get("fragment_global_index", [])
            token_id_lists = processed.get("fragment_token_ids", [])

            fragments: list[_FragmentRecord] = []
            for idx, text in enumerate(fragment_texts):
                tokens = list(token_id_lists[idx]) if idx < len(token_id_lists) else []
                fragments.append(
                    _FragmentRecord(
                        text=text,
                        sentence_index=int(sentence_indices[idx])
                        if idx < len(sentence_indices)
                        else 0,
                        fragment_index=int(fragment_indices[idx])
                        if idx < len(fragment_indices)
                        else 0,
                        global_index=int(global_indices[idx])
                        if idx < len(global_indices)
                        else idx,
                        token_length=len(tokens),
                        token_ids=tokens,
                    )
                )

            sentences: list[str] = processed.get("sentences", [])
            query_idx = job["query_idx"]
            context_idx = job["context_idx"]
            prefix_len = len(job.get("prefix_sentences", []))
            prefix_token_counts = job.get("prefix_token_counts", [])

            blocks = self._assemble_blocks_from_fragments(
                len(query_token_ids[query_idx]), len(sep_token_ids), fragments
            )

            contexts_info[(query_idx, context_idx)] = {
                "sentences": sentences,
                "fragments": fragments,
                "blocks": blocks,
                "prefix_length": prefix_len,
                "prefix_sentences": job.get("prefix_sentences", []),
                "prefix_token_counts": prefix_token_counts,
                "title_is_first_sentence": job.get("title_is_first_sentence", False),
                "original_text": job["context_text"],
                "raw_blocks": [],
            }

            for block_idx, block in enumerate(blocks):
                inference_jobs.append(
                    {
                        "query_idx": query_idx,
                        "context_idx": context_idx,
                        "block_idx": block_idx,
                        "texts": [fragment.text for fragment in block],
                    }
                )

        assembly_time = perf_counter() - assembly_start
        return contexts_info, inference_jobs, timing_totals, assembly_time

    def _run_inference_batches(
        self,
        inference_jobs: list[dict[str, Any]],
        batch_size: int,
        queries: list[str],
        query_token_ids: list[list[int]],
        contexts_info: dict[tuple[int, int], dict[str, Any]],
        *,
        show_inference_progress: bool,
        show_progress: bool,
    ) -> float:
        """Execute model inference over prepared jobs and attach raw predictions."""

        inference_time = 0.0
        total_inference_jobs = len(inference_jobs)
        progress_bar: Any | None = None

        if not total_inference_jobs:
            return inference_time

        if show_inference_progress:
            from tqdm import tqdm  # inline import to avoid dependency when unused

            total_batches = (total_inference_jobs + batch_size - 1) // batch_size
            progress_bar = tqdm(
                range(0, total_inference_jobs, batch_size),
                total=total_batches,
                desc="Model inference",
                unit="batch",
                leave=False,
            )
            batch_indices: Iterable[int] = progress_bar
        else:
            batch_indices = range(0, total_inference_jobs, batch_size)

        pad_token_raw = getattr(self.tokenizer, "pad_token_id", None)
        pad_token_id = int(pad_token_raw) if pad_token_raw is not None else 0

        for start in batch_indices:
            chunk_jobs = inference_jobs[start : start + batch_size]
            if not chunk_jobs:
                continue
            chunk_queries = [queries[job["query_idx"]] for job in chunk_jobs]
            chunk_context_texts = [job["texts"] for job in chunk_jobs]
            chunk_query_tokens = [query_token_ids[job["query_idx"]] for job in chunk_jobs]

            prepared_inputs: list[dict[str, Any]] = []
            ranges_per_job: list[list[tuple[int, int]]] = []

            for job_entry, query_tokens_entry in zip(chunk_jobs, chunk_query_tokens):
                block_fragments = contexts_info[
                    (job_entry["query_idx"], job_entry["context_idx"])
                ]["blocks"][job_entry["block_idx"]]
                (
                    input_ids_prepared,
                    attention_mask_prepared,
                    token_type_ids,
                    context_ranges,
                ) = self._prepare_block_inputs(
                    query_tokens_entry,
                    block_fragments,
                )
                prepared_inputs.append(
                    {
                        "input_ids": input_ids_prepared,
                        "attention_mask": attention_mask_prepared,
                        "token_type_ids": token_type_ids,
                    }
                )
                ranges_per_job.append(context_ranges)

            max_len = (
                max(len(entry["input_ids"]) for entry in prepared_inputs) if prepared_inputs else 0
            )
            input_tensor = torch.full(
                (len(prepared_inputs), max_len),
                pad_token_id,
                dtype=torch.long,
                device=self._runtime_device,
            )
            attention_tensor = torch.zeros(
                (len(prepared_inputs), max_len),
                dtype=torch.long,
                device=self._runtime_device,
            )
            token_type_tensor: torch.Tensor | None = (
                torch.zeros(
                    (len(prepared_inputs), max_len), dtype=torch.long, device=self._runtime_device
                )
                if any(entry.get("token_type_ids") for entry in prepared_inputs)
                else None
            )

            for tensor_idx, entry in enumerate(prepared_inputs):
                ids_list = entry["input_ids"]
                attn_list = entry["attention_mask"]
                seq_len = len(ids_list)
                if seq_len == 0:
                    continue
                input_tensor[tensor_idx, :seq_len] = torch.tensor(
                    ids_list,
                    dtype=torch.long,
                    device=self._runtime_device,
                )
                attention_tensor[tensor_idx, :seq_len] = torch.tensor(
                    attn_list if attn_list else [1] * seq_len,
                    dtype=torch.long,
                    device=self._runtime_device,
                )
                if token_type_tensor is not None:
                    type_ids = entry.get("token_type_ids") or [0] * seq_len
                    if len(type_ids) > seq_len:
                        type_ids = type_ids[:seq_len]
                    if len(type_ids) < seq_len:
                        type_ids = list(type_ids) + [type_ids[-1]] * (seq_len - len(type_ids))
                    token_type_tensor[tensor_idx, :seq_len] = torch.tensor(
                        type_ids,
                        dtype=torch.long,
                        device=self._runtime_device,
                    )

            infer_start = perf_counter()
            model_inputs = {
                "input_ids": input_tensor,
                "attention_mask": attention_tensor,
            }
            if token_type_tensor is not None:
                model_inputs["token_type_ids"] = token_type_tensor

            model_outputs = self.forward(return_dict=True, **model_inputs)
            inference_time += perf_counter() - infer_start

            ranking_logits = (
                self._extract_model_output(model_outputs, "ranking_logits").detach().cpu()
            )
            pruning_logits = (
                self._extract_model_output(model_outputs, "pruning_logits").detach().cpu()
            )

            if ranking_logits.dtype != torch.float32:
                ranking_logits = ranking_logits.to(dtype=torch.float32)
            if pruning_logits.dtype != torch.float32:
                pruning_logits = pruning_logits.to(dtype=torch.float32)

            for job_dict, raw_query, raw_contexts, ranges, rank_logits, prune_logits in zip(
                chunk_jobs,
                chunk_queries,
                chunk_context_texts,
                ranges_per_job,
                ranking_logits,
                pruning_logits,
            ):
                if rank_logits.ndim == 0 or rank_logits.numel() == 1:
                    ranking_score = torch.sigmoid(rank_logits.flatten())[0].item()
                else:
                    ranking_score = torch.sigmoid(rank_logits[..., 0]).item()

                pruning_probs = torch.softmax(prune_logits, dim=-1).numpy()
                if pruning_probs.ndim == 2 and pruning_probs.shape[1] == 2:
                    pruning_probs = pruning_probs[:, 1]
                elif pruning_probs.ndim == 1:
                    pruning_probs = pruning_probs
                else:
                    pruning_probs = pruning_probs.reshape(-1)

                contexts_info[(job_dict["query_idx"], job_dict["context_idx"])][
                    "raw_blocks"
                ].append(
                    (
                        job_dict["block_idx"],
                        OpenProvenceRawPrediction(
                            query=raw_query,
                            contexts=list(raw_contexts),
                            ranking_score=ranking_score,
                            pruning_probs=pruning_probs,
                            context_ranges=ranges,
                        ),
                    )
                )

        if progress_bar is not None:
            try:
                progress_bar.close()
            except Exception:  # pragma: no cover - harmless
                pass

            if show_progress:
                try:
                    progress_bar.write(
                        f"Model inference time: {inference_time:.2f}s "
                        f"({total_inference_jobs} blocks)"
                    )
                except Exception:  # pragma: no cover - best effort fallback
                    print(
                        f"[OpenProvenceModel] Model inference took {inference_time:.2f}s "
                        f"({total_inference_jobs} blocks)",
                        flush=True,
                    )

        return inference_time

    def _postprocess_contexts(
        self,
        queries: list[str],
        contexts: list[list[Any]],
        contexts_info: dict[tuple[int, int], dict[str, Any]],
        *,
        threshold: float,
        always_select_title: bool,
        use_best_reranker_score: bool,
        sentence_probability_groups_requested: bool,
        collect_sentence_texts: bool,
        first_line_as_title: bool,
        zero_score_when_empty: bool,
    ) -> tuple[
        list[list[str]],
        list[list[float | None]],
        list[list[float]],
        list[list[list[str]]] | None,
        list[list[list[str]]] | None,
        list[list[Any]],
        list[list[list[float]]] | None,
        float,
    ]:
        """Aggregate pruning outputs into user-facing structures."""

        post_start = perf_counter()
        pruned_contexts: list[list[str]] = []
        reranking_scores: list[list[float | None]] = []
        compression_rates: list[list[float]] = []
        if collect_sentence_texts:
            kept_sentences: list[list[list[str]]] | None = []
            removed_sentences: list[list[list[str]]] | None = []
        else:
            kept_sentences = None
            removed_sentences = None
        title_values: list[list[Any]] = []
        sentence_probability_groups: list[list[list[float]]] | None = (
            [] if sentence_probability_groups_requested else None
        )

        for query_idx, _ in enumerate(queries):
            query_pruned: list[str] = []
            query_scores: list[float | None] = []
            query_compression: list[float] = []
            query_kept: list[list[str]] | None = [] if collect_sentence_texts else None
            query_removed: list[list[str]] | None = [] if collect_sentence_texts else None
            query_titles: list[Any] = []
            query_sentence_probabilities: list[list[float]] | None = (
                [] if sentence_probability_groups is not None else None
            )

            for context_idx, context_entry in enumerate(contexts[query_idx]):
                info = contexts_info.get((query_idx, context_idx))
                prefix_sentences_value: Sequence[str] = ()
                if info:
                    raw_prefix = info.get("prefix_sentences", [])
                    if isinstance(raw_prefix, str):
                        prefix_sentences_value = (raw_prefix,)
                    elif isinstance(raw_prefix, Sequence):
                        prefix_sentences_value = tuple(str(item) for item in raw_prefix)
                if first_line_as_title and prefix_sentences_value:
                    if len(prefix_sentences_value) == 1:
                        fallback_title: Any = prefix_sentences_value[0]
                    else:
                        fallback_title = list(prefix_sentences_value)
                else:
                    fallback_title = None

                context_sentence_probs: list[float] | None = (
                    [] if sentence_probability_groups is not None else None
                )

                if not info or not info.get("fragments"):
                    query_pruned.append(context_entry)
                    query_scores.append(None)
                    query_compression.append(0.0)
                    if query_kept is not None:
                        query_kept.append([context_entry] if context_entry else [])
                    if query_removed is not None:
                        query_removed.append([])
                    query_titles.append(fallback_title)
                    if query_sentence_probabilities is not None:
                        query_sentence_probabilities.append(context_sentence_probs or [])
                    continue

                blocks = info["blocks"]
                raw_blocks = sorted(info["raw_blocks"], key=lambda x: x[0])

                if not blocks or not raw_blocks:
                    query_pruned.append(context_entry)
                    query_scores.append(None)
                    query_compression.append(0.0)
                    if query_kept is not None:
                        query_kept.append(info["sentences"])
                    if query_removed is not None:
                        query_removed.append([])
                    query_titles.append(fallback_title)
                    if context_sentence_probs is not None:
                        context_sentence_probs.extend([1.0] * len(info["sentences"]))
                    if query_sentence_probabilities is not None:
                        query_sentence_probabilities.append(context_sentence_probs or [])
                    continue

                fragment_scores: dict[int, list[float]] = defaultdict(list)
                ranking_score: float | None = None

                for (_, raw), block in zip(raw_blocks, blocks):
                    block_probs = raw.pruning_probs
                    ranges = raw.context_ranges
                    prefix_counts = contexts_info[(query_idx, context_idx)].get(
                        "prefix_token_counts", []
                    )

                    for fragment, (start, end) in zip(block, ranges):
                        offset = sum(prefix_counts[: fragment.sentence_index])
                        start = max(0, start - offset)
                        end = max(start, end - offset)
                        end = min(end, len(block_probs))
                        start = min(start, len(block_probs))
                        mean_prob = 1.0 if end <= start else float(block_probs[start:end].mean())
                        fragment_scores[fragment.global_index].append(mean_prob)

                    if raw.ranking_score is not None:
                        if use_best_reranker_score:
                            if ranking_score is None:
                                ranking_score = raw.ranking_score
                            else:
                                ranking_score = max(ranking_score, raw.ranking_score)
                        else:
                            if ranking_score is None:
                                ranking_score = raw.ranking_score

                sentence_scores: dict[int, list[float]] = defaultdict(list)
                for fragment in info["fragments"]:
                    if fragment.global_index in fragment_scores:
                        sentence_scores[fragment.sentence_index].extend(
                            fragment_scores[fragment.global_index]
                        )

                kept_sentence_texts: list[str] = []
                removed_sentence_texts: list[str] = []
                sentences = info["sentences"]
                prefix_len = info["prefix_length"]
                title_sentence_index: int | None = None
                sentence_keep_flags: list[bool] = []

                if always_select_title:
                    if prefix_len > 0:
                        title_sentence_index = 0
                    elif info.get("title_is_first_sentence") and len(sentences) > prefix_len:
                        title_sentence_index = prefix_len

                sentence_avg_probabilities: list[float] = []
                has_sentence_above_threshold = False
                for sentence_index in range(len(sentences)):
                    probabilities = sentence_scores.get(sentence_index)
                    avg_probability = float(np.mean(probabilities)) if probabilities else 0.0
                    avg_probability = max(0.0, min(avg_probability, 1.0))
                    sentence_avg_probabilities.append(avg_probability)
                    if avg_probability > threshold:
                        has_sentence_above_threshold = True

                force_keep_title = (
                    title_sentence_index is not None and has_sentence_above_threshold
                )

                for sentence_index in range(len(sentences)):
                    avg_probability = sentence_avg_probabilities[sentence_index]
                    keep_flag = avg_probability > threshold
                    if force_keep_title and sentence_index == title_sentence_index:
                        keep_flag = True

                    sentence_keep_flags.append(keep_flag)
                    if context_sentence_probs is not None:
                        context_sentence_probs.append(avg_probability)

                kept_sentence_texts = [
                    sentences[idx] for idx, keep in enumerate(sentence_keep_flags) if keep
                ]
                removed_sentence_texts = [
                    sentences[idx] for idx, keep in enumerate(sentence_keep_flags) if not keep
                ]

                content_kept_sentences = [
                    sentences[idx]
                    for idx, keep in enumerate(sentence_keep_flags)
                    if idx >= prefix_len and keep
                ]
                pruned_text = "".join(content_kept_sentences)
                original_text = info["original_text"]
                original_length = max(len(original_text), 1)
                compression = (len(original_text) - len(pruned_text)) / original_length * 100.0

                if zero_score_when_empty and not pruned_text.strip():
                    ranking_score = 0.0

                prefix_sentences_value = info.get("prefix_sentences", [])
                if prefix_sentences_value:
                    if len(prefix_sentences_value) == 1:
                        title_value = prefix_sentences_value[0]
                    else:
                        title_value = list(prefix_sentences_value)
                else:
                    title_value = None

                query_pruned.append(pruned_text)
                query_scores.append(ranking_score)
                query_compression.append(compression)
                if query_kept is not None:
                    query_kept.append(kept_sentence_texts)
                if query_removed is not None:
                    query_removed.append(removed_sentence_texts)
                query_titles.append(title_value)
                if query_sentence_probabilities is not None:
                    query_sentence_probabilities.append(context_sentence_probs or [])

            pruned_contexts.append(query_pruned)
            reranking_scores.append(query_scores)
            compression_rates.append(query_compression)
            if kept_sentences is not None and query_kept is not None:
                kept_sentences.append(query_kept)
            if removed_sentences is not None and query_removed is not None:
                removed_sentences.append(query_removed)
            title_values.append(query_titles)
            if (
                sentence_probability_groups is not None
                and query_sentence_probabilities is not None
            ):
                sentence_probability_groups.append(query_sentence_probabilities)

        post_time = perf_counter() - post_start
        return (
            pruned_contexts,
            reranking_scores,
            compression_rates,
            kept_sentences,
            removed_sentences,
            title_values,
            sentence_probability_groups,
            post_time,
        )

    def _apply_reordering(
        self,
        pruned_contexts: list[list[str]],
        reranking_scores: list[list[float | None]],
        compression_rates: list[list[float]],
        kept_sentences: list[list[list[str]]] | None,
        removed_sentences: list[list[list[str]]] | None,
        title_values: list[list[Any]],
        sentence_probability_groups: list[list[list[float]]] | None,
        *,
        top_k: int | None,
    ) -> tuple[
        list[list[str]],
        list[list[float | None]],
        list[list[float]],
        list[list[list[str]]] | None,
        list[list[list[str]]] | None,
        list[list[Any]],
        list[list[list[float]]] | None,
    ]:
        """Reorder contexts by reranker score and apply optional top-k truncation."""

        if not pruned_contexts:
            return (
                pruned_contexts,
                reranking_scores,
                compression_rates,
                kept_sentences,
                removed_sentences,
                title_values,
                sentence_probability_groups,
            )

        if top_k is None:
            effective_top_k = None
        else:
            effective_top_k = max(0, int(top_k))

        reordered_pruned: list[list[str]] = []
        reordered_scores: list[list[float | None]] = []
        reordered_compression: list[list[float]] = []
        reordered_kept: list[list[list[str]]] | None = [] if kept_sentences is not None else None
        reordered_removed: list[list[list[str]]] | None = (
            [] if removed_sentences is not None else None
        )
        reordered_titles: list[list[Any]] = []
        reordered_probs: list[list[list[float]]] | None = (
            [] if sentence_probability_groups is not None else None
        )

        for query_idx, scores in enumerate(reranking_scores):
            if not scores:
                reordered_pruned.append(pruned_contexts[query_idx])
                reordered_scores.append(scores)
                reordered_compression.append(compression_rates[query_idx])
                if reordered_kept is not None and kept_sentences is not None:
                    reordered_kept.append(kept_sentences[query_idx])
                if reordered_removed is not None and removed_sentences is not None:
                    reordered_removed.append(removed_sentences[query_idx])
                reordered_titles.append(title_values[query_idx])
                if reordered_probs is not None:
                    reordered_probs.append(
                        sentence_probability_groups[query_idx]
                        if sentence_probability_groups is not None
                        else []
                    )
                continue

            def _score_key(idx: int) -> float:
                value = scores[idx]
                if value is None:
                    return float("-inf")
                return float(value)

            ranking_indices = sorted(range(len(scores)), key=_score_key, reverse=True)

            if effective_top_k is None:
                limited_indices = ranking_indices
            else:
                limited_indices = ranking_indices[:effective_top_k]

            reordered_pruned.append([pruned_contexts[query_idx][idx] for idx in limited_indices])
            reordered_scores.append([scores[idx] for idx in limited_indices])
            reordered_compression.append(
                [compression_rates[query_idx][idx] for idx in limited_indices]
            )
            if reordered_kept is not None and kept_sentences is not None:
                reordered_kept.append([kept_sentences[query_idx][idx] for idx in limited_indices])
            if reordered_removed is not None and removed_sentences is not None:
                reordered_removed.append(
                    [removed_sentences[query_idx][idx] for idx in limited_indices]
                )
            reordered_titles.append([title_values[query_idx][idx] for idx in limited_indices])
            if reordered_probs is not None:
                reordered_probs.append(
                    [sentence_probability_groups[query_idx][idx] for idx in limited_indices]
                    if sentence_probability_groups is not None
                    else []
                )

        return (
            reordered_pruned,
            reordered_scores,
            reordered_compression,
            reordered_kept,
            reordered_removed,
            reordered_titles,
            reordered_probs if reordered_probs is not None else None,
        )

    def process(
        self,
        question: str | Sequence[str],
        context: str | Sequence[str] | Sequence[Sequence[str]],
        title: None | str | Sequence[str] | Sequence[Sequence[str]] = "first_sentence",
        first_line_as_title: bool = False,
        *,
        batch_size: int = 32,
        threshold: float | None = None,
        always_select_title: bool = False,
        reorder: bool = False,
        top_k: int | None = None,
        sentence_splitter: SentenceSplitter | Mapping[str, SentenceSplitter] | None = None,
        language: str | None = None,
        use_best_reranker_score: bool = True,
        zero_score_when_empty: bool = True,
        show_progress: bool = True,
        debug_messages: bool | Callable[[str], None] = False,
        enable_warnings: bool = True,
        strip_sentences: bool = False,
        respect_sentence_boundaries: bool = False,
        return_sentence_metrics: bool = False,
        return_sentence_texts: bool = False,
        show_inference_progress: bool | None = None,
        preprocess_workers: int | None = None,
        preprocess_batch_size: int | None = None,
        torch_dataloader_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prune long contexts by chunking them while preserving sentence boundaries.

        Args:
            question: Query text or list of queries.
            context: Context text(s) corresponding to each query.
            title: Optional title sentences to prepend. Use "first_sentence" to reuse the
                initial sentence per context (legacy default).
            first_line_as_title: When True, split the first non-empty line of each context and
                treat it as the title. Cannot be combined with explicit title overrides.
            batch_size: GPU batch size for inference.
            threshold: Pruning probability threshold. When omitted, the method first attempts to
                read ``self.config.default_threadshold`` (legacy spelling) from the checkpoint's
                ``config.json``. If that field is absent, the module constant
                ``DEFAULT_PROCESS_THRESHOLD`` (set to ``0.1``) is used.
            always_select_title: Force keeping title sentence.
            reorder: When True, sort contexts for each query by descending reranker score.
            top_k: When set along with ``reorder=True``, keep only the first ``top_k`` contexts
                per query after sorting.
            sentence_splitter: Callable that splits text into sentences or a mapping from language
                code to splitter. If omitted, the ``language`` parameter selects one of the built-in
                splitters.
            language: Language code used when choosing the default splitter or resolving a
                splitter mapping. When None, ``"auto"`` is assumed, which automatically handles
                Japanese and English text. Supported values remain ``"auto"``, ``"ja"`` (fast-bunkai),
                and ``"en"`` (NLTK Punkt with additional heuristics) for backwards compatibility.
            use_best_reranker_score: When True (default), store the highest reranker score among all
                processed blocks for each context. When False, keep the score from the first block
                only (original behaviour). If all sentences are discarded, the reranker score is set
                to 0.0 when ``zero_score_when_empty`` is enabled.
            zero_score_when_empty: When True (default), force the reranker score to ``0.0`` when
                the pruned context becomes empty after stripping whitespace. Disable to preserve the
                original score even when no sentences are kept.
            show_progress: When True, display progress bars for preprocessing and inference stages.
            debug_messages: Enable verbose timing diagnostics. When True, messages are logged via
                this module's logger. Provide a callable to redirect messages elsewhere. Timing
                summaries are also attached to the return payload.
            enable_warnings: Suppress warning output from dependencies when set to False.
            strip_sentences: When True, trim sentence text with `strip()` after splitting and filter
                out blank sentences (legacy behaviour). When False (default), preserve leading and
                trailing whitespace for downstream scoring.
            respect_sentence_boundaries: When True, keep each sentence produced by the splitter as
                a single fragment whenever it fits within the model's maximum token window, only
                falling back to token-level splitting when a sentence exceeds the allowed length.
            return_sentence_metrics: When True, include per-sentence probabilities in the
                response payload under ``sentence_probabilities``.
            return_sentence_texts: When True, include ``kept_sentences`` / ``removed_sentences``
                in the response payload. Defaults to False to minimise payload size.
            preprocess_workers: Number of DataLoader worker processes to use while fragmentizing
                contexts. When None, respects the ``OPEN_PROVENCE_PREPROCESS_WORKERS``
                environment variable and defaults to 0 (main-process preprocessing).
            preprocess_batch_size: Number of contexts processed per preprocessing batch. Defaults
                to ``batch_size`` when omitted.
            torch_dataloader_kwargs: Optional mapping forwarded directly to the preprocessing
                ``DataLoader`` to fine-tune worker behaviour (e.g., setting a custom
                ``worker_init_fn`` or pinning strategy).

        .. caution::
            Input shape determines how batching behaves. Passing ``question: str`` with
            ``context: List[str]`` is interpreted as *one* query paired with multiple
            documents. To batch distinct question–context pairs, provide
            ``question: List[str]`` and ``context: List[str]`` of equal length. If you
            supply ``context: List[List[str]]`` the inner lists are assumed to be
            pre-split sentences and the sentence splitter is skipped—use this form only
            when you have already segmented the text yourself.
        """

        progress_restore: Callable[[], None] | None = None
        original_progress_enabled = is_progress_bar_enabled()
        if show_progress and not original_progress_enabled:
            enable_progress_bar()
            progress_restore = disable_progress_bar
        elif not show_progress and original_progress_enabled:
            disable_progress_bar()
            progress_restore = enable_progress_bar

        try:
            batch_size = max(1, batch_size)
            threshold = self._resolve_process_threshold(threshold)

            start_total = perf_counter()

            splitter = OpenProvenceModel._resolve_sentence_splitter(
                self, sentence_splitter, language
            )

            debug_callback: Callable[[str], None] | None
            if isinstance(debug_messages, bool):
                debug_callback = LOGGER.info if debug_messages else None
            elif callable(debug_messages):
                debug_callback = debug_messages
            else:
                raise TypeError(
                    "debug_messages must be a bool or a callable that accepts a string"
                )

            def _log_debug(message: str) -> None:
                if debug_callback is not None:
                    debug_callback(message)

            if show_inference_progress is None:
                show_inference_progress = show_progress

            warnings_cm: contextlib.AbstractContextManager[Any]
            warnings_entered = False
            if enable_warnings:
                warnings_cm = contextlib.nullcontext()
            else:  # pragma: no cover - depends on caller preference
                warnings_cm = warnings.catch_warnings()
                warnings_cm.__enter__()
                warnings.simplefilter("ignore")
                warnings_entered = True

            preprocess_time = 0.0
            assembly_time = 0.0
            inference_time = 0.0
            post_time = 0.0
            timing_totals: dict[str, float] = {
                "sentence_collect_seconds": 0.0,
                "sentence_normalize_seconds": 0.0,
                "tokenize_seconds": 0.0,
                "fragment_split_seconds": 0.0,
                "fragment_decode_seconds": 0.0,
            }

            queries: list[str] = []
            contexts: list[list[Any]] = []
            structure = "str"
            preprocess_jobs: list[dict[str, Any]] = []
            query_token_ids: list[list[int]] = []
            contexts_info: dict[tuple[int, int], dict[str, Any]] = {}
            pruned_contexts: list[list[str]] = []
            reranking_scores: list[list[float | None]] = []
            compression_rates: list[list[float]] = []
            kept_sentences: list[list[list[str]]] | None = None
            removed_sentences: list[list[list[str]]] | None = None
            title_values: list[list[Any]] = []
            sentence_probability_groups: list[list[list[float]]] | None = None

            try:
                queries, contexts, structure = OpenProvenceModel._normalize_inputs(
                    self, question, context
                )
                contexts, titles = self._resolve_titles(
                    queries,
                    contexts,
                    title,
                    first_line_as_title=first_line_as_title,
                )
                if respect_sentence_boundaries:
                    max_fragment_tokens = max(16, self.max_length - 2)
                else:
                    max_fragment_tokens = max(16, self.max_length // 2)
                sep_token_ids = self.tokenizer.encode(
                    self.tokenizer.sep_token or "", add_special_tokens=False
                )

                preprocess_jobs, query_token_ids = self._build_preprocess_jobs(
                    queries,
                    contexts,
                    titles,
                    splitter,
                    strip_sentences=strip_sentences,
                    show_progress=show_progress,
                )

                resolved_workers = self._resolve_preprocess_workers(preprocess_workers)
                preprocess_batch = max(1, int(preprocess_batch_size or batch_size))

                dataset = _PreprocessDataset(
                    preprocess_jobs,
                    self.tokenizer,
                    splitter,
                    max_fragment_tokens,
                    strip_sentences,
                    respect_sentence_boundaries,
                )

                loader_kwargs: dict[str, Any] = {
                    "batch_size": preprocess_batch,
                    "shuffle": False,
                    "num_workers": resolved_workers,
                    "collate_fn": _preprocess_collate_fn,
                    "pin_memory": False,
                    "persistent_workers": resolved_workers > 0,
                }

                total_jobs = len(preprocess_jobs)
                workers_explicit = preprocess_workers is not None
                batch_explicit = preprocess_batch_size is not None
                prefetch_explicit = False

                if not workers_explicit and preprocess_workers is None:
                    env_workers_raw = os.getenv("OPEN_PROVENCE_PREPROCESS_WORKERS")
                    if env_workers_raw:
                        try:
                            workers_explicit = int(env_workers_raw) > 0
                        except ValueError:
                            workers_explicit = False

                if torch_dataloader_kwargs:
                    custom_kwargs = dict(torch_dataloader_kwargs)
                    if "num_workers" in custom_kwargs:
                        workers_explicit = True
                    if "batch_size" in custom_kwargs:
                        batch_explicit = True
                    if "prefetch_factor" in custom_kwargs:
                        prefetch_explicit = True
                    loader_kwargs.update(custom_kwargs)

                resolved_workers = int(loader_kwargs.get("num_workers", resolved_workers))
                preprocess_batch = int(loader_kwargs.get("batch_size", preprocess_batch))
                current_prefetch_raw = loader_kwargs.get("prefetch_factor")
                current_prefetch: int | None
                if isinstance(current_prefetch_raw, (int, float)):
                    current_prefetch = int(current_prefetch_raw)
                elif isinstance(current_prefetch_raw, str) and current_prefetch_raw.isdigit():
                    current_prefetch = int(current_prefetch_raw)
                else:
                    current_prefetch = None

                if "multiprocessing_context" in loader_kwargs:
                    loader_kwargs.pop("multiprocessing_context")

                (
                    resolved_workers,
                    preprocess_batch,
                    tuned_prefetch,
                ) = self._auto_tune_preprocess_loader(
                    total_jobs=total_jobs,
                    inference_batch_size=batch_size,
                    current_workers=resolved_workers,
                    current_preprocess_batch=preprocess_batch,
                    current_prefetch=current_prefetch,
                    workers_explicit=workers_explicit,
                    batch_explicit=batch_explicit,
                    prefetch_explicit=prefetch_explicit,
                )

                loader_kwargs["num_workers"] = resolved_workers
                loader_kwargs["batch_size"] = preprocess_batch
                loader_kwargs["persistent_workers"] = resolved_workers > 0

                if tuned_prefetch is not None:
                    loader_kwargs["prefetch_factor"] = tuned_prefetch
                elif not prefetch_explicit and "prefetch_factor" in loader_kwargs:
                    loader_kwargs.pop("prefetch_factor", None)

                loader = DataLoader(dataset, **loader_kwargs)

                if debug_callback is not None:
                    _log_debug(
                        "[OpenProvenceModel] "
                        f"preprocess_workers={resolved_workers} "
                        f"preprocess_batch={preprocess_batch} "
                        f"default_workers={_default_preprocess_workers()}"
                    )

                total_blocks_processed = 0

                loader_iter = iter(loader)
                shutdown_workers = getattr(loader_iter, "_shutdown_workers", None)

                try:
                    for jobs_batch, entries_batch in loader_iter:
                        if not jobs_batch:
                            continue

                        (
                            batch_contexts,
                            batch_inference_jobs,
                            batch_timing_totals,
                            batch_assembly,
                        ) = self._assemble_inference_inputs(
                            jobs_batch,
                            entries_batch,
                            query_token_ids,
                            sep_token_ids,
                        )

                        assembly_time += batch_assembly
                        preprocess_time += sum(batch_timing_totals.values())
                        for key, value in batch_timing_totals.items():
                            timing_totals[key] += value

                        for key, info in batch_contexts.items():
                            existing = contexts_info.get(key)
                            if existing is None:
                                contexts_info[key] = info
                                continue

                            existing_raw = existing.setdefault("raw_blocks", [])
                            existing_raw.extend(info.get("raw_blocks", []))

                        if not batch_inference_jobs:
                            continue

                        inference_time += self._run_inference_batches(
                            batch_inference_jobs,
                            batch_size,
                            queries,
                            query_token_ids,
                            contexts_info,
                            show_inference_progress=False,
                            show_progress=show_progress,
                        )

                        total_blocks_processed += len(batch_inference_jobs)
                finally:
                    if shutdown_workers is not None:
                        shutdown_workers()

                if show_progress and total_blocks_processed:
                    message = (
                        f"[OpenProvenceModel] Model inference time: {inference_time:.2f}s "
                        f"({total_blocks_processed} blocks)"
                    )
                    if debug_callback is None:
                        print(message, flush=True)
                    else:
                        _log_debug(message)

                (
                    pruned_contexts,
                    reranking_scores,
                    compression_rates,
                    kept_sentences,
                    removed_sentences,
                    title_values,
                    sentence_probability_groups,
                    post_time,
                ) = self._postprocess_contexts(
                    queries,
                    contexts,
                    contexts_info,
                    threshold=threshold,
                    always_select_title=always_select_title,
                    use_best_reranker_score=use_best_reranker_score,
                    sentence_probability_groups_requested=return_sentence_metrics,
                    collect_sentence_texts=return_sentence_texts,
                    first_line_as_title=first_line_as_title,
                    zero_score_when_empty=zero_score_when_empty,
                )
            finally:
                if warnings_entered:  # pragma: no cover - depends on caller preference
                    warnings_cm.__exit__(None, None, None)

            total_time = perf_counter() - start_total

            performance_trace = ProcessPerformanceTrace(
                preprocess_seconds=preprocess_time,
                assembly_seconds=assembly_time,
                inference_seconds=inference_time,
                postprocess_seconds=post_time,
                total_seconds=total_time,
                sentence_collect_seconds=timing_totals.get("sentence_collect_seconds", 0.0),
                sentence_normalize_seconds=timing_totals.get("sentence_normalize_seconds", 0.0),
                tokenize_seconds=timing_totals.get("tokenize_seconds", 0.0),
                fragment_split_seconds=timing_totals.get("fragment_split_seconds", 0.0),
                fragment_decode_seconds=timing_totals.get("fragment_decode_seconds", 0.0),
            )
            timing_summary = performance_trace.as_dict()

            timing_line = (
                "Timing: "
                f"preprocess={performance_trace.preprocess_seconds:.2f}s "
                f"[collect={performance_trace.sentence_collect_seconds:.2f}s "
                f"normalize={performance_trace.sentence_normalize_seconds:.2f}s "
                f"tokenize={performance_trace.tokenize_seconds:.2f}s "
                f"fragment_split={performance_trace.fragment_split_seconds:.2f}s "
                f"fragment_decode={performance_trace.fragment_decode_seconds:.2f}s] "
                f"assembly={performance_trace.assembly_seconds:.2f}s "
                f"inference={performance_trace.inference_seconds:.2f}s "
                f"postprocess={performance_trace.postprocess_seconds:.2f}s "
                f"total={performance_trace.total_seconds:.2f}s"
            )

            _log_debug(f"[OpenProvenceModel] {timing_line}")

            if reorder:
                (
                    pruned_contexts,
                    reranking_scores,
                    compression_rates,
                    kept_sentences,
                    removed_sentences,
                    title_values,
                    sentence_probability_groups,
                ) = self._apply_reordering(
                    pruned_contexts,
                    reranking_scores,
                    compression_rates,
                    kept_sentences,
                    removed_sentences,
                    title_values,
                    sentence_probability_groups,
                    top_k=top_k,
                )

            pruned_output: Any = pruned_contexts
            score_output: Any = reranking_scores
            compression_output: Any = compression_rates
            kept_output: Any = kept_sentences if kept_sentences is not None else None
            removed_output: Any = removed_sentences if removed_sentences is not None else None
            title_output: Any = title_values
            sentence_prob_output: Any = sentence_probability_groups

            if structure == "str" and pruned_contexts:
                pruned_output = pruned_contexts[0][0] if pruned_contexts[0] else ""
                score_output = reranking_scores[0][0] if reranking_scores[0] else None
                compression_output = compression_rates[0][0] if compression_rates[0] else 0.0
                if kept_sentences is not None:
                    kept_output = kept_sentences[0][0] if kept_sentences[0] else []
                if removed_sentences is not None:
                    removed_output = removed_sentences[0][0] if removed_sentences[0] else []
                title_output = title_values[0][0] if title_values[0] else None
                if (
                    sentence_probability_groups is not None
                    and sentence_probability_groups
                    and sentence_probability_groups[0]
                ):
                    sentence_prob_output = sentence_probability_groups[0][0]
            elif structure == "list" and pruned_contexts:
                pruned_output = pruned_contexts[0]
                score_output = reranking_scores[0]
                compression_output = compression_rates[0]
                if kept_sentences is not None:
                    kept_output = kept_sentences[0]
                if removed_sentences is not None:
                    removed_output = removed_sentences[0]
                title_output = title_values[0]
                if sentence_probability_groups is not None:
                    sentence_prob_output = (
                        sentence_probability_groups[0] if sentence_probability_groups else []
                    )

            result_payload = {
                "pruned_context": pruned_output,
                "reranking_score": score_output,
                "compression_rate": compression_output,
                "title": title_output,
                "timing": timing_summary,
                "performance_trace": performance_trace,
            }
            if kept_output is not None:
                result_payload["kept_sentences"] = kept_output
            if removed_output is not None:
                result_payload["removed_sentences"] = removed_output
            if sentence_prob_output is not None:
                result_payload["sentence_probabilities"] = sentence_prob_output

            return result_payload
        finally:
            if progress_restore is not None:
                progress_restore()


# Hugging Face integration -------------------------------------------------


class OpenProvenceForSequenceClassification(OpenProvenceModel):
    """Sequence classification wrapper compatible with transformers.AutoModel."""

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            **kwargs,
        )


class OpenProvenceForTokenClassification(OpenProvenceModel):
    """Token classification wrapper that exposes pruning logits."""

    def __init__(
        self,
        config: OpenProvenceConfig,
        *model_args: Any,
        device: str | torch.device | None = None,
        **model_kwargs: Any,
    ) -> None:
        super().__init__(config, *model_args, device=device, **model_kwargs)
        self.num_labels = config.num_pruning_labels

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ):
        effective_return_dict = return_dict if return_dict is not None else True

        base_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            return_dict=True,
            **kwargs,
        )

        classifier_output = cast(SequenceClassifierOutput, base_output)
        pruning_logits = cast(Tensor, getattr(classifier_output, "pruning_logits"))
        ranking_logits = cast(Tensor, getattr(classifier_output, "ranking_logits"))
        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = pruning_logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                if active_logits.numel() > 0:
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = torch.tensor(0.0, device=pruning_logits.device, requires_grad=True)
            else:
                loss = loss_fct(pruning_logits.view(-1, self.num_labels), labels.view(-1))

        if not effective_return_dict:
            output: tuple[torch.Tensor, ...] = (pruning_logits,)
            if loss is not None:
                return (loss,) + output
            return output

        logits_output = cast(FloatTensor, pruning_logits)
        loss_output: FloatTensor | None = None
        if loss is not None:
            loss_output = cast(FloatTensor, loss.to(dtype=logits_output.dtype))

        result = TokenClassifierOutput(
            loss=loss_output,
            logits=logits_output,
            hidden_states=classifier_output.hidden_states,
            attentions=classifier_output.attentions,
        )
        setattr(result, "ranking_logits", ranking_logits)
        return result


OpenProvenceEncoderConfig = OpenProvenceConfig
OpenProvenceEncoderForSequenceClassification = OpenProvenceForSequenceClassification
OpenProvenceEncoderForTokenClassification = OpenProvenceForTokenClassification

__all__ = [
    "OpenProvenceModel",
    "OpenProvenceRawPrediction",
    "OpenProvenceConfig",
    "OpenProvenceForSequenceClassification",
    "OpenProvenceForTokenClassification",
]
ContextItem: TypeAlias = str | Sequence[str]
ContextInput: TypeAlias = str | Sequence[ContextItem] | Sequence[Sequence[ContextItem]]
