"""
Compatibility shim: legacy imports for OpenProvence Hugging Face helpers.

All functionality now resides in ``modeling_open_provence_standalone``. This module keeps the
old import path working for downstream tooling that still references
``open_provence.modeling_open_provence_transformers``.
"""

from __future__ import annotations

from .modeling_open_provence_standalone import (
    OpenProvenceConfig,
    OpenProvenceEncoderConfig,
    OpenProvenceEncoderForSequenceClassification,
    OpenProvenceEncoderForTokenClassification,
    OpenProvenceForSequenceClassification,
    OpenProvenceForTokenClassification,
)

__all__ = [
    "OpenProvenceConfig",
    "OpenProvenceForSequenceClassification",
    "OpenProvenceForTokenClassification",
    "OpenProvenceEncoderConfig",
    "OpenProvenceEncoderForSequenceClassification",
    "OpenProvenceEncoderForTokenClassification",
]
