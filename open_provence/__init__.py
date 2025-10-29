"""
Query-dependent text pruning and reranking for efficient RAG pipelines.

This module provides functionality for pruning irrelevant content from documents
based on queries, with optional reranking capabilities.
"""

from __future__ import annotations

from .data_collator import OpenProvenceDataCollator
from .data_structures import (
    OpenProvenceConfig,
    OpenProvenceOnlyOutput,
    OpenProvenceOutput,
    RerankingOpenProvenceOutput,
)
from .encoder import OpenProvenceEncoder
from .losses import OpenProvenceLoss
from .trainer import OpenProvenceTrainer

# Import runner module at the end to avoid circular imports
# It will be imported after other modules are initialized

__all__ = [
    "OpenProvenceConfig",
    "RerankingOpenProvenceOutput",
    "OpenProvenceOutput",
    "OpenProvenceOnlyOutput",
    "OpenProvenceEncoder",
    "OpenProvenceTrainer",
    "OpenProvenceLoss",
    "OpenProvenceDataCollator",
    "runner",
]

# Import runner after other modules are initialized
from . import runner
