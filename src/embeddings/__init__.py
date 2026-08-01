"""
Embeddings Package

Provides utilities for generating dense vector embeddings
from source code, SQL, and natural language using
Sentence Transformers.

Modules
-------
code_embedder.py
    Main embedding model wrapper.

embedding_utils.py
    Helper functions for batching, saving, loading,
    normalization, and similarity computations.
"""

from .code_embedder import CodeEmbedder
from .embedding_utils import (
    save_embeddings,
    load_embeddings,
    cosine_similarity,
    normalize_embeddings,
)

__all__ = [
    "CodeEmbedder",
    "save_embeddings",
    "load_embeddings",
    "cosine_similarity",
    "normalize_embeddings",
]

__version__ = "1.0.0"