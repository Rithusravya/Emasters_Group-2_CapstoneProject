"""
Step 8 — Embedding Generation. Task 3.2 requires embeddings from the small code-LM
itself for semantic search; a fast sentence-transformer embedder is offered as the
practical default for building large indices. Both return the same L2-normalized
(n_docs, dim) array shape so downstream indexing/retrieval code is embedder-agnostic.
"""
from __future__ import annotations
from typing import List

import numpy as np

from src.config_loader import CFG
from src.embeddings.embedding_utils import l2_normalize


def embed_with_sentence_transformer(docs: List[str], model_name: str | None = None,
                                     batch_size: int = 64) -> np.ndarray:
    """Fast, general-purpose embedder — good default for large retrieval indices."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name or CFG.model.embedding_name)
    return model.encode(docs, batch_size=batch_size, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True)


def embed_with_code_lm(docs: List[str], code_lm) -> np.ndarray:
    """Task 3.2 — use the small code-LM's own hidden states for semantic search."""
    vectors = code_lm.embed(docs)
    return l2_normalize(np.array(vectors, dtype="float32"))


def default_sentence_transformer_embedder(model_name: str | None = None):
    """Returns a callable(texts) -> np.ndarray, used to embed *query* text with the
    same embedder that built the index."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name or CFG.model.embedding_name)

    def embed(texts: List[str]) -> np.ndarray:
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    return embed
