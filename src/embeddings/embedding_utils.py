"""
Shared helpers for the embedding stage (Step 8): corpus loading and vector
normalization, used by both the raw-text and AST embedders.
"""
from __future__ import annotations

import json
from typing import List

import numpy as np

from src.config_loader import CFG


def load_corpus(jsonl_path: str, text_field: str = "code", limit: int | None = None) -> List[str]:
    limit = limit or CFG.rag.codeparrot_subset_size
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs.append(row[text_field])
    return docs


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
