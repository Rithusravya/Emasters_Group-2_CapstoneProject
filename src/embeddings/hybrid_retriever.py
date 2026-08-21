import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.embeddings.ast_indexing import ASTIndexer

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z_][\w]*")


class HybridRetriever:
    """
    Wires the AST/structural index into retrieval instead of leaving it as a
    standalone artifact (Task 3.2). Retrieval works in two stages:

      1. Dense stage: pull `overfetch_factor * k` candidates from the FAISS
         semantic index (via `SemanticIndexManager.search`), same as before.
      2. Structural rerank stage: for each candidate, look up its
         precomputed structural fingerprint (function/class names for
         Python, tables/aggregates/clauses for SQL) and score its token
         overlap against the query's own tokens. Blend the normalized
         dense score with the structural overlap score and re-sort.

    The blended score is `alpha * dense_score + (1 - alpha) * structural_score`,
    so alpha=1.0 reduces to pure semantic search (the previous behavior),
    and this class stays a drop-in replacement for `SemanticIndexManager`
    wherever a `.search(query_embedding, k)` object is expected (e.g.
    `RAGPipeline`), while also accepting the raw query text when the caller
    has it, via an optional `query_text` kwarg.
    """

    def __init__(
        self,
        semantic_index: Any,
        ast_store: Optional[List[Dict[str, Any]]] = None,
        language: str = "sql",
        alpha: float = 0.7,
        overfetch_factor: int = 4,
    ):
        self.semantic_index = semantic_index
        self.language = language
        self.alpha = alpha
        self.overfetch_factor = max(1, overfetch_factor)
        self.ast_indexer = ASTIndexer(language=language)

        # Maps a metadata_store index -> structural token set, for O(1) lookup
        # during reranking. Built from a pre-computed ast_store (as produced
        # by the indexing notebook cell) if provided.
        self._structure_by_index: Dict[int, set] = {}
        if ast_store:
            self.load_ast_store(ast_store)

    def load_ast_store(self, ast_store: List[Dict[str, Any]]) -> None:
        for entry in ast_store:
            idx = entry.get("index")
            structure = entry.get("structure")
            if idx is None or structure is None:
                continue
            self._structure_by_index[idx] = ASTIndexer.structure_tokens(structure)

    def build_ast_store(self, corpus: List[Dict[str, Any]], text_field_candidates: Tuple[str, ...] = ("code", "SQL", "sql", "query")) -> List[Dict[str, Any]]:
        """
        Builds (and caches in-memory) the structural fingerprint for every
        item in the corpus, aligned by index with the metadata_store used by
        the semantic index. Returns the same list format the notebook
        previously pickled to `ast_store.pkl`, so it stays compatible with
        existing save/load code.
        """
        ast_store = []
        for idx, item in enumerate(corpus):
            code_text = self._extract_code(item, text_field_candidates)
            if not code_text:
                continue
            structure = self.ast_indexer.parse_structure(code_text)
            if structure.get("status") == "success":
                ast_store.append({"index": idx, "structure": structure, "code_preview": code_text[:100]})
                self._structure_by_index[idx] = ASTIndexer.structure_tokens(structure)
        logger.info(f"Built structural ({self.language}) index for {len(ast_store)}/{len(corpus)} corpus items.")
        return ast_store

    @staticmethod
    def _extract_code(item: Any, candidates: Tuple[str, ...]) -> str:
        if isinstance(item, dict):
            for key in candidates:
                if key in item and isinstance(item[key], str) and item[key].strip():
                    return item[key]
        elif isinstance(item, str):
            return item
        return ""

    @staticmethod
    def _query_tokens(query_text: str) -> set:
        return {t.lower() for t in _WORD_RE.findall(query_text or "")}

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        if not scores:
            return []
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-9:
            return [1.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    def _structural_overlap(self, query_tokens: set, candidate_idx: Optional[int]) -> float:
        if candidate_idx is None or not query_tokens:
            return 0.0
        structure_tokens = self._structure_by_index.get(candidate_idx)
        if not structure_tokens:
            return 0.0
        overlap = query_tokens & structure_tokens
        return len(overlap) / len(structure_tokens)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        query_text: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not self._structure_by_index or not query_text:
            # No structural signal available: behave exactly like plain
            # semantic search rather than silently degrading quality.
            return self.semantic_index.search(query_embedding, k=k)

        overfetch_k = min(k * self.overfetch_factor, max(k, self.semantic_index.index.ntotal))
        dense_results = self.semantic_index.search(query_embedding, k=overfetch_k)
        if not dense_results:
            return []

        # Recover each candidate's position in metadata_store to look up its
        # structural fingerprint (metadata dicts alone aren't hashable/unique).
        metadata_store = self.semantic_index.metadata_store
        candidate_indices = []
        for metadata, _score in dense_results:
            try:
                candidate_indices.append(metadata_store.index(metadata))
            except ValueError:
                candidate_indices.append(None)

        dense_scores = [score for _metadata, score in dense_results]
        dense_norm = self._normalize_scores(dense_scores)
        query_tokens = self._query_tokens(query_text)

        blended = []
        for (metadata, raw_score), dense_s, cand_idx in zip(dense_results, dense_norm, candidate_indices):
            structural_s = self._structural_overlap(query_tokens, cand_idx)
            final_score = self.alpha * dense_s + (1 - self.alpha) * structural_s
            blended.append((metadata, final_score, raw_score))

        blended.sort(key=lambda t: t[1], reverse=True)
        return [(metadata, raw_score) for metadata, _blended_score, raw_score in blended[:k]]

    # Kept for interface parity with SemanticIndexManager.
    def hybrid_search(self, query_embedding: np.ndarray, k: int = 5, query_text: Optional[str] = None):
        return self.search(query_embedding, k=k, query_text=query_text)
