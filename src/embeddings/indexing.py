import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class SemanticIndexManager:
    def __init__(self, embedding_dim: int, index_type: str = "Flat"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = self._build_index(embedding_dim, index_type)

        # Stores raw code/metadata items corresponding to indexed vector IDs.
        self.metadata_store: List[Dict[str, Any]] = []

    @staticmethod
    def _build_index(embedding_dim: int, index_type: str):
        if index_type == "Flat":
            # Inner Product index, for normalized (cosine-similarity) vectors.
            return faiss.IndexFlatIP(embedding_dim)
        if index_type == "L2":
            return faiss.IndexFlatL2(embedding_dim)
        raise ValueError(f"Unsupported index_type: {index_type}. Use 'Flat' or 'L2'.")

    @staticmethod
    def _format_metadata_item(item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return item
        return {"code": item if isinstance(item, str) else str(item)}

    def add_codes(
        self,
        embeddings: np.ndarray,
        corpus: Union[List[str], List[Dict[str, Any]], Any],
    ) -> None:
        if len(embeddings) == 0:
            logger.warning("Empty embeddings array passed. Skipping FAISS index population.")
            return

        # FAISS requires float32, contiguous memory.
        embeddings_faiss = np.ascontiguousarray(embeddings.astype(np.float32))
        self.index.add(embeddings_faiss)

        if isinstance(corpus, list):
            self.metadata_store.extend(self._format_metadata_item(item) for item in corpus)
        else:
            self.metadata_store.append(self._format_metadata_item(corpus))

        logger.info(
            f"Successfully added {len(embeddings_faiss)} items to FAISS index. "
            f"Total vectors indexed: {self.index.ntotal}"
        )

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if self.index.ntotal == 0:
            logger.warning("Search attempted on an empty FAISS index.")
            return []

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        query_faiss = np.ascontiguousarray(query_embedding.astype(np.float32))

        # Don't ask FAISS for more neighbors than exist in the index.
        k_actual = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_faiss, k_actual)

        return [
            (self.metadata_store[idx], float(score))
            for idx, score in zip(indices[0], distances[0])
            if idx != -1 and idx < len(self.metadata_store)
        ]

    def hybrid_search(self, query_embedding: np.ndarray, k: int = 5):
        return self.search(query_embedding, k=k)

    def save(self, index_path: Union[str, Path], metadata_path: Optional[Union[str, Path]] = None) -> None:
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        logger.info(f"FAISS index saved to {index_path}")

    def load(self, index_path: Union[str, Path]) -> None:
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")

        self.index = faiss.read_index(str(index_path))
        self.embedding_dim = self.index.d
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors from {index_path}")
