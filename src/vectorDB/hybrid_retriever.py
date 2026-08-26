import json
import logging
from typing import List, Dict, Any, Tuple
import torch

logger = logging.getLogger(__name__)


def _stable_doc_key(metadata: Dict[str, Any]) -> str:
    try:
        return json.dumps(metadata, sort_keys=True, default=str)
    except TypeError:
        # Fallback for anything truly unhashable/unserializable
        return str(sorted(metadata.items(), key=lambda kv: kv[0]))

class HybridRetriever:
    
    def __init__(
        self, 
        semantic_index, 
        ast_index, 
        embedder,
        semantic_weight: float = 0.7,
        ast_weight: float = 0.3
    ):
        self.semantic_index = semantic_index
        self.ast_index = ast_index
        self.embedder = embedder
        self.semantic_weight = semantic_weight
        self.ast_weight = ast_weight
        
        logger.info(f"Hybrid retriever initialized: semantic={semantic_weight}, ast={ast_weight}")

    def search(
            self,
            query: str,
            k: int = 5,
            query_type: str = "natural_language"
    ) -> List[Tuple[Dict[str, Any], float]]:
        # Semantic search
        query_embedding = self.embedder.encode(query, show_progress=False)
        semantic_results = self.semantic_index.search(query_embedding, k=k * 2)

        # AST search (use query as code)
        ast_results = self.ast_index.search(query, k=k * 2)

        combined = {}  # doc_id -> {"score": float, "metadata": dict}

        # Normalize scores to [0, 1] range for fair combination
        if semantic_results:
            max_sem_score = max(score for _, score in semantic_results)
            min_sem_score = min(score for _, score in semantic_results)
            sem_range = max_sem_score - min_sem_score if max_sem_score != min_sem_score else 1.0
            for metadata, score in semantic_results:
                doc_id = _stable_doc_key(metadata)
                normalized_score = (score - min_sem_score) / sem_range
                if doc_id not in combined:
                    combined[doc_id] = {"score": 0.0, "metadata": metadata}
                combined[doc_id]["score"] += normalized_score * self.semantic_weight

        if ast_results:
            max_ast_score = max(score for _, score in ast_results)
            min_ast_score = min(score for _, score in ast_results)
            ast_range = max_ast_score - min_ast_score if max_ast_score != min_ast_score else 1.0
            for metadata, score in ast_results:
                doc_id = _stable_doc_key(metadata)
                normalized_score = (score - min_ast_score) / ast_range
                if doc_id not in combined:
                    combined[doc_id] = {"score": 0.0, "metadata": metadata}
                combined[doc_id]["score"] += normalized_score * self.ast_weight

        # Sort by combined score
        sorted_docs = sorted(combined.items(), key=lambda x: x[1]["score"], reverse=True)[:k]

        results = [(item["metadata"], item["score"]) for _, item in sorted_docs]
        return results