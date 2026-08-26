import logging
import faiss
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import pickle

logger = logging.getLogger(__name__)

class SemanticIndex:
    """FAISS-based semantic vector index for CPU."""
    
    def __init__(self, embedding_dim: int, index_type: str = "Flat"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.metadata_store = []  # Store metadata for each vector
        
        logger.info(f"Initializing FAISS index: dim={embedding_dim}, type={index_type}")
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index based on type."""
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for normalized vectors
        elif self.index_type == "IVF":
            # For larger datasets, use IVF (Inverted File Index)
            nlist = 100  # number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def add(
        self, 
        embeddings: torch.Tensor, 
        metadata_list: List[Dict[str, Any]],
        show_progress: bool = True
    ):
        """
        Add embeddings and their metadata to the index.
        
        Args:
            embeddings: Tensor of shape (num_vectors, embedding_dim)
            metadata_list: List of metadata dicts for each vector
            show_progress: Whether to show progress bar
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Convert to numpy for FAISS
        embeddings_np = embeddings.numpy().astype(np.float32)
        
        # Train index if needed (for IVF)
        if self.index_type == "IVF" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings_np)
        
        # Add vectors with progress bar
        if show_progress:
            logger.info(f"Adding {len(embeddings_np)} vectors to FAISS index...")
        
        self.index.add(embeddings_np)
        self.metadata_store.extend(metadata_list)
        
        logger.info(f"✅ Index now contains {self.index.ntotal} vectors")
    
    def search(
        self, 
        query_embedding: torch.Tensor, 
        k: int = 5,
        show_progress: bool = False
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            k: Number of results to return
            show_progress: Whether to show progress (only useful for batch queries)
            
        Returns:
            List of (metadata, score) tuples, sorted by score descending
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, returning empty results")
            return []
        
        # Convert to numpy and ensure correct shape
        query_np = query_embedding.numpy().astype(np.float32)
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_np, k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata_store):
                results.append((self.metadata_store[idx], float(score)))
        
        return results
    
    def save(self, path: str):
        """Save index and metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path.with_suffix(".faiss")
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = path.with_suffix(".metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata_store, f)
        
        logger.info(f"✅ Semantic index saved to {index_path}")
    
    @classmethod
    def load(cls, path: str) -> "SemanticIndex":
        """Load index and metadata from disk."""
        path = Path(path)
        
        index_path = path.with_suffix(".faiss")
        metadata_path = path.with_suffix(".metadata.pkl")
        
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Index files not found at {path}")
        
        # Load FAISS index
        index = faiss.read_index(str(index_path))
        embedding_dim = index.d
        
        # Create instance
        instance = cls(embedding_dim=embedding_dim, index_type="Flat")
        instance.index = index
        
        # Load metadata
        with open(metadata_path, "rb") as f:
            instance.metadata_store = pickle.load(f)
        
        logger.info(f"✅ Loaded semantic index with {index.ntotal} vectors from {path}")
        return instance