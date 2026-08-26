import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm

from vectorDB.embedder import HFEmbedder
from vectorDB.semantic_index import SemanticIndex
from vectorDB.ast_index import ASTIndex
from vectorDB.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

class IndexManager:
    """
    Manages the complete indexing pipeline: embedding, semantic indexing, AST indexing,
    and hybrid retrieval. Handles save/load and auto-detection.
    """
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        language: str = "sql",
        semantic_weight: float = 0.7,
        ast_weight: float = 0.3,
        save_dir: str = "data/indices"
    ):
        self.embedding_model = embedding_model
        self.language = language
        self.semantic_weight = semantic_weight
        self.ast_weight = ast_weight
        self.save_dir = Path(save_dir)
        
        self.embedder = None
        self.semantic_index = None
        self.ast_index = None
        self.hybrid_retriever = None
        
        logger.info(f"IndexManager initialized: model={embedding_model}, lang={language}")
    
    def build_indices(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        code_field: Optional[str] = None,
        batch_size: int = 32,
        force_rebuild: bool = False
    ):
        """
        Build both semantic and AST indices from documents.
        
        Args:
            documents: List of document dicts
            text_field: Field name for natural language text
            code_field: Field name for code/SQL (if different from text_field)
            batch_size: Batch size for embedding
            force_rebuild: Whether to rebuild even if indices exist
        """
        # Check if indices already exist
        if not force_rebuild and self._indices_exist():
            logger.info("Existing indices detected. Loading...")
            self.load_indices()
            return
        
        logger.info(f"Building indices for {len(documents)} documents...")
        
        # Initialize embedder
        self.embedder = HFEmbedder(model_name=self.embedding_model)
        
        # Extract texts and codes
        texts = [doc.get(text_field, "") for doc in documents]
        codes = [doc.get(code_field, doc.get(text_field, "")) for doc in documents] if code_field else texts
        
        # Build semantic index
        logger.info("\n=== Building Semantic Index ===")
        embeddings = self.embedder.encode(
            texts, 
            batch_size=batch_size,
            show_progress=True
        )
        
        embedding_dim = embeddings.shape[1]
        self.semantic_index = SemanticIndex(embedding_dim=embedding_dim)
        self.semantic_index.add(embeddings, documents, show_progress=True)
        
        # Build AST index
        logger.info("\n=== Building AST Index ===")
        self.ast_index = ASTIndex(language=self.language)
        self.ast_index.add(codes, documents, show_progress=True)
        
        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            semantic_index=self.semantic_index,
            ast_index=self.ast_index,
            embedder=self.embedder,
            semantic_weight=self.semantic_weight,
            ast_weight=self.ast_weight
        )
        
        # Save indices
        self.save_indices()
        
        logger.info("\n✅ All indices built and saved successfully!")
    
    def _indices_exist(self) -> bool:
        """Check if all index files exist."""
        semantic_path = self.save_dir / "semantic_index.faiss"
        semantic_meta = self.save_dir / "semantic_index.metadata.pkl"
        ast_path = self.save_dir / "ast_index.pkl"
        
        return semantic_path.exists() and semantic_meta.exists() and ast_path.exists()
    
    def save_indices(self):
        """Save all indices to disk."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.semantic_index:
            self.semantic_index.save(self.save_dir / "semantic_index")
        
        if self.ast_index:
            self.ast_index.save(self.save_dir / "ast_index.pkl")
        
        logger.info(f"✅ Indices saved to {self.save_dir}")
    
    def load_indices(self):
        """Load all indices from disk."""
        if not self._indices_exist():
            raise FileNotFoundError(f"No indices found at {self.save_dir}")
        
        # Load embedder
        self.embedder = HFEmbedder(model_name=self.embedding_model)
        
        # Load semantic index
        self.semantic_index = SemanticIndex.load(self.save_dir / "semantic_index")
        
        # Load AST index
        self.ast_index = ASTIndex.load(self.save_dir / "ast_index.pkl")
        
        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            semantic_index=self.semantic_index,
            ast_index=self.ast_index,
            embedder=self.embedder,
            semantic_weight=self.semantic_weight,
            ast_weight=self.ast_weight
        )
        
        logger.info("✅ All indices loaded successfully!")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search using hybrid retriever.
        
        Args:
            query: Query string
            k: Number of results
            
        Returns:
            List of metadata dicts
        """
        if self.hybrid_retriever is None:
            raise RuntimeError("Indices not initialized. Call build_indices() or load_indices() first.")
        
        results = self.hybrid_retriever.search(query, k=k)
        return [metadata for metadata, score in results]