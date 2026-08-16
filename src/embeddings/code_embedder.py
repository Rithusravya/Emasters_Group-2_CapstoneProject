"""
code_embedder.py

Generates dense embeddings for code, SQL, and natural language.

Author : Emasters Group-2
"""

from typing import List, Union
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config_loader import CFG


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeEmbedder:
    """
    Wrapper around SentenceTransformer for embedding generation.

    Example
    -------
    >>> embedder = CodeEmbedder()
    >>> vector = embedder.encode("Write a Python function")
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        normalize: bool = None,
    ):

        self.model_name = (
            model_name
            if model_name
            else CFG.embedding.model_name
        )

        self.device = (
            device
            if device
            else CFG.embedding.device
        )

        self.normalize = (
            normalize
            if normalize is not None
            else True
        )

        logger.info(f"Loading embedding model: {self.model_name}")

        self.model = SentenceTransformer(
            self.model_name,
            device="cpu",
        )

        logger.info("Embedding model loaded successfully.")

    # ---------------------------------------------------
    # Encode Single Text
    # ---------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding.

        Parameters
        ----------
        text : str

        Returns
        -------
        numpy.ndarray
        """

        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        return embedding

    # ---------------------------------------------------
    # Encode Batch
    # ---------------------------------------------------

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 4,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Encode multiple texts.

        Returns
        -------
        numpy.ndarray
            Shape:
            (num_samples, embedding_dimension)
        """

        if batch_size is None:
            batch_size = CFG.embedding.batch_size

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress_bar,
        )

        return embeddings

    # ---------------------------------------------------
    # Encode Documents
    # ---------------------------------------------------

    def encode_documents(
        self,
        documents: List[dict],
        field: str = "context",
    ):
        """
        Generate embeddings from document dictionaries.

        Parameters
        ----------
        documents : list
            List of dictionaries.

        field : str
            Which field should be embedded.

        Returns
        -------
        numpy.ndarray
        """

        texts = [
            doc.get(field, "")
            for doc in documents
        ]

        return self.encode_batch(texts)

    # ---------------------------------------------------
    # Query Embedding
    # ---------------------------------------------------

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Encode a user query.
        """

        return self.encode(query)

    # ---------------------------------------------------
    # Embedding Dimension
    # ---------------------------------------------------

    @property
    def embedding_dimension(self):
        """
        Returns embedding size.
        """

        return self.model.get_sentence_embedding_dimension()

    # ---------------------------------------------------
    # Similarity
    # ---------------------------------------------------

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Cosine similarity between two vectors.
        """

        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        return float(np.dot(embedding1, embedding2))

    # ---------------------------------------------------
    # Utility
    # ---------------------------------------------------

    def print_model_info(self):

        print("=" * 60)
        print("Embedding Model Information")
        print("=" * 60)
        print("Model :", self.model_name)
        print("Device:", self.device)
        print("Dimension:", self.embedding_dimension)
        print("Normalize:", self.normalize)
        print("=" * 60)


# ---------------------------------------------------
# Testing
# ---------------------------------------------------

if __name__ == "__main__":

    embedder = CodeEmbedder()

    embedder.print_model_info()

    sample = "Write a Python function to reverse a string."

    embedding = embedder.encode(sample)

    print()

    print("Embedding Shape :", embedding.shape)

    print()

    samples = [
        "Write SQL query",
        "Generate documentation",
        "Create commit message",
    ]

    vectors = embedder.encode_batch(samples)

    print("Batch Shape :", vectors.shape)