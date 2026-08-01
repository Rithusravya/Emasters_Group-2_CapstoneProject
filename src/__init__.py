"""
Emasters Group-2 Capstone Project

A Retrieval-Augmented Generation (RAG) based Code Generation System.

Modules
-------
data        : Dataset loading and preprocessing
embeddings  : Code embedding generation
indexing    : Semantic and AST indexing
rag         : Retrieval-Augmented Generation pipeline
generators  : Code, SQL, documentation, and commit generation
evaluation  : Metrics and visualization
"""

__version__ = "1.0.0"
__author__ = "Emasters Group-2"

__all__ = [
    "data",
    "embeddings",
    "indexing",
    "rag",
    "generators",
    "evaluation",
]