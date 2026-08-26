import logging
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ASTIndex:
    """
    Abstract Syntax Tree index for structural code/SQL pattern matching.
    Extracts key structural features (keywords, table names, function calls, etc.)
    """
    
    def __init__(self, language: str = "sql"):
        self.language = language
        self.inverted_index = defaultdict(set)  # feature -> set of doc_ids
        self.doc_store = {}  # doc_id -> metadata
        self.doc_counter = 0
        
        logger.info(f"Initializing AST index for language: {language}")

    def _extract_features(self, code: str) -> List[str]:
        """Extract structural features from code/SQL/MQL or natural language."""
        features = []
        if self.language == "mongodb":
            features.extend(self._extract_mongodb_features(code))
            # If no MQL features found, try NL feature extraction
            if not features:
                features.extend(self._extract_nl_mongo_features(code))
        elif self.language == "sql":
            features.extend(self._extract_sql_features(code))
        else:
            features.extend(self._extract_python_features(code))
        return features

    def _extract_mongodb_features(self, mql: str) -> List[str]:
        """Extract MongoDB/MQL-specific structural features."""
        features = []
        mql_lower = mql.lower()

        # Collection names: db.<collection>.
        collection_pattern = r'db\.([a-zA-Z_][a-zA-Z0-9_]*)\.'
        collections = re.findall(collection_pattern, mql_lower)
        for col in collections:
            features.append(f"collection:{col}")

        # MQL operations
        operations = [
            "find", "aggregate", "insertone", "insertmany",
            "updateone", "updatemany", "deleteone", "deletemany",
            "countdocuments", "estimatedocumentcount", "distinct",
            "sort", "limit", "skip", "group", "match", "project",
            "lookup", "unwind", "count"
        ]
        for op in operations:
            if op in mql_lower:
                features.append(f"op:{op}")

        # Aggregation pipeline stages
        stages = ["$match", "$group", "$project", "$sort", "$limit", "$skip",
                  "$lookup", "$unwind", "$count", "$addfields", "$set"]
        for stage in stages:
            if stage.lower() in mql_lower:
                features.append(f"stage:{stage}")

        # Common operators
        operators = ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in",
                     "$and", "$or", "$not", "$regex", "$exists"]
        for op in operators:
            if op.lower() in mql_lower:
                features.append(f"filter:{op}")

        return features

    def _extract_nl_mongo_features(self, text: str) -> List[str]:
        """Extract features from natural language questions that map to MongoDB patterns."""
        features = []
        text_lower = text.lower()

        # Intent keywords → MQL operations
        intent_map = {
            "how many": "op:count",
            "count": "op:count",
            "number of": "op:count",
            "total number of": "op:count",
            "find": "op:find",
            "get": "op:find",
            "show": "op:find",
            "list": "op:find",
            "select": "op:find",
            "average": "op:aggregate",
            "avg": "op:aggregate",
            "mean": "op:aggregate",
            "sum": "op:aggregate",
            "total": "op:aggregate",
            "maximum": "op:aggregate",
            "max": "op:aggregate",
            "minimum": "op:aggregate",
            "min": "op:aggregate",
            "group by": "op:aggregate",
            "sort": "op:sort",
            "order by": "op:sort",
            "top": "op:sort",
            "limit": "op:limit",
            "first": "op:limit",
            "join": "op:lookup",
            "distinct": "op:distinct",
            "unique": "op:distinct",
        }
        for phrase, feature in intent_map.items():
            if phrase in text_lower:
                features.append(feature)

        # Try to extract collection/entity names (simple heuristic)
        # Remove common question words and extract remaining nouns
        stopwords = {"how", "many", "what", "which", "who", "is", "are", "the",
                     "a", "an", "of", "in", "on", "for", "do", "we", "have",
                     "does", "that", "with", "and", "or", "to", "from", "by"}
        words = re.findall(r'[a-zA-Z_]+', text_lower)
        for word in words:
            if word not in stopwords and len(word) > 2:
                features.append(f"entity:{word}")

        return features
    
    def add(
        self, 
        code_snippets: List[str], 
        metadata_list: List[Dict[str, Any]],
        show_progress: bool = True
    ):
        """
        Add code snippets and their metadata to the AST index.
        
        Args:
            code_snippets: List of code/SQL strings
            metadata_list: List of metadata dicts for each snippet
            show_progress: Whether to show progress bar
        """
        if len(code_snippets) != len(metadata_list):
            raise ValueError("Number of code snippets must match number of metadata entries")
        
        iterator = zip(code_snippets, metadata_list)
        if show_progress:
            iterator = tqdm(
                iterator, 
                total=len(code_snippets),
                desc="Building AST index",
                unit="snippet"
            )
        
        for code, metadata in iterator:
            doc_id = self.doc_counter
            self.doc_counter += 1
            
            # Extract features
            features = self._extract_features(code)
            
            # Add to inverted index
            for feature in features:
                self.inverted_index[feature].add(doc_id)
            
            # Store metadata
            self.doc_store[doc_id] = metadata
        
        logger.info(f"✅ AST index built with {len(self.doc_store)} documents and {len(self.inverted_index)} unique features")
    
    def search(
        self, 
        query_code: str, 
        k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for documents matching the query's structural features.
        
        Args:
            query_code: Query code/SQL string
            k: Number of results to return
            
        Returns:
            List of (metadata, score) tuples, sorted by score descending
        """
        if not self.doc_store:
            logger.warning("AST index is empty, returning empty results")
            return []
        
        # Extract query features
        query_features = self._extract_features(query_code)
        
        if not query_features:
            logger.warning("No features extracted from query")
            return []
        
        # Score documents based on feature overlap
        doc_scores = defaultdict(float)
        
        for feature in query_features:
            if feature in self.inverted_index:
                matching_docs = self.inverted_index[feature]
                # Simple scoring: each matching feature adds 1 point
                for doc_id in matching_docs:
                    doc_scores[doc_id] += 1.0
        
        # Sort by score and return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for doc_id, score in sorted_docs:
            if doc_id in self.doc_store:
                results.append((self.doc_store[doc_id], score))
        
        return results
    
    def save(self, path: str):
        """Save AST index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "language": self.language,
                "inverted_index": dict(self.inverted_index),
                "doc_store": self.doc_store,
                "doc_counter": self.doc_counter
            }, f)
        
        logger.info(f"✅ AST index saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "ASTIndex":
        """Load AST index from disk."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"AST index not found at {path}")
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        instance = cls(language=data["language"])
        instance.inverted_index = defaultdict(set, data["inverted_index"])
        instance.doc_store = data["doc_store"]
        instance.doc_counter = data["doc_counter"]
        
        logger.info(f"✅ Loaded AST index with {len(instance.doc_store)} documents from {path}")
        return instance