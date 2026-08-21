import ast
import logging
import re
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)

# Lightweight SQL structural extraction. Deliberately regex-based (no extra
# dependency like sqlglot) since this project's corpus (Spider text-to-SQL)
# is well-formed, single-statement SQL.
_SQL_TABLE_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w]*)", re.IGNORECASE)
_SQL_AGG_FN_RE = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)
_SQL_CLAUSE_KEYWORDS = [
    "WHERE", "GROUP BY", "ORDER BY", "HAVING", "JOIN", "DISTINCT", "LIMIT", "UNION",
]


class ASTIndexer:
    """
    Builds a structural fingerprint for a corpus item so retrieval can be
    reranked/filtered on structure, not just embedding similarity.

    - language="python": uses the stdlib `ast` module to extract function
      and class names (as before).
    - language="sql": this project's corpus is Spider text-to-SQL, which
      contains no Python to parse via `ast` (this is why the AST store was
      previously ending up empty). Instead this extracts an analogous
      structural fingerprint for SQL: referenced tables, aggregate
      functions, and clause keywords.
    """

    def __init__(self, language: str = "python"):
        self.language = language

    def parse_structure(self, code: str) -> Dict[str, Any]:
        if self.language == "sql":
            return self._parse_sql_structure(code)
        return self._parse_python_structure(code)

    @staticmethod
    def _parse_python_structure(code: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            return {"functions": functions, "classes": classes, "status": "success"}
        except Exception as e:
            logger.warning(f"AST Parsing failed: {e}")
            return {"functions": [], "classes": [], "status": "failed"}

    @staticmethod
    def _parse_sql_structure(sql: str) -> Dict[str, Any]:
        if not sql or not isinstance(sql, str) or not sql.strip():
            return {"tables": [], "aggregates": [], "clauses": [], "status": "failed"}

        try:
            tables = sorted({m.group(1).lower() for m in _SQL_TABLE_RE.finditer(sql)})
            aggregates = sorted({m.group(1).upper() for m in _SQL_AGG_FN_RE.finditer(sql)})
            clauses = sorted({kw for kw in _SQL_CLAUSE_KEYWORDS if kw.upper() in sql.upper()})
            return {"tables": tables, "aggregates": aggregates, "clauses": clauses, "status": "success"}
        except Exception as e:
            logger.warning(f"SQL structural parsing failed: {e}")
            return {"tables": [], "aggregates": [], "clauses": [], "status": "failed"}

    @staticmethod
    def structure_tokens(structure: Dict[str, Any]) -> Set[str]:
        """Flattens a parsed structure dict into a token set for overlap scoring."""
        tokens: List[str] = []
        for key in ("functions", "classes", "tables", "aggregates", "clauses"):
            tokens.extend(str(t).lower() for t in structure.get(key, []))
        return set(tokens)
