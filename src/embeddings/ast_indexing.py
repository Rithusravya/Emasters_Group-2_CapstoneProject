import ast
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ASTIndexer:
    def __init__(self, language: str = "python"):
        self.language = language

    def parse_structure(self, code: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            return {"functions": functions, "classes": classes, "status": "success"}
        except Exception as e:
            logger.warning(f"AST Parsing failed: {e}")
            return {"functions": [], "classes": [], "status": "failed"}
