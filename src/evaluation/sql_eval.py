"""Spider-style SQL evaluation: exact-match response accuracy and
result-set execution accuracy, computed against SQLite databases.
"""

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SpiderSQLEvaluator:
    """Computes:
    1. Response Accuracy / Exact Match Accuracy
    2. Execution Accuracy
    """

    def __init__(self, database_root: str):
        self.database_root = Path(database_root)

    @staticmethod
    def normalize_sql(sql: str) -> str:
        """Normalizes SQL for exact-match comparison (lowercase, collapsed
        whitespace, no trailing semicolon)."""
        if not sql:
            return ""
        sql = re.sub(r"\s+", " ", sql.lower().strip())
        return sql.rstrip(";").strip()

    def response_accuracy(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Computes exact-match response accuracy across all records."""
        if not records:
            return {"total": 0, "correct": 0, "response_accuracy": 0.0}

        correct = 0
        for record in records:
            gold_sql = self.normalize_sql(record.get("gold_sql", ""))
            pred_sql = self.normalize_sql(record.get("pred_sql", ""))
            if gold_sql and gold_sql == pred_sql:
                correct += 1

        total = len(records)
        return {"total": total, "correct": correct, "response_accuracy": correct / total if total else 0.0}

    def execute_sql(self, db_path: Path, sql: str) -> Dict[str, Any]:
        """Executes a single SQL statement (read-only) and returns its result or error."""
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            conn.close()
            return {"success": True, "result": result, "error": None}
        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}

    @staticmethod
    def _results_match(gold_result: list, pred_result: list) -> bool:
        """Compares two result sets order-independently where possible."""
        try:
            return sorted(gold_result) == sorted(pred_result)
        except TypeError:
            return gold_result == pred_result

    def execution_accuracy(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Executes gold and predicted SQL for each record and compares result sets."""
        if not records:
            return {"total": 0, "correct": 0, "execution_accuracy": 0.0}

        correct = 0
        errors = 0

        for record in records:
            db_id = record.get("db_id", "")
            db_file = self.database_root / db_id / f"{db_id}.sqlite"

            if not db_file.exists():
                logger.warning(f"Database not found: {db_file}")
                errors += 1
                continue

            gold_result = self.execute_sql(db_file, record.get("gold_sql", ""))
            pred_result = self.execute_sql(db_file, record.get("pred_sql", ""))

            if not gold_result["success"]:
                errors += 1
                continue
            if not pred_result["success"]:
                continue

            if self._results_match(gold_result["result"], pred_result["result"]):
                correct += 1

        total = len(records)
        return {
            "total": total,
            "correct": correct,
            "execution_accuracy": correct / total if total else 0.0,
            "errors": errors,
        }

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Computes both response and execution accuracy for a batch of records."""
        response_metrics = self.response_accuracy(records)
        execution_metrics = self.execution_accuracy(records)

        return {
            "response_accuracy": response_metrics["response_accuracy"],
            "response_correct": response_metrics["correct"],
            "execution_accuracy": execution_metrics["execution_accuracy"],
            "execution_correct": execution_metrics["correct"],
            "total": response_metrics["total"],
            "errors": execution_metrics["errors"],
        }
