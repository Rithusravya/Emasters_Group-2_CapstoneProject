import logging
import subprocess
import tempfile
import sqlite3
from pathlib import Path
from typing import Dict, Union
from src.evaluation.metrics import EVAL_DEVICE, EvaluationMetrics

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

class EvaluationEngine:
    """
    Single-example evaluation engine for Code Generation + RAG.
    A thin convenience wrapper around `EvaluationMetrics`: it exposes the same
    BLEU / CodeBERTScore / execution-accuracy metrics but for one
    prediction/reference pair at a time, instead of a batch.
    """
    def __init__(self, device: str = EVAL_DEVICE):
        self.device = device

    def calculate_bleu(self, prediction: str, reference: str) -> float:
        return EvaluationMetrics.compute_bleu([reference], [prediction])

    def calculate_bert_score(self, prediction: str, reference: str) -> float:
        return EvaluationMetrics.compute_bertscore([reference], [prediction], device=self.device)

    def execution_accuracy(self, generated_code: str, test_code: str = None) -> float:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                file = Path(tmp) / "solution.py"
                file.write_text(generated_code)
                result = subprocess.run(["python3", str(file)], timeout=10, capture_output=True)
                return 1.0 if result.returncode == 0 else 0.0
        except Exception as exc:
            logger.error(f"Execution error: {exc}")
            return 0.0

    def evaluate_code(self, generated_code: str, reference_code: str) -> Dict[str, float]:
        return {
            "BLEU": self.calculate_bleu(generated_code, reference_code),
            "CodeBERTScore": self.calculate_bert_score(generated_code, reference_code),
            "ExecutionAccuracy": self.execution_accuracy(generated_code)
        }

    def evaluate_sql(self, generated_sql: str, reference_sql: str) -> Dict[str, float]:
        generated = generated_sql.strip().lower()
        reference = reference_sql.strip().lower()
        return {
            "ExactMatchAccuracy": float(generated == reference),
            "BLEU": self.calculate_bleu(generated, reference),
            "CodeBERTScore": self.calculate_bert_score(generated, reference)
        }

    def evaluate_sql_response_accuracy(
        self,
        generated_sql: str,
        reference_sql: str,
        db_path: str
    ) -> Dict[str, Union[float, int, str]]:
        """
        Executes both generated and reference SQL queries against the SQLite database
        and compares the actual returned result sets. This is the true 'Response Accuracy'.
        """
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute(reference_sql)
            ref_results = cursor.fetchall()
            cursor.execute(generated_sql)
            gen_results = cursor.fetchall()
            conn.close()
            try:
                is_match = sorted(ref_results) == sorted(gen_results)
            except TypeError:
                is_match = ref_results == gen_results
            return {
                "ResponseAccuracy": float(is_match),
                "RefResultCount": len(ref_results),
                "GenResultCount": len(gen_results)
            }
        except Exception as e:
            logger.warning(f"SQL execution failed for DB {db_path}: {e}")
            return {"ResponseAccuracy": 0.0, "Error": str(e)}
