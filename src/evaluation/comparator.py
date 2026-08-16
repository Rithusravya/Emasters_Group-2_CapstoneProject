import logging
from typing import Any, Dict, List, Union

from src.evaluation.metrics import EVAL_DEVICE, EvaluationMetrics

logger = logging.getLogger(__name__)


class ModelComparator:
    def __init__(self, device: str = EVAL_DEVICE):
        self.device = device

    def compare(
        self,
        references: List[Union[str, Dict[str, Any]]],
        base_preds: List[str],
        lora_preds: List[str],
        rag_preds: List[str],
        test_cases: List[str] = None,
    ) -> Dict[str, Dict[str, float]]:

        pipelines = {
            "Base_Model": base_preds,
            "LoRA_Model": lora_preds,
            "RAG_Pipeline": rag_preds,
        }

        results = {}
        for name, preds in pipelines.items():
            if not preds:
                continue
            results[name] = self._compute_metrics(references, preds, test_cases)

        return results

    def _compute_metrics(
        self,
        references: List[Union[str, Dict[str, Any]]],
        preds: List[str],
        test_cases: List[str] = None,
    ) -> Dict[str, float]:
        rouge_scores = EvaluationMetrics.compute_rouge(references, preds)

        metrics = {
            "BLEU": round(EvaluationMetrics.compute_bleu(references, preds), 4),
            "CodeBERTScore": round(
                EvaluationMetrics.compute_bertscore(references, preds, device=self.device), 4
            ),
            "F1_Score": round(EvaluationMetrics.compute_f1(references, preds), 4),
            "ROUGE-1": round(rouge_scores["ROUGE-1"], 4),
            "ROUGE-L": round(rouge_scores["ROUGE-L"], 4),
        }

        if test_cases:
            metrics["Execution_Accuracy"] = round(
                EvaluationMetrics.evaluate_execution_accuracy(preds, test_cases), 4
            )

        return metrics
