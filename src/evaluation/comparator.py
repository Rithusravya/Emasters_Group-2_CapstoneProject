import logging
from typing import List, Dict, Any, Union
from src.evaluation.metrics import EVAL_DEVICE, EvaluationMetrics

logger = logging.getLogger(__name__)


class ModelComparator:
    """Compares metrics across Base, LoRA, and RAG pipelines."""

    def __init__(self, device: str = EVAL_DEVICE):
        self.device = device

    def compare(
            self,
            references: List[Union[str, Dict[str, Any]]],
            base_preds: List[str],
            lora_preds: List[str],
            rag_preds: List[str],
            test_cases: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        pipelines = {
            "Base_Model": base_preds,
            "LoRA_Model": lora_preds,
            "RAG_Pipeline": rag_preds
        }

        results = {}
        for name, preds in pipelines.items():
            if not preds:
                continue

            bleu_score = EvaluationMetrics.compute_bleu(references, preds)
            bert_score = EvaluationMetrics.compute_bertscore(references, preds, device=self.device)
            f1_score = EvaluationMetrics.compute_f1(references, preds)
            rouge_scores = EvaluationMetrics.compute_rouge(references, preds)

            metrics_dict = {
                "BLEU": round(bleu_score, 4),
                "CodeBERTScore": round(bert_score, 4),
                "F1_Score": round(f1_score, 4),
                "ROUGE-1": round(rouge_scores["ROUGE-1"], 4),
                "ROUGE-L": round(rouge_scores["ROUGE-L"], 4)
            }

            if test_cases:
                exec_acc = EvaluationMetrics.evaluate_execution_accuracy(preds, test_cases)
                metrics_dict["Execution_Accuracy"] = round(exec_acc, 4)

            results[name] = metrics_dict

        return results
