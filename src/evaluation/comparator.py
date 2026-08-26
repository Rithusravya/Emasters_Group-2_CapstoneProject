import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


def _default_bertscore_model() -> str:
    local_dir = Path("models/codebert-base")
    if local_dir.is_dir() and (local_dir / "config.json").exists():
        return str(local_dir.resolve())
    return "microsoft/codebert-base"


class ModelComparator:
    """
    Compares generation outputs across 4 categories:
    1. Base_Model (Small Code LM)
    2. LoRA_Model (Fine-tuned)
    3. RAG_Pipeline (Retrieval Augmented)
    4. LLM_Baseline (OpenAI/Gemini API)
    """
    def __init__(self, device: str = "cpu", bertscore_model: Optional[str] = None):
        self.device = device
        self.bertscore_model = bertscore_model or _default_bertscore_model()
        logger.info(f"ModelComparator using BERTScore/CodeBERTScore backbone: {self.bertscore_model}")
        self.metrics = EvaluationMetrics()

    def compare(
        self,
        references: List[str],
        base_preds: List[str],
        lora_preds: List[str],
        rag_preds: Optional[List[str]] = None,
        llm_preds: Optional[List[str]] = None,
        test_cases: Optional[List[str]] = None,
        task_type: str = "sql" # "sql" or "doc"
    ) -> Dict[str, Dict[str, float]]:
        
        results = {}
        categories = {
            "Base_Model": base_preds,
            "LoRA_Model": lora_preds
        }
        if rag_preds: categories["RAG_Pipeline"] = rag_preds
        if llm_preds: categories["LLM_Baseline"] = llm_preds

        for name, preds in categories.items():
            logger.info(f"Calculating metrics for {name}...")
            cat_metrics = {}
            
            # Text-based metrics (BLEU, ROUGE, BERTScore)
            cat_metrics["BLEU"] = self.metrics.compute_bleu(references, preds)
            rouge_scores = self.metrics.compute_rouge(references, preds)
            cat_metrics.update(rouge_scores)
            cat_metrics["BERTScore"] = self.metrics.compute_bertscore(
                references, preds, device=self.device, model_type=self.bertscore_model
            )
            
            # Exact & Response Accuracy
            exact_matches = [self.metrics.compute_exact_match(r, p) for r, p in zip(references, preds)]
            cat_metrics["Exact_Match_Accuracy"] = round(sum(exact_matches) / len(exact_matches), 4) if exact_matches else 0.0
            
            resp_acc = [self.metrics.compute_response_accuracy(r, p) for r, p in zip(references, preds)]
            cat_metrics["Response_Accuracy"] = round(sum(resp_acc) / len(resp_acc), 4) if resp_acc else 0.0

            # SQL Execution Accuracy (if task is SQL and test cases/DBs are provided)
            if task_type == "sql" and test_cases:
                pass 

            results[name] = cat_metrics

        return results
