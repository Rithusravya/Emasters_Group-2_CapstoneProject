import logging
from typing import Any, Dict, List, Optional, Union

from src.evaluation.comparator import ModelComparator
from src.evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)

# Default sweep range: 0 acts as a "no retrieval" control so the sweep table
# also reports the RAG-vs-no-RAG gain at K=0, in addition to gains across K.
DEFAULT_TOPK_VALUES = [1, 2, 3, 5, 8]


class RAGEvaluator:

    def __init__(self, comparator: Optional[ModelComparator] = None):
        self.comparator = comparator or ModelComparator()

    # -------------------------------------------------------------------
    # Task 3.1 — measure RAG improvement
    # -------------------------------------------------------------------
    def generate_rag_predictions(self, rag_pipeline, queries: List[str], top_k: int = 3) -> List[str]:
        preds = []
        for query in queries:
            generated_text, _context = rag_pipeline.generate_with_rag(query, top_k=top_k)
            preds.append(generated_text)
        return preds

    def measure_rag_improvement(
        self,
        rag_pipeline,
        references: List[Union[str, Dict[str, Any]]],
        queries: List[str],
        base_preds: Optional[List[str]] = None,
        lora_preds: Optional[List[str]] = None,
        top_k: int = 3,
        test_cases: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        rag_preds = self.generate_rag_predictions(rag_pipeline, queries, top_k=top_k)

        eval_results = self.comparator.compare(
            references=references,
            base_preds=base_preds or [],
            lora_preds=lora_preds or [],
            rag_preds=rag_preds,
            test_cases=test_cases,
        )

        baseline_name = "LoRA_Model" if "LoRA_Model" in eval_results else "Base_Model"
        gain = self._compute_gain(eval_results.get(baseline_name, {}), eval_results.get("RAG_Pipeline", {}))

        return {
            "top_k": top_k,
            "rag_preds": rag_preds,
            "metrics": eval_results,
            "baseline_used_for_gain": baseline_name,
            "gain_over_baseline": gain,
        }

    @staticmethod
    def _compute_gain(baseline_metrics: Dict[str, float], rag_metrics: Dict[str, float]) -> Dict[str, float]:
        gain = {}
        for metric, rag_value in rag_metrics.items():
            base_value = baseline_metrics.get(metric)
            if base_value is None:
                continue
            gain[metric] = round(rag_value - base_value, 4)
        return gain

    # -------------------------------------------------------------------
    # Task 3.3 — top-K sweep with gain evaluation
    # -------------------------------------------------------------------
    def sweep_top_k(
        self,
        rag_pipeline,
        references: List[Union[str, Dict[str, Any]]],
        queries: List[str],
        baseline_preds: List[str],
        k_values: Optional[List[int]] = None,
        test_cases: Optional[List[str]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Recomputes RAG predictions and metrics at each K in `k_values`, and
        reports the gain over `baseline_preds` (typically the LoRA model's
        output on the same queries, i.e. the no-RAG baseline) at each K.

        Returns a dict keyed by K:
            {k: {"metrics": {...}, "gain_over_baseline": {...}}}
        so the best K per metric can be read off directly, e.g.:
            best_k = max(results, key=lambda k: results[k]["gain_over_baseline"]["BLEU"])
        """
        k_values = k_values or DEFAULT_TOPK_VALUES
        baseline_metrics = self._score_predictions(references, baseline_preds, test_cases)

        results: Dict[int, Dict[str, Any]] = {}
        for k in k_values:
            rag_preds = self.generate_rag_predictions(rag_pipeline, queries, top_k=k)
            rag_metrics = self._score_predictions(references, rag_preds, test_cases)
            gain = self._compute_gain(baseline_metrics, rag_metrics)
            results[k] = {"metrics": rag_metrics, "gain_over_baseline": gain, "rag_preds": rag_preds}
            logger.info(f"[top_k={k}] metrics={rag_metrics} gain_over_baseline={gain}")

        return results

    @staticmethod
    def _score_predictions(
        references: List[Union[str, Dict[str, Any]]],
        preds: List[str],
        test_cases: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        if not preds:
            return {}
        rouge_scores = EvaluationMetrics.compute_rouge(references, preds)
        metrics = {
            "BLEU": round(EvaluationMetrics.compute_bleu(references, preds), 4),
            "F1_Score": round(EvaluationMetrics.compute_f1(references, preds), 4),
            "ROUGE-1": round(rouge_scores["ROUGE-1"], 4),
            "ROUGE-L": round(rouge_scores["ROUGE-L"], 4),
        }
        if test_cases:
            metrics["Execution_Accuracy"] = round(
                EvaluationMetrics.evaluate_execution_accuracy(preds, test_cases), 4
            )
        return metrics

    @staticmethod
    def best_k(sweep_results: Dict[int, Dict[str, Any]], metric: str = "BLEU") -> Optional[int]:
        """Returns the K with the highest gain for `metric`, or None if unavailable."""
        candidates = {
            k: v["gain_over_baseline"].get(metric)
            for k, v in sweep_results.items()
            if metric in v.get("gain_over_baseline", {})
        }
        if not candidates:
            return None
        return max(candidates, key=candidates.get)

    def plot_gain_vs_k(self, sweep_results: Dict[int, Dict[str, Any]], metrics: Optional[List[str]] = None, save_path: str = "output/plots/topk_gain_sweep.png"):
        """Line plot of gain-over-baseline vs K for each metric, saved to `save_path`."""
        import matplotlib.pyplot as plt
        from pathlib import Path

        k_values = sorted(sweep_results.keys())
        if not k_values:
            logger.warning("No sweep results to plot.")
            return None

        all_metrics = metrics or sorted(sweep_results[k_values[0]]["gain_over_baseline"].keys())

        fig, ax = plt.subplots(figsize=(8, 5))
        for metric in all_metrics:
            ys = [sweep_results[k]["gain_over_baseline"].get(metric, 0.0) for k in k_values]
            ax.plot(k_values, ys, marker="o", label=metric)

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("top_k (retrieved passages)")
        ax.set_ylabel("Gain over no-RAG baseline")
        ax.set_title("RAG Improvement vs. top_k")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Top-k gain sweep plot saved to: {save_path}")
        return save_path
