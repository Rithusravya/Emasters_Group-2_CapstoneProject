import logging
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path
from evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluates RAG pipeline improvements and optimizes Top-K retrieval."""
    
    def __init__(self, comparator):
        self.comparator = comparator
        self.metrics = EvaluationMetrics()
        logger.info(f"RAGEvaluator using BERTScore/CodeBERTScore backbone: {getattr(comparator, 'bertscore_model', 'models/codebert-base')}")

    def measure_rag_improvement(self, rag_pipeline, references, queries, lora_preds, top_k=3):
        logger.info(f"Measuring RAG improvement over baseline LoRA (top_k={top_k})...")
        rag_preds = []
        for q in queries:
            out, _ = rag_pipeline.generate_with_rag(q, top_k=top_k)
            rag_preds.append(out)
            
        bertscore_model = getattr(self.comparator, "bertscore_model", "models/codebert-base")

        rag_metrics = {
            "BLEU": self.metrics.compute_bleu(references, rag_preds),
            "BERTScore": self.metrics.compute_bertscore(references, rag_preds, device=self.comparator.device, model_type=bertscore_model)
            # "Exact_Match_Accuracy": round(sum([self.metrics.compute_exact_match(r, p) for r, p in zip(references, rag_preds)]) / len(references), 4)
        }
        
        baseline_metrics = {
            "BLEU": self.metrics.compute_bleu(references, lora_preds),
            "BERTScore": self.metrics.compute_bertscore(references, lora_preds, device=self.comparator.device, model_type=bertscore_model)
            # "Exact_Match_Accuracy": round(sum([self.metrics.compute_exact_match(r, p) for r, p in zip(references, lora_preds)]) / len(references), 4)
        }
        
        gain = {k: round(rag_metrics[k] - baseline_metrics[k], 4) for k in rag_metrics}
        
        return {
            "metrics": {"RAG_Pipeline": rag_metrics, "Baseline_LoRA": baseline_metrics},
            "gain_over_baseline": gain,
            "baseline_used_for_gain": "Baseline_LoRA"
        }

    def sweep_top_k(self, rag_pipeline, references, queries, baseline_preds, k_values=[1, 2, 3, 5, 8], verbose=True):
        sweep_results = {}
        
        # Pre-compute baseline metrics once to save time
        baseline_bleu = self.metrics.compute_bleu(references, baseline_preds)
        baseline_rouge = self.metrics.compute_rouge(references, baseline_preds).get("ROUGE-1", 0.0)
        
        for k in k_values:
            logger.info(f"Sweeping top_k={k}...")
            preds = []
            for q in queries:
                out, _ = rag_pipeline.generate_with_rag(q, top_k=k)
                preds.append(out)
                
            # Use BLEU and ROUGE-1 for sweeping 
            bleu = self.metrics.compute_bleu(references, preds)
            rouge1 = self.metrics.compute_rouge(references, preds).get("ROUGE-1", 0.0)
            
            metrics = {"BLEU": bleu, "ROUGE-1": rouge1}
            gain = {
                "BLEU": round(bleu - baseline_bleu, 4),
                "ROUGE-1": round(rouge1 - baseline_rouge, 4)
            }
            
            sweep_results[k] = {"metrics": metrics, "gain_over_baseline": gain, "predictions": preds}

        if verbose:
            print("\n" + self.format_sweep_table(sweep_results))

        return sweep_results

    @staticmethod
    def format_sweep_table(sweep_results: Dict[int, Dict[str, Any]]) -> str:
        ks = list(sweep_results.keys())
        if not ks:
            return "(no sweep results to display)"

        columns = [
            ("Top-K", lambda k, r: str(k)),
            ("BLEU", lambda k, r: f"{r['metrics'].get('BLEU', 0.0):.4f}"),
            ("ROUGE-1", lambda k, r: f"{r['metrics'].get('ROUGE-1', 0.0):.4f}"),
            ("BLEU Gain", lambda k, r: f"{r['gain_over_baseline'].get('BLEU', 0.0):+.4f}"),
            ("ROUGE-1 Gain", lambda k, r: f"{r['gain_over_baseline'].get('ROUGE-1', 0.0):+.4f}"),
        ]

        rows = [[render(k, sweep_results[k]) for _, render in columns] for k in ks]
        headers = [name for name, _ in columns]

        widths = [
            max(len(headers[i]), *(len(row[i]) for row in rows))
            for i in range(len(columns))
        ]

        def fmt_row(cells):
            return "| " + " | ".join(c.center(w) for c, w in zip(cells, widths)) + " |"

        separator = "+-" + "-+-".join("-" * w for w in widths) + "-+"

        lines = ["Top-K Retrieval Sweep Results", separator, fmt_row(headers), separator]
        lines += [fmt_row(row) for row in rows]
        lines.append(separator)
        return "\n".join(lines)

    def best_k(self, sweep_results, metric="BLEU"):
        best_k_val = -999
        best_k = None
        for k, res in sweep_results.items():
            val = res["gain_over_baseline"].get(metric, -999)
            if val > best_k_val:
                best_k_val = val
                best_k = k
        return best_k if best_k is not None else list(sweep_results.keys())[0]

    def plot_gain_vs_k(self, sweep_results, save_path="output/plots/topk_gain_sweep.png"):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ks = list(sweep_results.keys())
        bleu_gains = [sweep_results[k]["gain_over_baseline"].get("BLEU", 0) for k in ks]
        
        plt.figure(figsize=(8, 5))
        plt.plot(ks, bleu_gains, marker='o', linestyle='-', color='b')
        plt.title("RAG Gain over Baseline by Top-K")
        plt.xlabel("Top-K Retrieved Contexts")
        plt.ylabel("BLEU Score Gain")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"✅ Saved Top-K sweep plot to {save_path}")
