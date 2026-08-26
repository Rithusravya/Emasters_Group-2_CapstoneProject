import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

class ResultVisualizer:
    """Visualizes evaluation metrics across Base, LoRA, RAG, and LLM categories."""
    
    def __init__(self, output_dir: str = "output/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")

    def plot_comparison(self, eval_results: Dict[str, Dict[str, float]], save_name: str = "model_comparison.png", bertscore_model_label: str = None):
        # Flatten dictionary for seaborn
        data = []
        for model_name, metrics in eval_results.items():
            for metric_name, value in metrics.items():
                data.append({"Model": model_name, "Metric": metric_name, "Score": value})
                
        df = pd.DataFrame(data)
        
        if df.empty:
            logger.warning("No data to plot.")
            return

        plt.figure(figsize=(14, 8))
        sns.barplot(x="Metric", y="Score", hue="Model", data=df)
        
        plt.title("Comprehensive Model Comparison: Base vs LoRA vs RAG vs LLM", fontsize=16, fontweight='bold')
        if bertscore_model_label:
            plt.suptitle(f"BERTScore backbone: {bertscore_model_label}", fontsize=9, y=0.93, color="gray")
        plt.ylabel("Score", fontsize=12)
        plt.xlabel("Evaluation Metric", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Model Category", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"✅ Saved comparison plot to {save_path}")

    def plot_radar_chart(self, eval_results: Dict[str, Dict[str, float]], save_name: str = "radar_comparison.png"):
        """Generates a radar chart for a holistic view of model capabilities."""
        import numpy as np
        
        categories = list(next(iter(eval_results.values())).keys())
        N = len(categories)
        if N == 0: return

        # Compute the angle for each axis
        angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        for model_name, metrics in eval_results.items():
            values = list(metrics.values())
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        plt.title("Model Capability Radar", size=15, y=1.1, fontweight='bold')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Saved radar chart to {save_path}")