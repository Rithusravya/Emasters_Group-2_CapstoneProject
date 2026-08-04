import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class ResultVisualizer:
    """Generates comparison charts saved to the specified output directory."""

    def __init__(self, output_dir: str = "output/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_comparison(
            self,
            results: Dict[str, Dict[str, float]],
            save_name: str = "model_comparison.png"
    ) -> Optional[Path]:
        """Plots grouped bar charts, splitting normalized scores from other metrics."""
        models = list(results.keys())
        if not models:
            print("⚠️ No models to compare.")
            return None

        all_metrics = sorted({m for scores in results.values() for m in scores})

        score_metrics = []
        other_metrics = []

        for metric in all_metrics:
            max_val = max(results[m].get(metric, 0.0) for m in models)
            if max_val <= 1.05:
                score_metrics.append(metric)
            else:
                other_metrics.append(metric)

        primary_save_path = None

        if score_metrics:
            n_metrics = len(score_metrics)
            fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.5), 6))
            x = np.arange(len(models))
            width = 0.8 / max(n_metrics, 1)
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']

            for i, metric in enumerate(score_metrics):
                values = [results[m].get(metric, 0.0) for m in models]
                ax.bar(x + i * width, values, width=width, label=metric, color=colors[i % len(colors)])

            ax.set_xticks(x + width * (n_metrics - 1) / 2)
            ax.set_xticklabels(models, rotation=15, ha='right')
            ax.set_ylabel("Score (0.0 - 1.0)")
            ax.set_title("Model Performance Comparison (Normalized Scores)")
            ax.set_ylim(0, 1.1)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            primary_save_path = self.output_dir / f"scores_{save_name}"
            plt.savefig(primary_save_path, bbox_inches="tight")
            plt.close()
            print(f"✅ Score chart saved to: {primary_save_path}")

        if other_metrics:
            n_other = len(other_metrics)
            fig, ax = plt.subplots(figsize=(max(8, n_other * 1.5), 6))
            x = np.arange(len(models))
            width = 0.8 / max(n_other, 1)
            colors = ['#e67e22', '#95a5a6', '#d35400', '#8e44ad']

            for i, metric in enumerate(other_metrics):
                values = [results[m].get(metric, 0.0) for m in models]
                ax.bar(x + i * width, values, width=width, label=metric, color=colors[i % len(colors)])

            ax.set_xticks(x + width * (n_other - 1) / 2)
            ax.set_xticklabels(models, rotation=15, ha='right')
            ax.set_ylabel("Value")
            ax.set_title("Model Performance Comparison (Other Metrics)")
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            other_save_path = self.output_dir / f"other_{save_name}"
            plt.savefig(other_save_path, bbox_inches="tight")
            plt.close()
            print(f"✅ Other metrics chart saved to: {other_save_path}")

            print("\n--- Other Metrics Summary ---")
            for metric in other_metrics:
                print(f"  {metric}:")
                for m in models:
                    print(f"    - {m}: {results[m].get(metric, 0.0)}")

        if not score_metrics and not other_metrics:
            print("⚠️ No valid metrics found to plot.")
            return None

        return primary_save_path if primary_save_path else other_save_path
