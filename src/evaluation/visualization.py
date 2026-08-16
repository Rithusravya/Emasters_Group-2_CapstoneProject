"""Generates grouped bar-chart comparisons of model metrics, saved as PNGs."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Palettes for the two chart types: normalized 0-1 scores vs. everything else.
SCORE_COLORS = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c", "#34495e"]
OTHER_COLORS = ["#e67e22", "#95a5a6", "#d35400", "#8e44ad"]


class ResultVisualizer:
    """Generates comparison charts saved to the specified output directory."""

    def __init__(self, output_dir: str = "output/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _split_metrics_by_scale(results: Dict[str, Dict[str, float]], models: List[str]):
        """Separates metric names into normalized 0-1 'scores' and everything else,
        so the two groups aren't squashed onto the same y-axis."""
        all_metrics = sorted({m for scores in results.values() for m in scores})
        score_metrics, other_metrics = [], []
        for metric in all_metrics:
            max_val = max(results[m].get(metric, 0.0) for m in models)
            (score_metrics if max_val <= 1.05 else other_metrics).append(metric)
        return score_metrics, other_metrics

    def _plot_grouped_bars(
        self,
        results: Dict[str, Dict[str, float]],
        models: List[str],
        metrics: List[str],
        colors: List[str],
        title: str,
        ylabel: str,
        save_path: Path,
        y_limit: Optional[float] = None,
    ) -> None:
        """Draws one grouped bar chart (one group of bars per model) and saves it to disk."""
        n_metrics = len(metrics)
        fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.5), 6))
        x = np.arange(len(models))
        width = 0.8 / max(n_metrics, 1)

        for i, metric in enumerate(metrics):
            values = [results[m].get(metric, 0.0) for m in models]
            ax.bar(x + i * width, values, width=width, label=metric, color=colors[i % len(colors)])

        ax.set_xticks(x + width * (n_metrics - 1) / 2)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if y_limit is not None:
            ax.set_ylim(0, y_limit)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def _print_other_metrics_summary(self, results: Dict[str, Dict[str, float]], models: List[str], metrics: List[str]) -> None:
        """Prints a plain-text summary of the non-normalized metrics (e.g. counts)."""
        print("\n--- Other Metrics Summary ---")
        for metric in metrics:
            print(f"  {metric}:")
            for m in models:
                print(f"    - {m}: {results[m].get(metric, 0.0)}")

    def plot_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        save_name: str = "model_comparison.png",
    ) -> Optional[Path]:
        """Plots grouped bar charts, splitting normalized scores from other metrics."""
        models = list(results.keys())
        if not models:
            print("⚠️ No models to compare.")
            return None

        score_metrics, other_metrics = self._split_metrics_by_scale(results, models)

        primary_save_path = None
        other_save_path = None

        if score_metrics:
            primary_save_path = self.output_dir / f"scores_{save_name}"
            self._plot_grouped_bars(
                results, models, score_metrics, SCORE_COLORS,
                title="Model Performance Comparison (Normalized Scores)",
                ylabel="Score (0.0 - 1.0)",
                save_path=primary_save_path,
                y_limit=1.1,
            )
            print(f"✅ Score chart saved to: {primary_save_path}")

        if other_metrics:
            other_save_path = self.output_dir / f"other_{save_name}"
            self._plot_grouped_bars(
                results, models, other_metrics, OTHER_COLORS,
                title="Model Performance Comparison (Other Metrics)",
                ylabel="Value",
                save_path=other_save_path,
            )
            print(f"✅ Other metrics chart saved to: {other_save_path}")
            self._print_other_metrics_summary(results, models, other_metrics)

        if not score_metrics and not other_metrics:
            print("⚠️ No valid metrics found to plot.")
            return None

        return primary_save_path if primary_save_path else other_save_path
