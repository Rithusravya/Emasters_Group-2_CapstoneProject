from __future__ import annotations
import json
import os
from typing import Any, Dict
import matplotlib
matplotlib.use("Agg")  # safe for headless/notebook execution
import matplotlib.pyplot as plt
import numpy as np

class ResultsVisualizer:

    @staticmethod
    def save_metrics(metrics: Dict[str, Any], output_path: str = "outputs/metrics/results.json") -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

    @staticmethod
    def generate_chart(
        comparison: Dict[str, Dict[str, float]],
        output_dir: str = "outputs/plots",
        filename: str = "model_comparison.png",
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        archs = list(comparison.keys())
        codebleu = [comparison[a].get("CodeBLEU", 0.0) for a in archs]
        exec_acc = [comparison[a].get("ExecAccuracy", 0.0) for a in archs]

        x = np.arange(len(archs))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width / 2, codebleu, width, label="CodeBLEU", color="#1f77b4")
        ax.bar(x + width / 2, exec_acc, width, label="Execution Accuracy", color="#ff7f0e")

        ax.set_ylabel("Score")
        ax.set_title("Step 13 & 14: Evaluation & Model Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(archs, rotation=15, ha="right")
        ax.set_ylim(0, 1.0)
        ax.legend()

        chart_path = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300)
        plt.close(fig)
        return chart_path