from __future__ import annotations
import itertools
import math
from typing import Any, Dict, List

class CodeMetricsEvaluator:

    @staticmethod
    def calculate_codebleu(predictions: List[str], references: List[str], lang: str = "python") -> float:
        try:
            from codebleu import calc_codebleu
            res = calc_codebleu(references, predictions, lang=lang)
            return float(res["codebleu"])
        except ImportError:
            return 0.0

    @staticmethod
    def calculate_code_bertscore(predictions: List[str], references: List[str], lang: str = "python") -> float:
        try:
            from codebert_score import score
            _, _, f1 = score(cands=predictions, refs=references, lang=lang)
            return float(f1.mean())
        except ImportError:
            return 0.0

    @staticmethod
    def calculate_bleu(predictions: List[str], references: List[str]) -> float:
        try:
            from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
            refs = [[r.split()] for r in references]
            hyps = [p.split() for p in predictions]
            return float(corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1))
        except ImportError:
            return 0.0

    @staticmethod
    def calculate_rouge_l(predictions: List[str], references: List[str]) -> float:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]
            return sum(scores) / len(scores) if scores else 0.0
        except ImportError:
            return 0.0

    @staticmethod
    def exact_match_accuracy(predictions: List[str], references: List[str]) -> float:
        if not predictions:
            return 0.0
        matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
        return matches / len(predictions)

    @staticmethod
    def pass_at_k(n_samples: int, n_correct: int, k: int) -> float:
        """Unbiased pass@k estimator (Chen et al., 2021 / HumanEval)."""
        if n_samples - n_correct < k:
            return 1.0
        return 1.0 - math.prod(
            (n_samples - n_correct - i) / (n_samples - i) for i in range(k)
        )

    @staticmethod
    def evaluate_generation(predictions: List[str], references: List[str], lang: str = "python") -> Dict[str, Any]:
        """Convenience bundle: runs every applicable metric on one (predictions,
        references) pair and returns them as a single results dict."""
        return {
            "n": len(predictions),
            "bleu": CodeMetricsEvaluator.calculate_bleu(predictions, references),
            "rouge_l": CodeMetricsEvaluator.calculate_rouge_l(predictions, references),
            "codebleu": CodeMetricsEvaluator.calculate_codebleu(predictions, references, lang=lang),
            "code_bertscore": CodeMetricsEvaluator.calculate_code_bertscore(predictions, references, lang=lang),
            "exact_match": CodeMetricsEvaluator.exact_match_accuracy(predictions, references),
        }