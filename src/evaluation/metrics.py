<<<<<<< Updated upstream
"""Semantic, structural, and execution metrics for generated code and SQL.
=======
from __future__ import annotations
import math
from typing import Any, Dict, List
from codebleu import calc_codebleu
from nltk.translate.bleu_score import (
                SmoothingFunction,
                corpus_bleu,
            )
>>>>>>> Stashed changes

Implements small, dependency-light versions of BLEU, CodeBERTScore, token F1,
ROUGE, and execution/response accuracy so the pipeline doesn't need heavy
external scoring libraries.
"""

import ast
import logging
import os
import re
import sqlite3
import subprocess
import tempfile
from collections import Counter
from math import exp, log
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

# CodeBERT is forced onto CPU by default so the notebook runs anywhere without
# a GPU. Flip this to "cuda" (or pass device="cuda" to compute_bertscore)
# once you move to a machine with a GPU available.
EVAL_DEVICE = "cpu"


class EvaluationMetrics:
    """Provides semantic, structural, and execution metrics for generated code."""

    # Cached CodeBERT tokenizer/model so repeated calls don't reload weights.
    _codebert_tokenizer = None
    _codebert_model = None
    _codebert_device = None


    @staticmethod
<<<<<<< Updated upstream
    def extract_code_str(item: Union[str, Dict[str, Any], Any]) -> str:
        """Extracts a code/text string safely from a dict, string, or dataset item."""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            candidate_keys = ["code", "SQL", "sql", "canonical_solution", "query", "text"]
            for key in candidate_keys:
                if key in item and isinstance(item[key], str):
                    return item[key]
            return str(item)
        return str(item)

    # -------------------------------------------------------------------------
    # 1. Text & Structural Metrics: BLEU
    # -------------------------------------------------------------------------
    @staticmethod
    def _ngram_counts(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    @classmethod
    def _sentence_bleu(cls, ref_tokens: List[str], cand_tokens: List[str], max_n: int = 4) -> float:
        """Computes smoothed sentence-level BLEU with a brevity penalty."""
        if not cand_tokens or not ref_tokens:
            return 0.0

        precisions = []
        for n in range(1, max_n + 1):
            cand_ngrams = cls._ngram_counts(cand_tokens, n)
            if not cand_ngrams:
                precisions.append(0.0)
                continue
            ref_ngrams = cls._ngram_counts(ref_tokens, n)
            overlap = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in cand_ngrams.items())
            total = sum(cand_ngrams.values())
            # +1 smoothing avoids zero precision (and thus zero BLEU) on short candidates.
            precisions.append((overlap + 1) / (total + 1))

        geo_mean = exp(sum(log(p) for p in precisions) / len(precisions))
        ref_len, cand_len = len(ref_tokens), len(cand_tokens)
        brevity_penalty = 1.0 if cand_len > ref_len else exp(1 - ref_len / cand_len)
        return brevity_penalty * geo_mean

    @classmethod
    def compute_bleu(cls, references: List[Union[str, Dict[str, Any]]], predictions: List[str]) -> float:
        """Averages sentence-level BLEU across all reference/prediction pairs."""
        if not references or not predictions:
            return 0.0

        scores = [
            cls._sentence_bleu(cls.extract_code_str(ref_item).strip().split(), pred.strip().split())
            for ref_item, pred in zip(references, predictions)
        ]
        return float(sum(scores) / len(scores)) if scores else 0.0

    # -------------------------------------------------------------------------
    # 2. Semantic Representation: CodeBERTScore
    # -------------------------------------------------------------------------
    @classmethod
    def _load_codebert(cls, model_name: str, device: str):
        """Loads (and caches) the CodeBERT tokenizer/model for the given device."""
        if cls._codebert_model is not None and cls._codebert_tokenizer is not None and cls._codebert_device == device:
            return cls._codebert_tokenizer, cls._codebert_model

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("`torch` and `transformers` are required for CodeBERTScore.") from exc

        logger.info(f"Loading CodeBERT model '{model_name}' on device='{device}' ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = cls._load_codebert_weights(model_name)

        model.to(device)
        model.eval()
        cls._codebert_tokenizer = tokenizer
        cls._codebert_model = model
        cls._codebert_device = device
        return tokenizer, model

    @staticmethod
    def _load_codebert_weights(model_name: str):
        """Loads CodeBERT weights, preferring safetensors and explaining the
        torch>=2.6 requirement if the legacy `.bin` checkpoint path is blocked.
        """
        try:
            from transformers import AutoModel
            return AutoModel.from_pretrained(model_name, use_safetensors=True)
        except Exception as safetensors_exc:
            try:
                import torch
                torch_major, torch_minor = (int(x) for x in torch.__version__.split(".")[:2])
                torch_is_recent_enough = (torch_major, torch_minor) >= (2, 6)
            except Exception:
                torch_is_recent_enough = False

            if torch_is_recent_enough:
                raise

            raise RuntimeError(
                f"Could not load '{model_name}' with safetensors weights "
                f"({safetensors_exc}), and this checkpoint requires the legacy "
                "torch.load path. `transformers` blocks that path unless "
                "torch>=2.6 (see CVE-2025-32434). Fix by running "
                "`pip install -U torch` (>=2.6) in this environment, then "
                "re-running evaluation."
            ) from safetensors_exc

    @classmethod
    def _embed_codebert(cls, texts: List[str], tokenizer, model, device: str, max_length: int = 512):
        """Mean-pools CodeBERT hidden states into a single normalized embedding per text."""
        import torch

        with torch.no_grad():
            encoded = tokenizer(
                texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            hidden_states = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            mean_pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

    @classmethod
    def compute_bertscore(
        cls,
        references: List[Union[str, Dict[str, Any]]],
        predictions: List[str],
        model_type: str = "microsoft/codebert-base",
        device: str = EVAL_DEVICE,
    ) -> float:
        """Computes mean cosine similarity between CodeBERT embeddings of refs vs. predictions."""
        if not references or not predictions:
            return 0.0

        ref_texts = [cls.extract_code_str(r) for r in references]
        pred_texts = [cls.extract_code_str(p) for p in predictions]

        try:
            tokenizer, model = cls._load_codebert(model_type, device)
            ref_embeddings = cls._embed_codebert(ref_texts, tokenizer, model, device)
            pred_embeddings = cls._embed_codebert(pred_texts, tokenizer, model, device)
            cosine_sim = (ref_embeddings * pred_embeddings).sum(dim=1)
            return float(cosine_sim.mean().item())
        except Exception as exc:
            logger.error(f"Error calculating CodeBERTScore: {exc}")
            return 0.0

    # -------------------------------------------------------------------------
    # 3. Functional Verification: Execution Accuracy (Pass@1)
    # -------------------------------------------------------------------------
    @staticmethod
    def _strip_markdown_code_fence(code_str: str) -> str:
        """Extracts code from a ```python ... ``` fence if present, else strips
        any stray fence markers."""
        match = re.search(r"```(?:python|py)?\s*\n?(.*?)\n?\s*```", code_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        code_str = re.sub(r"^```[a-zA-Z]*\s*", "", code_str.strip(), flags=re.MULTILINE)
        return re.sub(r"\s*```$", "", code_str, flags=re.MULTILINE)

    @classmethod
    def _run_program_and_check_success(cls, program: str, timeout: int) -> bool:
        """Writes `program` to a temp file and runs it, returning True on a clean exit."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(program)
            temp_file = f.name

        try:
            result = subprocess.run(["python3", temp_file], capture_output=True, text=True, timeout=timeout)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception as exc:
            logger.error(f"Execution error: {exc}")
            return False
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    @classmethod
    def evaluate_execution_accuracy(cls, predictions: List[str], test_cases: List[str], timeout: int = 3) -> float:
        """Executes generated code against executable test assertions in a sandboxed process."""
        if not predictions or not test_cases:
            return 0.0

        passed_count = 0
        for code, test in zip(predictions, test_cases):
            code_str = cls._strip_markdown_code_fence(cls.extract_code_str(code))
            full_program = f"{code_str}\n\n{test}"

            try:
                ast.parse(full_program)
            except SyntaxError as e:
                logger.debug(f"SyntaxError in generated code: {e}")
                continue

            if cls._run_program_and_check_success(full_program, timeout):
                passed_count += 1

        return passed_count / len(predictions)

    # -------------------------------------------------------------------------
    # 4. Token-level F1 Score
    # -------------------------------------------------------------------------
    @classmethod
    def compute_f1(cls, references: List[Union[str, Dict[str, Any]]], predictions: List[str]) -> float:
        """Averages token-set F1 (precision/recall over whitespace-split tokens)."""
        if not references or not predictions:
            return 0.0

        f1_scores = []
        for ref_item, pred in zip(references, predictions):
            ref_tokens = set(cls.extract_code_str(ref_item).strip().split())
            pred_tokens = set(pred.strip().split())

            if not ref_tokens and not pred_tokens:
                f1_scores.append(1.0)
                continue
            if not ref_tokens or not pred_tokens:
                f1_scores.append(0.0)
                continue

            num_same = len(ref_tokens & pred_tokens)
            precision = num_same / len(pred_tokens)
            recall = num_same / len(ref_tokens)
            f1_scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))

        return float(sum(f1_scores) / len(f1_scores))

    # -------------------------------------------------------------------------
    # 5. ROUGE Scores (ROUGE-1 and ROUGE-L)
    # -------------------------------------------------------------------------
    @staticmethod
    def _lcs_length(x, y) -> int:
        """Longest common subsequence length via classic O(m*n) dynamic programming."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    @staticmethod
    def _f1_from_overlap(overlap: int, pred_len: int, ref_len: int) -> float:
        """Shared precision/recall/F1 computation used by both ROUGE-1 and ROUGE-L."""
        precision = overlap / pred_len if pred_len else 0.0
        recall = overlap / ref_len if ref_len else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    @classmethod
    def compute_rouge(cls, references: List[Union[str, Dict[str, Any]]], predictions: List[str]) -> Dict[str, float]:
        """Computes averaged ROUGE-1 (unigram overlap) and ROUGE-L (longest common subsequence)."""
        if not references or not predictions:
            return {"ROUGE-1": 0.0, "ROUGE-L": 0.0}

        rouge1_scores, rougel_scores = [], []
        for ref_item, pred in zip(references, predictions):
            ref_tokens = cls.extract_code_str(ref_item).strip().split()
            pred_tokens = pred.strip().split()

            unigram_overlap = sum((Counter(ref_tokens) & Counter(pred_tokens)).values())
            rouge1_scores.append(cls._f1_from_overlap(unigram_overlap, len(pred_tokens), len(ref_tokens)))

            lcs_overlap = cls._lcs_length(ref_tokens, pred_tokens)
            rougel_scores.append(cls._f1_from_overlap(lcs_overlap, len(pred_tokens), len(ref_tokens)))

        return {
            "ROUGE-1": float(sum(rouge1_scores) / len(rouge1_scores)),
            "ROUGE-L": float(sum(rougel_scores) / len(rougel_scores)),
        }

    # -------------------------------------------------------------------------
    # 6. SQL Response/Execution Accuracy (Result-based)
    # -------------------------------------------------------------------------
    @staticmethod
    def _result_sets_match(gold_results: list, pred_results: list) -> bool:
        """Compares two SQL result sets order-independently where possible."""
        try:
            return sorted(gold_results) == sorted(pred_results)
        except TypeError:
            # Unsortable row types (e.g. mixed None/str) - fall back to direct comparison.
            return gold_results == pred_results

    @classmethod
    def compute_sql_execution_accuracy(
        cls, gold_sqls: List[str], pred_sqls: List[str], db_paths: List[str]
    ) -> float:
        """Executes gold and predicted SQL queries against SQLite databases and compares
        the result sets. This is the standard 'Execution Accuracy' for Text-to-SQL tasks.
        """
        if not gold_sqls or not pred_sqls or not db_paths:
            return 0.0

        passed = 0
        for gold, pred, db_path in zip(gold_sqls, pred_sqls, db_paths):
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                cursor = conn.cursor()
                cursor.execute(gold)
                gold_results = cursor.fetchall()
                cursor.execute(pred)
                pred_results = cursor.fetchall()
                conn.close()

                if cls._result_sets_match(gold_results, pred_results):
                    passed += 1
            except Exception as e:
                logger.warning(f"SQL execution failed for DB {db_path}: {e}")
                continue

        return passed / len(gold_sqls)
=======
    def calculate_codebleu(
        predictions: List[str],
        references: List[str],
        lang: str = "python",
    ) -> float:

        try:
            result = calc_codebleu(
                references,
                predictions,
                lang=lang,
            )

            return float(
                result["codebleu"]
            )

        except Exception as e:

            print(
                "CodeBLEU error:",
                e
            )

            return 0.0

    @staticmethod
    def calculate_code_bertscore(
            predictions: List[str],
            references: List[str],
            lang: str = "python",
    ) -> float:
        """
        Calculate CodeBERT similarity using microsoft/codebert-base.
        Runs on CPU.
        """

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            import torch.nn.functional as F

            device = torch.device("cpu")

            model_name = "microsoft/codebert-base"

            # Load once and cache
            if not hasattr(CodeMetricsEvaluator, "_codebert_tokenizer"):
                CodeMetricsEvaluator._codebert_tokenizer = (
                    AutoTokenizer.from_pretrained(model_name)
                )

                CodeMetricsEvaluator._codebert_model = (
                    AutoModel.from_pretrained(model_name)
                    .to(device)
                    .eval()
                )

            tokenizer = CodeMetricsEvaluator._codebert_tokenizer
            model = CodeMetricsEvaluator._codebert_model

            scores = []

            with torch.no_grad():

                for pred, ref in zip(predictions, references):
                    pred_inputs = tokenizer(
                        pred,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256,
                        padding=True,
                    ).to(device)

                    ref_inputs = tokenizer(
                        ref,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256,
                        padding=True,
                    ).to(device)

                    pred_emb = model(**pred_inputs).last_hidden_state[:, 0, :]

                    ref_emb = model(**ref_inputs).last_hidden_state[:, 0, :]

                    similarity = F.cosine_similarity(
                        pred_emb,
                        ref_emb
                    )

                    scores.append(
                        float(similarity.item())
                    )

            return sum(scores) / len(scores) if scores else 0.0


        except Exception as e:

            print(
                "CodeBERT error:",
                e
            )

            return 0.0


    @staticmethod
    def calculate_bleu(
        predictions: List[str],
        references: List[str],
    ) -> float:

        try:
            refs = [
                [r.split()]
                for r in references
            ]

            hyps = [
                p.split()
                for p in predictions
            ]

            return float(
                corpus_bleu(
                    refs,
                    hyps,
                    smoothing_function=
                    SmoothingFunction().method1,
                )
            )

        except Exception as e:

            print(
                "BLEU error:",
                e
            )
            return 0.0

    @staticmethod
    def exact_match_accuracy(
        predictions: List[str],
        references: List[str],
    ) -> float:

        if not predictions:
            return 0.0

        matches = sum(
            p.strip() == r.strip()
            for p, r in zip(
                predictions,
                references,
            )
        )

        return matches / len(predictions)



    @staticmethod
    def pass_at_k(
        n_samples: int,
        n_correct: int,
        k: int,
    ) -> float:

        if n_samples - n_correct < k:
            return 1.0

        return 1.0 - math.prod(
            (
                (n_samples - n_correct - i)
                /
                (n_samples - i)
            )

            for i in range(k)
        )


    @staticmethod
    def evaluate_generation(
        predictions: List[str],
        references: List[str],
        lang: str = "python",
    ) -> Dict[str, Any]:

        return {

            "n":
                len(predictions),

            "bleu":
                CodeMetricsEvaluator.calculate_bleu(
                    predictions,
                    references,
                ),

            # Keep CodeBLEU for code tasks
            "codebleu":
                CodeMetricsEvaluator.calculate_codebleu(
                    predictions,
                    references,
                    lang,
                ),

            # Normal BERTScore for docs
            "code_bertscore":
                CodeMetricsEvaluator.calculate_bertscore(
                    predictions,
                    references,
                ),

            "exact_match":
                CodeMetricsEvaluator.exact_match_accuracy(
                    predictions,
                    references,
                ),
        }
>>>>>>> Stashed changes
