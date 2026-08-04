import ast
import logging
import os
import re
import subprocess
import sqlite3
import tempfile
from math import exp, log
from collections import Counter
from typing import Any, Dict, List, Tuple, Union

logger = logging.getLogger(__name__)
EVAL_DEVICE = "cpu"


class EvaluationMetrics:
    """Provides semantic, structural, and execution metrics for generated code."""

    _codebert_tokenizer = None
    _codebert_model = None
    _codebert_device = None

    @staticmethod
    def extract_code_str(item: Union[str, Dict[str, Any], Any]) -> str:
        """Extracts code string safely from dict, string, or dataset items."""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
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
            precisions.append((overlap + 1) / (total + 1))
        geo_mean = exp(sum(log(p) for p in precisions) / len(precisions))
        ref_len, cand_len = len(ref_tokens), len(cand_tokens)
        brevity_penalty = 1.0 if cand_len > ref_len else exp(1 - ref_len / cand_len)
        return brevity_penalty * geo_mean

    @classmethod
    def compute_bleu(cls, references: List[Union[str, Dict[str, Any]]], predictions: List[str]) -> float:
        if not references or not predictions:
            return 0.0
        scores = []
        for ref_item, pred in zip(references, predictions):
            ref_tokens = cls.extract_code_str(ref_item).strip().split()
            pred_tokens = pred.strip().split()
            scores.append(cls._sentence_bleu(ref_tokens, pred_tokens))
        return float(sum(scores) / len(scores)) if scores else 0.0

    # -------------------------------------------------------------------------
    # 2. Semantic Representation: CodeBERTScore
    # -------------------------------------------------------------------------
    @classmethod
    def _load_codebert(cls, model_name: str, device: str):
        if (cls._codebert_model is not None and cls._codebert_tokenizer is not None and cls._codebert_device == device):
            return cls._codebert_tokenizer, cls._codebert_model
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("`torch` and `transformers` are required for CodeBERTScore.") from exc
        logger.info(f"Loading CodeBERT model '{model_name}' on device='{device}' ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        cls._codebert_tokenizer = tokenizer
        cls._codebert_model = model
        cls._codebert_device = device
        return tokenizer, model

    @classmethod
    def _embed_codebert(cls, texts: List[str], tokenizer, model, device: str, max_length: int = 512):
        import torch
        with torch.no_grad():
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(
                device)
            hidden_states = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = summed / counts
            return torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

    @classmethod
    def compute_bertscore(cls, references: List[Union[str, Dict[str, Any]]], predictions: List[str],
                          model_type: str = "microsoft/codebert-base", device: str = EVAL_DEVICE) -> float:
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
    # 3. Functional Verification: Execution Accuracy (Pass@1) - FIXED
    # -------------------------------------------------------------------------
    @classmethod
    def evaluate_execution_accuracy(cls, predictions: List[str], test_cases: List[str], timeout: int = 3) -> float:
        """Executes generated code against executable test assertions in a sandboxed process."""
        if not predictions or not test_cases:
            return 0.0

        passed_count = 0
        for code, test in zip(predictions, test_cases):
            code_str = cls.extract_code_str(code)

            # ROBUST FIX: Extract code from markdown blocks (not just strip markers)
            match = re.search(r"```(?:python|py)?\s*\n?(.*?)\n?\s*```", code_str, re.DOTALL)
            if match:
                code_str = match.group(1).strip()
            else:
                code_str = re.sub(r"^```[a-zA-Z]*\s*", "", code_str.strip(), flags=re.MULTILINE)
                code_str = re.sub(r"\s*```$", "", code_str, flags=re.MULTILINE)

            full_program = f"{code_str}\n\n{test}"

            try:
                ast.parse(full_program)
            except SyntaxError as e:
                logger.debug(f"SyntaxError in generated code: {e}")
                continue

            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(full_program)
                temp_file = f.name

            try:
                result = subprocess.run(["python3", temp_file], capture_output=True, text=True, timeout=timeout)
                if result.returncode == 0:
                    passed_count += 1
            except subprocess.TimeoutExpired:
                pass
            except Exception as exc:
                logger.error(f"Execution error: {exc}")
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        return passed_count / len(predictions)

    # -------------------------------------------------------------------------
    # 4. Token-level F1 Score
    # -------------------------------------------------------------------------
    @classmethod
    def compute_f1(cls, references: List[Union[str, Dict[str, Any]]], predictions: List[str]) -> float:
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
            common = ref_tokens.intersection(pred_tokens)
            num_same = len(common)
            precision = num_same / len(pred_tokens)
            recall = num_same / len(ref_tokens)
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))
        return float(sum(f1_scores) / len(f1_scores))

    # -------------------------------------------------------------------------
    # 5. ROUGE Scores (ROUGE-1 and ROUGE-L)
    # -------------------------------------------------------------------------
    @staticmethod
    def _lcs_length(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    @classmethod
    def compute_rouge(cls, references: List[Union[str, Dict[str, Any]]], predictions: List[str]) -> Dict[str, float]:
        if not references or not predictions:
            return {"ROUGE-1": 0.0, "ROUGE-L": 0.0}
        rouge1_scores = []
        rougel_scores = []
        for ref_item, pred in zip(references, predictions):
            ref_tokens = cls.extract_code_str(ref_item).strip().split()
            pred_tokens = pred.strip().split()
            ref_unigrams = Counter(ref_tokens)
            pred_unigrams = Counter(pred_tokens)
            overlap = sum((ref_unigrams & pred_unigrams).values())
            prec1 = overlap / len(pred_tokens) if pred_tokens else 0.0
            rec1 = overlap / len(ref_tokens) if ref_tokens else 0.0
            f1_1 = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0
            rouge1_scores.append(f1_1)
            lcs = cls._lcs_length(ref_tokens, pred_tokens)
            prec_l = lcs / len(pred_tokens) if pred_tokens else 0.0
            rec_l = lcs / len(ref_tokens) if ref_tokens else 0.0
            f1_l = 2 * prec_l * rec_l / (prec_l + rec_l) if (prec_l + rec_l) > 0 else 0.0
            rougel_scores.append(f1_l)
        return {
            "ROUGE-1": float(sum(rouge1_scores) / len(rouge1_scores)),
            "ROUGE-L": float(sum(rougel_scores) / len(rougel_scores))
        }

    # -------------------------------------------------------------------------
    # 6. SQL Response/Execution Accuracy (Result-based) - PROFESSOR'S REQUIREMENT
    # -------------------------------------------------------------------------
    @classmethod
    def compute_sql_execution_accuracy(
            cls,
            gold_sqls: List[str],
            pred_sqls: List[str],
            db_paths: List[str]
    ) -> float:
        """
        Executes gold and predicted SQL queries against SQLite databases and compares the result sets.
        This is the standard 'Execution Accuracy' for Text-to-SQL tasks.
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

                try:
                    if sorted(gold_results) == sorted(pred_results):
                        passed += 1
                except TypeError:
                    if gold_results == pred_results:
                        passed += 1

            except Exception as e:
                logger.warning(f"SQL execution failed for DB {db_path}: {e}")
                continue

        return passed / len(gold_sqls)
