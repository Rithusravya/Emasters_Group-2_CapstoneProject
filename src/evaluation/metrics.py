import re
import torch
import logging
from typing import List, Dict, Optional
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

logger = logging.getLogger(__name__)

_BERTSCORE_NUM_LAYERS_OVERRIDES = {
    "models/codebert-base": 10,
    "models/codebert-base-mlm": 10
}


def _resolve_bertscore_num_layers(model_type: str) -> Optional[int]:
    try:
        from bert_score.utils import model2layers
    except Exception:
        model2layers = {}

    if model_type in model2layers:
        return model2layers[model_type]

    short_name = model_type.split("/")[-1]
    if short_name in model2layers:
        return model2layers[short_name]

    if model_type in _BERTSCORE_NUM_LAYERS_OVERRIDES:
        return _BERTSCORE_NUM_LAYERS_OVERRIDES[model_type]

    name_lower = model_type.replace("\\", "/").lower()
    for key in sorted(_BERTSCORE_NUM_LAYERS_OVERRIDES, key=len, reverse=True):
        key_short = key.split("/")[-1].lower()
        if key_short in name_lower:
            return _BERTSCORE_NUM_LAYERS_OVERRIDES[key]

    return None


class EvaluationMetrics:
    
    @staticmethod
    def _tokenize_sql(sql: str) -> List[str]:
        """Basic SQL tokenization for BLEU/ROUGE."""
        if not sql: return []
        sql = sql.lower().strip()
        sql = re.sub(r'[^a-z0-9\s_\*\(\)\,\.\'\"\=\>\<\!\+\-\%]', ' ', sql)
        return sql.split()

    @staticmethod
    def compute_bleu(references: List[str], predictions: List[str]) -> float:
        try:
            refs = [[EvaluationMetrics._tokenize_sql(r)] for r in references]
            preds = [EvaluationMetrics._tokenize_sql(p) for p in predictions]
            smoother = SmoothingFunction().method1
            score = corpus_bleu(refs, preds, smoothing_function=smoother)
            return round(score, 4)
        except Exception as e:
            logger.error(f"BLEU calculation failed: {e}")
            return 0.0

    @staticmethod
    def compute_rouge(references: List[str], predictions: List[str]) -> Dict[str, float]:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            rouge1_scores, rougeL_scores = [], []
            
            for ref, pred in zip(references, predictions):
                scores = scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
                
            return {
                "ROUGE-1": round(sum(rouge1_scores) / len(rouge1_scores), 4) if rouge1_scores else 0.0,
                "ROUGE-L": round(sum(rougeL_scores) / len(rougeL_scores), 4) if rougeL_scores else 0.0
            }
        except Exception as e:
            logger.error(f"ROUGE calculation failed: {e}")
            return {"ROUGE-1": 0.0, "ROUGE-L": 0.0}

    @staticmethod
    def compute_bertscore(references: List[str], predictions: List[str], device: str = "cpu", model_type: str = "models/codebert-base") -> float:
        """CodeBERTScore: BERTScore F1 computed with a code-aware encoder (CodeBERT by default).

        NOTE: `model_type` values that aren't in bert_score's built-in
        `model2layers` table (which is the case for both "microsoft/codebert-base"
        and "FacebookAI/roberta-base") need an explicit `num_layers`, or the
        underlying `bert_score.score()` call raises a KeyError. See
        `_resolve_bertscore_num_layers` above -- this is what was previously
        making CodeBERTScore silently return 0.0 for every model category.
        """
        try:
            from bert_score import score
            valid_pairs = [(r, p) for r, p in zip(references, predictions) if r.strip() and p.strip()]
            if not valid_pairs:
                return 0.0
            refs, preds = zip(*valid_pairs)

            num_layers = _resolve_bertscore_num_layers(model_type)
            score_kwargs = {"num_layers": num_layers} if num_layers is not None else {}

            with torch.no_grad():
                P, R, F1 = score(
                    list(preds), list(refs),
                    model_type=model_type,
                    verbose=False,
                    device=device,
                    **score_kwargs,
                )
            return round(F1.mean().item(), 4)
        except Exception as e:
            logger.error(f"BERTScore ({model_type}) calculation failed: {e}")
            return 0.0

    @staticmethod
    def _find_top_level_groups(s: str) -> List[tuple]:
        """Locates every top-level `{...}`/`[...]` span in `s`, skipping over
        brace/bracket characters that appear inside double-quoted string
        literals (so e.g. a `$regex: "a{2,3}"` value doesn't confuse the
        bracket matching)."""
        groups = []
        depth = 0
        start = None
        in_string = False
        i, n = 0, len(s)
        while i < n:
            ch = s[i]
            if in_string:
                if ch == '\\':
                    i += 2
                    continue
                if ch == '"':
                    in_string = False
                i += 1
                continue
            if ch == '"':
                in_string = True
                i += 1
                continue
            if ch in '{[':
                if depth == 0:
                    start = i
                depth += 1
            elif ch in '}]':
                depth -= 1
                if depth == 0 and start is not None:
                    groups.append((start, i + 1))
                    start = None
            i += 1
        return groups

    @staticmethod
    def _split_top_level(inner: str, sep: str) -> List[str]:
        """Splits `inner` on `sep`, but only at bracket-depth 0 and never
        inside a double-quoted string, so nested `{...}`/`[...]`/`(...)`
        and comma-containing string values aren't split incorrectly."""
        parts, depth, current = [], 0, []
        in_string = False
        i = 0
        while i < len(inner):
            ch = inner[i]
            if in_string:
                current.append(ch)
                if ch == '\\' and i + 1 < len(inner):
                    current.append(inner[i + 1])
                    i += 1
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            if ch == '"':
                in_string = True
                current.append(ch)
                i += 1
                continue
            if ch in '{[(':
                depth += 1
            elif ch in '}])':
                depth -= 1
            if ch == sep and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
            i += 1
        parts.append(''.join(current))
        return [p for p in parts if p.strip() != '']

    @classmethod
    def _canon_group(cls, s: str) -> str:
        s = s.strip()
        if s.startswith('{') and s.endswith('}'):
            inner = s[1:-1]
            items = cls._split_top_level(inner, ',')
            kvs = []
            for item in items:
                if ':' not in item:
                    return s  # malformed for our purposes -- bail out safely
                k, v = item.split(':', 1)
                kvs.append((k.strip(), cls._canon_group(v.strip())))
            kvs.sort(key=lambda kv: kv[0])
            return '{' + ','.join(f'{k}:{v}' for k, v in kvs) + '}'
        elif s.startswith('[') and s.endswith(']'):
            inner = s[1:-1]
            items = cls._split_top_level(inner, ',')
            return '[' + ','.join(cls._canon_group(i.strip()) for i in items) + ']'
        return s

    @classmethod
    def _canonicalize_mql(cls, s: str) -> str:
        try:
            if s.count('{') != s.count('}') or s.count('[') != s.count(']'):
                return s
            groups = cls._find_top_level_groups(s)
            if not groups:
                return s
            out, last = [], 0
            for gstart, gend in groups:
                out.append(s[last:gstart])
                out.append(cls._canon_group(s[gstart:gend]))
                last = gend
            out.append(s[last:])
            return ''.join(out)
        except Exception:
            return s

    # @classmethod
    # def compute_exact_match(cls, gold: str, pred: str) -> float:
    #     if not gold or not pred:
    #         return 0.0

    #     def normalize(s):
    #         s = s.strip().rstrip(';')
    #         s = s.replace("'", '"')            # Normalize single quotes to double quotes
    #         s = cls._canonicalize_mql(s)       # Order-independent object-key comparison
    #         s = re.sub(r'\s+', '', s)          # Remove all remaining whitespace
    #         return s.lower()

    #     return 1.0 if normalize(gold) == normalize(pred) else 0.0

    @staticmethod
    def _tokenize_mql(query: str) -> List[str]:
        """Tokenizer optimized for MongoDB Query Language (MQL) brackets and punctuation."""
        if not query: return []
        # Add spaces around brackets, commas, colons, and dots
        query = re.sub(r'([(){}[\],:.])', r' \1 ', query)
        return [tok.lower() for tok in query.split() if tok.strip()]

    @staticmethod
    def compute_response_accuracy(gold: str, pred: str) -> float:
        """Semantic response accuracy (recall-based token overlap)."""
        if not gold or not pred: return 0.0
        g_tokens = set(EvaluationMetrics._tokenize_mql(gold))
        p_tokens = set(EvaluationMetrics._tokenize_mql(pred))
        if not g_tokens: return 0.0
        overlap = len(g_tokens.intersection(p_tokens))
        return round(overlap / len(g_tokens), 4)