"""
Step 6 / Task 2 — Interactive database querying: generate SQL from natural-language
text and evaluate it against Spider / BirdBench-style (schema, question, gold_sql)
examples. Metrics: Execution Accuracy, Exact Match.
"""
from __future__ import annotations
import sqlite3
from typing import Any, Dict, List, Optional

import sqlparse

PROMPT_TEMPLATE = (
    "# Database schema:\n{schema}\n\n"
    "# Question: {question}\n"
    "# Write a single valid SQL query that answers the question.\n"
    "SQL:"
)


def build_prompt(schema: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(schema=schema, question=question)


def generate_sql(lm, schema: str, question: str) -> str:
    prompt = build_prompt(schema, question)
    raw = lm.generate(prompt, max_new_tokens=128, num_return_sequences=1)[0]
    sql = raw.split(";")[0].strip()
    return sql + ";" if sql else sql


def normalize_sql(sql: str) -> str:
    formatted = sqlparse.format(
        sql, keyword_case="upper", identifier_case="lower",
        strip_comments=True, reindent=False,
    )
    return " ".join(formatted.split())


def exact_match(pred_sql: str, gold_sql: str) -> bool:
    return normalize_sql(pred_sql) == normalize_sql(gold_sql)


def execute_sql(db_path: str, sql: str) -> Optional[List[tuple]]:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows
    except sqlite3.Error:
        return None


def execution_accuracy(db_path: str, pred_sql: str, gold_sql: str) -> bool:
    pred_rows = execute_sql(db_path, pred_sql)
    gold_rows = execute_sql(db_path, gold_sql)
    if pred_rows is None or gold_rows is None:
        return False
    return sorted(pred_rows) == sorted(gold_rows)


def evaluate_batch(lm, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """examples: list of {"schema": str, "question": str, "gold_sql": str, "db_path": str}"""
    n = len(examples)
    n_exact, n_exec = 0, 0
    predictions = []
    for ex in examples:
        pred = generate_sql(lm, ex["schema"], ex["question"])
        predictions.append(pred)
        if exact_match(pred, ex["gold_sql"]):
            n_exact += 1
        if ex.get("db_path") and execution_accuracy(ex["db_path"], pred, ex["gold_sql"]):
            n_exec += 1
    return {
        "n": n,
        "exact_match_acc": n_exact / n if n else 0.0,
        "execution_acc": n_exec / n if n else 0.0,
        "predictions": predictions,
    }
