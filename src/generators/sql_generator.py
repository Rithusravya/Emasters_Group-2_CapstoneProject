"""
Step 6 / Task 2 — Text-to-SQL generation

Generate SQL from a natural-language question.

Evaluation:
- Exact Match
- Execution Accuracy (when a local SQLite DB is available)
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, List, Optional

import sqlparse


PROMPT_TEMPLATE = """
You are an expert SQLite developer.

Given the database schema and the user's question,
write ONE valid SQLite SQL query.

Rules:
- Return ONLY SQL.
- Do not explain.
- Do not use markdown.
- End with a semicolon.

Database schema:

{schema}

Question:

{question}

SQL:
"""


def build_prompt(schema: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(
        schema=schema,
        question=question,
    )


def extract_sql(text: str) -> str:
    """
    Extract SQL from model output.
    """

    if not text:
        return ""

    text = text.strip()

    # remove markdown
    text = text.replace("```sql", "")
    text = text.replace("```", "")

    # remove prompt echo
    if "SQL:" in text:
        text = text.split("SQL:")[-1]

    text = text.strip()

    # first SELECT/WITH/INSERT/UPDATE/DELETE statement
    m = re.search(
        r"(SELECT|WITH|INSERT|UPDATE|DELETE).*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if m:
        text = m.group(0)

    text = text.split(";")[0].strip()

    if text:
        text += ";"

    return text


def generate_sql(lm, schema: str, question: str) -> str:
    prompt = build_prompt(schema, question)

    raw = lm.generate(
        prompt,
        max_new_tokens=128,
        num_return_sequences=1,
    )[0]

    return extract_sql(raw)


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL formatting.
    """

    if not sql:
        return ""

    sql = sqlparse.format(
        sql,
        keyword_case="upper",
        identifier_case="lower",
        strip_comments=True,
        reindent=False,
    )

    sql = " ".join(sql.split())

    return sql.strip()


def exact_match(pred_sql: str, gold_sql: str) -> bool:
    """
    Case-insensitive normalized exact match.
    """
    return normalize_sql(pred_sql) == normalize_sql(gold_sql)


def execute_sql(db_path: str, sql: str) -> Optional[List[tuple]]:
    """
    Execute SQL on SQLite database.
    """

    try:
        conn = sqlite3.connect(db_path)

        cur = conn.cursor()

        cur.execute(sql)

        rows = cur.fetchall()

        conn.close()

        return rows

    except sqlite3.Error:

        return None


def execution_accuracy(
    db_path: str,
    pred_sql: str,
    gold_sql: str,
) -> bool:

    pred_rows = execute_sql(db_path, pred_sql)

    gold_rows = execute_sql(db_path, gold_sql)

    if pred_rows is None or gold_rows is None:
        return False

    return sorted(pred_rows) == sorted(gold_rows)


def evaluate_batch(
    lm,
    examples: List[Dict[str, Any]],
) -> Dict[str, Any]:

    predictions = []

    exact = 0
    execution = 0

    for ex in examples:

        pred = generate_sql(
            lm,
            ex["schema"],
            ex["question"],
        )

        predictions.append(pred)

        if exact_match(pred, ex["gold_sql"]):
            exact += 1

        if ex.get("db_path"):

            if execution_accuracy(
                ex["db_path"],
                pred,
                ex["gold_sql"],
            ):
                execution += 1

    n = len(examples)

    return {
        "n": n,
        "exact_match_acc": exact / n if n else 0.0,
        "execution_acc": execution / n if n else 0.0,
        "predictions": predictions,
    }