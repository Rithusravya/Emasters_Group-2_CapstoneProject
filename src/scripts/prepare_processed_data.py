"""Builds the small processed-data JSONL files used for training/evaluation:
Mongo-augmented Spider records, a retrieval corpus, and a few hand-written
samples for the program/doc/commit generation tasks.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Default MongoDB query used when a source record doesn't have one of its own.
_DEFAULT_MONGO_QUERY = {"collection": "unknown", "operation": "find", "filter": {}}


def read_jsonl(path: Path) -> list:
    """Reads a JSONL file into a list of dicts. Returns [] if the file is missing."""
    if not path.exists():
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list) -> None:
    """Writes a list of dicts to a JSONL file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {path}")


def _load_spider_eval_or_warn() -> list:
    """Loads spider_eval.jsonl, warning (rather than failing) if it's missing/empty."""
    spider_eval_path = PROCESSED_DIR / "spider_eval.jsonl"
    spider_eval = read_jsonl(spider_eval_path)
    if not spider_eval:
        print(f"WARNING: {spider_eval_path} not found or empty.")
    return spider_eval


def create_spider_mongo_from_spider_eval() -> None:
    """Augments each Spider eval record with a (placeholder) MongoDB query field."""
    output_path = PROCESSED_DIR / "spider_mongo_converted.jsonl"
    spider_eval = _load_spider_eval_or_warn()
    if not spider_eval:
        return

    records = [
        {
            "id": item.get("id", ""),
            "task": "text_to_sql",
            "db_id": item.get("db_id", ""),
            "question": item.get("question", ""),
            "schema": item.get("schema", ""),
            "gold_sql": item.get("gold_sql", ""),
            "pred_sql": item.get("pred_sql", ""),
            "sql_query": item.get("gold_sql", ""),
            "mongodb_query": dict(_DEFAULT_MONGO_QUERY),
        }
        for item in spider_eval
    ]
    write_jsonl(output_path, records)


def create_retrieval_corpus_from_spider_eval() -> None:
    """Builds a (question, gold SQL) retrieval corpus from Spider eval records."""
    output_path = PROCESSED_DIR / "retrieval_corpus.jsonl"
    spider_eval = _load_spider_eval_or_warn()
    if not spider_eval:
        return

    records = []
    for item in spider_eval:
        question = item.get("question", "")
        gold_sql = item.get("gold_sql", "")
        if not question and not gold_sql:
            continue

        records.append({
            "id": item.get("id", ""),
            "task": "text_to_sql",
            "text": question,
            "code": gold_sql,
            "metadata": {"db_id": item.get("db_id", ""), "source": "spider_eval"},
        })
    write_jsonl(output_path, records)


def create_program_generation_sample() -> None:
    """Writes a few hand-crafted program-generation examples for smoke-testing."""
    records = [
        {
            "id": "program_sample_001",
            "task": "program_generation",
            "instruction": "Write a Python function that returns the area of a circle given its radius.",
            "prompt": "Write a Python function that returns the area of a circle given its radius.",
            "code": "import math\n\ndef circle_area(radius):\n    return math.pi * radius ** 2",
        },
        {
            "id": "program_sample_002",
            "task": "program_generation",
            "instruction": "Write a Python function that returns the sum of a list of numbers.",
            "prompt": "Write a Python function that returns the sum of a list of numbers.",
            "code": "def sum_list(numbers):\n    return sum(numbers)",
        },
        {
            "id": "program_sample_003",
            "task": "program_generation",
            "instruction": "Write a Python function that checks whether a number is even.",
            "prompt": "Write a Python function that checks whether a number is even.",
            "code": "def is_even(number):\n    return number % 2 == 0",
        },
    ]
    write_jsonl(PROCESSED_DIR / "program_generation.jsonl", records)


def create_doc_generation_sample() -> None:
    """Writes a few hand-crafted doc-generation examples for smoke-testing."""
    records = [
        {
            "id": "doc_sample_001",
            "task": "doc_generation",
            "code": "def add(a, b):\n    return a + b",
            "docstring": "Adds two numbers and returns the result.",
        },
        {
            "id": "doc_sample_002",
            "task": "doc_generation",
            "code": "def multiply(a, b):\n    return a * b",
            "docstring": "Multiplies two numbers and returns the result.",
        },
        {
            "id": "doc_sample_003",
            "task": "doc_generation",
            "code": "def is_positive(number):\n    return number > 0",
            "docstring": "Checks whether the given number is positive.",
        },
    ]
    write_jsonl(PROCESSED_DIR / "doc_generation.jsonl", records)


def create_commit_generation_sample() -> None:
    """Writes a few hand-crafted commit-message examples for smoke-testing."""
    records = [
        {
            "id": "commit_sample_001",
            "task": "commit_generation",
            "diff": "diff --git a/utils.py b/utils.py\n+ def add(a, b):\n+     return a + b",
            "commit_message": "feat: add utility function for addition",
        },
        {
            "id": "commit_sample_002",
            "task": "commit_generation",
            "diff": "diff --git a/main.py b/main.py\n- print('hello')\n+ print('Hello, world!')",
            "commit_message": "fix: update greeting message",
        },
        {
            "id": "commit_sample_003",
            "task": "commit_generation",
            "diff": "diff --git a/config.py b/config.py\n+ DEBUG = True",
            "commit_message": "chore: enable debug mode in config",
        },
    ]
    write_jsonl(PROCESSED_DIR / "commit_generation.jsonl", records)


if __name__ == "__main__":
    print("Preparing processed dataset files...")

    create_spider_mongo_from_spider_eval()
    create_retrieval_corpus_from_spider_eval()
    create_program_generation_sample()
    create_doc_generation_sample()
    create_commit_generation_sample()

    print("Done.")
