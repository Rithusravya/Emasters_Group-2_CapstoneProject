"""
Step 2 — Load Datasets (Spider, BirdBench, CoDocBench).

Thin, format-aware loaders for the three benchmark datasets named in the project
brief. Each dataset must be downloaded into its configured `datasets.<name>.path`
directory yourself (this environment has no network access to the dataset sites) —
these functions just read the JSONL files once they're there.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.config_loader import CFG, ROOT_DIR


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset into this directory first — "
            f"see README references for Spider/BirdBench/CoDocBench sources."
        )
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_codocbench(split: str | None = None) -> List[Dict]:
    """CoDocBench: (code, docstring) pairs for the documentation-generation task."""
    split = split or CFG.datasets.codocbench.split
    path = ROOT_DIR / CFG.datasets.codocbench.path.lstrip("./") / f"{split}.jsonl"
    return _read_jsonl(path)


def load_spider(split: str | None = None) -> List[Dict]:
    """Spider: (db_schema, question, gold_sql) examples for text-to-SQL."""
    split = split or CFG.datasets.spider.split
    path = ROOT_DIR / CFG.datasets.spider.path.lstrip("./") / f"{split}.jsonl"
    return _read_jsonl(path)


def load_birdbench(split: str | None = None) -> List[Dict]:
    """BirdBench: harder multi-domain text-to-SQL examples, same schema as Spider."""
    split = split or CFG.datasets.birdBench.split
    path = ROOT_DIR / CFG.datasets.birdBench.path.lstrip("./") / f"{split}.jsonl"
    return _read_jsonl(path)


def load_raw_language_corpus(language: str) -> List[Dict]:
    """(code, docstring) pairs for the new-language fine-tune target, expected at
    data/<language>_raw.jsonl with schema {"code": ..., "docstring": ...}. You are
    expected to populate this yourself (e.g. via the GitHub API / CodeParrot-style
    scraping) since this sandbox cannot reach source repos for the target language."""
    path = ROOT_DIR / "data" / f"{language.lower()}_raw.jsonl"
    return _read_jsonl(path)
