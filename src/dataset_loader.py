"""
Step 2 — Canonical dataset loader (plain module, promoted out of
`src/data/data_loader.ipynb`).

data/raw/ layout is *folders of files*, not single files, e.g.:

    data/raw/
      Spider/       spider2-dbt.jsonl, spider2-lite.jsonl, spider2-snow.jsonl, spider2-snow-0713.jsonl
      BirdBench/    dev.json                         (a single JSON array, not jsonl)
      CoDocBench/   codocbench.jsonl, train.jsonl, test.jsonl

`load_dataset_folder` globs every matching file in a folder and merges the
rows, tagging each with which file it came from, so adding a new split/file
to a dataset folder just works without touching this code.

`src/data/data_loader.ipynb` re-exports everything below for notebook use —
edit the logic here, not there.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def load_jsonl_file(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load one JSON-Lines file (one JSON object per line)."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON line in %s", path)
                continue
            row["_source_file"] = os.path.basename(path)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows

def load_json_array_file(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load one .json file that holds a top-level JSON array of objects
    (e.g. BirdBench's dev.json), or a single object (wrapped into a 1-item list)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    rows = []
    for row in data:
        if isinstance(row, dict):
            row = dict(row)
            row["_source_file"] = os.path.basename(path)
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows

def load_dataset_folder(
    folder: str,
    limit: Optional[int] = None,
    jsonl_pattern: str = "*.jsonl",
    json_pattern: str = "*.json",
) -> List[Dict[str, Any]]:
    """Merge every .jsonl and .json file found directly under `folder` into one
    list of records. Files are read in sorted order for reproducibility. If
    `limit` is set, it caps the *combined* row count across all files in the
    folder (not per-file), so callers get a predictably-sized sample."""
    if not os.path.isdir(folder):
        logger.warning("Dataset folder not found: %s", folder)
        return []

    rows: List[Dict[str, Any]] = []
    jsonl_paths = sorted(glob.glob(os.path.join(folder, jsonl_pattern)))
    json_paths = sorted(
        p for p in glob.glob(os.path.join(folder, json_pattern))
        if p not in jsonl_paths
    )

    for path in jsonl_paths:
        remaining = None if limit is None else max(0, limit - len(rows))
        if limit is not None and remaining == 0:
            break
        rows.extend(load_jsonl_file(path, limit=remaining))

    for path in json_paths:
        remaining = None if limit is None else max(0, limit - len(rows))
        if limit is not None and remaining == 0:
            break
        rows.extend(load_json_array_file(path, limit=remaining))

    logger.info(
        "Loaded %d rows from %s (%d .jsonl file(s), %d .json file(s))",
        len(rows), folder, len(jsonl_paths), len(json_paths),
    )
    return rows

# Fallback samples used only if a dataset folder is missing/empty, so the rest
# of the pipeline (generation, indexing, RAG, eval) can still run end-to-end
# on a fresh checkout without the full data/ directory present.
_FALLBACKS: Dict[str, List[Dict[str, Any]]] = {
    "spider": [
        {"instance_id": "demo-0", "db": "demo_db",
         "question": "Find all active users.",
         "gold_sql": "SELECT * FROM users WHERE status = 'active';"}
    ],
    "birdbench": [
        {"question_id": 0, "db_id": "demo_db",
         "question": "What is the highest score in the exams table?",
         "SQL": "SELECT MAX(score) FROM exams;", "difficulty": "simple"}
    ],
    "codocbench": [
        {"file": "demo.py", "function": "calc_area",
         "version_data": [{"code": "def calc_area(r):\n    return 3.14 * r ** 2",
                            "docstring": "Calculates circle area."}]}
    ],
}

class UnifiedDatasetLoader:
    """Loads every benchmark used by the pipeline from data/raw/<subfolder>/,
    where each subfolder may hold multiple .jsonl/.json files."""

    @staticmethod
    def load_datasets(config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        raw_dir = config["data"]["raw_dir"]
        dataset_dirs = config["data"]["datasets"]
        limit = config["data"].get("sample_limit")

        logger.info("Loading benchmarks and datasets from %s ...", raw_dir)
        loaded: Dict[str, List[Dict[str, Any]]] = {}

        for key, subfolder in dataset_dirs.items():
            folder = os.path.join(raw_dir, subfolder)
            try:
                rows = load_dataset_folder(folder, limit=limit)
            except Exception:
                logger.exception("Failed loading dataset '%s' from %s", key, folder)
                rows = []

            if not rows:
                logger.warning("No rows found for '%s' — using built-in fallback sample.", key)
                rows = _FALLBACKS.get(key, [])

            loaded[key] = rows

        return loaded

    # Backwards-compatible alias for older notebooks that call load_dataset(...)
    load_dataset = load_datasets