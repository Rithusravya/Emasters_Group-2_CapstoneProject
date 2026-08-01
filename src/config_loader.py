"""
Step 1 — Central configuration loader.

Every other module (`src.embeddings.*`, `src.indexing.*`, `src.rag.*`, `main.ipynb`, ...)
imports the single `CFG` object defined here instead of re-parsing YAML themselves.
`CFG` supports both dict-style (`CFG["model"]["name_or_path"]`) and dotted attribute
access (`CFG.model.name_or_path`), and falls back to sane defaults if
`configs/config.yaml` is missing or a key isn't present, so notebooks/scripts don't
crash on a fresh checkout before the config file has been customized.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "CodeGen-Pipeline",
        "device": "cuda",
    },
    "model": {
        "name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "embedding_name": "BAAI/bge-m3",
        "max_length": 512,
    },
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["qkv_proj"],
    },
    "data": {
        "raw_dir": "data/raw",
        # dataset key -> subfolder name under raw_dir
        "datasets": {
            "spider": "Spider",
            "birdbench": "BirdBench",
            "codocbench": "CoDocBench",
        },
        "sample_limit": 200,
    },
    "rag": {
        "codeparrot_subset_size": 500,
    },
    "indexing": {
        "top_k": 3,
        "vector_dim": 384,
    },
    "outputs": {
        "metrics_dir": "outputs/metrics",
        "plots_dir": "outputs/plots",
        "logs_dir": "outputs/logs",
        "checkpoints_dir": "models/checkpoints/codegen_350m_multi_lora",
    },
}


class DotDict(dict):
    """A dict that also allows attribute access, recursively.

    CFG.model.name_or_path  ==  CFG["model"]["name_or_path"]
    Missing keys resolve to None instead of raising, so optional config sections
    (e.g. a dataset not yet added to config.yaml) don't blow up notebook cells.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError:
            return None
        return DotDict(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge `override` onto `base`, recursing into nested dicts so a partial
    config.yaml (e.g. only overriding `model.name_or_path`) still inherits every
    other default instead of dropping the rest of that section."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str = "configs/config.yaml") -> DotDict:
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)
    return DotDict(config)


# Singleton used across the codebase: `from src.config_loader import CFG`
CFG = load_config()


def reload_config(config_path: str = "configs/config.yaml") -> DotDict:
    """Re-read config.yaml and refresh the CFG singleton in place (useful in
    notebooks after editing the YAML file without restarting the kernel)."""
    global CFG
    CFG.clear()
    CFG.update(load_config(config_path))
    return CFG
