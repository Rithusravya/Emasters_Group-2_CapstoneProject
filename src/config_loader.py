"""
Step 1 — Load Configuration.

Parses configs/default.yaml (hyperparameters, model paths, system settings,
execution/logging flags) into a Config object used across the whole pipeline.
Also resolves and creates the on-disk directories the rest of the codebase
depends on (data/, models/, outputs/, indices).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
FINETUNED_DIR = MODELS_DIR / "fine_tuned"
OUTPUTS_DIR = ROOT_DIR / "outputs"
INDEX_DIR = OUTPUTS_DIR / "indices"

for _d in (DATA_DIR, MODELS_DIR, CHECKPOINT_DIR, FINETUNED_DIR, OUTPUTS_DIR, INDEX_DIR,
           OUTPUTS_DIR / "metrics", OUTPUTS_DIR / "charts", OUTPUTS_DIR / "logs",
           OUTPUTS_DIR / "benchmarks"):
    _d.mkdir(parents=True, exist_ok=True)


def _to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _to_namespace(v) if isinstance(v, dict) else v)
    return ns


@dataclass
class Config:
    """Dot-accessible view of configs/*.yaml, e.g. cfg.model.name, cfg.rag.top_k."""
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for section, value in self.raw.items():
            setattr(self, section, _to_namespace(value) if isinstance(value, dict) else value)

    def get(self, dotted_path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted_path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


def load_config(path: str | Path = ROOT_DIR / "configs" / "default.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw=raw)


def load_named_config(name: str) -> Config:
    """Convenience loader for the other named configs, e.g. load_named_config('benchmark')."""
    return load_config(ROOT_DIR / "configs" / f"{name}.yaml")


# Module-level default, imported as `from src.config_loader import CFG` by the rest
# of the pipeline. Reload with `CFG = load_config(other_path)` if you need a variant
# (e.g. benchmark.yaml) inside a script.
CFG = load_config()

# --- Environment overrides for secrets, never stored in YAML -----------------
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
