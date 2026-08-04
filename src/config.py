import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any, Dict
import yaml

logger = logging.getLogger(__name__)

@dataclass
class PathConfig:
    data_dir: Path = Path("../data/raw")
    checkpoints_dir: Path = Path("../models/checkpoints")
    lora_save_path: Path = Path("../models/checkpoints/lora_finetuned")
    faiss_index_path: Path = Path("data/indices/faiss_index.bin")

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.checkpoints_dir = Path(self.checkpoints_dir)
        self.lora_save_path = Path(self.lora_save_path)
        self.faiss_index_path = Path(self.faiss_index_path)


@dataclass
class ModelConfig:
    base_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    embedder_model_name: str = "BAAI/bge-small-en-v1.5"


# ============================================================
# LoRA Configuration
# ============================================================
@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    batch_size: int = 2
    learning_rate: float = 3e-4
    epochs: int = 2
    output_dir: str = "models/checkpoints/lora_finetuned"


@dataclass
class EmbeddingConfig:
    max_length: int = 512
    batch_size: int = 32
    query_instruction: str = "Represent this sentence for searching relevant code: "


@dataclass
class RAGConfig:
    top_k: int = 3


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.8
    top_k: int = 20
    do_sample: bool = True


@dataclass
class EvaluationConfig:
    bertscore_model: str = "microsoft/codebert-base"
    timeout_seconds: int = 3
    # CodeBERTScore device, independent of the main model's device.
    # "cpu" for now; set to "cuda" once GPU evaluation is needed.
    device: str = "cpu"


class PipelineConfig:
    """Master Pipeline Configuration loaded from YAML."""

    def __init__(self, yaml_path: Optional[str] = None):
        if yaml_path is None:
            # Resolve relative to project root
            base_path = Path(__file__).resolve().parent.parent.parent
            yaml_path = base_path / "configs" / "config.yaml"
        self.yaml_path = Path(yaml_path)
        self._load_yaml()

    def _load_yaml(self):
        if not self.yaml_path.exists():
            logger.warning(
                f"Config file not found at {self.yaml_path}. Falling back to default settings."
            )
            data = {}
        else:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        self.paths = PathConfig(**data.get("paths", {}))
        self.models = ModelConfig(**data.get("models", {}))
        self.lora = LoRAConfig(**data.get("lora", {}))
        self.embedding = EmbeddingConfig(**data.get("embedding", {}))
        self.rag = RAGConfig(**data.get("rag", {}))
        self.generation = GenerationConfig(**data.get("generation", {}))
        self.evaluation = EvaluationConfig(**data.get("evaluation", {}))
        logger.info(f"Loaded configuration from {self.yaml_path}")


def load_config(yaml_path: Optional[str] = None) -> PipelineConfig:
    """Helper function to return a PipelineConfig instance."""
    return PipelineConfig(yaml_path=yaml_path)
