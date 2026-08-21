import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any
import yaml

logger = logging.getLogger(__name__)


def _get_nested(data: dict, key_path: str, default=None):
    """Helper to get value from nested dict using dot notation."""
    keys = key_path.split('.')
    val = data
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, default)
        else:
            return default
    return val


@dataclass
class PathConfig:
    data_dir: Path = Path("data/raw")
    checkpoints_dir: Path = Path("models/checkpoints")
    lora_save_path: Path = Path("models/checkpoints/lora_finetuned")
    faiss_index_path: Path = Path("data/indices/faiss_semantic.index")
    ast_store_path: Path = Path("data/indices/ast_store.pkl")


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    embedding_name: str = "BAAI/bge-small-en-v1.5"


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    dropout: float = 0.05
    batch_size: int = 4
    learning_rate: float = 3e-4
    epochs: int = 5
    output_dir: str = "models/checkpoints/lora_finetuned"


@dataclass
class EmbeddingConfig:
    source: str = "code_lm"  # 'code_lm' or 'external'
    external_model: str = "BAAI/bge-small-en-v1.5"
    max_length: int = 512
    batch_size: int = 32


@dataclass
class RetrievalConfig:
    hybrid: bool = True
    structural_language: str = "sql"
    structural_weight_alpha: float = 0.7
    topk_sweep: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 8])


@dataclass
class GenerationConfig:
    max_length: int = 512
    temperature: float = 0.2
    top_k: int = 50
    top_p: float = 0.95


@dataclass
class EvaluationConfig:
    device: str = "cpu"
    bertscore_model: str = "microsoft/codebert-base"


@dataclass
class OutputConfig:
    logs_dir: str = "output/logs"
    metrics_path: str = "output/metrics/evaluation_metrics.json"
    plots_dir: str = "output/plots"


class PipelineConfig:
    def __init__(self, yaml_path: Optional[str] = None):
        if yaml_path is None:
            base_path = Path(__file__).resolve().parent.parent
            yaml_path = base_path / "configs" / "config.yaml"

        self.yaml_path = Path(yaml_path)
        self._load_yaml()

    def _load_yaml(self):
        if self.yaml_path.exists():
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Config not found at {self.yaml_path}. Using defaults.")
            data = {}

        # Map nested YAML to flat dataclasses
        self.paths = PathConfig(
            data_dir=data.get('data_paths', {}).get('raw_dir', "data/raw"),
            faiss_index_path=data.get('indices_paths', {}).get('faiss_index', "data/indices/faiss_semantic.index"),
            ast_store_path=data.get('indices_paths', {}).get('ast_store', "data/indices/ast_store.pkl")
        )

        self.model = ModelConfig(
            name_or_path=data.get('model_name', "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
            embedding_name=data.get('embedding_model', "BAAI/bge-small-en-v1.5")
        )

        self.lora = LoRAConfig(**data.get('lora', {}))

        emb_data = data.get('embedding', {})
        self.embedding = EmbeddingConfig(
            source=emb_data.get('source', "code_lm"),
            external_model=emb_data.get('external_model', "BAAI/bge-small-en-v1.5")
        )

        ret_data = data.get('retrieval', {})
        self.retrieval = RetrievalConfig(
            structural_language=ret_data.get('structural_language', "sql"),
            structural_weight_alpha=ret_data.get('structural_weight_alpha', 0.7),
            topk_sweep=ret_data.get('topk_sweep', [1, 2, 3, 5, 8])
        )

        gen_data = data.get('generation', {})
        self.generation = GenerationConfig(**gen_data)

        eval_data = data.get('evaluation', {})
        self.evaluation = EvaluationConfig(**eval_data)

        out_data = data.get('outputs', {})
        self.outputs = OutputConfig(**out_data)


def load_config(yaml_path: Optional[str] = None) -> PipelineConfig:
    return PipelineConfig(yaml_path=yaml_path)