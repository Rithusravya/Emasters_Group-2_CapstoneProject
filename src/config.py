import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Any, Type, TypeVar
import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _build(dataclass_type: Type[T], data: dict) -> T:
    valid_keys = {f.name for f in fields(dataclass_type)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return dataclass_type(**filtered)


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
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    dropout: float = 0.05
    batch_size: int = 1
    learning_rate: float = 2e-4
    epochs: int = 1
    output_dir: str = "models/checkpoints/lora_finetuned"
    trainable_fraction: float = 1.0

@dataclass
class EmbeddingConfig:
    source: str = "external"  # 'code_lm' or 'external'
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
class LLMBaselineConfig:
    backend: str = "api"  # 'local' or 'api'
    local_model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    api_model: str = "gemini-3.6-flash"
    api_key_env: str = "AQ.Ab8RN6Ix4XRtM1cRSu3_KD7opq0fzJmOvq7TGXgLhofNnv1-QQ"
    max_new_tokens: int = 256
    temperature: float = 0.2

@dataclass
class GenerationConfig:
    max_length: int = 512
    temperature: float = 0.2
    top_k: int = 50
    top_p: float = 0.95
    num_return_sequences: int = 1

@dataclass
class EvaluationConfig:
    device: str = "cpu"
    bertscore_model: str = "microsoft/codebert-base"
    bertscore_local_path: str = "models/codebert-base"
    timeout_seconds: int = 3

    def resolve_bertscore_model(self, base_dir: Optional[Path] = None) -> str:
        candidates = []
        local = Path(self.bertscore_local_path)
        if base_dir is not None:
            candidates.append(Path(base_dir) / self.bertscore_local_path)
        candidates.append(local)
        # Fallback: resolve relative to this config.py's own directory (the
        # project root), since that's where `models/codebert-base` is
        # expected to live per the project layout.
        candidates.append(Path(__file__).resolve().parent / self.bertscore_local_path)

        for candidate in candidates:
            if candidate.is_dir() and (candidate / "config.json").exists():
                return str(candidate.resolve())

        logger.info(
            f"Local CodeBERT checkpoint not found at '{self.bertscore_local_path}' "
            f"(tried: {[str(c) for c in candidates]}); falling back to "
            f"downloading '{self.bertscore_model}' from the Hugging Face Hub."
        )
        return self.bertscore_model

@dataclass
class OutputConfig:
    logs_dir: str = "output/logs"
    metrics_path: str = "output/metrics/evaluation_metrics.json"
    plots_dir: str = "output/plots"
    generated_dir: str = "outputs/generated"

@dataclass
class IndexingConfig:
    top_k: int = 3
    vector_dim: int = 1024

class PipelineConfig:
    def __init__(self, yaml_path: Optional[str] = None):
        if yaml_path is None:
            base_path = Path(__file__).resolve().parent
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

        path_data = data.get('data_paths', {})
        idx_data = data.get('indices_paths', {})
        self.paths = PathConfig(
            data_dir=Path(path_data.get('raw_dir', "data/raw")),
            faiss_index_path=Path(idx_data.get('faiss_index', "data/indices/faiss_semantic.index")),
            ast_store_path=Path(idx_data.get('ast_store', "data/indices/ast_store.pkl"))
        )

        self.model = ModelConfig(
            name_or_path=data.get('model_name', "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
            embedding_name=data.get('embedding_model', "BAAI/bge-small-en-v1.5")
        )

        self.lora = _build(LoRAConfig, data.get('lora', {}))
        self.embedding = _build(EmbeddingConfig, data.get('embedding', {}))
        self.retrieval = _build(RetrievalConfig, data.get('retrieval', {}))
        self.llm_baseline = _build(LLMBaselineConfig, data.get('llm_baseline', {}))
        self.generation = _build(GenerationConfig, data.get('generation', {}))
        self.evaluation = _build(EvaluationConfig, data.get('evaluation', {}))
        self.outputs = _build(OutputConfig, data.get('outputs', {}))
        self.indexing = _build(IndexingConfig, data.get('indexing', {}))

        # Top-level attributes for backward compatibility
        self.model_name = self.model.name_or_path
        self.embedding_model = self.model.embedding_name
        self.data_paths = self.paths
        self.indices_paths = self.paths

def load_config(yaml_path: Optional[str] = None) -> PipelineConfig:
    return PipelineConfig(yaml_path=yaml_path)
