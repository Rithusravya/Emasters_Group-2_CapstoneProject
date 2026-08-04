import logging
from typing import List, Union, Any
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class CodeEmbedder:
    """Generates vector embeddings for code and queries with fp16/bf16 precision and batching."""

    def __init__(
            self,
            model_name: Union[str, Any] = "BAAI/bge-small-en-v1.5",
            device: str = None,
            dtype: torch.dtype = torch.float16,
            tokenizer: AutoTokenizer = None,
            query_instruction: str = "Represent this sentence for searching relevant code: "
    ):
        self.query_instruction = query_instruction

        # 1. Resolve Computing Device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Use float16/bf16 only on CUDA; default to float32 for CPU/MPS stability
        if self.device == "cuda":
            self.dtype = torch.float16
        elif self.device == "mps":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # 2. Resolve Model & Tokenizer
        if isinstance(model_name, str):
            logger.info(f"Loading embedder model '{model_name}' on {self.device}...")
            self.model_name = model_name
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(
                model_name, torch_dtype=self.dtype
            ).to(self.device)
        else:
            # Pre-loaded model object passed in
            self.model = model_name.to(self.device)
            self.model_name = getattr(model_name, "name_or_path", "BAAI/bge-small-en-v1.5")

            # Fallback tokenizer loading if tokenizer was omitted
            if tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                logger.warning(
                    f"No tokenizer passed with pre-loaded model. Loading default for '{self.model_name}'."
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model.eval()

    def _extract_text_field(self, item: dict) -> str:
        """Helper to extract text/code strings across different dataset formats."""
        candidate_keys = ["code", "SQL", "sql", "query", "question", "text", "instruction", "docstring"]
        for key in candidate_keys:
            if key in item and isinstance(item[key], str):
                return item[key]
        return str(item)

    def _mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean Pooling - Take attention mask into account for correct averaging."""
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(self.dtype)
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def generate_embedding(
            self,
            data: Union[str, List[str], pd.DataFrame, List[dict]],
            is_query: bool = False,
            normalize: bool = True,
            batch_size: int = 4
    ) -> np.ndarray:
        """Parses input dataset/queries and generates normalized float32 embeddings."""

        # 1. Parse Input Data Structures into standard List[str]
        if isinstance(data, pd.DataFrame):
            # Check candidate columns in DataFrame
            col_match = next((c for c in ["code", "SQL", "sql", "query", "question", "text"] if c in data.columns),
                             None)
            if col_match:
                texts = data[col_match].astype(str).tolist()
            else:
                texts = data.iloc[:, 0].astype(str).tolist()
        elif isinstance(data, list):
            if len(data) == 0:
                return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
            if isinstance(data[0], dict):
                texts = [self._extract_text_field(item) for item in data]
            else:
                texts = [str(x) for x in data]
        elif isinstance(data, str):
            texts = [data]
        else:
            texts = [str(data)]

        # 2. Add Query Instruction Prefix if applicable
        if is_query and self.query_instruction:
            processed_texts = [f"{self.query_instruction}{t}" for t in texts]
        else:
            processed_texts = texts

        all_embeddings = []

        # 3. Batched Vector Generation
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i: i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                sentence_embeddings = self._mean_pooling(outputs, inputs["attention_mask"])

                if normalize:
                    sentence_embeddings = torch.nn.functional.normalize(
                        sentence_embeddings, p=2, dim=1
                    )

                all_embeddings.append(sentence_embeddings.to(torch.float32).cpu().numpy())

        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
