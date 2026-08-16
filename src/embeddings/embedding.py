"""Generates vector embeddings for code/queries using a HuggingFace encoder
model (e.g. BAAI/bge-small-en-v1.5), with automatic device selection.
"""

import logging
from typing import Any, List, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# Fields checked, in order, when pulling a text/code string out of a dataset item.
TEXT_FIELD_CANDIDATES = ["code", "SQL", "sql", "query", "question", "text", "instruction", "docstring"]


class CodeEmbedder:
    """Generates normalized embeddings for code and natural-language queries."""

    def __init__(
        self,
        model_name: Union[str, Any] = "BAAI/bge-small-en-v1.5",
        device: str = None,
        dtype: torch.dtype = torch.float32,
        tokenizer: AutoTokenizer = None,
        query_instruction: str = "Represent this sentence for searching relevant code: ",
    ):
        self.query_instruction = query_instruction
        self.device = device or self._detect_device()
        self.dtype = self._select_dtype(self.device)
        self.model, self.tokenizer, self.model_name = self._load_model(model_name, tokenizer)
        self.model.eval()

    @staticmethod
    def _detect_device() -> str:
        """Picks the best available device: CUDA > MPS > CPU."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _select_dtype(device: str) -> torch.dtype:
        """Uses float16 on GPU/MPS for speed, float32 on CPU for numerical stability."""
        return torch.float16 if device in ("cuda", "mps") else torch.float32

    def _load_model(self, model_name: Union[str, Any], tokenizer: AutoTokenizer):
        """Loads the model (and tokenizer, if not already provided) onto `self.device`."""
        if isinstance(model_name, str):
            logger.info(f"Loading embedder model '{model_name}' on {self.device}...")
            resolved_tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device)
            return model, resolved_tokenizer, model_name

        # A pre-loaded model instance was passed in directly.
        model = model_name.to(self.device)
        resolved_name = getattr(model_name, "name_or_path", "BAAI/bge-small-en-v1.5")
        if tokenizer is not None:
            resolved_tokenizer = tokenizer
        else:
            logger.warning(
                f"No tokenizer passed with pre-loaded model. Loading default for '{resolved_name}'."
            )
            resolved_tokenizer = AutoTokenizer.from_pretrained(resolved_name)
        return model, resolved_tokenizer, resolved_name

    def _extract_text_field(self, item: dict) -> str:
        """Pulls the first matching text/code field out of a dataset item dict."""
        for key in TEXT_FIELD_CANDIDATES:
            if key in item and isinstance(item[key], str):
                return item[key]
        return str(item)

    def _mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pools token embeddings, weighting out padded positions via the attention mask."""
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(self.dtype)
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _texts_from_dataframe(self, data: pd.DataFrame) -> List[str]:
        """Extracts a text column from a DataFrame, preferring known field names."""
        col_match = next((c for c in TEXT_FIELD_CANDIDATES[:6] if c in data.columns), None)
        if col_match:
            return data[col_match].astype(str).tolist()
        return data.iloc[:, 0].astype(str).tolist()

    def _texts_from_input(self, data: Union[str, List[str], pd.DataFrame, List[dict]]) -> List[str]:
        """Normalizes any supported input type into a flat list of strings."""
        if isinstance(data, pd.DataFrame):
            return self._texts_from_dataframe(data)
        if isinstance(data, list):
            if not data:
                return []
            if isinstance(data[0], dict):
                return [self._extract_text_field(item) for item in data]
            return [str(x) for x in data]
        if isinstance(data, str):
            return [data]
        return [str(data)]

    def generate_embedding(
        self,
        data: Union[str, List[str], pd.DataFrame, List[dict]],
        is_query: bool = False,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encodes `data` into normalized float32 embeddings, batched for efficiency.

        Returns an empty (0, hidden_size) array if `data` contains no items.
        """
        texts = self._texts_from_input(data)
        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)

        if is_query and self.query_instruction:
            texts = [f"{self.query_instruction}{t}" for t in texts]

        batches = [self._embed_batch(texts[i : i + batch_size], normalize) for i in range(0, len(texts), batch_size)]
        return np.vstack(batches)

    def _embed_batch(self, batch_texts: List[str], normalize: bool) -> np.ndarray:
        """Runs the model on a single batch of texts and returns pooled embeddings."""
        inputs = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.to(torch.float32).cpu().numpy()
