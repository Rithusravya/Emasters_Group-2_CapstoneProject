import logging
from typing import Any, List, Union

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Fields checked, in order, when pulling a text/code string out of a dataset item.
TEXT_FIELD_CANDIDATES = ["code", "SQL", "sql", "query", "question", "text", "instruction", "docstring"]


class CodeLMEmbedder:
    """
    Produces retrieval embeddings using the project's own small code LM
    (e.g. Qwen2.5-Coder-0.5B-Instruct) rather than a separate off-the-shelf
    embedding model.

    This directly reuses the already-loaded causal LM + tokenizer (the same
    ones used for generation), so no second model is downloaded. Embeddings
    are produced by mean-pooling the last hidden state of the LM over the
    non-padded tokens, matching the same pooling strategy used elsewhere in
    this project's `CodeEmbedder` so the two are drop-in interchangeable.

    Satisfies the "index + embeddings from the small code LM" requirement
    (Task 3.2), as distinct from CodeEmbedder, which uses an external
    sentence-embedding model (BAAI/bge-small-en-v1.5) and remains available
    for callers who explicitly want that baseline.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = None,
        max_length: int = 512,
        query_instruction: str = "Represent this sentence for searching relevant code: ",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.query_instruction = query_instruction
        self.device = device or next(model.parameters()).device
        self.model_name = getattr(model, "name_or_path", "code-lm")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _extract_text_field(self, item: dict) -> str:
        for key in TEXT_FIELD_CANDIDATES:
            if key in item and isinstance(item[key], str):
                return item[key]
        return str(item)

    def _texts_from_dataframe(self, data: pd.DataFrame) -> List[str]:
        col_match = next((c for c in TEXT_FIELD_CANDIDATES[:6] if c in data.columns), None)
        if col_match:
            return data[col_match].astype(str).tolist()
        return data.iloc[:, 0].astype(str).tolist()

    def _texts_from_input(self, data: Union[str, List[str], pd.DataFrame, List[dict]]) -> List[str]:
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

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def _embed_batch(self, batch_texts: List[str], normalize: bool) -> np.ndarray:
        inputs = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Last transformer block's hidden states, before the LM head.
            last_hidden_state = outputs.hidden_states[-1]
            embeddings = self._mean_pooling(last_hidden_state, inputs["attention_mask"])
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.to(torch.float32).cpu().numpy()

    def generate_embedding(
        self,
        data: Union[str, List[str], pd.DataFrame, List[dict]],
        is_query: bool = False,
        normalize: bool = True,
        batch_size: int = 16,
    ) -> np.ndarray:
        texts = self._texts_from_input(data)
        if not texts:
            hidden_size = getattr(self.model.config, "hidden_size", 896)
            return np.empty((0, hidden_size), dtype=np.float32)

        if is_query and self.query_instruction:
            texts = [f"{self.query_instruction}{t}" for t in texts]

        was_training = self.model.training
        self.model.eval()
        try:
            batches = [
                self._embed_batch(texts[i : i + batch_size], normalize)
                for i in range(0, len(texts), batch_size)
            ]
        finally:
            if was_training:
                self.model.train()

        return np.vstack(batches)
