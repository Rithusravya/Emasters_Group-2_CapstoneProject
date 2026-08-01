"""
Tokenizer loader — split out from the model wrapper so the tokenizer can be
loaded/inspected independently (e.g. for prompt-length checks or dataset
pre-tokenization) without pulling in the full causal-LM weights.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase


class CodeTokenizer:
    """Thin wrapper around a HF tokenizer that guarantees a pad token exists
    (many small code-LM checkpoints, e.g. codegen-350M, ship without one)."""

    def __init__(self, model_name: str, max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(self, text: str, truncation: bool = True) -> List[int]:
        return self.tokenizer.encode(text, truncation=truncation, max_length=self.max_length)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(self, texts: List[str], padding: bool = True):
        """Returns a dict of tensors (input_ids, attention_mask, ...) ready for
        model.generate() / model(**inputs)."""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=padding,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))
