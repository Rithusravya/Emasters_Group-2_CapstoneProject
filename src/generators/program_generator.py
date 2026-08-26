import logging
import torch
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from typing import Any

logger = logging.getLogger(__name__)


class GenerationPipeline:
    def __init__(self, model: Any, tokenizer: Any, config: Any = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def _build_generation_kwargs(self) -> dict:
        """Builds generation kwargs based on config or defaults."""
        # Safely extract token IDs as standard Python ints to prevent PyTorch warnings
        eos_id = self.tokenizer.eos_token_id
        if isinstance(eos_id, torch.Tensor):
            eos_id = eos_id.item()

        kwargs = {
            "max_new_tokens": 128,  # Hard cap: MongoDB queries are short
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True,
            "pad_token_id": eos_id,
            "eos_token_id": eos_id,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,  # CRITICAL: Prevents repeating blocks of numbers/text
        }
        if self.config:
            kwargs["max_new_tokens"] = min(getattr(self.config, "max_length", 256), 128)
            kwargs["temperature"] = getattr(self.config, "temperature", 0.2)
            kwargs["top_p"] = getattr(self.config, "top_p", 0.95)
            kwargs["top_k"] = getattr(self.config, "top_k", 50)
        return kwargs

    def generate(self, prompt: str) -> str:
        """Generates text from a given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self._build_generation_kwargs())

        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        generated_text = generated_text.strip()

        generated_text = self._clean_output(generated_text)
        return generated_text

    @staticmethod
    def _clean_output(text: str) -> str:
        """Remove hallucinated refusals or trailing junk after the actual answer."""
        # Common patterns that indicate the model went off-track
        stop_patterns = [
            "I'm sorry", "I can't", "I cannot", "I am sorry",
            "I apologize", "As an AI", "As a language model",
            "```", "\n\n", "###", "Human:", "User:", "Assistant:"
        ]
        for pattern in stop_patterns:
            idx = text.find(pattern)
            if idx > 0:  # Only cut if it's not at the very start
                text = text[:idx]
        return text.strip()
