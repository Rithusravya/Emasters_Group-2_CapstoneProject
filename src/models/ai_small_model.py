from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

class CodeGenModelWrapper:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        device: str = "auto",
        max_length: int = 512,
        use_lora: bool = True,
        lora_kwargs = None,
    ):
        self.max_length = max_length
        self.model_name = model_name
        self.device = (
            "cuda" if device in ("cuda", "auto") and torch.cuda.is_available()
            else "mps" if device in ("mps", "auto") and torch.backends.mps.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.use_lora = use_lora
        if use_lora:
            cfg = lora_kwargs or {}
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.get("r", 16),
                lora_alpha=cfg.get("lora_alpha", 32),
                lora_dropout=cfg.get("lora_dropout", 0.05),
                target_modules=cfg.get("target_modules", ["qkv_proj"]),
            )
            self.model = get_peft_model(base_model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = base_model

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        do_sample: Optional[bool] = None,
    ) -> List[str]:
        """Generate `num_return_sequences` completions for `prompt`. Returns the
        continuation only (prompt text stripped from the front of each output)."""
        if do_sample is None:
            do_sample = num_return_sequences > 1

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        completions = []
        for seq in outputs:
            text = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            completions.append(text.strip())
        return completions

    # kept for notebook/demo compatibility with the single-string call style
    def generate_code(self, prompt: str, max_new_tokens: int = 64) -> str:
        return self.generate(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1)[0]

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    def _underlying_model(self):
        return self.model.base_model.model if hasattr(self.model, "base_model") else self.model

    def embed(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Mean-pool the last hidden state as a code/text embedding. Returns an
        (len(texts), hidden_dim) float32 array."""
        model = self._underlying_model()
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            ).to(self.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]                     # (B, T, H)
                mask = inputs["attention_mask"].unsqueeze(-1)          # (B, T, 1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors.append(pooled.float().cpu().numpy())
        return np.concatenate(vectors, axis=0).astype("float32")

    def extract_embeddings(self, text: str) -> np.ndarray:
        """Single-text convenience wrapper around embed(), returns a 1D vector."""
        return self.embed([text])[0]

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def save_lora_weights(self, output_dir: str) -> None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_lora_weights(self, checkpoint_dir: str) -> None:
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model = PeftModel.from_pretrained(base_model, checkpoint_dir).to(self.device)