"""
Step 3 — Load CodeGen-350M Model (tokenizer + weights), with VRAM pre-allocation
and dynamic device mapping. Provides generate() for text/code generation and
embed() for retrieval embeddings used later by the RAG indexing stages.
"""
from __future__ import annotations

from typing import List, Optional

from src.config_loader import CFG, CHECKPOINT_DIR
from src.models.tokenizer import load_tokenizer


class CodeGenModel:
    """Loads a causal-LM code model and exposes generate/embed methods.

    Import of torch/transformers is deferred to __init__ so this module can be
    imported (e.g. by tests) without those heavy deps installed.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        model_name = model_name or CFG.model.name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        num_return_sequences: int = 1,
    ) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=CFG.model.max_length).to(self.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [d[len(prompt):].strip() if d.startswith(prompt) else d.strip() for d in decoded]

    def embed(self, texts: List[str]) -> "list[list[float]]":
        """Mean-pooled last-hidden-state embedding — used when the RAG index is
        built with embeddings from the small code-LM itself (Task 3.2)."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True,
                                 truncation=True, max_length=CFG.model.max_length).to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        return pooled.cpu().tolist()


def load_finetuned(checkpoint_path: str, base_model_name: Optional[str] = None) -> CodeGenModel:
    """Load a LoRA-fine-tuned checkpoint on top of the base small code-LM."""
    from peft import PeftModel

    lm = CodeGenModel(model_name=base_model_name)
    lm.model = PeftModel.from_pretrained(lm.model, checkpoint_path).to(lm.device)
    lm.model.eval()
    return lm


def latest_checkpoint(subdir: str = "subset") -> Optional[str]:
    """Convenience helper: resolve models/checkpoints/<subdir> if it exists."""
    path = CHECKPOINT_DIR / subdir
    return str(path) if path.exists() else None
