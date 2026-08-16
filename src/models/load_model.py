import logging
from pathlib import Path
from typing import Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Default LoRA target modules for Qwen / LLaMA-style architectures.
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class ModelLoader:
    def __init__(self, config, dtype: torch.dtype = torch.float16):
        self.config = config
        self.base_model = None
        self.lora_model = None
        self.tokenizer = None
        self.device, self.dtype = self._resolve_device_and_dtype(dtype)
        logger.info(f"Using device: {self.device} with dtype: {self.dtype}")

    @staticmethod
    def _resolve_device_and_dtype(requested_dtype: torch.dtype):
        if torch.cuda.is_available():
            dtype = (
                torch.bfloat16
                if (requested_dtype == torch.bfloat16 and torch.cuda.is_bf16_supported())
                else torch.float16
            )
            return "cuda", dtype
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def load_tokenizer(self):
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_base_model(self):
        logger.info(f"Loading BASE model {self.config.model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.base_model.to(self.device)
        self.base_model.eval()
        return self.base_model

    def setup_lora_training(self, model=None, r=8, alpha=32, dropout=0.05, target_modules=None):
        target_model = model if model is not None else self.base_model
        if target_model is None:
            raise ValueError("Base model must be loaded before setting up LoRA.")

        target_modules = target_modules or self._resolve_lora_target_modules()

        logger.info(f"Configuring LoRA (r={r}, alpha={alpha}) for target modules: {target_modules}")
        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.lora_model = get_peft_model(target_model, peft_config)
        self.lora_model.print_trainable_parameters()
        return self.lora_model

    def _resolve_lora_target_modules(self):
        if hasattr(self.config, "lora") and hasattr(self.config.lora, "target_modules"):
            return self.config.lora.target_modules
        return DEFAULT_LORA_TARGET_MODULES

    def train_lora(
        self,
        lora_model,
        tokenizer,
        train_examples,  # list of (prompt, completion) tuples, not flat strings
        epochs: int = 3,
        batch_size: int = 1,
        grad_accum_steps: int = 4,
        learning_rate: float = 1e-4,  # lowered from 3e-4 - less risk of destabilizing a tiny model
        max_length: int = 256,
    ):
        if not train_examples:
            logger.warning("No training examples provided; skipping LoRA training.")
            return lora_model

        device = next(lora_model.parameters()).device
        is_mps = device.type == "mps"
        self._prepare_model_for_training(lora_model, is_mps)

        encoded = [self._build_training_example(tokenizer, p, c, max_length) for p, c in train_examples]
        loader = DataLoader(
            encoded, batch_size=batch_size, shuffle=True,
            collate_fn=lambda batch: self._collate_batch(batch, tokenizer.pad_token_id),
        )
        optimizer = torch.optim.AdamW(
            (p for p in lora_model.parameters() if p.requires_grad), lr=learning_rate
        )

        logger.info(
            f"Starting LoRA training: {len(encoded)} examples, {epochs} epochs, "
            f"batch_size={batch_size}, grad_accum={grad_accum_steps}, lr={learning_rate}, device={device}"
        )

        for epoch in range(epochs):
            avg_loss = self._run_training_epoch(lora_model, loader, optimizer, device, grad_accum_steps, is_mps)
            logger.info(f"  epoch {epoch + 1}/{epochs} - avg loss: {avg_loss:.4f}")
            if is_mps:
                torch.mps.empty_cache()

        lora_model.eval()
        return lora_model

    @staticmethod
    def _prepare_model_for_training(lora_model, is_mps: bool) -> None:
        import os

        if is_mps:
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        try:
            lora_model.gradient_checkpointing_enable()
            lora_model.enable_input_require_grads()
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")
        lora_model.train()

    @staticmethod
    def _build_training_example(tokenizer, prompt: str, completion: str, max_length: int) -> dict:
        full_text = prompt + completion + tokenizer.eos_token
        full_ids = tokenizer(full_text, truncation=True, max_length=max_length)["input_ids"]
        prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
        prompt_len = min(len(prompt_ids), len(full_ids))

        labels = full_ids.copy()
        for i in range(prompt_len):
            labels[i] = -100

        return {"input_ids": full_ids, "labels": labels}

    @staticmethod
    def _collate_batch(batch, pad_id: int) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, attn_mask, labels = [], [], []

        for ex in batch:
            pad_n = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_id] * pad_n)
            attn_mask.append([1] * len(ex["input_ids"]) + [0] * pad_n)
            labels.append(ex["labels"] + [-100] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn_mask),
            "labels": torch.tensor(labels),
        }

    @staticmethod
    def _run_training_epoch(lora_model, loader, optimizer, device, grad_accum_steps: int, is_mps: bool) -> float:
        total_loss, n_steps = 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            try:
                outputs = lora_model(**batch)
                loss = outputs.loss / grad_accum_steps
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"OOM at step {step}; skipping batch.")
                    optimizer.zero_grad()
                    del batch
                    if is_mps:
                        torch.mps.empty_cache()
                    continue
                raise

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            n_steps += 1
            del batch, outputs, loss
            if is_mps and (step + 1) % 5 == 0:
                torch.mps.empty_cache()

        # Flush any accumulated gradients from a partial final batch.
        optimizer.step()
        optimizer.zero_grad()
        return total_loss / max(n_steps, 1)

    def load_lora_model(self, adapter_path: str):
        adapter_path = Path(adapter_path)
        if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
            logger.warning(f"Invalid LoRA path or missing config: {adapter_path}")
            return None

        if hasattr(self.base_model, "peft_config"):
            logger.warning(
                "self.base_model already has a LoRA adapter injected in place "
                "(e.g. via setup_lora_training in this session). Re-wrapping it "
                "here would compare the adapted model against itself. Returning "
                "the existing in-memory LoRA model instead - use "
                "generate_base_vs_lora() to correctly isolate base vs. adapted "
                "outputs from a single trained model."
            )

            if self.lora_model is None or not isinstance(self.lora_model, PeftModel):
                raise RuntimeError(
                    "self.base_model carries an injected LoRA adapter but no "
                    "corresponding PeftModel wrapper was found in self.lora_model. "
                    "This should not happen if setup_lora_training() was called "
                    "normally - check the call order."
                )
            self.lora_model.eval()
            return self.lora_model

        logger.info(f"Loading LoRA adapter from {adapter_path}")
        self.lora_model = PeftModel.from_pretrained(self.base_model, str(adapter_path), torch_dtype=self.dtype)
        self.lora_model.eval()
        return self.lora_model

    def generate_base_vs_lora(
        self,
        lora_model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> Tuple[str, str]:
        device = next(lora_model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

        lora_model.eval()
        with torch.no_grad():
            # Adapter OFF -> genuine base-model behavior on the shared weights.
            with lora_model.disable_adapter():
                base_ids = lora_model.generate(**inputs, **gen_kwargs)
            # Adapter ON (default) -> LoRA-adapted behavior.
            lora_ids = lora_model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        base_text = tokenizer.decode(base_ids[0][prompt_len:], skip_special_tokens=True).strip()
        lora_text = tokenizer.decode(lora_ids[0][prompt_len:], skip_special_tokens=True).strip()
        return base_text, lora_text

    def load_models(self, lora_path=None):
        if self.tokenizer is None:
            self.load_tokenizer()
        if self.base_model is None:
            self.load_base_model()

        models = {"base": self.base_model, "lora": None}
        if lora_path and Path(lora_path).exists():
            models["lora"] = self.load_lora_model(lora_path)
        return models, self.tokenizer
