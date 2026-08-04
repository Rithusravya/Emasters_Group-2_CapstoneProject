import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

logger = logging.getLogger(__name__)

class ModelLoader:
    """Loads base models and configures PEFT/LoRA modules on target hardware."""

    def __init__(self, config, dtype: torch.dtype = torch.float16):
        self.config = config
        self.base_model = None
        self.lora_model = None
        self.tokenizer = None

        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = (
                torch.bfloat16
                if (dtype == torch.bfloat16 and torch.cuda.is_bf16_supported())
                else torch.float16
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        logger.info(f"Using device: {self.device} with dtype: {self.dtype}")

    def load_tokenizer(self):
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_fast=True
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
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device != "cuda":
            self.base_model.to(self.device)
        self.base_model.eval()
        return self.base_model

    def setup_lora_training(self, model=None, r=8, alpha=32, dropout=0.05, target_modules=None):
        """
        Configures and wraps base model with LoRA layers using HuggingFace PEFT.
        Supports Qwen, LLaMA, CodeGen, and generic CausalLM architectures.
        """
        target_model = model if model is not None else self.base_model
        if target_model is None:
            raise ValueError("Base model must be loaded before setting up LoRA.")

        # Default fallback target modules for Qwen / LLaMA architectures
        if target_modules is None:
            if hasattr(self.config, "lora") and hasattr(self.config.lora, "target_modules"):
                target_modules = self.config.lora.target_modules
            else:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        logger.info(f"Configuring LoRA (r={r}, alpha={alpha}) for target modules: {target_modules}")
        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.lora_model = get_peft_model(target_model, peft_config)
        self.lora_model.print_trainable_parameters()
        return self.lora_model

    def load_lora_model(self, adapter_path: str):
        adapter_path = Path(adapter_path)
        if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
            logger.warning(f"Invalid LoRA path or missing config: {adapter_path}")
            return None

        logger.info(f"Loading LoRA adapter from {adapter_path}")
        self.lora_model = PeftModel.from_pretrained(
            self.base_model,
            str(adapter_path),
            torch_dtype=self.dtype
        )
        self.lora_model.eval()
        return self.lora_model

    def load_models(self, lora_path=None):
        if self.tokenizer is None:
            self.load_tokenizer()
        if self.base_model is None:
            self.load_base_model()

        models = {"base": self.base_model, "lora": None}
        if lora_path and Path(lora_path).exists():
            models["lora"] = self.load_lora_model(lora_path)
        return models, self.tokenizer
