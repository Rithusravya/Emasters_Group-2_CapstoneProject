import json
import logging
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, asdict

import torch
from transformers import GenerationConfig

logger = logging.getLogger(__name__)


class GenerationPipeline:
    """Wraps a causal LM + tokenizer for prompt-based generation."""

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.generation_config = GenerationConfig(
            max_length=getattr(config, "max_length", 512),
            temperature=getattr(config, "temperature", 0.2),
            top_k=getattr(config, "top_k", 50),
            top_p=getattr(config, "top_p", 0.95),
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            repetition_penalty=1.1,
        )

    def generate_program(self, prompt: str, clean_output: bool = True) -> str:
        try:
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if clean_output and prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()

            return generated_text
        except Exception as e:
            logger.error(f"Error during code generation: {e}")
            return ""
