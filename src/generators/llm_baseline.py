import json
import logging
import os
import urllib.error
import urllib.request
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_LLM = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

# Default models and environment variables for each provider
DEFAULT_MODELS = {
    "openai": "gpt-5.4-mini",
    "gemini": "gemini-3.7-flash"
}

DEFAULT_API_KEYS = {
    "openai": "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",
    "gemini": "AIzaSyC_rcB6oFqk1e8Oixf1vn74x-L5dir_HSk"  # or GOOGLE_API_KEY
}

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
}


class LargeLLMGenerator:
    """
    Supports multiple LLM backends:
    - "openai": OpenAI API (and compatible endpoints like DeepSeek/Groq)
    - "gemini": Google Gemini API
    - "anthropic": Anthropic Claude API
    - "local": Local Hugging Face causal LM via transformers
    """

    def __init__(self, config: Optional[object] = None):
        cfg = getattr(config, "llm_baseline", None)

        # Determine backend/provider (defaults to local)
        self.backend = getattr(cfg, "backend", "local") if cfg else "local"

        # If legacy "api" backend is specified, map to "openai" or check provider config
        if self.backend == "api":
            self.backend = getattr(cfg, "provider", "openai")

        # Set provider-specific defaults
        default_model = DEFAULT_MODELS.get(self.backend, DEFAULT_MODELS["openai"])
        default_key_env = DEFAULT_API_KEYS.get(self.backend, "OPENAI_API_KEY")
        default_url = DEFAULT_BASE_URLS.get(self.backend, DEFAULT_BASE_URLS["openai"])

        self.api_model = getattr(cfg, "api_model", default_model) if cfg else default_model
        self.api_key_env = getattr(cfg, "api_key_env", default_key_env) if cfg else default_key_env
        self.api_base_url = getattr(cfg, "api_base_url", default_url) if cfg else default_url

        self.local_model_name = getattr(cfg, "local_model", DEFAULT_LOCAL_LLM) if cfg else DEFAULT_LOCAL_LLM
        self.max_new_tokens = getattr(cfg, "max_new_tokens", 256) if cfg else 256
        self.temperature = getattr(cfg, "temperature", 0.2) if cfg else 0.2

        self._local_model = None
        self._local_tokenizer = None

        # Validate API key availability if using an API backend
        if self.backend in DEFAULT_MODELS:
            # Fallback check for alternative Gemini key names
            if self.backend == "gemini" and not os.environ.get(self.api_key_env) and os.environ.get("GOOGLE_API_KEY"):
                self.api_key_env = "GOOGLE_API_KEY"

            if not os.environ.get(self.api_key_env):
                logger.warning(
                    f"llm_baseline.backend='{self.backend}' but ${self.api_key_env} is not set; "
                    "falling back to backend='local'."
                )
                self.backend = "local"

    # -------------------------------------------------------------------
    # API Handlers
    # -------------------------------------------------------------------
    def _generate_openai(self, prompt: str) -> str:
        api_key = os.environ[self.api_key_env]
        payload = json.dumps({
            "model": self.api_model,
            "messages": [
                {"role": "system", "content": "You are an expert software engineer."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }).encode("utf-8")

        request = urllib.request.Request(
            self.api_base_url,
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"OpenAI API call failed ({self.api_model}): {e}")
            return ""

    def _generate_gemini(self, prompt: str) -> str:
        api_key = os.environ[self.api_key_env]
        url = self.api_base_url.format(model=self.api_model) + f"?key={api_key}"

        payload = json.dumps({
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"You are an expert software engineer.\n\n{prompt}"}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": self.max_new_tokens,
                "temperature": self.temperature,
            }
        }).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Gemini API call failed ({self.api_model}): {e}")
            return ""

    # -------------------------------------------------------------------
    # Local backend
    # -------------------------------------------------------------------
    def _load_local_model(self):
        if self._local_model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading large local LLM baseline: {self.local_model_name}")
        self._local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_name, trust_remote_code=True)
        if self._local_tokenizer.pad_token is None:
            self._local_tokenizer.pad_token = self._local_tokenizer.eos_token

        device_map = "auto" if torch.cuda.is_available() else None
        self._local_model = AutoModelForCausalLM.from_pretrained(
            self.local_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map=device_map,
        )
        if device_map is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._local_model.to(device)
        self._local_model.eval()

    def _generate_local(self, prompt: str) -> str:
        import torch

        self._load_local_model()
        device = next(self._local_model.parameters()).device
        inputs = self._local_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            output_ids = self._local_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._local_tokenizer.eos_token_id,
            )

        generated = self._local_tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return generated.strip()

    # -------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        if self.backend == "openai":
            return self._generate_openai(prompt)
        elif self.backend == "gemini":
            return self._generate_gemini(prompt)
        else:
            return self._generate_local(prompt)

    def generate_batch(self, prompts: List[str]) -> List[str]:
        return [self.generate(p) for p in prompts]