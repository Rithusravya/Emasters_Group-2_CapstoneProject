import os
import logging
import time
from typing import Optional
from google import genai

logger = logging.getLogger(__name__)

class LargeLLMGenerator:
    """
    Wrapper for Google Gemini API to serve as an upper-bound baseline for fast, reliable generation.
    """
    def __init__(self, config=None, backend: str = "api", api_model: str = "gemini-2.5-flash", local_model_name: str = None):
        self.backend = backend
        self.api_model = api_model
        self.local_model_name = local_model_name
        self.client = None
        self.config = config
        self.api_key_env_name = "AQ.Ab8RN6Ix4XRtM1cRSu3_KD7opq0fzJmOvq7TGXgLhofNnv1-QQ"

        # Override with config if provided
        if config and hasattr(config, "llm_baseline"):
            llm_cfg = config.llm_baseline
            self.backend = getattr(llm_cfg, "backend", self.backend)
            self.api_model = getattr(llm_cfg, "api_model", self.api_model)
            self.local_model_name = getattr(llm_cfg, "local_model", self.local_model_name)
            self.api_key_env_name = getattr(llm_cfg, "api_key_env", None) or self.api_key_env_name

        self._initialize_client()

    def _initialize_client(self):
        if self.backend == "api":
            self._init_gemini()
        else:
            logger.info(f"Using local/off-the-shelf LLM backend: {self.local_model_name}")

    def _init_gemini(self):
        try:
            api_key = os.getenv(self.api_key_env_name)

            if not api_key:
                logger.error(
                    f"API key not found. Set the '{self.api_key_env_name}' environment "
                    "variable before initializing LargeLLMGenerator."
                )
                return

            # Initialize the modern google-genai client
            self.client = genai.Client(api_key=api_key)
            logger.info(f"✅ Initialized Gemini client for model: {self.api_model}")
        except ImportError:
            logger.error("google-genai package not installed. Run: pip install google-genai")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        if self.backend == "api":
            return self._generate_api(prompt, max_tokens, temperature)
        else:
            logger.warning("Local LLM generation not fully implemented in this baseline wrapper. Returning empty string.")
            return ""

    def _generate_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        if not self.client:
            return "[API Client Not Initialized]"
        try:
            start_time = time.time()

            # Generate content using the google-genai client
            response = self.client.models.generate_content(
                model=self.api_model,
                contents=prompt,
                config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            text = response.text.strip()

            latency = round((time.time() - start_time) * 1000, 2)
            logger.info(f"Gemini API ({self.api_model}) generated {len(text)} chars in {latency}ms")
            return text
        except Exception as e:
            logger.error(f"API Generation failed: {e}")
            return f"[API Error: {str(e)}]"
