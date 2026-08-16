import logging
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)

# Fields checked, in order, when pulling a display string out of retrieved metadata.
METADATA_TEXT_FIELDS = ["code", "SQL", "question", "text"]


class RAGPipeline:
    def __init__(self, model: Any, tokenizer: Any, embedder: Any, index_manager: Any, config: Any = None):
        self.model = model
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.index_manager = index_manager
        self.config = config

    def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        query_embedding = self.embedder.generate_embedding(query, is_query=True)
        results = self.index_manager.search(query_embedding, k=top_k)
        return [self._metadata_to_text(metadata) for metadata, _score in results]

    @staticmethod
    def _metadata_to_text(metadata: Any) -> str:
        if not isinstance(metadata, dict):
            return str(metadata)
        for field in METADATA_TEXT_FIELDS:
            if metadata.get(field):
                return metadata[field]
        return str(metadata)

    def build_prompt(self, query: str, context: List[str]) -> str:
        context_str = "\n\n".join(f"--- Reference {i + 1} ---\n{c}" for i, c in enumerate(context))
        return (
            f"You are an expert code assistant. Use the following references if helpful.\n\n"
            f"Context:\n{context_str}\n\n"
            f"User Question: {query}\n\n"
            f"Answer:"
        )

    def generate_program(self, prompt: str):
        generated_text, _ = self.generate_with_rag(query=prompt, top_k=3)
        return generated_text

    def _build_generation_kwargs(self) -> dict:
        return {
            "max_new_tokens": getattr(self.config, "max_length", 256) if self.config else 256,
            "temperature": getattr(self.config, "temperature", 0.2) if self.config else 0.2,
            "top_p": getattr(self.config, "top_p", 0.8) if self.config else 0.8,
            "top_k": getattr(self.config, "top_k", 20) if self.config else 20,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

    def generate_with_rag(self, query: str, top_k: int = 1) -> Tuple[str, List[str]]:
        context = self.retrieve_context(query, top_k=top_k)
        prompt = self.build_prompt(query, context)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, **self._build_generation_kwargs())
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        return generated_text.strip(), context
