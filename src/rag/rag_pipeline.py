import logging
import torch
from typing import Any, List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

METADATA_TEXT_FIELDS = ["generated_mongodb_query", "mongodb_query", "code", "sql", "SQL", "question", "text", "schema"]

class RAGPipeline:
    def __init__(
        self, 
        retriever: Any, 
        generator: Optional[Any] = None, 
        model: Optional[Any] = None, 
        tokenizer: Optional[Any] = None, 
        config: Any = None
    ):
        """
        Args:
            retriever: An instance of IndexManager or HybridRetriever that has a `.search(query, k)` method.
            generator: An instance of GenerationPipeline (has `.generate(prompt)`). 
                       If provided, `model` and `tokenizer` are ignored.
            model: HuggingFace model (used if generator is None).
            tokenizer: HuggingFace tokenizer (used if generator is None).
            config: Generation configuration object.
        """
        self.retriever = retriever
        self.generator = generator
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        if self.generator is None and (self.model is None or self.tokenizer is None):
            raise ValueError("Either `generator` or both `model` and `tokenizer` must be provided.")

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves relevant context using the hybrid retriever.
        Returns a list of metadata dictionaries.
        """
        try:
            # The new HybridRetriever/IndexManager returns List[Dict] or List[Tuple[Dict, float]]
            results = self.retriever.search(query, k=top_k)
            
            # Handle different return formats gracefully
            if results and isinstance(results[0], tuple):
                return [metadata for metadata, score in results]
            elif results and isinstance(results[0], dict):
                return results
            else:
                return []
        except Exception as e:
            logger.error(f"Error during context retrieval: {e}")
            return []

    @staticmethod
    def _metadata_to_text(metadata: Any) -> str:
        """Extracts the most relevant text field from a metadata dictionary."""
        if not isinstance(metadata, dict):
            return str(metadata)
        
        for field in METADATA_TEXT_FIELDS:
            if metadata.get(field):
                return str(metadata[field])
        
        # Fallback: concatenate all string values
        return " | ".join([f"{k}: {v}" for k, v in metadata.items() if isinstance(v, str)])

    def build_prompt(self, query: str, context_items: List[Dict[str, Any]]) -> str:
        context_str = ""
        for i, item in enumerate(context_items):
            mongodb_query = (
                    item.get('generated_mongodb_query', '')
                    or item.get('mongodb_query', '')
            )
            question = item.get('question', '')

            if isinstance(mongodb_query, list) and len(mongodb_query) > 0:
                mongodb_query = mongodb_query[0]

            if mongodb_query:
                context_str += f"--- Example {i + 1} ---\n"
                if question:
                    context_str += f"Question: {question}\n"
                context_str += f"MongoDB Query: {mongodb_query}\n\n"

        prompt = (
            "You are an expert MongoDB query generator. Given a question, output ONLY the MongoDB query.\n"
            "Do not include explanations, apologies, or extra text.\n\n"
            f"Context Examples:\n{context_str}\n"
            f"### Task: Generate MongoDB Query (MQL)\n"
            f"### Question:\n{query}\n\n"
            f"### MongoDB Query:\ndb."
        )
        return prompt

    def _build_generation_kwargs(self) -> dict:
        eos_id = self.tokenizer.eos_token_id if self.tokenizer else None
        if isinstance(eos_id, torch.Tensor):
            eos_id = eos_id.item()

        kwargs = {
            "max_new_tokens": 128,
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True,
            "pad_token_id": eos_id,
            "eos_token_id": eos_id,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
        }
        if self.config:
            kwargs["max_new_tokens"] = min(getattr(self.config, "max_length", 256), 128)
            kwargs["temperature"] = getattr(self.config, "temperature", 0.2)
            kwargs["top_p"] = getattr(self.config, "top_p", 0.95)
            kwargs["top_k"] = getattr(self.config, "top_k", 50)
        return kwargs

    @staticmethod
    def _clean_mongo_output(raw_output: str) -> str:
        stripped = raw_output.strip()
        if stripped.lower().startswith("db."):
            restored = stripped
        else:
            restored = "db." + stripped

        cleaned = restored.replace("```javascript", "").replace("```json", "").replace("```", "").strip()
        stop_markers = ["--->", "\n\n", "\nUser:", "\nQuestion:", "```", "###", "Human:"]
        for marker in stop_markers:
            idx = cleaned.find(marker)
            if idx != -1:
                cleaned = cleaned[:idx]
        lines = cleaned.split('\n')
        for line in lines:
            if line.strip().startswith("db."):
                return line.strip().rstrip(';')
        if "db." in cleaned:
            idx = cleaned.find("db.")
            return cleaned[idx:].split('\n')[0].strip().rstrip(';')
        return cleaned.strip()

    # Update generate_with_rag to clean the output before returning:
    def generate_with_rag(self, query: str, top_k: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        context_metadata = self.retrieve_context(query, top_k=top_k)
        prompt = self.build_prompt(query, context_metadata)

        if self.generator:
            generated_text = self.generator.generate(prompt)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            gen_kwargs = self._build_generation_kwargs()
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

        generated_text = self._clean_mongo_output(generated_text)
        return generated_text, context_metadata

    def generate_program(self, prompt: str, top_k: int = 3) -> str:
        """Wrapper for simple program generation using RAG."""
        generated_text, _ = self.generate_with_rag(query=prompt, top_k=top_k)
        return generated_text