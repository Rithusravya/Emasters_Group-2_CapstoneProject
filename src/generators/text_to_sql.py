import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.generators.program_generator import GenerationPipeline

_EXPLANATION_PHRASES = [
    "to convert", "understand the", "here's", "explain", "step",
    "\"status\"", "we need to", "follow these steps", "natural language",
]

_PROMPT_TEMPLATE = (
    "You are a database expert. Convert a question into SQL and MongoDB.\n\n"
    "Dialect: {dialect}\n"
    "Schema: {schema}\n\n"
    "Question: {question}\n\n"
    "Example:\n"
    "Question: List all products with price greater than 100.\n"
    "Schema: Table products(id, name, price, category)\n"
    "Output:\n"
    "{{\n"
    '  "sql_query": "SELECT * FROM products WHERE price > 100",\n'
    '  "mongodb_query": {{\n'
    '    "collection": "products",\n'
    '    "operation": "find",\n'
    '    "filter": {{\n'
    '      "price": {{ "$gt": 100 }}\n'
    "    }}\n"
    "  }}\n"
    "}}\n\n"
    "Instructions:\n"
    "- sql_query: Write a valid {dialect} SQL statement\n"
    "- mongodb_query.collection: The table/collection name\n"
    "- mongodb_query.operation: The MongoDB operation (find, countDocuments, aggregate, etc.)\n"
    "- mongodb_query.filter: The query conditions as a MongoDB filter object\n\n"
    "Now convert the question above. Output ONLY the JSON object. "
    "No explanations, no markdown, no extra text.\n"
    "Output:"
)

_DEFAULT_MONGO_QUERY = {"collection": "collection", "operation": "find", "filter": {}}
_FALLBACK_SQL = "SELECT * FROM table; -- Model failed to generate valid SQL."


class TextToSQLGenerator:
    def __init__(self, pipeline: GenerationPipeline):
        self.pipeline = pipeline

    def _build_prompt(self, question: str, schema: Optional[str], dialect: str) -> str:
        return _PROMPT_TEMPLATE.format(dialect=dialect, schema=schema or "N/A", question=question)

    def generate_queries(
        self,
        question: str,
        schema: Optional[str] = None,
        dialect: str = "sqlite",
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(question, schema, dialect)
        raw_output = self._call_pipeline(prompt)
        raw_text = self._normalize_pipeline_output(raw_output)
        return self._parse_json_response(raw_text)

    @staticmethod
    def _normalize_pipeline_output(raw_output: Any) -> str:
        if isinstance(raw_output, list):
            raw_output = raw_output[0] if raw_output else ""
        if isinstance(raw_output, dict):
            raw_output = raw_output.get("generated_text") or raw_output.get("text") or ""
        return str(raw_output)

    def save_result(
        self,
        result: Dict[str, Any],
        output_dir: Union[str, Path] = "outputs/generated/text_to_sql",
        question: Optional[str] = None,
        schema: Optional[str] = None,
        dialect: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"text_to_sql_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        payload = {
            "question": question,
            "schema": schema,
            "dialect": dialect,
            "sql_query": result.get("sql_query", ""),
            "mongodb_query": result.get("mongodb_query", {}),
        }

        output_path = output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return output_path

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        parsed = self._extract_json_object(text)

        sql_query = parsed.get("sql_query", "").strip() if isinstance(parsed, dict) else ""
        mongodb_query = parsed.get("mongodb_query", {}) if isinstance(parsed, dict) else {}

        if self._looks_like_explanation(sql_query):
            sql_query = self._extract_sql_fallback(text)

        return {
            "sql_query": sql_query,
            "mongodb_query": self._normalize_mongodb_query(mongodb_query),
        }

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text, flags=re.MULTILINE)

        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {}

    @staticmethod
    def _looks_like_explanation(sql_query: str) -> bool:
        if not sql_query:
            return True
        lowered = sql_query.lower()
        return any(phrase in lowered for phrase in _EXPLANATION_PHRASES)

    @staticmethod
    def _extract_sql_fallback(text: str) -> str:
        sql_match = re.search(r"(SELECT\s+.*?)(?:\n\n|```|$)", text, re.IGNORECASE | re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()

        sql_match = re.search(r"```sql\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()

        return _FALLBACK_SQL

    @staticmethod
    def _normalize_mongodb_query(mongodb_query: Any) -> Dict[str, Any]:
        if isinstance(mongodb_query, str):
            collection_match = re.search(r"db\.([a-zA-Z_]+)\.", mongodb_query)
            collection = collection_match.group(1) if collection_match else "collection"

            operation = "find"
            if "countDocuments" in mongodb_query:
                operation = "countDocuments"
            elif "aggregate" in mongodb_query:
                operation = "aggregate"

            return {"collection": collection, "operation": operation, "filter": {}}

        if isinstance(mongodb_query, dict):
            mongodb_query.setdefault("collection", "collection")
            mongodb_query.setdefault("operation", "find")
            mongodb_query.setdefault("filter", {})
            return mongodb_query

        return dict(_DEFAULT_MONGO_QUERY)

    def _call_pipeline(self, prompt: str) -> str:
        pipeline = self.pipeline

        # 1) Try common generation method names.
        for method_name in ("generate", "generate_text", "generate_code", "run", "infer", "predict", "generate_response"):
            fn = getattr(pipeline, method_name, None)
            if callable(fn):
                return fn(prompt)

        # 2) Try __call__ (i.e., pipeline(prompt)).
        if callable(pipeline):
            return pipeline(prompt)

        # 3) Fallback: generate directly with the pipeline's model + tokenizer.
        return self._generate_from_model_and_tokenizer(pipeline, prompt)

    @staticmethod
    def _generate_from_model_and_tokenizer(pipeline: Any, prompt: str) -> str:
        model = getattr(pipeline, "model", None)
        tokenizer = getattr(pipeline, "tokenizer", None)

        if model is None or tokenizer is None:
            available = [a for a in dir(pipeline) if not a.startswith("_")]
            raise AttributeError(
                f"GenerationPipeline has no generate-like method and no "
                f"model/tokenizer attributes. Available: {available}"
            )

        gen_config = getattr(pipeline, "config", None) or getattr(pipeline, "generation_config", None)
        max_new_tokens = getattr(gen_config, "max_new_tokens", 256) if gen_config else 256
        temperature = getattr(gen_config, "temperature", 0.2) if gen_config else 0.2

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
