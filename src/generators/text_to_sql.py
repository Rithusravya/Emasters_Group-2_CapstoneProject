import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from src.generators.program_generator import GenerationPipeline


class TextToSQLGenerator:
    """Task module for Natural Language to SQL & MongoDB translation producing structured JSON output."""

    def __init__(self, pipeline: GenerationPipeline):
        self.pipeline = pipeline

    def generate_queries(
            self,
            question: str,
            schema: Optional[str] = None,
            dialect: str = "sqlite"
    ) -> Dict[str, Any]:
        """Generates both SQL and MongoDB queries for a natural language question."""
        prompt = (
            f"You are a database expert. Convert a question into SQL and MongoDB.\n\n"
            f"Dialect: {dialect}\n"
            f"Schema: {schema or 'N/A'}\n\n"
            f"Question: {question}\n\n"
            f"Example:\n"
            f"Question: List all products with price greater than 100.\n"
            f"Schema: Table products(id, name, price, category)\n"
            f"Output:\n"
            f'{{\n'
            f'  "sql_query": "SELECT * FROM products WHERE price > 100",\n'
            f'  "mongodb_query": {{\n'
            f'    "collection": "products",\n'
            f'    "operation": "find",\n'
            f'    "filter": {{\n'
            f'      "price": {{ "$gt": 100 }}\n'
            f'    }}\n'
            f'  }}\n'
            f'}}\n\n'
            f"Instructions:\n"
            f"- sql_query: Write a valid {dialect} SQL statement\n"
            f"- mongodb_query.collection: The table/collection name\n"
            f"- mongodb_query.operation: The MongoDB operation (find, countDocuments, aggregate, etc.)\n"
            f"- mongodb_query.filter: The query conditions as a MongoDB filter object\n\n"
            f"Now convert the question above. Output ONLY the JSON object. No explanations, no markdown, no extra text.\n"
            f"Output:"
        )
        raw_output = self.pipeline.generate_program(prompt)
        return self._parse_json_response(raw_output)

    def save_result(
            self,
            result: Dict[str, Any],
            output_dir: Union[str, Path] = "outputs/generated/text_to_sql",
            question: Optional[str] = None,
            schema: Optional[str] = None,
            dialect: Optional[str] = None,
            filename: Optional[str] = None
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
        """Parses the LLM output into a dictionary, with robust fallbacks for small models."""
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text, flags=re.MULTILINE)

        parsed = {}
        try:
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        sql_query = parsed.get("sql_query", "").strip() if isinstance(parsed, dict) else ""
        mongodb_query = parsed.get("mongodb_query", {}) if isinstance(parsed, dict) else {}

        # VALIDATION: If sql_query looks like an explanation, trigger fallback
        is_garbage = not sql_query or any(phrase in sql_query.lower() for phrase in [
            "to convert", "understand the", "here's", "explain", "step",
            "\"status\"", "we need to", "follow these steps", "natural language"
        ])

        if is_garbage:
            sql_match = re.search(r"(SELECT\s+.*?)(?:\n\n|```|$)", text, re.IGNORECASE | re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1).strip()
            else:
                sql_match = re.search(r"```sql\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
                if sql_match:
                    sql_query = sql_match.group(1).strip()
                else:
                    sql_query = "SELECT * FROM table; -- Model failed to generate valid SQL."

        # VALIDATION: Ensure mongodb_query is a properly structured dict
        if isinstance(mongodb_query, str):
            collection_match = re.search(r"db\.([a-zA-Z_]+)\.", mongodb_query)
            collection = collection_match.group(1) if collection_match else "collection"

            operation = "find"
            if "countDocuments" in mongodb_query:
                operation = "countDocuments"
            elif "aggregate" in mongodb_query:
                operation = "aggregate"

            mongodb_query = {
                "collection": collection,
                "operation": operation,
                "filter": {}
            }
        elif isinstance(mongodb_query, dict):
            if "collection" not in mongodb_query:
                mongodb_query["collection"] = "collection"
            if "operation" not in mongodb_query:
                mongodb_query["operation"] = "find"
            if "filter" not in mongodb_query:
                mongodb_query["filter"] = {}
        else:
            mongodb_query = {
                "collection": "collection",
                "operation": "find",
                "filter": {}
            }

        return {
            "sql_query": sql_query,
            "mongodb_query": mongodb_query
        }