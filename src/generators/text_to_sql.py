import re
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from src.generators.program_generator import GenerationPipeline

# Strict SQL-only prompt
_PROMPT_TEMPLATE = """You are an expert SQLite developer.
Given the database schema and the user's question, write ONE valid SQLite SQL query.

Rules:
- Return ONLY the SQL query.
- Do NOT output JSON, MongoDB queries, or any explanations.
- Do NOT use markdown formatting (no ```sql blocks).
- Ensure the SQL is valid and directly answers the question.

Database schema:
{schema}

Question:
{question}

SQL:
"""


class TextToSQLGenerator:
    def __init__(self, pipeline: GenerationPipeline):
        self.pipeline = pipeline

    def _build_prompt(self, question: str, schema: Optional[str], dialect: str) -> str:
        return _PROMPT_TEMPLATE.format(
            schema=schema or "No schema provided",
            question=question
        )

    def generate_queries(
            self,
            question: str,
            schema: Optional[str] = None,
            dialect: str = "sqlite",
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(question, schema, dialect)
        raw_output = self.pipeline.generate_program(prompt)
        sql_query = self._extract_sql(raw_output)

        valid = bool(re.match(r"^\s*(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b", sql_query, re.IGNORECASE))

        return {
            "sql_query": sql_query,
            "valid": valid,
            "parse_error": None if valid else "invalid_sql_query"
        }

    @staticmethod
    def _extract_sql(text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        # 1. Remove markdown code fences first
        text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"\s*```\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"\s*```\s*", " ", text)

        # 2. Try JSON parse (fallback)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "sql_query" in parsed:
                return str(parsed["sql_query"]).strip().rstrip(';').strip()
        except (json.JSONDecodeError, TypeError):
            pass

        # 3. Extract only the first SQL statement, stop at explanation markers
        lines = text.split('\n')
        sql_lines = []
        for line in lines:
            stripped = line.strip()
            # Stop at JSON keys, markdown, or explanation markers
            if re.match(r'^\s*["\']?(sql_query|mongodb_query|valid|parse_error)["\']?\s*:', stripped, re.IGNORECASE):
                break
            if stripped.startswith('```') or stripped.startswith('---'):
                break
            if re.match(r'^(Here|Note|Explanation|Solution|This|The|In|To|We|Use|Step)\b', stripped, re.IGNORECASE):
                break
            # Skip lines that are clearly explanation (contain backticks with text)
            if '`' in stripped and not re.search(r'SELECT|FROM|WHERE|JOIN|ORDER|GROUP|HAVING|LIMIT|UNION', stripped,
                                                 re.IGNORECASE):
                continue
            if stripped:
                sql_lines.append(stripped)

        if not sql_lines:
            return ""

        sql = " ".join(sql_lines)

        # 4. Clean up: remove trailing garbage, fix quotes
        sql = re.sub(r'[\s,"\'}\]]+$', '', sql)
        # Fix unclosed quotes
        single_quotes = sql.count("'")
        if single_quotes % 2 != 0:
            sql += "'"
        double_quotes = sql.count('"')
        if double_quotes % 2 != 0:
            sql += '"'

        # 5. Extract only up to the first semicolon (single statement)
        if ';' in sql:
            sql = sql.split(';')[0]

        # 6. Remove any remaining backticks
        sql = sql.replace('`', '')

        return sql.strip()

    def save_result(
            self,
            result: Dict[str, Any],
            output_dir: str = "outputs/generated/text_to_sql",
            question: Optional[str] = None,
            schema: Optional[str] = None,
            dialect: Optional[str] = None,
            filename: Optional[str] = None,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            # Use timezone-aware datetime to avoid Python 3.12+ deprecation warnings
            filename = f"text_to_sql_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

        payload = {
            "question": question,
            "schema": schema,
            "dialect": dialect,
            "sql_query": result.get("sql_query", ""),
            "valid": bool(result.get("valid", False)),
            "parse_error": result.get("parse_error"),
        }

        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return output_path