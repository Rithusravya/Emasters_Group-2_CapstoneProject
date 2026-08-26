import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TextToMongoGenerator:
    def __init__(self, pipeline, config: Any = None):
        self.pipeline = pipeline
        self.config = config

    def _build_prompt(self, question: str, schema: str = "") -> str:
        prompt = "### Task: Generate MongoDB Query (MQL)\n"
        if schema:
            prompt += f"### Collection Schema / Structure:\n{schema}\n\n"
        prompt += f"### Question:\n{question}\n\n### MongoDB Query:\n db."
        return prompt

    def generate_single(self, question: str, schema: str = "") -> Dict[str, Any]:
        """Generates a single MongoDB query and measures latency."""
        prompt = self._build_prompt(question, schema)
        start_time = time.perf_counter()
        raw_output = self.pipeline.generate(prompt)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Clean up hallucinations and repetition loops
        generated_query = self._clean_mongo_output(raw_output)

        return {
            "generated_query": generated_query,
            "latency_ms": round(latency_ms, 2)
        }

    @staticmethod
    def clean_mongo_output(raw_output: str) -> str:
        stripped = raw_output.strip()
        if stripped.lower().startswith("db."):
            # Model re-emitted the prefix itself; don't duplicate it.
            restored = stripped
        else:
            restored = "db." + stripped

        cleaned = (
            restored.replace("```javascript", "")
            .replace("```json", "")
            .replace("```js", "")
            .replace("```", "")
            .strip()
        )

        # Stop at common hallucination markers
        stop_markers = ["--->", "\n\n", "\nUser:", "\nQuestion:", "```", "###", "Human:"]
        for marker in stop_markers:
            idx = cleaned.find(marker)
            if idx != -1:
                cleaned = cleaned[:idx]

        # Extract the first valid MongoDB query line
        lines = cleaned.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("db."):
                return line.rstrip(';').strip()

        # Fallback: find "db." anywhere in the text
        if "db." in cleaned:
            idx = cleaned.find("db.")
            end_idx = len(cleaned)
            for marker in stop_markers + ["\n"]:
                pos = cleaned.find(marker, idx)
                if pos != -1:
                    end_idx = min(end_idx, pos)
            return cleaned[idx:end_idx].strip().rstrip(';')

        return cleaned.strip()

    _clean_mongo_output = clean_mongo_output

    def generate_batch(self, dataset: List[Dict[str, Any]], output_path: str = None) -> List[Dict[str, Any]]:
        results = []
        logger.info(f"🚀 Starting batch MongoDB query generation for {len(dataset)} examples...")

        for item in tqdm(dataset, desc="Generating MongoDB Queries", unit="query"):
            query_id = item.get("query_id", f"sample_{len(results)}")
            database_id = item.get("database_id", "unknown")
            question = item.get("question", "")
            schema = item.get("schema", "")
            gold_query = item.get("generated_mongodb_query", item.get("gold_query", ""))
            if isinstance(gold_query, list) and len(gold_query) > 0:
                gold_query = gold_query[0]

            if not question:
                continue

            gen_result = self.generate_single(question, schema)

            result_entry = {
                "query_id": query_id,
                "database_id": database_id,
                "question": question,
                "gold_query": gold_query,
                "generated_query": gen_result["generated_query"],
                "latency_ms": gen_result["latency_ms"],
            }
            results.append(result_entry)

        if output_path:
            self.save_results(results, output_path)

        return results

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {"results": results}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        logger.info(f"✅ Successfully saved {len(results)} generated queries to {path}")
