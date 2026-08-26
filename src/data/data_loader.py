import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetLoader:
    def __init__(self, target_filename="qwen_spider_mongodb_conversion.json"):
        self.target_filename = target_filename
        self.data = None
        self.file_path = None

    def scan_and_find_file(self, search_dirs=None):
        """Scans directories recursively to find the target JSON file."""
        if search_dirs is None:
            search_dirs = [
                Path.cwd(),
                Path.cwd().parent,
                Path.cwd() / "data",
                Path.cwd() / "data" / "raw",
                Path.cwd() / "data" / "raw" / "Spider",
                Path.cwd() / "outputs",
            ]

        search_dirs = list({d.resolve() for d in search_dirs if d.exists()})

        for directory in search_dirs:
            for path in directory.rglob(self.target_filename):
                if path.is_file():
                    logger.info(f"✅ Found {self.target_filename} at: {path}")
                    return path

        logger.warning(f"File not found in predefined paths. Scanning entire {Path.cwd()}...")
        for path in Path.cwd().rglob(self.target_filename):
            if path.is_file():
                logger.info(f"✅ Found {self.target_filename} at: {path}")
                return path

        raise FileNotFoundError(f"❌ Could not find {self.target_filename} in any scanned directories.")

    def load_data(self, file_path=None):
        """Loads the JSON data and parses it into a list of dictionaries."""
        if file_path is None:
            file_path = self.scan_and_find_file()
        else:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"❌ Provided path does not exist: {file_path}")

        self.file_path = file_path
        logger.info(f"📂 Loading data from {file_path}...")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if "detailed_results" in raw_data:
            self.data = raw_data["detailed_results"]
        elif isinstance(raw_data, list):
            self.data = raw_data
        else:
            self.data = [raw_data]

        # Normalize: flatten generated_mongodb_query if it's a list
        for item in self.data:
            if "generated_mongodb_query" in item and isinstance(item["generated_mongodb_query"], list) and len(item["generated_mongodb_query"]) > 0:
                item["generated_mongodb_query"] = item["generated_mongodb_query"][0]

        logger.info(f"✅ Successfully loaded {len(self.data)} records.")
        return self.data

    def get_training_data(self):
        """Returns data formatted for LoRA training (Question -> MongoDB Query)."""
        if self.data is None:
            self.load_data()

        training_data = []
        for item in self.data:
            question = item.get("question", "")
            mongodb_query = item.get("generated_mongodb_query", "")

            # Flatten if still a list (safety net)
            if isinstance(mongodb_query, list) and len(mongodb_query) > 0:
                mongodb_query = mongodb_query[0]

            if question and mongodb_query:
                training_data.append({
                    "question": question,
                    "generated_mongodb_query": mongodb_query,
                    "database_id": item.get("database_id", "")
                })

        logger.info(f"✅ Prepared {len(training_data)} training examples.")
        return training_data