import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads benchmark datasets (Spider, BirdBench, CoDocBench) from data/raw/ subdirectories.

    Supports both .json and .jsonl files.
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def load_single_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Loads a single .json or .jsonl file safely."""
        path = Path(file_path)

        # Ensure path points to data_dir without duplicating path prefixes
        if not path.is_absolute() and not str(path).startswith(str(self.data_dir)):
            path = self.data_dir / path

        if not path.exists():
            logger.error(f"File not found: {path}")
            return []

        data = []
        file_suffix = path.suffix.lower()

        try:
            with open(path, "r", encoding="utf-8") as f:
                if file_suffix == ".jsonl":
                    # Parse line-by-line JSONL
                    for line_idx, line in enumerate(f):
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                logger.warning(f"Error parsing line {line_idx} in {path}: {e}")

                elif file_suffix == ".json":
                    # Parse standard single JSON file
                    content = json.load(f)
                    if isinstance(content, list):
                        data.extend(content)
                    elif isinstance(content, dict):
                        data.append(content)
                    else:
                        logger.warning(f"Unexpected JSON root structure in {path}: {type(content)}")
                else:
                    logger.warning(f"Unsupported file format '{file_suffix}' for file: {path}")
                    return []

        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            return []

        logger.info(f"Loaded {len(data)} items from {path.name}")
        return data

    def load_folder_data(self, folder_name: str) -> List[Dict[str, Any]]:
        """Recursively finds and loads all .json and .jsonl files within a specific subfolder."""
        folder_path = self.data_dir / folder_name

        if not folder_path.exists() or not folder_path.is_dir():
            logger.error(f"Folder not found or is not a directory: {folder_path}")
            return []

        # Find all .json and .jsonl files inside the target benchmark directory
        json_files = sorted(
            list(folder_path.rglob("*.json")) + list(folder_path.rglob("*.jsonl"))
        )

        if not json_files:
            logger.warning(f"No .json or .jsonl files found in directory: {folder_path}")
            return []

        logger.info(f"Found {len(json_files)} data file(s) in {folder_path.name}/")

        combined_data = []
        for file_path in json_files:
            file_data = self.load_single_file(file_path)
            combined_data.extend(file_data)

        logger.info(f"Total items loaded for '{folder_name}': {len(combined_data)}")
        return combined_data

    def load_spider(self, target: str = "Spider") -> List[Dict[str, Any]]:
        """Loads Spider benchmark data from folder or specific file."""
        target_path = self.data_dir / target
        if target_path.is_dir():
            return self.load_folder_data(target)
        return self.load_single_file(target)

    def load_birdbench(self, target: str = "BirdBench") -> List[Dict[str, Any]]:
        """Loads BirdBench benchmark data from folder or specific file."""
        target_path = self.data_dir / target
        if target_path.is_dir():
            return self.load_folder_data(target)
        return self.load_single_file(target)

    def load_codocbench(self, target: str = "CoDocBench") -> List[Dict[str, Any]]:
        """Loads CoDocBench benchmark data from folder or specific file."""
        target_path = self.data_dir / target
        if target_path.is_dir():
            return self.load_folder_data(target)
        return self.load_single_file(target)

    def load_all_benchmarks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Loads data from all three benchmarks into a single dictionary."""
        return {
            "spider": self.load_spider(),
            "birdbench": self.load_birdbench(),
            "codocbench": self.load_codocbench(),
        }
