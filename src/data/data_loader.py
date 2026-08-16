"""Loads benchmark datasets (Spider, and the still-experimental BirdBench /
CoDocBench) from `data/raw/` subdirectories. Supports .json and .jsonl files.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads benchmark datasets from `data_dir` subfolders."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def load_single_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Loads a single .json or .jsonl file safely, returning [] on any failure."""
        path = self._resolve_path(file_path)

        if not path.exists():
            logger.error(f"File not found: {path}")
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix.lower() == ".jsonl":
                    return self._parse_jsonl(f, path)
                elif path.suffix.lower() == ".json":
                    return self._parse_json(f, path)
                logger.warning(f"Unsupported file format '{path.suffix}' for file: {path}")
                return []
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            return []

    def _resolve_path(self, file_path: Union[str, Path]) -> Path:
        """Ensures relative paths are resolved against `data_dir` without duplication."""
        path = Path(file_path)
        if not path.is_absolute() and not str(path).startswith(str(self.data_dir)):
            path = self.data_dir / path
        return path

    def _parse_jsonl(self, file_obj, path: Path) -> List[Dict[str, Any]]:
        """Parses a JSONL file line-by-line, skipping (and logging) malformed lines."""
        records = []
        for line_idx, line in enumerate(file_obj):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {line_idx} in {path}: {e}")

        logger.info(f"Loaded {len(records)} items from {path.name}")
        return records

    def _parse_json(self, file_obj, path: Path) -> List[Dict[str, Any]]:
        """Parses a standard JSON file, accepting either a list or a single object as root."""
        content = json.load(file_obj)

        if isinstance(content, list):
            records = content
        elif isinstance(content, dict):
            records = [content]
        else:
            logger.warning(f"Unexpected JSON root structure in {path}: {type(content)}")
            records = []

        logger.info(f"Loaded {len(records)} items from {path.name}")
        return records

    def load_folder_data(self, folder_name: str) -> List[Dict[str, Any]]:
        """Recursively finds and loads all .json/.jsonl files within a subfolder."""
        folder_path = self.data_dir / folder_name

        if not folder_path.exists() or not folder_path.is_dir():
            logger.error(f"Folder not found or is not a directory: {folder_path}")
            return []

        data_files = sorted(folder_path.rglob("*.json")) + sorted(folder_path.rglob("*.jsonl"))
        if not data_files:
            logger.warning(f"No .json or .jsonl files found in directory: {folder_path}")
            return []

        logger.info(f"Found {len(data_files)} data file(s) in {folder_path.name}/")

        combined_data = []
        for file_path in data_files:
            combined_data.extend(self.load_single_file(file_path))

        logger.info(f"Total items loaded for '{folder_name}': {len(combined_data)}")
        return combined_data

    def load_spider(self, target: str = "Spider") -> List[Dict[str, Any]]:
        """Loads Spider benchmark data from a folder or a specific file."""
        target_path = self.data_dir / target
        if target_path.is_dir():
            return self.load_folder_data(target)
        return self.load_single_file(target)

    # NOTE: BirdBench / CoDocBench loaders are not wired up yet. They follow the
    # exact same folder-or-file pattern as `load_spider` above; kept here,
    # commented out, as a placeholder for when those benchmarks are added.
    #
    # def load_birdbench(self, target: str = "BirdBench") -> List[Dict[str, Any]]:
    #     """Loads BirdBench benchmark data from folder or specific file."""
    #     target_path = self.data_dir / target
    #     if target_path.is_dir():
    #         return self.load_folder_data(target)
    #     return self.load_single_file(target)
    #
    # def load_codocbench(self, target: str = "CoDocBench") -> List[Dict[str, Any]]:
    #     """Loads CoDocBench benchmark data from folder or specific file."""
    #     target_path = self.data_dir / target
    #     if target_path.is_dir():
    #         return self.load_folder_data(target)
    #     return self.load_single_file(target)

    def load_all_benchmarks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Loads data from all currently-supported benchmarks into one dictionary."""
        return {
            "spider": self.load_spider(),
            # "birdbench": self.load_birdbench(),
            # "codocbench": self.load_codocbench(),
        }
