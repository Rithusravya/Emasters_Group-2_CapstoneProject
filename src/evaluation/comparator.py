from typing import Dict, Any
from typing import Any, Dict, Optional

_DEMO_RESULTS: Dict[str, Dict[str, float]] = {
    "CodeGen-350M Base":        {"CodeBLEU": 0.41, "CodeBERTScore": 0.50, "ExecAccuracy": 0.381},
    "Zero-Shot LLM":            {"CodeBLEU": 0.49, "CodeBERTScore": 0.68, "ExecAccuracy": 0.546},
    "Fine-Tuned + Hybrid RAG":  {"CodeBLEU": 0.58, "CodeBERTScore": 0.82, "ExecAccuracy": 0.684},
}

class ArchitectureComparator:

    @staticmethod
    def compare_architectures(
        results: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        return results if results else dict(_DEMO_RESULTS)