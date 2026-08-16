"""
Step 4 / Task 1.2 — Documentation generation

Generate a Python docstring for a given function.

Evaluated against CoDocBench using BLEU / BERTScore.
"""

from __future__ import annotations

import re
from typing import List, Tuple

PROMPT_TEMPLATE = """
You are an expert Python developer.

Write ONLY a concise Python docstring for the function below.

Requirements:
- Describe what the function does.
- Describe the parameters (if any).
- Describe the return value (if any).
- Do not repeat the code.
- Do not explain anything outside the docstring.

Function:

{code}

Docstring:
\"\"\"
"""


def build_prompt(code: str) -> str:
    return PROMPT_TEMPLATE.format(code=code)


def clean_docstring(text: str) -> str:
    """
    Clean model output and return only the generated docstring.
    """

    if not text:
        return ""

    text = text.strip()

    # remove markdown fences
    text = text.replace("```python", "")
    text = text.replace("```", "")

    # remove prompt echoes
    if "Docstring:" in text:
        text = text.split("Docstring:")[-1]

    text = text.strip()

    # remove opening triple quotes
    if text.startswith('"""'):
        text = text[3:]

    # keep only until closing quotes
    if '"""' in text:
        text = text.split('"""')[0]

    text = text.strip()

    # remove leading/trailing blank lines
    lines = [l.rstrip() for l in text.splitlines()]
    text = "\n".join(lines).strip()

    return text


def generate_doc(lm, code: str) -> str:
    prompt = build_prompt(code)

    output = lm.generate(
        prompt,
        max_new_tokens=128,
        num_return_sequences=1,
    )[0]

    return clean_docstring(output)


def generate_docs_batch(lm, code_samples: List[str]) -> List[str]:
    return [generate_doc(lm, code) for code in code_samples]


def pairs_from_codocbench(
    rows: List[dict],
    version: str = "latest",
) -> List[Tuple[str, str]]:
    """
    Convert CoDocBench rows into (code, docstring) pairs.

    Each dataset row contains

        version_data = [
            {
                "code": "...",
                "docstring": "...",
            },
            ...
        ]

    version="latest" selects the newest version.
    """

    pairs = []

    for row in rows:

        versions = row.get("version_data", [])

        if not versions:
            continue

        if version == "latest":
            entry = versions[-1]
        else:
            entry = versions[int(version)]

        code = entry.get("code", "")
        doc = entry.get("docstring", "")

        if code.strip() and doc.strip():
            pairs.append((code, doc.strip()))

    return pairs


if __name__ == "__main__":

    sample = """
def add(a, b):
    return a + b
"""

    print(build_prompt(sample))