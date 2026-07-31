"""
Step 4 / Task 1.2 — Documentation generation: generate a docstring for a given
program. Evaluated against CoDocBench (code, reference-docstring) pairs with
BLEU / BERTScore / CodeBLEU-lite.
"""
from __future__ import annotations
from typing import List, Tuple

PROMPT_TEMPLATE = (
    '{code}\n'
    '"""\n'
    "Write a concise docstring describing what the function above does, "
    "its parameters, and its return value.\n"
    '"""\n'
)


def build_prompt(code: str) -> str:
    return PROMPT_TEMPLATE.format(code=code)


def generate_doc(lm, code: str) -> str:
    prompt = build_prompt(code)
    return lm.generate(prompt, max_new_tokens=128, num_return_sequences=1)[0]


def generate_docs_batch(lm, code_samples: List[str]) -> List[str]:
    return [generate_doc(lm, c) for c in code_samples]


def pairs_from_codocbench(rows: List[dict], version: str = "latest") -> List[Tuple[str, str]]:
    """Adapt real CoDocBench rows into (code, docstring) pairs.

    Real schema (kunpai/codocbench): each row has a `version_data` list, where
    each entry carries its own `code`/`docstring` for that commit version — NOT
    a flat {"code":..., "docstring":...} per line. `version="latest"` takes the
    last (most recent) version per function; pass an int index to pick a specific
    version_data entry instead.
    """
    pairs = []
    for row in rows:
        versions = row.get("version_data", [])
        if not versions:
            continue
        entry = versions[-1] if version == "latest" else versions[version]
        code, doc = entry.get("code"), entry.get("docstring")
        if code and doc:
            pairs.append((code, doc))
    return pairs
