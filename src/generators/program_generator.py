"""
Step 4 / Task 1.1 — Program synthesis: generate a program for a given problem
description. Evaluated with Pass@k, CodeBLEU-lite, execution accuracy
(see src/evaluation/metrics.py).
"""
from __future__ import annotations
from typing import List

PROMPT_TEMPLATE = (
    "# Problem: {problem}\n"
    "# Language: {language}\n"
    "# Write a correct, complete solution below.\n"
)


def build_prompt(problem: str, language: str = "python") -> str:
    return PROMPT_TEMPLATE.format(problem=problem, language=language)


def synthesize(lm, problem: str, language: str = "python", n: int = 1) -> List[str]:
    """Generate `n` candidate programs for a problem description using the given
    CodeGenModel instance (base or RAG-augmented — see src/rag/context_injector.py)."""
    prompt = build_prompt(problem, language)
    return lm.generate(prompt, max_new_tokens=256, num_return_sequences=n)


def synthesize_batch(lm, problems: List[str], language: str = "python") -> List[str]:
    return [synthesize(lm, p, language, n=1)[0] for p in problems]


if __name__ == "__main__":
    print(build_prompt("Return the maximum value in a list of integers.", "python"))
