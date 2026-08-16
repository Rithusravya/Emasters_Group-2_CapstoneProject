<<<<<<< Updated upstream
"""Wraps a causal LM + tokenizer for prompt-based text/code generation."""

import logging

import torch
from transformers import GenerationConfig

logger = logging.getLogger(__name__)


class GenerationPipeline:
    """Wraps a causal LM + tokenizer for prompt-based generation."""
=======
"""
Step 4 / Task 1.1 — Program synthesis

Generate a complete program from a problem description.

Evaluation:
- Pass@k
- CodeBLEU-lite
- Execution accuracy
"""

from __future__ import annotations

import re
from typing import List


PROMPT_TEMPLATE = """
You are an expert competitive programmer.

Write ONLY the complete solution code.

Requirements:
- Use {language}.
- Do not explain the solution.
- Do not include markdown.
- Do not include ``` blocks.
- Include all required imports.
- The program must solve the problem correctly.

Problem:

{problem}

Solution code:
"""


def build_prompt(
    problem: str,
    language: str = "python",
) -> str:

    return PROMPT_TEMPLATE.format(
        problem=problem,
        language=language,
    )
>>>>>>> Stashed changes

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

<<<<<<< Updated upstream
        self.generation_config = GenerationConfig(
            max_length=getattr(config, "max_length", 512),
            temperature=getattr(config, "temperature", 0.2),
            top_k=getattr(config, "top_k", 50),
            top_p=getattr(config, "top_p", 0.95),
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            repetition_penalty=1.1,
        )
=======
def clean_code(output: str) -> str:
    """
    Remove markdown and explanations from generated code.
    """

    if not output:
        return ""

    output = output.strip()

    # remove markdown fences
    output = output.replace("```python", "")
    output = output.replace("```", "")

    # remove common explanation sections
    patterns = [
        r"^Here.*?:",
        r"^Solution.*?:",
        r"^Explanation.*?:",
    ]

    for p in patterns:
        output = re.sub(
            p,
            "",
            output,
            flags=re.IGNORECASE,
        )

    return output.strip()
>>>>>>> Stashed changes

    def generate_program(self, prompt: str, clean_output: bool = True) -> str:
        """Generates text from `prompt`, optionally stripping the echoed prompt
        from the start of the model's output. Returns "" on any generation error.
        """
        try:
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(device)

<<<<<<< Updated upstream
            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=self.generation_config)
=======
def synthesize(
    lm,
    problem: str,
    language: str = "python",
    n: int = 1,
) -> List[str]:

    prompt = build_prompt(
        problem,
        language,
    )

    outputs = lm.generate(
        prompt,
        max_new_tokens=256,
        num_return_sequences=n,
    )

    return [
        clean_code(x)
        for x in outputs
    ]


def synthesize_batch(
    lm,
    problems: List[str],
    language: str = "python",
) -> List[str]:

    results = []

    for problem in problems:

        code = synthesize(
            lm,
            problem,
            language,
            n=1,
        )[0]

        results.append(code)

    return results
>>>>>>> Stashed changes

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

<<<<<<< Updated upstream
            if clean_output and prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()

            return generated_text
        except Exception as e:
            logger.error(f"Error during code generation: {e}")
            return ""
=======
if __name__ == "__main__":

    print(
        build_prompt(
            "Return the maximum value in a list of integers."
        )
    )
>>>>>>> Stashed changes
