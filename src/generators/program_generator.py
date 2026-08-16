from __future__ import annotations

import logging
import re
from typing import List
import torch
from transformers import GenerationConfig

logger = logging.getLogger(__name__)


class GenerationPipeline:
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

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

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

    @classmethod
    def build_prompt(
            cls,
            problem: str,
            language: str = "python",
    ) -> str:
        return cls.PROMPT_TEMPLATE.format(
            problem=problem,
            language=language,
        )

    @staticmethod
    def clean_code(output: str) -> str:
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

    def generate_program(self, prompt: str, clean_output: bool = True) -> str:
        try:
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, generation_config=self.generation_config
                )

            generated_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            if clean_output and prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()

            return generated_text

        except Exception as e:
            logger.error(f"Error during code generation: {e}")
            return ""

    def synthesize(
            self,
            problem: str,
            language: str = "python",
            n: int = 1,
    ) -> List[str]:
        prompt = self.build_prompt(
            problem,
            language,
        )

        # Assuming lm/self usage aligns with generate_program
        outputs = [self.generate_program(prompt) for _ in range(n)]

        return [self.clean_code(x) for x in outputs]

    def synthesize_batch(
            self,
            problems: List[str],
            language: str = "python",
    ) -> List[str]:
        results = []

        for problem in problems:
            code = self.synthesize(
                problem,
                language,
                n=1,
            )[0]

            results.append(code)

        return results


if __name__ == "__main__":
    print(
        GenerationPipeline.build_prompt(
            "Return the maximum value in a list of integers."
        )
    )