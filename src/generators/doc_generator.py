import logging

logger = logging.getLogger(__name__)

class DocGenerator:
    """
    Generates documentation/docstrings for code snippets using 5-shot prompting.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def _build_prompt(self, code_snippet: str) -> str:
        # 5 high-quality examples for few-shot learning
        examples = [
            {
                "code": "def add(a, b):\n    return a + b",
                "doc": "Adds two numbers together and returns their sum."
            },
            {
                "code": "import os\ndef list_files(directory):\n    return os.listdir(directory)",
                "doc": "Returns a list containing the names of all files and directories in the specified directory path."
            },
            {
                "code": "class Dog:\n    def __init__(self, name):\n        self.name = name\n    def bark(self):\n        return 'Woof!'",
                "doc": "A simple Dog class that initializes with a name and provides a method to make the dog bark."
            },
            {
                "code": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
                "doc": "Calculates the factorial of a given non-negative integer n using a recursive approach."
            },
            {
                "code": "import json\ndef parse_json(file_path):\n    with open(file_path, 'r') as f:\n        return json.load(f)",
                "doc": "Reads a JSON file from the specified file path and returns the parsed content as a Python dictionary."
            }
        ]
        
        prompt = "You are an expert programmer. Generate a concise, accurate, and professional docstring/description for the given code.\n\n"
        for ex in examples:
            prompt += f"Code:\n{ex['code']}\nDescription: {ex['doc']}\n\n"
            
        prompt += f"Code:\n{code_snippet}\nDescription:"
        return prompt

    def generate_docstring(self, code_snippet: str) -> str:
        """Generates a docstring for the provided code snippet."""
        prompt = self._build_prompt(code_snippet)
        return self.pipeline.generate(prompt)