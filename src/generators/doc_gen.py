from src.generators.program_generator import GenerationPipeline


class DocGenerator:
    def __init__(self, pipeline: GenerationPipeline):
        self.pipeline = pipeline

    def generate_docstring(self, code: str, style: str = "google") -> str:
        prompt = f"Generate a {style} style docstring for the following Python code:\n\n{code}\n\nDocstring:"
        return self.pipeline.generate_program(prompt)
