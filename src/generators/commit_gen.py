from src.generators.program_generator import GenerationPipeline


class CommitMessageGenerator:
    def __init__(self, pipeline: GenerationPipeline):
        self.pipeline = pipeline

    def generate_commit_msg(self, diff: str) -> str:
        prompt = f"Write a clear, concise conventional commit message for this diff:\n\n{diff}\n\nCommit Message:"
        return self.pipeline.generate_program(prompt)
