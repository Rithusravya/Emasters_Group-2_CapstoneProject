import logging

logger = logging.getLogger(__name__)

class CommitMessageGenerator:
    """
    Generates conventional commit messages based on git diffs.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def _build_prompt(self, diff: str) -> str:
        prompt = (
            "You are an expert software developer. Generate a concise, conventional commit message "
            "for the following git diff. The message should start with a standard type "
            "(feat, fix, docs, style, refactor, test, chore) followed by a colon and a short, "
            "imperative summary of the changes.\n\n"
            f"Diff:\n{diff}\n\n"
            "Commit Message:"
        )
        return prompt

    def generate_commit_msg(self, diff: str) -> str:
        """Generates a commit message for the provided git diff."""
        prompt = self._build_prompt(diff)
        return self.pipeline.generate(prompt)