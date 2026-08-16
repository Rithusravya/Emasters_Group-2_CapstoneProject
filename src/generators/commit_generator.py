"""
Step 7 / Task 1.3 — Commit message generation

Generate a commit message from a git diff.

Evaluation:
- BLEU
- ROUGE
"""

from __future__ import annotations

import json
import subprocess

from typing import List, Tuple


PROMPT_TEMPLATE = """
You are an experienced software engineer.

Write ONE concise git commit message based on the diff.

Rules:
- Use imperative mood.
- Keep it under 72 characters.
- Write only the commit message.
- No explanation.
- No markdown.

Git diff:

{diff}

Commit message:
"""


def build_prompt(diff: str) -> str:

    return PROMPT_TEMPLATE.format(
        diff=diff
    )


def clean_commit_message(text: str) -> str:

    if not text:
        return ""

    text = text.strip()

    text = text.replace(
        "```",
        ""
    )

    if "Commit message:" in text:

        text = text.split(
            "Commit message:"
        )[-1]

    # only first line
    text = text.splitlines()[0]

    return text.strip()


def generate_commit_message(
    lm,
    diff: str,
) -> str:

    prompt = build_prompt(diff)

    output = lm.generate(
        prompt,
        max_new_tokens=32,
        num_return_sequences=1,
    )[0]

    return clean_commit_message(output)


def mine_diff_commit_pairs_from_repo(
    repo_path: str,
    limit: int = 500,
) -> List[Tuple[str, str]]:

    log = subprocess.run(
        [
            "git",
            "-C",
            repo_path,
            "log",
            f"-{limit}",
            "--no-merges",
            "--pretty=format:%H|||%s",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()


    pairs = []


    for line in log:

        if "|||" not in line:
            continue


        commit_hash, message = line.split(
            "|||",
            1,
        )


        result = subprocess.run(
            [
                "git",
                "-C",
                repo_path,
                "show",
                commit_hash,
                "--no-stat",
                "--patch",
                "--no-color",
                "--unified=0",
            ],
            capture_output=True,
            text=True,
        )


        if result.returncode != 0:
            continue


        diff = result.stdout.strip()


        if diff:

            pairs.append(
                (
                    diff[:4000],
                    message.strip(),
                )
            )


    return pairs



def dump_pairs_jsonl(
    pairs: List[Tuple[str, str]],
    out_path: str,
) -> None:


    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:

        for diff, msg in pairs:

            f.write(
                json.dumps(
                    {
                        "diff": diff,
                        "message": msg,
                    }
                )
                + "\n"
            )