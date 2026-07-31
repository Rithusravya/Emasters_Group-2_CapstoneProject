"""
Step 7 / Task 1.3 -- Commit message generation from a file diff. Metrics: BLEU, ROUGE.

No external labeled commit-message benchmark is specified in the project brief, so
eval pairs are mined directly from local git history via
mine_diff_commit_pairs_from_repo() rather than cited against a named dataset.
"""
from __future__ import annotations
import json
import subprocess
from typing import List, Tuple

PROMPT_TEMPLATE = (
    "# The following is a unified diff of a code change.\n"
    "{diff}\n"
    "# Write a single-line, imperative-mood commit message summarizing this change:\n"
)


def build_prompt(diff: str) -> str:
    return PROMPT_TEMPLATE.format(diff=diff)


def generate_commit_message(lm, diff: str) -> str:
    prompt = build_prompt(diff)
    out = lm.generate(prompt, max_new_tokens=32, num_return_sequences=1)[0]
    return out.splitlines()[0].strip() if out else ""


def mine_diff_commit_pairs_from_repo(repo_path: str, limit: int = 500) -> List[Tuple[str, str]]:
    """Mine (diff, commit_message) pairs from local git history -- a lightweight way
    to build a training/eval set for this task without an external dataset.

    Robust to shallow/partial clones: `git log` can list commits whose full
    trees/blobs were never fetched, so `git show` on them fails with exit 128.
    We detect a shallow repo up front and try to unshallow it, and otherwise
    just skip any individual commit `git show` can't produce a diff for
    instead of raising, so one bad commit doesn't abort the whole pass.
    """
    is_shallow = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "--is-shallow-repository"],
        capture_output=True, text=True,
    ).stdout.strip()
    if is_shallow == "true":
        subprocess.run(
            ["git", "-C", repo_path, "fetch", "--unshallow"],
            capture_output=True, text=True,
        )

    log = subprocess.run(
        ["git", "-C", repo_path, "log", f"-{limit}", "--no-merges", "--pretty=format:%H|||%s"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()

    pairs = []
    skipped = 0
    for line in log:
        if "|||" not in line:
            continue
        commit_hash, message = line.split("|||", 1)
        result = subprocess.run(
            ["git", "-C", repo_path, "show", commit_hash, "--no-stat", "--patch",
             "--no-color", "--unified=0"],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            skipped += 1
            continue
        pairs.append((result.stdout[:4000], message.strip()))

    if skipped:
        print(f"Skipped {skipped} commit(s) git could not produce a diff for "
              f"(e.g. missing objects in a shallow/partial clone).")

    return pairs


def dump_pairs_jsonl(pairs: List[Tuple[str, str]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for diff, msg in pairs:
            f.write(json.dumps({"diff": diff, "message": msg}) + "\n")
