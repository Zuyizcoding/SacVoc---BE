"""Local source identity shared by analysis and retrieval experiments.

A Git commit is insufficient when the working tree has edits. Hash the files that
actually execute as well. This records identity; it never invokes a remote Git
operation and never reads environment secrets.
"""

import subprocess
from pathlib import Path

from app.services.analysis.tracing import content_hash

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_fingerprint() -> str:
    paths = [
        *PROJECT_ROOT.joinpath("app").rglob("*.py"),
        *PROJECT_ROOT.joinpath("scripts").rglob("*.py"),
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "requirements-test.txt",
        PROJECT_ROOT / "requirements-learning.txt",
        PROJECT_ROOT / "requirements-learning-torch.txt",
    ]
    identities = "\n".join(
        f"{path.relative_to(PROJECT_ROOT).as_posix()}:{content_hash(path.read_bytes())}"
        for path in sorted(paths)
        if path.is_file()
    )
    return content_hash(identities)
