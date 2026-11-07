"""Verify a bundle before consuming its data; never initialize the app database.

Intuition: a file that parses can still be stale or edited. Artifact checksums
bind downstream preparation to one exact bundle; baseline checksums separately
confirm that earlier committed inputs remain unchanged on this local workspace.
"""

import json
from pathlib import Path

from .encoder import file_sha256


def verify_manifest(directory: Path) -> dict:
    """Reject altered bundle files before exercising any serving path."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, digest in manifest["artifact_sha256"].items():
        path = directory / filename
        if path.resolve().parent != directory.resolve():
            raise ValueError("Manifest artifact must be directly inside the bundle")
        if file_sha256(path) != digest:
            raise ValueError(f"Bundle checksum mismatch: {filename}")
    for filename, digest in manifest["baseline_sha256"].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"Preserved baseline changed: {filename}")
    return {
        "manifest_sha256": file_sha256(manifest_path),
        "artifact_count": len(manifest["artifact_sha256"]),
        "baseline_unchanged": True,
    }
