"""Hugging Face Hub sync for model representations.

Tensors are not stored in Git. They live in a Hub dataset repo and are pinned
per run by ``data/<run_id>/representations.lock.json``.
"""

import json
from pathlib import Path

LOCK_NAME = "representations.lock.json"
DEFAULT_REPO_ID = "llm-behaviour-intensity/activations"


def lock_path(data_root: str | Path) -> Path:
    """``data/<run_id>/representations.lock.json``."""
    return Path(data_root) / LOCK_NAME


def write_lock(
    data_root: str | Path,
    *,
    repo_id: str,
    revision: str,
    run_id: str,
    files: list[str],
) -> Path:
    """Record which Hub revision holds this run's tensors. Returns the lock path."""
    path = lock_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "run_id": run_id,
        "files": sorted(files),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def allow_patterns(
    run_id: str, *, model: str | None = None, trait: str | None = None
) -> list[str]:
    """Hub glob patterns selecting a subset of one run's representations.

    The unembedding covariance is model-level rather than per-trait, so it stays
    included whenever a trait filter narrows the layer files.
    """
    base = f"{run_id}/representations"
    m = model or "*"
    if trait is None:
        return [f"{base}/{m}/**"]
    return [f"{base}/{m}/{trait}/**", f"{base}/{m}/unembeddings_covariance.pt"]


def read_lock(data_root: str | Path) -> dict:
    """Load the lock file, or explain how to create it."""
    path = lock_path(data_root)
    if not path.exists():
        run_id = Path(data_root).name
        raise FileNotFoundError(
            f"No representation lock at {path}.\n"
            f"Upload this run's tensors first:\n"
            f"    python src/data_sync.py push --run {run_id}"
        )
    return json.loads(path.read_text())
