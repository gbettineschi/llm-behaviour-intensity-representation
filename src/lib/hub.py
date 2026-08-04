"""Hugging Face Hub sync for model representations.

Tensors are not stored in Git. They live in a Hub dataset repo and are pinned
per run by ``data/<run_id>/representations.lock.json``.

The lock records a sha256 per file, not just the Hub revision id. A revision id
alone is a *reference*: it says where the tensors came from, but nothing can
check that the bytes on your disk are still those tensors. The digests make it
an *integrity binding* — ``verify_lock`` turns "these results came from revision
X" from a claim into something checkable.
"""

import hashlib
import json
from pathlib import Path

LOCK_NAME = "representations.lock.json"
DEFAULT_REPO_ID = "llm-behaviour-intensity/activations"


def lock_path(data_root: str | Path) -> Path:
    """``data/<run_id>/representations.lock.json``."""
    return Path(data_root) / LOCK_NAME


def file_digest(path: str | Path) -> str:
    """sha256 of a file, streamed so a multi-GB sweep stays cheap in memory.

    Matches the digest the Hub reports for an LFS object, so a local file and
    its Hub counterpart are comparable without downloading anything.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_lock(
    data_root: str | Path,
    *,
    repo_id: str,
    revision: str,
    run_id: str,
    files: dict[str, str],
) -> Path:
    """Record which Hub revision holds this run's tensors, and their digests.

    ``files`` maps a path relative to ``data/<run_id>/`` to its sha256.
    Returns the lock path.
    """
    path = lock_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "run_id": run_id,
        "files": {k: files[k] for k in sorted(files)},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def verify_lock(
    data_root: str | Path, *, model: str | None = None, trait: str | None = None
) -> dict:
    """Compare the tensors on disk against the digests the lock pins.

    Only files present on disk are hashed, so a selective pull is not reported
    as corruption: ``missing`` is a normal state, ``mismatched`` is not.
    Returns ``{"matched": [...], "mismatched": [...], "missing": [...]}``.
    """
    data_root = Path(data_root)
    lock = read_lock(data_root)
    out: dict[str, list[str]] = {"matched": [], "mismatched": [], "missing": []}
    for rel, want in lock["files"].items():
        parts = Path(rel).parts  # representations/<model>/<trait>/<pooling>_token/layer_N.pt
        if model and (len(parts) < 2 or parts[1] != model):
            continue
        if trait and (len(parts) < 3 or parts[2] != trait):
            continue
        path = data_root / rel
        if not path.exists():
            out["missing"].append(rel)
        elif file_digest(path) == want:
            out["matched"].append(rel)
        else:
            out["mismatched"].append(rel)
    return out


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
