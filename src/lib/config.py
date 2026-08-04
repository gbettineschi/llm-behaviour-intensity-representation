"""Model registry, path conventions, and seed derivation shared by the
analysis drivers.

Focal layer rule: ``num_hidden_layers // 2`` (mid depth), declared explicitly
per model so it is visible and auditable. ``params_b`` is the nominal
(marketing) parameter count in billions — the scale axis for cross-model
summaries, not used in any analysis.
"""

from __future__ import annotations

import subprocess
import zlib
from datetime import datetime
from importlib import metadata as _im
from pathlib import Path

import numpy as np

MODELS: dict[str, dict] = {
    "gemma-2-2b": {"hf_id": "google/gemma-2-2b", "focal_layer": 13, "params_b": 2.0},  # 26 layers (extracted 1-22)
    "llama-3.2-3b": {"hf_id": "meta-llama/Llama-3.2-3B", "focal_layer": 14, "params_b": 3.0},  # 28 layers, gated
    "qwen2.5-1.5b": {"hf_id": "Qwen/Qwen2.5-1.5B", "focal_layer": 14, "params_b": 1.5},  # 28 layers
    "qwen2.5-0.5b": {"hf_id": "Qwen/Qwen2.5-0.5B", "focal_layer": 12, "params_b": 0.5},  # 24 layers
    "qwen2.5-1.5b-instruct": {"hf_id": "Qwen/Qwen2.5-1.5B-Instruct", "focal_layer": 14, "params_b": 1.5},  # 28 layers
    "qwen2.5-3b": {"hf_id": "Qwen/Qwen2.5-3B", "focal_layer": 18, "params_b": 3.0},  # 36 layers
    "qwen2.5-7b": {"hf_id": "Qwen/Qwen2.5-7B", "focal_layer": 14, "params_b": 7.0},  # 28 layers, cloud-GPU extraction
}
DEFAULT_MODEL = "gemma-2-2b"
DEFAULT_SEEDS = (0, 1, 2)

# The run every driver reads unless told otherwise. Declared once so adding a
# run is a one-line change here rather than a sweep through every script.
DEFAULT_RUN = "20260530_001930"
DEFAULT_DATA_ROOT = Path("data") / DEFAULT_RUN


# --- paths ------------------------------------------------------------------
def dataset_path(data_root: Path, trait: str) -> Path:
    """``data/<ts>/sentences/<trait>/sentences_filtered.jsonl``."""
    return data_root / "sentences" / trait / "sentences_filtered.jsonl"


def rep_dir(data_root: Path, model: str, trait: str, token_pooling: str) -> Path:
    """``data/<ts>/representations/<model>/<trait>/<pooling>_token``."""
    return data_root / "representations" / model / trait / f"{token_pooling}_token"


def unembed_cov_path(data_root: Path, model: str) -> Path:
    """``data/<ts>/representations/<model>/unembeddings_covariance.pt``.

    Model-level, not per-trait: the unembedding covariance depends only on the
    model's output embeddings.
    """
    return data_root / "representations" / model / "unembeddings_covariance.pt"


def seeds_base_dir(dataset_name: str, analysis: str, model: str, trait: str, token_pooling: str) -> Path:
    """Parent of the per-seed result dirs (and of ``aggregated/``)."""
    return Path("results") / dataset_name / analysis / model / trait / f"{token_pooling}_token"


def results_dir(dataset_name: str, analysis: str, model: str, trait: str, token_pooling: str, seed: int) -> Path:
    """``results/<ts>/<analysis>/<model>/<trait>/<pooling>_token/seed_<seed>``."""
    return seeds_base_dir(dataset_name, analysis, model, trait, token_pooling) / f"seed_{seed}"


# --- seeds --------------------------------------------------------------
def child_seed(master: int, name: str) -> int:
    """Stable per-component seed derived from the master seed.

    Component names are hashed with crc32 (stable across processes, unlike
    ``hash()``); the result feeds both ``default_rng`` and sklearn
    ``random_state`` consumers.
    """
    seq = np.random.SeedSequence([master, zlib.crc32(name.encode())])
    return int(seq.generate_state(1)[0])


# --- provenance --------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]


def git_commit() -> str | None:
    """HEAD, suffixed ``-dirty`` when ``src/`` has uncommitted changes.

    Without the suffix a result stamped with commit X may have been produced by
    code that was never committed, so checking X out would not reproduce it.

    Scoped to ``src/`` on purpose: a run writes its own tracked artifacts into
    ``results/`` as it goes, so a whole-tree check would stamp every result
    ``-dirty`` from its own output and the flag would carry no information.
    Only the code that computed the numbers bears on reproducibility.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src"],
            cwd=_REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def run_metadata(*, model: str, trait: str, seed: int, token_pooling: str, dataset: Path, focal_layer: int) -> dict:
    """Provenance record written as ``run_metadata.json`` in each seed dir."""
    commit = git_commit()
    versions = {}
    for pkg in ("numpy", "torch", "transformers", "scikit-learn"):
        try:
            versions[pkg] = _im.version(pkg)
        except _im.PackageNotFoundError:
            versions[pkg] = None
    return {
        "model": model,
        "hf_id": MODELS[model]["hf_id"],
        "trait": trait,
        "seed": seed,
        "token_pooling": token_pooling,
        "focal_layer": focal_layer,
        "dataset": str(dataset),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": commit,
        "versions": versions,
    }
