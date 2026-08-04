"""Generate a trait sentence dataset into data/<timestamp>/sentences/<trait>/.

This script owns the output location (data/ + a timestamp); the library just writes into
the directory it's given. Traits are declared in src/lib/traits.py. Pass --data-root to
add a trait to an existing dataset root (e.g. data/20260530_001930) so all traits share
one root; by default a fresh timestamp is created. Needs OPENROUTER_API_KEY in a
repo-root .env (copy .env.example).

Run from the repo root:  uv run python src/generate_sentences.py --trait politeness
"""

import argparse
from datetime import datetime
from pathlib import Path

from lib.sentences import generate_sentences
from lib.traits import DEFAULT_TRAIT, TRAITS

# The LLMs that *build* the dataset — unrelated to lib.config.MODELS, which is
# the registry of models whose activations are studied.
PIPELINE_LLMS = {
    "generator": {
        "model": "openrouter/google/gemini-2.0-flash-001",
        "family": "google",
        "temperature": 0.5,
        "max_output_tokens": 2048,
        "litellm_kwargs": {
            "extra_body": {
                "provider": {"order": ["google-vertex"], "allow_fallbacks": True}
            }
        },
    },
    "judge": {
        "model": "openrouter/deepseek/deepseek-v4-flash",
        "family": "deepseek",
        "temperature": 0.0,
        "max_output_tokens": 8192,
        "litellm_kwargs": {
            "extra_body": {"provider": {"order": ["alibaba"], "allow_fallbacks": True}}
        },
    },
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trait", choices=sorted(TRAITS), default=DEFAULT_TRAIT)
    ap.add_argument(
        "--data-root", type=Path, default=None,
        help="Existing data/<timestamp> root to add this trait to (default: a fresh timestamp).",
    )
    args = ap.parse_args()

    root = args.data_root or Path("data") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out = generate_sentences(
        root / "sentences" / args.trait,
        trait=args.trait,
        n_scenarios=100,
        paraphrases_per_level=3,
        min_acceptance_score=0.70,
        models=PIPELINE_LLMS,
    )
    print(f"Done. Dataset written to {out}")
