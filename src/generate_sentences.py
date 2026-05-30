"""Generate a politeness sentence dataset into data/<timestamp>/sentences/.

This script owns the output location (data/ + a fresh timestamp); the library just writes into
the directory it's given. Needs OPENROUTER_API_KEY in a repo-root .env (copy .env.example).

Run from the repo root:  uv run python src/generate_sentences.py
"""

from datetime import datetime
from pathlib import Path

from lib.sentences import generate_sentences

MODELS = {
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
    out = generate_sentences(
        Path("data") / datetime.now().strftime("%Y%m%d_%H%M%S") / "sentences",
        n_scenarios=100,
        paraphrases_per_level=3,
        min_acceptance_score=0.70,
        models=MODELS,
    )
    print(f"Done. Dataset written to {out}")
