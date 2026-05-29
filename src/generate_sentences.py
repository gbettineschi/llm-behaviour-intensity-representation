"""Generate a politeness sentence dataset into data/<timestamp>/sentences/.

This script owns the output location (data/ + a fresh timestamp); the library just writes into
the directory it's given. Needs OPENROUTER_API_KEY in a repo-root .env (copy .env.example).

Run from the repo root:  uv run python src/generate_sentences.py
"""

from datetime import datetime
from pathlib import Path

from lib.sentences import generate_sentences

if __name__ == "__main__":
    out = generate_sentences(
        Path("data") / datetime.now().strftime("%Y%m%d_%H%M%S") / "sentences",
        n_scenarios=300,
        paraphrases_per_level=3,
        min_acceptance_score=0.70,
    )
    print(f"Done. Dataset written to {out}")
