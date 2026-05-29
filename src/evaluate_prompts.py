"""Run the blind human-eval web app and save its results.

Opens a browser: pick one of the data/<timestamp> prompt datasets, blindly rank its triplets,
and finish. The collected results are saved as a single file under
results/human_eval/human_eval_<timestamp>.json (this script owns the saving).

Run from the repo root:  uv run python src/evaluate_prompts.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from lib.human_eval import human_eval

if __name__ == "__main__":
    results = human_eval(Path("data"))
    if not results.get("answers"):
        print("No answers recorded; nothing saved.")
        sys.exit(0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("results") / "human_eval" / f"human_eval_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"result_timestamp": ts, **results}, indent=2, ensure_ascii=False)
    )
    print(f"Saved {out}")
