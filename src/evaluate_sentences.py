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
        json.dumps({"eval_timestamp": ts, **results}, indent=2, ensure_ascii=False)
    )
    print(f"Saved {out}")
