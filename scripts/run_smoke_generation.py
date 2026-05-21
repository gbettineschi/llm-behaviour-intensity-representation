"""Smoke run of the data-generation pipeline with the research-grade gates.

Writes to data/v1_smoke/ so existing data/v1/ artifacts are untouched.

Usage:
    OPENROUTER_API_KEY=... python scripts/run_smoke_generation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lib.data_generation_v1 import Pipeline  # noqa: E402


SMOKE_CONFIG = {
    "output_dir": "data/v1_smoke_large",
    "traits": ["politeness", "hedging_confidence"],
    "models": {
        "generator": {
            "model": "openrouter/openai/gpt-oss-120b",
            "family": "openai",
            "temperature": 0.5,
            "max_output_tokens": 2048,
        },
        "judge": {
            # Reasoning model — needs headroom for reasoning + JSON output.
            "model": "openrouter/openai/gpt-5-nano",
            "family": "openai",
            "temperature": 0.0,
            "max_output_tokens": 4096,
        },
        "tie_breaker_judge": {
            "model": "openrouter/openai/gpt-5-nano",
            "family": "openai",
            "temperature": 0.0,
            "max_output_tokens": 4096,
        },
    },
    "pipeline": {
        "scenarios_per_trait": 50,
        "scenario_batch_size": 5,
        "paraphrases_per_level": 3,
        "min_acceptance_score": 0.80,
        "max_duplicate_jaccard_scenarios": 0.85,
        "max_duplicate_jaccard": 0.85,
        "lexical_baseline_warning_accuracy": 0.55,
        "human_validation_fraction": 0.15,
        # 3 workers (not 4) — fewer OpenRouter "peer closed connection" drops.
        "max_workers": 3,
        "resume": True,
        # Research-grade knobs added in this version:
        "enable_intensity_scorer": True,
        "intensity_min_gap": 0.10,
        "max_length_ratio": 1.15,
        "max_retries": 2,
        "max_shortcut_repair_rounds": 2,
        "tie_breaker_margin": 0.10,
    },
    "validation": {
        "warn_if_same_family_generator_and_judge": True,
        "reject_if_same_family_generator_and_judge": False,
    },
}


def main() -> None:
    pipeline = Pipeline(SMOKE_CONFIG)
    for trait in SMOKE_CONFIG["traits"]:
        print("\n" + "=" * 60)
        print(f"  {trait}")
        print("=" * 60)
        pipeline.make_all(trait)
    pipeline.export_prompts()
    print("\nSmoke run finished. See data/v1_smoke/.")

    # Print a compact summary of accept/reject rates per trait.
    out_root = Path("data/v1_smoke")
    print("\n── Smoke summary ──")
    for trait in SMOKE_CONFIG["traits"]:
        td = out_root / trait
        para = (td / "paraphrases.jsonl")
        judg = (td / "judged.jsonl")
        acc = (td / "accepted.jsonl")
        n_para = sum(1 for _ in para.open()) if para.exists() else 0
        n_judg = sum(1 for _ in judg.open()) if judg.exists() else 0
        n_acc = sum(1 for _ in acc.open()) if acc.exists() else 0
        rate = (n_acc / n_judg) if n_judg else 0.0
        print(f"  {trait}: paraphrases={n_para}  judged={n_judg}  accepted={n_acc}  rate={rate:.2%}")

    # Print top of audit report if present.
    for trait in SMOKE_CONFIG["traits"]:
        rpt = out_root / trait / "audit_report.json"
        if rpt.exists():
            data = json.loads(rpt.read_text())
            print(
                f"  audit[{trait}]: lex_baseline_acc="
                f"{data.get('lexical_baseline_accuracy')}, "
                f"length_imbalance_warning={data.get('length_imbalance_warning')}, "
                f"shortcut_repair_rounds={data.get('shortcut_repair_rounds')}"
            )


if __name__ == "__main__":
    main()
