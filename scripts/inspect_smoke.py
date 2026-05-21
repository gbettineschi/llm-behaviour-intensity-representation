"""Inspect a generation run: rejection rates, gate fires, length spreads, intensity monotonicity.

Usage: python scripts/inspect_smoke.py [output_dir]
       output_dir defaults to data/v1_smoke_large.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/v1_smoke_large")
    if not root.exists():
        print(f"{root} does not exist — run scripts/run_smoke_generation.py first.")
        return

    for trait_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        trait = trait_dir.name
        scenarios = jsonl(trait_dir / "scenarios.jsonl")
        ladders = jsonl(trait_dir / "ladders.jsonl")
        paraphrases = jsonl(trait_dir / "paraphrases.jsonl")
        judged = jsonl(trait_dir / "judged.jsonl")
        accepted = jsonl(trait_dir / "accepted.jsonl")
        failed_para = jsonl(trait_dir / "failed_paraphrases.jsonl")
        failed_lad = jsonl(trait_dir / "failed_ladders.jsonl")
        audit_path = trait_dir / "audit_report.json"
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}

        print("\n" + "=" * 60)
        print(f"  {trait}")
        print("=" * 60)
        print(
            f"scenarios={len(scenarios)}  ladders={len(ladders)}  "
            f"paraphrases={len(paraphrases)}  judged={len(judged)}  "
            f"accepted={len(accepted)}"
        )
        print(f"failed_ladders={len(failed_lad)}  failed_paraphrases={len(failed_para)}")

        # Per-bundle: was it accepted, and which gates fired?
        bundles_judged: dict[str, list[dict]] = defaultdict(list)
        for r in judged:
            bundles_judged[r["scenario_id"]].append(r)

        accepted_bundles = 0
        gate_failures_counter: Counter[str] = Counter()
        intensity_means_log: list[tuple[str, dict]] = []
        for sid, rows in bundles_judged.items():
            judge_blk = rows[0].get("judge", {})
            any_accepted = any(r.get("accepted") for r in rows)
            if any_accepted:
                accepted_bundles += 1
            for f in (judge_blk.get("gate_failures") or []):
                # Normalize the failure family
                if f.startswith("length-balance"):
                    gate_failures_counter["length_balance"] += 1
                elif f.startswith("intensity-monotonicity"):
                    gate_failures_counter["intensity_monotonicity"] += 1
                else:
                    gate_failures_counter["other"] += 1
            gate = judge_blk.get("intensity_gate", {})
            if gate.get("intensity_means_by_level"):
                intensity_means_log.append((sid, gate["intensity_means_by_level"]))

        n_bundles = len(bundles_judged)
        rate = (accepted_bundles / n_bundles) if n_bundles else 0.0
        print(
            f"bundle-level acceptance: {accepted_bundles}/{n_bundles} "
            f"({rate:.2%})"
        )
        if gate_failures_counter:
            print(f"gate failures by family: {dict(gate_failures_counter)}")
        else:
            print("gate failures by family: none recorded")

        # Item-level acceptance breakdown
        item_total = len(judged)
        item_accepted = sum(1 for r in judged if r.get("accepted"))
        print(
            f"item-level acceptance: {item_accepted}/{item_total} "
            f"({(item_accepted / item_total) if item_total else 0.0:.2%})"
        )

        # Length spread per bundle (within-bundle max/min word-count ratio)
        bundle_length_ratios: list[float] = []
        for sid, rows in bundles_judged.items():
            wc_by_level: dict[str, list[int]] = defaultdict(list)
            for r in rows:
                wc_by_level[r["level"]].append(len(r["text"].split()))
            means = [mean(v) for v in wc_by_level.values() if v]
            if len(means) >= 2 and min(means) > 0:
                bundle_length_ratios.append(max(means) / min(means))
        if bundle_length_ratios:
            print(
                f"per-bundle length-ratio (max-mean/min-mean across levels): "
                f"mean={mean(bundle_length_ratios):.3f}  "
                f"max={max(bundle_length_ratios):.3f}  "
                f"n>1.15={sum(1 for r in bundle_length_ratios if r > 1.15)}/"
                f"{len(bundle_length_ratios)}"
            )

        # Intensity monotonicity summary
        mono_ok = 0
        mono_total = 0
        gap_min: list[float] = []
        gap_max: list[float] = []
        for _, means_dict in intensity_means_log:
            vals = [means_dict.get(l) for l in ("low", "mid", "high")]
            if any(v is None for v in vals):
                continue
            mono_total += 1
            if vals[0] < vals[1] < vals[2]:
                mono_ok += 1
            gaps = [vals[1] - vals[0], vals[2] - vals[1]]
            gap_min.append(min(gaps))
            gap_max.append(max(gaps))
        if mono_total:
            print(
                f"blind intensity monotonic per bundle: {mono_ok}/{mono_total} "
                f"({mono_ok / mono_total:.2%}); "
                f"min-gap mean={mean(gap_min):.3f}; max-gap mean={mean(gap_max):.3f}"
            )

        # Audit report headline numbers
        if audit:
            print(
                f"audit: lex_baseline_acc={audit.get('lexical_baseline_accuracy')}, "
                f"length_imbalance_warning={audit.get('length_imbalance_warning')}, "
                f"shortcut_repair_rounds={audit.get('shortcut_repair_rounds')}, "
                f"duplicate_pairs={audit.get('num_duplicate_pairs')}"
            )

        # Word counts in accepted data
        wc_by_level_acc: dict[str, list[int]] = defaultdict(list)
        for r in accepted:
            wc_by_level_acc[r["level"]].append(len(r["text"].split()))
        if wc_by_level_acc:
            print("accepted word counts (mean ± stdev):")
            for lvl in ("low", "mid", "high"):
                ws = wc_by_level_acc.get(lvl, [])
                if not ws:
                    print(f"  {lvl}: <empty>")
                    continue
                m = mean(ws)
                sd = math.sqrt(sum((w - m) ** 2 for w in ws) / max(1, len(ws) - 1)) if len(ws) > 1 else 0.0
                print(f"  {lvl}: {m:.2f} ± {sd:.2f}  (n={len(ws)})")


if __name__ == "__main__":
    main()
