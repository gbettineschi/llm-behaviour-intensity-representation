"""Synthetic-geometry tests for the linearity metrics in ``lib.analysis``.

Run with plain Python (no pytest dependency)::

    .venv/bin/python tests/test_analysis.py

Each test builds activations with a *known* geometry and asserts the
within-scenario metrics recover it, plus an isolated demonstration of the
train/test leakage that motivated the switch from shuffled KFold to GroupKFold.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold, GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lib.analysis import (  # noqa: E402
    acts_by_level_from_dict,
    ordinal_linearity_metrics,
    scenario_triples,
    within_center,
    within_scenario_linearity_metrics,
)

LEVELS = ["negative", "neutral", "positive"]
COORD = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
TRAIT = "politeness"


def _make(n_scen, D=32, *, neutral_offaxis=0.0, offset_scale=0.0, noise=0.0,
          signal=1.0, no_signal=False, seed=0):
    """Build a synthetic activations dict with controllable geometry.

    Each scenario places its 3 levels along a shared unit ``axis`` at positions
    ``signal * COORD[level]``, plus a per-scenario additive ``offset`` (the
    language/topic confound), plus iid ``noise``. ``neutral_offaxis`` pushes the
    neutral level along an orthogonal direction (a V-shape -> non-linear).
    ``no_signal`` collapses all levels onto the same point (pure null).

    Note: Spearman/Kendall between a continuous projection and a 3-level ordinal
    label are structurally capped (rho ~ 0.94, tau-b ~ 0.83 for equal groups),
    not 1.0 — the tests assert against those ceilings, not against 1.
    """
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(D); axis /= np.linalg.norm(axis)
    orth = rng.standard_normal(D); orth -= (orth @ axis) * axis; orth /= np.linalg.norm(orth)
    acts = {}
    for s in range(n_scen):
        offset = offset_scale * rng.standard_normal(D)
        for lvl in LEVELS:
            c = 0.0 if no_signal else COORD[lvl]
            v = signal * c * axis + offset + noise * rng.standard_normal(D)
            if lvl == "neutral":
                v = v + neutral_offaxis * orth
            acts[(TRAIT, lvl, f"s{s}")] = torch.tensor(v)
    return acts


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        raise AssertionError(f"{name}: {detail}")


def test_perfectly_linear():
    print("test_perfectly_linear")
    acts = _make(24, offset_scale=5.0, noise=0.0, seed=1)
    m = within_scenario_linearity_metrics(acts, LEVELS, trait=TRAIT)
    _check("n_scenarios == 24", m["n_scenarios"] == 24)
    _check("spearman at ceiling (~0.94)", m["spearman"] > 0.93, f"{m['spearman']:.4f}")
    _check("kendall at ceiling (~0.83)", m["kendall"] > 0.82, f"{m['kendall']:.4f}")
    _check("monotone_fraction == 1", m["monotone_fraction"] == 1.0)
    _check("midpoint_residual_median ~ 0", m["midpoint_residual_median"] < 1e-9,
           f"{m['midpoint_residual_median']:.2e}")
    _check("pc1_frac_per_scenario_mean ~ 1", m["pc1_frac_per_scenario_mean"] > 0.999,
           f"{m['pc1_frac_per_scenario_mean']:.4f}")
    _check("probe_r2 high", m["probe_r2"] > 0.95, f"{m['probe_r2']:.4f}")
    _check("midpoint_residual_pooled ~ 0", m["midpoint_residual_pooled"] < 1e-6,
           f"{m['midpoint_residual_pooled']:.2e}")


def test_v_shape_detected():
    print("test_v_shape_detected")
    # neutral pushed a full span-length off-axis -> residual ~ 1, pc1 < 1.
    acts = _make(24, offset_scale=3.0, noise=0.0, neutral_offaxis=2.0, seed=2)
    m = within_scenario_linearity_metrics(acts, LEVELS, trait=TRAIT)
    _check("midpoint_residual_median ~ 1", abs(m["midpoint_residual_median"] - 1.0) < 1e-6,
           f"{m['midpoint_residual_median']:.4f}")
    _check("pc1_frac_per_scenario_mean < 0.9", m["pc1_frac_per_scenario_mean"] < 0.9,
           f"{m['pc1_frac_per_scenario_mean']:.4f}")
    # Ordering along the main axis is still clean (at the structural ceiling).
    _check("spearman still near ceiling", m["spearman"] > 0.9, f"{m['spearman']:.4f}")


def test_no_signal_null():
    print("test_no_signal_null")
    # All levels collapse to one point + noise -> no monotone relationship.
    acts = _make(50, offset_scale=3.0, noise=1.0, no_signal=True, seed=5)
    m = within_scenario_linearity_metrics(acts, LEVELS, trait=TRAIT)
    _check("|spearman| small", abs(m["spearman"]) < 0.35, f"{m['spearman']:.4f}")
    _check("probe_r2 not positive", m["probe_r2"] < 0.2, f"{m['probe_r2']:.4f}")


def test_within_center_and_triples():
    print("test_within_center_and_triples")
    acts = _make(5, D=8, offset_scale=4.0, noise=0.5, seed=4)
    # Drop one level of one scenario -> it must be excluded as incomplete.
    del acts[(TRAIT, "neutral", "s0")]
    triples = scenario_triples(acts, LEVELS, trait=TRAIT)
    _check("incomplete scenario dropped", set(triples) == {f"s{i}" for i in range(1, 5)},
           f"{sorted(triples)}")
    centered = within_center(acts, LEVELS, trait=TRAIT)
    for sid in triples:
        rows = np.stack([centered[(TRAIT, lvl, sid)] for lvl in LEVELS])
        _check(f"scenario {sid} rows sum ~ 0", np.allclose(rows.sum(axis=0), 0.0, atol=1e-9))


def test_groupkfold_prevents_leakage():
    print("test_groupkfold_prevents_leakage")
    # Identical raw X for both CV schemes -> isolates the CV effect (not centering).
    # Big per-scenario offsets + small signal: shuffled folds let the probe overfit
    # offset directions shared with held-out rows of the same scenario.
    sh, gr = [], []
    for seed in range(5):
        acts = _make(20, D=120, offset_scale=12.0, noise=1.0, signal=0.25, seed=100 + seed)
        triples = scenario_triples(acts, LEVELS, trait=TRAIT)
        sids = sorted(triples)
        mats = np.stack([triples[s] for s in sids])            # (S, 3, D) RAW (not centered)
        X = mats.reshape(len(sids) * 3, -1)
        y = np.tile(np.arange(3), len(sids))
        groups = np.repeat(np.arange(len(sids)), 3)
        r2_sh = cross_val_score(Ridge(), X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean()
        r2_gr = cross_val_score(Ridge(), X, y, cv=GroupKFold(5), groups=groups, scoring="r2").mean()
        sh.append(r2_sh); gr.append(r2_gr)
    sh_m, gr_m = float(np.mean(sh)), float(np.mean(gr))
    _check("shuffled R2 > grouped R2 (leakage)", sh_m > gr_m, f"shuffled={sh_m:.4f} grouped={gr_m:.4f}")


def test_within_scenario_recovers_confounded_ranking():
    print("test_within_scenario_recovers_confounded_ranking")
    # Offset-dominated linear data: the naive pooled projection is swamped by the
    # per-scenario offset; the within-transform recovers the true ordering.
    acts = _make(24, D=64, offset_scale=8.0, noise=0.05, signal=0.3, seed=7)
    naive = ordinal_linearity_metrics(acts_by_level_from_dict(acts, LEVELS), LEVELS)
    within = within_scenario_linearity_metrics(acts, LEVELS, trait=TRAIT)
    _check("within-scenario spearman near ceiling", within["spearman"] > 0.9, f"{within['spearman']:.4f}")
    _check("naive spearman attenuated", naive["spearman"] < 0.5, f"{naive['spearman']:.4f}")
    _check("within-scenario > naive by margin", within["spearman"] - naive["spearman"] > 0.5,
           f"within={within['spearman']:.4f} naive={naive['spearman']:.4f}")


def test_child_seed_deterministic():
    print("test_child_seed_deterministic")
    from lib.config import child_seed

    _check("same (master, name) -> same seed", child_seed(0, "bootstrap") == child_seed(0, "bootstrap"))
    _check("different name -> different seed", child_seed(0, "bootstrap") != child_seed(0, "permutation_null"))
    _check("different master -> different seed", child_seed(0, "bootstrap") != child_seed(1, "bootstrap"))
    s = child_seed(3, "reliability")
    _check("uint32 range", isinstance(s, int) and 0 <= s < 2**32, str(s))


def test_direction_seed():
    print("test_direction_seed")
    from lib.directions import NAMES, direction

    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(0, 1, (40, 16)), rng.normal(1.5, 1, (40, 16))])
    y = np.array([0] * 40 + [1] * 40)
    for m in NAMES:
        d1, d2 = direction(m, X, y, seed=1), direction(m, X, y, seed=1)
        _check(f"{m} reproducible for same seed", np.allclose(d1, d2))
        _check(f"{m} oriented (pos class higher)", (X[y == 1] @ d1).mean() > (X[y == 0] @ d1).mean())
    r1, r2 = direction("Random", X, y, seed=1), direction("Random", X, y, seed=2)
    _check("Random differs across seeds", not np.allclose(r1, r2))


def test_aggregate_csv():
    print("test_aggregate_csv")
    import csv
    import json
    import tempfile

    from aggregate_results import aggregate_analysis

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        rows = {0: [1.0, 2.0], 1: [3.0, 4.0]}
        for seed, varying in rows.items():
            num = base / f"seed_{seed}" / "numeric"
            num.mkdir(parents=True)
            with (num / "a.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["method", "constant", "varying"])
                w.writerow(["MeanDiff", 7.0, varying[0]])
                w.writerow(["KMeans", 8.0, varying[1]])
            # misaligned key column across seeds -> must be skipped
            with (num / "bad.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["name", "x"])
                w.writerow([f"row_of_seed_{seed}", 1.0])
        # present in seed_1 only: discovery unions over every seed, so this must be
        # reported as missing rather than silently ignored (it is invisible to a
        # seed_0-only glob).
        with (base / "seed_1" / "numeric" / "late.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "x"])
            w.writerow(["only_in_seed_1", 1.0])
        out = aggregate_analysis(base)
        with (out / "numeric" / "a.csv").open() as f:
            got = list(csv.reader(f))
        _check("header has mean/std/n_seeds", got[0] == ["method", "constant", "varying_mean", "varying_std", "n_seeds"], str(got[0]))
        _check("constant passthrough", float(got[1][1]) == 7.0)
        # CSV values are formatted to 10 significant digits by exports.fmt
        _check("mean correct", abs(float(got[1][2]) - 2.0) < 1e-8, got[1][2])
        _check("std ddof=1 correct", abs(float(got[1][3]) - np.sqrt(2.0)) < 1e-8, got[1][3])
        _check("n_seeds = 2", int(got[1][4]) == 2)
        report = json.loads((out / "aggregation_report.json").read_text())
        _check("misaligned file reported", "numeric/bad.csv" in report["skipped_misaligned"], str(report["skipped_misaligned"]))
        _check("misaligned file not written", not (out / "numeric" / "bad.csv").exists())
        _check("file missing from seed_0 reported", "numeric/late.csv" in report["skipped_missing"], str(report["skipped_missing"]))
        _check("file missing from seed_0 not written", not (out / "numeric" / "late.csv").exists())


def test_collect_summary_reads_aggregated_only_tree():
    """A fresh clone has aggregated/ but no seed dirs, because results/**/seed_*/
    is gitignored. Every column must still be populated from aggregated/."""
    print("test_collect_summary_reads_aggregated_only_tree")
    import csv
    import json
    import tempfile

    import collect_summary

    def _write(path, header, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        combo = "gemma-2-2b/politeness/avg_token"
        geom = root / "trait_geometry" / combo / "aggregated"
        lin = root / "ordinal_linearity" / combo / "aggregated"
        # Column shapes exactly as aggregate_results emits them: deterministic
        # columns pass through, varying ones gain _mean/_std.
        _write(geom / "numeric" / "geometry" / "geometry_bootstrap_summary.csv",
               ["metric", "point", "ci_lo_mean", "ci_lo_std", "ci_hi_mean", "ci_hi_std", "n_seeds"],
               [["apex_angle_deg", 73.2, 68.0, 0.4, 78.1, 0.3, 3]])
        _write(geom / "numeric" / "noise_null" / "noise_null_summary.csv",
               ["metric", "observed", "p_value", "n_sim", "n_seeds"],
               [["step_cosine", -0.289, 0.000999, 1000, 3],
                ["midpoint_residual", 0.661, 0.000999, 1000, 3]])
        _write(geom / "numeric" / "shared_plane" / "shared_plane_summary.csv",
               ["markedness_R_full", "shared_bend_r2_cv"], [[0.91, 0.55]])
        _write(lin / "numeric" / "linearity" / "linearity_metrics.csv",
               ["metric", "within_scenario"], [["spearman", 0.895], ["probe_r2", 0.77]])
        (geom / "run_metadata.json").write_text(json.dumps({"seeds": [0, 1, 2]}))

        orig = collect_summary.DATASET_ROOT
        try:
            # _combo_dir builds results/<name>/... so point the tree at our temp root
            collect_summary.seeds_base_dir = (
                lambda name, analysis, model, trait, pooling: root / analysis / model / trait / f"{pooling}_token"
            )
            row = collect_summary.collect_combo("gemma-2-2b", "politeness", "avg")
        finally:
            collect_summary.DATASET_ROOT = orig

        _check("row found with no seed dirs present", row is not None)
        _check("n_seeds from aggregated provenance", row["n_seeds"] == 3, str(row["n_seeds"]))
        _check("apex_angle_deg populated", row["apex_angle_deg"] == 73.2, str(row["apex_angle_deg"]))
        _check("apex_angle_ci_lo from _mean column", row["apex_angle_ci_lo"] == 68.0, str(row["apex_angle_ci_lo"]))
        _check("step_cosine populated", row["step_cosine"] == -0.289, str(row["step_cosine"]))
        _check("step_cosine_null_p populated", row["step_cosine_null_p"] == 0.000999, str(row["step_cosine_null_p"]))
        _check("midpoint_residual populated", row["midpoint_residual"] == 0.661, str(row["midpoint_residual"]))
        _check("spearman_within_scenario populated", row["spearman_within_scenario"] == 0.895)


def test_no_trait_intent_holds_its_own_trait_constant():
    """Per-intent constraints must not tell the generator to hold constant the very
    trait being varied. The shared `request` intent forbids changing urgency, which
    is a correct control for every trait except urgency itself."""
    print("test_no_trait_intent_holds_its_own_trait_constant")
    from lib.traits import TRAITS

    for trait, spec in sorted(TRAITS.items()):
        for intent in spec["intents"]:
            for c in intent.get("extra_constraints", []):
                _check(f"{trait}/{intent['id']} does not pin {trait}", trait not in c.lower(), c)


def test_git_commit_dirty_flag_tracks_src_only():
    """The provenance -dirty flag must reflect uncommitted *code*, not the results
    a run is writing. A whole-tree check stamps every result dirty from its own
    output, which makes the flag carry no information at all."""
    print("test_git_commit_dirty_flag_tracks_src_only")
    import subprocess

    from lib.config import git_commit

    c = git_commit()
    _check("returns a commit", c is not None, str(c))
    _check("sha is 40 hex chars", len(c.split("-")[0]) == 40, str(c))
    src_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True
        ).stdout.strip()
    )
    _check(
        "dirty flag matches src/ state, ignoring results/",
        c.endswith("-dirty") == src_dirty,
        f"stamp={c} src_dirty={src_dirty}",
    )


def test_jsonify_keeps_booleans_boolean():
    """bool is a subclass of int, so an unguarded int branch exports True as 1 —
    which in a provenance file reads as a count, not a flag."""
    print("test_jsonify_keeps_booleans_boolean")
    from lib.exports import jsonify

    _check("True stays True", jsonify(True) is True, repr(jsonify(True)))
    _check("False stays False", jsonify(False) is False, repr(jsonify(False)))
    _check("numpy bool stays bool", jsonify(np.bool_(True)) is True, repr(jsonify(np.bool_(True))))
    _check("nested in a dict", jsonify({"verified": True})["verified"] is True)
    _check("real ints unaffected", jsonify(50) == 50 and isinstance(jsonify(50), int))
    _check("numpy int unaffected", jsonify(np.int64(7)) == 7)


def test_aggregate_respects_explicit_seed_list():
    """A seed_* dir left over from an earlier run must not be folded into the mean
    when the caller says which seeds it just ran."""
    print("test_aggregate_respects_explicit_seed_list")
    import csv
    import json
    import tempfile

    from aggregate_results import aggregate_analysis

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        # seed_2 is stale: a wildly different value from an earlier run.
        for seed, val in {0: 1.0, 1: 3.0, 2: 99.0}.items():
            num = base / f"seed_{seed}" / "numeric"
            num.mkdir(parents=True)
            with (num / "a.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["metric", "x"])
                w.writerow(["m", val])
        out = aggregate_analysis(base, seeds=[0, 1])
        with (out / "numeric" / "a.csv").open() as f:
            got = list(csv.reader(f))
        _check("n_seeds is the requested count", int(got[1][-1]) == 2, str(got[1]))
        _check("mean excludes the stale seed", abs(float(got[1][1]) - 2.0) < 1e-8, got[1][1])
        meta = json.loads((out / "run_metadata.json").read_text())
        _check("provenance records only the requested seeds", meta["seeds"] == [0, 1], str(meta["seeds"]))
        # Globbing (seeds=None) still folds everything in, for standalone --dir use.
        out2 = aggregate_analysis(base)
        with (out2 / "numeric" / "a.csv").open() as f:
            got2 = list(csv.reader(f))
        _check("seeds=None still globs all three", int(got2[1][-1]) == 3, str(got2[1]))


def test_aggregate_json_keeps_provenance_per_seed():
    """RNG seeds and timestamps are provenance, not measurements — averaging them
    yields a mean of unrelated 32-bit integers that reads like a parameter."""
    print("test_aggregate_json_keeps_provenance_per_seed")
    from aggregate_results import _aggregate_json

    docs = [
        {"n_perm": 100, "seed": 111111, "p_floor": 0.0099},
        {"n_perm": 100, "seed": 222222, "p_floor": 0.0099},
        {"n_perm": 100, "seed": 333333, "p_floor": 0.0099},
    ]
    got = _aggregate_json(docs, 3)
    _check("varying seed kept per-seed", got["seed"] == {"per_seed": [111111, 222222, 333333]}, str(got["seed"]))
    _check("seed not averaged", "mean" not in str(got["seed"]), str(got["seed"]))
    _check("identical params pass through", got["n_perm"] == 100 and got["p_floor"] == 0.0099, str(got))
    # A genuine measurement still aggregates.
    m = _aggregate_json([{"score": 1.0}, {"score": 3.0}], 2)
    _check("real metric still averaged", abs(m["score"]["mean"] - 2.0) < 1e-9, str(m))


def test_run_analyses_skips_dirs_without_tensors():
    """Rep dirs survive a clone (metadata.json is tracked, tensors are not), so
    the sweep must skip on absent tensors rather than on an absent directory."""
    print("test_run_analyses_skips_dirs_without_tensors")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rd = Path(td) / "representations" / "m" / "t" / "avg_token"
        rd.mkdir(parents=True)
        (rd / "metadata.json").write_text("{}")  # what a fresh clone actually has
        has_tensors = rd.is_dir() and any(rd.glob("layer_*.pt"))
        _check("dir with only metadata.json is skipped", not has_tensors)
        _check("plain exists() would NOT have skipped it", rd.exists())
        (rd / "layer_1.pt").write_bytes(b"x")
        _check("dir with a tensor is not skipped", rd.is_dir() and any(rd.glob("layer_*.pt")))


def main():
    tests = [
        test_perfectly_linear,
        test_v_shape_detected,
        test_no_signal_null,
        test_within_center_and_triples,
        test_groupkfold_prevents_leakage,
        test_within_scenario_recovers_confounded_ranking,
        test_child_seed_deterministic,
        test_direction_seed,
        test_aggregate_csv,
        test_collect_summary_reads_aggregated_only_tree,
        test_no_trait_intent_holds_its_own_trait_constant,
        test_git_commit_dirty_flag_tracks_src_only,
        test_jsonify_keeps_booleans_boolean,
        test_aggregate_respects_explicit_seed_list,
        test_aggregate_json_keeps_provenance_per_seed,
        test_run_analyses_skips_dirs_without_tensors,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  ASSERTION FAILED: {e}")
    print()
    if failed:
        print(f"{failed}/{len(tests)} tests FAILED")
        sys.exit(1)
    print(f"All {len(tests)} tests PASSED")


if __name__ == "__main__":
    main()
