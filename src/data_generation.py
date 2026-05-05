from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

try:
    from litellm import completion
except Exception:  # pragma: no cover
    completion = None


LEVELS = ["low", "mid", "high"]
TRAITS = ["politeness", "hedging_confidence"]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def jsonl_write(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def jsonl_read(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:80]


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def token_set(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def jaccard(a: str, b: str) -> float:
    sa = token_set(a)
    sb = token_set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


@dataclass
class ModelSpec:
    model: str
    family: str
    temperature: float = 0.0
    max_output_tokens: int = 1200


class LLMClient:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        if completion is None:
            raise RuntimeError(
                "litellm is not installed. Install requirements before using the pipeline."
            )

    def _extract_json(self, raw: str) -> Any:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", raw, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(1))

    def call_json(self, system: str, user: str, schema_hint: Optional[str] = None) -> Any:
        prompt = user
        if schema_hint:
            prompt = f"{user}\n\nReturn JSON only. Expected structure:\n{schema_hint}"
        resp = completion(
            model=self.spec.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=self.spec.temperature,
            max_tokens=self.spec.max_output_tokens,
        )
        content = resp["choices"][0]["message"]["content"]
        return self._extract_json(content)


class Pipeline:
    def __init__(self, config_path: Path):
        self.config = load_yaml(config_path)
        self.config_path = config_path
        self.root = config_path.parent.parent
        self.output_root = self.root / self.config.get("output_dir", "output")
        ensure_dir(self.output_root)

        gen_cfg = self.config["models"]["generator"]
        judge_cfg = self.config["models"]["judge"]
        tiebreak_cfg = self.config["models"]["tie_breaker_judge"]

        self.generator = LLMClient(ModelSpec(**gen_cfg))
        self.judge = LLMClient(ModelSpec(**judge_cfg))
        self.tie_breaker = LLMClient(ModelSpec(**tiebreak_cfg))

        self.levels: List[str] = list(self.config["pipeline"].get("levels", LEVELS))
        self._validate_model_separation()

    def trait_dir(self, trait: str) -> Path:
        out = self.output_root / trait
        ensure_dir(out)
        return out

    def _validate_model_separation(self) -> None:
        gen_family = self.config["models"]["generator"]["family"]
        judge_family = self.config["models"]["judge"]["family"]
        if gen_family == judge_family:
            msg = (
                f"Generator family ({gen_family}) and judge family ({judge_family}) are the same. "
                "This increases leakage risk."
            )
            if self.config["validation"].get("reject_if_same_family_generator_and_judge", False):
                raise ValueError(msg)
            if self.config["validation"].get("warn_if_same_family_generator_and_judge", True):
                print(f"WARNING: {msg}")

    def _scenario_system_prompt(self, trait: str) -> str:
        return (
            "You are building a research dataset for representation geometry. "
            "Create base scenarios, not labels or explanations. "
            "Scenarios must keep future ordinal rewrites content-controlled."
        )

    def _scenario_user_prompt(self, trait: str, n: int) -> str:
        schema = load_yaml(self.root / "config" / "scenario_schema.yaml")
        shared = schema["definitions"]["shared_fields"]
        trait_info = schema["traits"][trait]
        return (
            f"Generate {n} base scenarios for the trait '{trait}'.\n"
            f"Use three-level ordinal rewriting later: low, mid, high.\n"
            f"Speech-act restriction: {trait_info['speech_act']}.\n"
            f"Required trait-specific fields: {trait_info['required_fields']}.\n"
            f"Generation constraints: {trait_info['generation_constraints']}.\n"
            f"Shared fields to fill: {list(shared.keys())}.\n"
            "Return a JSON array of scenario objects. Each object must contain concise field values."
        )

    def make_scenarios(self, trait: str) -> None:
        n = int(self.config["pipeline"]["scenarios_per_trait"])
        system = self._scenario_system_prompt(trait)
        user = self._scenario_user_prompt(trait, n)
        schema_hint = json.dumps(
            [
                {
                    "scenario_id": f"{trait}-001",
                    "trait": trait,
                    "domain": "workplace",
                    "audience_relation": "peer",
                    "communicative_goal": "request a file",
                    "proposition_or_request": "speaker requests the latest budget spreadsheet",
                    "speech_act": "request",
                    "requested_action": "send the latest budget spreadsheet",
                    "imposition_level": "medium",
                    "urgency_level": "low",
                    "social_distance": "moderate",
                    "notes": "keep requested action fixed across rewrites",
                }
            ],
            ensure_ascii=False,
        )
        scenarios = self.generator.call_json(system, user, schema_hint)
        cleaned: List[Dict[str, Any]] = []
        for i, row in enumerate(scenarios, start=1):
            row["trait"] = trait
            row.setdefault("scenario_id", f"{trait}-{i:03d}")
            cleaned.append(row)
        jsonl_write(self.trait_dir(trait) / "scenarios.jsonl", cleaned)
        print(f"Wrote {len(cleaned)} scenarios to {self.trait_dir(trait) / 'scenarios.jsonl'}")

    def _ladder_system_prompt(self, trait: str) -> str:
        rubric = (self.root / "config" / "construct_rubric.md").read_text(encoding="utf-8")
        return (
            "You create controlled ordinal ladders for NLP research. "
            "Keep content fixed and vary only trait intensity.\n\n"
            f"Rubric:\n{rubric}"
        )

    def _ladder_user_prompt(self, trait: str, scenario: Dict[str, Any]) -> str:
        return (
            f"Create one 3-level canonical ladder for trait '{trait}'.\n"
            f"Scenario:\n{json.dumps(scenario, ensure_ascii=False, indent=2)}\n\n"
            "Levels must be low, mid, high.\n"
            "The proposition or requested action must remain invariant.\n"
            "Return JSON with keys: scenario_id, trait, invariant_content, ladder.\n"
            "'ladder' must map each level to a single sentence."
        )

    def make_ladders(self, trait: str) -> None:
        scenarios = jsonl_read(self.trait_dir(trait) / "scenarios.jsonl")
        if not scenarios:
            raise FileNotFoundError("Run make-scenarios first.")
        out_rows: List[Dict[str, Any]] = []
        system = self._ladder_system_prompt(trait)
        for scenario in scenarios:
            schema_hint = json.dumps(
                {
                    "scenario_id": scenario["scenario_id"],
                    "trait": trait,
                    "invariant_content": scenario.get("proposition_or_request", ""),
                    "ladder": {"low": "...", "mid": "...", "high": "..."},
                },
                ensure_ascii=False,
            )
            obj = self.generator.call_json(system, self._ladder_user_prompt(trait, scenario), schema_hint)
            obj["scenario"] = scenario
            out_rows.append(obj)
        jsonl_write(self.trait_dir(trait) / "ladders.jsonl", out_rows)
        print(f"Wrote {len(out_rows)} ladders")

    def _paraphrase_system_prompt(self, trait: str) -> str:
        return (
            "You generate paraphrases for a representation-geometry benchmark. "
            "Preserve meaning exactly while varying lexical realization. "
            "Do not produce near-duplicates. Use different cue families when possible."
        )

    def _paraphrase_user_prompt(self, ladder_obj: Dict[str, Any], per_level: int) -> str:
        return (
            f"Given this canonical 3-level ladder:\n{json.dumps(ladder_obj['ladder'], ensure_ascii=False, indent=2)}\n\n"
            f"Generate {per_level} paraphrases per level.\n"
            "For each paraphrase, provide: level, cue_family, text.\n"
            "Cue families should differ when possible, such as lexical marker, syntactic framing, gratitude framing, evidential framing, indirectness, modal framing.\n"
            "Keep the proposition or requested action unchanged.\n"
            "Return JSON with keys: scenario_id, trait, items."
        )

    def make_paraphrases(self, trait: str) -> None:
        ladders = jsonl_read(self.trait_dir(trait) / "ladders.jsonl")
        if not ladders:
            raise FileNotFoundError("Run make-ladders first.")
        per_level = int(self.config["pipeline"]["paraphrases_per_level"])
        system = self._paraphrase_system_prompt(trait)
        rows: List[Dict[str, Any]] = []
        for ladder in ladders:
            schema_hint = json.dumps(
                {
                    "scenario_id": ladder["scenario_id"],
                    "trait": trait,
                    "items": [
                        {
                            "level": "low",
                            "cue_family": "syntactic indirectness",
                            "text": "...",
                        }
                    ],
                },
                ensure_ascii=False,
            )
            obj = self.generator.call_json(system, self._paraphrase_user_prompt(ladder, per_level), schema_hint)
            scenario_id = obj["scenario_id"]
            for idx, item in enumerate(obj["items"], start=1):
                row = {
                    "scenario_id": scenario_id,
                    "trait": trait,
                    "level": item["level"],
                    "cue_family": item.get("cue_family", "unspecified"),
                    "text": normalize_text(item["text"]),
                    "canonical": ladder["ladder"][item["level"]],
                    "invariant_content": ladder.get("invariant_content", ""),
                    "paraphrase_id": f"{scenario_id}-{item['level']}-{idx:02d}",
                }
                rows.append(row)
        jsonl_write(self.trait_dir(trait) / "paraphrases.jsonl", rows)
        print(f"Wrote {len(rows)} paraphrase rows")

    def _judge_system_prompt(self) -> str:
        rubric = (self.root / "config" / "construct_rubric.md").read_text(encoding="utf-8")
        return (
            "You are an independent validation judge for a benchmark. "
            "Your job is to reject content drift, wrong ordering, poor fluency, weak cue diversity, and shortcut-heavy bundles.\n\n"
            f"Rubric:\n{rubric}"
        )

    def _judge_user_prompt(self, trait: str, scenario_id: str, bundle: List[Dict[str, Any]]) -> str:
        grouped = {level: [r["text"] for r in bundle if r["level"] == level] for level in self.levels}
        return (
            f"Validate this bundle for trait '{trait}' and scenario '{scenario_id}'.\n"
            f"Texts by level:\n{json.dumps(grouped, ensure_ascii=False, indent=2)}\n\n"
            "Check: proposition/request preservation, correct low<mid<high ordering, naturalness, paraphrase diversity, and obvious lexical shortcut risk.\n"
            "Return JSON with keys: scenario_id, accepted, overall_score, checks, notes, item_scores.\n"
            "'checks' must include content_preservation, monotonic_order, naturalness, cue_diversity, shortcut_risk.\n"
            "'item_scores' must be a list with paraphrase_id and score in [0,1]."
        )

    def judge_bundles(self, trait: str) -> None:
        rows = jsonl_read(self.trait_dir(trait) / "paraphrases.jsonl")
        if not rows:
            raise FileNotFoundError("Run make-paraphrases first.")
        system = self._judge_system_prompt()
        min_score = float(self.config["pipeline"]["min_acceptance_score"])
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["scenario_id"], []).append(row)

        judged: List[Dict[str, Any]] = []
        accepted_rows: List[Dict[str, Any]] = []
        for scenario_id, bundle in grouped.items():
            schema_hint = json.dumps(
                {
                    "scenario_id": scenario_id,
                    "accepted": True,
                    "overall_score": 0.92,
                    "checks": {
                        "content_preservation": True,
                        "monotonic_order": True,
                        "naturalness": True,
                        "cue_diversity": True,
                        "shortcut_risk": "low",
                    },
                    "notes": "brief explanation",
                    "item_scores": [
                        {"paraphrase_id": bundle[0]["paraphrase_id"], "score": 0.91}
                    ],
                },
                ensure_ascii=False,
            )
            primary = self.judge.call_json(system, self._judge_user_prompt(trait, scenario_id, bundle), schema_hint)
            accepted = bool(primary.get("accepted", False)) and float(primary.get("overall_score", 0.0)) >= min_score
            if not accepted and float(primary.get("overall_score", 0.0)) >= max(0.0, min_score - 0.15):
                tie = self.tie_breaker.call_json(system, self._judge_user_prompt(trait, scenario_id, bundle), schema_hint)
                accepted = bool(tie.get("accepted", False)) and float(tie.get("overall_score", 0.0)) >= min_score
                primary["tie_breaker"] = tie
            item_scores = {x["paraphrase_id"]: x["score"] for x in primary.get("item_scores", []) if "paraphrase_id" in x}
            for row in bundle:
                rec = dict(row)
                rec["judge"] = primary
                rec["accepted"] = accepted and item_scores.get(row["paraphrase_id"], 0.0) >= min_score
                rec["item_score"] = item_scores.get(row["paraphrase_id"], None)
                judged.append(rec)
                if rec["accepted"]:
                    accepted_rows.append(rec)
        jsonl_write(self.trait_dir(trait) / "judged.jsonl", judged)
        jsonl_write(self.trait_dir(trait) / "accepted.jsonl", accepted_rows)
        print(f"Judged {len(judged)} rows; accepted {len(accepted_rows)} rows")

    def audit(self, trait: str) -> None:
        rows = jsonl_read(self.trait_dir(trait) / "accepted.jsonl")
        if not rows:
            raise FileNotFoundError("Run judge first.")
        max_dup = float(self.config["pipeline"]["max_duplicate_jaccard"])
        warning_acc = float(self.config["pipeline"]["lexical_baseline_warning_accuracy"])

        duplicates: List[Dict[str, Any]] = []
        for a, b in itertools.combinations(rows, 2):
            score = jaccard(a["text"], b["text"])
            if score >= max_dup:
                duplicates.append(
                    {
                        "a": a["paraphrase_id"],
                        "b": b["paraphrase_id"],
                        "jaccard": round(score, 4),
                    }
                )

        df = pd.DataFrame(rows)
        lexical_accuracy = None
        top_ngrams: List[Dict[str, Any]] = []
        if len(df) >= 12 and df["level"].nunique() >= 2:
            vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
            X = vectorizer.fit_transform(df["text"])
            y = df["level"]
            n_splits = min(5, y.value_counts().min())
            if n_splits >= 2:
                clf = LogisticRegression(max_iter=2000, multi_class="auto")
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
                preds = cross_val_predict(clf, X, y, cv=cv)
                lexical_accuracy = float(accuracy_score(y, preds))
                clf.fit(X, y)
                vocab = vectorizer.get_feature_names_out()
                coef = clf.coef_
                for idx, label in enumerate(clf.classes_):
                    top_ids = coef[idx].argsort()[-10:][::-1]
                    top_ngrams.append(
                        {
                            "level": label,
                            "top_positive_ngrams": [vocab[i] for i in top_ids],
                        }
                    )

        scenario_balance = (
            df.groupby(["scenario_id", "level"]).size().reset_index(name="count").to_dict(orient="records")
        )

        report = {
            "trait": trait,
            "num_rows": len(rows),
            "duplicate_pairs_over_threshold": duplicates,
            "num_duplicate_pairs": len(duplicates),
            "lexical_baseline_accuracy": lexical_accuracy,
            "lexical_baseline_warning": lexical_accuracy is not None and lexical_accuracy >= warning_acc,
            "top_ngrams_by_level": top_ngrams,
            "scenario_level_balance": scenario_balance,
        }
        with (self.trait_dir(trait) / "audit_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Wrote audit report to {self.trait_dir(trait) / 'audit_report.json'}")

    def export_human_validation(self, trait: str) -> None:
        rows = jsonl_read(self.trait_dir(trait) / "accepted.jsonl")
        if not rows:
            raise FileNotFoundError("Run judge first.")
        frac = float(self.config["pipeline"]["human_validation_fraction"])
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["scenario_id"], []).append(row)
        scenario_ids = sorted(grouped.keys())
        sample_n = max(1, math.ceil(len(scenario_ids) * frac))
        random.Random(7).shuffle(scenario_ids)
        sampled = scenario_ids[:sample_n]

        human_rows: List[Dict[str, Any]] = []
        for sid in sampled:
            bundle = grouped[sid]
            for row in bundle:
                human_rows.append(
                    {
                        "scenario_id": sid,
                        "paraphrase_id": row["paraphrase_id"],
                        "trait": row["trait"],
                        "level": row["level"],
                        "text": row["text"],
                        "check_order_within_bundle": "",
                        "check_content_preserved": "",
                        "check_natural": "",
                        "comments": "",
                    }
                )
        pd.DataFrame(human_rows).to_csv(self.trait_dir(trait) / "human_validation.csv", index=False)

        if self.config["pipeline"].get("use_bws_exports", True):
            bws_rows: List[Dict[str, Any]] = []
            items = [r for r in rows if r["scenario_id"] in sampled]
            rng = random.Random(11)
            for i in range(min(100, max(0, len(items) // 2))):
                quad = rng.sample(items, 4)
                bws_rows.append(
                    {
                        "task_id": f"{trait}-bws-{i:03d}",
                        "trait": trait,
                        "item_a": quad[0]["text"],
                        "item_b": quad[1]["text"],
                        "item_c": quad[2]["text"],
                        "item_d": quad[3]["text"],
                        "most_target_like": "",
                        "least_target_like": "",
                    }
                )
            pd.DataFrame(bws_rows).to_csv(self.trait_dir(trait) / "bws_tasks.csv", index=False)
        print(f"Exported human validation files for {trait}")

    def export_prompts(self, out_path: Optional[Path] = None) -> None:
        """Export accepted paraphrases from all traits to data/prompts.json.

        Produces the {prompt, trait, intensity} format consumed by src/data.py.
        """
        if out_path is None:
            out_path = self.root / "data" / "prompts.json"
        rows = []
        for trait in self.config["traits"]:
            for rec in jsonl_read(self.trait_dir(trait) / "accepted.jsonl"):
                rows.append({"prompt": rec["text"], "trait": rec["trait"], "intensity": rec["level"]})
        ensure_dir(out_path.parent)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(rows)} prompts to {out_path}")

    def make_all(self, trait: str) -> None:
        self.make_scenarios(trait)
        self.make_ladders(trait)
        self.make_paraphrases(trait)
        self.judge_bundles(trait)
        self.audit(trait)
        self.export_human_validation(trait)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ordinal dataset creation pipeline")
    parser.add_argument("command", choices=[
        "make-scenarios",
        "make-ladders",
        "make-paraphrases",
        "judge",
        "audit",
        "export-human-validation",
        "export-prompts",
        "make-all",
    ])
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--trait", choices=TRAITS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = Pipeline(Path(args.config))
    if args.command == "make-scenarios":
        pipe.make_scenarios(args.trait)
    elif args.command == "make-ladders":
        pipe.make_ladders(args.trait)
    elif args.command == "make-paraphrases":
        pipe.make_paraphrases(args.trait)
    elif args.command == "judge":
        pipe.judge_bundles(args.trait)
    elif args.command == "audit":
        pipe.audit(args.trait)
    elif args.command == "export-human-validation":
        pipe.export_human_validation(args.trait)
    elif args.command == "export-prompts":
        pipe.export_prompts()
    elif args.command == "make-all":
        pipe.make_all(args.trait)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
