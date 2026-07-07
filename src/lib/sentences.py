"""Trait sentence-dataset generation (English-only).

One module: shared types/IO (Sample, load_accepted, …), scenario schema, LLM client,
and the Pipeline. Trait-specific content (rubric guide, intents, invariant field, …)
comes from the registry in ``lib.traits``; the pipeline itself is trait-agnostic.
`generate_sentences(out_dir, trait=..., ...)` runs the stages and writes into out_dir;
the caller owns where that is. Stages run in order:

    scenarios -> base_sentences -> paraphrases -> judge_and_filter -> export
"""

from __future__ import annotations

import json
import math
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

try:
    import litellm

    litellm.suppress_debug_info = True
    from litellm import completion
except Exception:  # pragma: no cover
    completion = None

from json_repair import repair_json

from jsonschema import Draft202012Validator

from lib.traits import DEFAULT_TRAIT, LEVELS, TRAITS

# --- shared types & IO helpers (this module is `lib.sentences`)


class Sample(NamedTuple):
    prompt: str
    trait: str
    intensity: str
    scenario_id: str = ""
    paraphrase_id: str = ""


def load_accepted(path: str | Path) -> list[Sample]:
    """Load samples from a sentences_filtered.jsonl dataset (text/trait/level/scenario_id/paraphrase_id)."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                samples.append(
                    Sample(
                        prompt=r["text"],
                        trait=r["trait"],
                        intensity=r["level"],
                        scenario_id=r["scenario_id"],
                        paraphrase_id=r["paraphrase_id"],
                    )
                )
    return samples


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def jsonl_write(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


# --- scenario schema

RUBRIC_VERSION = "v3"
DATASET_VERSION = "v3"

# Fields every scenario must carry regardless of trait; each trait's spec adds
# its `required_fields` (e.g. politeness's `intent_target`, the fixed thing the
# act is about).
_SHARED_REQUIRED = [
    "scenario_id",
    "trait",
    "intent",
    "domain",
    "audience_relation",
    "communicative_goal",
    "target_word_count",
]


def scenario_json_schema(trait: str) -> Dict[str, Any]:
    properties: Dict[str, Any] = {
        "scenario_id": {"type": "string", "minLength": 3},
        "trait": {"const": trait},
        "rubric_version": {"type": "string"},
        "dataset_version": {"type": "string"},
        "intent": {"type": "string", "minLength": 2},
        "domain": {"type": "string", "minLength": 2},
        "audience_relation": {"type": "string"},
        "communicative_goal": {"type": "string", "minLength": 5},
        "target_word_count": {"type": "integer", "minimum": 4, "maximum": 60},
    }
    for field_name in TRAITS[trait]["required_fields"]:
        properties[field_name] = {"type": "string", "minLength": 3}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": True,
        "properties": properties,
        "required": _SHARED_REQUIRED + TRAITS[trait]["required_fields"],
    }


def make_validator(trait: str) -> Draft202012Validator:
    return Draft202012Validator(scenario_json_schema(trait))


# --- llm client


@dataclass
class ModelSpec:
    model: str
    family: str
    temperature: float = 0.0
    max_output_tokens: int = 1200
    litellm_kwargs: Dict[str, Any] = field(default_factory=dict)


# Reasoning models (e.g. gpt-5-nano) sometimes leave `content` empty and put the
# actual answer in one of these alternate message fields. Checked in order.
_REASONING_CONTENT_FIELDS = ("reasoning_content", "reasoning", "text")


class LLMClient:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        if completion is None:
            raise RuntimeError(
                "litellm is not installed. Install requirements before using the pipeline."
            )

    def call_json(
        self, system: str, user: str, schema_hint: Optional[str] = None
    ) -> Any:
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
            **self.spec.litellm_kwargs,
        )
        msg = resp["choices"][0]["message"]
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if not content:
            # Reasoning models can leave `content` empty; the answer is in an alt field.
            for alt in _REASONING_CONTENT_FIELDS:
                v = msg.get(alt) if isinstance(msg, dict) else getattr(msg, alt, None)
                if isinstance(v, str) and v.strip():
                    content = v
                    break
        text = (content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        # json_repair closes whatever the model left unterminated (e.g. a batch cut off at
        # max_tokens) and returns the parsed value. Callers check the shape they expect.
        return repair_json(text, return_objects=True)


# --- pipeline

_RANDOM_SEED = (
    17  # fixed so shuffles (intent queue, blind scoring) are reproducible
)


class Pipeline:
    def __init__(
        self,
        output_dir,
        generator: ModelSpec,
        judge: ModelSpec,
        *,
        n_scenarios: int = 300,
        paraphrases_per_level: int = 3,
        min_acceptance_score: float = 0.70,
        intensity_min_gap: float = 0.10,
        max_length_ratio: float = 1.15,
        scenario_batch_size: int = 5,
        max_duplicate_jaccard: float = 0.85,
        max_workers: int = 3,
        trait: str = DEFAULT_TRAIT,
    ):
        self.output_root = Path(output_dir)
        ensure_dir(self.output_root)
        self.generator = LLMClient(generator)
        self.judge = LLMClient(judge)
        self._warn_if_same_family(generator.family, judge.family)

        self.n_scenarios = n_scenarios
        self.paraphrases_per_level = paraphrases_per_level
        self.min_acceptance_score = min_acceptance_score
        self.intensity_min_gap = intensity_min_gap
        self.max_length_ratio = max_length_ratio
        self.scenario_batch_size = scenario_batch_size
        self.max_duplicate_jaccard = max_duplicate_jaccard
        self.max_workers = max_workers
        self.trait = trait
        self._trait_info(trait)  # fail fast on an unknown trait

        self.levels: List[str] = list(LEVELS)
        self._validator = make_validator(trait)
        self.dataset_version = DATASET_VERSION
        self.rubric_version = RUBRIC_VERSION
        self.random_seed = _RANDOM_SEED

    # --- helpers

    @staticmethod
    def _warn_if_same_family(generator_family: str, judge_family: str) -> None:
        if generator_family == judge_family:
            print(
                f"WARNING: generator and judge are both '{generator_family}' family — using the "
                "same model family to generate and judge increases leakage risk."
            )

    def _run_items(self, items, work):
        # Run work(item) over items in a thread pool; a failing item comes back as None
        # (callers filter those out). Errors are printed; counts go into metadata.json.
        def _safe(item):
            try:
                return work(item)
            except Exception as exc:
                print(f"  ! item failed: {type(exc).__name__}: {str(exc)[:160]}")
                return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(_safe, items))

    def _trait_info(self, trait: str) -> Dict[str, Any]:
        try:
            return TRAITS[trait]
        except KeyError as e:
            raise KeyError(
                f"Trait '{trait}' is not defined in the lib.traits registry."
            ) from e

    def _intents_for(self, trait: str) -> List[Dict[str, Any]]:
        acts = self._trait_info(trait).get("intents", [])
        if not acts:
            raise ValueError(f"No intents configured for trait '{trait}'")
        return acts

    def _build_intent_queue(self, trait: str, n: int) -> List[str]:
        # Intent ids split as evenly as possible across n scenarios, then shuffled
        # with a fixed seed so a given trait always yields the same order.
        ids = [a["id"] for a in self._intents_for(trait)]
        per = n // len(ids)
        rem = n - per * len(ids)
        queue: List[str] = []
        for a in ids:
            queue.extend([a] * per)
        for i in range(rem):
            queue.append(ids[i % len(ids)])
        rng = random.Random(self.random_seed + sum(ord(c) for c in trait))
        rng.shuffle(queue)
        return queue

    def _content_invariant(self, scenario: Dict[str, Any]) -> str:
        v = scenario.get(self._trait_info(self.trait)["invariant_field"])
        return v.strip() if isinstance(v, str) else ""

    def _validate_scenario(self, scenario: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = [e.message for e in self._validator.iter_errors(scenario)]
        return (not errors, errors)

    def _seed_example(self, trait: str) -> Dict[str, Any]:
        # Shown to the generator as a worked example before any real scenarios exist.
        example = dict(self._trait_info(trait)["seed_example"])
        example["rubric_version"] = self.rubric_version
        example["dataset_version"] = self.dataset_version
        return example

    # --- stage 1: scenarios

    def _scenario_system_prompt(self, trait: str) -> str:
        return (
            "You are building a research dataset for representation geometry. "
            "Create base scenarios, not labels or explanations. "
            "Scenarios must keep future ordinal rewrites content-controlled.\n\n"
            "CRITICAL: You MUST return only valid JSON. Do not include any text before or after the JSON. "
            "Do not add explanations, preambles, or comments. The entire response must be parseable as JSON."
        )

    _SHARED_FIELDS = [
        "scenario_id",
        "trait",
        "rubric_version",
        "dataset_version",
        "intent",
        "domain",
        "audience_relation",
        "communicative_goal",
        "target_word_count",
    ]

    def _scenario_user_prompt_batched(
        self,
        trait: str,
        intent_ids: List[str],
        previous_scenarios: List[Dict[str, Any]],
    ) -> str:
        # Each scenario's intent is assigned up front so the batch stays balanced.
        trait_info = self._trait_info(trait)
        shared_fields = list(self._SHARED_FIELDS)
        levels_str = ", ".join(self.levels)
        n = len(intent_ids)

        intents_by_id = {a["id"]: a for a in self._intents_for(trait)}
        assignment_lines = "\n".join(
            f"  scenario {i + 1}: intent = {sid}"
            for i, sid in enumerate(intent_ids)
        )
        unique_intents = sorted(set(intent_ids))
        intent_blocks = []
        for sid in unique_intents:
            a = intents_by_id[sid]
            block = f"- {sid}: {a.get('description', '')}"
            example_goal = a.get("example_goal")
            if example_goal:
                block += f"\n  example goal: {example_goal}"
            extra = a.get("extra_constraints") or []
            if extra:
                block += "\n  per-intent constraints:\n" + "\n".join(
                    f"    - {c}" for c in extra
                )
            intent_blocks.append(block)

        prompt = (
            f"Generate exactly {n} base scenarios for the trait '{trait}'.\n"
            f"Trait description: {trait_info.get('description', '')}\n"
            f"Future ordinal rewriting will use levels: {levels_str}.\n\n"
            f"Each scenario MUST use the intent assigned to it below. Do not reassign or merge intents:\n"
            f"{assignment_lines}\n\n"
            f"Intent guide (only the intents you need this batch):\n"
            + "\n".join(intent_blocks)
            + "\n\n"
            f"Required trait-specific fields: {trait_info['required_fields']}.\n"
            f"Trait-level generation constraints:\n"
            + "\n".join(f"  - {c}" for c in trait_info["generation_constraints"])
            + "\n\n"
            f"Shared fields each scenario must fill: {shared_fields}.\n"
            "Hard requirements for every scenario object:\n"
            "  - scenario_id is a stable string (you may use 'auto' and the pipeline will reassign).\n"
            "  - trait must equal the trait above.\n"
            f"  - rubric_version = '{self.rubric_version}', dataset_version = '{self.dataset_version}'.\n"
            "  - target_word_count is an integer between 8 and 40.\n"
            "  - intent MUST equal the value assigned above for that scenario index.\n"
            "  - All natural-language fields must be written in English.\n"
            "  - Vary domains and audience_relations across the batch.\n"
        )
        if previous_scenarios:
            slim = [
                {
                    "scenario_id": s.get("scenario_id"),
                    "intent": s.get("intent"),
                    "domain": s.get("domain"),
                    "communicative_goal": s.get("communicative_goal"),
                }
                for s in previous_scenarios[-3:]
            ]
            prompt += (
                f"\nRECENT SCENARIOS — DO NOT repeat these topics/goals:\n"
                f"{json.dumps(slim, indent=2, ensure_ascii=False)}\n"
            )
        prompt += (
            "\nReturn a JSON array of scenario objects only. No prose, no markdown."
        )
        return prompt

    def make_scenarios(self, trait: str) -> List[Dict[str, Any]]:
        total_n = self.n_scenarios
        batch_size = self.scenario_batch_size
        max_batch_retries = 3
        max_dup_jaccard = self.max_duplicate_jaccard
        system = self._scenario_system_prompt(trait)

        cleaned: List[Dict[str, Any]] = []
        remaining_queue = self._build_intent_queue(trait, total_n)
        invariant_strings: set[str] = set()

        # Stop on: target reached, an absolute batch cap, a fatal API error, or too many
        # batches in a row that add nothing (model failing or every scenario a duplicate).
        max_total_batches = max(8, ((total_n + batch_size - 1) // batch_size) * 4)
        FATAL_ERRORS = (
            "more credits",
            "insufficient_quota",
            "insufficient credits",
            "billing",
            "401",
            "403",
        )
        attempted_batches = batch_index = unproductive_streak = 0

        while len(cleaned) < total_n:
            if attempted_batches >= max_total_batches:
                print(
                    f"  ! batch cap ({max_total_batches}) reached; stopping at {len(cleaned)}/{total_n}."
                )
                break
            if unproductive_streak >= 5:
                print(
                    f"  ! 5 unproductive batches; stopping at {len(cleaned)}/{total_n}."
                )
                break
            batch_index += 1
            if (
                not remaining_queue
            ):  # ran dry because dedup/validation dropped some scenarios
                remaining_queue = self._build_intent_queue(
                    trait, total_n - len(cleaned)
                )
            batch_intents = remaining_queue[:batch_size]
            remaining_queue = remaining_queue[batch_size:]
            batch_n = len(batch_intents)
            attempted_batches += 1

            user = self._scenario_user_prompt_batched(trait, batch_intents, cleaned)
            schema_hint = json.dumps(
                cleaned[-2:] if cleaned else [self._seed_example(trait)],
                ensure_ascii=False,
            )

            scenarios_raw = None
            last_err: Optional[Exception] = None
            for batch_retry in range(max_batch_retries):
                try:
                    scenarios_raw = self.generator.call_json(system, user, schema_hint)
                    break
                except Exception as e:
                    last_err = e
                    print(
                        f"  [batch {batch_index} retry {batch_retry + 1}/{max_batch_retries}] "
                        f"{type(e).__name__}: {str(e)[:160]}"
                    )

            if scenarios_raw is None:
                err_msg = str(last_err) if last_err else ""
                print(
                    f"  ! batch {batch_index} ({batch_intents}) skipped after {max_batch_retries} retries."
                )
                unproductive_streak += 1
                if any(p in err_msg.lower() for p in FATAL_ERRORS):
                    print(
                        f"  ! fatal API error (credits/quota/auth); stopping at {len(cleaned)}/{total_n}.\n    {err_msg[:240]}"
                    )
                    break
                continue

            if not isinstance(scenarios_raw, list):
                if isinstance(scenarios_raw, dict):
                    for key in ("scenarios", "items", "data"):
                        if isinstance(scenarios_raw.get(key), list):
                            scenarios_raw = scenarios_raw[key]
                            break
            if not isinstance(scenarios_raw, list):
                print(f"  ! batch {batch_index} produced non-list output; skipping.")
                unproductive_streak += 1
                continue

            kept_in_batch = 0
            for i, raw_row in enumerate(scenarios_raw):
                if not isinstance(raw_row, dict):
                    continue
                row = dict(raw_row)
                row["trait"] = trait
                row.setdefault("rubric_version", self.rubric_version)
                row.setdefault("dataset_version", self.dataset_version)
                if i < len(batch_intents):
                    row["intent"] = batch_intents[i]
                next_idx = len(cleaned) + 1
                intent_slug = slugify(str(row.get("intent", "intent")))
                row["scenario_id"] = f"{trait}-{intent_slug}-{next_idx:03d}"

                ok, _ = self._validate_scenario(row)
                if not ok:
                    continue
                inv = normalize_text(self._content_invariant(row)).lower()
                if inv:
                    if any(
                        jaccard(inv, prev) >= max_dup_jaccard
                        for prev in invariant_strings
                    ):
                        continue
                    invariant_strings.add(inv)

                cleaned.append(row)
                kept_in_batch += 1
                if len(cleaned) >= total_n:
                    break

            unproductive_streak = 0 if kept_in_batch else unproductive_streak + 1
            print(
                f"  batch {batch_index}: kept {kept_in_batch}/{batch_n} (total {len(cleaned)}/{total_n})"
            )

        jsonl_write(self.output_root / "scenarios.jsonl", cleaned)
        print(f"Wrote {len(cleaned)}/{total_n} scenarios.")
        return cleaned

    # --- stage 2: base_sentences

    def _base_sentences_system_prompt(self, trait: str) -> str:
        return (
            "You create controlled ordinal ladders for NLP research. "
            "Keep content fixed and vary only trait intensity.\n\n"
            f"{trait.capitalize()} guide:\n{self._trait_info(trait)['guide']}\n\n"
            "CRITICAL: You MUST return only valid JSON. Do not include any text before or after the JSON. "
            "Do not add explanations, preambles, or comments. The entire response must be parseable as JSON."
        )

    def _base_sentences_user_prompt(self, trait: str, scenario: Dict[str, Any]) -> str:
        levels_str = ", ".join(self.levels)
        invariant = self._content_invariant(scenario)
        intent = scenario.get("intent", "")
        return (
            f"Create one {len(self.levels)}-level set of base sentences for trait '{trait}', "
            f"intent '{intent}'.\n"
            f"Scenario:\n{json.dumps(scenario, ensure_ascii=False, indent=2)}\n\n"
            "Write every sentence in English; use natural, idiomatic phrasing.\n"
            f"Levels must be {levels_str}.\n"
            f"The invariant content '{invariant}' must remain identical in meaning across levels.\n"
            f"Vary only the trait intensity. Write all {len(self.levels)} sentences at about "
            f"{scenario.get('target_word_count', 18)} words each — equal length across "
            "levels, so sentence length never cues the trait.\n"
            "Return JSON with keys: scenario_id, trait, invariant_content, base_sentences.\n"
            f"'base_sentences' must map each of these levels to a single sentence: {levels_str}."
        )

    def make_base_sentences(
        self, trait: str, scenarios: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        system = self._base_sentences_system_prompt(trait)

        def _one(scenario: Dict[str, Any]) -> Dict[str, Any]:
            schema_hint = json.dumps(
                {
                    "scenario_id": scenario["scenario_id"],
                    "trait": trait,
                    "invariant_content": self._content_invariant(scenario),
                    "base_sentences": {lvl: "..." for lvl in self.levels},
                },
                ensure_ascii=False,
            )
            obj = self.generator.call_json(
                system, self._base_sentences_user_prompt(trait, scenario), schema_hint
            )
            sentences = obj.get("base_sentences") if isinstance(obj, dict) else None
            if not isinstance(sentences, dict) or any(
                lvl not in sentences for lvl in self.levels
            ):
                raise ValueError(
                    f"Base sentences for {scenario.get('scenario_id')} missing levels; "
                    f"got {list(sentences) if isinstance(sentences, dict) else type(sentences).__name__}"
                )
            return {
                "scenario_id": scenario["scenario_id"],
                "trait": trait,
                "invariant_content": obj.get(
                    "invariant_content", self._content_invariant(scenario)
                ),
                "base_sentences": sentences,
                "scenario": scenario,
            }

        out_rows = [r for r in self._run_items(scenarios, _one) if r is not None]
        print(
            f"Built {len(out_rows)} base-sentence sets ({len(scenarios) - len(out_rows)} failed)."
        )
        return out_rows

    # --- stage 3: paraphrases

    def _paraphrase_system_prompt(self, trait: str) -> str:
        return (
            "You generate paraphrases for a representation-geometry benchmark. "
            "Preserve meaning exactly while varying lexical realization. "
            "Do not produce near-duplicates. Use different cue families when possible. "
            "Write fluent, natural, well-formed English. "
            "Critical: all paraphrases must be similar in length across levels. "
            "Do not use sentence length or verbosity as a cue for the trait level.\n\n"
            "CRITICAL: You MUST return only valid JSON. Do not include any text before or after the JSON. "
            "Do not add explanations, preambles, or comments. The entire response must be parseable as JSON."
        )

    def _paraphrase_user_prompt(
        self,
        base_obj: Dict[str, Any],
        used_cue_families: set[str],
        prior_texts_by_level: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        avoid = sorted(used_cue_families - {""})
        sentences = base_obj["base_sentences"]
        n_levels = len(sentences)
        target_len = base_obj.get("scenario", {}).get("target_word_count", 18)
        spec = self._trait_info(self.trait)
        prompt = (
            f"Given these canonical {n_levels}-level base sentences:\n"
            f"{json.dumps(sentences, ensure_ascii=False, indent=2)}\n\n"
            "Write every paraphrase in fluent, natural English.\n"
            "Generate one paraphrase per level.\n"
            "For each paraphrase, provide: level, cue_family, text.\n"
            "DIVERSITY — hard requirement: each paraphrase must be a genuinely distinct "
            "sentence — its own syntactic structure and its own cue family "
            f"({', '.join(spec['cue_families'])}). Do not reuse the structure of any other "
            f"paraphrase for this scenario. {spec['paraphrase_note']}\n"
            "Keep the proposition or requested action unchanged.\n"
            f"LENGTH — hard requirement: write every paraphrase, at every level, at about "
            f"{target_len} words. Negative, neutral and positive paraphrases must all "
            "average the same length; a systematic length difference across levels is a "
            "disqualifying confound. Convey the trait through word choice and framing, "
            "never by adding or cutting words.\n"
            "Return JSON with keys: scenario_id, trait, items."
        )
        if avoid:
            prompt += f"\n\nAlready used in this dataset — vary away from these cue families: {avoid}."
        if prior_texts_by_level and any(prior_texts_by_level.values()):
            shown = {
                lvl: prior_texts_by_level[lvl]
                for lvl in self.levels
                if prior_texts_by_level.get(lvl)
            }
            prompt += (
                "\n\nParaphrases ALREADY generated for this scenario are below. Yours "
                "MUST be genuinely different sentences — a different syntactic structure "
                "and a different cue family from every one of them. Do not re-paraphrase "
                "them:\n" + json.dumps(shown, ensure_ascii=False, indent=2)
            )
        return prompt

    def make_paraphrases(
        self, trait: str, base_rows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        base_rows = sorted(base_rows, key=lambda x: x["scenario_id"])
        per_level = self.paraphrases_per_level
        system = self._paraphrase_system_prompt(trait)

        used_cue_families: set[str] = set()
        lock = threading.Lock()

        def _one(base_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
            scenario_id = base_obj["scenario_id"]
            schema_hint = json.dumps(
                {
                    "scenario_id": scenario_id,
                    "trait": trait,
                    "items": [
                        {
                            "level": "negative",
                            "cue_family": "syntactic indirectness",
                            "text": "...",
                        }
                    ],
                },
                ensure_ascii=False,
            )
            with lock:
                avoid = set(used_cue_families)
            # One paraphrase per level per pass, showing the model what it already wrote
            # for this scenario so it can't just copy a level's earlier paraphrases.
            prior_by_level: Dict[str, List[str]] = {lvl: [] for lvl in self.levels}
            counters: Dict[str, int] = {lvl: 0 for lvl in self.levels}
            batch: List[Dict[str, Any]] = []
            for _pass in range(per_level):
                try:
                    obj = self.generator.call_json(
                        system,
                        self._paraphrase_user_prompt(
                            base_obj, avoid, prior_by_level
                        ),
                        schema_hint,
                    )
                except Exception as exc:
                    print(
                        f"  ! paraphrase pass failed for {scenario_id}: {type(exc).__name__}: {str(exc)[:100]}"
                    )
                    continue
                items = obj.get("items") if isinstance(obj, dict) else None
                if not isinstance(items, list):
                    continue
                seen: set[str] = set()
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    lvl = item.get("level")
                    txt = item.get("text")
                    if lvl not in self.levels or lvl in seen:
                        continue
                    if not isinstance(txt, str) or not txt.strip():
                        continue
                    seen.add(lvl)
                    txt = normalize_text(txt)
                    counters[lvl] += 1
                    batch.append(
                        {
                            "scenario_id": scenario_id,
                            "trait": trait,
                            "level": lvl,
                            "cue_family": item.get("cue_family", "unspecified"),
                            "text": txt,
                            "base_sentence": base_obj["base_sentences"].get(lvl, ""),
                            "invariant_content": base_obj.get("invariant_content", ""),
                            "paraphrase_id": f"{scenario_id}-{lvl}-{counters[lvl]:02d}",
                        }
                    )
                    prior_by_level[lvl].append(txt)
            if not batch:
                raise ValueError(
                    f"paraphrase generation produced no valid items for {scenario_id}"
                )
            with lock:
                for r in batch:
                    used_cue_families.add(r["cue_family"])
            return batch

        results = self._run_items(base_rows, _one)
        rows = [row for b in results if b for row in b]
        skipped = sum(1 for r in results if not r)
        print(f"Built {len(rows)} paraphrase rows ({skipped} scenarios failed).")
        return rows

    # --- stage 4: judge_and_filter (three checks, in order)

    # --- criterion 1: context (per-sentence faithfulness to content)

    def _context_system_prompt(self, trait: str) -> str:
        return (
            f"You verify CONTENT PRESERVATION for a controlled {trait} dataset. "
            "Each text expresses a fixed underlying content (the 'invariant content'); "
            f"{trait} and tone vary deliberately and MUST NOT affect your judgement. "
            "For each text, score in [0.0, 1.0] how faithfully it preserves the invariant "
            "content: 1.0 = same content, intent, and polarity; lower = content drift, "
            "added/dropped/flipped meaning, or a different request/answer. Do NOT reward or "
            "penalise how polite, rude, or fluent a text is — only content fidelity.\n\n"
            "CRITICAL: Return ONLY valid JSON. No prose, no markdown."
        )

    def _context_user_prompt(
        self,
        invariant: str,
        items: List[Tuple[str, str]],
    ) -> str:
        items_json = json.dumps(
            [{"paraphrase_id": pid, "text": t} for pid, t in items],
            ensure_ascii=False,
            indent=2,
        )
        return (
            f"Invariant content every text must preserve:\n{invariant}\n\n"
            f"Texts:\n{items_json}\n\n"
            'Return JSON with key "scores": a list of {"paraphrase_id": "<id>", "score": <float 0..1>}, '
            "one entry per input id. No other keys."
        )

    def _context_scores(
        self,
        trait: str,
        bundle: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        # paraphrase_id -> faithfulness score in [0,1]; 0.0 if the judge omits an id.
        if not bundle:
            return {}
        invariant = bundle[0].get("invariant_content", "")
        items = [(r["paraphrase_id"], r["text"]) for r in bundle]
        system = self._context_system_prompt(trait)
        user = self._context_user_prompt(invariant, items)
        schema_hint = json.dumps(
            {"scores": [{"paraphrase_id": "<id>", "score": "<float 0..1>"}]},
            ensure_ascii=False,
        )
        out: Dict[str, float] = {r["paraphrase_id"]: 0.0 for r in bundle}
        try:
            obj = self.judge.call_json(system, user, schema_hint)
        except Exception as exc:
            print(
                f"  context judge failed for {bundle[0].get('scenario_id', '?')}: {exc}"
            )
            return out
        scores_raw = (
            obj.get("scores")
            if isinstance(obj, dict)
            else (obj if isinstance(obj, list) else None)
        )
        if not isinstance(scores_raw, list):
            return out
        for s in scores_raw:
            if not isinstance(s, dict):
                continue
            pid, val = s.get("paraphrase_id"), s.get("score")
            if pid in out and isinstance(val, (int, float)):
                out[pid] = float(val)
        return out

    # --- criterion 2: trait_intensity (blind; scores must increase across levels)

    def _intensity_system_prompt(self, trait: str) -> str:
        trait_info = self._trait_info(trait)
        return (
            "You are a calibrated rater of linguistic trait intensity. "
            "You will receive a list of short texts identified only by opaque codes. "
            f"For each text, rate the intensity of the trait '{trait}' on a continuous "
            "scale in [0.0, 1.0], where 0.0 is minimum intensity and 1.0 is maximum. "
            "Use the FULL range. Do not anchor to 0.5. You will NOT be told which texts "
            "belong together or which level they represent. Rate each on its own merits.\n\n"
            f"Trait definition: {trait_info.get('description', '')}\n\n"
            "CRITICAL: Return ONLY valid JSON. No prose, no markdown."
        )

    def _intensity_user_prompt(
        self, trait: str, anon_items: List[Tuple[str, str]]
    ) -> str:
        items_json = json.dumps(
            [{"code": c, "text": t} for c, t in anon_items],
            ensure_ascii=False,
            indent=2,
        )
        return (
            f"Rate the intensity of trait '{trait}' for each text below.\n\n"
            "Use the full [0.0, 1.0] range. Do NOT mode-collapse to 0.5 or to a narrow "
            "band. Texts with markedly different intensity must receive markedly different "
            f"scores.\n\nTexts:\n{items_json}\n\n"
            'Return JSON with key "scores": a list of objects '
            '{"code": "<code>", "intensity": <float in [0,1]>}. '
            "One entry per input code, in any order. Do not include any other keys."
        )

    def _score_intensity_blind(
        self, trait: str, bundle: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        # Rate each paraphrase without showing its level. paraphrase_id -> [0,1], NaN if missing.
        if not bundle:
            return {}
        rng = random.Random(self.random_seed + sum(ord(c) for c in trait) + len(bundle))
        order = list(range(len(bundle)))
        rng.shuffle(order)
        codes = [f"q{i:03d}" for i in range(len(bundle))]
        code_to_pid: Dict[str, str] = {}
        anon_items: List[Tuple[str, str]] = []
        for code, idx in zip(codes, order):
            row = bundle[idx]
            code_to_pid[code] = row["paraphrase_id"]
            anon_items.append((code, row["text"]))
        schema_hint = json.dumps(
            {"scores": [{"code": "<code>", "intensity": "<float in [0,1]>"}]},
            ensure_ascii=False,
        )
        try:
            obj = self.judge.call_json(
                self._intensity_system_prompt(trait),
                self._intensity_user_prompt(trait, anon_items),
                schema_hint,
            )
        except Exception as exc:
            print(
                f"  intensity-scorer failed for {bundle[0].get('scenario_id', '?')}: {exc}"
            )
            return {row["paraphrase_id"]: float("nan") for row in bundle}
        scores_raw = obj.get("scores") if isinstance(obj, dict) else None
        out: Dict[str, float] = {row["paraphrase_id"]: float("nan") for row in bundle}
        if not isinstance(scores_raw, list):
            return out
        for s in scores_raw:
            if not isinstance(s, dict):
                continue
            code, val = s.get("code"), s.get("intensity")
            if (
                code in code_to_pid
                and isinstance(val, (int, float))
                and 0.0 <= float(val) <= 1.0
            ):
                out[code_to_pid[code]] = float(val)
        return out

    def _check_intensity_order(
        self,
        bundle: List[Dict[str, Any]],
        intensity_scores: Dict[str, float],
        min_gap: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        # Pass when each level's mean blind score is higher than the previous one (in
        # self.levels order) by at least min_gap. Needs at least one score per level.
        by_level: Dict[str, List[float]] = {lvl: [] for lvl in self.levels}
        for row in bundle:
            s = intensity_scores.get(row["paraphrase_id"], float("nan"))
            if isinstance(s, float) and not math.isnan(s):
                by_level.setdefault(row["level"], []).append(s)
        means: Dict[str, Optional[float]] = {
            lvl: (sum(v) / len(v) if v else None)
            for lvl, v in ((l, by_level.get(l, [])) for l in self.levels)
        }
        diag = {
            "intensity_means_by_level": {
                k: (round(v, 4) if v is not None else None) for k, v in means.items()
            },
            "intensity_n_by_level": {k: len(by_level.get(k, [])) for k in self.levels},
            "intensity_min_gap_required": min_gap,
        }
        if any(v is None for v in means.values()):
            diag["intensity_monotonic"] = False
            diag["intensity_failure_reason"] = "missing scores for at least one level"
            return False, diag
        ordered = [means[lvl] for lvl in self.levels]
        for a, b in zip(ordered, ordered[1:]):
            if b - a < min_gap:
                diag["intensity_monotonic"] = False
                diag["intensity_failure_reason"] = (
                    f"adjacent gap {b - a:.3f} < required {min_gap:.3f}"
                )
                return False, diag
        diag["intensity_monotonic"] = True
        return True, diag

    # --- criterion 3: length balance

    def _check_length_balance(
        self, bundle: List[Dict[str, Any]], max_ratio: float
    ) -> Tuple[bool, Dict[str, float]]:
        # Pass when the longest level's mean word count is at most max_ratio x the shortest.
        by_level: Dict[str, List[int]] = {}
        for r in bundle:
            by_level.setdefault(r["level"], []).append(len(r["text"].split()))
        means = {lvl: (sum(ws) / len(ws)) for lvl, ws in by_level.items() if ws}
        diag = {k: round(v, 2) for k, v in means.items()}
        mw = list(means.values())
        if len(mw) >= 2 and min(mw) > 0 and max(mw) > min(mw) * max_ratio:
            return False, diag
        return True, diag

    # --- run the three checks, in order

    def _judge_and_filter_bundle(
        self,
        trait: str,
        bundle: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        # Tag every paraphrase with its scores, `passed`, and the `failed_check` that dropped it.
        min_score = self.min_acceptance_score
        min_gap = self.intensity_min_gap
        max_ratio = self.max_length_ratio

        failed_check: Dict[str, str] = {}

        # criterion 1: context (per-sentence faithfulness)
        ctx_scores = self._context_scores(trait, bundle)
        for r in bundle:
            if ctx_scores.get(r["paraphrase_id"], 0.0) < min_score:
                failed_check[r["paraphrase_id"]] = "context"
        survivors = [r for r in bundle if r["paraphrase_id"] not in failed_check]

        # criterion 2: trait_intensity (blind, per bundle)
        intensity_scores: Dict[str, float] = {}
        mono_diag: Dict[str, Any] = {}
        if survivors:
            intensity_scores = self._score_intensity_blind(trait, survivors)
            ok, mono_diag = self._check_intensity_order(
                survivors, intensity_scores, min_gap
            )
            if not ok:
                for r in survivors:
                    failed_check[r["paraphrase_id"]] = "trait_intensity"
                survivors = []

        # criterion 3: length (per bundle, over survivors)
        length_means: Dict[str, float] = {}
        if survivors:
            ok, length_means = self._check_length_balance(survivors, max_ratio)
            if not ok:
                for r in survivors:
                    failed_check[r["paraphrase_id"]] = "length"
                survivors = []

        judged: List[Dict[str, Any]] = []
        for r in bundle:
            pid = r["paraphrase_id"]
            rec = dict(r)
            rec["context_score"] = ctx_scores.get(pid)
            rec["intensity_score_blind"] = intensity_scores.get(pid)
            rec["intensity_gate"] = mono_diag or None
            rec["length_means"] = length_means or None
            rec["failed_check"] = failed_check.get(pid)
            rec["passed"] = pid not in failed_check
            judged.append(rec)
        return judged

    def judge_and_filter(
        self,
        trait: str,
        paraphrases: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        # Judge each scenario's paraphrases and tag every row with `passed` + `failed_check`.
        # Returns all rows (the unfiltered set); the caller keeps the passing ones.
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in paraphrases:
            grouped.setdefault(row["scenario_id"], []).append(row)
        bundles = sorted(grouped.items())

        def _one(item: Tuple[str, List[Dict[str, Any]]]):
            sid, bundle = item
            return self._judge_and_filter_bundle(trait, bundle)

        results = [r for r in self._run_items(bundles, _one) if r is not None]
        judged = [row for bundle_rows in results for row in bundle_rows]
        n_passed = sum(1 for r in judged if r["passed"])
        print(
            f"Judged {len(judged)} rows; {n_passed} passed, {len(judged) - n_passed} dropped."
        )
        return judged


# --- entry point

# Rows kept in sentences_filtered.jsonl (the dataset); diagnostics (scores, passed, failed_check) stay in sentences_unfiltered.
_DATASET_FIELDS = (
    "scenario_id",
    "trait",
    "level",
    "paraphrase_id",
    "cue_family",
    "text",
    "base_sentence",
    "invariant_content",
)


def generate_sentences(
    out_dir,
    *,
    trait: str = DEFAULT_TRAIT,
    n_scenarios: int = 300,
    paraphrases_per_level: int = 3,
    min_acceptance_score: float = 0.70,
    intensity_min_gap: float = 0.10,
    max_length_ratio: float = 1.15,
    max_workers: int = 3,
    models: Dict[str, Any],
) -> Path:
    """Generate the dataset into out_dir (created if needed) and return it.

    ``models`` must be a dict with ``"generator"`` and ``"judge"`` entries, each a
    kwargs dict for ``ModelSpec`` (model id, family, temperature, ...). The caller
    (the entry script) owns the concrete model ids so this library stays
    model-agnostic.

    Writes scenarios.jsonl, sentences_unfiltered.jsonl, sentences_filtered.jsonl, metadata.json into out_dir. The caller
    chooses the directory (e.g. data/<timestamp>/sentences/<trait>). Needs OPENROUTER_API_KEY (a repo-root .env
    is loaded automatically).
    """
    out = Path(out_dir)
    pipe = Pipeline(
        out,
        ModelSpec(**models["generator"]),
        ModelSpec(**models["judge"]),
        n_scenarios=n_scenarios,
        paraphrases_per_level=paraphrases_per_level,
        min_acceptance_score=min_acceptance_score,
        intensity_min_gap=intensity_min_gap,
        max_length_ratio=max_length_ratio,
        max_workers=max_workers,
        trait=trait,
    )
    scenarios = pipe.make_scenarios(trait)
    base = pipe.make_base_sentences(trait, scenarios)
    paraphrases = pipe.make_paraphrases(trait, base)
    judged = pipe.judge_and_filter(trait, paraphrases)
    passed = [r for r in judged if r["passed"]]

    jsonl_write(out / "sentences_unfiltered.jsonl", judged)
    jsonl_write(
        out / "sentences_filtered.jsonl",
        [{k: r[k] for k in _DATASET_FIELDS if k in r} for r in passed],
    )

    by_failed: Dict[str, int] = {}
    for r in judged:
        if r["failed_check"]:
            by_failed[r["failed_check"]] = by_failed.get(r["failed_check"], 0) + 1
    by_intent: Dict[str, int] = {}
    for s in scenarios:
        by_intent[s.get("intent", "?")] = by_intent.get(s.get("intent", "?"), 0) + 1
    manifest = {
        "timestamp": out.parent.parent.name,  # out is data/<ts>/sentences/<trait>
        "trait": trait,
        "params": {
            "n_scenarios": n_scenarios,
            "paraphrases_per_level": paraphrases_per_level,
            "min_acceptance_score": min_acceptance_score,
            "intensity_min_gap": intensity_min_gap,
            "max_length_ratio": max_length_ratio,
        },
        "models": {
            "generator": models["generator"]["model"],
            "judge": models["judge"]["model"],
        },
        "counts": {
            "scenarios": len(scenarios),
            "paraphrases": len(judged),
            "passed": len(passed),
            "failed": len(judged) - len(passed),
            "by_failed_check": by_failed,
            "by_intent": by_intent,
        },
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Done. {len(passed)} accepted paraphrases → {out}")
    return out
