from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml
from jsonschema import Draft202012Validator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

try:
    import litellm
    litellm.suppress_debug_info = True
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

    def _iter_balanced_json(self, raw: str) -> Iterable[str]:
        """Yield every top-level balanced JSON value (object or array) in raw."""
        i = 0
        n = len(raw)
        while i < n:
            ch = raw[i]
            if ch != "{" and ch != "[":
                i += 1
                continue
            opener = ch
            closer = "}" if ch == "{" else "]"
            depth = 1
            j = i + 1
            in_string = False
            escaped = False
            while j < n and depth > 0:
                cj = raw[j]
                if in_string:
                    if escaped:
                        escaped = False
                    elif cj == "\\":
                        escaped = True
                    elif cj == '"':
                        in_string = False
                else:
                    if cj == '"':
                        in_string = True
                    elif cj == opener:
                        depth += 1
                    elif cj == closer:
                        depth -= 1
                j += 1
            if depth == 0:
                yield raw[i:j]
                i = j
            else:
                return

    def _extract_balanced_json(self, raw: str) -> str:
        """Return the longest balanced top-level JSON value found in raw."""
        candidates = list(self._iter_balanced_json(raw))
        if not candidates:
            raise json.JSONDecodeError("Could not find a complete JSON value", raw, 0)
        candidates.sort(key=len, reverse=True)
        return candidates[0]

    def _recover_partial_array(self, raw: str) -> Optional[List[Any]]:
        """If raw looks like a truncated JSON array, salvage the complete object elements.

        Models sometimes hit max_tokens mid-array; the outer ']' never arrives.
        We scan for balanced top-level objects/arrays and attempt to JSON-parse each.
        """
        s = raw.lstrip()
        if not s.startswith("["):
            return None
        # Scan inside the outer '[' for balanced inner values.
        inner = s[1:]
        items: List[Any] = []
        for chunk in self._iter_balanced_json(inner):
            try:
                items.append(json.loads(chunk))
            except json.JSONDecodeError:
                continue
        return items if items else None

    def _recover_partial_object(self, raw: str) -> Optional[Dict[str, Any]]:
        """Salvage a top-level object whose inner array (e.g. 'items') was truncated.

        Returns a dict with as many top-level scalar fields as parseable, plus an 'items'
        list of every complete object found inside the array.
        """
        s = raw.lstrip()
        if not s.startswith("{"):
            return None
        # Find the array key (default 'items', also try 'scenarios', 'data').
        array_key = None
        array_start = -1
        for key in ("items", "scenarios", "data", "paraphrases"):
            m = re.search(r'"' + re.escape(key) + r'"\s*:\s*\[', s)
            if m:
                array_key = key
                array_start = m.end()
                break
        if array_key is None:
            return None
        salvaged_items: List[Any] = []
        for chunk in self._iter_balanced_json(s[array_start:]):
            try:
                salvaged_items.append(json.loads(chunk))
            except json.JSONDecodeError:
                continue
        # Try to extract simple top-level string fields before the array.
        prefix = s[1:s.index('"' + array_key + '"')]
        meta: Dict[str, Any] = {}
        for k, v in re.findall(r'"([A-Za-z_][\w\-]*)"\s*:\s*"((?:[^"\\]|\\.)*)"', prefix):
            meta[k] = bytes(v, "utf-8").decode("unicode_escape", errors="replace")
        if not salvaged_items and not meta:
            return None
        meta[array_key] = salvaged_items
        return meta

    def _extract_json(self, raw: str) -> Any:
        if raw is None:
            raise json.JSONDecodeError("API returned None content", "", 0)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                balanced = self._extract_balanced_json(raw)
                return json.loads(balanced)
            except json.JSONDecodeError:
                # Last resort: a truncated array, or an object with a truncated inner array.
                salvaged_arr = self._recover_partial_array(raw)
                if salvaged_arr:
                    return salvaged_arr
                salvaged_obj = self._recover_partial_object(raw)
                if salvaged_obj:
                    return salvaged_obj
                raise

    def call_json(self, system: str, user: str, schema_hint: Optional[str] = None) -> Any:
        base_prompt = user
        if schema_hint:
            base_prompt = f"{user}\n\nReturn JSON only. Expected structure:\n{schema_hint}"
        last_error: Exception | None = None
        for attempt in range(3):
            prompt = base_prompt
            temperature = self.spec.temperature
            if attempt > 0:
                # Nudge the model away from the previous broken response.
                prompt = (
                    base_prompt
                    + "\n\nYour previous response was not valid JSON. "
                    "Return ONLY a JSON value, with no prose, no markdown, no comments."
                )
                # Add a small jitter so a deterministic miss doesn't repeat verbatim.
                temperature = min(1.0, max(self.spec.temperature, 0.0) + 0.2 * attempt)
            resp = completion(
                model=self.spec.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=self.spec.max_output_tokens,
            )
            content = resp["choices"][0]["message"]["content"]
            try:
                return self._extract_json(content)
            except json.JSONDecodeError as exc:
                last_error = exc
                if attempt < 2:
                    preview = content[:200] if len(content) <= 200 else content[:197] + "..."
                    print(f"  [retry {attempt+1}/3] JSON parse failed. Response preview: {preview}")
        if last_error is not None:
            raise last_error
        raise RuntimeError("call_json failed without producing an error")


class Pipeline:
    def __init__(self, config_path: Path, model_override: Optional[str] = None):
        self.config = load_yaml(config_path)
        self.config_path = config_path
        self.root = config_path.parent.parent
        self.output_root = self.root / self.config.get("output_dir", "output")
        ensure_dir(self.output_root)

        if model_override:
            family = model_override.split("/")[0]
            for role in ("generator", "judge", "tie_breaker_judge"):
                self.config["models"][role]["model"] = model_override
                self.config["models"][role]["family"] = family

        gen_cfg = self.config["models"]["generator"]
        judge_cfg = self.config["models"]["judge"]
        tiebreak_cfg = self.config["models"]["tie_breaker_judge"]

        self.generator = LLMClient(ModelSpec(**gen_cfg))
        self.judge = LLMClient(ModelSpec(**judge_cfg))
        self.tie_breaker = LLMClient(ModelSpec(**tiebreak_cfg))

        self.levels: List[str] = list(self.config["pipeline"].get("levels", LEVELS))
        if not model_override:
            self._validate_model_separation()

        self.scenario_schema = load_yaml(self.root / "config" / "scenario_schema.yaml")
        self._validator = Draft202012Validator(self.scenario_schema["json_schema"])
        self.dataset_version = str(self.scenario_schema.get("dataset_version", "v3"))
        self.rubric_version = str(self.scenario_schema.get("rubric_version", "construct_rubric.md@v2"))
        self.random_seed = int(self.config["pipeline"].get("random_seed", 17))

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

    # ── Run-control helpers ────────────────────────────────────────────────────

    def _resume_enabled(self) -> bool:
        return bool(self.config["pipeline"].get("resume", True))

    # ── Schema helpers ─────────────────────────────────────────────────────────

    def _trait_info(self, trait: str) -> Dict[str, Any]:
        try:
            return self.scenario_schema["traits"][trait]
        except KeyError as e:
            raise KeyError(f"Trait '{trait}' is not defined in scenario_schema.yaml") from e

    def _speech_acts_for(self, trait: str) -> List[Dict[str, Any]]:
        acts = self._trait_info(trait).get("speech_acts", [])
        if not acts:
            raise ValueError(f"No speech_acts configured for trait '{trait}'")
        return acts

    def _build_speech_act_queue(self, trait: str, n: int) -> List[str]:
        """Return a stratified, deterministic list of speech_act ids of length n."""
        acts = [a["id"] for a in self._speech_acts_for(trait)]
        per = n // len(acts)
        rem = n - per * len(acts)
        queue = []
        for a in acts:
            queue.extend([a] * per)
        # Distribute remainder round-robin across acts (deterministic).
        for i in range(rem):
            queue.append(acts[i % len(acts)])
        rng = random.Random(self.random_seed + sum(ord(c) for c in trait))
        rng.shuffle(queue)
        return queue

    def _content_invariant(self, scenario: Dict[str, Any]) -> str:
        """Return the invariant content string for a scenario, regardless of trait."""
        for key in ("speech_act_target", "proposition", "proposition_or_request", "requested_action"):
            v = scenario.get(key)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    def _validate_scenario(self, scenario: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = [e.message for e in self._validator.iter_errors(scenario)]
        return (not errors, errors)

    def _seed_example(self, trait: str) -> Dict[str, Any]:
        """A trait-aware seed example used when no scenarios exist yet."""
        if trait == "politeness":
            return {
                "scenario_id": f"{trait}-001",
                "trait": trait,
                "rubric_version": self.rubric_version,
                "dataset_version": self.dataset_version,
                "speech_act": "request",
                "domain": "workplace",
                "topic_cluster": "budget_request",
                "audience_relation": "peer",
                "register": "neutral",
                "communicative_goal": "ask a teammate to share the latest budget spreadsheet",
                "speech_act_target": "send the latest budget spreadsheet by end of day",
                "content_probes": [
                    {"question": "Is the speaker asking for the budget spreadsheet?", "expected_answer": "yes"},
                    {"question": "Is the speaker offering to do something for the listener?", "expected_answer": "no"},
                ],
                "allowed_named_entities": [],
                "forbidden_content_changes": ["do not change the requested document", "do not change the deadline"],
                "forbidden_cue_tokens": ["please", "kindly"],
                "forbidden_lexical_shortcuts": ["if you don't mind"],
                "target_word_count": 18,
                "length_tolerance_pct": 20,
                "imposition_level": "medium",
                "urgency_level": "low",
                "social_distance": "moderate",
                "notes": "keep target action and deadline fixed across rewrites",
            }
        if trait == "hedging_confidence":
            return {
                "scenario_id": f"{trait}-001",
                "trait": trait,
                "rubric_version": self.rubric_version,
                "dataset_version": self.dataset_version,
                "speech_act": "forecast",
                "domain": "workplace",
                "topic_cluster": "quarterly_forecast",
                "audience_relation": "senior",
                "register": "neutral",
                "communicative_goal": "tell a manager whether churn will rise next quarter",
                "speech_act_target": "predict that customer churn will rise next quarter",
                "content_probes": [
                    {"question": "Does the speaker predict that churn will rise?", "expected_answer": "yes"},
                    {"question": "Does the speaker give a numerical probability?", "expected_answer": "no"},
                ],
                "allowed_named_entities": [],
                "forbidden_content_changes": ["do not flip the predicted direction", "do not introduce numbers"],
                "forbidden_cue_tokens": ["maybe", "definitely"],
                "forbidden_lexical_shortcuts": ["I'm not sure but"],
                "target_word_count": 18,
                "length_tolerance_pct": 20,
                "proposition": "customer churn will rise next quarter",
                "evidence_state": "internal usage and renewal data trends",
                "answer_type": "forecast",
                "consequence_sensitivity": "medium",
                "notes": "vary only confidence in the forecast across levels",
            }
        return {
            "scenario_id": f"{trait}-001",
            "trait": trait,
            "rubric_version": self.rubric_version,
            "dataset_version": self.dataset_version,
            "speech_act": self._speech_acts_for(trait)[0]["id"],
            "domain": "workplace",
            "topic_cluster": "general",
            "audience_relation": "peer",
            "register": "neutral",
            "communicative_goal": "placeholder",
            "content_probes": [{"question": "placeholder?", "expected_answer": "yes"}],
            "forbidden_cue_tokens": [],
            "target_word_count": 18,
        }

    # ── Scenario generation ────────────────────────────────────────────────────

    def _scenario_system_prompt(self, trait: str) -> str:
        return (
            "You are building a research dataset for representation geometry. "
            "Create base scenarios, not labels or explanations. "
            "Scenarios must keep future ordinal rewrites content-controlled.\n\n"
            "CRITICAL: You MUST return only valid JSON. Do not include any text before or after the JSON. "
            "Do not add explanations, preambles, or comments. The entire response must be parseable as JSON."
        )

    def _scenario_user_prompt_batched(
        self,
        trait: str,
        speech_act_ids: List[str],
        previous_scenarios: List[Dict[str, Any]],
    ) -> str:
        """Build the per-batch prompt. Each scenario's assigned speech_act is fixed up front."""
        trait_info = self._trait_info(trait)
        shared = self.scenario_schema["definitions"]["shared_fields"]
        levels_str = ", ".join(self.levels)
        n = len(speech_act_ids)

        # Build per-scenario assignment lines and per-act guidance blocks.
        acts_by_id = {a["id"]: a for a in self._speech_acts_for(trait)}
        assignment_lines = "\n".join(
            f"  scenario {i + 1}: speech_act = {sid}" for i, sid in enumerate(speech_act_ids)
        )
        unique_acts = sorted(set(speech_act_ids))
        act_blocks = []
        for sid in unique_acts:
            a = acts_by_id[sid]
            block = f"- {sid}: {a.get('description', '')}"
            example_goal = a.get("example_communicative_goal")
            if example_goal:
                block += f"\n  example communicative_goal: {example_goal}"
            extra = a.get("extra_constraints") or []
            if extra:
                block += "\n  per-act constraints:\n" + "\n".join(f"    - {c}" for c in extra)
            act_blocks.append(block)

        prompt = (
            f"Generate exactly {n} base scenarios for the trait '{trait}'.\n"
            f"Trait description: {trait_info.get('description', '')}\n"
            f"Future ordinal rewriting will use levels: {levels_str}.\n\n"
            f"Each scenario MUST use the speech_act assigned to it below. Do not reassign or merge speech acts:\n"
            f"{assignment_lines}\n\n"
            f"Speech-act guide (only the acts you need this batch):\n"
            + "\n".join(act_blocks)
            + "\n\n"
            f"Required trait-specific fields: {trait_info['required_fields']}.\n"
            f"Trait-level generation constraints:\n"
            + "\n".join(f"  - {c}" for c in trait_info["generation_constraints"])
            + "\n\n"
            f"Shared fields each scenario must fill: {list(shared.keys())}.\n"
            "Hard requirements for every scenario object:\n"
            "  - scenario_id is a stable string (you may use 'auto' and the pipeline will reassign).\n"
            "  - trait must equal the trait above.\n"
            f"  - rubric_version = '{self.rubric_version}', dataset_version = '{self.dataset_version}'.\n"
            "  - content_probes is a list of 1-3 yes/no questions, each with expected_answer in [yes, no].\n"
            "  - target_word_count is an integer between 8 and 40.\n"
            "  - length_tolerance_pct is an integer (default 20).\n"
            "  - forbidden_cue_tokens is a list of surface tokens that must not dominate any single level.\n"
            "  - speech_act MUST equal the value assigned above for that scenario index.\n"
            "  - Vary domains, topic_clusters, audience_relations, registers across the batch.\n"
        )
        if previous_scenarios:
            slim = [
                {
                    "scenario_id": s.get("scenario_id"),
                    "speech_act": s.get("speech_act"),
                    "domain": s.get("domain"),
                    "topic_cluster": s.get("topic_cluster"),
                    "communicative_goal": s.get("communicative_goal"),
                }
                for s in previous_scenarios[-3:]
            ]
            prompt += (
                f"\nRECENT SCENARIOS — DO NOT repeat these topics/goals:\n"
                f"{json.dumps(slim, indent=2, ensure_ascii=False)}\n"
            )
        prompt += "\nReturn a JSON array of scenario objects only. No prose, no markdown."
        return prompt

    def make_scenarios(self, trait: str) -> None:
        total_n = int(self.config["pipeline"]["scenarios_per_trait"])
        batch_size = int(self.config["pipeline"].get("scenario_batch_size", 5))
        max_batch_retries = int(self.config["pipeline"].get("max_scenario_batch_retries", 3))
        max_dup_jaccard = float(self.config["pipeline"].get("max_duplicate_jaccard_scenarios", 0.85))
        system = self._scenario_system_prompt(trait)

        scenarios_file = self.trait_dir(trait) / "scenarios.jsonl"
        failed_batches_file = self.trait_dir(trait) / "failed_batches.jsonl"
        failed_scenarios_file = self.trait_dir(trait) / "failed_scenarios.jsonl"

        if not self._resume_enabled():
            for path in (scenarios_file, failed_batches_file, failed_scenarios_file):
                if path.exists():
                    path.unlink()
            print(f"resume=false → starting fresh for trait '{trait}'")
            cleaned: List[Dict[str, Any]] = []
        else:
            cleaned = jsonl_read(scenarios_file)
            if cleaned:
                print(f"resume=true → {len(cleaned)} scenarios already on disk, target={total_n}")

        # Build the full speech_act queue, then slice past what's already been done.
        full_queue = self._build_speech_act_queue(trait, total_n)
        # Honour speech_acts already recorded for existing scenarios (don't double-assign).
        # We just take the suffix corresponding to remaining slots.
        remaining_queue = full_queue[len(cleaned):]
        all_act_ids = [a["id"] for a in self._speech_acts_for(trait)]
        # Safety cap to prevent runaway loops on persistent model failures or repeated dedup hits.
        max_total_batches = max(8, ((total_n + batch_size - 1) // batch_size) * 4)
        consecutive_empty_batches = 0

        invariant_strings: set[str] = {
            normalize_text(self._content_invariant(s)).lower()
            for s in cleaned
            if self._content_invariant(s)
        }

        succeeded_batches = 0
        attempted_batches = 0
        batch_index = 0
        consecutive_failed_batches = 0
        FATAL_ERROR_PATTERNS = (
            "more credits",
            "insufficient_quota",
            "insufficient credits",
            "billing",
            "401",
            "403",
        )

        while len(cleaned) < total_n:
            if attempted_batches >= max_total_batches:
                print(f"  ! reached safety cap of {max_total_batches} batches; stopping at {len(cleaned)}/{total_n}.")
                break
            if consecutive_empty_batches >= 5:
                print(f"  ! 5 consecutive empty batches; stopping at {len(cleaned)}/{total_n}.")
                break
            batch_index += 1
            # Refill the queue if we burned through it but still need scenarios.
            if not remaining_queue:
                needed = total_n - len(cleaned)
                # Round-robin top-up, deterministic per (trait, batch_index).
                rng = random.Random(self.random_seed + 1000 * batch_index + sum(ord(c) for c in trait))
                topup = [all_act_ids[(rng.randrange(len(all_act_ids)) + i) % len(all_act_ids)] for i in range(needed)]
                rng.shuffle(topup)
                remaining_queue = topup
            batch_acts = remaining_queue[:batch_size]
            remaining_queue = remaining_queue[batch_size:]
            batch_n = len(batch_acts)
            attempted_batches += 1

            user = self._scenario_user_prompt_batched(trait, batch_acts, cleaned)
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
                # Skip & continue. Persist failure record, push acts back to queue front.
                err_msg = str(last_err) if last_err else ""
                failure_record = {
                    "trait": trait,
                    "batch_index": batch_index,
                    "speech_acts": batch_acts,
                    "error_type": type(last_err).__name__ if last_err else "Unknown",
                    "error_message": err_msg[:500],
                }
                with failed_batches_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(failure_record, ensure_ascii=False) + "\n")
                print(f"  ! batch {batch_index} ({batch_acts}) skipped after {max_batch_retries} retries.")
                consecutive_failed_batches += 1
                # Bail immediately on credit/quota/auth errors — retrying just burns time.
                if any(p in err_msg.lower() for p in FATAL_ERROR_PATTERNS):
                    print(
                        f"  ! fatal API error detected (credits/quota/auth). Stopping at {len(cleaned)}/{total_n}.\n"
                        f"    {err_msg[:240]}"
                    )
                    break
                if consecutive_failed_batches >= 3:
                    print(f"  ! 3 consecutive failed batches; stopping at {len(cleaned)}/{total_n}.")
                    break
                continue
            consecutive_failed_batches = 0

            if not isinstance(scenarios_raw, list):
                # Some models wrap the array under a key. Try common shapes.
                if isinstance(scenarios_raw, dict):
                    for key in ("scenarios", "items", "data"):
                        if isinstance(scenarios_raw.get(key), list):
                            scenarios_raw = scenarios_raw[key]
                            break
            if not isinstance(scenarios_raw, list):
                with failed_batches_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "trait": trait, "batch_index": batch_index,
                        "speech_acts": batch_acts, "error_type": "BadShape",
                        "error_message": "generator did not return a JSON array",
                    }, ensure_ascii=False) + "\n")
                print(f"  ! batch {batch_index} produced non-list output; skipping.")
                continue

            kept_in_batch = 0
            for i, raw_row in enumerate(scenarios_raw):
                if not isinstance(raw_row, dict):
                    continue
                # Force fields that the pipeline owns.
                row = dict(raw_row)
                row["trait"] = trait
                row.setdefault("rubric_version", self.rubric_version)
                row.setdefault("dataset_version", self.dataset_version)
                row.setdefault("length_tolerance_pct", 20)
                # Assign / overwrite speech_act from the queue (defensive).
                if i < len(batch_acts):
                    row["speech_act"] = batch_acts[i]
                # Generate scenario_id deterministically.
                next_idx = len(cleaned) + 1
                act_slug = slugify(str(row.get("speech_act", "act")))
                row["scenario_id"] = f"{trait}-{act_slug}-{next_idx:03d}"
                # Coerce content_probes if model returned a single dict.
                cp = row.get("content_probes")
                if isinstance(cp, dict):
                    row["content_probes"] = [cp]
                # Coerce list-typed fields if absent.
                row.setdefault("forbidden_cue_tokens", [])
                row.setdefault("forbidden_content_changes", [])
                row.setdefault("forbidden_lexical_shortcuts", [])
                row.setdefault("allowed_named_entities", [])

                ok, errors = self._validate_scenario(row)
                if not ok:
                    with failed_scenarios_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "trait": trait, "batch_index": batch_index,
                            "scenario_attempt": row, "errors": errors,
                        }, ensure_ascii=False) + "\n")
                    continue

                # Deduplication on invariant content.
                inv = normalize_text(self._content_invariant(row)).lower()
                if inv:
                    duplicate = False
                    for prev in invariant_strings:
                        if jaccard(inv, prev) >= max_dup_jaccard:
                            duplicate = True
                            break
                    if duplicate:
                        with failed_scenarios_file.open("a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "trait": trait, "batch_index": batch_index,
                                "scenario_attempt": row, "errors": ["near-duplicate invariant content"],
                            }, ensure_ascii=False) + "\n")
                        continue
                    invariant_strings.add(inv)

                cleaned.append(row)
                kept_in_batch += 1
                if len(cleaned) >= total_n:
                    break

            jsonl_write(scenarios_file, cleaned)
            succeeded_batches += 1
            if kept_in_batch == 0:
                consecutive_empty_batches += 1
            else:
                consecutive_empty_batches = 0
            print(
                f"  batch {batch_index}: kept {kept_in_batch}/{batch_n} "
                f"(total {len(cleaned)}/{total_n})"
            )

        # Diversity report: counts per (speech_act, domain).
        per_act: Dict[str, int] = {}
        per_domain: Dict[str, int] = {}
        for s in cleaned:
            per_act[s.get("speech_act", "?")] = per_act.get(s.get("speech_act", "?"), 0) + 1
            per_domain[s.get("domain", "?")] = per_domain.get(s.get("domain", "?"), 0) + 1
        diversity_path = self.trait_dir(trait) / "diversity_report.json"
        with diversity_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "trait": trait,
                    "total_scenarios": len(cleaned),
                    "by_speech_act": per_act,
                    "by_domain": per_domain,
                    "attempted_batches": attempted_batches,
                    "succeeded_batches": succeeded_batches,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(
            f"Wrote {len(cleaned)}/{total_n} scenarios to {scenarios_file}. "
            f"Batches: {succeeded_batches}/{attempted_batches} ok. "
            f"Speech-act distribution: {per_act}"
        )

    # ── Ladder generation ──────────────────────────────────────────────────────

    def _ladder_system_prompt(self, trait: str) -> str:
        rubric = (self.root / "config" / "construct_rubric.md").read_text(encoding="utf-8")
        return (
            "You create controlled ordinal ladders for NLP research. "
            "Keep content fixed and vary only trait intensity.\n\n"
            f"Rubric:\n{rubric}\n\n"
            "CRITICAL: You MUST return only valid JSON. Do not include any text before or after the JSON. "
            "Do not add explanations, preambles, or comments. The entire response must be parseable as JSON."
        )

    def _ladder_user_prompt(self, trait: str, scenario: Dict[str, Any]) -> str:
        levels_str = ", ".join(self.levels)
        invariant = self._content_invariant(scenario)
        speech_act = scenario.get("speech_act", "")
        return (
            f"Create one {len(self.levels)}-level canonical ladder for trait '{trait}', "
            f"speech_act '{speech_act}'.\n"
            f"Scenario:\n{json.dumps(scenario, ensure_ascii=False, indent=2)}\n\n"
            f"Levels must be {levels_str}.\n"
            f"The invariant content '{invariant}' must remain identical in meaning across levels.\n"
            "Vary only the trait intensity. Keep all paraphrases similar in length.\n"
            "Return JSON with keys: scenario_id, trait, invariant_content, ladder.\n"
            f"'ladder' must map each of these levels to a single sentence: {levels_str}."
        )

    def make_ladders(self, trait: str) -> None:
        scenarios = jsonl_read(self.trait_dir(trait) / "scenarios.jsonl")
        if not scenarios:
            raise FileNotFoundError("Run make-scenarios first.")
        system = self._ladder_system_prompt(trait)
        max_workers = int(self.config["pipeline"].get("max_workers", 1))

        ladders_file = self.trait_dir(trait) / "ladders.jsonl"
        if not self._resume_enabled():
            if ladders_file.exists():
                ladders_file.unlink()
            existing_ladders: List[Dict[str, Any]] = []
            print(f"resume=false → regenerating all ladders for trait '{trait}'")
        else:
            existing_ladders = jsonl_read(ladders_file)
            if existing_ladders:
                print(f"resume=true → {len(existing_ladders)} ladders already on disk")
        done_ids = {l["scenario_id"] for l in existing_ladders}
        scenarios = [s for s in scenarios if s["scenario_id"] not in done_ids]
        if not scenarios:
            print("All ladders already generated")
            return

        def _one(scenario: Dict[str, Any]) -> Dict[str, Any]:
            schema_hint = json.dumps(
                {
                    "scenario_id": scenario["scenario_id"],
                    "trait": trait,
                    "invariant_content": self._content_invariant(scenario),
                    "ladder": {lvl: "..." for lvl in self.levels},
                },
                ensure_ascii=False,
            )
            obj = self.generator.call_json(system, self._ladder_user_prompt(trait, scenario), schema_hint)
            # Ensure ladder has all required levels; if not, raise so the worker captures it.
            ladder = obj.get("ladder") if isinstance(obj, dict) else None
            if not isinstance(ladder, dict) or any(lvl not in ladder for lvl in self.levels):
                raise ValueError(
                    f"Ladder for {scenario.get('scenario_id')} missing levels; got {list(ladder) if isinstance(ladder, dict) else type(ladder).__name__}"
                )
            obj["scenario"] = scenario
            return obj

        failed_ladders_file = self.trait_dir(trait) / "failed_ladders.jsonl"

        def _safe_one(scenario: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            try:
                return _one(scenario)
            except Exception as exc:
                with failed_ladders_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "scenario_id": scenario.get("scenario_id"),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:500],
                    }, ensure_ascii=False) + "\n")
                print(f"  ! ladder failed for {scenario.get('scenario_id')}: {type(exc).__name__}: {str(exc)[:120]}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            out_rows = [r for r in ex.map(_safe_one, scenarios) if r is not None]
        all_rows = existing_ladders + out_rows
        jsonl_write(ladders_file, all_rows)
        skipped = len(scenarios) - len(out_rows)
        print(f"Wrote {len(out_rows)} new ladders ({len(all_rows)} total). Skipped {skipped}.")

    # ── Paraphrase generation (with cross-ladder cue-family diversity) ─────────

    def _paraphrase_system_prompt(self, trait: str) -> str:
        return (
            "You generate paraphrases for a representation-geometry benchmark. "
            "Preserve meaning exactly while varying lexical realization. "
            "Do not produce near-duplicates. Use different cue families when possible. "
            "Critical: all paraphrases must be similar in length across levels. "
            "Do not use sentence length or verbosity as a cue for the trait level. "
            "A high-intensity paraphrase must not be longer than a low-intensity one.\n\n"
            "CRITICAL: You MUST return only valid JSON. Do not include any text before or after the JSON. "
            "Do not add explanations, preambles, or comments. The entire response must be parseable as JSON."
        )

    def _paraphrase_user_prompt(
        self,
        ladder_obj: Dict[str, Any],
        per_level: int,
        used_cue_families: set[str],
    ) -> str:
        avoid = sorted(used_cue_families - {""})
        n_levels = len(ladder_obj["ladder"])
        prompt = (
            f"Given this canonical {n_levels}-level ladder:\n"
            f"{json.dumps(ladder_obj['ladder'], ensure_ascii=False, indent=2)}\n\n"
            f"Generate {per_level} paraphrases per level.\n"
            "For each paraphrase, provide: level, cue_family, text.\n"
            "Cue families should differ when possible, such as lexical marker, syntactic "
            "framing, gratitude framing, evidential framing, indirectness, modal framing.\n"
            "Keep the proposition or requested action unchanged.\n"
            "Keep all paraphrases similar in length regardless of level. "
            "Express intensity through word choice and framing, not sentence length.\n"
            "Return JSON with keys: scenario_id, trait, items."
        )
        if avoid:
            prompt += (
                f"\n\nAlready used in this dataset — vary away from these cue families: {avoid}."
            )
        return prompt

    def _repair_paraphrases_user_prompt(
        self,
        ladder_obj: Dict[str, Any],
        per_level: int,
        judge_result: Dict[str, Any],
        used_cue_families: set[str],
    ) -> str:
        failed = [
            k for k, v in judge_result.get("checks", {}).items()
            if v is False or v == "high"
        ]
        notes = judge_result.get("notes", "no notes provided")
        avoid = sorted(used_cue_families - {""})
        prompt = (
            f"The previous paraphrases for this ladder were rejected by the validation judge.\n"
            f"Failed checks: {failed or 'none listed'}.\n"
            f"Judge notes: {notes}\n\n"
            f"Canonical ladder:\n"
            f"{json.dumps(ladder_obj['ladder'], ensure_ascii=False, indent=2)}\n\n"
            f"Generate {per_level} corrected paraphrases per level, addressing the issues above.\n"
            "For each paraphrase, provide: level, cue_family, text.\n"
            "Keep the proposition or requested action unchanged.\n"
            "Return JSON with keys: scenario_id, trait, items."
        )
        if avoid:
            prompt += f"\nAvoid these already-used cue families: {avoid}."
        return prompt

    def make_paraphrases(self, trait: str) -> None:
        ladders = jsonl_read(self.trait_dir(trait) / "ladders.jsonl")
        if not ladders:
            raise FileNotFoundError("Run make-ladders first.")
        ladders = sorted(ladders, key=lambda x: x["scenario_id"])
        per_level = int(self.config["pipeline"]["paraphrases_per_level"])
        system = self._paraphrase_system_prompt(trait)
        max_workers = int(self.config["pipeline"].get("max_workers", 1))

        # Load existing paraphrases to avoid re-processing
        paraphrases_file = self.trait_dir(trait) / "paraphrases.jsonl"
        existing_paraphrases: List[Dict[str, Any]] = []
        processed_scenario_ids: set[str] = set()

        if not self._resume_enabled():
            if paraphrases_file.exists():
                paraphrases_file.unlink()
            print(f"resume=false → regenerating all paraphrases for trait '{trait}'")
        elif paraphrases_file.exists():
            existing_paraphrases = jsonl_read(paraphrases_file)
            processed_scenario_ids = {row["scenario_id"] for row in existing_paraphrases}
            print(f"resume=true → {len(processed_scenario_ids)} scenarios already processed")
        
        # Filter to remaining ladders
        remaining_ladders = [l for l in ladders if l["scenario_id"] not in processed_scenario_ids]
        
        if not remaining_ladders:
            print("All paraphrases already generated")
            return

        used_cue_families: set[str] = set()
        lock = threading.Lock()

        def _one(ladder: Dict[str, Any]) -> List[Dict[str, Any]]:
            schema_hint = json.dumps(
                {
                    "scenario_id": ladder["scenario_id"],
                    "trait": trait,
                    "items": [{"level": "low", "cue_family": "syntactic indirectness", "text": "..."}],
                },
                ensure_ascii=False,
            )
            with lock:
                avoid = set(used_cue_families)
            obj = self.generator.call_json(
                system,
                self._paraphrase_user_prompt(ladder, per_level, avoid),
                schema_hint,
            )
            scenario_id = obj["scenario_id"]
            batch: List[Dict[str, Any]] = []
            for idx, item in enumerate(obj["items"], start=1):
                cue_family = item.get("cue_family", "unspecified")
                batch.append({
                    "scenario_id": scenario_id,
                    "trait": trait,
                    "level": item["level"],
                    "cue_family": cue_family,
                    "text": normalize_text(item["text"]),
                    "canonical": ladder["ladder"].get(item["level"], ""),
                    "invariant_content": ladder.get("invariant_content", ""),
                    "paraphrase_id": f"{scenario_id}-{item['level']}-{idx:02d}",
                })
            with lock:
                for r in batch:
                    used_cue_families.add(r["cue_family"])
            return batch

        failed_paraphrases_file = self.trait_dir(trait) / "failed_paraphrases.jsonl"

        def _safe_one(ladder: Dict[str, Any]) -> List[Dict[str, Any]]:
            try:
                return _one(ladder)
            except Exception as exc:
                with failed_paraphrases_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "scenario_id": ladder.get("scenario_id"),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:500],
                    }, ensure_ascii=False) + "\n")
                print(f"  ! paraphrases failed for {ladder.get('scenario_id')}: {type(exc).__name__}: {str(exc)[:120]}")
                return []

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_safe_one, remaining_ladders))
        new_rows = [row for batch in results for row in batch]
        all_rows = existing_paraphrases + new_rows
        jsonl_write(paraphrases_file, all_rows)
        skipped = sum(1 for r in results if not r)
        print(f"Wrote {len(new_rows)} new paraphrase rows (total: {len(all_rows)}). Skipped {skipped} ladders.")

    # ── Judging (with per-scenario retry using judge feedback) ─────────────────

    def _judge_system_prompt(self) -> str:
        rubric = (self.root / "config" / "construct_rubric.md").read_text(encoding="utf-8")
        return (
            "You are an independent validation judge for a benchmark. "
            "Your job is to reject content drift, wrong ordering, poor fluency, "
            "weak cue diversity, and shortcut-heavy bundles.\n\n"
            f"Rubric:\n{rubric}\n\n"
            "CRITICAL: You MUST return only valid JSON. Do not include any text before or after the JSON. "
            "Do not add explanations, preambles, or comments. The entire response must be parseable as JSON."
        )

    def _judge_user_prompt(self, trait: str, scenario_id: str, bundle: List[Dict[str, Any]]) -> str:
        levels_order = " < ".join(self.levels)
        grouped = {
            level: [{"id": r["paraphrase_id"], "text": r["text"]}
                    for r in bundle if r["level"] == level]
            for level in self.levels
        }
        return (
            f"Validate this bundle for trait '{trait}'.\n"
            f"Texts by level (use the exact 'id' values in your item_scores):\n"
            f"{json.dumps(grouped, ensure_ascii=False, indent=2)}\n\n"
            f"Check: proposition/request preservation, correct {levels_order} ordering, "
            "naturalness, paraphrase diversity, obvious lexical shortcut risk, and "
            "length_balance (flag if one level's texts are substantially longer/shorter "
            "than the others — length must not be a trait cue).\n"
            "Return JSON with keys: accepted, overall_score, checks, notes, item_scores.\n"
            "'checks' must include content_preservation, monotonic_order, naturalness, "
            "cue_diversity, shortcut_risk, length_balance.\n"
            "'item_scores' must be a list where each entry has the exact 'id' string from "
            "above as 'paraphrase_id', and a 'score' in [0,1]."
        )

    def _judge_scenario_bundle(
        self,
        trait: str,
        scenario_id: str,
        bundle: List[Dict[str, Any]],
        system: str,
        min_score: float,
    ) -> Tuple[Dict[str, Any], bool, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run primary + optional tie-breaker judgment for one scenario bundle.

        Returns (verdict, bundle_accepted, all_judged_rows, accepted_rows).
        """
        example_ids = [r["paraphrase_id"] for r in bundle[:2]] if bundle else ["x"]
        schema_hint = json.dumps(
            {
                "accepted": True,
                "overall_score": 0.92,
                "checks": {
                    "content_preservation": True,
                    "monotonic_order": True,
                    "naturalness": True,
                    "cue_diversity": True,
                    "shortcut_risk": "low",
                    "length_balance": True,
                },
                "notes": "brief explanation",
                "item_scores": [{"paraphrase_id": pid, "score": 0.91} for pid in example_ids],
            },
            ensure_ascii=False,
        )
        tie_breaker_margin = float(
            self.config["pipeline"].get("tie_breaker_margin", 0.15)
        )
        primary = self.judge.call_json(
            system, self._judge_user_prompt(trait, scenario_id, bundle), schema_hint
        )
        bundle_accepted = (
            bool(primary.get("accepted", False))
            and float(primary.get("overall_score", 0.0)) >= min_score
        )
        # Hard programmatic length-balance gate. Don't trust the judge LLM alone.
        max_ratio = float(self.config["pipeline"].get("max_length_ratio", 1.2))
        word_counts_by_level: Dict[str, List[int]] = {}
        for r in bundle:
            word_counts_by_level.setdefault(r["level"], []).append(len(r["text"].split()))
        if word_counts_by_level:
            mean_words = [sum(v) / len(v) for v in word_counts_by_level.values() if v]
            if len(mean_words) >= 2 and min(mean_words) > 0:
                if max(mean_words) > min(mean_words) * max_ratio:
                    primary["length_balance_hard_gate"] = False
                    primary.setdefault("notes", "")
                    primary["notes"] = (primary["notes"] or "") + (
                        f" [length-balance hard gate fired: per-level mean words {dict(zip(word_counts_by_level.keys(), [round(m, 1) for m in mean_words]))}]"
                    )
                    bundle_accepted = False
                else:
                    primary["length_balance_hard_gate"] = True
        if not bundle_accepted and float(primary.get("overall_score", 0.0)) >= max(0.0, min_score - tie_breaker_margin):
            tie = self.tie_breaker.call_json(
                system, self._judge_user_prompt(trait, scenario_id, bundle), schema_hint
            )
            bundle_accepted = (
                bool(tie.get("accepted", False))
                and float(tie.get("overall_score", 0.0)) >= min_score
            )
            primary["tie_breaker"] = tie
        item_scores = {
            x["paraphrase_id"]: x["score"]
            for x in primary.get("item_scores", [])
            if "paraphrase_id" in x
        }
        # If the bundle is accepted overall, a missing item score means the judge
        # didn't explicitly reject that item — give benefit of the doubt.
        # Only items with an explicit score below min_score get filtered out.
        default_item_score = min_score if bundle_accepted else 0.0
        judged_rows: List[Dict[str, Any]] = []
        accepted_rows: List[Dict[str, Any]] = []
        for row in bundle:
            rec = dict(row)
            rec["judge"] = primary
            rec["accepted"] = bundle_accepted and item_scores.get(row["paraphrase_id"], default_item_score) >= min_score
            rec["item_score"] = item_scores.get(row["paraphrase_id"])
            judged_rows.append(rec)
            if rec["accepted"]:
                accepted_rows.append(rec)
        if bundle_accepted and not accepted_rows:
            print(
                f"  WARNING: bundle {scenario_id} accepted overall but 0 items passed "
                f"item-level threshold {min_score}. "
                f"item_scores keys: {list(item_scores.keys())[:5]}"
            )
        return primary, bundle_accepted, judged_rows, accepted_rows

    def _repair_bundle(
        self,
        trait: str,
        ladder: Dict[str, Any],
        per_level: int,
        judge_result: Dict[str, Any],
        used_cue_families: set[str],
    ) -> Optional[List[Dict[str, Any]]]:
        """Regenerate paraphrases for a rejected bundle using the judge's critique."""
        scenario_id = ladder["scenario_id"]
        system = self._paraphrase_system_prompt(trait)
        user = self._repair_paraphrases_user_prompt(ladder, per_level, judge_result, used_cue_families)
        schema_hint = json.dumps(
            {
                "scenario_id": scenario_id,
                "trait": trait,
                "items": [{"level": "low", "cue_family": "...", "text": "..."}],
            },
            ensure_ascii=False,
        )
        try:
            obj = self.generator.call_json(system, user, schema_hint)
        except Exception as exc:
            print(f"  Repair generation failed for {scenario_id}: {exc}")
            return None
        return [
            {
                "scenario_id": scenario_id,
                "trait": trait,
                "level": item["level"],
                "cue_family": item.get("cue_family", "unspecified"),
                "text": normalize_text(item["text"]),
                "canonical": ladder["ladder"].get(item["level"], ""),
                "invariant_content": ladder.get("invariant_content", ""),
                "paraphrase_id": f"{scenario_id}-{item['level']}-r{idx:02d}",
            }
            for idx, item in enumerate(obj.get("items", []), start=1)
        ]

    def judge_bundles(self, trait: str) -> None:
        rows = jsonl_read(self.trait_dir(trait) / "paraphrases.jsonl")
        if not rows:
            raise FileNotFoundError("Run make-paraphrases first.")
        ladders = jsonl_read(self.trait_dir(trait) / "ladders.jsonl")
        ladder_map = {l["scenario_id"]: l for l in ladders}

        system = self._judge_system_prompt()
        min_score = float(self.config["pipeline"]["min_acceptance_score"])
        per_level = int(self.config["pipeline"]["paraphrases_per_level"])
        max_retries = int(self.config["pipeline"].get("max_retries", 3))

        # Load existing results to avoid re-judging
        judged_file = self.trait_dir(trait) / "judged.jsonl"
        accepted_file = self.trait_dir(trait) / "accepted.jsonl"
        existing_judged: List[Dict[str, Any]] = []
        existing_accepted: List[Dict[str, Any]] = []
        processed_scenario_ids: set[str] = set()
        
        if not self._resume_enabled():
            for path in (judged_file, accepted_file):
                if path.exists():
                    path.unlink()
            print(f"resume=false → re-judging all bundles for trait '{trait}'")
        elif judged_file.exists() and accepted_file.exists():
            existing_judged = jsonl_read(judged_file)
            existing_accepted = jsonl_read(accepted_file)
            processed_scenario_ids = {row["scenario_id"] for row in existing_judged}
            print(f"resume=true → {len(processed_scenario_ids)} scenarios already judged")

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["scenario_id"], []).append(row)

        # Filter to remaining bundles
        remaining_bundles = [(sid, bundle) for sid, bundle in sorted(grouped.items()) 
                            if sid not in processed_scenario_ids]
        
        if not remaining_bundles:
            print("All bundles already judged")
            return

        used_cue_families: set[str] = set()
        lock = threading.Lock()
        max_workers = int(self.config["pipeline"].get("max_workers", 1))

        def _one(item: tuple) -> tuple:
            scenario_id, bundle = item
            ladder = ladder_map.get(scenario_id)
            verdict, accepted, judged_bundle, accepted_bundle = self._judge_scenario_bundle(
                trait, scenario_id, bundle, system, min_score
            )
            for attempt in range(max_retries):
                if accepted or ladder is None:
                    break
                print(f"  Retrying {scenario_id} (attempt {attempt + 1}/{max_retries})")
                with lock:
                    avoid = set(used_cue_families)
                repaired = self._repair_bundle(trait, ladder, per_level, verdict, avoid)
                if repaired is None:
                    break
                verdict, accepted, judged_bundle, accepted_bundle = self._judge_scenario_bundle(
                    trait, scenario_id, repaired, system, min_score
                )
            with lock:
                for r in accepted_bundle:
                    if r.get("cue_family"):
                        used_cue_families.add(r["cue_family"])
            return judged_bundle, accepted_bundle

        failed_judged_file = self.trait_dir(trait) / "failed_judged.jsonl"

        def _safe_one(item: tuple) -> tuple:
            try:
                return _one(item)
            except Exception as exc:
                sid = item[0] if isinstance(item, tuple) else "?"
                with failed_judged_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "scenario_id": sid,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:500],
                    }, ensure_ascii=False) + "\n")
                print(f"  ! judging failed for {sid}: {type(exc).__name__}: {str(exc)[:120]}")
                return ([], [])

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_safe_one, remaining_bundles))

        new_judged: List[Dict[str, Any]] = []
        new_accepted: List[Dict[str, Any]] = []
        for judged_bundle, accepted_bundle in results:
            new_judged.extend(judged_bundle)
            new_accepted.extend(accepted_bundle)

        all_judged = existing_judged + new_judged
        all_accepted = existing_accepted + new_accepted
        
        jsonl_write(judged_file, all_judged)
        jsonl_write(accepted_file, all_accepted)
        print(f"Judged {len(new_judged)} new rows (total: {len(all_judged)}); accepted {len(new_accepted)} new rows (total: {len(all_accepted)})")

    # ── Audit (with shortcut hard gate that triggers targeted regeneration) ────

    def _find_shortcut_scenarios(
        self, rows: List[Dict[str, Any]], shortcut_ngrams: List[str]
    ) -> List[str]:
        """Return scenario IDs where a shortcut n-gram appears at only one level."""
        by_scenario: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            by_scenario.setdefault(row["scenario_id"], []).append(row)
        flagged = []
        for sid, bundle in by_scenario.items():
            for ngram in shortcut_ngrams:
                levels_with = {r["level"] for r in bundle if ngram in r["text"].lower()}
                if len(levels_with) == 1:
                    flagged.append(sid)
                    break
        return flagged

    def _regenerate_shortcut_scenarios(
        self,
        trait: str,
        scenario_ids: List[str],
        forbidden_ngrams: List[str],
        used_cue_families: set[str],
    ) -> None:
        """Regenerate paraphrases and re-judge for shortcut-flagged scenarios."""
        ladders = jsonl_read(self.trait_dir(trait) / "ladders.jsonl")
        ladder_map = {l["scenario_id"]: l for l in ladders}
        per_level = int(self.config["pipeline"]["paraphrases_per_level"])
        min_score = float(self.config["pipeline"]["min_acceptance_score"])
        system_para = self._paraphrase_system_prompt(trait)
        system_judge = self._judge_system_prompt()
        forbidden_str = ", ".join(forbidden_ngrams[:10])

        accepted = [
            r for r in jsonl_read(self.trait_dir(trait) / "accepted.jsonl")
            if r["scenario_id"] not in scenario_ids
        ]

        for sid in scenario_ids:
            ladder = ladder_map.get(sid)
            if ladder is None:
                continue
            user = (
                self._paraphrase_user_prompt(ladder, per_level, used_cue_families)
                + f"\n\nIMPORTANT: do not use these shortcut expressions: {forbidden_str}."
            )
            schema_hint = json.dumps(
                {
                    "scenario_id": sid,
                    "trait": trait,
                    "items": [{"level": "low", "cue_family": "...", "text": "..."}],
                },
                ensure_ascii=False,
            )
            try:
                obj = self.generator.call_json(system_para, user, schema_hint)
            except Exception as exc:
                print(f"  Shortcut repair generation failed for {sid}: {exc}")
                continue
            new_rows = [
                {
                    "scenario_id": sid,
                    "trait": trait,
                    "level": item["level"],
                    "cue_family": item.get("cue_family", "unspecified"),
                    "text": normalize_text(item["text"]),
                    "canonical": ladder["ladder"].get(item["level"], ""),
                    "invariant_content": ladder.get("invariant_content", ""),
                    "paraphrase_id": f"{sid}-{item['level']}-s{idx:02d}",
                }
                for idx, item in enumerate(obj.get("items", []), start=1)
            ]
            try:
                _, _, _, new_accepted = self._judge_scenario_bundle(
                    trait, sid, new_rows, system_judge, min_score
                )
            except Exception as exc:
                print(f"  Shortcut repair judging failed for {sid}: {exc}")
                continue
            accepted.extend(new_accepted)

        jsonl_write(self.trait_dir(trait) / "accepted.jsonl", accepted)
        print(f"Shortcut repair done; accepted.jsonl now has {len(accepted)} rows")

    def audit(self, trait: str, _repair_round: int = 0) -> None:
        rows = jsonl_read(self.trait_dir(trait) / "accepted.jsonl")
        if not rows:
            raise FileNotFoundError("Run judge first.")
        max_dup = float(self.config["pipeline"]["max_duplicate_jaccard"])
        warning_acc = float(self.config["pipeline"]["lexical_baseline_warning_accuracy"])
        max_rounds = int(self.config["pipeline"].get("max_shortcut_repair_rounds", 2))

        duplicates: List[Dict[str, Any]] = []
        for a, b in itertools.combinations(rows, 2):
            score = jaccard(a["text"], b["text"])
            if score >= max_dup:
                duplicates.append({
                    "a": a["paraphrase_id"],
                    "b": b["paraphrase_id"],
                    "jaccard": round(score, 4),
                })

        df = pd.DataFrame(rows)
        lexical_accuracy = None
        top_ngrams: List[Dict[str, Any]] = []
        shortcut_ngrams: List[str] = []

        if len(df) >= 12 and df["level"].nunique() >= 2:
            vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
            X = vectorizer.fit_transform(df["text"])
            y = df["level"]
            n_splits = min(5, y.value_counts().min())
            if n_splits >= 2:
                clf = LogisticRegression(max_iter=2000)
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
                preds = cross_val_predict(clf, X, y, cv=cv)
                lexical_accuracy = float(accuracy_score(y, preds))
                clf.fit(X, y)
                vocab = vectorizer.get_feature_names_out()
                for idx, label in enumerate(clf.classes_):
                    top_ids = clf.coef_[idx].argsort()[-10:][::-1]
                    level_top = [vocab[i] for i in top_ids]
                    top_ngrams.append({"level": label, "top_positive_ngrams": level_top})
                    shortcut_ngrams.extend(level_top[:5])

        used_cue_families: set[str] = {
            r["cue_family"] for r in rows if r.get("cue_family")
        }

        # Hard gate: if lexical accuracy is too high, trigger targeted regeneration
        if (
            lexical_accuracy is not None
            and lexical_accuracy >= warning_acc
            and _repair_round < max_rounds
            and shortcut_ngrams
        ):
            flagged = self._find_shortcut_scenarios(rows, shortcut_ngrams)
            if flagged:
                print(
                    f"Shortcut gate (round {_repair_round + 1}/{max_rounds}): "
                    f"regenerating {len(flagged)} scenarios"
                )
                self._regenerate_shortcut_scenarios(trait, flagged, shortcut_ngrams, used_cue_families)
                self.audit(trait, _repair_round=_repair_round + 1)
                return

        scenario_balance = (
            df.groupby(["scenario_id", "level"])
            .size()
            .reset_index(name="count")
            .to_dict(orient="records")
        )

        df["text_len"] = df["text"].str.len()
        df["word_count"] = df["text"].str.split().str.len()
        char_len_by_level = df.groupby("level")["text_len"].mean().to_dict()
        word_len_by_level = df.groupby("level")["word_count"].mean().to_dict()
        max_ratio = float(self.config["pipeline"].get("max_length_ratio", 1.2))
        char_lengths = list(char_len_by_level.values())
        word_lengths = list(word_len_by_level.values())
        char_imbalanced = (
            bool(max(char_lengths) > min(char_lengths) * max_ratio)
            if len(char_lengths) >= 2 and min(char_lengths) > 0 else False
        )
        word_imbalanced = (
            bool(max(word_lengths) > min(word_lengths) * max_ratio)
            if len(word_lengths) >= 2 and min(word_lengths) > 0 else False
        )
        length_imbalanced = char_imbalanced or word_imbalanced
        if length_imbalanced:
            print(
                f"WARNING: length imbalance (>{max_ratio}×) across levels — "
                f"chars={char_len_by_level}, words={word_len_by_level}"
            )

        report = {
            "trait": trait,
            "num_rows": len(rows),
            "shortcut_repair_rounds": _repair_round,
            "duplicate_pairs_over_threshold": duplicates,
            "num_duplicate_pairs": len(duplicates),
            "lexical_baseline_accuracy": lexical_accuracy,
            "lexical_baseline_warning": lexical_accuracy is not None and lexical_accuracy >= warning_acc,
            "top_ngrams_by_level": top_ngrams,
            "scenario_level_balance": scenario_balance,
            "mean_char_length_by_level": {k: round(v, 1) for k, v in char_len_by_level.items()},
            "mean_word_count_by_level": {k: round(v, 2) for k, v in word_len_by_level.items()},
            "length_ratio_threshold": max_ratio,
            "length_imbalance_warning": length_imbalanced,
        }
        with (self.trait_dir(trait) / "audit_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Wrote audit report to {self.trait_dir(trait) / 'audit_report.json'}")

    # ── Human validation export (stratified BWS tasks) ─────────────────────────

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
            for row in grouped[sid]:
                human_rows.append({
                    "scenario_id": sid,
                    "paraphrase_id": row["paraphrase_id"],
                    "trait": row["trait"],
                    "level": row["level"],
                    "text": row["text"],
                    "check_order_within_bundle": "",
                    "check_content_preserved": "",
                    "check_natural": "",
                    "comments": "",
                })
        pd.DataFrame(human_rows).to_csv(self.trait_dir(trait) / "human_validation.csv", index=False)

        if self.config["pipeline"].get("use_bws_exports", True):
            items = [r for r in rows if r["scenario_id"] in sampled]
            bws_rows: List[Dict[str, Any]] = []
            rng = random.Random(11)
            if len(items) >= 4:
                for i in range(min(100, max(0, len(items) // 2))):
                    quad = rng.sample(items, 4)
                    bws_rows.append({
                        "task_id": f"{trait}-bws-{i:03d}",
                        "trait": trait,
                        "item_a": quad[0]["text"],
                        "item_b": quad[1]["text"],
                        "item_c": quad[2]["text"],
                        "item_d": quad[3]["text"],
                        "most_target_like": "",
                        "least_target_like": "",
                    })
            else:
                print(f"WARNING: only {len(items)} items in BWS pool; skipping BWS export (need ≥4)")
            pd.DataFrame(bws_rows).to_csv(self.trait_dir(trait) / "bws_tasks.csv", index=False)

        print(f"Exported human validation files for {trait}")

    # ── Export to main pipeline format ────────────────────────────────────────

    def export_prompts(self, out_path: Optional[Path] = None) -> None:
        """Export accepted paraphrases from all traits to data/prompts.json.

        Produces the {prompt, trait, intensity} format consumed by src/data.py.
        """
        if out_path is None:
            out_path = self.root / "data" / "prompts.json"
        rows = []
        for trait in self.config["traits"]:
            for rec in jsonl_read(self.trait_dir(trait) / "accepted.jsonl"):
                rows.append({
                    "prompt": rec["text"],
                    "trait": rec["trait"],
                    "intensity": rec["level"],
                    "scenario_id": rec.get("scenario_id", ""),
                })
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
    parser.add_argument(
        "--model",
        help="Override all model roles with a single litellm model string, e.g. anthropic/claude-haiku-4-5-20251001",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command != "export-prompts" and args.trait is None:
        raise SystemExit("error: --trait is required for this command")
    pipe = Pipeline(Path(args.config), model_override=args.model)
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
