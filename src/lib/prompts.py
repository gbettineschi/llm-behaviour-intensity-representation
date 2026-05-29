"""Politeness prompt-dataset generation (English-only).

One module: shared types/IO (Sample, load_accepted, …), rubric, scenario schema, LLM client,
and the Pipeline. `generate_prompts(out_dir, ...)` runs the stages and writes into out_dir;
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

# --- shared types & IO helpers (this module is `lib.prompts`)

TRAITS = ["politeness"]
LEVELS = ["negative", "neutral", "positive"]


class Sample(NamedTuple):
    prompt: str
    trait: str
    intensity: str
    scenario_id: str = ""


def load_accepted(path: str | Path) -> list[Sample]:
    """Load samples from a filtered.jsonl dataset (text/trait/level/scenario_id)."""
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


# --- construct rubric

CONSTRUCT_RUBRIC = """\
# Politeness rubric (signed 3-point scale)

Rewrite one fixed message at three politeness levels — negative (impolite), neutral
(unmarked), positive (polite) — changing only politeness, never the content. Neutral is a
true zero: neither courtesy nor rudeness markers. Negative and positive are equal-and-opposite
departures from it.

## Keep constant across the three levels
- the speech-act target (what is requested / refused / criticised / etc.)
- named entities, dates, deadlines, and the core intent
- the polarity of the act (a refusal stays a refusal; an apology stays an apology)
- urgency, scope, and the amount imposed
- length: all three levels should be about the same number of words

## The three levels
**Negative — impolite.** Dismissive, curt, grudging, impatient, or condescending; it must read
as clearly rude to an ordinary reader, not merely plain or short. Rudeness comes from tone and
framing, never from changing the content, and never from profanity, threats, or attacks on the
person (those would be lexical giveaways).

**Neutral — unmarked.** Plain, matter-of-fact, even-toned; no courtesy and no rudeness markers.

**Positive — polite.** Respectful and mitigated: appreciation, deference, acknowledging the
imposition. Politeness is in the framing, not extra words — keep it the same length.

## Per-speech-act examples
**request.** Negative: rude, impatient demand ("Just send me the file already."). Neutral: plain direct request with no softeners or brusqueness ("Can you send me the file?"). Positive: deferential framing with gratitude or imposition acknowledgment ("I'd really appreciate it if you could send me the file when you get a chance."). The requested action stays identical.

**refusal.** Negative: abrupt, dismissive decline with no acknowledgment of the offer ("No. I'm not doing that."). Neutral: plain decline with brief reason ("I won't be able to make it."). Positive: appreciative refusal acknowledging the offer and apologising ("Thank you so much for the invitation — I'm afraid I won't be able to make it this time."). The refusal target stays identical.

**disagreement.** Negative: dismissive, contemptuous contradiction ("That's flat-out wrong."). Neutral: plain contradiction ("I don't think that's right."). Positive: respectful disagreement with framing ("I see your point, but I'd respectfully push back — I don't think that holds."). The disagreed-with claim stays identical.

**criticism_or_feedback.** Negative: harsh, belittling judgment ("This report is sloppy and nowhere near good enough."). Neutral: plain feedback ("This report needs more work."). Positive: appreciative, face-saving feedback ("There's a lot of good material here; I think the report would benefit from some additional work in a few places."). The criticised aspect stays identical.

**bad_news_delivery.** Negative: blunt, dismissive delivery that shuts the listener down ("Your refund is denied — that's final."). Neutral: plain delivery with brief reason ("We can't approve your refund."). Positive: empathetic delivery with appreciation and apology ("I'm really sorry to have to tell you this, but we won't be able to approve your refund."). The bad news stays identical.

**apology.** Negative: grudging, dismissive non-apology that minimises the fault ("Yeah, I missed the deadline. It happens."). Neutral: plain apology with brief explanation ("I'm sorry I missed the deadline — I should have flagged it earlier."). Positive: full face-restoring apology with acknowledgment of impact ("I really do apologise for missing the deadline; I know it put extra pressure on the team and I should have raised it sooner."). The apologised-for action stays identical.

**complaint.** Negative: hostile, accusatory venting ("This is unacceptable — sort out the noise now."). Neutral: plain statement of the grievance ("The room next door is very noisy and it's keeping me awake."). Positive: courteous complaint with framing ("I'm sorry to raise this, but the neighbouring room has been quite noisy — would it be possible to help?"). The grievance stays identical.

**reminder.** Negative: nagging, exasperated prod ("You still haven't filed that expense report. Do it."). Neutral: plain reminder ("Just a reminder that the expense report is still outstanding."). Positive: gentle, considerate reminder ("Whenever you have a moment, it would be great to get the expense report filed — no rush."). The outstanding item stays identical.

**inquiry_sensitive.** Negative: blunt, prying question with no tact ("Why were you out all last week?"). Neutral: plain question ("Can I ask why you were away last week?"). Positive: tactful, considerate framing ("I hope everything's alright — if you're comfortable sharing, I wondered about last week."). The question's content stays identical.

**correction.** Negative: contemptuous put-down ("That's wrong — you're using the tool completely incorrectly."). Neutral: plain correction ("That's not quite right; the tool should be used this way."). Positive: gentle, face-saving correction ("Easy mistake — I think it actually works a little differently; may I show you?"). The corrected fact stays identical.

## Cue diversity
Within a level, do not lean on one marker. Spread across lexical courtesy markers, syntactic
indirectness, gratitude framing, imposition acknowledgment, and softened or impersonal phrasing.
"""


# --- scenario schema

# The trait-specific field every politeness scenario must carry: the fixed thing the act is
# about. Named once so the trait catalogue and the json_schema `required` list stay in sync.
POLITENESS_REQUIRED = ["speech_act_target"]

SCENARIO_SCHEMA: Dict[str, Any] = {
    "rubric_version": "v3",
    "dataset_version": "v3",
    # Keyed by trait so a future trait can be added by restoring a block here + in rubric.py.
    "traits": {
        "politeness": {
            "description": "Mitigation of face threat, deference, social consideration. Negative = impolite/rude; neutral = plain/matter-of-fact; positive = polite/mitigated.",
            "required_fields": POLITENESS_REQUIRED,
            "speech_acts": [
                {
                    "id": "request",
                    "description": "Asking the listener to do or provide something.",
                    "example_communicative_goal": "ask a colleague to send a file",
                    "extra_constraints": [
                        "Keep the requested action fixed across levels.",
                        "Do not change urgency or scope across levels.",
                    ],
                },
                {
                    "id": "refusal",
                    "description": "Declining a request, invitation, proposal, or offer made by the listener.",
                    "example_communicative_goal": "turn down a meeting invitation",
                    "extra_constraints": [
                        "The refusal target (what is being declined) must remain identical across levels.",
                        "Do not change the refusal into a partial acceptance or a counter-offer.",
                    ],
                },
                {
                    "id": "disagreement",
                    "description": "Expressing a contrary opinion, correction, or pushback on a claim.",
                    "example_communicative_goal": "push back on a colleague's analysis",
                    "extra_constraints": [
                        "The point of disagreement must remain identical across levels.",
                        "Do not soften disagreement into agreement at any level.",
                    ],
                },
                {
                    "id": "criticism_or_feedback",
                    "description": "Pointing out a problem with the listener's work, output, or behaviour.",
                    "example_communicative_goal": "tell a junior their report needs rework",
                    "extra_constraints": [
                        "The criticised aspect must remain identical across levels.",
                        "Do not turn criticism into pure praise.",
                    ],
                },
                {
                    "id": "bad_news_delivery",
                    "description": "Telling the listener something they will not want to hear (denial, rejection, negative outcome).",
                    "example_communicative_goal": "inform a customer their refund is denied",
                    "extra_constraints": [
                        "The bad news content must remain identical across levels.",
                        "Do not change a denial into an approval or a hedged maybe.",
                    ],
                },
                {
                    "id": "apology",
                    "description": "Acknowledging fault or expressing regret for a specific wrongdoing.",
                    "example_communicative_goal": "apologise for missing a deadline",
                    "extra_constraints": [
                        "The thing being apologised for must remain identical across levels.",
                        "Do not change which party is at fault.",
                    ],
                },
                {
                    "id": "complaint",
                    "description": "Voicing a grievance about a problem or situation affecting the speaker (service issue, environmental nuisance, missed commitment, etc.). Distinct from criticism_or_feedback, which targets the listener's work.",
                    "example_communicative_goal": "complain to a hotel manager about a noisy neighbouring room",
                    "extra_constraints": [
                        "The grievance (what is wrong) must remain identical across levels.",
                        "Do not turn the complaint into pure praise or into a refusal of service.",
                    ],
                },
                {
                    "id": "reminder",
                    "description": "Prompting the listener about an outstanding obligation, deadline, or commitment they owe.",
                    "example_communicative_goal": "remind a colleague that an expense report is overdue",
                    "extra_constraints": [
                        "The reminded item (what is outstanding) must remain identical across levels.",
                        "Do not change the reminder into a new request or an apology.",
                    ],
                },
                {
                    "id": "inquiry_sensitive",
                    "description": "Asking a personal, awkward, or socially delicate question.",
                    "example_communicative_goal": "ask a coworker why they missed work last week",
                    "extra_constraints": [
                        "The question's content must remain identical across levels.",
                        "Do not change the topic or scope of the inquiry.",
                    ],
                },
                {
                    "id": "correction",
                    "description": "Pointing out a factual or procedural mistake the listener made.",
                    "example_communicative_goal": "correct a junior's misuse of a tool",
                    "extra_constraints": [
                        "The corrected fact or step must remain identical across levels.",
                        "Do not change the correction into agreement or an unrelated tip.",
                    ],
                },
            ],
            "generation_constraints": [
                "Speech_act_target must be fixed across levels and paraphrases.",
                "Urgency and imposition must stay constant across levels.",
                "No insults, threats, or profanity even at the negative (impolite) pole.",
                "Diversify politeness across directness, deference, gratitude, softeners, and impersonal phrasing. Do not let any single forbidden_cue_token dominate one level.",
                "Every paraphrase, at every level, must satisfy every content_probe with the same expected_answer.",
                "Keep every paraphrase, at every level, close to target_word_count.",
            ],
        },
    },
    "json_schema": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "scenario_id": {"type": "string", "minLength": 3},
            "trait": {"const": "politeness"},
            "rubric_version": {"type": "string"},
            "dataset_version": {"type": "string"},
            "speech_act": {"type": "string", "minLength": 2},
            "domain": {"type": "string", "minLength": 2},
            "audience_relation": {"type": "string"},
            "communicative_goal": {"type": "string", "minLength": 5},
            "speech_act_target": {"type": "string", "minLength": 3},
            "content_probes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "minLength": 5},
                        "expected_answer": {"type": "string", "enum": ["yes", "no"]},
                    },
                    "required": ["question", "expected_answer"],
                },
            },
            "forbidden_cue_tokens": {"type": "array", "items": {"type": "string"}},
            "target_word_count": {"type": "integer", "minimum": 4, "maximum": 60},
        },
        "required": [
            "scenario_id",
            "trait",
            "speech_act",
            "domain",
            "audience_relation",
            "communicative_goal",
            "content_probes",
            "forbidden_cue_tokens",
            "target_word_count",
        ]
        + POLITENESS_REQUIRED,
    },
}


def make_validator() -> Draft202012Validator:
    return Draft202012Validator(SCENARIO_SCHEMA["json_schema"])


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
    17  # fixed so shuffles (speech-act queue, blind scoring) are reproducible
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
        traits=("politeness",),
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
        self.traits = list(traits)

        self.levels: List[str] = list(LEVELS)
        self._validator = make_validator()
        self.dataset_version = SCENARIO_SCHEMA["dataset_version"]
        self.rubric_version = SCENARIO_SCHEMA["rubric_version"]
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
        # (callers filter those out). Errors are printed; counts go into run.json.
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
            return SCENARIO_SCHEMA["traits"][trait]
        except KeyError as e:
            raise KeyError(
                f"Trait '{trait}' is not defined in the scenario schema."
            ) from e

    def _speech_acts_for(self, trait: str) -> List[Dict[str, Any]]:
        acts = self._trait_info(trait).get("speech_acts", [])
        if not acts:
            raise ValueError(f"No speech_acts configured for trait '{trait}'")
        return acts

    def _build_speech_act_queue(self, trait: str, n: int) -> List[str]:
        # Speech_act ids split as evenly as possible across n scenarios, then shuffled
        # with a fixed seed so a given trait always yields the same order.
        acts = [a["id"] for a in self._speech_acts_for(trait)]
        per = n // len(acts)
        rem = n - per * len(acts)
        queue: List[str] = []
        for a in acts:
            queue.extend([a] * per)
        for i in range(rem):
            queue.append(acts[i % len(acts)])
        rng = random.Random(self.random_seed + sum(ord(c) for c in trait))
        rng.shuffle(queue)
        return queue

    def _content_invariant(self, scenario: Dict[str, Any]) -> str:
        v = scenario.get("speech_act_target")
        return v.strip() if isinstance(v, str) else ""

    def _validate_scenario(self, scenario: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = [e.message for e in self._validator.iter_errors(scenario)]
        return (not errors, errors)

    def _seed_example(self, trait: str) -> Dict[str, Any]:
        # Shown to the generator as a worked example before any real scenarios exist.
        return {
            "scenario_id": f"{trait}-001",
            "trait": trait,
            "rubric_version": self.rubric_version,
            "dataset_version": self.dataset_version,
            "speech_act": "request",
            "domain": "workplace",
            "audience_relation": "peer",
            "communicative_goal": "ask a teammate to share the latest budget spreadsheet",
            "speech_act_target": "send the latest budget spreadsheet by end of day",
            "content_probes": [
                {
                    "question": "Is the speaker asking for the budget spreadsheet?",
                    "expected_answer": "yes",
                },
                {
                    "question": "Is the speaker offering to do something for the listener?",
                    "expected_answer": "no",
                },
            ],
            "forbidden_cue_tokens": ["please", "kindly"],
            "target_word_count": 18,
        }

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
        "speech_act",
        "domain",
        "audience_relation",
        "communicative_goal",
        "content_probes",
        "forbidden_cue_tokens",
        "target_word_count",
    ]

    def _scenario_user_prompt_batched(
        self,
        trait: str,
        speech_act_ids: List[str],
        previous_scenarios: List[Dict[str, Any]],
    ) -> str:
        # Each scenario's speech_act is assigned up front so the batch stays balanced.
        trait_info = self._trait_info(trait)
        shared_fields = list(self._SHARED_FIELDS)
        levels_str = ", ".join(self.levels)
        n = len(speech_act_ids)

        acts_by_id = {a["id"]: a for a in self._speech_acts_for(trait)}
        assignment_lines = "\n".join(
            f"  scenario {i + 1}: speech_act = {sid}"
            for i, sid in enumerate(speech_act_ids)
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
                block += "\n  per-act constraints:\n" + "\n".join(
                    f"    - {c}" for c in extra
                )
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
            f"Shared fields each scenario must fill: {shared_fields}.\n"
            "Hard requirements for every scenario object:\n"
            "  - scenario_id is a stable string (you may use 'auto' and the pipeline will reassign).\n"
            "  - trait must equal the trait above.\n"
            f"  - rubric_version = '{self.rubric_version}', dataset_version = '{self.dataset_version}'.\n"
            "  - content_probes is a list of 1-3 yes/no questions, each with expected_answer in [yes, no].\n"
            "  - target_word_count is an integer between 8 and 40.\n"
            "  - forbidden_cue_tokens is a list of surface tokens that must not dominate any single level.\n"
            "  - speech_act MUST equal the value assigned above for that scenario index.\n"
            "  - All natural-language fields must be written in English.\n"
            "  - Vary domains and audience_relations across the batch.\n"
        )
        if previous_scenarios:
            slim = [
                {
                    "scenario_id": s.get("scenario_id"),
                    "speech_act": s.get("speech_act"),
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
        remaining_queue = self._build_speech_act_queue(trait, total_n)
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
                remaining_queue = self._build_speech_act_queue(
                    trait, total_n - len(cleaned)
                )
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
                err_msg = str(last_err) if last_err else ""
                print(
                    f"  ! batch {batch_index} ({batch_acts}) skipped after {max_batch_retries} retries."
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
                if i < len(batch_acts):
                    row["speech_act"] = batch_acts[i]
                next_idx = len(cleaned) + 1
                act_slug = slugify(str(row.get("speech_act", "act")))
                row["scenario_id"] = f"{trait}-{act_slug}-{next_idx:03d}"
                cp = row.get("content_probes")
                if isinstance(cp, dict):
                    row["content_probes"] = [cp]
                row.setdefault("forbidden_cue_tokens", [])

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
            f"Rubric:\n{CONSTRUCT_RUBRIC}\n\n"
            "CRITICAL: You MUST return only valid JSON. Do not include any text before or after the JSON. "
            "Do not add explanations, preambles, or comments. The entire response must be parseable as JSON."
        )

    def _base_sentences_user_prompt(self, trait: str, scenario: Dict[str, Any]) -> str:
        levels_str = ", ".join(self.levels)
        invariant = self._content_invariant(scenario)
        speech_act = scenario.get("speech_act", "")
        return (
            f"Create one {len(self.levels)}-level set of base sentences for trait '{trait}', "
            f"speech_act '{speech_act}'.\n"
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
        forbidden_cue_tokens: Optional[List[str]] = None,
        prior_texts_by_level: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        avoid = sorted(used_cue_families - {""})
        sentences = base_obj["base_sentences"]
        n_levels = len(sentences)
        target_len = base_obj.get("scenario", {}).get("target_word_count", 18)
        prompt = (
            f"Given these canonical {n_levels}-level base sentences:\n"
            f"{json.dumps(sentences, ensure_ascii=False, indent=2)}\n\n"
            "Write every paraphrase in fluent, natural English.\n"
            "Generate one paraphrase per level.\n"
            "For each paraphrase, provide: level, cue_family, text.\n"
            "DIVERSITY — hard requirement: each paraphrase must be a genuinely distinct "
            "sentence — its own syntactic structure and its own cue family (lexical "
            "marker, syntactic framing, gratitude framing, evidential framing, "
            "indirectness, modal framing). Do not reuse the structure of any other "
            "paraphrase for this scenario. The polite level especially must not collapse "
            "onto a single 'I appreciate your X, but Y' scaffold.\n"
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
        if forbidden_cue_tokens:
            prompt += (
                "\n\nDo not lean on these tokens to mark politeness (they are scenario-specific "
                f"shortcuts to avoid as level cues): {sorted(set(forbidden_cue_tokens))}."
            )
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
            forbidden = base_obj.get("scenario", {}).get("forbidden_cue_tokens") or []
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
                            base_obj, avoid, forbidden, prior_by_level
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
            "You verify CONTENT PRESERVATION for a controlled politeness dataset. "
            "Each text expresses a fixed underlying content (the 'invariant content'); "
            "politeness and tone vary deliberately and MUST NOT affect your judgement. "
            "For each text, score in [0.0, 1.0] how faithfully it preserves the invariant "
            "content: 1.0 = same content, intent, and polarity; lower = content drift, "
            "added/dropped/flipped meaning, or a different request/answer. Do NOT reward or "
            "penalise how polite, rude, or fluent a text is — only content fidelity.\n\n"
            "CRITICAL: Return ONLY valid JSON. No prose, no markdown."
        )

    def _context_user_prompt(
        self,
        invariant: str,
        content_probes: List[Dict[str, Any]],
        items: List[Tuple[str, str]],
    ) -> str:
        probes_str = (
            json.dumps(content_probes, ensure_ascii=False) if content_probes else "[]"
        )
        items_json = json.dumps(
            [{"paraphrase_id": pid, "text": t} for pid, t in items],
            ensure_ascii=False,
            indent=2,
        )
        return (
            f"Invariant content every text must preserve:\n{invariant}\n\n"
            f"Content probes — every text must satisfy each with the stated expected_answer:\n{probes_str}\n\n"
            f"Texts:\n{items_json}\n\n"
            'Return JSON with key "scores": a list of {"paraphrase_id": "<id>", "score": <float 0..1>}, '
            "one entry per input id. No other keys."
        )

    def _context_scores(
        self,
        trait: str,
        bundle: List[Dict[str, Any]],
        content_probes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        # paraphrase_id -> faithfulness score in [0,1]; 0.0 if the judge omits an id.
        if not bundle:
            return {}
        invariant = bundle[0].get("invariant_content", "")
        items = [(r["paraphrase_id"], r["text"]) for r in bundle]
        system = self._context_system_prompt(trait)
        user = self._context_user_prompt(invariant, content_probes or [], items)
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
        scenario: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        # Tag every paraphrase with its scores, `passed`, and the `failed_check` that dropped it.
        min_score = self.min_acceptance_score
        min_gap = self.intensity_min_gap
        max_ratio = self.max_length_ratio
        content_probes = (scenario or {}).get("content_probes", [])

        failed_check: Dict[str, str] = {}

        # criterion 1: context (per-sentence faithfulness)
        ctx_scores = self._context_scores(trait, bundle, content_probes)
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
        scenarios_by_id: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        # Judge each scenario's paraphrases and tag every row with `passed` + `failed_check`.
        # Returns all rows (the unfiltered set); the caller keeps the passing ones.
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in paraphrases:
            grouped.setdefault(row["scenario_id"], []).append(row)
        bundles = sorted(grouped.items())

        def _one(item: Tuple[str, List[Dict[str, Any]]]):
            sid, bundle = item
            return self._judge_and_filter_bundle(
                trait, bundle, scenarios_by_id.get(sid)
            )

        results = [r for r in self._run_items(bundles, _one) if r is not None]
        judged = [row for bundle_rows in results for row in bundle_rows]
        n_passed = sum(1 for r in judged if r["passed"])
        print(
            f"Judged {len(judged)} rows; {n_passed} passed, {len(judged) - n_passed} dropped."
        )
        return judged


# --- entry point

DEFAULT_MODELS = {
    # Generator: creative, follows complex JSON schemas. Routed through google-vertex.
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
    # Judge: a different family from the generator to limit leakage. Does both the label-aware
    # content check and the label-blind intensity rating.
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


# Rows kept in filtered.jsonl (the dataset); diagnostics (scores, passed, failed_check) stay in unfiltered.
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


def generate_prompts(
    out_dir,
    *,
    n_scenarios: int = 300,
    paraphrases_per_level: int = 3,
    min_acceptance_score: float = 0.70,
    intensity_min_gap: float = 0.10,
    max_length_ratio: float = 1.15,
    max_workers: int = 3,
    models: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate the dataset into out_dir (created if needed) and return it.

    Writes scenarios.jsonl, unfiltered.jsonl, filtered.jsonl, run.json into out_dir. The caller
    chooses the directory (e.g. data/<timestamp>). Needs OPENROUTER_API_KEY (a repo-root .env is
    loaded automatically).
    """
    models = models or DEFAULT_MODELS
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
    )
    trait = pipe.traits[
        0
    ]  # politeness-only; a second trait would need per-trait filenames
    scenarios = pipe.make_scenarios(trait)
    base = pipe.make_base_sentences(trait, scenarios)
    paraphrases = pipe.make_paraphrases(trait, base)
    judged = pipe.judge_and_filter(
        trait, paraphrases, {s["scenario_id"]: s for s in scenarios}
    )
    passed = [r for r in judged if r["passed"]]

    jsonl_write(out / "unfiltered.jsonl", judged)
    jsonl_write(
        out / "filtered.jsonl",
        [{k: r[k] for k in _DATASET_FIELDS if k in r} for r in passed],
    )

    by_failed: Dict[str, int] = {}
    for r in judged:
        if r["failed_check"]:
            by_failed[r["failed_check"]] = by_failed.get(r["failed_check"], 0) + 1
    by_act: Dict[str, int] = {}
    for s in scenarios:
        by_act[s.get("speech_act", "?")] = by_act.get(s.get("speech_act", "?"), 0) + 1
    manifest = {
        "timestamp": out.name,
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
            "by_speech_act": by_act,
        },
    }
    with (out / "run.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Done. {len(passed)} accepted paraphrases → {out}")
    return out
