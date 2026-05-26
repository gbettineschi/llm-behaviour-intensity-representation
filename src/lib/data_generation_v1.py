from __future__ import annotations

import itertools
import json
import math
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from jsonschema import Draft202012Validator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from dotenv import load_dotenv
load_dotenv()

try:
    import litellm
    litellm.suppress_debug_info = True
    from litellm import completion
except Exception:  # pragma: no cover
    completion = None

from lib.data_typing import LEVELS, TRAITS

SCENARIO_SCHEMA: Dict[str, Any] = {
    "rubric_version": "construct_rubric.md@v3",
    "dataset_version": "v3",
    "traits": {
        "politeness": {
            "description": "Mitigation of face threat, deference, social consideration. Negative = impolite/rude; neutral = plain/matter-of-fact; positive = polite/mitigated.",
            "required_fields": ["speech_act_target", "imposition_level", "urgency_level", "social_distance"],
            "field_descriptions": {
                "speech_act_target": "The exact action / refusal target / news / criticism etc. that must remain invariant across levels.",
                "imposition_level": "low | medium | high — how costly the act is for the listener",
                "urgency_level": "low | medium | high — must remain constant across levels",
                "social_distance": "close | moderate | distant",
            },
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
                "Urgency_level and imposition scope must be constant across levels.",
                "No insults, threats, or profanity even at the negative (impolite) pole.",
                "Diversify politeness realisation across: directness, deference, gratitude framing, softeners, impersonalisation. Do not let any single token in forbidden_cue_tokens dominate one level.",
                "All paraphrases at all levels must satisfy every content_probe with the same expected_answer.",
                "Word count of every paraphrase must be within ±length_tolerance_pct of target_word_count.",
            ],
        },
        "hedging_confidence": {
            "description": "Speaker commitment to a proposition. Negative = tentative/hedged; neutral = balanced; positive = strongly committed/direct.",
            "required_fields": ["proposition", "evidence_state", "answer_type", "consequence_sensitivity"],
            "field_descriptions": {
                "proposition": "The exact proposition that must stay fixed across levels.",
                "evidence_state": "Brief description of evidence basis; the *amount* of evidence claimed must not change across levels.",
                "answer_type": "assertion | recommendation | explanation | forecast | answer | estimate",
                "consequence_sensitivity": "low | medium | high",
            },
            "speech_acts": [
                {
                    "id": "factual_assertion",
                    "description": "Stating a proposition as true.",
                    "example_communicative_goal": "state that the new policy reduces costs",
                    "extra_constraints": ["Polarity of the assertion must not flip across levels."],
                },
                {
                    "id": "recommendation",
                    "description": "Advising a particular course of action.",
                    "example_communicative_goal": "advise switching vendors",
                    "extra_constraints": ["The recommended action must remain identical across levels."],
                },
                {
                    "id": "forecast",
                    "description": "Predicting a future outcome.",
                    "example_communicative_goal": "predict next quarter's churn will rise",
                    "extra_constraints": [
                        "The predicted outcome and its direction must remain identical.",
                        "Do not introduce numerical probabilities unless the scenario explicitly allows them.",
                    ],
                },
                {
                    "id": "causal_explanation",
                    "description": "Explaining why something happened.",
                    "example_communicative_goal": "explain why the deployment failed",
                    "extra_constraints": ["The causal claim must remain identical across levels."],
                },
                {
                    "id": "yes_no_answer",
                    "description": "Answering a direct yes/no question.",
                    "example_communicative_goal": "answer whether the data supports the hypothesis",
                    "extra_constraints": ["The polarity of the answer (yes/no) must not flip across levels."],
                },
                {
                    "id": "estimation",
                    "description": "Giving a qualitative or coarse quantitative estimate.",
                    "example_communicative_goal": "estimate how long the migration will take",
                    "extra_constraints": [
                        "The central estimate (rough magnitude) must remain identical across levels.",
                        "Hedging modifies confidence in the estimate, not its value.",
                    ],
                },
                {
                    "id": "comparison",
                    "description": "Comparing two options, products, or approaches on a stated dimension.",
                    "example_communicative_goal": "compare two cloud providers on reliability",
                    "extra_constraints": [
                        "The compared items and the dimension of comparison must remain identical across levels.",
                        "Do not flip which option is judged better across levels.",
                    ],
                },
                {
                    "id": "diagnosis",
                    "description": "Identifying the underlying cause of an observed problem.",
                    "example_communicative_goal": "diagnose why the dashboard is slow",
                    "extra_constraints": [
                        "The diagnosed cause must remain identical across levels.",
                        "Hedging modifies confidence in the diagnosis, not the cause itself.",
                    ],
                },
                {
                    "id": "generalization",
                    "description": "Drawing a general claim from specific cases or evidence.",
                    "example_communicative_goal": "generalise from recent incidents to a team-wide pattern",
                    "extra_constraints": [
                        "The general claim and its scope must remain identical across levels.",
                        "Do not narrow or widen the population covered across levels.",
                    ],
                },
                {
                    "id": "evaluation",
                    "description": "Judging the quality, suitability, or correctness of something.",
                    "example_communicative_goal": "evaluate whether a draft proposal is ready",
                    "extra_constraints": [
                        "The evaluated object and the direction of judgement must remain identical across levels.",
                        "Hedging modifies confidence in the judgement, not its direction.",
                    ],
                },
            ],
            "generation_constraints": [
                "Proposition must be fixed across levels and paraphrases.",
                "Do not change the answer polarity or recommended action.",
                "Do not introduce numerical probabilities unless explicitly allowed.",
                "Use ordinary natural-language hedging and commitment cues.",
                "Diversify cues across: lexical hedges, evidential framing, modal verbs, syntactic structure. Avoid letting forbidden_cue_tokens become level markers.",
                "All paraphrases at all levels must satisfy every content_probe with the same expected_answer.",
                "Word count of every paraphrase must be within ±length_tolerance_pct of target_word_count.",
            ],
        },
    },
    "json_schema": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "scenario_id": {"type": "string", "minLength": 3},
            "trait": {"type": "string", "enum": ["politeness", "hedging_confidence"]},
            "rubric_version": {"type": "string"},
            "dataset_version": {"type": "string"},
            "split": {"type": "string", "enum": ["train", "val", "test"]},
            "speech_act": {"type": "string", "minLength": 2},
            "domain": {"type": "string", "minLength": 2},
            "topic_cluster": {"type": "string", "minLength": 2},
            "audience_relation": {"type": "string"},
            "register": {"type": "string", "minLength": 2},
            "communicative_goal": {"type": "string", "minLength": 5},
            "speech_act_target": {"type": "string", "minLength": 3},
            "proposition_or_request": {"type": "string"},
            "language": {"type": "string", "enum": ["en", "it"]},
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
            "allowed_named_entities": {"type": "array", "items": {"type": "string"}},
            "forbidden_content_changes": {"type": "array", "items": {"type": "string"}},
            "forbidden_cue_tokens": {"type": "array", "items": {"type": "string"}},
            "forbidden_lexical_shortcuts": {"type": "array", "items": {"type": "string"}},
            "target_word_count": {"type": "integer", "minimum": 4, "maximum": 60},
            "length_tolerance_pct": {"type": "integer", "minimum": 5, "maximum": 50},
            "imposition_level": {"type": "string", "minLength": 2},
            "urgency_level": {"type": "string", "minLength": 2},
            "social_distance": {"type": "string", "minLength": 2},
            "proposition": {"type": "string"},
            "evidence_state": {"type": "string"},
            "answer_type": {"type": "string", "enum": ["assertion", "recommendation", "explanation", "forecast", "answer", "estimate"]},
            "consequence_sensitivity": {"type": "string", "minLength": 2},
            "notes": {"type": "string"},
        },
        "required": [
            "scenario_id", "trait", "speech_act", "domain", "topic_cluster",
            "audience_relation", "register", "communicative_goal",
            "content_probes", "forbidden_cue_tokens", "target_word_count",
            "language",
        ],
        "allOf": [
            {
                "if": {"properties": {"trait": {"const": "politeness"}}, "required": ["trait"]},
                "then": {"required": ["speech_act_target", "imposition_level", "urgency_level", "social_distance"]},
            },
            {
                "if": {"properties": {"trait": {"const": "hedging_confidence"}}, "required": ["trait"]},
                "then": {"required": ["proposition", "evidence_state", "answer_type", "consequence_sensitivity"]},
            },
        ],
    },
}

CONSTRUCT_RUBRIC = """\
# Construct Rubric (Signed 3-Point Benchmark) — v3

This rubric is written for a **representation-geometry benchmark**, not a generic classification dataset. The key requirement is that all levels within a ladder preserve the same underlying content while varying only the target trait. The rubric is multi–speech-act: each trait is realised across several speech acts, and intensity is defined consistently across all of them.

Each trait is realised on a **signed three-point scale**: a **negative** pole, a **neutral** centre, and a **positive** pole. The neutral point is a genuine zero — neither pole's markers are present — and the negative and positive points are departures of comparable magnitude in opposite directions.

---

## Trait A: Politeness

### Core construct
Politeness is the degree to which an utterance mitigates face threat, softens imposition, signals respect or deference, and frames the act in a socially considerate way. It applies to any speech act whose surface form is socially loaded.

### Scope
Politeness is realised across the following speech acts (same construct, different acts):
- **request** — asking for an action or item
- **refusal** — declining an offer/request
- **disagreement** — pushing back on a claim
- **criticism_or_feedback** — pointing out a problem with someone's work
- **bad_news_delivery** — telling the listener something they will not want to hear
- **apology** — acknowledging fault or expressing regret

### What must remain constant across levels (per scenario)
- the speech-act target (what is requested / refused / criticised / forecast / etc.)
- named entities, dates, deadlines
- core practical intent and truth conditions
- the polarity of the act (a refusal stays a refusal; an apology stays an apology)

### What may vary
- directness vs indirectness
- mitigation and softening
- deference and respect markers
- gratitude or appreciation framing
- imposition acknowledgment
- syntactic form

### What must not vary
- urgency or scope
- amount of work / cost imposed
- whether the act is optional
- sentiment unrelated to the act
- sentence length — positive paraphrases must not be substantially longer than negative ones

### Signed three-point scale (universal across speech acts)

#### Negative — impolite
- the act is performed rudely: dismissive, curt, grudging, impatient, or condescending
- it MUST read as clearly impolite to an ordinary reader — actively face-threatening, not merely plain or brief
- a plain, inoffensive, unadorned sentence is NOT negative; that is neutral. The negative point must carry visible discourtesy
- impoliteness is carried by tone and framing — brusqueness, irritation, dismissiveness, the absence of any acknowledgment — never by changing the underlying content
- NEVER profane, threatening, slur-based, or abusive: rudeness comes from curt, dismissive framing, not from profanity or attacks on the person (those would be lexical confounds)
- the same length as the neutral and positive realisations — do not make the negative point shorter

#### Neutral — neither polite nor impolite
- plain, matter-of-fact realisation of the act
- no courtesy markers and no rudeness markers
- even-toned and transactional; the unmarked default

#### Positive — polite
- clearly respectful and mitigated
- strong face-saving framing
- appreciation, deference, imposition acknowledgment without changing what is being said
- politeness is in framing, not verbosity — do not pad with extra clauses; keep it close in length to the neutral and negative realisations

### Per-speech-act guidance

**request.** Negative: rude, impatient demand ("Just send me the file already."). Neutral: plain direct request with no softeners or brusqueness ("Can you send me the file?"). Positive: deferential framing with gratitude or imposition acknowledgment ("I'd really appreciate it if you could send me the file when you get a chance."). The requested action stays identical.

**refusal.** Negative: abrupt, dismissive decline with no acknowledgment of the offer ("No. I'm not doing that."). Neutral: plain decline with brief reason ("I won't be able to make it."). Positive: appreciative refusal acknowledging the offer and apologising ("Thank you so much for the invitation — I'm afraid I won't be able to make it this time."). The refusal target stays identical.

**disagreement.** Negative: dismissive, contemptuous contradiction ("That's flat-out wrong."). Neutral: plain contradiction ("I don't think that's right."). Positive: respectful disagreement with framing ("I see your point, but I'd respectfully push back — I don't think that holds."). The disagreed-with claim stays identical.

**criticism_or_feedback.** Negative: harsh, belittling judgment ("This report is sloppy and nowhere near good enough."). Neutral: plain feedback ("This report needs more work."). Positive: appreciative, face-saving feedback ("There's a lot of good material here; I think the report would benefit from some additional work in a few places."). The criticised aspect stays identical.

**bad_news_delivery.** Negative: blunt, dismissive delivery that shuts the listener down ("Your refund is denied — that's final."). Neutral: plain delivery with brief reason ("We can't approve your refund."). Positive: empathetic delivery with appreciation and apology ("I'm really sorry to have to tell you this, but we won't be able to approve your refund."). The bad news stays identical.

**apology.** Negative: grudging, dismissive non-apology that minimises the fault ("Yeah, I missed the deadline. It happens."). Neutral: plain apology with brief explanation ("I'm sorry I missed the deadline — I should have flagged it earlier."). Positive: full face-restoring apology with acknowledgment of impact ("I really do apologise for missing the deadline; I know it put extra pressure on the team and I should have raised it sooner."). The apologised-for action stays identical.

### Cue-diversity requirement
For a given level, do not rely on one marker repeatedly. Spread realisations across:
- lexical courtesy markers
- syntactic indirectness
- gratitude framing
- imposition acknowledgment
- depersonalised or softened phrasing

---

## Trait B: Hedging / Linguistic Confidence

### Core construct
Hedging/confidence is the degree of speaker commitment to a proposition. The negative pole expresses uncertainty or tentativeness; the positive pole expresses strong commitment.

### Scope
Hedging is realised across the following speech acts:
- **factual_assertion** — stating something as true
- **recommendation** — advising a course of action
- **forecast** — predicting a future outcome
- **causal_explanation** — explaining why something happened
- **yes_no_answer** — answering a yes/no question
- **estimation** — giving a qualitative or coarse quantitative estimate

### What must remain constant across levels
- the proposition / recommended action / forecast outcome / causal claim / yes-or-no polarity / central estimate
- named entities and factual content
- the amount of evidence claimed (unless the scenario explicitly varies it — it should not)

### What may vary
- strength of commitment
- epistemic stance and evidential framing
- hedge markers and modal verbs
- discourse framing of uncertainty

### What must not vary
- the proposition itself
- polarity or factual answer
- specificity of the recommendation
- whether numerical probabilities are introduced (forbidden unless explicitly allowed)

### Signed three-point scale (universal across speech acts)

#### Negative — hedged / low confidence
- tentative stance, explicit uncertainty
- usable but clearly cautious
- "I think it might…", "based on what I can tell…", "it's possible that…"

#### Neutral — balanced commitment
- ordinary qualified claim, neither markedly hedged nor markedly assertive
- the unmarked default
- "it seems…", "I think…", "it looks like…"

#### Positive — confident / minimally hedged
- strong commitment, direct statement
- still natural, not boastful or aggressive
- "it is…", "I'm confident that…", a bare assertion

### Per-speech-act guidance

**factual_assertion.** Negative: "It might be that the policy reduces costs." Neutral: "It looks like the policy reduces costs." Positive: "The policy reduces costs." Polarity stays identical.

**recommendation.** Negative: "You might want to consider switching vendors." Neutral: "I'd suggest switching vendors." Positive: "You should switch vendors." The recommended action stays identical.

**forecast.** Negative: "Churn could rise next quarter." Neutral: "Churn is likely to rise next quarter." Positive: "Churn will rise next quarter." Direction of the prediction stays identical; do not insert numbers.

**causal_explanation.** Negative: "The deployment may have failed because of the config change." Neutral: "The deployment likely failed because of the config change." Positive: "The deployment failed because of the config change." The causal claim stays identical.

**yes_no_answer.** Negative: "I think the answer is probably yes, though I'm not certain." Neutral: "I'd say yes." Positive: "Yes." Polarity stays identical.

**estimation.** Negative: "It might take roughly two weeks, give or take." Neutral: "It'll likely take about two weeks." Positive: "It'll take two weeks." The central estimate stays identical; only confidence in it varies.

### Cue-diversity requirement
Diversify across:
- lexical hedges
- evidential framing
- modal verbs
- discourse-softening phrases
- syntax and clause structure

Avoid mapping one point to one token (e.g. "maybe" ↔ negative only).

---

## Universal acceptance criteria

Every final item must satisfy all of the following:

1. **Content preservation**: same core content (target / proposition / answer / forecast / cause / estimate) across levels. Every content_probe must have the same expected_answer at every level.
2. **Ordered intensity**: human or validated judge ordering matches negative < neutral < positive for the target trait.
3. **Naturalness**: each sentence is fluent and plausible in ordinary usage.
4. **No overt artifacts**: no single cue or template uniquely identifies one level across the dataset.
5. **Paraphrase diversity**: at least two distinct phrasings per level, not near-duplicates.
6. **No domain leakage**: scenario metadata, topic, named entities, or speech act do not uniquely determine the level.
7. **Length balance**: paraphrase length must not correlate with level. All paraphrases within a scenario must be within ±length_tolerance_pct of target_word_count.
"""


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


class ShortcutTracker:
    """Streaming per-level token counter for cross-bundle shortcut detection.

    Used live during paraphrase generation and judging: as each bundle is
    processed, its texts are added to the tracker, and the next bundle is
    given the top tokens that *currently* distinguish each level so the
    generator can avoid them and the judge can reject paraphrases that lean
    on them.
    """

    _STOP = frozenset(
        "the a an and or but if of to in on at for with is are was were be "
        "been being have has had do does did this that these those it its as "
        "by from not no so we i you he she they them us our your his her their "
        "my me into out over under than then there here will would can could "
        "should may might must shall about up down".split()
    )

    def __init__(self, levels: List[str]):
        self.levels = list(levels)
        self.counts: Dict[str, Dict[str, int]] = {l: {} for l in self.levels}
        self._lock = threading.Lock()

    @classmethod
    def _tokens(cls, text: str) -> List[str]:
        return [
            t for t in re.findall(r"[a-z']+", text.lower())
            if len(t) > 2 and t not in cls._STOP
        ]

    def add(self, level: str, text: str) -> None:
        toks = self._tokens(text)
        if not toks:
            return
        with self._lock:
            d = self.counts.setdefault(level, {})
            for t in toks:
                d[t] = d.get(t, 0) + 1

    def add_rows(self, rows: Iterable[Dict[str, Any]]) -> None:
        for r in rows:
            lvl, txt = r.get("level"), r.get("text")
            if lvl and txt:
                self.add(lvl, txt)

    def top_per_level(self, k: int = 10, min_count: int = 2) -> Dict[str, List[str]]:
        """Return top-K log-odds tokens per level, only those biased toward that level."""
        with self._lock:
            snap = {L: dict(d) for L, d in self.counts.items()}
        result: Dict[str, List[str]] = {}
        for L in self.levels:
            level_counts = snap.get(L, {})
            others_counts: Dict[str, int] = {}
            for OL, d in snap.items():
                if OL == L:
                    continue
                for t, c in d.items():
                    others_counts[t] = others_counts.get(t, 0) + c
            scored = []
            for t, c in level_counts.items():
                if c < min_count:
                    continue
                o = others_counts.get(t, 0)
                score = math.log((c + 1) / (o + 1))
                if score <= 0:
                    continue
                scored.append((score, c, t))
            scored.sort(reverse=True)
            result[L] = [t for _, _, t in scored[:k]]
        return result


@dataclass
class ModelSpec:
    model: str
    family: str
    temperature: float = 0.0
    max_output_tokens: int = 1200
    litellm_kwargs: Dict[str, Any] = field(default_factory=dict)


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
                **self.spec.litellm_kwargs,
            )
            msg = resp["choices"][0]["message"]
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            # Reasoning models (e.g. gpt-5-nano) sometimes put the JSON in
            # alternate fields and leave content as None. Try common fallbacks.
            if not content:
                for alt in ("reasoning_content", "reasoning", "text"):
                    v = (
                        msg.get(alt)
                        if isinstance(msg, dict)
                        else getattr(msg, alt, None)
                    )
                    if isinstance(v, str) and v.strip():
                        content = v
                        break
            try:
                return self._extract_json(content)
            except json.JSONDecodeError as exc:
                last_error = exc
                if attempt < 2:
                    safe = content if isinstance(content, str) else ""
                    preview = safe[:200] if len(safe) <= 200 else safe[:197] + "..."
                    print(f"  [retry {attempt+1}/3] JSON parse failed. Response preview: {preview!r}")
        if last_error is not None:
            raise last_error
        raise RuntimeError("call_json failed without producing an error")


class Pipeline:
    def __init__(self, config: Dict[str, Any], model_override: Optional[str] = None):
        self.config = config
        self.root = Path(__file__).parent.parent.parent
        self.output_root = self.root / self.config.get("output_dir", "data/v1")
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

        self.scenario_schema = SCENARIO_SCHEMA
        self._validator = Draft202012Validator(self.scenario_schema["json_schema"])
        self.dataset_version = str(self.scenario_schema.get("dataset_version", "v3"))
        self.rubric_version = str(self.scenario_schema.get("rubric_version", "construct_rubric.md@v3"))
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

    def _build_language_queue(self, trait: str, n: int) -> List[str]:
        """Return a balanced, deterministic list of language ids of length n."""
        langs = list(self.LANGUAGES)
        per = n // len(langs)
        rem = n - per * len(langs)
        queue: List[str] = []
        for lang in langs:
            queue.extend([lang] * per)
        for i in range(rem):
            queue.append(langs[i % len(langs)])
        rng = random.Random(self.random_seed + sum(ord(c) for c in trait) + 17)
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
                "language": "en",
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
                "language": "en",
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
            "language": "en",
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

    # Names of fields that every scenario must populate regardless of trait.
    # (Replaces a stale reference to scenario_schema["definitions"]["shared_fields"]
    # that crashed make_scenarios on a fresh run.)
    _SHARED_FIELDS = [
        "scenario_id", "trait", "rubric_version", "dataset_version",
        "speech_act", "domain", "topic_cluster", "audience_relation",
        "register", "communicative_goal", "content_probes",
        "forbidden_cue_tokens", "target_word_count", "length_tolerance_pct",
        "language",
    ]

    LANGUAGES = ["en", "it"]
    _LANGUAGE_LABELS = {"en": "English", "it": "Italian"}

    def _scenario_user_prompt_batched(
        self,
        trait: str,
        speech_act_ids: List[str],
        previous_scenarios: List[Dict[str, Any]],
        language_ids: Optional[List[str]] = None,
    ) -> str:
        """Build the per-batch prompt. Each scenario's assigned speech_act is fixed up front."""
        trait_info = self._trait_info(trait)
        shared_fields = list(self._SHARED_FIELDS)
        levels_str = ", ".join(self.levels)
        n = len(speech_act_ids)
        if language_ids is None or len(language_ids) != n:
            # Fallback: alternate en/it deterministically.
            language_ids = [self.LANGUAGES[i % len(self.LANGUAGES)] for i in range(n)]

        # Build per-scenario assignment lines and per-act guidance blocks.
        acts_by_id = {a["id"]: a for a in self._speech_acts_for(trait)}
        assignment_lines = "\n".join(
            f"  scenario {i + 1}: speech_act = {sid}, language = {language_ids[i]} ({self._LANGUAGE_LABELS.get(language_ids[i], language_ids[i])})"
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
            f"Shared fields each scenario must fill: {shared_fields}.\n"
            "Hard requirements for every scenario object:\n"
            "  - scenario_id is a stable string (you may use 'auto' and the pipeline will reassign).\n"
            "  - trait must equal the trait above.\n"
            f"  - rubric_version = '{self.rubric_version}', dataset_version = '{self.dataset_version}'.\n"
            "  - content_probes is a list of 1-3 yes/no questions, each with expected_answer in [yes, no].\n"
            "  - target_word_count is an integer between 8 and 40.\n"
            "  - length_tolerance_pct is an integer (default 20).\n"
            "  - forbidden_cue_tokens is a list of surface tokens that must not dominate any single level.\n"
            "  - speech_act MUST equal the value assigned above for that scenario index.\n"
            "  - language MUST equal the value assigned above for that scenario index. "
            "Use 'en' or 'it' verbatim. The communicative_goal and all natural-language fields "
            "in that scenario must be written in the assigned language. Italian scenarios must use "
            "culturally natural Italian framing (e.g. tu/Lei distinctions, Italian workplace norms), "
            "not transliterated English.\n"
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

        # Parallel queue for languages, balanced en/it across the run.
        full_lang_queue = self._build_language_queue(trait, total_n)
        remaining_lang_queue = full_lang_queue[len(cleaned):]
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
            if not remaining_lang_queue:
                needed = total_n - len(cleaned)
                remaining_lang_queue = [
                    self.LANGUAGES[i % len(self.LANGUAGES)] for i in range(needed)
                ]
            batch_acts = remaining_queue[:batch_size]
            remaining_queue = remaining_queue[batch_size:]
            batch_langs = remaining_lang_queue[:batch_size]
            remaining_lang_queue = remaining_lang_queue[batch_size:]
            # Pad languages if shorter than acts (defensive).
            while len(batch_langs) < len(batch_acts):
                batch_langs.append(self.LANGUAGES[len(batch_langs) % len(self.LANGUAGES)])
            batch_n = len(batch_acts)
            attempted_batches += 1

            user = self._scenario_user_prompt_batched(trait, batch_acts, cleaned, batch_langs)
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
                # Assign / overwrite speech_act and language from the queues (defensive).
                if i < len(batch_acts):
                    row["speech_act"] = batch_acts[i]
                if i < len(batch_langs):
                    row["language"] = batch_langs[i]
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
        rubric = CONSTRUCT_RUBRIC
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
        language = scenario.get("language", "en")
        language_label = self._LANGUAGE_LABELS.get(language, language)
        return (
            f"Create one {len(self.levels)}-level canonical ladder for trait '{trait}', "
            f"speech_act '{speech_act}'.\n"
            f"Scenario:\n{json.dumps(scenario, ensure_ascii=False, indent=2)}\n\n"
            f"LANGUAGE — hard requirement: every ladder rung MUST be written in {language_label} "
            f"(language code '{language}'). Do not mix languages. Use culturally natural phrasing "
            "for that language; do not produce a literal translation of an English template.\n"
            f"Levels must be {levels_str}.\n"
            f"The invariant content '{invariant}' must remain identical in meaning across levels.\n"
            f"Vary only the trait intensity. Write all {len(self.levels)} rungs at about "
            f"{scenario.get('target_word_count', 18)} words each — equal length across "
            "levels, so sentence length never cues the trait.\n"
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
            "Paraphrases must not be systematically longer at any point on the scale than at the others.\n\n"
            "CRITICAL: You MUST return only valid JSON. Do not include any text before or after the JSON. "
            "Do not add explanations, preambles, or comments. The entire response must be parseable as JSON."
        )

    def _paraphrase_user_prompt(
        self,
        ladder_obj: Dict[str, Any],
        per_level: int,
        used_cue_families: set[str],
        forbidden_tokens_by_level: Optional[Dict[str, List[str]]] = None,
        prior_texts_by_level: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        avoid = sorted(used_cue_families - {""})
        n_levels = len(ladder_obj["ladder"])
        target_len = ladder_obj.get("scenario", {}).get("target_word_count", 18)
        language = ladder_obj.get("scenario", {}).get("language", "en")
        language_label = self._LANGUAGE_LABELS.get(language, language)
        prompt = (
            f"Given this canonical {n_levels}-level ladder:\n"
            f"{json.dumps(ladder_obj['ladder'], ensure_ascii=False, indent=2)}\n\n"
            f"LANGUAGE — hard requirement: every paraphrase MUST be written in "
            f"{language_label} (language code '{language}'). Do not mix languages. Use "
            "culturally natural phrasing; do not produce a literal translation of an "
            "English template.\n"
            f"Generate {per_level} paraphrase(s) per level.\n"
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
            prompt += (
                f"\n\nAlready used in this dataset — vary away from these cue families: {avoid}."
            )
        if forbidden_tokens_by_level:
            forbidden_lines = [
                f"  {lvl}: {forbidden_tokens_by_level[lvl]}"
                for lvl in self.levels
                if forbidden_tokens_by_level.get(lvl)
            ]
            if forbidden_lines:
                prompt += (
                    "\n\nForbidden tokens by level — these words have been overused at the "
                    "listed level in earlier scenarios and now leak the level. Do NOT use "
                    "them at that level (other levels are fine). Find different lexical "
                    "realizations:\n" + "\n".join(forbidden_lines)
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
        target_len = ladder_obj.get("scenario", {}).get("target_word_count", 18)
        language = ladder_obj.get("scenario", {}).get("language", "en")
        language_label = self._LANGUAGE_LABELS.get(language, language)
        prompt = (
            f"The previous paraphrases for this ladder were rejected by the validation judge.\n"
            f"Failed checks: {failed or 'none listed'}.\n"
            f"Judge notes: {notes}\n\n"
            f"Canonical ladder:\n"
            f"{json.dumps(ladder_obj['ladder'], ensure_ascii=False, indent=2)}\n\n"
            f"LANGUAGE — hard requirement: every corrected paraphrase MUST be written in "
            f"{language_label} (language code '{language}'). Do not mix languages.\n"
            f"Generate {per_level} corrected paraphrases per level, addressing the issues above.\n"
            "For each paraphrase, provide: level, cue_family, text.\n"
            "Keep the proposition or requested action unchanged.\n"
            f"DIVERSITY — hard requirement: the {per_level} paraphrases within each level "
            "must each use a distinct syntactic structure and cue family; no template "
            "clones or synonym swaps, especially at the polite level.\n"
            f"LENGTH — hard requirement: every paraphrase at every level must be about "
            f"{target_len} words; equal length across levels, length must not cue the trait.\n"
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

        # Streaming shortcut tracker: pre-seed with already-generated paraphrases
        # (resume case) so the next ladder sees forbidden tokens from the start.
        shortcut_top_k = int(self.config["pipeline"].get("shortcut_top_k", 10))
        shortcut_min_count = int(self.config["pipeline"].get("shortcut_min_count", 2))
        tracker = ShortcutTracker(self.levels)
        tracker.add_rows(existing_paraphrases)

        def _one(ladder: Dict[str, Any]) -> List[Dict[str, Any]]:
            scenario_id = ladder["scenario_id"]
            scenario_language = ladder.get("scenario", {}).get("language", "en")
            schema_hint = json.dumps(
                {
                    "scenario_id": scenario_id,
                    "trait": trait,
                    "items": [{"level": "negative", "cue_family": "syntactic indirectness", "text": "..."}],
                },
                ensure_ascii=False,
            )
            with lock:
                avoid = set(used_cue_families)
            # Generate one paraphrase per level per pass, each pass seeing the
            # paraphrases already produced for this scenario, so the model cannot
            # template-clone a level's set within a single response.
            prior_by_level: Dict[str, List[str]] = {lvl: [] for lvl in self.levels}
            counters: Dict[str, int] = {lvl: 0 for lvl in self.levels}
            batch: List[Dict[str, Any]] = []
            for _pass in range(per_level):
                forbidden = tracker.top_per_level(k=shortcut_top_k, min_count=shortcut_min_count)
                try:
                    obj = self.generator.call_json(
                        system,
                        self._paraphrase_user_prompt(ladder, 1, avoid, forbidden, prior_by_level),
                        schema_hint,
                    )
                except Exception as exc:
                    print(f"  ! paraphrase pass failed for {scenario_id}: {type(exc).__name__}: {str(exc)[:100]}")
                    continue
                items = obj.get("items") if isinstance(obj, dict) else None
                if not isinstance(items, list):
                    continue
                pass_rows: List[Dict[str, Any]] = []
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
                    pass_rows.append({
                        "scenario_id": scenario_id,
                        "trait": trait,
                        "level": lvl,
                        "language": scenario_language,
                        "cue_family": item.get("cue_family", "unspecified"),
                        "text": txt,
                        "canonical": ladder["ladder"].get(lvl, ""),
                        "invariant_content": ladder.get("invariant_content", ""),
                        "paraphrase_id": f"{scenario_id}-{lvl}-{counters[lvl]:02d}",
                    })
                    prior_by_level[lvl].append(txt)
                batch.extend(pass_rows)
                tracker.add_rows(pass_rows)
            if not batch:
                raise ValueError(
                    f"paraphrase generation produced no valid items for {scenario_id}"
                )
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
        rubric = CONSTRUCT_RUBRIC
        return (
            "You are an independent validation judge for a benchmark. "
            "Your job is to reject content drift, wrong ordering, poor fluency, "
            "weak cue diversity, and shortcut-heavy bundles.\n"
            "You score paraphrase QUALITY — how faithfully each text realises its "
            "ASSIGNED level while preserving content and reading naturally — never the "
            "magnitude of the trait itself.\n\n"
            f"Rubric:\n{rubric}\n\n"
            "CRITICAL: You MUST return only valid JSON. Do not include any text before or after the JSON. "
            "Do not add explanations, preambles, or comments. The entire response must be parseable as JSON."
        )

    def _judge_user_prompt(
        self,
        trait: str,
        scenario_id: str,
        bundle: List[Dict[str, Any]],
        shortcut_hints: Optional[Dict[str, List[str]]] = None,
        min_score: float = 0.75,
    ) -> str:
        levels_order = " < ".join(self.levels)
        grouped = {
            level: [{"id": r["paraphrase_id"], "text": r["text"]}
                    for r in bundle if r["level"] == level]
            for level in self.levels
        }
        bundle_language = next((r.get("language") for r in bundle if r.get("language")), "en")
        language_label = self._LANGUAGE_LABELS.get(bundle_language, bundle_language)
        prompt = (
            f"Validate this bundle for trait '{trait}'.\n"
            f"LANGUAGE: all texts in this bundle are written in {language_label} "
            f"(code '{bundle_language}'). Apply the rubric and your fluency judgement in that "
            "language. Do not penalise correct non-English phrasing as 'unnatural'.\n"
            f"Texts by level (use the exact 'id' values in your item_scores):\n"
            f"{json.dumps(grouped, ensure_ascii=False, indent=2)}\n\n"
            f"Check: proposition/request preservation, correct {levels_order} ordering, "
            "naturalness, paraphrase diversity, obvious lexical shortcut risk, and "
            "length_balance (flag if one level's texts are substantially longer/shorter "
            "than the others — length must not be a trait cue).\n\n"
            "SCORING — read carefully. 'overall_score' and every 'item_score' rate "
            "PARAPHRASE QUALITY, not how much of the trait the text expresses. A "
            "paraphrase is high quality when it correctly and naturally realises its "
            "ASSIGNED level, preserves the shared content, and is a fluent, non-duplicate "
            f"sentence. The levels '{levels_order}' are target labels, NOT a 0-to-1 "
            "scale: a text assigned the lowest level (the negative pole) that is a clear, "
            "well-written realisation of that level deserves a HIGH score — just as much "
            "as a well-written highest-level text. Do NOT give a text a low score merely "
            "for sitting at a low level. Score a paraphrase low ONLY when it is a bad "
            "paraphrase: wrong level, content drift, unnatural, or a near-duplicate.\n"
            "Return JSON with keys: accepted, overall_score, checks, notes, item_scores.\n"
            "'checks' must include content_preservation, monotonic_order, naturalness, "
            "cue_diversity, shortcut_risk, length_balance.\n"
            "'item_scores' must be a list where each entry has the exact 'id' string from "
            "above as 'paraphrase_id', and a quality 'score' in [0,1] as defined above."
        )
        if shortcut_hints and any(shortcut_hints.values()):
            hint_lines = [
                f"  {lvl}: {shortcut_hints[lvl]}"
                for lvl in self.levels
                if shortcut_hints.get(lvl)
            ]
            if hint_lines:
                prompt += (
                    "\n\nFrequently-used tokens by level (for awareness — these recur at "
                    "the listed level across earlier bundles):\n" + "\n".join(hint_lines) + "\n"
                    "Note: this trait is partly lexical — ordinary trait vocabulary (such "
                    "as courtesy markers) is expected and is NOT itself a shortcut. Set "
                    "checks.shortcut_risk='high' and score items low ONLY when the "
                    "paraphrases at a level lack genuine variety — i.e. they are near-"
                    "duplicates of one another or all hinge on the same single token or "
                    "template. Diverse sentences that share common trait vocabulary are fine."
                )
        return prompt

    # ── Blind intensity scorer (independent LLM, no level labels visible) ──────

    def _intensity_scorer_system_prompt(self, trait: str) -> str:
        trait_info = self._trait_info(trait)
        return (
            "You are a calibrated rater of linguistic trait intensity. "
            "You will receive a list of short texts identified only by opaque codes. "
            f"For each text, rate the intensity of the trait '{trait}' on a "
            "continuous scale in [0.0, 1.0], where 0.0 is minimum intensity and "
            "1.0 is maximum intensity. Use the FULL range. Do not anchor to 0.5. "
            "You will NOT be told which texts belong together or which level they "
            "are supposed to represent. Rate each text on its own merits.\n\n"
            f"Trait definition: {trait_info.get('description', '')}\n\n"
            "CRITICAL: Return ONLY valid JSON. No prose, no markdown."
        )

    def _intensity_scorer_user_prompt(
        self,
        trait: str,
        anon_items: List[Tuple[str, str]],
    ) -> str:
        """anon_items: list of (code, text) — opaque codes, randomized order."""
        items_json = json.dumps(
            [{"code": c, "text": t} for c, t in anon_items],
            ensure_ascii=False,
            indent=2,
        )
        return (
            f"Rate the intensity of trait '{trait}' for each text below.\n\n"
            "Use the full [0.0, 1.0] range. Do NOT mode-collapse to 0.5 or to a "
            "narrow band. Texts with markedly different intensity must receive "
            "markedly different scores.\n\n"
            f"Texts:\n{items_json}\n\n"
            'Return JSON with key "scores": a list of objects '
            '{"code": "<code>", "intensity": <float in [0,1]>}. '
            "One entry per input code, in any order. "
            "Do not include any other keys."
        )

    def _intensity_scorer_language_hint(self, bundle: List[Dict[str, Any]]) -> str:
        lang = next((r.get("language") for r in bundle if r.get("language")), "en")
        return self._LANGUAGE_LABELS.get(lang, lang)

    def _score_intensity_blind(
        self,
        trait: str,
        bundle: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Score each paraphrase blind to its assigned level.

        Returns paraphrase_id → float in [0,1]. Missing rows are returned as NaN.
        Uses the tie_breaker_judge to keep it independent from the bundle judge.
        """
        if not bundle:
            return {}
        rng = random.Random(self.random_seed + sum(ord(c) for c in trait) + len(bundle))
        order = list(range(len(bundle)))
        rng.shuffle(order)
        # Opaque codes — never include level/trait/scenario in the code.
        codes = [f"q{i:03d}" for i in range(len(bundle))]
        code_to_pid: Dict[str, str] = {}
        anon_items: List[Tuple[str, str]] = []
        for code, idx in zip(codes, order):
            row = bundle[idx]
            code_to_pid[code] = row["paraphrase_id"]
            anon_items.append((code, row["text"]))
        system = self._intensity_scorer_system_prompt(trait)
        user = self._intensity_scorer_user_prompt(trait, anon_items)
        lang_label = self._intensity_scorer_language_hint(bundle)
        user = f"All texts below are written in {lang_label}. Rate them in that language.\n\n" + user
        schema_hint = json.dumps(
            {"scores": [{"code": "<code>", "intensity": "<float in [0,1]>"}]},
            ensure_ascii=False,
        )
        try:
            obj = self.tie_breaker.call_json(system, user, schema_hint)
        except Exception as exc:
            print(f"  intensity-scorer failed for {bundle[0].get('scenario_id', '?')}: {exc}")
            return {row["paraphrase_id"]: float("nan") for row in bundle}
        scores_raw = obj.get("scores") if isinstance(obj, dict) else None
        if not isinstance(scores_raw, list):
            return {row["paraphrase_id"]: float("nan") for row in bundle}
        out: Dict[str, float] = {row["paraphrase_id"]: float("nan") for row in bundle}
        for s in scores_raw:
            if not isinstance(s, dict):
                continue
            code = s.get("code")
            val = s.get("intensity")
            if code in code_to_pid and isinstance(val, (int, float)):
                v = float(val)
                if 0.0 <= v <= 1.0:
                    out[code_to_pid[code]] = v
        return out

    def _check_intensity_monotonicity(
        self,
        bundle: List[Dict[str, Any]],
        intensity_scores: Dict[str, float],
        min_gap: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Return (passed, diagnostic_dict).

        Passes iff per-level mean of blind intensity scores is strictly monotone
        in self.levels order and adjacent means differ by at least min_gap.
        Coverage: requires at least one finite score per level.
        """
        by_level: Dict[str, List[float]] = {lvl: [] for lvl in self.levels}
        for row in bundle:
            s = intensity_scores.get(row["paraphrase_id"], float("nan"))
            if isinstance(s, float) and not math.isnan(s):
                by_level.setdefault(row["level"], []).append(s)
        means: Dict[str, Optional[float]] = {}
        for lvl in self.levels:
            vals = by_level.get(lvl, [])
            means[lvl] = (sum(vals) / len(vals)) if vals else None
        diag = {
            "intensity_means_by_level": {k: (round(v, 4) if v is not None else None) for k, v in means.items()},
            "intensity_n_by_level": {k: len(by_level.get(k, [])) for k in self.levels},
            "intensity_min_gap_required": min_gap,
        }
        # All levels must have at least one score.
        if any(v is None for v in means.values()):
            diag["intensity_monotonic"] = False
            diag["intensity_failure_reason"] = "missing scores for at least one level"
            return False, diag
        ordered = [means[lvl] for lvl in self.levels]  # e.g. [negative, neutral, positive]
        # Strict monotonic increase with min gap.
        for a, b in zip(ordered, ordered[1:]):
            if b - a < min_gap:
                diag["intensity_monotonic"] = False
                diag["intensity_failure_reason"] = (
                    f"adjacent gap {b - a:.3f} < required {min_gap:.3f}"
                )
                return False, diag
        diag["intensity_monotonic"] = True
        return True, diag

    def _judge_scenario_bundle(
        self,
        trait: str,
        scenario_id: str,
        bundle: List[Dict[str, Any]],
        system: str,
        min_score: float,
        shortcut_hints: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[Dict[str, Any], bool, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run primary + optional tie-breaker judgment for one scenario bundle.

        Returns (verdict, bundle_accepted, all_judged_rows, accepted_rows).
        """
        example_ids = [r["paraphrase_id"] for r in bundle[:2]] if bundle else ["x"]
        # NOTE: schema describes types/ranges only. Do not include concrete numeric
        # example values here — judge LLMs mode-collapse onto them and return the
        # same score for every bundle.
        schema_hint = json.dumps(
            {
                "accepted": "<bool>",
                "overall_score": "<float in [0.0, 1.0]>",
                "checks": {
                    "content_preservation": "<bool>",
                    "monotonic_order": "<bool>",
                    "naturalness": "<bool>",
                    "cue_diversity": "<bool>",
                    "shortcut_risk": "<one of: low | medium | high>",
                    "length_balance": "<bool>",
                },
                "notes": "<string: cite specific evidence from the texts>",
                "item_scores": [
                    {"paraphrase_id": pid, "score": "<float in [0.0, 1.0]>"}
                    for pid in example_ids
                ],
            },
            ensure_ascii=False,
        )
        primary = self.judge.call_json(
            system,
            self._judge_user_prompt(trait, scenario_id, bundle, shortcut_hints, min_score),
            schema_hint,
        )
        # Defensive: judge LLM occasionally returns a JSON array (e.g. a list of
        # item-score objects) instead of the expected verdict dict. Coerce or
        # treat as a rejection rather than crashing the whole bundle.
        if isinstance(primary, list):
            primary = {"item_scores": primary, "accepted": False, "overall_score": 0.0}
        elif not isinstance(primary, dict):
            primary = {"accepted": False, "overall_score": 0.0, "notes": f"judge returned {type(primary).__name__}"}
        # Bundle acceptance is decoupled from the judge's scalar overall_score,
        # which mode-collapses to a narrow band and is not a reliable signal.
        # Accept on the judge's core correctness checks; the programmatic gates
        # below can still veto, and per-item scores filter individual rows.
        _checks = primary.get("checks")
        if not isinstance(_checks, dict):
            _checks = {}
        bundle_accepted = (
            _checks.get("content_preservation") is True
            and _checks.get("monotonic_order") is True
            and _checks.get("naturalness") is True
        )

        # ─── Programmatic gates (not trusted to the judge LLM) ────────────────
        gate_failures: List[str] = []

        # (1) Length-balance: per-level mean word count must not span > max_ratio.
        max_ratio = float(self.config["pipeline"].get("max_length_ratio", 1.15))
        word_counts_by_level: Dict[str, List[int]] = {}
        for r in bundle:
            word_counts_by_level.setdefault(r["level"], []).append(len(r["text"].split()))
        mean_words_per_level: Dict[str, float] = {}
        if word_counts_by_level:
            for lvl, ws in word_counts_by_level.items():
                if ws:
                    mean_words_per_level[lvl] = sum(ws) / len(ws)
            mw = list(mean_words_per_level.values())
            if len(mw) >= 2 and min(mw) > 0:
                if max(mw) > min(mw) * max_ratio:
                    primary["length_balance_hard_gate"] = False
                    primary["length_means"] = {k: round(v, 2) for k, v in mean_words_per_level.items()}
                    gate_failures.append(
                        f"length-balance: per-level mean words "
                        f"{primary['length_means']} exceeds ratio {max_ratio:.2f}"
                    )
                else:
                    primary["length_balance_hard_gate"] = True
                    primary["length_means"] = {k: round(v, 2) for k, v in mean_words_per_level.items()}

        # (2) Blind intensity monotonicity: independent LLM rates each paraphrase
        #     blind to its level. Per-level means must be monotone in self.levels
        #     order, with adjacent gap ≥ intensity_min_gap.
        enable_intensity = bool(self.config["pipeline"].get("enable_intensity_scorer", True))
        intensity_scores: Dict[str, float] = {}
        if enable_intensity:
            min_gap = float(self.config["pipeline"].get("intensity_min_gap", 0.10))
            intensity_scores = self._score_intensity_blind(trait, bundle)
            ok_mono, mono_diag = self._check_intensity_monotonicity(
                bundle, intensity_scores, min_gap
            )
            primary["intensity_gate"] = mono_diag
            if not ok_mono:
                gate_failures.append(
                    f"intensity-monotonicity: {mono_diag.get('intensity_failure_reason', 'failed')}; "
                    f"means={mono_diag.get('intensity_means_by_level')}"
                )

        if gate_failures:
            bundle_accepted = False
            primary.setdefault("notes", "")
            primary["notes"] = (primary["notes"] or "") + " [gate failures: " + " | ".join(gate_failures) + "]"
            primary["gate_failures"] = gate_failures

        # ─── Per-item acceptance ───────────────────────────────────────────────
        # Require an explicit per-item score from the judge for each paraphrase.
        # Missing or non-numeric → 0.0 (NOT min_score). This closes the
        # "judge says accepted=true and forgets item_scores" loophole.
        item_scores: Dict[str, float] = {}
        for x in primary.get("item_scores", []) or []:
            if not isinstance(x, dict):
                continue
            pid = x.get("paraphrase_id")
            sc = x.get("score")
            if isinstance(pid, str) and isinstance(sc, (int, float)):
                item_scores[pid] = float(sc)
        judged_rows: List[Dict[str, Any]] = []
        accepted_rows: List[Dict[str, Any]] = []
        for row in bundle:
            rec = dict(row)
            rec["judge"] = primary
            rec["item_score"] = item_scores.get(row["paraphrase_id"])
            rec["intensity_score_blind"] = (
                intensity_scores.get(row["paraphrase_id"]) if enable_intensity else None
            )
            rec["accepted"] = (
                bundle_accepted
                and rec["item_score"] is not None
                and rec["item_score"] >= min_score
            )
            judged_rows.append(rec)
            if rec["accepted"]:
                accepted_rows.append(rec)
        if bundle_accepted and not accepted_rows:
            missing_pids = [r["paraphrase_id"] for r in bundle if r["paraphrase_id"] not in item_scores]
            print(
                f"  WARNING: bundle {scenario_id} accepted overall but 0 items passed "
                f"item-level threshold {min_score}. "
                f"Missing item_scores for: {missing_pids[:3]}"
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
                "items": [{"level": "negative", "cue_family": "...", "text": "..."}],
            },
            ensure_ascii=False,
        )
        try:
            obj = self.generator.call_json(system, user, schema_hint)
        except Exception as exc:
            print(f"  Repair generation failed for {scenario_id}: {exc}")
            return None
        if not isinstance(obj, dict):
            return None
        items = obj.get("items")
        if not isinstance(items, list):
            return None
        out: List[Dict[str, Any]] = []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            lvl = item.get("level")
            txt = item.get("text")
            if lvl not in self.levels or not isinstance(txt, str) or not txt.strip():
                continue
            out.append({
                "scenario_id": scenario_id,
                "trait": trait,
                "level": lvl,
                "language": ladder.get("scenario", {}).get("language", "en"),
                "cue_family": item.get("cue_family", "unspecified"),
                "text": normalize_text(txt),
                "canonical": ladder["ladder"].get(lvl, ""),
                "invariant_content": ladder.get("invariant_content", ""),
                "paraphrase_id": f"{scenario_id}-{lvl}-r{idx:02d}",
            })
        return out or None

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

        # Streaming shortcut tracker: pre-seed with already-judged rows so the
        # next bundle's judge prompt includes shortcut hints from prior runs.
        shortcut_top_k = int(self.config["pipeline"].get("shortcut_top_k", 10))
        shortcut_min_count = int(self.config["pipeline"].get("shortcut_min_count", 2))
        tracker = ShortcutTracker(self.levels)
        tracker.add_rows(existing_judged)

        def _one(item: tuple) -> tuple:
            scenario_id, bundle = item
            ladder = ladder_map.get(scenario_id)
            hints = tracker.top_per_level(k=shortcut_top_k, min_count=shortcut_min_count)
            verdict, accepted, judged_bundle, accepted_bundle = self._judge_scenario_bundle(
                trait, scenario_id, bundle, system, min_score, shortcut_hints=hints
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
                hints = tracker.top_per_level(k=shortcut_top_k, min_count=shortcut_min_count)
                verdict, accepted, judged_bundle, accepted_bundle = self._judge_scenario_bundle(
                    trait, scenario_id, repaired, system, min_score, shortcut_hints=hints
                )
            with lock:
                for r in accepted_bundle:
                    if r.get("cue_family"):
                        used_cue_families.add(r["cue_family"])
            tracker.add_rows(bundle)
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
                    "items": [{"level": "negative", "cue_family": "...", "text": "..."}],
                },
                ensure_ascii=False,
            )
            try:
                obj = self.generator.call_json(system_para, user, schema_hint)
            except Exception as exc:
                print(f"  Shortcut repair generation failed for {sid}: {exc}")
                continue
            if not isinstance(obj, dict):
                continue
            items = obj.get("items")
            if not isinstance(items, list):
                continue
            new_rows: List[Dict[str, Any]] = []
            for idx, item in enumerate(items, start=1):
                if not isinstance(item, dict):
                    continue
                lvl = item.get("level")
                txt = item.get("text")
                if lvl not in self.levels or not isinstance(txt, str) or not txt.strip():
                    continue
                new_rows.append({
                    "scenario_id": sid,
                    "trait": trait,
                    "level": lvl,
                    "cue_family": item.get("cue_family", "unspecified"),
                    "text": normalize_text(txt),
                    "canonical": ladder["ladder"].get(lvl, ""),
                    "invariant_content": ladder.get("invariant_content", ""),
                    "paraphrase_id": f"{sid}-{lvl}-s{idx:02d}",
                })
            if not new_rows:
                continue
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
        # Ensure language column exists so older rows (pre-language pipeline) still group.
        if "language" not in df.columns:
            df["language"] = "en"
        else:
            df["language"] = df["language"].fillna("en")

        def _fit_lexical_baseline(sub_df: "pd.DataFrame") -> Tuple[Optional[float], List[Dict[str, Any]], List[str]]:
            """Run the per-subset lexical-baseline classifier and return
            (accuracy, top_ngrams_by_level, shortcut_ngrams)."""
            if len(sub_df) < 12 or sub_df["level"].nunique() < 2:
                return None, [], []
            vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2)
            X = vectorizer.fit_transform(sub_df["text"])
            y = sub_df["level"]
            n_splits = min(5, y.value_counts().min())
            if n_splits < 2:
                return None, [], []
            clf = LogisticRegression(max_iter=2000)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
            preds = cross_val_predict(clf, X, y, cv=cv)
            acc = float(accuracy_score(y, preds))
            clf.fit(X, y)
            vocab = vectorizer.get_feature_names_out()
            coef_rows = list(clf.coef_)
            if len(coef_rows) == 1 and len(clf.classes_) == 2:
                coef_rows = [-coef_rows[0], coef_rows[0]]
            tops: List[Dict[str, Any]] = []
            shortcuts: List[str] = []
            for idx, label in enumerate(clf.classes_):
                top_ids = coef_rows[idx].argsort()[-10:][::-1]
                level_top = [vocab[i] for i in top_ids]
                tops.append({"level": label, "top_positive_ngrams": level_top})
                shortcuts.extend(level_top[:5])
            return acc, tops, shortcuts

        # Per-language baselines — mixed-language pooling would trivially separate
        # on vocabulary, so each language gets its own classifier.
        per_language_baseline: Dict[str, Dict[str, Any]] = {}
        for lang_code, lang_df in df.groupby("language"):
            acc, tops, shortcuts = _fit_lexical_baseline(lang_df)
            per_language_baseline[str(lang_code)] = {
                "num_rows": int(len(lang_df)),
                "lexical_baseline_accuracy": acc,
                "top_ngrams_by_level": tops,
                "shortcut_ngrams": shortcuts,
            }

        # Headline accuracy for the report: max across languages (most concerning value).
        # The warning gate uses this same headline.
        accs = [v["lexical_baseline_accuracy"] for v in per_language_baseline.values()
                if v["lexical_baseline_accuracy"] is not None]
        lexical_accuracy = max(accs) if accs else None
        top_ngrams = [
            {"language": lang, **entry}
            for lang, info in per_language_baseline.items()
            for entry in info["top_ngrams_by_level"]
        ]
        shortcut_ngrams = [
            s for info in per_language_baseline.values() for s in info["shortcut_ngrams"]
        ]

        # The lexical-baseline accuracy is reported (below), not gated on:
        # politeness is intrinsically lexical, so a bag-of-words classifier
        # beating chance is expected. Cue diversity is enforced upstream
        # (generation + judge), not by destructive post-hoc regeneration.
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
            "lexical_baseline_by_language": {
                lang: {
                    "num_rows": info["num_rows"],
                    "lexical_baseline_accuracy": info["lexical_baseline_accuracy"],
                }
                for lang, info in per_language_baseline.items()
            },
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
            out_path = self.output_root / "prompts.json"
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


