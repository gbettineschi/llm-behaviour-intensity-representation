# Construct Rubric (3-Level Benchmark)

This rubric is written for a **representation-geometry benchmark**, not for a generic classification dataset. The key requirement is that all levels within a ladder preserve the same underlying proposition or requested action while varying only the target trait.

---

## Trait A: Politeness

### Core construct
Politeness is the degree to which the utterance mitigates face threat, softens imposition, signals respect or deference, and frames the request in a socially considerate way.

### Scope restriction
Use **requests and request-like directives only**.
Do not mix in apologies, thanks-only utterances, praise, or general emotional warmth unless they are part of the request framing.

### What must remain constant across levels
- requested action
- target object or information
- named entities
- core practical intent
- truth conditions

### What may vary
- directness vs indirectness
- mitigation and softening
- deference and respect markers
- framing with gratitude or appreciation
- level of imposition management
- syntactic form of the request

### What must not vary
- urgency unless explicitly fixed in the scenario
- amount of requested work
- stance on whether the request is optional
- sentiment unrelated to the request
- extra content not present in the base scenario

### Three-level scale

#### Level 0: Low politeness
Characteristics:
- direct or blunt request
- little or no mitigation
- no courtesy markers required
- may sound terse or mildly imposing
- still must remain plausible and not become abusive or insulting

Allowed cues:
- imperative or strongly direct form
- minimal framing
- absence of softeners

Forbidden failure modes:
- open rudeness, insult, sarcasm, profanity
- changing the action requested
- making the request more urgent than the scenario specifies

#### Level 1: Mid politeness
Characteristics:
- clear request with moderate mitigation
- socially acceptable and neutral-professional
- some softening, but not highly deferential

Allowed cues:
- modal requests such as "could you"
- light softeners such as "please" or "when you have a moment"
- neutral professional framing

Forbidden failure modes:
- sounding as blunt as level 0
- sounding strongly deferential or overly apologetic like level 2

#### Level 2: High politeness
Characteristics:
- clearly respectful and mitigated request
- stronger face-saving framing
- signals appreciation or deference without changing the request

Allowed cues:
- appreciation framing such as "I'd appreciate it"
- deferential modal constructions
- acknowledgement of imposition
- respectful softening and indirectness

Forbidden failure modes:
- changing the request into a suggestion or different action
- adding excessive explanation that changes content
- becoming florid or unnatural

### Cue-diversity requirement
For a given level, do not rely on one marker repeatedly. Spread realizations across:
- lexical courtesy markers
- syntactic indirectness
- gratitude framing
- imposition acknowledgment
- depersonalized or softened phrasing

### Canonical scenario types
- ask colleague for file or document
- request deadline extension
- ask reviewer or teammate for clarification
- ask someone to revise or resend something
- ask for meeting reschedule
- ask for access or permission

---

## Trait B: Hedging / Linguistic Confidence

### Core construct
Hedging/confidence is the degree of speaker commitment to the proposition. The low end expresses uncertainty or tentativeness; the high end expresses strong commitment or confidence.

### Scope restriction
Use **assertions, answers, recommendations, forecasts, and explanations**.
Do not mix in explicit probability numbers unless the scenario specifically allows them.

### What must remain constant across levels
- underlying proposition
- recommended action or answer
- named entities and factual content
- intended semantic answer to the scenario

### What may vary
- strength of commitment
- epistemic stance
- hedge markers
- discourse framing of evidence or uncertainty
- degree of tentativeness in recommendation or answer

### What must not vary
- proposition itself
- amount of evidence claimed, unless the scenario explicitly encodes the same evidence base across levels
- polarity or factual answer
- specificity of the recommendation

### Three-level scale

#### Level 0: Low confidence / strongly hedged
Characteristics:
- tentative stance
- explicit uncertainty or limited commitment
- answer remains usable but clearly cautious

Allowed cues:
- "it might be"
- "I think it may"
- "it seems possible"
- "based on what I know"
- tentative recommendation framing

Forbidden failure modes:
- changing the answer itself
- making the speaker ignorant rather than uncertain
- adding irrelevant justifications

#### Level 1: Mid confidence / moderate commitment
Characteristics:
- balanced answer
- some uncertainty but not strongly hedged
- sounds like an ordinary qualified claim

Allowed cues:
- "it's likely"
- "I think it's"
- "it seems"
- moderate commitment without categorical tone

Forbidden failure modes:
- sounding as categorical as level 2
- sounding as doubtful as level 0

#### Level 2: High confidence / minimally hedged
Characteristics:
- strong commitment to the same proposition
- direct assertion or recommendation
- still natural and not boastful

Allowed cues:
- direct declarative answer
- firm recommendation
- explicit commitment without added aggression

Forbidden failure modes:
- changing the proposition
- adding evidence that changes scenario semantics
- turning the answer into an imperative if it was originally an assertion

### Cue-diversity requirement
For a given level, diversify across:
- lexical hedges
- evidential framing
- modal verbs
- discourse-softening phrases
- syntax and clause structure

Avoid mapping one level to one token such as:
- level 0 always contains "maybe"
- level 1 always contains "likely"
- level 2 never contains first-person stance

### Canonical scenario types
- answer a factual question under ordinary uncertainty
- give a recommendation
- make a forecast
- summarize a likely cause
- give a diagnosis-style tentative explanation
- state an interpretation of evidence

---

## Universal acceptance criteria

Every final item must satisfy all of the following:

1. **Content preservation**: the same proposition or requested action is preserved across levels.
2. **Ordered intensity**: human or validated judge ordering matches low < mid < high for the target trait.
3. **Naturalness**: each sentence is fluent and plausible in ordinary usage.
4. **No overt artifacts**: no single cue or template uniquely identifies one level across the dataset.
5. **Paraphrase diversity**: at least two distinct phrasings per level are not near-duplicates.
6. **No domain leakage**: scenario metadata, topic, or named entities do not uniquely determine the level.
