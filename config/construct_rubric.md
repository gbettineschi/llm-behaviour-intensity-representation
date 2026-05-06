# Construct Rubric (3-Level Benchmark) — v2

This rubric is written for a **representation-geometry benchmark**, not a generic classification dataset. The key requirement is that all levels within a ladder preserve the same underlying content while varying only the target trait. The rubric is now multi–speech-act: each trait is realised across several speech acts, and intensity is defined consistently across all of them.

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
- sentence length — high-politeness paraphrases must not be substantially longer than low-politeness ones

### Three-level scale (universal across speech acts)

#### Level 0 — Low politeness
- direct, blunt realisation of the act
- little or no mitigation
- terse but never abusive, insulting, or profane

#### Level 1 — Mid politeness
- clear realisation with moderate mitigation
- neutral-professional register
- some softeners ("could you", "I'm afraid", "please") but not strongly deferential

#### Level 2 — High politeness
- clearly respectful and mitigated
- strong face-saving framing
- appreciation, deference, imposition acknowledgment without changing what is being said

### Per-speech-act guidance

**request.** Low: imperative or near-imperative ("Send me the file."). Mid: modal request with light softener ("Could you send me the file when you have a moment?"). High: deferential framing with gratitude or imposition acknowledgment ("I'd really appreciate it if you could send me the file when you get a chance.").

**refusal.** Low: bare "no" plus minimal reason ("I can't make it."). Mid: softened decline with brief reason ("I won't be able to make it, sorry."). High: appreciative refusal acknowledging the offer and apologising ("Thank you so much for the invitation — I'm afraid I won't be able to make it this time."). The refusal target stays identical.

**disagreement.** Low: flat contradiction ("That's wrong."). Mid: hedged contradiction ("I don't think that's quite right."). High: respectful disagreement with framing ("I see your point, but I'd respectfully push back — I don't think that holds."). The disagreed-with claim stays identical.

**criticism_or_feedback.** Low: direct judgment ("This report is inadequate."). Mid: feedback with mitigation ("This report needs more work in places."). High: appreciative, face-saving feedback ("There's a lot of good material here; I think the report would benefit from some additional work in a few places."). The criticised aspect stays identical.

**bad_news_delivery.** Low: blunt delivery ("Your refund is denied."). Mid: softened delivery with brief reason ("Unfortunately we can't approve your refund."). High: empathetic delivery with appreciation and apology ("I'm really sorry to have to tell you this, but we won't be able to approve your refund."). The bad news stays identical.

**apology.** Low: minimal acknowledgement ("Sorry I missed the deadline."). Mid: ordinary apology with brief explanation ("I'm sorry I missed the deadline — I should have flagged it earlier."). High: full face-restoring apology with acknowledgment of impact ("I really do apologise for missing the deadline; I know it put extra pressure on the team and I should have raised it sooner."). The apologised-for action stays identical.

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
Hedging/confidence is the degree of speaker commitment to a proposition. The low end expresses uncertainty or tentativeness; the high end expresses strong commitment.

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

### Three-level scale (universal across speech acts)

#### Level 0 — Low confidence / strongly hedged
- tentative stance, explicit uncertainty
- usable but clearly cautious
- "I think it might…", "based on what I can tell…", "it's possible that…"

#### Level 1 — Mid confidence
- balanced commitment
- ordinary qualified claim
- "it's likely…", "I think…", "it seems…"

#### Level 2 — High confidence / minimally hedged
- strong commitment, direct statement
- still natural, not boastful or aggressive
- "it is…", "I'm confident that…", a bare assertion

### Per-speech-act guidance

**factual_assertion.** Low: "It might be that the policy reduces costs." Mid: "It looks like the policy reduces costs." High: "The policy reduces costs." Polarity stays identical.

**recommendation.** Low: "You might want to consider switching vendors." Mid: "I'd suggest switching vendors." High: "You should switch vendors." The recommended action stays identical.

**forecast.** Low: "Churn could rise next quarter." Mid: "Churn is likely to rise next quarter." High: "Churn will rise next quarter." Direction of the prediction stays identical; do not insert numbers.

**causal_explanation.** Low: "The deployment may have failed because of the config change." Mid: "The deployment likely failed because of the config change." High: "The deployment failed because of the config change." The causal claim stays identical.

**yes_no_answer.** Low: "I think the answer is probably yes, though I'm not certain." Mid: "I'd say yes." High: "Yes." Polarity stays identical.

**estimation.** Low: "It might take roughly two weeks, give or take." Mid: "It'll likely take about two weeks." High: "It'll take two weeks." The central estimate stays identical; only confidence in it varies.

### Cue-diversity requirement
Diversify across:
- lexical hedges
- evidential framing
- modal verbs
- discourse-softening phrases
- syntax and clause structure

Avoid mapping one level to one token (e.g. "maybe" ↔ low only).

---

## Universal acceptance criteria

Every final item must satisfy all of the following:

1. **Content preservation**: same core content (target / proposition / answer / forecast / cause / estimate) across levels. Every content_probe must have the same expected_answer at every level.
2. **Ordered intensity**: human or validated judge ordering matches low < mid < high for the target trait.
3. **Naturalness**: each sentence is fluent and plausible in ordinary usage.
4. **No overt artifacts**: no single cue or template uniquely identifies one level across the dataset.
5. **Paraphrase diversity**: at least two distinct phrasings per level, not near-duplicates.
6. **No domain leakage**: scenario metadata, topic, named entities, or speech act do not uniquely determine the level.
7. **Length balance**: paraphrase length must not correlate with level. All paraphrases within a scenario must be within ±length_tolerance_pct of target_word_count.
