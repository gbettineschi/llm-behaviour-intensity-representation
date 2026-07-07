"""Trait registry: every trait the pipeline can generate and analyse.

Pure data — no third-party imports — so the generation pipeline, the analysis
drivers, and the stdlib-only human-eval app can all load it. Adding a trait
means adding one entry to ``TRAITS`` (rubric guide, intents, trait-specific
fields); every script then accepts it via ``--trait`` and reads/writes the
per-trait directories (``data/<ts>/sentences/<trait>``,
``data/<ts>/representations/<model>/<trait>``,
``results/<ts>/<analysis>/<model>/<trait>``).

Per-trait keys:
    description      one-line definition shown to generator and judge
    axis             human-eval axis label (least -> most)
    guide            full rubric given to the sentence generator
    invariant_field  scenario field whose value must stay fixed across levels
    required_fields  trait-specific scenario fields (schema ``required`` extras)
    cue_families     cue families paraphrases must spread across
    paraphrase_note  extra diversity warning for the paraphrase prompt
    seed_example     worked scenario example (versions filled in by the pipeline)
    intents          speech acts scenarios are balanced across
    generation_constraints  trait-level hard constraints for scenario writing
"""

LEVELS = ["negative", "neutral", "positive"]  # ordinal scale, least -> most

DEFAULT_TRAIT = "politeness"


_POLITENESS_GUIDE = """\
# Politeness rubric (signed 3-point scale)

Rewrite one fixed message at three politeness levels — negative (impolite), neutral
(unmarked), positive (polite) — changing only politeness, never the content. Neutral is a
true zero: neither courtesy nor rudeness markers. Negative and positive are equal-and-opposite
departures from it.

## Keep constant across the three levels
- the intent target (what is requested / refused / criticised / etc.)
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

## Per-intent examples
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


TRAITS: dict[str, dict] = {
    "politeness": {
        "description": "Mitigation of face threat, deference, social consideration. Negative = impolite/rude; neutral = plain/matter-of-fact; positive = polite/mitigated.",
        "axis": "least polite  →  most polite",
        "guide": _POLITENESS_GUIDE,
        "invariant_field": "intent_target",
        "required_fields": ["intent_target"],
        "cue_families": [
            "lexical marker",
            "syntactic framing",
            "gratitude framing",
            "evidential framing",
            "indirectness",
            "modal framing",
        ],
        "paraphrase_note": "The polite level especially must not collapse onto a single 'I appreciate your X, but Y' scaffold.",
        "seed_example": {
            "scenario_id": "politeness-001",
            "trait": "politeness",
            "intent": "request",
            "domain": "workplace",
            "audience_relation": "peer",
            "communicative_goal": "ask a teammate to share the latest budget spreadsheet",
            "intent_target": "send the latest budget spreadsheet by end of day",
            "target_word_count": 18,
        },
        "intents": [
            {
                "id": "request",
                "description": "Asking the listener to do or provide something.",
                "example_goal": "ask a colleague to send a file",
                "extra_constraints": [
                    "Keep the requested action fixed across levels.",
                    "Do not change urgency or scope across levels.",
                ],
            },
            {
                "id": "refusal",
                "description": "Declining a request, invitation, proposal, or offer made by the listener.",
                "example_goal": "turn down a meeting invitation",
                "extra_constraints": [
                    "The refusal target (what is being declined) must remain identical across levels.",
                    "Do not change the refusal into a partial acceptance or a counter-offer.",
                ],
            },
            {
                "id": "disagreement",
                "description": "Expressing a contrary opinion, correction, or pushback on a claim.",
                "example_goal": "push back on a colleague's analysis",
                "extra_constraints": [
                    "The point of disagreement must remain identical across levels.",
                    "Do not soften disagreement into agreement at any level.",
                ],
            },
            {
                "id": "criticism_or_feedback",
                "description": "Pointing out a problem with the listener's work, output, or behaviour.",
                "example_goal": "tell a junior their report needs rework",
                "extra_constraints": [
                    "The criticised aspect must remain identical across levels.",
                    "Do not turn criticism into pure praise.",
                ],
            },
            {
                "id": "bad_news_delivery",
                "description": "Telling the listener something they will not want to hear (denial, rejection, negative outcome).",
                "example_goal": "inform a customer their refund is denied",
                "extra_constraints": [
                    "The bad news content must remain identical across levels.",
                    "Do not change a denial into an approval or a hedged maybe.",
                ],
            },
            {
                "id": "apology",
                "description": "Acknowledging fault or expressing regret for a specific wrongdoing.",
                "example_goal": "apologise for missing a deadline",
                "extra_constraints": [
                    "The thing being apologised for must remain identical across levels.",
                    "Do not change which party is at fault.",
                ],
            },
            {
                "id": "complaint",
                "description": "Voicing a grievance about a problem or situation affecting the speaker (service issue, environmental nuisance, missed commitment, etc.). Distinct from criticism_or_feedback, which targets the listener's work.",
                "example_goal": "complain to a hotel manager about a noisy neighbouring room",
                "extra_constraints": [
                    "The grievance (what is wrong) must remain identical across levels.",
                    "Do not turn the complaint into pure praise or into a refusal of service.",
                ],
            },
            {
                "id": "reminder",
                "description": "Prompting the listener about an outstanding obligation, deadline, or commitment they owe.",
                "example_goal": "remind a colleague that an expense report is overdue",
                "extra_constraints": [
                    "The reminded item (what is outstanding) must remain identical across levels.",
                    "Do not change the reminder into a new request or an apology.",
                ],
            },
            {
                "id": "inquiry_sensitive",
                "description": "Asking a personal, awkward, or socially delicate question.",
                "example_goal": "ask a coworker why they missed work last week",
                "extra_constraints": [
                    "The question's content must remain identical across levels.",
                    "Do not change the topic or scope of the inquiry.",
                ],
            },
            {
                "id": "correction",
                "description": "Pointing out a factual or procedural mistake the listener made.",
                "example_goal": "correct a junior's misuse of a tool",
                "extra_constraints": [
                    "The corrected fact or step must remain identical across levels.",
                    "Do not change the correction into agreement or an unrelated tip.",
                ],
            },
        ],
        "generation_constraints": [
            "Intent_target must be fixed across levels and paraphrases.",
            "Urgency and imposition must stay constant across levels.",
            "No insults, threats, or profanity even at the negative (impolite) pole.",
            "Diversify politeness across directness, deference, gratitude, softeners, and impersonal phrasing.",
            "Keep every paraphrase, at every level, close to target_word_count.",
        ],
    },
}
