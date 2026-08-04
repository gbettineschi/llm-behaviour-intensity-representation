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


_FORMALITY_GUIDE = """\
# Formality rubric (signed 3-point scale)

Rewrite one fixed message at three formality levels — negative (casual/colloquial), neutral
(plain, unmarked register), positive (formal/ceremonial) — changing only register, never the
content. Neutral is a true zero: neither casual markers nor formal markers. Negative and
positive are equal-and-opposite departures from it.

## Keep constant across the three levels
- the intent target (what is requested / refused / criticised / etc.)
- named entities, dates, deadlines, and the core intent
- the polarity of the act (a refusal stays a refusal; an apology stays an apology)
- urgency, scope, and the amount imposed
- length: all three levels should be about the same number of words

## The three levels
**Negative — casual/colloquial.** Relaxed, conversational register: contractions, informal
address, colloquial phrasing; it must read as clearly informal to an ordinary reader, not
merely plain or short. Casualness comes from register and syntax, never from slang that would
be unintelligible, profanity, or dropping content.

**Neutral — plain register.** Matter-of-fact, standard written register; no colloquial markers
and no ceremonial markers.

**Positive — formal/ceremonial.** Elevated register: full forms, impersonal constructions,
formal address, ceremonial framing. Formality is in the register, not extra words — keep it
the same length.

## Per-intent examples
**request.** Negative: "Hey, can you shoot me the file?" Neutral: "Can you send me the file?" Positive: "Would you kindly forward the file at your earliest convenience?" The requested action stays identical.

**refusal.** Negative: "Nah, can't make it, sorry." Neutral: "I won't be able to make it." Positive: "I regret that I am unable to attend." The refusal target stays identical.

**disagreement.** Negative: "Nah, that's not it." Neutral: "I don't think that's right." Positive: "I must respectfully submit that this is not accurate." The disagreed-with claim stays identical.

**criticism_or_feedback.** Negative: "This report's kinda sloppy, ngl." Neutral: "This report needs more work." Positive: "This report would benefit from further refinement." The criticised aspect stays identical.

**bad_news_delivery.** Negative: "Bad news — refund's a no." Neutral: "We can't approve your refund." Positive: "We regret to inform you that your refund request cannot be approved." The bad news stays identical.

**apology.** Negative: "Yeah my bad, missed the deadline." Neutral: "I'm sorry I missed the deadline." Positive: "I offer my sincere apologies for having missed the deadline." The apologised-for action stays identical.

**complaint.** Negative: "The room next door's way too loud, ugh." Neutral: "The room next door is very noisy." Positive: "I wish to bring to your attention that the adjoining room has been excessively noisy." The grievance stays identical.

**reminder.** Negative: "Hey, still waiting on that expense report, lol." Neutral: "Just a reminder that the expense report is still outstanding." Positive: "This is a reminder that the expense report remains outstanding." The outstanding item stays identical.

**inquiry_sensitive.** Negative: "Where were you all last week?" Neutral: "Can I ask why you were away last week?" Positive: "May I inquire as to the reason for your absence last week?" The question's content stays identical.

**correction.** Negative: "Nope, you're doing that totally wrong." Neutral: "That's not quite right; the tool should be used this way." Positive: "I would note that the tool is more appropriately used as follows." The corrected fact stays identical.

## Cue diversity
Within a level, do not lean on one marker. Spread across register vocabulary, contractions vs.
full forms, syntactic complexity, address forms, impersonal constructions, and discourse
markers.
"""


_CERTAINTY_GUIDE = """\
# Certainty rubric (signed 3-point scale)

Rewrite one fixed message at three certainty levels — negative (hedged/doubtful), neutral
(plain assertion), positive (confident/emphatic) — changing only the speaker's expressed
confidence, never the claim itself. Neutral is a true zero: a plain assertion with no hedges
and no boosters. Negative and positive are equal-and-opposite departures from it.

## Keep constant across the three levels
- the intent target (what is requested / refused / criticised / etc.) and the claim being made
- named entities, dates, deadlines, and the core intent
- the polarity of the act (a refusal stays a refusal; an apology stays an apology)
- the actual truth-value or content of the claim — only expressed confidence in it varies
- length: all three levels should be about the same number of words

## The three levels
**Negative — hedged/doubtful.** Tentative, qualified, self-doubting; it must read as clearly
uncertain to an ordinary reader, not merely plain or brief. Doubt comes from modality and
framing, never from actually retracting or changing the claim.

**Neutral — plain assertion.** Matter-of-fact statement; no hedges and no boosters.

**Positive — confident/emphatic.** Assured, emphatic, unequivocal. Confidence is in the
framing, not extra words — keep it the same length.

## Per-intent examples
**request.** Negative: "I think maybe you could possibly send me the file, if that's okay?" Neutral: "Can you send me the file?" Positive: "Send me the file — I need it." The requested action stays identical.

**refusal.** Negative: "I don't think I can make it, I'm not sure." Neutral: "I won't be able to make it." Positive: "I definitely won't be able to make it." The refusal target stays identical.

**disagreement.** Negative: "I'm not totally sure, but I don't think that's quite right?" Neutral: "I don't think that's right." Positive: "That is simply not right." The disagreed-with claim stays identical.

**criticism_or_feedback.** Negative: "This might need a bit more work, maybe?" Neutral: "This report needs more work." Positive: "This report clearly needs more work." The criticised aspect stays identical.

**bad_news_delivery.** Negative: "I think we probably can't approve your refund." Neutral: "We can't approve your refund." Positive: "We cannot approve your refund." The bad news stays identical.

**apology.** Negative: "I guess I might have missed the deadline, sorry." Neutral: "I'm sorry I missed the deadline." Positive: "I clearly missed the deadline, and I'm sorry." The apologised-for action stays identical.

**complaint.** Negative: "I think the room next door might be a bit noisy, maybe?" Neutral: "The room next door is very noisy." Positive: "The room next door is unmistakably noisy." The grievance stays identical.

**reminder.** Negative: "I think the expense report might still be outstanding?" Neutral: "Just a reminder that the expense report is still outstanding." Positive: "The expense report is still outstanding — this needs to be filed." The outstanding item stays identical.

**inquiry_sensitive.** Negative: "I'm not sure, but were you maybe away last week?" Neutral: "Can I ask why you were away last week?" Positive: "I need to know why you were away last week." The question's content stays identical.

**correction.** Negative: "I could be wrong, but I don't think that's quite right." Neutral: "That's not quite right; the tool should be used this way." Positive: "That is wrong; the tool must be used this way." The corrected fact stays identical.

## Cue diversity
Within a level, do not lean on one marker. Spread across modal verbs, epistemic adverbs,
evidential framing, hedges vs. boosters, question tags, and subjective openers.
"""


_URGENCY_GUIDE = """\
# Urgency rubric (signed 3-point scale)

Rewrite one fixed message at three urgency levels — negative (relaxed/no-rush), neutral
(plain), positive (urgent/pressing) — changing only the expressed pressure, never the
underlying facts. Neutral is a true zero: no rush markers and no pressure markers. Negative
and positive are equal-and-opposite departures from it.

## Keep constant across the three levels
- the intent target (what is requested / refused / criticised / etc.)
- named entities, dates, and the actual deadline or timeframe stated in the content
- the polarity of the act (a refusal stays a refusal; an apology stays an apology)
- scope and the amount imposed
- length: all three levels should be about the same number of words

## The three levels
**Negative — relaxed/no-rush.** Unhurried, low-priority framing; it must read as clearly
relaxed to an ordinary reader, not merely plain or short. Relaxedness comes from pacing and
framing, never from changing the stated deadline itself.

**Neutral — plain.** Matter-of-fact statement of the timeframe; no rush markers and no
pressure markers.

**Positive — urgent/pressing.** Time-critical, high-priority framing built from consequence
and prioritization language — never from lexical giveaways like "ASAP", "urgent", or
"immediately", and never from moving the actual deadline earlier. Urgency is in the framing,
not extra words — keep it the same length.

## Per-intent examples
**request.** Negative: "Whenever you get a chance, could you send me the file? No rush." Neutral: "Can you send me the file by Friday?" Positive: "I need that file before Friday's meeting starts, or we can't proceed." The requested action and deadline stay identical.

**refusal.** Negative: "I won't be able to make it Friday, it's fine either way." Neutral: "I won't be able to make it on Friday." Positive: "I won't be able to make it Friday, and that leaves us without cover for the deadline." The refusal target stays identical.

**disagreement.** Negative: "I don't think that's right, but there's time to sort it out." Neutral: "I don't think that's right." Positive: "That's not right, and we need to fix it before it goes out today." The disagreed-with claim stays identical.

**criticism_or_feedback.** Negative: "This report needs more work, whenever works for you." Neutral: "This report needs more work before Friday." Positive: "This report needs more work now — it can't go out as is before Friday's deadline." The criticised aspect stays identical.

**bad_news_delivery.** Negative: "We can't approve your refund just yet, no immediate action needed." Neutral: "We can't approve your refund." Positive: "We can't approve your refund, and you'll need to resubmit before the window closes today." The bad news stays identical.

**apology.** Negative: "Sorry I missed the deadline, I'll get to it when I can." Neutral: "I'm sorry I missed the deadline." Positive: "I'm sorry I missed the deadline — this needs to be fixed right away before it holds anyone else up." The apologised-for action stays identical.

**complaint.** Negative: "The room next door is noisy, whenever it's convenient to look into it." Neutral: "The room next door is very noisy." Positive: "The room next door is noisy and I need this resolved before I can get any sleep tonight." The grievance stays identical.

**reminder.** Negative: "Whenever you have a moment, the expense report is still outstanding." Neutral: "Just a reminder that the expense report is still outstanding." Positive: "The expense report is still outstanding and needs to be filed before today's close." The outstanding item stays identical.

**inquiry_sensitive.** Negative: "No rush, but can I ask why you were away last week?" Neutral: "Can I ask why you were away last week?" Positive: "I need to know why you were away last week before I sign off on this today." The question's content stays identical.

**correction.** Negative: "That's not quite right; worth fixing whenever you get a chance." Neutral: "That's not quite right; the tool should be used this way." Positive: "That's not quite right, and it needs fixing before this ships today." The corrected fact stays identical.

## Cue diversity
Within a level, do not lean on one marker, and never on the words "ASAP", "urgent", or
"immediately" alone. Spread across temporal framing, consequence framing, imperative force,
prioritization language, and pacing/brevity of framing.
"""


_ENTHUSIASM_GUIDE = """\
# Enthusiasm rubric (signed 3-point scale)

Rewrite one fixed message at three enthusiasm levels — negative (reluctant/flat), neutral
(plain), positive (enthusiastic/eager) — changing only the speaker's expressed eagerness,
never the content. Neutral is a true zero: no reluctance markers and no eagerness markers.
Negative and positive are equal-and-opposite departures from it.

## Keep constant across the three levels
- the intent target (what is requested / refused / criticised / etc.)
- named entities, dates, deadlines, and the core intent
- the polarity of the act (a refusal stays a refusal; an apology stays an apology)
- urgency, scope, and the amount imposed
- length: all three levels should be about the same number of words

## The three levels
**Negative — reluctant/flat.** Weary, unenthusiastic, going-through-the-motions; it must read
as clearly reluctant to an ordinary reader, not merely plain or short. Reluctance comes from
volition and evaluative framing, never from refusing outright or changing the content.

**Neutral — plain.** Matter-of-fact statement; no reluctance markers and no eagerness markers.

**Positive — enthusiastic/eager.** Eager, animated, positively evaluative. Enthusiasm is in
the framing, not extra words — keep it the same length. Closest to Tigges-style valence, a
bridge to prior linear-representation work.

## Per-intent examples
**request.** Negative: "I guess you could send me the file, if you have to." Neutral: "Can you send me the file?" Positive: "Would you send me the file? I'd love to dig into it!" The requested action stays identical.

**refusal.** Negative: "I won't be able to make it, whatever." Neutral: "I won't be able to make it." Positive: "I really wish I could, but I won't be able to make it — I was so looking forward to it!" The refusal target stays identical.

**disagreement.** Negative: "I guess I don't really think that's right." Neutral: "I don't think that's right." Positive: "I'd love to dig into this — I really don't think that's right!" The disagreed-with claim stays identical.

**criticism_or_feedback.** Negative: "This report needs more work, I suppose." Neutral: "This report needs more work." Positive: "There's some great material here — I'd love to see this report get a bit more work!" The criticised aspect stays identical.

**bad_news_delivery.** Negative: "We can't approve your refund, whatever that's worth." Neutral: "We can't approve your refund." Positive: "I really wish this were different, but we can't approve your refund!" The bad news stays identical.

**apology.** Negative: "I guess I missed the deadline, sorry or whatever." Neutral: "I'm sorry I missed the deadline." Positive: "I'm so sorry I missed the deadline — I really wanted to get this right for you!" The apologised-for action stays identical.

**complaint.** Negative: "The room next door is noisy, I guess, if it even matters." Neutral: "The room next door is very noisy." Positive: "I have to say, I'd really love it if we could sort out how noisy the room next door has been!" The grievance stays identical.

**reminder.** Negative: "The expense report's still outstanding, whenever, I guess." Neutral: "Just a reminder that the expense report is still outstanding." Positive: "Quick reminder — I'd love to get that expense report filed, it'll feel great to have it done!" The outstanding item stays identical.

**inquiry_sensitive.** Negative: "I guess I could ask why you were away last week." Neutral: "Can I ask why you were away last week?" Positive: "I'd love to hear about it — why were you away last week?" The question's content stays identical.

**correction.** Negative: "I suppose that's not quite right; the tool works differently." Neutral: "That's not quite right; the tool should be used this way." Positive: "Oh, I'd love to show you — it actually works a little differently!" The corrected fact stays identical.

## Cue diversity
Within a level, do not lean on one marker. Spread across affective vocabulary, intensifiers,
exclamatory framing, volition markers ("happy to" / "if I must"), and evaluative framing.
"""


# Shared across all traits: the 10 speech acts are orthogonal to what varies (politeness,
# formality, ...), so every trait entry's "intents" reuses this list.
_SHARED_INTENTS: list[dict] = [
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
]


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
        "intents": _SHARED_INTENTS,
        "generation_constraints": [
            "Intent_target must be fixed across levels and paraphrases.",
            "Urgency and imposition must stay constant across levels.",
            "No insults, threats, or profanity even at the negative (impolite) pole.",
            "Diversify politeness across directness, deference, gratitude, softeners, and impersonal phrasing.",
            "Keep every paraphrase, at every level, close to target_word_count.",
        ],
    },
    "formality": {
        "description": "Register level: colloquial vs. ceremonial phrasing. Negative = casual/colloquial; neutral = plain/standard register; positive = formal/ceremonial.",
        "axis": "least formal  →  most formal",
        "guide": _FORMALITY_GUIDE,
        "invariant_field": "intent_target",
        "required_fields": ["intent_target"],
        "cue_families": [
            "register vocabulary",
            "contractions vs full forms",
            "syntactic complexity",
            "address forms",
            "impersonal constructions",
            "discourse markers",
        ],
        "paraphrase_note": "The formal level especially must not collapse onto a single 'kindly' / 'would you be so kind as to' scaffold.",
        "seed_example": {
            "scenario_id": "formality-001",
            "trait": "formality",
            "intent": "request",
            "domain": "workplace",
            "audience_relation": "peer",
            "communicative_goal": "ask a teammate to share the latest budget spreadsheet",
            "intent_target": "send the latest budget spreadsheet by end of day",
            "target_word_count": 18,
        },
        "intents": _SHARED_INTENTS,
        "generation_constraints": [
            "Intent_target must be fixed across levels and paraphrases.",
            "Urgency and imposition must stay constant across levels.",
            "No slang that would be unintelligible, profanity, or content changes even at the casual pole.",
            "Diversify formality across register vocabulary, contractions, syntax, address forms, and discourse markers.",
            "Keep every paraphrase, at every level, close to target_word_count.",
        ],
    },
    "certainty": {
        "description": "Speaker's expressed confidence in a claim, independent of the claim's content. Negative = hedged/doubtful; neutral = plain assertion; positive = confident/emphatic.",
        "axis": "least certain  →  most certain",
        "guide": _CERTAINTY_GUIDE,
        "invariant_field": "intent_target",
        "required_fields": ["intent_target"],
        "cue_families": [
            "modal verbs",
            "epistemic adverbs",
            "evidential framing",
            "hedges vs boosters",
            "question tags",
            "subjective openers",
        ],
        "paraphrase_note": "The confident level especially must not collapse onto a single 'definitely' / 'clearly' scaffold.",
        "seed_example": {
            "scenario_id": "certainty-001",
            "trait": "certainty",
            "intent": "request",
            "domain": "workplace",
            "audience_relation": "peer",
            "communicative_goal": "ask a teammate to share the latest budget spreadsheet",
            "intent_target": "send the latest budget spreadsheet by end of day",
            "target_word_count": 18,
        },
        "intents": _SHARED_INTENTS,
        "generation_constraints": [
            "Intent_target and the underlying claim must be fixed across levels and paraphrases.",
            "Urgency and imposition must stay constant across levels.",
            "Do not retract, hedge away, or strengthen the underlying claim's content — only expressed confidence may vary.",
            "Diversify certainty across modal verbs, epistemic adverbs, evidential framing, hedges, boosters, and question tags.",
            "Keep every paraphrase, at every level, close to target_word_count.",
        ],
    },
    "urgency": {
        "description": "Expressed time pressure and priority, independent of the stated deadline. Negative = relaxed/no-rush; neutral = plain; positive = urgent/pressing.",
        "axis": "least urgent  →  most urgent",
        "guide": _URGENCY_GUIDE,
        "invariant_field": "intent_target",
        "required_fields": ["intent_target"],
        "cue_families": [
            "temporal framing",
            "consequence framing",
            "imperative force",
            "prioritization language",
            "pacing/brevity of framing",
        ],
        "paraphrase_note": "The urgent level especially must not collapse onto a single 'ASAP' / 'right away' scaffold — vary consequence and prioritization framing instead.",
        "seed_example": {
            "scenario_id": "urgency-001",
            "trait": "urgency",
            "intent": "request",
            "domain": "workplace",
            "audience_relation": "peer",
            "communicative_goal": "ask a teammate to share the latest budget spreadsheet",
            "intent_target": "send the latest budget spreadsheet by end of day",
            "target_word_count": 18,
        },
        "intents": _SHARED_INTENTS,
        "generation_constraints": [
            "Intent_target and the actual stated deadline must be fixed across levels and paraphrases.",
            "Scope and imposition must stay constant across levels.",
            "Never use the words 'ASAP', 'urgent', 'urgently', or 'immediately' as a shortcut for urgency — convey it through consequence and prioritization framing.",
            "Diversify urgency across temporal framing, consequence framing, imperative force, and prioritization language.",
            "Keep every paraphrase, at every level, close to target_word_count.",
        ],
    },
    "enthusiasm": {
        "description": "Expressed eagerness and positive affect toward the act, closest to Tigges-style valence. Negative = reluctant/flat; neutral = plain; positive = enthusiastic/eager.",
        "axis": "least enthusiastic  →  most enthusiastic",
        "guide": _ENTHUSIASM_GUIDE,
        "invariant_field": "intent_target",
        "required_fields": ["intent_target"],
        "cue_families": [
            "affective vocabulary",
            "intensifiers",
            "exclamatory framing",
            "volition markers",
            "evaluative framing",
        ],
        "paraphrase_note": "The enthusiastic level especially must not collapse onto a single 'I'd love to' scaffold.",
        "seed_example": {
            "scenario_id": "enthusiasm-001",
            "trait": "enthusiasm",
            "intent": "request",
            "domain": "workplace",
            "audience_relation": "peer",
            "communicative_goal": "ask a teammate to share the latest budget spreadsheet",
            "intent_target": "send the latest budget spreadsheet by end of day",
            "target_word_count": 18,
        },
        "intents": _SHARED_INTENTS,
        "generation_constraints": [
            "Intent_target must be fixed across levels and paraphrases.",
            "Urgency and imposition must stay constant across levels.",
            "Do not turn reluctance into an outright refusal or change the content even at the reluctant pole.",
            "Diversify enthusiasm across affective vocabulary, intensifiers, exclamatory framing, volition markers, and evaluative framing.",
            "Keep every paraphrase, at every level, close to target_word_count.",
        ],
    },
}
