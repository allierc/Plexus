"""hypothesis -- the scientific protocol of the campaign: HYPOTHESIS FIRST, then the test.

Nothing in this file simulates anything. It exists to make one discipline structural rather than
aspirational:

    a candidate is not run until its PREDICTION has been recorded.

That single rule is what converts a parameter sweep into an experiment, and it is what makes the
70/30 balance measurable instead of rhetorical.

--------------------------------------------------------------------------------------------
THE 70/30 BALANCE, AND WHY IT IS A SETPOINT AND NOT A QUOTA
--------------------------------------------------------------------------------------------
Each round proposes a BATCH. The batch is allocated between two intents:

    CONFIRMATORY (~70%)   an edit the current causal map predicts WILL work.
                          Consolidates the map. Individually low-information, collectively
                          what makes the map usable.
    ADVERSARIAL  (~30%)   an edit predicted to BREAK the current best explanation.
                          Individually high-variance; this is where surprise lives.

Because every candidate carries a prediction made BEFORE the run, each outcome falls into a 2x2:

                     | confirmed                    | refuted
    -----------------+------------------------------+------------------------------
    confirmatory     | consolidates (low info)      | ** SURPRISE ** (high info)
    adversarial      | ** SURPRISE ** (high info)   | breaks as expected (low info)

The information content of a round is therefore the SURPRISE RATE -- the fraction of prediction
errors. And that gives the Supervisor a control signal that neither a quota nor a human's taste
can provide:

    surprise < 0.10   the batch has drifted to 100/0. Everything confirms what we already
                      believe: near-zero information. -> push the mix ADVERSARIAL.
    surprise > 0.50   the batch has drifted to 0/100. Everything breaks as expected, nothing
                      consolidates: no map accumulates. -> push the mix CONFIRMATORY.
    target ~ 0.30     the productive regime.

This is the mechanism by which the loop keeps itself out of both failure modes the hand-run
campaign fell into: thirty rounds of confirmatory tuning (surprise ~ 0), then a stretch of
"try something wild" rounds that mostly broke as expected (surprise ~ 1).

--------------------------------------------------------------------------------------------
THE KNOWLEDGE DOCUMENT
--------------------------------------------------------------------------------------------
`knowledge.md` is append-only and is written in the order the science happened: the hypothesis,
its grounding, its prediction, then what actually occurred. A reader can audit what was believed
before the evidence arrived. Verdicts move between four classes (Established / Refuted /
Structural / Open); a claim's history is never rewritten, only extended.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

# `control` is an intent, but NOT a mechanism hypothesis.
#
# The causality rule makes slot 0 the unchanged parent, and `proposer.propose()` REJECTS any
# proposal whose slot 0 is not `intent: "control"`. But `control` was missing from this tuple, so
# `Hypothesis.__post_init__` raised on it -- meaning `--mode composition` crashed at the moment it
# posed its first hypothesis and could never have completed a single round. Two halves of the same
# protocol disagreed about whether a control exists, and both enforced their view with a hard
# error. Found by running the mode attended, which is the only way it could have been found.
#
# A control is kept OUT of MECHANISM_INTENTS because it is not a claim about an edit: it changes
# nothing, so it cannot confirm or refute a mechanism, and folding it into the surprise rate would
# dilute the campaign's only control signal (with a 6-slot batch, by a sixth). What a control CAN
# do is fail -- and that is `is_baseline_drift`, a louder signal than a surprise, because it means
# the baseline moved under us and the round's other five comparisons are all suspect.
# `exploratory` -- THE HALF OF TRACK A THAT DID NOT EXIST. The goal statement is explicit:
# "it must be allowed to be curious. A loop that may only test predictions will only ever propose
# things it can predict, which is how a campaign came to spend nineteen of thirty-four predictions
# on 'nothing will happen' and count eighteen confirmations that a sphere stayed a sphere. An
# exploratory slot states what it is VARYING and what it will REPORT, rather than what it EXPECTS."
#
# Until now `Hypothesis(intent="exploratory")` raised ValueError, and so did `predicted=""`. So the
# only way to pose a slot was as a bet, and the trap the goal names was structural rather than
# cultural: an agent that must predict will propose what it can predict.
#
# An exploratory slot is kept OUT of MECHANISM_INTENTS for the same reason a control is: it makes
# no claim, so it can neither confirm nor refute one, and folding it into the surprise rate would
# dilute the campaign's only control signal. It resolves `described` -- a real outcome, recorded,
# and outside the confirmed/refuted arithmetic. "A labyrinth nobody forecast is a finding, not a
# failed prediction."
# A REPLICATE IS ITS OWN INTENT, added 6 August. It arises when a proposed slot repeats an experiment
# already on file: instead of refusing it, the round re-seeds it and runs it, so the wasted slot becomes
# the campaign's only measurement of its own noise floor. It is kept OUT of MECHANISM_INTENTS for the
# same reason a control and an exploratory slot are -- it makes no NEW claim about a mechanism, it tests
# whether an old one survives a different seed. Its prediction is the original's, and whether that holds
# is the robustness result.
INTENTS = ("confirmatory", "adversarial", "control", "exploratory", "replicate")
MECHANISM_INTENTS = ("confirmatory", "adversarial")

# ---------------------------------------------------------------------------------------------
# THE SECOND 70/30 -- WHERE the experiment sits, not what we believe about it.
#
# The intent mix asks "do you expect this to work?". It says nothing about WHERE in the space you
# are looking, and for months every proposal sat on or near Okuda's four published points. Cedric:
# "during the Okuda understanding it can be nice to deviate a bit to look at other objects or
# EXTREMUM objects different from those depicted in Okuda, to bring another perspective -- in a
# ratio 70 (in paper distribution) / 30 (out of paper distribution)."
#
# The two axes are orthogonal and both are informative:
#
#                    | in_paper                        | excursion
#     ---------------+---------------------------------+---------------------------------
#     confirmatory   | reproduce Fig 5a                 | "at extreme chi I expect a flat sheet"
#     adversarial    | break the Fig 5a mechanism       | "I doubt anything coherent forms here"
#
# Three reasons the excursions earn their 30%:
#   * extrema often say what a lever DOES more plainly than the operating point does;
#   * running only the published settings teaches you to reproduce his figures, not to understand
#     the system -- and the stated objective is a MAP, not a target;
#   * a phenotype nobody has named is far likelier at an extremum than at a published point, so
#     this IS the serendipity budget -- and it now has machinery behind it (an open scoreboard row,
#     the genus test, metric authoring).
#
# It is affordable only now. Before the pre-flight, an excursion that blew up entered the record as
# a phenotype; today it is refused as not-evidence and costs compute alone.
TERRITORIES = ("in_paper", "excursion")
TERRITORY_TARGET = 0.70          # fraction of MECHANISM slots that should sit in Okuda's space
TERRITORY_LOW, TERRITORY_HIGH = 0.55, 0.85
# `described` -- what an EXPLORATORY slot resolves to. Not `inconclusive`: that word means "we
# tried to check a claim and could not", and it lands the slot in the wasted bucket the Supervisor
# reads. An exploratory slot made no claim, so it cannot be inconclusive about one -- it reported
# what it said it would report, and that is a completed experiment. Kept out of the surprise
# arithmetic because there was no prediction to be surprised by.
OUTCOMES = ("confirmed", "refuted", "inconclusive", "described")

# WHAT KIND OF CLAIM IS THIS. The distinction is not pedantry -- it is the difference between an
# experiment and an anecdote, and it decides what the Interpreter is ALLOWED to write.
#
#   sufficient   "adding X produces the phenotype".  Tested by the ADDITIVE direction alone: run
#                the base, run base+X, compare. The control at slot 0 already supplies the
#                "without", so this is enforceable today.
#   necessary    "without X the phenotype dies".     Tested by the SUBTRACTIVE direction: take a
#                composition that HAS X and remove it. Nothing in the campaign has ever required
#                this, which is why no necessity claim on the books is actually supported.
#   causal       both. The strongest claim, and it costs two runs, and it may not be asserted
#                from one of them.
#
# The Interpreter may write "causes" only for `causal`. For `sufficient` the word is "sufficient";
# for `necessary` it is "required". A claim that arrives labelled `causal` without both directions
# present in the same batch is REJECTED by the Critic before any compute is spent -- the same
# treatment the mandatory control already gets, and for the same reason: the author does not
# referee their own work.
CLAIM_KINDS = ("sufficient", "necessary", "causal", "descriptive")

# Supervisor control band on the surprise rate (see module docstring).
SURPRISE_TARGET = 0.30
SURPRISE_LOW = 0.10
SURPRISE_HIGH = 0.50


@dataclass
class Hypothesis:
    """One falsifiable claim about one composition, recorded BEFORE it is run."""
    hid: str                       # stable id
    comp_hash: str
    parent_hash: str | None
    edit: str                      # the single mutation under test
    intent: str                    # confirmatory | adversarial
    claim: str                     # plain-language, falsifiable
    metric: str                    # THE metric that decides it
    predicted: str                 # e.g. "Q > 0.5"  -- recorded before the run
    rationale: str = ""
    territory: str = "in_paper"    # in_paper | excursion -- see TERRITORIES
    claim_kind: str = "descriptive"  # sufficient | necessary | causal -- see CLAIM_KINDS
    # THE REVISIT FIELDS. A slot that challenges an existing claim names it, and names the thing
    # it VARIES against the runs that established it. Both are emitted by the Proposer, never
    # filled in by code after the batch is checked -- a code-inserted revisit would be the one
    # slot in the batch exempt from the duplicate and confounder gates, which is a licensed
    # re-run of exactly the bit-identical duplicates rounds 4-8 produced, recorded as compliance.
    # THE TRACK. Declared by the Proposer, printed by round.py, and never passed to this
    # constructor -- so `track` read MISSING for all 58 runs of the 2 August campaign and every
    # analysis entry said "Tracks: 0 Track A, 0 Track B". The two-track design the roster devotes
    # a section to was never once recorded. Found by an independent audit of the loop's
    # information flow, not by the person who wrote the tracks.
    track: str = ""                 # "A" understand the mechanism | "B" reproduce the figure
    revisits: str = ""              # claim id being challenged, or ""
    confounder: str = ""            # what this slot varies vs that claim's supporting runs
    grounding: list = field(default_factory=list)   # citations from the Grounder
    round_id: int = 0
    t_posed: float = field(default_factory=time.time)

    # --- filled in after the run ---
    # WHAT AN EXPLORATORY SLOT CARRIES INSTEAD OF A PREDICTION. Stated before the run, exactly as
    # a prediction is, so the slot is still a committed experiment and not a licence to look at
    # whatever came out and call it interesting afterwards.
    varying: str = ""              # the thing being turned up, and over what range
    will_report: str = ""          # the observable that will be described, chosen in advance
    observed: dict | None = None
    outcome: str | None = None     # confirmed | refuted | inconclusive | described
    run_ids: list = field(default_factory=list)
    note: str = ""

    def __post_init__(self):
        if self.intent not in INTENTS:
            raise ValueError(f"intent {self.intent!r} not in {INTENTS}")
        if self.territory not in TERRITORIES:
            raise ValueError(f"territory {self.territory!r} not in {TERRITORIES}")
        if self.claim_kind not in CLAIM_KINDS:
            raise ValueError(f"claim_kind {self.claim_kind!r} not in {CLAIM_KINDS}")
        # AN EXPLORATORY SLOT HAS NO PREDICTION, BY DESIGN -- it has `varying` and `will_report`.
        # The rule below is right for a bet and wrong for a question, and applying it to both is
        # what made curiosity unrepresentable. It still bites where it should: a confirmatory or
        # adversarial slot with no prediction is still refused, which is the case it was written
        # for (a prediction invented after the run is not a prediction).
        if self.intent == "exploratory":
            if not (self.varying and self.will_report):
                raise ValueError("an exploratory slot must say what it is VARYING and what it "
                                 "WILL REPORT -- those are its equivalent of a prediction, and "
                                 "a slot that states neither is not an experiment either")
        elif not self.predicted:
            raise ValueError("a hypothesis without a recorded prediction is not a hypothesis; "
                             "the prediction MUST exist before the run")

    def resolve(self, observed: dict, outcome: str, run_ids=(), note=""):
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome {outcome!r} not in {OUTCOMES}")
        if self.outcome is not None:
            raise ValueError(f"{self.hid} already resolved as {self.outcome}; the record is "
                             f"append-only -- pose a new hypothesis instead of rewriting this one")
        self.observed = dict(observed)
        self.outcome = outcome
        self.run_ids = list(run_ids)
        self.note = note
        return self

    @property
    def is_surprise(self) -> bool:
        """A prediction error: the informative quadrant of the 2x2.

        A control is never a surprise -- it asserts nothing about any mechanism. See
        `is_baseline_drift` for the thing a control can tell you.
        """
        if self.outcome is None or self.outcome == "inconclusive":
            return False
        return ((self.intent == "confirmatory" and self.outcome == "refuted") or
                (self.intent == "adversarial" and self.outcome == "confirmed"))

    @property
    def is_baseline_drift(self) -> bool:
        """The control's prediction failed -- the baseline moved.

        This is not a surprise, it is an ALARM: every other slot in the round is measured as a
        difference from this control, so if the control is not where it was believed to be, the
        round's whole causal reading is suspect. It must be reported separately and loudly rather
        than averaged into a rate.
        """
        return self.intent == "control" and self.outcome == "refuted"

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- the register
class HypothesisRegister:
    """Append-only store of hypotheses. The campaign's scientific record."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._items = {}
        self.amendments = []
        if os.path.exists(path):
            fields = set(Hypothesis.__dataclass_fields__)
            for line in open(path):
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("amended"):
                    # audit metadata rides alongside the record; keep it, but the dataclass
                    # itself only takes its own fields
                    self.amendments.append({k: d[k] for k in
                                            ("hid", "amend_reason", "amended_from", "t_amended")
                                            if k in d})
                self._items[d["hid"]] = Hypothesis(**{k: v for k, v in d.items() if k in fields})

    def pose(self, h: Hypothesis):
        """Record a hypothesis BEFORE its run. Refuses to overwrite."""
        if h.hid in self._items:
            raise ValueError(f"hypothesis {h.hid} already posed")
        self._items[h.hid] = h
        self._append(h)
        return h

    def free_hid(self, hid):
        """`(hid, attempt)` -- the given id if it is free, else the next `hid@n`.

        THE ID IS DETERMINISTIC, SO A COLLISION IS PERMANENT. `R{round}.{slot}.{hash}` is built
        from the round number, the slot index and the composition; run round 2 a second time from
        the same frontier and every id is identical to the first attempt's. `pose` then raises,
        `_run_round` had no handler, and the round died at the LAST line of Act 1 -- after the
        Proposer, the Critic, the peer-review and the reflection had all been paid for. Measured
        4 August: 4.06 agent-minutes and $1.41 spent, then `ValueError: hypothesis R2.0.f4907e
        already posed` and twelve configs written for a round that never submitted a job. Worse,
        it is not recoverable by retrying: the second attempt regenerates the same ids and dies in
        the same place, so the campaign is wedged at that round forever.

        `pose`'s refusal to overwrite is RIGHT and is not being relaxed -- a claim already on the
        record must not be quietly rewritten by a later one that happens to share coordinates.
        What was wrong is that the caller had no way to say "this is round 2 being run AGAIN".
        The `@n` suffix says exactly that: the first attempt keeps its id, and the new claim is
        recorded beside it as attempt n rather than replacing it or killing the round.
        """
        if hid not in self._items:
            return hid, 1
        n = 2
        while f"{hid}@{n}" in self._items:
            n += 1
        return f"{hid}@{n}", n

    def rounds_present(self, rid):
        """How many hypotheses the registry already holds for round `rid`.

        Asked BEFORE Act 1 spends anything. A registry that already knows round 2 while the loop
        is about to run round 2 is a real inconsistency -- here, campaign state restored from git
        into a campaign that then RESUMED instead of resetting -- and the cheap moment to say so
        is before the Proposer is called, not after.
        """
        pre = f"R{rid}."
        return sum(1 for k in self._items if k.startswith(pre))

    def resolve(self, hid, observed, outcome, run_ids=(), note=""):
        h = self._items[hid].resolve(observed, outcome, run_ids, note)
        self._append(h)                       # append the resolution; the posing line stays
        return h

    def amend(self, hid, reason, predicted=None, outcome=None, note=None):
        """Correct a RECORD that misstates what was done. Appends; never deletes.

        `resolve` refuses to touch a resolved hypothesis, and that rule is right: a claim's
        history must not be rewritten to suit a later belief. But it cannot cover the case where
        the record itself is wrong about what was PREDICTED -- and that case is real.

        The vcap round is the instance. Its five points were posed before F19 with the placeholder
        `predicted: "unknown -- this is a sensitivity sweep"`, and the old scorer returned True on
        any string it could not parse, so all five resolved `confirmed`. The persisted surprise
        rate for the campaign's most informative round was therefore 0.00 -- "nothing learned" --
        while the actual predictions (recorded in the session log and in the round's own output)
        were `< 1.5`, `< 2.0`, `< 2.0`, `>= 2.0`, `>= 2.0`, of which two were refuted.

        An amendment must say WHY, is written as a new line carrying `amended_from`, and leaves
        every prior line in place. The file remains a complete audit trail: `grep amended` shows
        every correction ever made.
        """
        if not reason:
            raise ValueError("an amendment without a stated reason is a rewrite; give a reason")
        h = self._items[hid]
        prior = {"predicted": h.predicted, "outcome": h.outcome, "note": h.note}
        if predicted is not None:
            h.predicted = predicted
        if outcome is not None:
            if outcome not in OUTCOMES:
                raise ValueError(f"outcome {outcome!r} not in {OUTCOMES}")
            h.outcome = outcome
        if note is not None:
            h.note = note
        d = h.to_dict()
        d.update({"amended": True, "amend_reason": reason, "amended_from": prior,
                  "t_amended": time.time()})
        with open(self.path, "a") as f:
            f.write(json.dumps(d) + "\n")
        return h

    def _append(self, h):
        with open(self.path, "a") as f:
            f.write(json.dumps(h.to_dict()) + "\n")

    def all(self):
        return list(self._items.values())

    def by_round(self, r):
        return [h for h in self._items.values() if h.round_id == r]

    # ----------------------------------------------------------------- the control signal
    def surprise_rate(self, round_id=None):
        """Fraction of resolved MECHANISM hypotheses whose outcome contradicted its prediction.

        Controls are excluded: they assert nothing about an edit, so counting them would dilute
        the rate by one slot per round. `inconclusive` is excluded too -- an unresolved or
        uncheckable prediction is not evidence either way, which is precisely why
        `predict.score()` returns it instead of guessing `confirmed` (see predict.py P1).
        """
        hs = [h for h in (self.by_round(round_id) if round_id is not None else self.all())
              if h.outcome in ("confirmed", "refuted") and h.intent in MECHANISM_INTENTS]
        if not hs:
            return None
        return sum(1 for h in hs if h.is_surprise) / len(hs)

    def baseline_drift(self, round_id=None):
        """The controls that failed their prediction. Non-empty means read the round with care."""
        return [h for h in (self.by_round(round_id) if round_id is not None else self.all())
                if h.is_baseline_drift]

    def mix(self, round_id=None):
        """Confirmatory/adversarial split among MECHANISM slots that produced evidence.

        The 70/30 setpoint is about how the batch spends its *edits*; the control is a fixed
        overhead of the design, not a choice, so including it would report 6 slots as 50/33 when
        the agent in fact chose 60/40.

        `inconclusive` is excluded so that this and `surprise_rate` describe the SAME population.
        They are printed side by side and read as one statement about the round; computing them
        over different sets makes that statement false. Concretely: eight hypotheses posed by
        aborted runs sat unresolved under round 1 (a consequence of the round counter restarting
        every process) and dragged the reported mix to 92/8 -- which is the steer the Supervisor
        gave the Proposer, and the Proposer duly obeyed it.
        """
        hs = [h for h in (self.by_round(round_id) if round_id is not None else self.all())
              if h.intent in MECHANISM_INTENTS and h.outcome in ("confirmed", "refuted")]
        n = len(hs) or 1
        return {i: sum(1 for h in hs if h.intent == i) / n for i in MECHANISM_INTENTS}

    def territory_mix(self, round_id=None):
        """Fraction of resolved mechanism slots that sat inside Okuda's space."""
        hs = [h for h in (self.by_round(round_id) if round_id is not None else self.all())
              if h.intent in MECHANISM_INTENTS and h.outcome in ("confirmed", "refuted")]
        n = len(hs) or 1
        return {t: sum(1 for h in hs if h.territory == t) / n for t in TERRITORIES}

    def novelty_yield(self, round_id=None, novel_phenotypes=()):
        """Of the excursions that produced evidence, what fraction found something new?

        The counterpart to the surprise rate, and the control signal for the territory mix. An
        excursion "found something" if its phenotype is one nobody had listed. If excursions stop
        yielding, spend the budget back inside the paper; if they keep yielding, widen.
        """
        ex = [h for h in (self.by_round(round_id) if round_id is not None else self.all())
              if h.territory == "excursion" and h.outcome in ("confirmed", "refuted")]
        if not ex:
            return None
        novel = {str(p).strip().lower() for p in novel_phenotypes}
        hit = sum(1 for h in ex
                  if str((h.observed or {}).get("analyst_consensus", "")).strip().lower() in novel)
        return hit / len(ex)

    def advise_territory(self, round_id=None, novel_phenotypes=()):
        """(in_paper_fraction, why). Steered by novelty yield, as the intent mix is by surprise."""
        y = self.novelty_yield(round_id, novel_phenotypes)
        cur = self.territory_mix(round_id)["in_paper"]
        if y is None:
            return TERRITORY_TARGET, ("no excursion has produced evidence yet -- hold the 70/30 "
                                      "default")
        if y <= 0.0:
            return TERRITORY_HIGH, (f"novelty yield {y:.2f}: the excursions are finding nothing "
                                    f"new. Spend the budget inside the paper (currently "
                                    f"{cur:.0%} in-paper).")
        if y >= 0.5:
            return TERRITORY_LOW, (f"novelty yield {y:.2f}: excursions keep finding phenotypes "
                                   f"nobody listed. Widen -- the map is bigger than the paper.")
        return TERRITORY_TARGET, f"novelty yield {y:.2f} -- hold 70/30"

    def advise_mix(self, round_id=None):
        """The Supervisor's rule. Returns (confirmatory_fraction, why).

        This is where the 70/30 discipline actually lives: not as a quota, but as a closed loop
        on the observed surprise rate.
        """
        s = self.surprise_rate(round_id)
        if s is None:
            return 0.70, "no resolved hypotheses yet -- start at the 70/30 default"
        if s < SURPRISE_LOW:
            return 0.50, (f"surprise {s:.2f} < {SURPRISE_LOW}: the batch is confirming what we "
                          f"already believe (drifting to 100/0, near-zero information). "
                          f"Push ADVERSARIAL.")
        if s > SURPRISE_HIGH:
            return 0.85, (f"surprise {s:.2f} > {SURPRISE_HIGH}: almost everything breaks "
                          f"(drifting to 0/100, no map accumulates). Push CONFIRMATORY.")
        return 0.70, f"surprise {s:.2f} in the productive band -- hold 70/30"

    # ----------------------------------------------------------------- the knowledge document
    def render_knowledge(self, path, ledger=None, round_id=None):
        """Append this round's science to the knowledge document, in the order it happened."""
        hs = sorted(self.by_round(round_id) if round_id is not None else self.all(),
                    key=lambda h: h.t_posed)
        s = self.surprise_rate(round_id)
        mix = self.mix(round_id)
        n_ctl = sum(1 for h in hs if h.intent == "control")
        drift = [h for h in hs if h.is_baseline_drift]
        lines = [f"\n## Round {round_id} — {time.strftime('%Y-%m-%d %H:%M')}", "",
                 f"- batch: {len(hs) - n_ctl} mechanism hypotheses "
                 f"({mix['confirmatory']:.0%} confirmatory / {mix['adversarial']:.0%} adversarial)"
                 + (f" + {n_ctl} control" if n_ctl else ""),
                 f"- **surprise rate: {s:.2f}**" if s is not None else "- surprise rate: n/a",
                 f"- supervisor: {self.advise_mix(round_id)[1]}", ""]
        if drift:
            lines += [f"> ⚠️ **BASELINE DRIFT — {len(drift)} control(s) failed their prediction.** "
                      f"Every other slot this round is measured as a difference from a control "
                      f"that is not where it was believed to be, so read the causal claims below "
                      f"with care.", ""]
            for h in drift:
                lines.append(f"> - `{h.comp_hash}` predicted `{h.predicted}` → "
                             f"observed `{json.dumps(h.observed)}`")
            lines.append("")

        for label, pred in (("### Validated", "confirmed"), ("### Refuted", "refuted"),
                            ("### Open / inconclusive", "inconclusive")):
            sel = [h for h in hs if h.outcome == pred]
            lines.append(label)
            if not sel:
                lines.append("- _(none this round)_")
            for h in sel:
                flag = " 🔥 **surprise**" if h.is_surprise else ""
                lines.append(f"- **{h.claim}**{flag}")
                lines.append(f"  - edit `{h.edit}` on `{h.comp_hash}` · intent *{h.intent}*")
                lines.append(f"  - predicted `{h.predicted}` on `{h.metric}` "
                             f"→ observed `{json.dumps(h.observed)}`")
                if h.grounding:
                    lines.append(f"  - grounding: {'; '.join(h.grounding)}")
                if h.note:
                    lines.append(f"  - {h.note}")
            lines.append("")

        if ledger:
            lines += ["### Ledger movement this round", ""]
            for k, v in ledger.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        with open(path, "a") as f:
            f.write("\n".join(lines))
        return path


# --------------------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        reg = HypothesisRegister(os.path.join(d, "hypotheses.jsonl"))

        # a hypothesis MUST carry a prediction
        try:
            Hypothesis(hid="h0", comp_hash="C1", parent_hash=None, edit="-extrude",
                       intent="adversarial", claim="x", metric="Q", predicted="")
            assert False
        except ValueError as e:
            print("refused as designed:", str(e)[:70], "...")

        # the campaign's central hypothesis, posed before any run
        h1 = reg.pose(Hypothesis(
            hid="R1.h1", comp_hash="Ca51d8ef877b", parent_hash="Cd13473bf1c3",
            edit="-extrude", intent="adversarial",
            claim="the tube survives removal of the extrusion forcing",
            metric="aspect_final", predicted="aspect > 3",
            rationale="Okuda's tube is a growth-driven quasi-static equilibrium, so a faithful "
                      "composition should not need an explicit outward force",
            grounding=["Okuda et al. 2018 Sci Rep 8:2386, Eq. 1 (quasi-static force balance)"],
            round_id=1))
        h2 = reg.pose(Hypothesis(
            hid="R1.h2", comp_hash="C5e315998af4", parent_hash=None,
            edit="theta: relax_iters 30->60", intent="confirmatory",
            claim="more relaxation degrades a forced tube",
            metric="aspect_final", predicted="aspect drops > 30%",
            grounding=["round 41 field notes"], round_id=1))

        reg.resolve("R1.h1", {"aspect_final": 1.02}, "refuted",
                    note="the forcing is necessary in THIS composition; consistent with R41")
        reg.resolve("R1.h2", {"aspect_final": 1.1}, "confirmed")

        # THE CONTROL -- this construction used to raise, which is why --mode composition could
        # never complete a round. It must be posable, must not enter the surprise rate, and must
        # raise a BASELINE DRIFT alarm when its own prediction fails.
        hc = reg.pose(Hypothesis(
            hid="R1.h0", comp_hash="C5e315998af4", parent_hash=None,
            edit="control (parent unchanged)", intent="control",
            claim="the parent, unchanged -- the control every knockout is read against",
            metric="protr_peak", predicted="protr_peak 2.0-3.5", round_id=1))
        assert reg.surprise_rate(1) == 0.0, "the control must not enter the surprise rate"
        assert abs(reg.mix(1)["confirmatory"] - 0.5) < 1e-9, "the control must not enter the mix"
        reg.resolve("R1.h0", {"protr_peak": 9.1}, "refuted", note="baseline moved")
        assert hc.is_baseline_drift and not hc.is_surprise, "a failed control is drift, not surprise"
        assert reg.surprise_rate(1) == 0.0, "a failed control still must not enter the rate"
        assert len(reg.baseline_drift(1)) == 1
        print("control: posable, excluded from surprise/mix, raises baseline drift -- OK")

        print(f"\nsurprise rate round 1: {reg.surprise_rate(1):.2f}")
        print("mix:", {k: f'{v:.0%}' for k, v in reg.mix(1).items()})
        frac, why = reg.advise_mix(1)
        print(f"supervisor -> {frac:.0%} confirmatory next round\n  because: {why}")

        kp = reg.render_knowledge(os.path.join(d, "knowledge.md"),
                                  ledger={"Established": 0, "Refuted": 1, "Open": 1}, round_id=1)
        print("\n--- knowledge.md ---")
        print(open(kp).read()[:900])
        print("hypothesis protocol OK")
