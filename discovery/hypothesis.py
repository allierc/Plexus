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
INTENTS = ("confirmatory", "adversarial", "control")
MECHANISM_INTENTS = ("confirmatory", "adversarial")
OUTCOMES = ("confirmed", "refuted", "inconclusive")

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
    grounding: list = field(default_factory=list)   # citations from the Grounder
    round_id: int = 0
    t_posed: float = field(default_factory=time.time)

    # --- filled in after the run ---
    observed: dict | None = None
    outcome: str | None = None     # confirmed | refuted | inconclusive
    run_ids: list = field(default_factory=list)
    note: str = ""

    def __post_init__(self):
        if self.intent not in INTENTS:
            raise ValueError(f"intent {self.intent!r} not in {INTENTS}")
        if not self.predicted:
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
