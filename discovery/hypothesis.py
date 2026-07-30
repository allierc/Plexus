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

INTENTS = ("confirmatory", "adversarial")
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
        """A prediction error: the informative quadrant of the 2x2."""
        if self.outcome is None or self.outcome == "inconclusive":
            return False
        return ((self.intent == "confirmatory" and self.outcome == "refuted") or
                (self.intent == "adversarial" and self.outcome == "confirmed"))

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- the register
class HypothesisRegister:
    """Append-only store of hypotheses. The campaign's scientific record."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._items = {}
        if os.path.exists(path):
            for line in open(path):
                if line.strip():
                    d = json.loads(line)
                    self._items[d["hid"]] = Hypothesis(**d)

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

    def _append(self, h):
        with open(self.path, "a") as f:
            f.write(json.dumps(h.to_dict()) + "\n")

    def all(self):
        return list(self._items.values())

    def by_round(self, r):
        return [h for h in self._items.values() if h.round_id == r]

    # ----------------------------------------------------------------- the control signal
    def surprise_rate(self, round_id=None):
        """Fraction of resolved hypotheses whose outcome contradicted its prediction."""
        hs = [h for h in (self.by_round(round_id) if round_id is not None else self.all())
              if h.outcome in ("confirmed", "refuted")]
        if not hs:
            return None
        return sum(1 for h in hs if h.is_surprise) / len(hs)

    def mix(self, round_id=None):
        hs = self.by_round(round_id) if round_id is not None else self.all()
        n = len(hs) or 1
        return {i: sum(1 for h in hs if h.intent == i) / n for i in INTENTS}

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
        lines = [f"\n## Round {round_id} — {time.strftime('%Y-%m-%d %H:%M')}", "",
                 f"- batch: {len(hs)} hypotheses "
                 f"({mix['confirmatory']:.0%} confirmatory / {mix['adversarial']:.0%} adversarial)",
                 f"- **surprise rate: {s:.2f}**" if s is not None else "- surprise rate: n/a",
                 f"- supervisor: {self.advise_mix(round_id)[1]}", ""]

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

        print(f"\nsurprise rate round 1: {reg.surprise_rate(1):.2f}")
        print("mix:", {k: f'{v:.0%}' for k, v in reg.mix(1).items()})
        frac, why = reg.advise_mix(1)
        print(f"supervisor -> {frac:.0%} confirmatory next round\n  because: {why}")

        kp = reg.render_knowledge(os.path.join(d, "knowledge.md"),
                                  ledger={"Established": 0, "Refuted": 1, "Open": 1}, round_id=1)
        print("\n--- knowledge.md ---")
        print(open(kp).read()[:900])
        print("hypothesis protocol OK")
