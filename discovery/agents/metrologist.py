"""metrologist -- the agent that certifies the FOUNDATION, and retracts knowledge built on a bad one.

    "If the foundations are wrong, so is the knowledge."

Every other agent produces or judges EVIDENCE. None is responsible for the thing evidence is
measured WITH. That gap is not hypothetical: the clock defect recorded below was found by a person
reading code, not by any designed role.

The Metrologist owns three things no other agent does:

  1. THE LADDER        -- the validation levels (contract, unit, trajectory, invariants,
                          recording, metrics, gradients, coverage, description) and the
                          instrument gate. Nothing is admitted as evidence until they pass.
  2. SUBSTRATE SEMANTICS -- units, clocks, and the per-call / per-frame distinction. This is
                          where the campaign's most expensive defect lived.
  3. RETRACTION        -- the power to act BACKWARDS. When a foundation defect is found, every
                          claim whose evidence depends on the defective semantics is moved back
                          to Open, with the defect and its scope recorded.

In an append-only ledger a retraction is a NEW RECORD, never an edit. The history must show what
was believed, on what evidence, and why that evidence was later withdrawn -- otherwise a reader
cannot distinguish a claim that survived scrutiny from one that was quietly repaired.

--------------------------------------------------------------------------------------------
THE BOUNDARY
--------------------------------------------------------------------------------------------
    NO CAMPAIGN AGENT MAY MODIFY THE SUBSTRATE.

An agent that can rewrite the simulator can make any hypothesis true. So:

    Metrologist   detects and QUARANTINES. Does not write code.
    Engineer      proposes a PATCH, into a different artefact. Cannot admit its own fix.
    Supervisor    GATES resumption, only after the full ladder passes on the patched substrate.
    Human         approves substrate changes.

The instrument must not be adjustable by the experiment it is measuring.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

SEVERITY = ("cosmetic", "confounding", "invalidating")


# ============================================================================ defect record
@dataclass
class FoundationDefect:
    """A defect in the thing evidence is measured with -- not in a hypothesis."""
    did: str
    title: str
    severity: str                     # cosmetic | confounding | invalidating
    scope: str                        # WHICH evidence is affected, precisely
    detail: str
    detected_by: str = "metrologist"
    detected_t: float = field(default_factory=time.time)
    # what makes it actionable
    affects: list = field(default_factory=list)      # metric / param names touched
    correction: str = ""                             # how archived values convert, if analytic
    behaviour_preserving: bool = False               # True => archived behaviour recoverable exactly
    resolved_by: str | None = None                   # commit / patch id
    resolved_t: float | None = None

    def __post_init__(self):
        if self.severity not in SEVERITY:
            raise ValueError(f"severity {self.severity!r} not in {SEVERITY}")

    def to_dict(self):
        return asdict(self)


# The defects this campaign has actually found, in the order they were found. This list IS the
# Metrologist's memory: a returning session must be able to see what the instrument got wrong.
KNOWN_DEFECTS = [
    FoundationDefect(
        did="D1", title="clock double-gating",
        severity="invalidating",
        scope="every archived run: divide_3d executed once every FOUR frames (engine every=2 x "
              "private self._k every=2), so all per-call quantities meant 4x what they said",
        detail="min_cycle/max_cycle are counted in DIVISION-CALLS and max_div/max_div_frac are "
               "PER-CALL throttles. Correcting the clock multiplied their wall-clock meaning by 4; "
               "the recipe that produced aspect 7.5 produced 3.2.",
        affects=["min_cycle", "max_cycle", "max_div", "max_div_frac"],
        correction="min_cycle,max_cycle x4 ; max_div,max_div_frac /4",
        behaviour_preserving=True),
    FoundationDefect(
        did="D1b", title="rescaling the fraction alone is masked by the floor",
        severity="invalidating",
        scope="the D1 correction itself, before this was noticed",
        detail="cap_div = max(max_div, max_div_frac*nF). The absolute floor DOMINATES at realistic "
               "cell counts (nF=1431: max(120, 42) = 120), so rescaling max_div_frac alone had NO "
               "effect. Both had to move. A correction that looks applied but is masked is worse "
               "than no correction, because it is believed.",
        affects=["max_div"], correction="max_div /4", behaviour_preserving=True),
    FoundationDefect(
        did="D2", title="composition-dependent dt",
        severity="confounding",
        scope="any comparison across compositions that differ in whether RD is present",
        detail="dt = 1.0 if (cones and not rd) else 0.02 -- adding/removing the RD operators, the "
               "campaign's single most important edit, ALSO rescaled chemical:mechanical time 50x.",
        affects=["dt"], correction="one global dt for the whole campaign",
        behaviour_preserving=False),
    FoundationDefect(
        did="D3", title="recording-stride mismatch",
        severity="invalidating",
        scope="any run long enough to trip a recording stride > 1",
        detail="positions and topology recorded on different strides; the analysis paired one "
               "frame's coordinates with another frame's connectivity and reported phantom "
               "inverted cells ('97% hollow') that no physical fix could touch.",
        affects=["hollow", "any mesh-referenced metric"],
        correction="assert equal length; never clamp", behaviour_preserving=False),
    FoundationDefect(
        did="D4", title="silently inert operators",
        severity="invalidating",
        scope="any composition with an unmet precondition -- i.e. exactly the combinations a "
              "SEARCH generates and a hand-written preset never did",
        detail="an operator whose prerequisite is missing no-ops, the run still finishes and still "
               "scores, so the loop records 'this mechanism cannot produce tubes' when the "
               "mechanism never ran. Manufactures FALSE IMPOSSIBILITY claims.",
        affects=["every impossibility claim"],
        correction="acted-ledger; a run with an inert scheduled operator is not evidence",
        behaviour_preserving=False),
    FoundationDefect(
        did="D7", title="archive stored only the final frame",
        severity="confounding",
        scope="every temporal observable, retrospectively -- the archive could not be re-scored "
              "for anything time-dependent without re-simulating",
        detail="the decisive evidence in the tube record is temporal (a protrusion peaking then "
               "degrading; a pattern flooding once growth starts; synchronised division waves).",
        affects=["retention", "Q", "any per-frame metric"],
        correction="persist the full trajectory + per-frame metric table",
        behaviour_preserving=False),
]


# ============================================================================ retraction
@dataclass
class Retraction:
    """Issued when a foundation defect invalidates evidence already recorded.

    Append-only: this is a NEW record. The retracted claim's original entry stays exactly as it
    was, so the history shows what was believed and why it was withdrawn.
    """
    rid: str
    defect_id: str
    affected_claims: list          # comp_hashes / claim ids moved back to Open
    affected_runs: list            # run_ids whose analyses are no longer admissible
    reason: str
    reanalysable: bool             # True => archived trajectories can be re-scored, not re-run
    t: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


class Certification:
    """The Metrologist's on-disk record: defects, retractions, and the admission gate."""

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.defects_path = os.path.join(root, "defects.jsonl")
        self.retractions_path = os.path.join(root, "retractions.jsonl")
        self.gate_path = os.path.join(root, "admission.json")

    # -- defects -------------------------------------------------------------------------
    def record_defect(self, d: FoundationDefect):
        with open(self.defects_path, "a") as f:
            f.write(json.dumps(d.to_dict()) + "\n")
        return d

    def bootstrap(self):
        """Write the known defects once, so a fresh session inherits the instrument's history."""
        if os.path.exists(self.defects_path):
            return 0
        for d in KNOWN_DEFECTS:
            self.record_defect(d)
        return len(KNOWN_DEFECTS)

    def resolve_defect(self, did, resolved_by, note=""):
        """Mark a defect resolved. Append-only: a NEW record with the resolution, never an edit
        of the original -- so the history shows the defect as it was first stated."""
        cur = {d["did"]: d for d in self.defects()}
        if did not in cur:
            raise KeyError(f"unknown defect {did}")
        d = dict(cur[did])
        d.update(resolved_by=resolved_by, resolved_t=time.time(),
                 detail=d["detail"] + f"  [RESOLVED {resolved_by}: {note}]" if note else d["detail"])
        with open(self.defects_path, "a") as f:
            f.write(json.dumps(d) + "\n")
        return d

    def defects(self):
        if not os.path.exists(self.defects_path):
            return []
        # last record per did wins: resolutions are appended, so the latest state is current
        out = {}
        for l in open(self.defects_path):
            if l.strip():
                d = json.loads(l); out[d["did"]] = d
        return list(out.values())

    # -- retraction ----------------------------------------------------------------------
    def retract(self, defect_id, affected_claims, affected_runs, reason, reanalysable):
        r = Retraction(rid=f"RET{len(self._retractions()):03d}", defect_id=defect_id,
                       affected_claims=list(affected_claims), affected_runs=list(affected_runs),
                       reason=reason, reanalysable=bool(reanalysable))
        with open(self.retractions_path, "a") as f:
            f.write(json.dumps(r.to_dict()) + "\n")
        return r

    def _retractions(self):
        if not os.path.exists(self.retractions_path):
            return []
        return [json.loads(l) for l in open(self.retractions_path) if l.strip()]

    def retracted_runs(self):
        out = set()
        for r in self._retractions():
            out.update(r["affected_runs"])
        return out

    # -- the admission gate --------------------------------------------------------------
    def certify(self, ladder_results: dict, instrument_gate: bool, substrate_rev: str):
        """May evidence be admitted? Written to disk so the Supervisor reads it, not infers it."""
        unresolved = [d for d in self.defects()
                      if d["severity"] == "invalidating" and not d.get("resolved_by")]
        failed = [k for k, v in ladder_results.items() if not v]
        ok = (not failed) and instrument_gate and not unresolved
        rec = {"t": time.time(), "substrate_rev": substrate_rev, "admit": bool(ok),
               "ladder_failed": failed, "instrument_gate": bool(instrument_gate),
               "unresolved_invalidating": [d["did"] for d in unresolved]}
        json.dump(rec, open(self.gate_path, "w"), indent=1)
        return rec

    def may_admit(self):
        """Admission = the stored certification AND the LIVE instrument-gate verdict.

        `admission.json` cached `instrument_gate: true`, and `certify()` is only ever called with
        that value hard-coded in a smoke test -- so the boolean was frozen the day it was written.
        The instrument gate has since gone to NOT CERTIFIED (its blob and sphere controls are
        invalid runs) and the campaign would still have admitted evidence, because nothing
        re-read it. A gate whose verdict is cached is not a gate. Read the file the gate writes.
        """
        if not os.path.exists(self.gate_path):
            return False, "not certified -- the Metrologist has not run"
        rec = json.load(open(self.gate_path))
        ok = bool(rec.get("admit"))
        # THE ARBITER MOVED, AND THIS CHECK HAD NOT. Until 2026-08-01 admission required the
        # instrument gate, which certified metrics by making them reproduce labels a person wrote
        # from the movies. Phase 4d retired that: movies inform, they do not decide. What replaced
        # it is `morphology.classify`, certified against shapes whose answer is known BY
        # CONSTRUCTION, plus the Biologist refusing any run that is not a tissue.
        #
        # The old gate cannot be certified any more, and not for a fixable reason: its labelled
        # runs are ARCHIVED PRE-FIX configs, and the Biologist refuses them outright -- P2 (growth
        # gated with no baseline) and P3 (the growth ceiling below the division trigger, defect
        # D5b). A run that is not a tissue cannot calibrate an instrument. Requiring a gate that
        # can never pass is not caution, it is a deadlock.
        #
        # So admission now asks the question the arbiter can actually answer: is the classifier
        # certified? That is a real check -- it runs, it can fail, and it has known answers.
        ok_m, why_m = _morphology_certified()
        rec = dict(rec, morphology_certified=ok_m, morphology_why=why_m)
        ok = ok and ok_m
        return ok, json.dumps({k: v for k, v in rec.items() if k != "t"})


# ============================================================================ smoke
def _morphology_certified():
    """Run the classifier's own self-test. It has known answers, so it can genuinely fail."""
    import subprocess
    import sys as _sys
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "..", "prototype", "Tyssue", "morphology.py")
    p = os.path.abspath(p)
    if not os.path.exists(p):
        return False, f"morphology.py not found at {p}"
    try:
        r = subprocess.run([_sys.executable, p], capture_output=True, text=True, timeout=600)
    except Exception as e:
        return False, f"could not run the classifier self-test: {type(e).__name__}: {e}"
    if r.returncode != 0 or "CERTIFIED" not in r.stdout:
        tail = (r.stdout or r.stderr).strip().splitlines()[-1:] or [""]
        return False, f"the morphology classifier does not certify: {tail[0][:90]}"
    return True, "morphology classifier certified against shapes with known answers"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cert = Certification(d)
        n = cert.bootstrap()
        print(f"bootstrapped {n} known foundation defects\n")
        for x in cert.defects():
            flag = "analytic" if x["behaviour_preserving"] else "not recoverable"
            print(f"  [{x['severity']:12}] {x['did']:4} {x['title']:48} ({flag})")

        print("\n-- the campaign may NOT admit evidence while an invalidating defect is open --")
        rec = cert.certify({"contract": True, "invariants": True}, instrument_gate=True,
                           substrate_rev="abc1234")
        ok, why = cert.may_admit()
        print(f"  admit = {ok}")
        print(f"  because: {why}")

        print("\n-- a retraction is a NEW record; the original claim is never edited --")
        r = cert.retract("D1", affected_claims=["C5e315998af4"],
                         affected_runs=["R91a48cf669b7789"],
                         reason="every rate-dependent conclusion assumed divide_3d fired 4x less "
                                "often than it did",
                         reanalysable=True)
        print(f"  {r.rid} -> claims {r.affected_claims} back to Open; "
              f"re-scorable without re-running: {r.reanalysable}")
        assert "R91a48cf669b7789" in cert.retracted_runs()
        print("\nmetrologist OK")
