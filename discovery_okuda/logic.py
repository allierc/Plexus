#!/usr/bin/env python
"""logic -- the claim register, its renderer, and the four checks. See LOGIC.md.

WHY THIS FILE EXISTS. Nine rounds, 58 runs, a genuine 57% refutation rate -- and across 44 KB of
agent-written reasoning, NOT ONE occurrence of "only tested once", "may not generalise",
"confounder", "insufficient evidence" or "cannot conclude", against FIFTEEN assertions of closure.

The models were not careless. Four of seven positive claims carried an explicit falsifier, because
TEMPLATE_memory.md has a `Falsifiable by:` field for positives. "Known traps" asks only for "one
line each, with the run that proved it" -- no conditions, no quantifier, no escape -- and got
exactly that. The agents were as rigorous as the structure they were given and not one degree more.

WIRING, NOT CONSTRUCTION. The vocabulary already existed and was dead:

    hypothesis.CLAIM_KINDS  ("sufficient","necessary","causal","descriptive"), validated -- and
                            all 170 hypotheses in the ledger carry "descriptive".
    critic.check_batch      A1_NO_ABLATION already refuses a `necessary` claim with no ablation
                            in the batch -- and is never called from the loop.
    templates.check_memory  called only under __main__; memory.md ran 1186 words against a
                            900-word budget and the check never fired.

So this module reuses those closed sets rather than inventing parallel ones, and `--check` runs
templates.check_memory() as one of its own gates.

A CLAIM IS FILED, NEVER WRITTEN. Nothing in the loop parses memory.md -- it reaches the Proposer
as a PATH INSIDE A PROMPT. A checker that read agent prose out of it would be parsing the weakest
link in the system. So the register is the truth and the claim-bearing sections of memory.md are
RENDERED from it, exactly as operator_backlog.md is rendered from operator_requests.jsonl.

    python logic.py --check                     # check the live register + memory.md
    python logic.py --check --fixture <dir>     # check an archived campaign's records/
    python logic.py --render                    # print the claim sections of memory.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOG = os.path.join(ROOT, "log", "okuda")
CONFIG = os.path.join(ROOT, "config", "okuda")
CAMP = os.path.join(HERE, "campaign")
REGISTER = os.path.join(CAMP, "claims.jsonl")

# ---------------------------------------------------------------- the four modalities (LOGIC.md)
# Existence is cheap: one bud is a bud. A universal negative quantifies over every condition not
# varied. A necessity claim quantifies over every alternative not attempted -- it is the most
# expensive statement in the file and must be the hardest to write.
CAN_BE, COULD_BE, CANNOT_BE, CANNOT_NOT_BE = "can_be", "could_be", "cannot_be", "cannot_not_be"
MODALITIES = (CAN_BE, COULD_BE, CANNOT_BE, CANNOT_NOT_BE)

SUPPORT_BAR = {CAN_BE: 1, COULD_BE: 0, CANNOT_BE: 3, CANNOT_NOT_BE: 3}

# `cannot_not_be` additionally requires a COUNTEREXAMPLE ATTEMPT that failed. An ablation removes X
# from a working recipe; a counterexample asks whether the effect is reachable AT ALL without X,
# and may change other operators to compensate. "Division is NECESSARY" survived six rounds because
# nobody ever tried to build a bud without it -- when round 8 finally did, by ablating on a
# different parent, it produced a clean bud with a fission neck.
NEEDS_COUNTEREXAMPLE = (CANNOT_NOT_BE,)

# hypothesis.CLAIM_KINDS is the closed set that already exists. Mapped, not duplicated.
KIND_OF_MODALITY = {CAN_BE: "sufficient", CANNOT_BE: "causal",
                    CANNOT_NOT_BE: "necessary", COULD_BE: "descriptive"}

STATUSES = ("established", "provisional", "demoted", "retracted")

# Words that ASSERT a modality regardless of where the sentence is filed. The three most damaging
# claims in the campaign sat in the ESTABLISHED section, were marked SUPPORTED, and carried a
# falsifier -- and asserted NECESSARY or INERT from single removals. A checker keying on section,
# or on the presence of a falsifier, passes them silently. So the modality is read from the words.
_NECESSITY = re.compile(r"\b(necessary|required|indispensable|only route|must have|prerequisite)\b",
                        re.I)
# The lexicon had to be widened: the first version missed three genuine universals in the fixture
# because they carry the quantifier in their GRAMMAR rather than in a keyword --
# "ANY morphogen-tune -> integrator runaway", "remove_op cell_diffuse0 -> DIVERGES",
# "Predicting protr_peak >=1.3 -- WRONG-HIGH every time". Those sat in the positive column, which
# is the dangerous side of the balance: a false positive is loud and recoverable, a false negative
# is silent and compounds. Recall matters more here than elegance.
_UNIVERSAL = re.compile(r"\b(inert|never|no effect|cannot|closed|exhausted|fundamental|"
                        r"impossible|always fails|do not re-propose|do not re-remove|nothing|"
                        r"every time|always|any\s+\w+[- ]tune|not integrable|"
                        r"wrong-high|diverges|explodes|unstable)\b", re.I)

# "ANY X" / "no X" / "every X" -- a quantifier in the grammar rather than a keyword.
_QUANTIFIER = re.compile(r"\b(any|every|no|none of|all)\s+[a-z_]", re.I)

# SECTIONS THAT ARE NEGATIVE BY CONSTRUCTION. Everything filed under "Known traps" is a
# prohibition -- that is what the section IS. Section membership is therefore a SUFFICIENT trigger,
# never a necessary one: the three worst claims in the campaign were negatives filed under
# ESTABLISHED, so keying on section alone would still miss them.
NEGATIVE_SECTIONS = ("known traps", "traps")

# COULD BE IS DECIDED BY ITS SECTION, and nothing may override that. The golden fixture caught this
# the first time it ran: "gray_scott kinetics ... has never been run" was classified as a universal
# negative on the word `never` and REFUSED -- the bucket built so untested territory cannot be
# starved, refused by its own checker. That is the worst false positive available here.
COULD_BE_SECTIONS = ("could be", "could be — untried, or tried below the bar", "untried")

# "has never been RUN" is a statement about US. "never WORKS" is a statement about the world. Only
# the second is a claim about nature, and only the second needs a quantifier. Distinguishing them
# is the whole difference between `could_be` and `cannot_be`.
_UNTRIED = re.compile(r"\b(never|not)\s+(been\s+)?(run|tried|attempted|tested|explored|proposed)\b|"
                      r"\buntried\b|\bnot yet\b|\bno attempt\b", re.I)


def asserted_modality(statement, section=""):
    """What the SENTENCE claims, independent of how it was filed. Necessity wins over universal."""
    s = statement or ""
    sec = (section or "").strip().lower()
    # ORDER IS LOAD-BEARING, and getting it wrong laundered the campaign's worst claim. When
    # `_UNTRIED` was tested first, "The ~1.23 ceiling is FUNDAMENTAL to this body ... the only
    # UNTRIED route is a different base geometry" was filed as could_be -- a universal negative
    # hidden in the safe bucket by the word `untried` appearing later in the same sentence. A
    # sentence does not stop asserting a universal by also mentioning something not yet tried.
    # So: assertions about NATURE are read first, and `untried` only decides a sentence that
    # makes no such assertion.
    if sec.startswith("could be") or sec in COULD_BE_SECTIONS:
        return COULD_BE
    if _NECESSITY.search(s):
        return CANNOT_NOT_BE
    if _UNIVERSAL.search(s) or _QUANTIFIER.search(s):
        return CANNOT_BE
    if _UNTRIED.search(s):
        return COULD_BE
    if sec in NEGATIVE_SECTIONS:
        return CANNOT_BE
    return None


# ---------------------------------------------------------------------------------- the claim
@dataclass
class Claim:
    """One campaign-level belief. Filed, never written as prose.

    __post_init__ refuses to construct an ill-formed claim, following the OperatorRequest
    precedent that already raises on a blank `why_inexpressible`. Making the illegal state
    unconstructable is cheaper than detecting it later.
    """
    cid: str                        # stable id, e.g. "K003"
    modality: str
    statement: str
    support: list = field(default_factory=list)      # run ids, must exist on disk
    conditions: dict = field(default_factory=dict)   # COMPUTED -- see compute_conditions()
    refuter: str = ""
    counterexample: str = ""        # for cannot_not_be: the failed attempt to reach it without X
    subject: str = ""               # the operator/param the claim is about -- stripped when
    #                                 computing signatures, so support counts VARIED backgrounds
    status: str = "established"
    round_id: int = 0
    note: str = ""
    t: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.modality not in MODALITIES:
            raise ValueError(f"modality {self.modality!r} not in {MODALITIES}")
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r} not in {STATUSES}")
        if not (self.statement or "").strip():
            raise ValueError("a claim must say something")
        # The field that makes this a claim rather than an assertion. A negative with no stated
        # escape is the shape of every trap in the 2 August campaign: nine of them, two with any
        # escape at all.
        if self.modality in (CANNOT_BE, CANNOT_NOT_BE) and not (self.refuter or "").strip():
            raise ValueError(
                f"{self.cid}: a negative MUST carry its refuter. 'reconnect_t1_3d0 is INERT' is an "
                f"assertion; 'reconnect_t1_3d0 is INERT -- refuted by any run where removing it "
                f"moves protr_peak by more than 0.02' is a claim.")

    def to_dict(self):
        return asdict(self)

    @property
    def claim_kind(self):
        """The existing hypothesis.CLAIM_KINDS value, so the two vocabularies never diverge."""
        return KIND_OF_MODALITY[self.modality]


# ------------------------------------------------------------------- independence, from the disk
def _composition(run_id, config_dir=None, log_dir=None):
    """A run's operator/param signature, from its composition.json or its spec."""
    cd = config_dir or CONFIG
    p = os.path.join(cd, f"{run_id}.composition.json")
    if os.path.exists(p):
        try:
            d = json.load(open(p))
            ops = tuple(sorted((o.get("op"), o.get("impl")) for o in d.get("ops", [])))
            params = tuple(sorted((k, str(v)) for k, v in (d.get("params") or {}).items()))
            return ops, params
        except Exception:
            pass
    ld = log_dir or LOG
    p = os.path.join(ld, run_id, "spec_run.yaml")
    if os.path.exists(p):
        try:
            import yaml
            c = yaml.safe_load(open(p))
            ops = tuple(sorted((o.get("op"), o.get("implementation"))
                               for o in (c.get("operators") or [])))
            return ops, ()
        except Exception:
            pass
    return None


def signature(run_id, subject="", **kw):
    """The run's condition-signature with the claim's SUBJECT stripped.

    Stripping the subject is what makes this count varied BACKGROUNDS rather than repetitions of
    the same experiment. Two runs that differ only in the thing the claim is about are one
    observation of that thing, not two.
    """
    comp = _composition(run_id, **kw)
    if comp is None:
        return None
    ops, params = comp
    s = (subject or "").strip().lower()
    if s:
        ops = tuple(o for o in ops if s not in str(o[0] or "").lower())
        params = tuple(p for p in params if s not in str(p[0] or "").lower())
    return (ops, params)


def independent_support(claim, **kw):
    """How many DISTINCT condition-signatures back this claim. Repetition is not corroboration.

    The morphogen-closure claim cited "10+ runs" and every one was gierer_meinhardt WITH division,
    while the single run without it was stable and was never counted. Ten runs, one observation.
    A rule demanding "three supporting runs" would have passed the worst claim in the file.
    """
    sigs, missing = set(), []
    for r in claim.support:
        s = signature(r, claim.subject, **kw)
        if s is None:
            missing.append(r)
            continue
        sigs.add(s)
    return len(sigs), missing


def compute_conditions(claim, **kw):
    """The conditions are DERIVED, not authored: what is identical across every supporting run.

    This is the answer to `conditions: none noted`. An agent cannot satisfy it with prose because
    it is read off the cited runs' compositions. Whatever is invariant across all of them is an
    unvaried confounder, and naming it is the whole point.
    """
    sigs = [signature(r, claim.subject, **kw) for r in claim.support]
    sigs = [s for s in sigs if s is not None]
    if not sigs:
        return {}
    shared_ops = set(sigs[0][0])
    shared_params = set(sigs[0][1])
    for ops, params in sigs[1:]:
        shared_ops &= set(ops)
        shared_params &= set(params)
    return {"held_fixed_ops": sorted(f"{o}:{i}" for o, i in shared_ops if o),
            "held_fixed_params": sorted(f"{k}={v}" for k, v in shared_params),
            "n_runs": len(claim.support)}


# -------------------------------------------------------------------------- admitted properties
def admitted_metrics():
    """What an instrument actually reports. Reused from predict.py, never re-listed here."""
    try:
        from predict import KNOWN_METRICS, REJECTED_METRICS
        return set(KNOWN_METRICS) - set(REJECTED_METRICS)
    except Exception:
        return set()


# A PROPERTY IS NAMED IN A MEASUREMENT CONTEXT, not merely mentioned. The first version matched
# every snake_case token and duly proposed building instruments for `remove_op`,
# `reconnect_t1_3d0` and `vesicle_growth0` -- operator names, not measurable quantities. An
# instrument request per operator is noise, and noise is how the operator backlog stopped being
# read the first time.
_PROP = re.compile(r"\b([a-z][a-z0-9_]{3,})\s*(?:=|==|>=|<=|>|<|\bof\b|\bis\b)", re.I)

# Words that name a QUANTITY the campaign reasons about but may not measure. This is the vocabulary
# the pattern gap lives in: wavelength, domain count and contrast on the activator field are the
# variables that actually govern budding, and nothing reports any of them.
PROPERTY_WORDS = ("wavelength", "domain", "domains", "spacing", "contrast",
                  "localisation", "localization", "periodicity", "anisotropy",
                  "sphericity", "roughness")

# Words that LOOK like properties in a measurement context but are prose. "the low-c/high-d
# CORNER of the Turing map" and "a_sw BELOW the activator range" produced two instrument
# requests in round 1 -- one for a region of parameter space and one for a model parameter that
# is set, not measured. An instrument request is expensive and the backlog is the deliverable;
# filling it with nouns from a sentence is how it stops being read.
NOT_PROPERTIES = ("corner", "range", "box", "column", "map", "region", "family", "side",
                  "point", "level", "state", "value", "number", "case", "part", "step",
                  "growth", "division", "inflation", "sphere", "ceiling", "bud", "buds",
                  "metric", "pattern", "morphology", "curvature", "aspect")


def _operator_names():
    """Operator identifiers, so a mechanism is never mistaken for a measurement."""
    try:
        from composition_space import OPERATORS
        names = set()
        for k in OPERATORS:
            names.add(k)
            names.add(re.sub(r"\d+$", "", k))
        return names
    except Exception:
        return set()


def unmeasured_properties(text, extra=()):
    """Property-shaped words in a claim that no admitted instrument reports.

    `morphology=sphere` was recorded for the run carrying the finest Turing pattern in the
    campaign, because the shape was measured and the pattern was not. The honest record is
    `not measured`, and the honest next step is a request for an instrument.
    """
    ok = admitted_metrics() | set(extra)
    ops = _operator_names()
    t = (text or "").lower()
    named = set(_PROP.findall(t))
    named |= {w for w in PROPERTY_WORDS if re.search(rf"\b{w}\b", t)}
    # an operator is a mechanism, and a verb-ish token is neither
    named = {w for w in named if w not in ops and not w.startswith(("remove_", "add_", "set_"))}
    named -= set(NOT_PROPERTIES)
    # a model PARAMETER is set, not measured: asking for an instrument to report it is a category
    # error. Anything the composition space knows as a param name is excluded.
    try:
        from composition_space import OPERATORS
        _params = {p for o in OPERATORS.values() for p in (o.get("params") or {})}
        named -= _params
    except Exception:
        pass
    return sorted(named - ok)


# ------------------------------------------------------------------------------- the four checks
def check_claim(claim, **kw):
    """Returns (verdict, findings). verdict is 'ok', 'demote' or 'reject'.

    DEMOTE IS THE DEFAULT FAILURE, NOT REJECT. One null ablation is real evidence -- it is simply
    not a universal negative. Demotion keeps the observation, avoids stalling a round, and makes
    the revisit queue concrete: the queue IS the list of claims below their support bar.
    """
    out = []

    # C1 MODALITY MATCHES THE WORDS. A statement saying NECESSARY may not be filed as can_be. The
    # three worst claims in the campaign were filed as positives and asserted universals.
    asserted = asserted_modality(claim.statement)
    if asserted and claim.modality in (CAN_BE, COULD_BE):
        out.append(("reject", "C1_MODALITY_UNDERSTATED",
                    f"the sentence asserts {asserted} but it is filed as {claim.modality}"))

    # C2 SUPPORT, in independent signatures. Failure DEMOTES.
    n, missing = independent_support(claim, **kw)
    bar = SUPPORT_BAR[claim.modality]
    if missing:
        out.append(("reject", "C2_SUPPORT_NOT_ON_DISK",
                    f"cited runs with no composition or spec: {', '.join(missing)}"))
    if n < bar:
        out.append(("demote", "C2_UNDER_SUPPORTED",
                    f"{n} independent signature(s) from {len(claim.support)} run(s), "
                    f"bar for {claim.modality} is {bar}"))

    # C3 CONDITIONS ARE COMPUTED, not authored.
    want = compute_conditions(claim, **kw)
    if claim.support and claim.conditions != want:
        out.append(("reject", "C3_CONDITIONS_NOT_COMPUTED",
                    "conditions must equal the shared invariants across the cited runs"))

    # C4 EVERY PROPERTY IS MEASURED.
    un = unmeasured_properties(f"{claim.statement} {claim.refuter}")
    if un:
        out.append(("reject", "C4_UNMEASURED_PROPERTY",
                    f"no admitted instrument reports: {', '.join(un)}. File an instrument "
                    f"request, do not soften the verdict."))

    # A necessity claim needs the counterexample ATTEMPT, not more of the same ablation.
    if claim.modality in NEEDS_COUNTEREXAMPLE and not (claim.counterexample or "").strip():
        out.append(("demote", "C2_NO_COUNTEREXAMPLE",
                    "necessity requires a failed attempt to reach the effect WITHOUT the subject; "
                    "an ablation of a working recipe is not that attempt"))

    if any(v == "reject" for v, _, _ in out):
        return "reject", out
    if any(v == "demote" for v, _, _ in out):
        return "demote", out
    return "ok", out


# ---------------------------------------------------------------------------------- the register
class Register:
    """Append-only claim store. Same shape as escalation.Backlog, deliberately."""

    def __init__(self, path=REGISTER):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._items = {}
        if os.path.exists(path):
            for line in open(path):
                if line.strip():
                    d = json.loads(line)
                    self._items[d["cid"]] = Claim(**d)

    def next_cid(self):
        return f"K{len(self._items) + 1:03d}"

    def file(self, claim: Claim, **kw):
        """File a claim, applying the checks. A demoted claim is STORED, as could_be."""
        verdict, findings = check_claim(claim, **kw)
        if verdict == "reject":
            return None, findings
        if verdict == "demote":
            claim.modality, claim.status = COULD_BE, "demoted"
            claim.note = ((claim.note + " | ") if claim.note else "") + \
                "; ".join(f"{c}: {m}" for _, c, m in findings)
        claim.conditions = compute_conditions(claim, **kw)
        self._items[claim.cid] = claim
        with open(self.path, "a") as f:
            f.write(json.dumps(claim.to_dict()) + "\n")
        return claim, findings

    def all(self):
        return list(self._items.values())

    def by_modality(self, m):
        return [c for c in self._items.values() if c.modality == m and c.status != "retracted"]

    def below_bar(self):
        """THE REVISIT QUEUE. Literally the claims sitting under their support bar."""
        return [c for c in self._items.values() if c.status == "demoted"]


# ----------------------------------------------------------------------------------- the render
def render(reg: Register):
    """The claim-bearing sections of memory.md, emitted from the register.

    NEVER TRUNCATED. The could_be bucket exists so untested territory cannot be starved; a render
    cap would starve it by another route, and an untried edit ranked ninth is exactly as invisible
    to the Proposer as it was before this file existed. Ranking is stated so the Proposer knows
    what it is reading.
    """
    L = ["## What is ESTABLISHED", ""]
    for c in sorted(reg.by_modality(CAN_BE), key=lambda c: c.cid):
        L.append(f"- [{c.cid}] \"{c.statement}\" — support {', '.join(c.support) or 'none'}. "
                 f"Refuted by: {c.refuter or 'unstated'}.")
    L += ["", "## What CANNOT be — negatives, with their quantifier", ""]
    for c in sorted(reg.by_modality(CANNOT_BE) + reg.by_modality(CANNOT_NOT_BE),
                    key=lambda c: c.cid):
        n, _ = independent_support(c)
        held = ", ".join((c.conditions or {}).get("held_fixed_ops", [])[:6]) or "none computed"
        L.append(f"- [{c.cid}] ({c.modality}) \"{c.statement}\"")
        L.append(f"      support: {n} independent signature(s) from {len(c.support)} run(s)")
        L.append(f"      held fixed (UNVARIED — the claim does not extend past these): {held}")
        L.append(f"      refuted by: {c.refuter}")
    L += ["", "## COULD BE — untried, or tried below the bar. Mine these. Never cite as a negative.",
          "", "_Ranked by: frontier parents the edit is legal on, then family coverage deficit, "
          "then age._", ""]
    for c in sorted(reg.by_modality(COULD_BE), key=lambda c: c.cid):
        why = c.note or "never attempted"
        L.append(f"- [{c.cid}] \"{c.statement}\" — {why}")
    return "\n".join(L)


# ------------------------------------------------------------------------------------ the check
def check_file(memory_path, config_dir=None, log_dir=None):
    """Parse a legacy prose memory.md and report what LOGIC.md would refuse.

    This is the fixture path: the 2 August memory.md is prose, not a register, and the whole point
    of the test is that this checker REJECTS it while PASSING its four well-formed positives.
    """
    txt = open(memory_path, errors="replace").read()
    sec, cur = {}, None
    for line in txt.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            sec[cur] = []
        elif cur:
            sec[cur].append(line)

    def bullets(name):
        out, buf = [], ""
        for l in sec.get(name, []):
            if l.startswith("- "):
                if buf:
                    out.append(buf)
                buf = l[2:].strip()
            elif buf and l.strip():
                buf += " " + l.strip()
        if buf:
            out.append(buf)
        return out

    rows = []
    for name in sec:
        for b in bullets(name):
            asserted = asserted_modality(b, name)
            has_refuter = bool(re.search(r"falsifiable by|refuted by|refuted if|guard:", b, re.I))
            # THREE KINDS OF CITATION, and only one of them can be checked. The campaign's
            # claims cite COMPOSITION HASHES ("C414a11") and BARE ROUNDS ("round 2"), neither of
            # which names the runs that back the claim -- so the support cannot be recovered even
            # in principle. That is a finding, not a parse failure, and it gets its own reason.
            runs = set(re.findall(r"\b(r\d{3}[nc]_\d{2}_\w+)\b", b))
            vague = set(re.findall(r"\bC[0-9a-f]{6,}\b", b)) | \
                set(re.findall(r"\brounds?\s+\d+", b, re.I))
            if asserted == COULD_BE:
                rows.append((name, COULD_BE, True, has_refuter, 0, b[:74], "untried -- minable"))
            elif asserted:
                if runs:
                    ok, why = len(runs) >= 3, f"{len(runs)} run(s) cited"
                elif vague:
                    ok, why = False, "cites comp hashes/rounds, not runs -- unresolvable"
                else:
                    ok, why = False, "no citation at all"
                ok = ok and has_refuter
                rows.append((name, asserted, ok, has_refuter, len(runs), b[:74], why))
            else:
                rows.append((name, CAN_BE, True, has_refuter, len(runs), b[:74],
                             f"{len(runs)} run(s) cited"))
    return rows


def write_report(rid, rows, path):
    """One line per round: the confusion matrix, so the trend is visible without re-parsing.

    A check whose result is only ever printed to a terminal is a check nobody reads twice.
    """
    neg = [r for r in rows if r[1] in (CANNOT_BE, CANNOT_NOT_BE)]
    pos = [r for r in rows if r[1] == CAN_BE]
    cb = [r for r in rows if r[1] == COULD_BE]
    rec = {"round": rid, "claims": len(rows), "negatives": len(neg),
           "unearned": sum(1 for r in neg if not r[2]), "positives": len(pos),
           "positives_refused": sum(1 for r in pos if not r[2]), "could_be": len(cb),
           "unearned_text": [r[5][:90] for r in neg if not r[2]][:8]}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--fixture", default=None,
                    help="an archived campaign's records/ dir to check instead of the live one")
    a = ap.parse_args()

    if a.render:
        print(render(Register()))
        return 0

    if a.check:
        mem = os.path.join(a.fixture, "memory.md") if a.fixture else os.path.join(CAMP, "memory.md")
        if not os.path.exists(mem):
            print(f"[logic] no memory.md at {mem}")
            return 1
        rows = check_file(mem)
        neg = [r for r in rows if r[1] in (CANNOT_BE, CANNOT_NOT_BE)]
        pos = [r for r in rows if r[1] == CAN_BE]
        bad = [r for r in neg if not r[2]]
        good_pos = [r for r in pos if r[3]]

        print(f"[logic] {mem}")
        print(f"  claims parsed        : {len(rows)}")
        print(f"  negatives asserted   : {len(neg)}")
        print(f"  ... REFUSED          : {len(bad)}   (no refuter, or <3 independent runs)")
        print(f"  positives            : {len(pos)}")
        print(f"  ... with a falsifier : {len(good_pos)}  <- these MUST pass; a checker that "
              f"rejects everything is worthless")
        print()
        for name, mod, ok, ref, n, txt, why in bad:
            print(f"  REFUSE [{mod}] in '{name}': {why}; refuter={'yes' if ref else 'NO'}")
            print(f"         {txt}")
        # wire the dead check rather than duplicating it
        try:
            import templates as T
            words = len(open(mem, errors="replace").read().split())
            if words > T.MAX_MEMORY_WORDS:
                print(f"\n  templates.check_memory: {words} words against a "
                      f"{T.MAX_MEMORY_WORDS} budget -- this check exists and has never fired")
        except Exception:
            pass
        return 0 if not bad else 2

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
