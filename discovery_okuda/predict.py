"""predict -- parse a natural-language prediction into checkable clauses, and score it.

WHY THIS FILE EXISTS (three defects in one three-line regex)
------------------------------------------------------------------------------------------------
The surprise rate is the campaign's ONLY control signal: it sets the confirmatory/adversarial
mix, it decides when a cluster is dry, and it is the headline number of every round. It was
computed by this:

    m = re.search(r"(>=|<=|>|<)\\s*([0-9.]+)", str(pred))
    if not m:
        return True                     # "only a genuinely UNSTATED prediction is uncounted"
    ...
    outcome = "confirmed" if _pred_holds(...) else "refuted"

Three things are wrong with it, and all three push the surprise rate DOWN -- i.e. toward
"nothing was learned", which is exactly the failure F19 was raised about.

  P1  AN UNPARSEABLE PREDICTION IS SCORED `confirmed`, NOT SKIPPED.  The comment says such a
      prediction is "uncounted"; the code returns True, and True means CONFIRMED at the call
      site. Measured on the first real LLM proposal: slot 0's `protr_peak 2.0-3.5` (a RANGE --
      a perfectly reasonable way to state a control's expectation) parses to nothing and is
      therefore recorded as a confirmed prediction, forever, without anyone looking.

  P2  THE PARSER IS METRIC-BLIND.  It takes the first comparison ANYWHERE in the string and
      applies it to `protr_peak`, whatever metric the sentence was actually about. The very
      first proposal contains `"mech_p_ratio drops toward ~1 with protr_peak >= 2.0"`; had the
      agent written the p_ratio clause with an operator, the p_ratio threshold would have been
      silently tested against protr_peak. This is the house error -- my metric vs their metric --
      sitting in the scoring path.

  P3  `REFUTED if ...` IS PARSED AS THE PREDICTION.  The prompt invites the agent to state the
      falsifier, and it does: `"protr_peak >= 2.0 (within ~0.5 of control); REFUTED if it drops
      below 1.5"`. A first-match regex is saved here only by clause order. Reverse the sentence
      and the falsifier becomes the prediction, inverting the outcome.

The fix is not a better regex. It is to (a) parse EVERY clause with its metric, (b) refuse to
guess when nothing checkable is present -- `inconclusive`, which is an existing, honest outcome
that is excluded from the surprise denominator -- and (c) make the failure legible in the record.

A prediction that cannot be checked is not a prediction. It must not be scored as a correct one.
"""
from __future__ import annotations

import re

# Metrics the instrument gate admitted, plus the diagnostics that are informative but unscored.
# A clause naming anything else is not checkable and is reported as such rather than guessed at.
KNOWN_METRICS = (
    "protr_peak", "protr_final", "ta_n_tubes_final",       # ADMITTED (instrument gate)
    "mech_p_ratio", "n_cells_final", "Q_drop",             # informative, not scored
    "ta_aspect_len_over_diam", "ta_tube_len_final", "retention",   # REJECTED -- see below
    # Turing x vertex study (turing_vertex_study.py). Added after the parser correctly REFUSED to
    # score 22 runs whose predictions were written against `protr` -- it returned `inconclusive`
    # 22 times rather than inventing 22 confirmations, which is the whole point of P1. An unknown
    # metric is a missing entry here, not a licence to guess.
    "protr", "protr_p99", "corr_act_rad", "hollow_frac", "vol_cv", "act_max", "r_cv", "cells_end",
    # THE PATTERN. Certified before this campaign began (n_spots exact at 3/5/12, spacing within
    # 13% of the analytic value) and never admitted, so no claim could name the variable that
    # actually governs budding: a fine field spreads growth evenly and keeps a sphere spherical,
    # one localised patch makes a bud. Without these a conclusion about the pattern is not a
    # weaker verdict, it is an unmeasured one.
    "n_spots_final", "spot_cells_med_final", "spot_cells_max_final", "spot_frac_final",
    "spot_spacing_cells_final",
    # `wavelength_cells_final` WAS ADMITTED HERE AND IS NOT PRODUCED. pattern_scale computes it
    # and stores it as `autocorr_hops_uncalibrated` -- renamed deliberately, because finding F010
    # withdrew it as uncalibrated -- so the admitted name has never once appeared in a summary,
    # and `agents/llm_agents.py` was advertising it to every agent as a NEW INSTRUMENT. A metric
    # that was decommissioned for lying must not be reachable through a name that survived it.
    # It is listed as REJECTED below under its real name, which is the honest record.
    # THE RESERVOIR. `divide_3d` counts the divisions it refuses for want of vertex buffer and
    # flags a full array; run_one records both in diag.json. They were never registered here, so
    # on 3 August the Metrologist asked for an instrument to measure `div_blocked` -- a quantity
    # ALREADY MEASURED and written to disk for every run. That is the campaign's recurring shape
    # in its purest form: built, recorded, and never handed over.
    #
    # They matter because they decide what a number MEANS. A run that stops at 1766 of 1778 cells
    # has not found where growth stops; it has found where the array ends, and a claim about
    # growth resting on it is a claim about a buffer. Registering them makes "the tissue saturated
    # its reservoir" a checkable prediction rather than a remark.
    "div_blocked", "buf_full", "div_blocked_first_frame",
    # CELL SHAPE. Measured every frame, and P7 FAILS a specimen on it -- shape index 22.65 against
    # 3.72 for a regular hexagon is what invalidated the campaign's three highest protr_peak runs.
    # It was never admitted, so the loop could be TOLD its cells were degenerate and could not
    # PROPOSE to fix it: no prediction may name a metric that is not here, so "this edit keeps the
    # cells regular" was unsayable and untestable.
    #
    # It is also the quantity that separates a grown tube from a stretched one. round40_mc8 holds
    # shape_idx_med at 3.83 -- near-regular -- for 900 frames while protr doubles, which is what a
    # healthy tube looks like; a sheet being pulled apart shows it here first.
    "shape_idx_med", "shape_idx_p95", "shape_idx_max", "shape_idx_mean",
    # THE PATTERN, not just its peak. Only act_max and corr_act_rad were admitted, so the loop
    # could see a spike and could not see a field die -- and okuda_route's activator reached
    # 17,678 at frame 350 and 0.0105 by frame 807 while every admitted number stayed sayable.
    # act_cv is the one that matters: a uniform field and a real Turing pattern have the SAME
    # MEAN and cv of 0.000 against 0.905.
    "act_mean", "act_sd", "act_cv", "act_occupancy", "red_frac",
    "act_sd_final", "act_cv_final", "act_occupancy_final", "act_mean_final",
    # ... and over the whole run, so "the pattern went extinct" is a measurement:
    "act_alive_frac", "act_extinct_frame", "act_peak_frame",
    # DOES THE CHEMISTRY GRIP THE SHAPE? The campaign's actual question. `corr_act_rad` was
    # admitted with NO PRODUCER anywhere in the codebase -- every prediction that named it scored
    # `not measured` and fell to inconclusive -- and `act_at_tip` (activator in the outermost
    # tenth of cells, over the tissue mean) asks it without assuming the relation is linear.
    "corr_act_rad_final", "act_at_tip", "act_at_tip_final",
    # SHAPE, NOT SIZE. protr is p95/median: a tail statistic, blind to whether the tissue is ONE
    # long tube or a lumpy ball with the same tail. The gyration tensor separates them, which is
    # Okuda's own phenotype axis (tube / undulation / branch), and r_cv and protr_p99 were both
    # admitted with no producer too.
    "gyr_prolate", "gyr_asphere", "gyr_oblate",
    "gyr_prolate_final", "gyr_asphere_final", "gyr_oblate_final", "r_cv_final", "protr_p99_final",
    # EXCESS AREA, the mechanical precondition for buckling: rv = 1 is a sphere, rv < 1 means the
    # shell holds more area than a sphere of that volume can and MUST wrinkle. Computed every
    # frame since the reduced-volume work and never admitted.
    "reduced_volume", "reduced_volume_final",
    # THE SELF-INTERSECTION FRACTION P11 KEYS ON, and which the evidence horizon now truncates at.
    # It decided which frames count as evidence while being unnameable in a prediction.
    "ray_single_frac", "ray_single_frac_final",
)

# Measured to lie by the instrument gate (F15/F16). A prediction resting on one of these is not
# evidence, and saying so is the whole point of having run the gate.
REJECTED_METRICS = ("ta_aspect_len_over_diam", "ta_tube_len_final", "retention",
                    "autocorr_hops_uncalibrated")

# WHAT THE AGENTS ARE TOLD IS ADMISSIBLE. Rendered from the registry, never hand-written into a
# prompt. The Reader's prompt carried a HARDCODED list -- "ADMITTED protr_peak, ta_n_tubes_final,
# protr_final" -- which was already stale against KNOWN_METRICS and, worse, meant a newly
# certified instrument could never be mentioned to the role that reads the numbers. An instrument
# nobody is told about is the same defect as an instrument that does not exist: pattern_scale was
# written, certified and computed every frame, and no agent could name it.
# WHAT EACH ONE MEANS, AND WHAT IT READS ON A SPHERE. Six of these existed for fifty-six
# admitted names, so a role was handed a comma-separated list of identifiers and asked to write a
# falsifiable prediction over them. A metric with no stated reference value cannot be predicted
# against: "act_cv > 0.3" is a guess unless you know a dead field reads 0.00 and a live Turing
# field reads about 1. Every note below states the number a NULL result gives, because that is
# what makes the prediction a bet rather than a description.
METRIC_NOTES = {
    # ---- shape of the tissue ------------------------------------------------------------
    "protr_peak": "the best protrusion ratio over the run. 1.0 = a sphere",
    "protr": "percentile(r,95)/median(r) about the tissue centroid. 1.0 = a sphere. A TAIL "
             "statistic: blind to a single thin spike, and equal for one tube and a lumpy ball",
    "protr_p99": "percentile(r,99)/median(r). Catches the thin spike protr's p95 misses",
    "r_cv": "spread of cell radius over its mean. 0 = a perfect sphere; rises with ANY departure "
            "from one, unlike protr which only sees the tail",
    "gyr_prolate": "largest gyration eigenvalue over the mean of the other two. 1.0 = a sphere, "
                   "> 1 = ELONGATED along one axis. This is what separates a TUBE from an "
                   "undulating ball: both raise protr, only a tube raises this",
    "gyr_asphere": "standard asphericity. 0 = a sphere, 1 = a rod",
    "gyr_oblate": "0 for a sphere OR a rod, positive for a FLATTENED shell -- a vesicle "
                  "collapsing into a disc, which reads as `not a tube` with no number of its own",
    "reduced_volume": "6 sqrt(pi) V / A^1.5. 1.0 = a sphere; BELOW 1 the shell holds more area "
                      "than a sphere of that volume can and MUST wrinkle, buckle or fold. The "
                      "mechanical precondition for budding, not a consequence of it",
    "ta_n_tubes_final": "tube count at the end",
    # ---- the cells ----------------------------------------------------------------------
    "shape_idx_med": "median perimeter/sqrt(area). 3.545 is a circle and is the FLOOR for any "
                     "shape; 3.81 is the rigidity transition -- above it the tissue FLOWS and "
                     "cannot hold a shape it is pushed into",
    "shape_idx_p95": "the same for the worst-shaped 5% of cells",
    # ---- the pattern --------------------------------------------------------------------
    "act_max": "peak activator over the cells. Says NOTHING about whether a pattern exists: it "
               "reads high for one exploding cell and for a healthy field alike",
    "act_mean": "mean activator. CANNOT SEE A PATTERN -- 0.5 everywhere and half-at-1/half-at-0 "
                "give the same number",
    "act_sd": "spatial spread of the activator across cells. This IS the pattern's amplitude; "
              "0 means the field is uniform or dead, whatever its mean",
    "act_cv": "act_sd / act_mean -- scale-free, so it survives a collapsing or exploding level. "
              "0.00 = uniform or dead, ~0.9-1.8 = a real Turing pattern. THE metric for `is "
              "there a pattern at all`",
    "act_occupancy": "fraction of cells in the upper half of the field's own range. ~0.5 for a "
                     "coarse two-phase pattern, small for isolated spots, 0 for a dead field",
    "red_frac": "fraction of cells above the GROWTH OPERATOR'S OWN switch a_sw -- i.e. the cells "
                "growth actually acts on. LOW = localised spots (distinct tubes), HIGH = the "
                "activator has spread over the shell (one fat lumpy lobe)",
    "act_alive_frac": "fraction of frames in which a pattern existed at all (act_cv > 0.05 AND "
                      "occupancy > 0.01). 1.0 = patterned throughout; 0.2 = a FLASH followed by "
                      "a dead run -- the run grew on a corpse for its last 80%",
    "act_extinct_frame": "the frame at which the pattern last stopped existing, or absent if it "
                         "never did. Turns `the chemistry died` into a measurement",
    "act_peak_frame": "the frame of maximum act_max. Early + extinct later = a blow-up, not a "
                      "pattern",
    "n_spots_final": "how many distinct activator domains (a fine field has many, a bud has one)",
    "spot_spacing_cells_final": "centre-to-centre domain spacing, in cells",
    "spot_frac_final": "fraction of cells above the activator threshold",
    # ---- does the pattern GRIP the shape? -----------------------------------------------
    "corr_act_rad": "Pearson correlation between a cell's activator and its radius. THE "
                    "campaign's question. REFUSED (not measured) when act_cv < 0.05: a "
                    "correlation on a dead field is a correlation of round-off, and it reads a "
                    "confident 0.29 on an activator whose entire spread is 8e-05",
    "act_at_tip": "mean activator in the outermost tenth of cells, over the tissue mean. 1.0 = "
                  "no relation to shape, > 1 = the activator sits at the protrusions. Asks the "
                  "same question as corr_act_rad without assuming the relation is a straight line",
    # ---- is this even evidence? ---------------------------------------------------------
    "ray_single_frac": "fraction of rays from the centroid that cross the surface EXACTLY once. "
                       "1.0 = a simple closed shell. Below 0.5 the sheet has folded through "
                       "itself, which a tissue cannot do -- every frame after that measures a "
                       "broken mesh, and the evidence horizon truncates there",
    "div_blocked": "divisions REFUSED for want of vertex buffer. Non-zero means growth stopped "
                   "where the ARRAY ended, not where biology did",
    "buf_full": "the vertex array filled. A run that ends here has not found a limit of growth",
    "mech_p_ratio": "pressure in protruding cells over body cells. ~3 = a FORCED protrusion, "
                    "~1 = a growth-driven equilibrium",
    "Q_drop": "protrusion lost when the forces are switched off and the tissue is allowed to "
              "relax. LARGE = the shape was being HELD by force; ~0 = an equilibrium shape",
    "div_blocked_first_frame": "the frame at which the buffer first refused a division. "
                               "Everything after it is a run against a wall",
    "hollow_frac": "the frozen legacy blend of folded|sliver|under-connected cells. Prefer "
                   "broken_frac: this cannot tell a slightly bent cell from a destroyed one",
    "vol_cv": "spread of cell volume over its mean. Rises when division stalls while cells keep "
              "growing -- a size distribution coming apart",
    "shape_idx_mean": "mean perimeter/sqrt(area); see shape_idx_med for the reference values",
    "shape_idx_max": "the single worst-shaped cell. Above ~5 that cell is a 2:1 sliver, and the "
                     "mesh rather than the tissue is what is being measured",
    "n_cells_final": "final cell count in the okuda loop",
    "protr_final": "protrusion ratio at the last VALID frame (truncated at the evidence horizon)",
    "cells_end": "final cell count. TURING x VERTEX PIPELINE ONLY -- the okuda loop produces "
                 "`n_cells_final`, and a prediction naming this one there scores `not measured`",
}


def admitted_block(new_since=()):
    """The admissible-metric block for a prompt, built from the registry.

    `new_since` names metrics admitted recently, so their arrival is ANNOUNCED rather than left
    to be noticed. A role that is not told an instrument exists will keep reasoning as though the
    property is unmeasurable -- which is how the finest Turing pattern in the campaign came to be
    recorded as a null sphere.
    """
    ok = [m for m in KNOWN_METRICS if m not in REJECTED_METRICS]
    lines = ["Only these metrics are admissible (the others were MEASURED to lie and are excluded):"]
    lines.append("  ADMITTED  " + ", ".join(ok))
    lines.append("  REJECTED  " + ", ".join(REJECTED_METRICS))
    notes = [f"    {m} -- {METRIC_NOTES[m]}" for m in ok if m in METRIC_NOTES]
    if notes:
        lines.append("  what the less obvious ones mean:")
        lines += notes
    if new_since:
        lines.append("")
        lines.append("  NEW INSTRUMENTS, admitted since the last campaign -- USE THEM. A property "
                     "you could not measure before is not still unmeasurable:")
        for m in new_since:
            lines.append(f"    {m}" + (f" -- {METRIC_NOTES[m]}" if m in METRIC_NOTES else ""))
    return "\n".join(lines)


_OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, "<": lambda a, b: a < b}

# "REFUTED if ...", "falsified if ...", "refuted when ..." -- everything from there on states the
# FALSIFIER, not the prediction. Parsing it as the prediction inverts the outcome.
_FALSIFIER = re.compile(r"\b(?:refuted|falsifie[sd]|disconfirmed|wrong)\b\s*(?:if|when|by)?",
                        re.I)

_METRIC_ALT = "|".join(sorted(KNOWN_METRICS, key=len, reverse=True))

# metric <op> value            e.g. "protr_peak >= 2.0"
_CLAUSE_OP = re.compile(rf"(?P<metric>{_METRIC_ALT})\s*(?P<op>>=|<=|>|<)\s*(?P<val>-?[0-9.]+)", re.I)
# metric <lo>-<hi> / "between" e.g. "protr_peak 2.0-3.5", "protr_peak between 2 and 3.5"
_CLAUSE_RANGE = re.compile(
    rf"(?P<metric>{_METRIC_ALT})\s*(?:of|:|is|=|~)?\s*"
    rf"(?:between\s+)?(?P<lo>[0-9.]+)\s*(?:-|–|to|and)\s*(?P<hi>[0-9.]+)", re.I)


class Clause:
    """One checkable statement: a metric, a test, and the text it came from."""

    def __init__(self, metric, kind, lo=None, hi=None, op=None, val=None, src=""):
        self.metric, self.kind, self.src = metric, kind, src
        self.lo, self.hi, self.op, self.val = lo, hi, op, val

    def check(self, observed: dict):
        """(holds, why). None if the metric was not measured -- NOT a pass."""
        if self.metric not in observed or observed[self.metric] is None:
            return None, f"{self.metric} not measured"
        try:
            got = float(observed[self.metric])
        except (TypeError, ValueError):
            return None, f"{self.metric}={observed[self.metric]!r} is not numeric"
        if self.kind == "range":
            ok = self.lo <= got <= self.hi
            return ok, f"{self.metric}={got:g} {'in' if ok else 'OUTSIDE'} [{self.lo:g},{self.hi:g}]"
        ok = _OPS[self.op](got, self.val)
        return ok, f"{self.metric}={got:g} {'satisfies' if ok else 'VIOLATES'} {self.op}{self.val:g}"

    def __repr__(self):
        return (f"<{self.metric} in [{self.lo},{self.hi}]>" if self.kind == "range"
                else f"<{self.metric}{self.op}{self.val}>")


def parse(pred: str):
    """Prediction text -> [Clause]. Text after a falsifier marker is discarded, not parsed."""
    if not pred:
        return []
    text = _FALSIFIER.split(str(pred))[0]        # keep only the assertion, drop the falsifier
    out, spans = [], []
    for m in _CLAUSE_RANGE.finditer(text):
        lo, hi = float(m.group("lo")), float(m.group("hi"))
        if lo > hi:
            lo, hi = hi, lo
        out.append(Clause(m.group("metric").lower(), "range", lo=lo, hi=hi, src=m.group(0)))
        spans.append(m.span())
    for m in _CLAUSE_OP.finditer(text):
        # a range already consumed this text (e.g. "2.0-3.5" contains no op, but be safe)
        if any(s <= m.start() < e for s, e in spans):
            continue
        out.append(Clause(m.group("metric").lower(), "op", op=m.group("op"),
                          val=float(m.group("val")), src=m.group(0)))
    return out


def score(pred: str, observed: dict, primary_metric: str = None):
    """(outcome, note) with outcome in confirmed | refuted | inconclusive.

    Rules, in order:
      * nothing checkable            -> inconclusive  (NOT confirmed -- defect P1)
      * every clause on a REJECTED   -> inconclusive  (the metric bank lies there)
      * the clause on `primary_metric` decides, if present; otherwise ALL clauses must hold
        (a conjunction -- the agent wrote them all, so it is on the hook for all of them)
      * a clause whose metric was not measured cannot pass; if none can be evaluated ->
        inconclusive
    """
    clauses = parse(pred)
    if not clauses:
        return "inconclusive", (f"NOT CHECKABLE: no clause of the form '<metric> <op> <value>' "
                                f"or '<metric> <lo>-<hi>' naming a known metric in "
                                f"{pred[:120]!r}. Recorded as inconclusive rather than guessed.")
    if all(c.metric in REJECTED_METRICS for c in clauses):
        return "inconclusive", (f"prediction rests only on instrument-gate-REJECTED metrics "
                                f"({', '.join(sorted({c.metric for c in clauses}))}); not evidence")

    ranked = [c for c in clauses if primary_metric and c.metric == primary_metric] or clauses
    results, whys = [], []
    for c in ranked:
        ok, why = c.check(observed)
        whys.append(why)
        if ok is not None:
            results.append(ok)
    if not results:
        return "inconclusive", "; ".join(whys)
    return ("confirmed" if all(results) else "refuted"), "; ".join(whys)


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    OK = "\033[92mok\033[0m"
    fails = []

    def eq(got, want, what):
        if got != want:
            fails.append(f"{what}: got {got!r} want {want!r}")
        print(f"  [{OK if got == want else 'FAIL'}] {what:62} -> {got}")

    print("P1 -- an unparseable prediction must NOT be scored `confirmed`")
    # the real slot-0 control prediction from the first LLM proposal
    eq(score("protr_peak 2.0-3.5, mech_p_ratio ~3 (forced), analyst phenotype tube/spike",
             {"protr_peak": 2.8})[0], "confirmed", "range form now parses (in range)")
    eq(score("protr_peak 2.0-3.5, mech_p_ratio ~3", {"protr_peak": 9.1})[0], "refuted",
       "range form, observed outside")
    eq(score("it should look like a nice tube", {"protr_peak": 2.8})[0], "inconclusive",
       "genuinely unstated -> inconclusive, NOT confirmed")
    eq(score("Q > 0.5", {"protr_peak": 2.8})[0], "inconclusive",
       "unknown metric `Q` -> inconclusive, NOT confirmed")

    print("\nP2 -- a threshold must be tested against the metric it names")
    # the demonstrating case: p_ratio's threshold must NOT be applied to protr_peak
    pred = "mech_p_ratio <= 1.5 with protr_peak >= 2.0"
    eq(score(pred, {"protr_peak": 2.4, "mech_p_ratio": 1.2})[0], "confirmed", "both clauses hold")
    eq(score(pred, {"protr_peak": 1.2, "mech_p_ratio": 1.2})[0], "refuted", "protr_peak fails")
    eq(score(pred, {"protr_peak": 2.4, "mech_p_ratio": 3.9})[0], "refuted", "p_ratio fails")
    # under the old regex this returned protr_peak(2.4) <= 1.5 -> False -> "refuted", wrongly
    eq(score(pred, {"protr_peak": 2.4, "mech_p_ratio": 1.2}, primary_metric="protr_peak")[0],
       "confirmed", "primary metric selects the right clause")

    print("\nP3 -- `REFUTED if ...` states the falsifier, not the prediction")
    p = "protr_peak >= 2.0 (within ~0.5 of control); REFUTED if it drops below 1.5"
    eq([c.metric for c in parse(p)], ["protr_peak"], "falsifier clause is discarded")
    eq(score(p, {"protr_peak": 2.4})[0], "confirmed", "assertion holds")
    eq(score(p, {"protr_peak": 1.2})[0], "refuted", "assertion fails")
    # reversed order -- the case a first-match regex gets backwards
    r = "REFUTED if protr_peak >= 2.0; I predict protr_peak < 1.5"
    eq(score(r, {"protr_peak": 1.2})[0], "inconclusive",
       "falsifier-first: nothing assertable remains -> inconclusive, not inverted")

    print("\nrejected metrics and missing measurements")
    eq(score("ta_tube_len_final >= 4.0", {"ta_tube_len_final": 9.3})[0], "inconclusive",
       "prediction resting only on a lying metric is not evidence")
    eq(score("protr_peak >= 2.0", {})[0], "inconclusive", "metric not measured -> inconclusive")
    eq(score("protr_peak >= 2.0", {"protr_peak": None})[0], "inconclusive", "null metric")

    print("\nthe five real predictions from the first LLM proposal (round 1)")
    real = [
        ("protr_peak 2.0-3.5, mech_p_ratio ~3 (forced), analyst phenotype tube/spike", "control"),
        ("protr_peak < 1.5 AND mech_p_ratio drops from ~3 toward ~1", "confirmatory"),
        ("protr_peak >= 2.0 (within ~0.5 of control); REFUTED if it drops below 1.5", "adversarial"),
        ("protr_peak < 1.5 or degenerate/collapsed morphology; REFUTED if protr_peak stays >= 2.0",
         "adversarial"),
        ("protr_peak >= 3.0 (>= control) and/or ta_n_tubes_final increases", "confirmatory"),
        ("mech_p_ratio drops toward ~1 with protr_peak >= 2.0; REFUTED if the body merely "
         "inflates (protr_peak < 1.5, no tube)", "confirmatory"),
    ]
    for i, (p, intent) in enumerate(real):
        cs = parse(p)
        assert cs, f"slot {i} still unparseable: {p!r}"
        print(f"  slot {i} [{intent:12}] {cs}")

    print("\n" + ("predict OK" if not fails else f"{len(fails)} FAILURES:\n  " + "\n  ".join(fails)))
    raise SystemExit(1 if fails else 0)
