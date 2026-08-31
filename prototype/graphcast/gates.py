"""The gate table: thresholds fixed BEFORE any run, per plexus2.tex sec. 14.6.

A gate is a number with a threshold decided in advance and a stated consequence if it fails. Three
tiers, and conflating them is the failure mode this form exists to prevent:

    bookkeeping   does the code do what the operator says?          -- verification, cheap
    closed_form   does it reproduce the physics it was GIVEN?       -- verification
    measurement   does it agree with something observed in cells?   -- validation, the only tier
                                                                       that can be wrong about the world

Two rules from the reference, both load-bearing:

  * "A threshold chosen after seeing the number is not a threshold." The thresholds below are
    literals in THIS FILE, not config values -- a threshold you can edit in a yaml between runs is
    not a threshold. This is the one place in the prototype where numbers are hardcoded, and that
    is the point of it.
  * "A gate's threshold belongs in the unit of the phenomenon, not of the mesh." Every row carries
    a `unit` string saying what the number is OF, and the measurement tier is only available
    because the spec declares `general.units:`.

Provenance of the non-obvious thresholds, so a later reader can check them rather than trust them:
  G11/G12  0.70 adjusted Rand index -- the flyvis connectome Ward-tree reference is 0.702 against
           the 65 true cell types (measured, connectome-gnn analysis 2026-08-30).
  G14      0.03 -- twice the 0.015 run-to-run resolution floor measured on flyvis_A in the
           weekend benchmark (papers/weekend_benchmark_results_2026_08_29.md sec. 4).
  G17      0.268 -- held-out R^2 of a parameter-free 8/64/512-nearest-neighbour spatial pool on
           ZAPBench dF/dt, which is ABOVE a rank-64 linear readout of the whole population (0.262).
  G19      GCaMP6 rise 50-200 ms, decay 0.5-2 s.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

TIERS = ("bookkeeping", "closed_form", "measurement")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
PENDING, DONE = "PENDING", "DONE"   # review status, orthogonal to outcome


@dataclass
class Gate:
    gid: str
    tier: str
    what: str                     # what is measured
    threshold: str                # human-readable, in the unit of the phenomenon
    unit: str                     # what the number is OF
    stage: int                    # the stage that must first pass it
    compare: Callable[[float], bool] = None   # measured -> passed
    measured: Optional[float] = None
    outcome: str = SKIP
    note: str = ""
    artifacts: list = field(default_factory=list)   # PNG / MP4 paths, relative to the run dir
    # STATUS IS NOT OUTCOME. `outcome` is what the number did against the threshold; `status` is
    # whether the gate has been walked through -- definition written, estimator sanity-checked,
    # negative control run, result read by a human. A gate can PASS mechanically and still be
    # PENDING review, and that distinction is the difference between a number and evidence.
    status: str = PENDING
    explain: str = ""                               # plain English, coarse to specific

    def record(self, measured: Optional[float], note: str = "",
               artifacts: Optional[list] = None) -> "Gate":
        """Record a measurement. A gate with NO artifact cannot pass.

        A number in a table cannot be checked by eye, and the failures this prototype is most
        exposed to -- a circuit sitting at a fixed point, a field that is identically zero, an
        embedding that clusters on position rather than on type -- are obvious in a picture and
        invisible in a scalar. Every one of those three was in fact found by looking at numbers
        that a picture would have shown at a glance. So an artifact is a condition of PASS, not a
        decoration on it: a green row with nothing to look at is the endorsement the reference
        warns against.
        """
        self.measured = measured
        self.note = note
        self.artifacts = list(artifacts or [])
        if measured is None or self.compare is None:
            self.outcome = SKIP
        elif not self.artifacts:
            self.outcome = SKIP
            self.note = (note + "; " if note else "") + "no artifact: a gate must point to a figure"
        else:
            self.outcome = PASS if self.compare(measured) else FAIL
        return self


# --- G2's definition ------------------------------------------------------------------------ #
# Dataset identity must never appear in code; paths and sizes are the two ways it leaks in. These
# patterns live HERE, with the thresholds, for two reasons: they are part of the gate's definition
# rather than of the runner, and a scanner whose pattern list sits in a scanned file finds itself.
# NO WORD BOUNDARIES, and that is the fix rather than an oversight. The first version wrote
# `\bzapbench\b`, which does not match inside "zapbench_dff_full.npy" because `_` is a word
# character -- and a dataset name embedded in a filename is the single most likely way one would
# actually appear. A planted-violation check caught 2 of 3 because of it. Substring matching can
# in principle fire on an innocent superstring (7870 inside 178700); for a gate whose threshold is
# zero that is the right direction to be wrong in, because a false positive fails loudly while a
# false negative passes silently.
FORBIDDEN_PATTERNS = [
    (r"/gr" + r"oups/", "an absolute data path"),
    (r"zap" + r"bench", "a dataset name"),
    (r"re" + r"dox", "a dataset name"),
    (r"(717" + r"21|78" + r"70|137" + r"41)", "a dataset dimension"),
]
# gates.py itself is exempt: it holds the pre-registered thresholds, some of which are DERIVED from
# a dataset (G17's 0.268 is a ZAPBench baseline). Those are thresholds, not dataset parameters, and
# the exemption is stated rather than silent.
G2_EXEMPT_FILES = {"gates.py"}
G2_EXEMPT_DIRS = {"__pycache__", "config", ".git", "log"}


def _lt(x):  return lambda v: v < x
def _gt(x):  return lambda v: v > x
def _le(x):  return lambda v: v <= x
def _eq(x):  return lambda v: v == x
def _abs_lt(x): return lambda v: abs(v) < x
def _within(lo, hi): return lambda v: lo <= v <= hi


def build_table() -> dict[str, Gate]:
    """The full table. Every threshold here predates the implementation it gates."""
    g = [
        # ---- tier 1: bookkeeping -------------------------------------------------------- #
        # G1 is split because the two halves become available at different stages, and a green
        # row that covers only half of what it claims is exactly the endorsement the reference
        # warns against. Both thresholds are set here, before either is run.
        Gate("G1", "bookkeeping", "every option combination can be READ",
             "24 of 24 option combinations load", "configs", 0, _eq(24.0), status=DONE,
             explain="The model has four switches, and the whole premise is that any setting of "
                     "them is a legal model -- options, not forks. This gate checks that "
                     "literally: every combination must be readable. It takes the reference "
                     "config, overwrites only the model block with each of the 24 settings, and "
                     "calls the loader. Why 24: `simple` carries no edge state, so running it "
                     "twice is a different model wearing the same name and the schema refuses it, "
                     "which leaves four legal (message, n_passes) pairs times 2 encoder/decoder "
                     "times 3 embeddings. It catches an option that exists only in the "
                     "documentation, and a combination that silently falls back to a default "
                     "instead of erroring."),
        Gate("G1b", "bookkeeping", "every option combination can be RUN",
             "24 of 24 option combinations run one forward step", "configs", 2, _eq(24.0),
             explain="The other half of G1. Reading a config proves the vocabulary is right; it "
                     "does not prove the model can be built from it. This gate constructs the "
                     "model object for each of the 24 settings and pushes one forward pass "
                     "through it. It is split from G1 because the two become available at "
                     "different stages -- G1 needs only the schema, G1b needs a model -- and a "
                     "row that goes green having checked half its claim reads as an endorsement "
                     "it has not earned. It catches shape mismatches that appear only when two "
                     "particular options meet."),
        # G2 IS SPLIT FOR THE SAME REASON G1 IS. The scanner half can run today; the half that
        # actually matters -- that ONE pipeline runs on three datasets with only the config
        # changing -- cannot, because the ZAPBench and redox loaders do not exist yet. The scanner
        # passing now is weak evidence: it passes partly because the most likely place for a
        # hardcoded path has not been written. Marking that green would be a row claiming more
        # than it has earned, which is the failure this whole form exists to prevent.
        Gate("G2a", "bookkeeping", "no dataset identity appears as a VALUE in the code",
             "0 offending constants outside config/", "literals", 0, _eq(0.0), status=DONE,
             explain="A scan of the prototype's own Python on the ABSTRACT SYNTAX TREE rather "
                     "than as text, checking every string and numeric constant except docstrings. "
                     "That distinction is the point: naming a dataset in prose is documentation, "
                     "while the same name used as a value is a hardcoded path. Scanning the text "
                     "would flag the first; skipping strings entirely would miss the second, "
                     "because a path IS a string. Reviewing it found a hole -- the patterns were "
                     "word-bounded, so a name inside a filename slipped through and a "
                     "planted-violation check caught only 2 of 3. The boundaries are gone and all "
                     "4 are caught, while docstring prose and innocent constants still are not. "
                     "Note what this does NOT establish: it is a necessary condition, not the "
                     "claim. See G2b."),
        Gate("G2b", "bookkeeping", "ONE pipeline actually runs on all three datasets",
             "3 of 3 datasets complete generate/train/test with only the config changed",
             "datasets", 8, _eq(3.0),
             explain="The claim G2a only gestures at. The point of forbidding dataset identity in "
                     "the code is that the same trainer should run on a toy, on a point-cloud "
                     "recording and on a field recording with nothing changing but the yaml. That "
                     "can only be checked once all three loaders exist and all three have been "
                     "run end to end, which is stage 8. Until then the scanner is passing partly "
                     "because the most likely place to hardcode a path -- the ZAPBench and redox "
                     "loaders -- has not been written yet."),
        Gate("G3", "bookkeeping", "the transfer pair returns what it was given",
             "< 1e-6 of the field value", "fraction of the field value", 4, _lt(1e-6),
             explain="The encoder/decoder option moves state onto a background grid and back. If "
                     "it is sound, depositing a constant and gathering it again returns the "
                     "constant. This is the end-to-end version of G4 and it catches what G4 "
                     "cannot: a transfer pair that is not each other's adjoint, an off-by-one in "
                     "the stencil, a normalisation applied on one side only. The threshold is a "
                     "FRACTION of the field value, so it does not depend on what the field is."),
        Gate("G4", "bookkeeping", "the transfer conserves what it moves",
             "|sum(w) - 1| < 1e-6", "dimensionless", 0, _lt(1e-6), status=DONE,
             explain="The local half of G3. Each transfer spreads a node's value over the corners "
                     "of the grid cell it sits in, and those weights must sum to one, or the "
                     "transfer quietly changes the total amount of stuff every time it is "
                     "applied. Summing to one is also exactly the condition that makes "
                     "interpolation reproduce a constant, which is why G3 tests the same property "
                     "from the outside. Dimensionless by construction. Its stage is 0, not 5, "
                     "because it tests `mpm_ops.bspline` directly -- the transfer the "
                     "encoder/decoder option wraps already exists, so no model and no wiring are "
                     "needed. Measured 2.4e-07 over 2-D and 3-D at three resolutions, which is "
                     "float32 machine precision, so the 1e-6 threshold is about eight epsilons: "
                     "sharp enough to catch a real error, loose enough not to fail on rounding. "
                     "The negative control -- dropping the middle B-spline lobe -- reads 9.4e-01."),
        Gate("G5", "bookkeeping", "the simple option IS the existing model, arithmetically",
             "< 1e-5 of the voltage range", "fraction of the voltage range", 2, _lt(1e-5),
             explain="The pivot of the whole prototype. With `simple`, one pass and no "
                     "encoder/decoder, this model is meant to be connectome-gnn's NeuralGNN term "
                     "for term. The gate copies NeuralGNN's weights across and requires the two "
                     "to produce the same numbers. If it passes, everything downstream is a "
                     "controlled variation on a model already known to reach R^2_W around 0.97. "
                     "If it fails, no later result can be interpreted at all, because a new "
                     "model's failure and a reimplementation bug are indistinguishable."),
        Gate("G6", "bookkeeping", "depth is an option, not a different model",
             "bit-identical (max |delta| == 0)", "absolute", 3, _eq(0.0),
             explain="Both message-passing MLPs have their final layer initialised to zero, so "
                     "every residual block is EXACTLY the identity before training. It follows "
                     "that one pass and sixteen passes must give the same numbers at step zero. "
                     "The threshold is exactly zero with no tolerance, because this is an "
                     "algebraic identity rather than a numerical one. It catches a residual that "
                     "is not a residual -- a missing skip connection, or an initialisation that "
                     "makes the stack a different model at every depth."),
        Gate("G7", "bookkeeping", "the spec is allowed to carry a unit",
             "units declared, and no measurement threshold in mesh units", "boolean", 0, _eq(1.0),
             status=DONE,
             explain="Two halves, both with teeth. The spec must declare a units block, because "
                     "plexus/units.py is explicit that a model without one is dimensionless and "
                     "no result from it may be quoted with a unit -- and every measurement-tier "
                     "gate is a comparison against a quantity. And no measurement threshold may "
                     "be denominated in grid cells, voxels or steps; that half is the lesson from "
                     "the ecm study, where a penetration of 0.82 grid cells sounded small and was "
                     "15 microns, nearly two cell diameters. A threshold in the mesh's own "
                     "currency is the easiest one to pass. Both halves were checked against "
                     "negative controls: a spec with no units block is refused, a spec declaring "
                     "a DERIVED unit is refused, and poisoning one measurement gate's unit to "
                     "'grid cells' makes this gate fail and name the offender. WHAT IT DOES NOT "
                     "ESTABLISH: it compares a unit LABEL against a blocklist, so it verifies "
                     "that the declaration is honest in form, not that the number is really in "
                     "that unit; and it says nothing about whether any result has actually been "
                     "converted. See G7b."),
        Gate("G7b", "bookkeeping", "a measurement result is REPORTED in the declared unit",
             "every tier-3 measured value carries its declared unit through the conversion",
             "boolean", 7, _eq(1.0),
             explain="The half G7 cannot reach. Declaring length_um = 100 does not convert "
                     "anything; it only makes a conversion possible. Whether a measured tier-3 "
                     "value is actually reported in seconds or micrometres rather than in "
                     "frames or cells can only be checked once a tier-3 gate has run, which is "
                     "stage 7. Until then G7 establishes that the spec is ALLOWED to carry a "
                     "unit, and nothing about whether it does."),
        Gate("G8", "bookkeeping", "one-step accuracy is not stability",
             "state norm stays < 2x the ground-truth norm", "ratio to the GT norm", 3, _lt(2.0),
             explain="A model can predict the next increment almost perfectly and still blow up "
                     "when it is fed its own output twenty times over, because a one-step fit "
                     "never sees its own error compound. This runs a 20-step rollout and requires "
                     "the state to stay bounded. The threshold is a RATIO to the ground-truth "
                     "norm, so it is dimensionless and means the same thing on the toy and on "
                     "real data."),
        # ---- tier 2: closed form -------------------------------------------------------- #
        # RESTATED for the two-scale wave toy. The thresholds are unchanged and none of these has
        # been run; only the objects they name have, because the toy they will be run on changed.
        # On this toy the fine rule IS a spatial derivative, so "did the model recover the
        # interaction" and "did the message become a gradient operator" are the same question, and
        # the second is the one that can be measured directly.
        Gate("G9", "closed_form", "the message becomes a gradient operator",
             "R^2 > 0.90 against the true field gradient", "R^2 against du/dx", 2, _gt(0.90)),
        Gate("G10", "closed_form", "recover the per-node time constant",
             "R^2 > 0.95 against the known tau", "R^2 against known tau", 2, _gt(0.95)),
        Gate("G11", "closed_form", "the embedding recovers the types (small toy)",
             "ARI > 0.70 against the 6 true types", "adjusted Rand index", 2, _gt(0.70)),
        Gate("G12", "closed_form", "the embedding recovers the types (large toy)",
             "ARI > 0.70 against the 65 true types", "adjusted Rand index", 5, _gt(0.70)),
        Gate("G13", "closed_form", "recover the per-node SIGNED GAIN (the heterogeneity)",
             "R^2 > 0.90 against the true g_i", "R^2 against the true g_i", 2, _gt(0.90)),
        Gate("G14", "closed_form", "encoder/decoder is a genuine option: on vs off",
             "|delta R^2(gradient)| < 0.03", "R^2 difference", 4, _abs_lt(0.03)),
        Gate("G15", "closed_form", "graphcast vs simple message is RESOLVED, either way",
             "|delta| reported against the 3-seed floor; below it is UNRESOLVED, not ranked",
             "R^2 difference vs the measured floor", 3, None),
        Gate("G16", "closed_form", "types are spatially mixed by construction",
             "spatial-cell type purity within 20% of chance (1/n_types)",
             "purity as a multiple of chance", 1, _lt(1.2)),
        # ---- stage 1b: the toy is a valid test bed (data only, no model) ----------------- #
        # Added after three toys failed for reasons that had nothing to do with the model: a
        # circuit at a fixed point, a stimulus field that was identically zero, and a travelling
        # wave whose gradient is recoverable node-locally (du/dx = -(1/c) du/dt) so the graph was
        # never necessary. In every case training was run before the data was known to pose the
        # problem it claimed to. These need no model and each would have caught one of them.
        # Thresholds are principled: a deterministic rule is recoverable at R^2 > 0.95 by
        # definition, and 0.80 excludes collinearity rather than describing what was seen.
        Gate("G21", "closed_form", "the coarse field is a travelling wave, cyclic left to right",
             "phase drift per frame within 5% of lambda/period", "fraction of lambda/period",
             1, _lt(0.05)),
        Gate("G22", "closed_form", "the fine rule is exactly recoverable from (v, grad u)",
             "minimum per-node R^2 > 0.90", "R^2, worst node", 1, _gt(0.90)),
        Gate("G23", "closed_form", "the gradient is reconstructible from neighbours' states",
             "R^2 > 0.95, else the graph cannot carry the fine rule", "R^2", 1, _gt(0.95)),
        Gate("G24", "closed_form", "the heterogeneity is linearly readable",
             "corr(fitted gain, true g_i) > 0.90", "Pearson correlation", 1, _gt(0.90)),
        # G26 IS THE GATE THAT WOULD HAVE CAUGHT THE TRAVELLING-WAVE DEFECT DIRECTLY. For any
        # u = f(x - ct) the gradient is du/dx = -(1/c) du/dt, so a model that can see the drive
        # recovers it from that node's own history and the graph is never needed -- measured, on
        # the first wave toy, as a loss of 0.005 with gradient recovery of 0.000. A test bed whose
        # fine rule is solvable node-locally cannot test a graph model at all, whatever else is
        # true of it. The threshold is a gap, not a level: the node-local baseline must FAIL where
        # the neighbour-informed fit succeeds.
        Gate("G26", "closed_form", "the graph is NECESSARY: a node-local baseline cannot fit",
             "node-local R^2 < 0.50 while (v, grad u) exceeds 0.90",
             "R^2 of a node-local predictor", 2, _lt(0.50)),
        Gate("G27", "closed_form", "which coarse rule forces the graph",
             "the three toys ranked by G26; reported, not tuned",
             "spread in node-local R^2 across the three coarse rules", 6, None),
        Gate("G25", "closed_form", "connected nodes are not collinear",
             "mean |corr| between connected nodes < 0.80", "Pearson correlation", 1, _lt(0.80)),
        # ---- tier 3: measurement -------------------------------------------------------- #
        Gate("G17", "measurement", "ZAPBench held-out prediction of d(dF/F)/dt",
             "R^2 > 0.268, the parameter-free kNN spatial pool", "held-out R^2", 6, _gt(0.268)),
        Gate("G18", "measurement", "the learned stimulus gain b_i is spatially structured",
             "Moran's I > 0.2 over the soma graph, against a permutation null",
             "Moran's I", 6, _gt(0.2)),
        Gate("G19", "measurement", "fitted calcium decay time constant",
             "0.5 - 2 s (GCaMP6)", "seconds", 6, _within(0.5, 2.0)),
        Gate("G20", "measurement", "redox field fit reproduces the washout response",
             "THRESHOLD TO BE FIXED from Development_Time_Trend.xlsx, before the run",
             "minutes and sign", 7, None),
    ]
    return {x.gid: x for x in g}


def _order(gid: str):
    """Sort G1, G1b, G2, ... G20 numerically with a lettered suffix after its number."""
    m = re.match(r"G(\d+)([a-z]*)$", gid)
    return (int(m.group(1)), m.group(2)) if m else (10**6, gid)


def tier_counts(table: dict[str, Gate]) -> dict[str, int]:
    out = {t: 0 for t in TIERS}
    for g in table.values():
        out[g.tier] += 1
    return out


def write_csv(table: dict[str, Gate], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "tier", "stage", "status", "gate", "threshold", "unit", "measured",
                    "outcome", "artifacts", "note", "explain"])
        for gid in sorted(table, key=_order):
            g = table[gid]
            w.writerow([g.gid, g.tier, g.stage, g.status, g.what, g.threshold, g.unit,
                        "" if g.measured is None else f"{g.measured:.6g}", g.outcome,
                        " ".join(g.artifacts), g.note, g.explain])
    return path


def _tex_escape(s: str) -> str:
    """Escape for LaTeX text mode. `<`, `>` and `|` need math mode or they render as inverted
    punctuation, which is the kind of silent corruption a generated table is most likely to carry
    into a note nobody re-reads."""
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("^", r"\^{}"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("<", r"$<$"), (">", r"$>$"), ("|", r"$|$")):
        s = s.replace(a, b)
    return s


def write_tex(table: dict[str, Gate], path: str, rel_to: str | None = None) -> str:
    """A \\tblGates macro for note_graphcast_plexus.tex, in the note_spheroid_bm_ecm.tex sec. 4
    format: id | gate | threshold | measured, grouped by tier, coloured by outcome."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    colour = {PASS: r"\gpass", FAIL: r"\gfail", SKIP: r"\gskip"}
    base = rel_to or os.path.dirname(os.path.abspath(path))
    lines = ["% GENERATED by gates.py -- do not edit",
             r"\newcommand{\tblGates}{%",
             r"{\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.15}",
             # FIXED COLUMN WIDTHS. An `l` column does not wrap, so a long gate description expands
             # without bound and crushes every X column beside it -- which is how the first version
             # of this table rendered its thresholds one word per line and ran its headers
             # together. Widths are explicit and sum to the text block.
             r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.7cm}>{\raggedright\arraybackslash}p{4.2cm}>{\raggedright\arraybackslash}p{3.3cm}r@{\hspace{6pt}}>{\raggedright\arraybackslash}p{1.25cm}>{\raggedright\arraybackslash}p{1.35cm}>{\raggedright\arraybackslash}p{2.0cm}@{}}",
             r"\toprule",
             r"\textbf{id} & \textbf{gate} & \textbf{threshold} & \textbf{measured} & "
             r"\textbf{outcome} & \textbf{status} & \textbf{figure}\\",
             r"\midrule"]
    label = {"bookkeeping": "Bookkeeping \\textnormal{--- does the code do what the operator says?}",
             "closed_form": "Closed form \\textnormal{--- does it reproduce the physics it was given?}",
             "measurement": "Measurement \\textnormal{--- does it agree with something observed?}"}
    for tier in TIERS:
        rows = [table[k] for k in sorted(table, key=_order) if table[k].tier == tier]
        if not rows:
            continue
        lines.append(r"\addlinespace[2pt]\multicolumn{7}{@{}l}{\textbf{" + label[tier] + r"}}\\[1pt]")
        for g in rows:
            meas = "---" if g.measured is None else f"{g.measured:.4g}"
            # CLICKABLE, not just named: the artifact is the evidence, so the table has to take a
            # reader to it. Paths are relative to the note, which is written beside the run dir.
            art = ", ".join(
                r"\href{" + os.path.relpath(a, base).replace(os.sep, "/") + "}{"
                + _tex_escape(os.path.basename(a)) + "}"
                for a in g.artifacts) or "---"
            lines.append(f"{g.gid} & {_tex_escape(g.what)} & {_tex_escape(g.threshold)} & "
                         f"{meas} & {colour[g.outcome]}{{{g.outcome}}} & {{\\tiny {art}}}\\\\")
    # THE FIGURES THEMSELVES, not only links. A gate's evidence should be visible in the document
    # a reader is already holding; a link is a promise that the reader will go and look.
    figs = [r"\newcommand{\gateFigures}{%"]
    for gid in sorted(table, key=_order):
        g = table[gid]
        pngs = [a for a in g.artifacts if a.lower().endswith(".png")]
        if not pngs:
            continue
        for a in pngs:
            rel = os.path.relpath(a, base).replace(os.sep, "/")
            figs += [r"\begin{figure}[htbp]\centering",
                     r"\includegraphics[width=\linewidth]{" + rel + "}",
                     r"\caption{\textbf{" + g.gid + r"} --- " + _tex_escape(g.what)
                     + r". Threshold: " + _tex_escape(g.threshold)
                     + (r". Measured: " + f"{g.measured:.4g}" if g.measured is not None else "")
                     + r". Outcome: " + g.outcome + r".}",
                     r"\end{figure}"]
        mp4s = [a for a in g.artifacts if a.lower().endswith(".mp4")]
        for a in mp4s:
            rel = os.path.relpath(a, base).replace(os.sep, "/")
            figs.append(r"\noindent\small " + g.gid + r" also has a movie: \href{" + rel + "}{"
                        + _tex_escape(os.path.basename(a)) + r"}.\par\medskip")
    figs.append("}")

    # PLAIN-ENGLISH DEFINITIONS, one per gate that has one. The table says what was measured;
    # this says what the gate is FOR, which a threshold alone never conveys.
    defs = [r"\newcommand{\gateDefinitions}{%"]
    for tier in TIERS:
        rows = [table[k] for k in sorted(table, key=_order)
                if table[k].tier == tier and table[k].explain]
        if not rows:
            continue
        defs.append(r"\subsection{" + label[tier].split(" \\textnormal")[0] + r"}")
        for g in rows:
            defs.append(r"\noindent\textbf{" + g.gid + r"} --- \emph{"
                        + _tex_escape(g.what) + r"}. " + _tex_escape(g.explain)
                        + r"\par\smallskip")
    defs.append("}")

    counts = tier_counts(table)
    lines += [r"\bottomrule\end{tabular}}}",]
    lines += defs
    lines += figs
    lines += [
              r"\newcommand{\tierProportion}{"
              + f"{counts['bookkeeping']} bookkeeping / {counts['closed_form']} closed form / "
                f"{counts['measurement']} measurement" + "}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _md(text: str) -> str:
    """Escape for a markdown TABLE CELL. The pipe is the column delimiter, so an unescaped one in
    a threshold like |sum(w) - 1| < 1e-6 silently splits the row into extra columns."""
    return str(text).replace("|", r"\|")


def write_md(table: dict[str, Gate], path: str, rel_to: str | None = None) -> str:
    """The gate report as markdown: a table, then a plain-English definition per gate.

    Markdown rather than LaTeX because the report is read in the editor beside the code, not
    printed. Figures are linked rather than embedded; a viewer that renders markdown shows them
    inline anyway.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    base = rel_to or os.path.dirname(os.path.abspath(path))
    label = {"bookkeeping": "Tier 1 — bookkeeping: does the code do what the operator says?",
             "closed_form": "Tier 2 — closed form: does it reproduce the physics it was given?",
             "measurement": "Tier 3 — measurement: does it agree with something observed?"}
    mark = {PASS: "**PASS**", FAIL: "**FAIL**", SKIP: "·"}
    out = ["# Gate report", "",
           f"{summary(table)}", "",
           "`status` is not `outcome`. **Outcome** is what the number did against the threshold;",
           "**status** is whether the gate has been walked through — definition written, estimator",
           "sanity-checked, negative control run, result read. A gate can pass mechanically and",
           "still be pending review.", ""]
    for tier in TIERS:
        rows = [table[k] for k in sorted(table, key=_order) if table[k].tier == tier]
        if not rows:
            continue
        out += [f"## {label[tier]}", "",
                "| id | gate | threshold | measured | outcome | status | figures |",
                "|---|---|---|---|---|---|---|"]
        for g in rows:
            meas = "—" if g.measured is None else f"{g.measured:.4g}"
            figs = ", ".join(f"[{os.path.basename(a)}]({os.path.relpath(a, base)})"
                             for a in g.artifacts) or "—"
            out.append(f"| {g.gid} | {_md(g.what)} | {_md(g.threshold)} | {meas} | "
                       f"{mark[g.outcome]} | {g.status.lower()} | {figs} |")
        out.append("")
        described = [g for g in rows if g.explain]
        if described:
            out += [f"### What each {tier.replace('_', ' ')} gate is for", ""]
            for g in described:
                out += [f"**{g.gid} — {g.what}.** {g.explain}", ""]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return path


def summary(table: dict[str, Gate]) -> str:
    n = {PASS: 0, FAIL: 0, SKIP: 0}
    for g in table.values():
        n[g.outcome] += 1
    c = tier_counts(table)
    return (f"{n[PASS]} pass, {n[FAIL]} fail, {n[SKIP]} not yet run "
            f"(of {len(table)}: {c['bookkeeping']} bookkeeping, {c['closed_form']} closed form, "
            f"{c['measurement']} measurement)")
