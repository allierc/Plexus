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
FORBIDDEN_PATTERNS = [
    (r"/gr" + r"oups/", "an absolute data path"),
    (r"\bzap" + r"bench\b", "a dataset name"),
    (r"\bre" + r"dox\b", "a dataset name"),
    (r"\b(717" + r"21|78" + r"70|137" + r"41)\b", "a dataset dimension"),
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
        Gate("G1", "bookkeeping", "all four options PARSE (24 combinations)",
             "24 of 24 option combinations load", "configs", 0, _eq(24.0)),
        Gate("G1b", "bookkeeping", "all four options BUILD and take one step",
             "24 of 24 option combinations run one forward step", "configs", 2, _eq(24.0)),
        Gate("G2", "bookkeeping", "nothing dataset-specific is hardcoded",
             "0 offending literals outside config/", "literals", 0, _eq(0.0)),
        Gate("G3", "bookkeeping", "scatter->gather round trip on a constant field",
             "< 1e-6 of the field value", "fraction of the field value", 4, _lt(1e-6)),
        Gate("G4", "bookkeeping", "transfer weights are a partition of unity",
             "|sum(w) - 1| < 1e-6", "dimensionless", 4, _lt(1e-6)),
        Gate("G5", "bookkeeping", "simple + 1 pass + no enc/dec is arithmetically NeuralGNN",
             "< 1e-5 of the voltage range", "fraction of the voltage range", 2, _lt(1e-5)),
        Gate("G6", "bookkeeping", "residual blocks start at identity: 1 vs 16 passes at init",
             "bit-identical (max |delta| == 0)", "absolute", 3, _eq(0.0)),
        Gate("G7", "bookkeeping", "units declared, and every measurement threshold is in "
             "phenomenon units", "1 = declared and checked", "boolean", 0, _eq(1.0)),
        Gate("G8", "bookkeeping", "a K=20 rollout on the toy does not diverge",
             "state norm stays < 2x the ground-truth norm", "ratio to the GT norm", 3, _lt(2.0)),
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
        w.writerow(["id", "tier", "stage", "gate", "threshold", "unit", "measured", "outcome",
                    "artifacts", "note"])
        for gid in sorted(table, key=_order):
            g = table[gid]
            w.writerow([g.gid, g.tier, g.stage, g.what, g.threshold, g.unit,
                        "" if g.measured is None else f"{g.measured:.6g}", g.outcome,
                        " ".join(g.artifacts), g.note])
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
             r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.75cm}>{\raggedright\arraybackslash}p{4.7cm}>{\raggedright\arraybackslash}p{3.7cm}r@{\hspace{8pt}}>{\raggedright\arraybackslash}p{1.5cm}>{\raggedright\arraybackslash}p{2.4cm}@{}}",
             r"\toprule",
             r"\textbf{id} & \textbf{gate} & \textbf{threshold} & \textbf{measured} & "
             r"\textbf{outcome} & \textbf{figure}\\",
             r"\midrule"]
    label = {"bookkeeping": "Bookkeeping \\textnormal{--- does the code do what the operator says?}",
             "closed_form": "Closed form \\textnormal{--- does it reproduce the physics it was given?}",
             "measurement": "Measurement \\textnormal{--- does it agree with something observed?}"}
    for tier in TIERS:
        rows = [table[k] for k in sorted(table, key=_order) if table[k].tier == tier]
        if not rows:
            continue
        lines.append(r"\addlinespace[2pt]\multicolumn{6}{@{}l}{\textbf{" + label[tier] + r"}}\\[1pt]")
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

    counts = tier_counts(table)
    lines += [r"\bottomrule\end{tabular}}}",]
    lines += figs
    lines += [
              r"\newcommand{\tierProportion}{"
              + f"{counts['bookkeeping']} bookkeeping / {counts['closed_form']} closed form / "
                f"{counts['measurement']} measurement" + "}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def summary(table: dict[str, Gate]) -> str:
    n = {PASS: 0, FAIL: 0, SKIP: 0}
    for g in table.values():
        n[g.outcome] += 1
    c = tier_counts(table)
    return (f"{n[PASS]} pass, {n[FAIL]} fail, {n[SKIP]} not yet run "
            f"(of {len(table)}: {c['bookkeeping']} bookkeeping, {c['closed_form']} closed form, "
            f"{c['measurement']} measurement)")
