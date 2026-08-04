"""predict -- the scoring registry, and the parser that turns a prediction into checkable clauses.

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


THE BANK IS A PRODUCT, NOT A LIST (rebuilt 2026-08-04)
------------------------------------------------------------------------------------------------
Until today the bank was forty hand-listed names, and every new instrument was one more line in a
comma-separated block that a role reads before writing a claim. Two things broke:

  * A FLAT LIST IS A LIST YOU READ THE FIRST ITEM OF. Round 2 wrote ALL TWELVE of its predictions
    on `protr_peak` while act_cv, corr_act_rad and act_alive_frac sat unnamed in the same block.
  * A NAME PER REDUCTION DOES NOT SCALE. `protr_peak` and `protr_final` were hand-minted; the
    thirteen `_final` twins deleted on 4 August were hand-minted the other way. Nobody could say
    which reductions existed, because there was no rule -- only precedent.

So the bank is now generated:  24 QUANTITIES  x  6 TEMPORAL REDUCTIONS, plus nine genuinely
run-level scalars that no reduction of a per-frame column can express. That is 153 admissible
names, and an agent reads THIRTY-NINE things: 24 quantities grouped by the five questions the
campaign actually asks, 6 suffixes, 9 scalars.

  A QUANTITY is a per-frame column. Every one of the 24 is produced by
  `tube_analysis.frame_metrics` (verified by reading it; `t_metrics_have_producers` re-verifies
  on every test run). Fourteen of them are ALSO produced every frame by `run_one.frame_metrics`
  -- the chemistry and centroid tiers -- which is where their reductions should be taken from,
  because the mesh tier is sampled every 25 frames and okuda_route's activator is a limit cycle
  of period 53 frames: 2.1 samples per cycle, below Nyquist.

  A SUFFIX is a reduction over the VALID samples -- those at or before the evidence horizon.
  THE HORIZON IS A FRAME NUMBER AND metrics.npz ROWS ARE SAMPLES, and confusing the two truncated
  folded runs to their opening frames (a real bug, fixed 2026-08-04). A reduction that indexes a
  column by the horizon FRAME is wrong by the stride.

WHAT IS NOT IN THE BANK IS AS DELIBERATE AS WHAT IS. Three of the columns dropped today were
dropped because they are RAILS -- measured on okuda_route's own per-frame record
(log/okuda/okuda_route/frames_1.npz, 901 frames, mesh tier at stride 25), inside its own evidence
horizon (last clean sample = frame 150):

    broken_frac      peak 0.0005 over the whole valid window. It is measured INSIDE the window it
                     itself defines -- the horizon is the last frame with broken_n < 1 -- so it is
                     ~0 there BY CONSTRUCTION, on every run, forever.
    ray_single_frac  final 1, peak 1, floor 1, span 0. Constant. Same reason: the horizon truncates
                     at ray_single_frac < 0.5, so within the window it cannot say anything. Note
                     this kills ALL SIX reductions, not merely `_peak`.
    V_enclosed       521.1 -> 522.3 inside the window: a raw volume in arbitrary units, with no
                     threshold that transfers between runs. `reduced_volume` is its dimensionless
                     form and IS admitted.

They remain COLUMNS. `broken_n` still sets the horizon, `ray_single_frac` still detects the fold
and still gates `morphology.classify`, `euler` still checks the topology, `V_total` still answers
premise #1. Not being predictable-against is not the same as not being measured.
"""
from __future__ import annotations

import re

# =================================================================================================
# THE SIX REDUCTIONS. A suffix says WHICH NUMBER of a trajectory the claim is about, and the six
# are the ones that distinguish the runs this campaign keeps confusing:
#
#     a tube that grew and held        final high, peak ~ final, trend +, span small
#     a spike at the moment it tore    peak high, final low, span >> |trend|
#     a rail                           span 0 -- it never moved, and no threshold on it means
#                                      anything (this is how broken_frac and ray_single_frac were
#                                      caught)
#     a refusal                        measured_frac < 1 -- the number does not exist for most of
#                                      the run, and the honest verdict is `inconclusive`
#
# `_floor` IS NOT CALLED `_min` because `shape_idx_min` is a QUANTITY whose name ends in `_min`,
# the parser's alternation is longest-first, and the product would then mint `shape_idx_min_min`
# beside `shape_idx_min` -- two names one edit apart for two different things.
SUFFIXES = ("_final", "_peak", "_floor", "_trend", "_span", "_measured_frac")

# WHAT EACH REDUCTION IS, COPIED FROM ITS PRODUCER RATHER THAN IMAGINED. `time_analysis.
# reduce_series` computes these and `REDUCTIONS` there is this tuple without the leading
# underscore; the numbers quoted are its own, measured on okuda_route with the horizon at frame
# 150. A note that describes a reduction the reducer does not compute is worse than no note: it
# is an instrument advertised to every role that does not exist. The two most likely to be
# misremembered are `_trend`, which is a RANK CORRELATION in [-1,+1] and not a difference, and
# `_span`, which is DIVIDED BY THE MEDIAN and so is dimensionless.
SUFFIX_NOTES = {
    "_final": "the value AT the evidence horizon -- V[h] exactly, not the last finite value "
              "before it, so all 24 series of one run are read at the SAME moment. When the "
              "instrument was refusing at that frame this is absent, and _measured_frac is how "
              "you tell a refusal from a breakage",
    "_peak": "max over the finite samples at or before the horizon. Says NOTHING about whether it "
             "was held: a plateau and a spike at the instant the mesh tore give the same number, "
             "which is why _final and _trend sit beside it",
    "_floor": "min over the same samples (called _floor, not _min, because shape_idx_min is a "
              "QUANTITY and the parser is longest-first). The one that catches a collapse",
    "_trend": "SPEARMAN RANK CORRELATION between the value and the frame number, in [-1,+1]: "
              "+1 = monotone climb, 0 = no direction, -1 = monotone fall. Rank-based, so one "
              "spike cannot manufacture a trend and units drop out. Absent below 4 finite "
              "samples. n_cells_trend = +0.999 on okuda_route -- growth, monotone",
    "_span": "(peak - floor) / |median|, so it is DIMENSIONLESS and 951,288 and 0.0065 sit on one "
             "axis. 0 = a RAIL, and no threshold on a rail discriminates anything; order 1 = a "
             "live quantity; act_max_span = 1.3e6 inside okuda_route's window while every shape "
             "word for that run says `sphere`",
    "_measured_frac": "fraction of the samples inside the window that are FINITE. Below 1 for two "
                      "reasons: the mesh tier is sampled every 25 frames, and some quantities "
                      "REFUSE to report (corr_act_rad returns nothing when act_cv < 0.05). "
                      "okuda_route: corr_act_rad_measured_frac = 0.046 inside the window, 0.230 "
                      "over the whole run -- read it BEFORE the other five",
}

# =================================================================================================
# THE 24 QUANTITIES, grouped by the FIVE QUESTIONS. Every one is produced by
# `tube_analysis.frame_metrics`; fourteen are also produced every frame by `run_one.frame_metrics`.
#
# The grouping is not decoration. A prediction that names only shape metrics has not asked whether
# there was a pattern, or whether the run was evidence at all, and the group headings are what
# makes that visible to the role writing the claim.
SERIES_METRICS = {
    "IS IT A TUBE -- the shape of the tissue": (
        "protr", "protr_p99", "r_cv", "gyr_prolate", "gyr_oblate", "reduced_volume",
        "n_tubes", "tube_diam", "n_tips", "protrusion_aspect_max"),
    "IS IT STILL MADE OF CELLS -- or is the mesh being measured": (
        "cells", "v_cell_mean", "shape_idx_med", "shape_idx_p95", "shape_idx_min"),
    "IS THERE A PATTERN AT ALL -- the Turing field": (
        "act_cv", "act_mean", "act_max", "red_frac", "n_spots", "spot_spacing_cells"),
    "DOES THE PATTERN GRIP THE SHAPE -- the campaign's question": (
        "corr_act_rad", "act_at_tip", "red_at_tip"),
    "IS THIS EVIDENCE AT ALL -- the apparatus, not the biology": (),
}

# WHY THE FIFTH QUESTION HAS NO PER-FRAME QUANTITY, and why that is the right answer rather than a
# gap. Every per-frame evidence column is measured INSIDE the window it defines: the horizon is
# the last frame with broken_n < 1 and the last frame with ray_single_frac >= 0.5, so within the
# valid window broken_frac is ~0 and ray_single_frac is 1.0, on every run, by construction (both
# verified on okuda_route: peak 0.0005 and span 0.0). The question is answered instead by
# RUN-LEVEL scalars, below, and by the horizon itself, which every prompt already carries.

# =================================================================================================
# THE RUN-LEVEL SCALARS. Admitted only where NO reduction of a per-frame column can express them:
# they come from a separate probe (mechanics, the quasi-static relaxation), from an operator's own
# history (divide_3d's refusal counter), or from a whole-run predicate that is not one of the six.
# Each was checked to be actually WRITTEN into the summary before being kept here.
SCALAR_METRICS = {
    "IS THERE A PATTERN AT ALL -- the Turing field": (
        "act_alive_frac", "act_extinct_frame", "act_peak_frame"),
    "IS THIS EVIDENCE AT ALL -- the apparatus, not the biology": (
        "mech_p_ratio", "Q_drop", "div_blocked", "buf_full", "div_blocked_first_frame"),
}

SERIES_QUANTITIES = tuple(q for v in SERIES_METRICS.values() for q in v)
SCALAR_QUANTITIES = tuple(q for v in SCALAR_METRICS.values() for q in v)

# Measured to lie by the instrument gate (F15/F16). Kept nameable so a prediction resting on one
# is REPORTED as such rather than silently unrecognised -- saying so is the point of the gate.
REJECTED_METRICS = ("ta_aspect_len_over_diam", "ta_tube_len_final", "retention",
                    "autocorr_hops_uncalibrated")

# WITHDRAWN -- admitted yesterday, not admitted today, WITH THE REASON. A withdrawn name that is
# simply deleted becomes "no clause naming a known metric", which tells the agent that wrote it
# nothing at all; the archive is full of predictions on these names and the loop will keep
# reaching for them until it is told why not to. None of these is wrong -- each is a SECOND NAME
# for something the bank already has, or a series quantity admitted only at its endpoint.
#
# KEYED BY QUANTITY, expanded over the six suffixes below, because a withdrawn COLUMN is withdrawn
# in all six of its reductions: `ray_single_frac_span` has to be answerable, not merely
# `ray_single_frac`, or the explanation is missing from exactly the names an agent is most likely
# to write now that the suffixes exist.
WITHDRAWN_QUANTITIES = {
    "broken_frac": "measured INSIDE the window it defines (the horizon is the last frame with "
                   "broken_n < 1), so all six reductions read ~0 there on every run. okuda_route: "
                   "peak 0.0005 inside its own horizon. broken_n still sets the horizon",
    "ray_single_frac": "same defect: the horizon truncates at ray_single_frac < 0.5, so inside "
                       "the window it is 1.0. okuda_route: final 1, peak 1, floor 1, SPAN 0. It "
                       "still detects the fold and still gates morphology.classify",
    "V_enclosed": "a raw volume in arbitrary units -- 521.1 to 522.3 inside okuda_route's "
                  "horizon. No threshold transfers between runs. Use reduced_volume, its "
                  "dimensionless form, or V_total which is still a column",
    "tip_frac": "a rank statistic of the radius distribution against its own percentiles, so it "
                "CANNOT read zero on a sphere: 0.062-0.089 through okuda_route, which is a "
                "sphere. Use red_at_tip (which does read 0) or protrusion_aspect_max",
    "branch_frac": "same defect and worse: 0.27-0.31 on a sphere. Branching is n_tips exceeding "
                   "the number of protrusion bases, which is what n_tips is for",
    "act_sd": "the scale-CARRYING twin of act_cv, on a field that spans six orders of magnitude "
              "(okuda_route: act_sd floor 0, peak 3029). act_cv is the same claim, scale-free",
    "act_occupancy": "the same question as red_frac asked of the field's own range instead of the "
                     "growth operator's switch. Kept as a column; red_frac is the one growth acts on",
    "hollow_frac": "the frozen legacy blend folded|sliver|under-connected. Correlates with the "
                   "TIP-CELL COUNT at r=+0.97 -- it counts the tube, not the damage",
    "vol_cv": "spread of cell volume. v_cell_mean plus cells says the same thing with a reference "
              "value (premise 3: mean cell volume roughly steady while cells divides)",
    "gyr_asphere": "monotone in gyr_prolate over the range this substrate produces; prolate has "
                   "the reference value (1.0 = sphere) and oblate covers the other failure",
    "shape_idx_mean": "a mean over a distribution with a 4:1 tail (okuda_route p95 = 7.0). "
                      "shape_idx_med and shape_idx_p95 separate the tissue from its worst cells",
    "shape_idx_max": "one cell. shape_idx_p95 is the same warning without resting on a single face",
    "spot_cells_med": "a pattern-scale column the 24 does not carry. n_spots and "
                      "spot_spacing_cells are the two that Okuda's own reading is stated in",
    "spot_cells_max": "as spot_cells_med",
    "spot_frac": "as spot_cells_med, and it is red_frac thresholded on the field's own range "
                 "rather than on the growth switch",
    "wavelength_cells": "never produced under this name. pattern_scale stores it as "
                        "autocorr_hops_uncalibrated because F010 withdrew it as uncalibrated, "
                        "and it is REJECTED under that name",
}

# Withdrawn names that are NOT a quantity plus a suffix -- one entry, one name.
WITHDRAWN_NAMES = {
    "ta_n_tubes_final": "bit-identical to n_tubes_final (both are series[-1]['n_tubes']) and "
                        "UNTRUNCATED, i.e. the pre-horizon behaviour the campaign fixed. Name "
                        "n_tubes_final",
    # TWO PRODUCERS, ONE KEY -- caught by writing the rule down. `n_cells` is the every-frame
    # table's own cell-count column, so `time_analysis.reduce_all` emits `n_cells_final` = the
    # count AT THE HORIZON, while `run_one` writes `n_cells_final` = the count at the LAST FRAME.
    # On okuda_route those are ~2,300 and 3,975. Admitting the name would put the horizon fix and
    # the thing it fixed under one word, which is the protr/ta_protr divergence exactly.
    "n_cells_final": "two producers write this key with two meanings -- run_one's LAST-FRAME "
                     "count (3,975 on okuda_route) and reduce_all's count at the HORIZON "
                     "(~2,300). Name cells_final or cells_peak, which are horizon-truncated by "
                     "construction, and read div_blocked/buf_full for whether the array stopped it",
}

WITHDRAWN = {**{q + s: why for q, why in WITHDRAWN_QUANTITIES.items()
                for s in ("",) + SUFFIXES},
             **WITHDRAWN_NAMES}

# =================================================================================================
# WHAT EACH QUANTITY MEASURES, AND WHAT IT READS WHEN THE ANSWER IS NO.
#
# ONE LINE PER QUANTITY, NOT PER NAME -- 24 lines cover 144 names. A metric with no stated
# reference value cannot be predicted against: "act_cv > 0.3" is a guess unless you know that a
# dead field reads 0.00 and a live Turing field reads about 1. Every note states the NULL reading,
# because that is what makes a prediction a bet rather than a description. Where a number is
# quoted it is measured, on okuda_route's per-frame record.
METRIC_NOTES = {
    # ---- IS IT A TUBE -------------------------------------------------------------------
    "protr": "p95/median of cell radius about the tissue centroid. NO = 1.0 (a sphere). A TAIL "
             "statistic: one long tube and a lumpy ball read alike, and a spike thinner than 5% "
             "of the cells is invisible to it",
    "protr_p99": "p99/median of cell radius. NO = 1.0. The only admitted statistic sensitive to a "
                 "spike in under 5% of cells -- which is what a narrow tube on a 4,000-cell shell is",
    "r_cv": "sd/mean of cell radius. NO = 0. Rises with ANY departure from a sphere, not only "
            "with the tail, so it moves when protr cannot",
    "gyr_prolate": "largest gyration eigenvalue over the mean of the other two. NO = 1.0. Above 1 "
                   "= ELONGATED along one axis: this is what separates a TUBE from an undulating "
                   "ball, which raise protr equally",
    "gyr_oblate": "NO = 0 for a sphere AND for a rod; POSITIVE means flattened -- a vesicle "
                  "collapsing into a disc, a failure that otherwise reads only as `not a tube`",
    "reduced_volume": "6 sqrt(pi) V / A^1.5. NO = 1.0 (a sphere). BELOW 1 the shell holds more "
                      "area than a sphere of that volume can and MUST wrinkle, buckle or fold: "
                      "the mechanical precondition for budding, not a consequence of it",
    "n_tubes": "angular clusters of protruding cells = protrusion BASES. NO = 0 (exactly 0 at all "
               "37 mesh samples of okuda_route). Does NOT move when a tube forks -- that is n_tips",
    "tube_diam": "2x median perpendicular distance of a tube's cells from its axis. NO = 0. "
                 "Okuda's own control target: thin_tube vs thick_tube differ in DIAMETER AT EQUAL "
                 "LENGTH, and n_tubes counts bases, not width",
    "n_tips": "distinct tips summed over all protrusions. NO = 0. A fork is ONE base with TWO "
              "tips, so this is the only admitted quantity that moves when a tube branches (Fig 6)",
    "protrusion_aspect_max": "length/width of the deepest protrusion. NO = 0; a shallow bump is "
                             "~1; >= 1.5 IS morphology.classify's own tube criterion. okuda_route "
                             "peaks at 1.006 -- bumps for 901 frames, never a tube",
    # ---- IS IT STILL MADE OF CELLS ------------------------------------------------------
    "cells": "live faces in the mesh, at the MESH TIER (every 25 frames). NO = the seed count, "
             "flat, _trend 0 (no division at all). A tail that is bit-identical sample after "
             "sample is the VERTEX ARRAY, not biology -- read div_blocked before calling it a "
             "limit of growth",
    "v_cell_mean": "mean cell volume. NO = flat (nothing grew and nothing divided). Roughly "
                   "steady while `cells` climbs = growth absorbed by division, which is premise "
                   "3; falling = division outruns growth; RISING = cells inflate instead of dividing",
    "shape_idx_med": "median perimeter/sqrt(area). NO = 3.545, a circle, which is the hard FLOOR "
                     "for any shape; 3.81 is the rigidity transition -- above it the tissue FLOWS "
                     "and cannot hold a shape it is pushed into. okuda_route sits at 3.85-3.90",
    "shape_idx_p95": "the same for the worst-shaped 5% of cells. NO = ~3.8, i.e. even the worst "
                     "cells are near-regular; okuda_route reaches 7.0, a 4:1 sliver, which means "
                     "the MESH and not the tissue is what the shape numbers describe",
    "shape_idx_min": "the best-shaped cell. Cannot fall below 3.5449 for ANY shape -- that is "
                     "geometry, not biology -- so a value below it is a BROKEN RULER, never a "
                     "finding. NO = ~3.55. It is the one statistic that can prove the ruler lying",
    # ---- IS THERE A PATTERN -------------------------------------------------------------
    "act_cv": "act_sd/act_mean, scale-free, so it survives a collapsing or exploding level. "
              "NO = 0.00 (uniform OR dead, whatever the mean). A real Turing pattern reads "
              "0.9-1.8. okuda_route's median over 901 frames is 0.025: there was no pattern",
    "act_mean": "mean activator. CANNOT SEE A PATTERN -- 0.5 everywhere and half-at-1/half-at-0 "
                "give the same number. Read it for the LEVEL (a collapse, a blow-up), never as "
                "evidence that a field exists",
    "act_max": "peak activator over the cells. NO answer of its own: it reads high for one "
               "exploding cell and for a healthy field alike. okuda_route peaks at 9.5e5, which "
               "is a singularity, not a pattern",
    "red_frac": "fraction of cells above the growth operator's OWN switch a_sw -- the cells growth "
                "actually acts on. NO = 0. LOW = localised spots (distinct tubes); 1.0 means "
                "growth is acting on EVERY cell at once, which grows a sphere (okuda_route: 0.997)",
    "n_spots": "distinct activator domains on the cell graph. NO = 0. Okuda's own reading is "
               "about five on a 2,000-cell ball; 48 is noise and 1 is a single bud",
    "spot_spacing_cells": "centre-to-centre domain spacing IN CELLS -- the only pattern length we "
                          "have that is comparable with the paper (chi is a solver rate, not a "
                          "scale). NOT MEASURED below 2 spots: okuda_route measures it on 70% of "
                          "samples, so read its _measured_frac first",
    # ---- DOES THE PATTERN GRIP THE SHAPE ------------------------------------------------
    "corr_act_rad": "Pearson r between a cell's activator and its radius. NO = 0. REFUSED (not "
                    "measured) when act_cv <= 0.05, because a correlation on a dead field is a "
                    "correlation of round-off -- it reads a confident 0.294 on a spread of 8e-05. "
                    "okuda_route: measured_frac 0.230, so 77% of that run has NO answer here",
    "act_at_tip": "mean activator in the outermost tenth of cells, over the tissue mean. NO = 1.0 "
                  "(no relation to shape); above 1 the activator sits at the protrusions. The "
                  "same question as corr_act_rad without assuming the relation is a straight "
                  "line, and refused on a dead field for the same reason",
    "red_at_tip": "fraction of ACTIVATED cells that are tip cells. NO = 0 -- which is also what it "
                  "reads when nothing is activated at all, so read it beside red_frac. 1.0 = the "
                  "activator is CONFINED to the tips, which is what a clean tube looks like",
    # ---- the run-level scalars ----------------------------------------------------------
    "act_alive_frac": "fraction of frames in which a pattern existed at all (act_cv > 0.05 AND "
                      "occupancy > 0.01). NO = 0.0; 1.0 = patterned throughout; 0.2 = a FLASH, "
                      "and the last 80% of the run grew on a corpse",
    "act_extinct_frame": "the frame at which the pattern LAST stopped existing; absent if it "
                         "never did. Turns `the chemistry died` into a measurement",
    "act_peak_frame": "the frame of maximum act_max. Early, with act_extinct_frame later, is a "
                      "blow-up followed by extinction -- not a pattern",
    "mech_p_ratio": "pressure in protruding cells over body cells. ~1 = a growth-driven "
                    "EQUILIBRIUM shape; ~3 = a FORCED protrusion, held by the driver",
    "Q_drop": "protrusion LOST when the forces are switched off and the tissue relaxes from its "
              "own end state. ~0 = an equilibrium shape; large = the shape was being held",
    "div_blocked": "divisions REFUSED for want of vertex buffer. NO = 0. Non-zero means growth "
                   "stopped where the ARRAY ended, not where the biology did",
    "buf_full": "the vertex array filled (1/0). A run that ends here has not found a limit of "
                "growth, and every growth number after that frame describes the reservoir",
    "div_blocked_first_frame": "the frame at which the buffer first refused a division; "
                               "everything after it is a run against a wall. Absent if it never did",
}

# =================================================================================================
# DERIVED. The bank is generated, so a quantity cannot be added without its six reductions and a
# reduction cannot exist for a quantity nobody declared.
KNOWN_METRICS = (
    tuple(q + s for q in SERIES_QUANTITIES for s in SUFFIXES)
    + SCALAR_QUANTITIES
    + REJECTED_METRICS
)

# A BARE QUANTITY NAME MEANS `_final`, and it is an ALIAS rather than an admitted name. The
# distinction matters twice: `t_no_final_twins` forbids a bank that holds both `act_cv` and
# `act_cv_final` (one quantity, two names, in a vocabulary an agent must read before it can write
# a claim), while the archive is full of predictions written as `protr >= 2.0` and the convention
# has always been that a bare name is the summary value at its last frame. Resolving the alias at
# PARSE time keeps the bank a product and keeps those predictions scorable, and the record shows
# the resolved name so nothing is silently reinterpreted.
ALIAS = {q: q + "_final" for q in SERIES_QUANTITIES}

METRIC_GROUP = {
    grp: tuple(q + s for q in SERIES_METRICS[grp] for s in SUFFIXES)
         + tuple(SCALAR_METRICS.get(grp, ()))
    for grp in SERIES_METRICS
}


def decompose(name):
    """(quantity, suffix) for an admitted name; suffix is None for a run-level scalar.

    THE ONE PLACE THAT KNOWS HOW A NAME IS BUILT. `t_metrics_have_producers` and
    `t_metrics_documented` both need it -- the producer of `act_cv_peak` is whatever produces
    `act_cv` plus whatever applies `_peak`, and the DOCUMENTATION of `act_cv_peak` is act_cv's
    note plus _peak's note. Returns (None, None) for a name that is not admitted.
    """
    if name in SCALAR_QUANTITIES or name in REJECTED_METRICS:
        return name, None
    for s in sorted(SUFFIXES, key=len, reverse=True):
        if name.endswith(s) and name[: -len(s)] in SERIES_QUANTITIES:
            return name[: -len(s)], s
    return (name, None) if name in SERIES_QUANTITIES else (None, None)


def admitted_block(new_since=()):
    """The admissible-metric block for a prompt: quantities by group, then the suffixes ONCE.

    A 53-NAME FLAT LIST IS WHY ROUND 2 WROTE ALL TWELVE PREDICTIONS ON protr_peak. This renders
    24 quantities under the five questions they answer, the six reductions once, and the nine
    run-level scalars -- about fifty lines instead of a comma-separated paragraph, and the group
    headings make "you have not asked whether there was a pattern" visible while the claim is
    being written.

    `new_since` names metrics admitted recently, so their arrival is ANNOUNCED rather than left to
    be noticed. A role that is not told an instrument exists will keep reasoning as though the
    property is unmeasurable -- which is how the finest Turing pattern in the campaign came to be
    recorded as a null sphere.
    """
    L = ["Only these metrics are admissible. EVERY NAME IS A QUANTITY PLUS A SUFFIX: "
         "<quantity><suffix>, e.g. act_cv_peak, protr_final, corr_act_rad_measured_frac. "
         f"{len(SERIES_QUANTITIES)} quantities x {len(SUFFIXES)} suffixes. A bare quantity name "
         "is read as _final. Each note says what the quantity measures AND WHAT IT READS WHEN "
         "THE ANSWER IS NO -- a threshold you cannot state a null for is not a prediction."]
    for grp, names in SERIES_METRICS.items():
        if not names:
            continue
        L += ["", f"  {grp}"]
        L += [f"    {m:22} {METRIC_NOTES.get(m, '')}" for m in names]
    L += ["", "  THE SIX SUFFIXES -- pick the one that states your claim. All are taken over the "
              "VALID samples only (at or before the evidence horizon)."]
    L += [f"    {s:22} {SUFFIX_NOTES[s]}" for s in SUFFIXES]
    L += ["", "  RUN-LEVEL SCALARS -- not per-frame series, so they take NO suffix. Grouped by the "
              "same questions."]
    for grp, names in SCALAR_METRICS.items():
        L.append(f"   [{grp}]")
        L += [f"    {m:22} {METRIC_NOTES.get(m, '')}" for m in names]
    L += ["", "  REJECTED, measured to lie -- a prediction resting on one is not evidence:  "
              + ", ".join(REJECTED_METRICS)]
    L += ["  WITHDRAWN (with all six of their suffixes), each because the bank already answers "
          "the question under a better name -- score() will tell you which:  "
          + ", ".join(sorted(WITHDRAWN_QUANTITIES) + sorted(WITHDRAWN_NAMES))]
    homeless = [m for m in KNOWN_METRICS if m not in REJECTED_METRICS
                and not any(m in v for v in METRIC_GROUP.values())]
    if homeless:
        L += ["", "  UNGROUPED (a metric with no home -- report this): " + ", ".join(homeless)]
    if new_since:
        # AN ANNOUNCEMENT LIST GOES STALE, and a stale one advertises an instrument that no longer
        # exists -- the same fault as the METRIC_NOTES entry for `wavelength_cells_final`, one
        # list over. `agents/llm_agents.NEW_INSTRUMENTS` is hand-kept, so names are resolved
        # against the registry here and the dropped ones are SHOWN rather than quietly filtered:
        # a role that was told about act_sd yesterday needs to be told it went, not to find it
        # missing.
        # NAMES ONLY, ON ONE LINE. Each one's meaning is already printed above, under its
        # quantity; repeating the note here cost fifteen lines of a block whose whole purpose is
        # to be short enough to read before writing a claim.
        live = [m for m in new_since if ALIAS.get(m, m) in KNOWN_METRICS]
        gone = [m for m in new_since if m not in live]
        if live:
            L += ["", "  NEW INSTRUMENTS, admitted since the last campaign -- USE THEM (each is "
                      "defined above, under its quantity). A property you could not measure "
                      "before is not still unmeasurable:", "    " + ", ".join(live)]
        if gone:
            L += ["  ANNOUNCED BEFORE, NO LONGER ADMITTED -- do not reach for these: "
                  + ", ".join(gone)]
    return "\n".join(L)


_OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, "<": lambda a, b: a < b}

# "REFUTED if ...", "falsified if ...", "refuted when ..." -- everything from there on states the
# FALSIFIER, not the prediction. Parsing it as the prediction inverts the outcome.
_FALSIFIER = re.compile(r"\b(?:refuted|falsifie[sd]|disconfirmed|wrong)\b\s*(?:if|when|by)?",
                        re.I)

# LONGEST FIRST, and it now has to carry its own weight: the bank contains `shape_idx_min` (a
# quantity), `shape_idx_min_floor` (its reduction) and `shape_idx_med_floor` (another quantity's),
# and a shortest-first alternation would match `shape_idx_min` inside `shape_idx_min_floor` and
# score a claim about the best-shaped cell's FLOOR against its last frame. This is exactly why the
# floor suffix is not spelled `_min`.
_NAMES = tuple(KNOWN_METRICS) + tuple(ALIAS) + tuple(WITHDRAWN)
_METRIC_ALT = "|".join(sorted(set(_NAMES), key=len, reverse=True))

# metric <op> value            e.g. "protr_peak >= 2.0"
_CLAUSE_OP = re.compile(rf"(?P<metric>{_METRIC_ALT})\s*(?P<op>>=|<=|>|<)\s*(?P<val>-?[0-9.]+)", re.I)
# metric <lo>-<hi> / "between" e.g. "protr_peak 2.0-3.5", "protr_peak between 2 and 3.5"
_CLAUSE_RANGE = re.compile(
    rf"(?P<metric>{_METRIC_ALT})\s*(?:of|:|is|=|~)?\s*"
    rf"(?:between\s+)?(?P<lo>[0-9.]+)\s*(?:-|–|to|and)\s*(?P<hi>[0-9.]+)", re.I)


class Clause:
    """One checkable statement: a metric, a test, and the text it came from."""

    def __init__(self, metric, kind, lo=None, hi=None, op=None, val=None, src=""):
        self.written = metric                       # exactly what the agent typed
        self.metric = ALIAS.get(metric, metric)     # the key the summary is keyed by
        self.kind, self.src = kind, src
        self.lo, self.hi, self.op, self.val = lo, hi, op, val

    def check(self, observed: dict):
        """(holds, why). None if the metric was not measured -- NOT a pass."""
        if self.metric in WITHDRAWN:
            return None, f"{self.metric} is WITHDRAWN: {WITHDRAWN[self.metric]}"
        if self.metric not in observed or observed[self.metric] is None:
            extra = (f" (you wrote `{self.written}`, read as `{self.metric}`)"
                     if self.written != self.metric else "")
            return None, f"{self.metric} not measured{extra}"
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
      * nothing checkable                    -> inconclusive  (NOT confirmed -- defect P1)
      * every clause on a REJECTED/WITHDRAWN -> inconclusive, WITH THE REASON
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
    if all(c.metric in WITHDRAWN for c in clauses):
        return "inconclusive", "; ".join(f"{m} is WITHDRAWN: {WITHDRAWN[m]}"
                                         for m in sorted({c.metric for c in clauses}))

    ranked = [c for c in clauses
              if primary_metric and c.metric == ALIAS.get(primary_metric, primary_metric)] or clauses
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

    print("the bank is a product")
    eq(len(SERIES_QUANTITIES), 24, "24 quantities")
    eq(len(set(SERIES_QUANTITIES)), 24, "no quantity in two groups")
    eq(len(KNOWN_METRICS), 24 * 6 + len(SCALAR_QUANTITIES) + len(REJECTED_METRICS),
       "product + scalars + rejected")
    eq(len(admitted_block().splitlines()) <= 60, True,
       f"admitted_block is {len(admitted_block().splitlines())} lines (budget 60)")
    _ann = admitted_block(new_since=("act_cv_span", "n_tips_peak", "act_sd"))
    eq(len(_ann.splitlines()) <= 60, True,
       f"...and {len(_ann.splitlines())} with announcements (budget 60)")
    eq("NO LONGER ADMITTED" in _ann and "act_sd" in _ann, True,
       "a stale announcement is shown as withdrawn, not recommended")

    print("\nthe reason `_floor` is not `_min`: longest-first must resolve shape_idx_min")
    eq([c.metric for c in parse("shape_idx_min >= 3.545")], ["shape_idx_min_final"],
       "bare shape_idx_min -> its _final")
    eq([c.metric for c in parse("shape_idx_min_floor >= 3.545")], ["shape_idx_min_floor"],
       "shape_idx_min_floor is NOT split into shape_idx_min")
    eq([c.metric for c in parse("shape_idx_min_peak < 3.5")], ["shape_idx_min_peak"],
       "shape_idx_min_peak resolves whole")
    eq([c.metric for c in parse("shape_idx_med_floor > 3.81")], ["shape_idx_med_floor"],
       "the sibling quantity is unaffected")
    eq([c.metric for c in parse("n_cells_final >= 3000")], ["n_cells_final"],
       "n_cells_final resolves whole (it is withdrawn, not unknown)")
    eq([c.metric for c in parse("corr_act_rad_measured_frac < 0.5")],
       ["corr_act_rad_measured_frac"], "the longest suffix wins over the shorter names inside it")

    print("\nP1 -- an unparseable prediction must NOT be scored `confirmed`")
    eq(score("protr_peak 2.0-3.5, mech_p_ratio ~3 (forced), analyst phenotype tube/spike",
             {"protr_peak": 2.8})[0], "confirmed", "range form now parses (in range)")
    eq(score("protr_peak 2.0-3.5, mech_p_ratio ~3", {"protr_peak": 9.1})[0], "refuted",
       "range form, observed outside")
    eq(score("it should look like a nice tube", {"protr_peak": 2.8})[0], "inconclusive",
       "genuinely unstated -> inconclusive, NOT confirmed")
    eq(score("Q > 0.5", {"protr_peak": 2.8})[0], "inconclusive",
       "unknown metric `Q` -> inconclusive, NOT confirmed")

    print("\nP2 -- a threshold must be tested against the metric it names")
    pred = "mech_p_ratio <= 1.5 with protr_peak >= 2.0"
    eq(score(pred, {"protr_peak": 2.4, "mech_p_ratio": 1.2})[0], "confirmed", "both clauses hold")
    eq(score(pred, {"protr_peak": 1.2, "mech_p_ratio": 1.2})[0], "refuted", "protr_peak fails")
    eq(score(pred, {"protr_peak": 2.4, "mech_p_ratio": 3.9})[0], "refuted", "p_ratio fails")
    eq(score(pred, {"protr_peak": 2.4, "mech_p_ratio": 1.2}, primary_metric="protr_peak")[0],
       "confirmed", "primary metric selects the right clause")

    print("\nP3 -- `REFUTED if ...` states the falsifier, not the prediction")
    p = "protr_peak >= 2.0 (within ~0.5 of control); REFUTED if it drops below 1.5"
    eq([c.metric for c in parse(p)], ["protr_peak"], "falsifier clause is discarded")
    eq(score(p, {"protr_peak": 2.4})[0], "confirmed", "assertion holds")
    eq(score(p, {"protr_peak": 1.2})[0], "refuted", "assertion fails")
    r = "REFUTED if protr_peak >= 2.0; I predict protr_peak < 1.5"
    eq(score(r, {"protr_peak": 1.2})[0], "inconclusive",
       "falsifier-first: nothing assertable remains -> inconclusive, not inverted")

    print("\nthe six reductions are scorable, and a rail is sayable")
    obs = {"act_cv_peak": 0.9, "act_cv_final": 0.02, "protr_trend": 0.06,
           "ray_single_frac_span": 0.0, "corr_act_rad_measured_frac": 0.23}
    eq(score("act_cv_peak > 0.5 and act_cv_final < 0.1", obs)[0], "confirmed",
       "a flash-then-extinction is now one prediction")
    eq(score("corr_act_rad_measured_frac >= 0.9", obs)[0], "refuted",
       "`the coupling was measurable throughout` is refutable")

    print("\nwithdrawn, rejected and unmeasured")
    o, w = score("ray_single_frac_span > 0.1", obs)
    eq(o, "inconclusive", "a WITHDRAWN metric is inconclusive, not guessed")
    eq("WITHDRAWN" in w, True, "and the record says why")
    eq(score("ta_tube_len_final >= 4.0", {"ta_tube_len_final": 9.3})[0], "inconclusive",
       "prediction resting only on a lying metric is not evidence")
    eq(score("protr_peak >= 2.0", {})[0], "inconclusive", "metric not measured -> inconclusive")
    eq(score("protr_peak >= 2.0", {"protr_peak": None})[0], "inconclusive", "null metric")

    print("\nthe bare-name alias (the archive is full of these)")
    eq(score("act_cv > 0.3", {"act_cv_final": 0.62})[0], "confirmed",
       "a bare quantity resolves to its _final")
    eq(score("act_cv > 0.3", {"act_cv": 0.62})[0], "inconclusive",
       "...and is NOT silently scored against a bare summary key")

    print("\nthe five real predictions from the first LLM proposal (round 1)")
    real = [
        ("protr_peak 2.0-3.5, mech_p_ratio ~3 (forced), analyst phenotype tube/spike", "control"),
        ("protr_peak < 1.5 AND mech_p_ratio drops from ~3 toward ~1", "confirmatory"),
        ("protr_peak >= 2.0 (within ~0.5 of control); REFUTED if it drops below 1.5", "adversarial"),
        ("protr_peak < 1.5 or degenerate/collapsed morphology; REFUTED if protr_peak stays >= 2.0",
         "adversarial"),
        ("protr_peak >= 3.0 (>= control) and/or n_tubes_final increases", "confirmatory"),
        ("mech_p_ratio drops toward ~1 with protr_peak >= 2.0; REFUTED if the body merely "
         "inflates (protr_peak < 1.5, no tube)", "confirmatory"),
    ]
    for i, (p, intent) in enumerate(real):
        cs = parse(p)
        assert cs, f"slot {i} still unparseable: {p!r}"
        print(f"  slot {i} [{intent:12}] {cs}")

    print("\n" + ("predict OK" if not fails else f"{len(fails)} FAILURES:\n  " + "\n  ".join(fails)))
    raise SystemExit(1 if fails else 0)
