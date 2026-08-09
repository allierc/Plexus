#!/usr/bin/env python
"""metrics -- one name, one quantity. A class per measurement, and a registry of the classes.

PHASE 2. WHY THIS FILE EXISTS, AND WHY IT INVENTS ALMOST NOTHING
================================================================================================
The measurements this campaign needs already existed. They were written over three weeks in
`prototype/cardio_mpm`, audited on 4 July, and the audit's own remedy survived. The failure was
never that the quantities were missing -- it is that **the same quantity acquired four names, in
four places, on four different sets of nodes, and nothing recorded which was which:**

    loopscore_residual   size | orientation | openness/aspect | chirality | shape-detail(k>=2)
    enclosure_row        energy peak | area loopiness | chir_match | minor
    morphology_row       size | open | chir+                  (SIM-ONLY -- withdrawn, still printed)
    descriptors.py       magnitude_peak | opening_area | direction_chirality | orientation_rad

The fourth is ours, added in Phase 1, and it was the same mistake one turn later. So each
measurement is now **one class**, in one place, carrying everything that decides whether it may be
believed: what it means in a sentence a person can check, where it came from, which axes it is
allowed to respond to, where it is defined, what a model that knows nothing scores on it, and what
is known to be wrong with it. Nothing about a measurement lives anywhere else.

TWO ARE NEW, AND ONLY BECAUSE NOTHING MEASURED THEM
------------------------------------------------------------------------------------------------
`OrientationError` existed inside the objective, as the phase of a product of Fourier
coefficients, and was never reported -- so that axis could move for a whole campaign with nobody
able to see it. `Coordination` did not exist at all, and its absence is why the objective scores a
sheet whose points beat in random order at exactly 1.0000: whether the tissue contracts *together*
was not a measurable property of any run in sixty batches.

WHAT A TIER MEANS
------------------------------------------------------------------------------------------------
  certified    it has a measured null, a measured noise floor, and it passed the battery below.
               Admissible evidence.
  provisional  it computes and one of those is missing. Usable, never citable.
  withdrawn    a defect was demonstrated. Kept with its cause of death; `compute` REFUSES, and no
               live definition may name it.

    python metrics.py --check      # the registry's own checks
    python metrics.py --certify    # the distortion battery
    python metrics.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CERTIFIED, PROVISIONAL, WITHDRAWN = "certified", "provisional", "withdrawn"

# A TIER SAYS HOW MUCH IS KNOWN ABOUT A MEASUREMENT. A ROLE SAYS WHAT IT IS FOR, AND THEY ARE
# INDEPENDENT.
# The quantity a fit DESCENDS and the quantity a claim CITES are not the same job and there is no
# reason one number should be good at both. An objective has to be smooth, differentiable and
# cheap; an instrument has to be precise and separable. `loopscore` is the objective, it is not
# defective, and it is the only number comparable with the 324 archived runs and the replay bar --
# so it is neither withdrawn nor promotable. It is simply not evidence, and that is enforced here
# rather than remembered.
EVIDENCE, OBJECTIVE = "evidence", "objective"

# WHERE A ZERO CAME FROM, BECAUSE A COPIED DIGIT DRIFTS.
# ANALYTIC nulls are derivable and need no run: circulating the right way round by chance is 0.5,
# two unrelated axes differ by pi/4 on average, timing agreement between unrelated nodes is 0.
# MEASURED nulls are read off the null bank in `floors.py`, and `check()` proves the number in the
# class still matches the artefact -- which is the whole difference between a measurement and a
# number somebody once typed.
ANALYTIC, MEASURED = "analytic", "measured"

# HOW MANY DISTINGUISHABLE STEPS A METRIC MUST OFFER BEFORE IT MAY CARRY A CLAIM.
# Declared as a rule, not fitted to an outcome: ranking runs into quartiles needs four steps
# between "knows nothing" and "as good as the tissue agrees with itself", and this allows one
# spare. Below it a metric can say "nothing" or "tissue-like" and has no opinion in between.
# It is applied to the WORKING UNIT -- the largest of the three noise floors -- never to the
# cheapest one.
MIN_LEVELS = 5.0

# The axes a loop can be wrong along. The first four are the campaign's own vocabulary, from
# `cardio_harmonic.loopscore_residual`; `coordination` is added because nothing covered it.
# The first four are the campaign's own vocabulary; `coordination` was added because nothing
# covered it. The last three were added by the battery, which refused to pass until each metric
# said honestly what it responds to:
#   placement       where the loop SITS -- a shape measure must ignore this, and interior_r2 does not
#   phase           when the whole beat starts -- likewise
#   heterogeneity   whether the error is EVEN across the tissue. loopscore_sd is a spread, so it is
#                   the only thing that responds to this and it responds to nothing else.
AXES = ("size", "openness", "chirality", "orientation", "coordination",
        "placement", "phase", "heterogeneity")


# =============================================================================================
# THE READING SURFACE -- the 10x10 grid of the tissue the campaign actually looks at
# =============================================================================================
GRID_SIDE, GRID_N = 137, 10
MARGIN_INHERITED = 10                  # what the prototype used
MARGIN_SAFE = 20                       # outside the pinned band; see grid_report()
SHEET_SPAN = 0.70                      # the sheet occupies [0.15, 0.85] of the world


def select_grid_nodes(nx=GRID_N, ny=GRID_N, side=GRID_SIDE, margin=MARGIN_INHERITED):
    """The canonical 10x10 dashboard selection, exactly as the prototype defines it."""
    rows = np.linspace(margin, side - 1 - margin, ny).round().astype(int)
    cols = np.linspace(margin, side - 1 - margin, nx).round().astype(int)
    return (rows[:, None] * side + cols[None, :]).ravel()


def grid_report(bwidth=0.06):
    """How many panels of each selection sit inside the pinned band. The defect, as a number.

    The first version omitted the sheet-span factor and reported none, against the July audit's
    thirty-six. The audit was right: the outer ring of a 10x10 grid is 4*10-4 = 36 panels, and at
    margin 10 it sits 0.052 from the edge against a band of 0.06.
    """
    out = {}
    for name, margin in (("inherited", MARGIN_INHERITED), ("corrected", MARGIN_SAFE)):
        idx = select_grid_nodes(margin=margin)
        r, c = idx // GRID_SIDE, idx % GRID_SIDE
        u = np.stack([c, r], 1) / (GRID_SIDE - 1)
        edge = SHEET_SPAN * np.minimum(u, 1 - u).min(1)
        out[name] = {"margin_nodes": margin, "n": int(idx.size),
                     "in_band": int((edge < bwidth).sum()),
                     "min_edge_distance": float(edge.min()), "band_width": bwidth}
    return out


# =============================================================================================
# SHARED GEOMETRY. Every metric that needs one of these uses THIS one, so two of them cannot
# quietly come to mean different things.
# =============================================================================================
def _sel(a, m):
    a = np.asarray(a, float)
    return a[:, np.asarray(m, bool)] if m is not None else a


def _centred(p):
    return p - p.mean(axis=0, keepdims=True)


def signed_area(p):
    """Shoelace area per node. Its sign is the direction of circulation."""
    x, y = p[..., 0], p[..., 1]
    return 0.5 * (x * np.roll(y, -1, 0) - np.roll(x, -1, 0) * y).sum(0)


def bbox_area(p):
    return np.ptp(p[..., 0], axis=0) * np.ptp(p[..., 1], axis=0)


def major_axis_angle(p):
    """Principal-axis angle per node, in [0, pi). An axis has no head or tail."""
    q = _centred(p)
    xx = (q[..., 0] ** 2).mean(0); yy = (q[..., 1] ** 2).mean(0)
    xy = (q[..., 0] * q[..., 1]).mean(0)
    return 0.5 * np.arctan2(2 * xy, xx - yy) % np.pi


def axis_difference(a, b):
    """Smallest angle between two axes, in [0, pi/2]. About 0.785 is what chance gives."""
    d = np.abs(a - b) % np.pi
    return np.minimum(d, np.pi - d)


def motion_signal(p):
    """Per node, how far it is from the centre of its own path, frame by frame. [G, M], real.

    A real scalar, so rotating, mirroring or translating the loop leaves it alone. That is what a
    timing measure needs: it must respond to WHEN things happen and to nothing else.
    """
    return np.linalg.norm(_centred(p), axis=-1)


def timing_lag(sim, real):
    """Per node, the whole-frame shift that best aligns the simulated motion with the recorded one.

    A cross-correlation rather than a Fourier phase, and the reason is worth keeping: the FIRST
    version read the phase of the fundamental of this signal and failed its own shift-invariance
    check, reading 0.50 where it had to read 1.0. The cause is that **the distance-from-centre of
    an ellipse peaks TWICE per beat** -- once at each end of the long axis -- so the beat lives in
    the second harmonic and the fundamental is weak noise whose phase means nothing. Which harmonic
    carries the beat depends on the shape of the path, so indexing one at all is fragile. A lag
    does not care.
    """
    a, b = motion_signal(sim), motion_signal(real)
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    G = a.shape[0]
    # circular cross-correlation per node, via the FFT
    c = np.fft.ifft(np.fft.fft(a, axis=0) * np.conj(np.fft.fft(b, axis=0)), axis=0).real
    return np.argmax(c, axis=0), G


# =============================================================================================
# THE BASE CLASS
# =============================================================================================
class Withdrawn(RuntimeError):
    """Raised when a withdrawn measurement is asked for."""


class NotEvidence(RuntimeError):
    """Raised when a measurement that is not an instrument is asked to support a claim."""


class Undefined(RuntimeError):
    """Raised when a measurement is asked about something outside its domain.

    The alternative is worse than an error: a metric that declares its own domain in prose and then
    returns a number anyway. `coordination` did exactly that -- it said "undefined for a node that
    does not move, so those must be masked out before this is read", which the caller cannot do,
    because the mask selects nodes moving in the RECORDING and says nothing about the model. So a
    model predicting no motion at all scored a perfect 1.0000.
    """


class Metric:
    """One measurement. Everything that decides whether it may be believed lives here."""

    name = "unnamed"
    definition = ""                 # one sentence a person can check
    source = ""                     # where it came from
    tier = PROVISIONAL
    responds_to: set = set()        # the axes it is ALLOWED to move on
    domain = ""                     # where it is defined
    known_defects: list = []
    cause_of_death = ""             # withdrawn only
    certified_because = ""          # CERTIFIED only: the judgement, and the evidence behind it
    undefined_on: set = set()       # distortions outside this metric's declared domain
    axis_separable = True           # False when its response is not attributable to single axes
    null = None                     # what a model that knows nothing scores
    null_source = ""                # ANALYTIC (derivable) or MEASURED (read off the null bank)
    higher_is_better = True
    role = EVIDENCE                 # EVIDENCE = an instrument; OBJECTIVE = a thing to descend
    not_evidence_because = ""       # required when role is OBJECTIVE

    def __call__(self, sim, real, mask=None):
        """Read the number. Reporting and optimising are both allowed; citing is not."""
        if self.tier == WITHDRAWN:
            raise Withdrawn(f"{self.name} may not be quoted: {self.cause_of_death}")
        return float(self.compute(_sel(sim, mask), _sel(real, mask)))

    def cite(self, sim, real, mask=None):
        """Read the number FOR A CLAIM. The gate every conclusion must come through.

        Deliberately narrower than `__call__`: a run record may print anything, but the moment a
        number is offered as evidence it has to be an instrument, and it has to be certified.
        """
        if self.role != EVIDENCE:
            raise NotEvidence(f"{self.name} is the {self.role}, not evidence: "
                              f"{self.not_evidence_because}")
        if self.tier != CERTIFIED:
            raise NotEvidence(f"{self.name} is {self.tier}, so it may be reported but not cited")
        return self(sim, real, mask)

    def compute(self, sim, real):                             # pragma: no cover - interface
        raise NotImplementedError

    def as_dict(self):
        return {"name": self.name, "definition": self.definition, "source": self.source,
                "tier": self.tier, "role": self.role,
                "not_evidence_because": self.not_evidence_because,
                "responds_to": sorted(self.responds_to), "domain": self.domain,
                "known_defects": list(self.known_defects), "cause_of_death": self.cause_of_death,
                "null": self.null, "higher_is_better": self.higher_is_better}

    def __repr__(self):
        return f"<{type(self).__name__} {self.name} [{self.tier}]>"


# =============================================================================================
# THE OBJECTIVE AND ITS DECOMPOSITION -- inherited implementations, wrapped and not rewritten
# =============================================================================================
def _harm():
    import harmonic_inherited as H
    return H


def _t(a):
    import torch
    return torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)


class LoopScore(Metric):
    name = "loopscore"
    definition = ("per node, the beat is a closed path; compare its low-order Fourier description "
                  "with the recording's, normalised by that node's own energy, and average over "
                  "nodes. 1 means the loops match.")
    source = "cardio_harmonic.harmonic_score (26 June, the objective shift)"
    responds_to = {"size", "openness", "chirality", "orientation"}
    domain = "one closed beat, at least 8 frames, on moving interior nodes"
    null = 0.0700
    null_source = MEASURED
    role = OBJECTIVE
    not_evidence_because = (
        "it is the quantity the fit descends, and it is too coarse to be an instrument. Between "
        "its null (+0.070) and what the recording scores against itself (+0.710) there is a range "
        "of 0.640, and its beat-to-beat spread is 0.129 -- about 1.6 distinguishable steps across "
        "the whole usable range, where every other metric with a measured null offers 10 to 130. "
        "It can say `knows nothing' or `tissue-like' and has no opinion in between, which is "
        "precisely the resolution at which the previous campaign ranked 324 runs and settled "
        "orderings on differences of 0.003. It is kept, reported and optimised -- it is the only "
        "number comparable with those runs and with the replay bar -- and it may not carry a claim.")
    known_defects = [
        "BLIND TO COORDINATION: give every node an independent random timing offset and it returns "
        "exactly 1.0000, measured on all 10 corpus runs. That is what Coordination is for.",
        "its zero is +0.070, not 0, and its own documentation says otherwise",
        "the area term is weighted three-to-one, hard-coded in a function signature",
        "the scoring window is 53 frames for a beat whose onsets are 50.5 apart"]

    def compute(self, sim, real):
        return _harm().harmonic_score(_t(sim), _t(real), None)


class LoopScoreSpread(Metric):
    name = "loopscore_sd"
    definition = ("the spread of the per-node loopscore. Uniformly mediocre tissue and a few "
                  "excellent nodes among wrong ones give the same mean; this separates them.")
    source = "cardio_harmonic.harmonic_stats"
    responds_to = {"heterogeneity"}
    axis_separable = False          # see below; the battery only holds it to the heterogeneity case
    domain = "as loopscore"
    higher_is_better = False
    known_defects = [
        "it is a SPREAD, so a distortion applied evenly to the whole tissue moves the mean and "
        "leaves this alone -- doubling every loop does not change it.",
        "IT IS NOT AXIS-SEPARABLE, and the battery is what established that. Its response depends "
        "on the interaction between a distortion and the variety already in the tissue: rotating "
        "every loop by one angle is uniform in the world but lands UNEVENLY on nodes whose own axes "
        "differ, so the spread moves; mirroring every loop lands evenly, so it does not. Contorting "
        "a list of axes to fit that would be a fiction. It answers one question -- is the error even "
        "across the tissue -- and that is the only thing it is held to."]

    def compute(self, sim, real):
        return _harm().harmonic_stats(_t(sim), _t(real), None)[1]


class InteriorR2(Metric):
    name = "interior_r2"
    definition = ("frame-locked goodness of fit, pooled over nodes, with the temporal mean removed "
                  "from the recording only.")
    source = "cardio_harmonic.interior_r2 (23 June, the original objective)"
    responds_to = {"size", "openness", "chirality", "orientation", "coordination",
                   "placement", "phase"}
    domain = "any window; blind to whether the path is a loop at all"
    null = -0.8308
    null_source = MEASURED
    known_defects = [
        "every one of the 324 archived fits scores below its own null",
        "NOT A SHAPE MEASURE, and the battery pinned it down: it moves when the loop is merely "
        "TRANSLATED and when the whole beat is merely SHIFTED IN TIME. Both are correct behaviour "
        "for a frame-locked residual and both disqualify it from answering a question about "
        "morphology. That is why it is a diagnostic."]

    def compute(self, sim, real):
        return _harm().interior_r2(_t(sim), _t(real), None)


class ResidualDimension(Metric):
    """One of the five named dimensions: how much loopscore is recovered by fixing that axis.

    The five are computed together, so each runs the whole decomposition and takes its own entry.
    Kept as separate classes because each is separately citable and separately defined.
    """
    key = ""
    source = "cardio_harmonic.loopscore_residual"
    domain = ("as loopscore, and UNDEFINED on a degenerate path: the decomposition works by "
              "correcting one property of an ellipse at a time, and a loop flattened onto a line "
              "has no aspect, no handedness and no meaningful higher harmonics to correct. The "
              "battery skips it there rather than recording a failure. ALSO undefined for a model "
              "with no motion, and that one is enforced below.")
    undefined_on = {"collapse to a line", "a model with no motion to correct"}
    known_defects = [
        "THESE MEASURE RECOVERABILITY, NOT ERROR. Each says how much loopscore is regained by "
        "correcting one property, which is not the same as how wrong that property is. The "
        "consequence was measurable: on the do-nothing model residual/shape_detail read +0.3031, "
        "the highest score in the whole null bank, because when a model does nothing every axis is "
        "maximally recoverable. Enforced away below rather than left as a caveat."]

    #: as Coordination, and for the same reason: a model with no motion must not be scored well for
    #: having nothing to correct
    DEAD_FRACTION = 0.01
    MIN_ALIVE = 0.25

    def compute(self, sim, real):
        a_sim = np.ptp(motion_signal(sim), axis=0)
        a_real = np.ptp(motion_signal(real), axis=0)
        ref = np.median(a_real[a_real > 0]) if np.any(a_real > 0) else 0.0
        alive = a_sim > self.DEAD_FRACTION * ref
        if ref <= 0 or alive.mean() < self.MIN_ALIVE:
            raise Undefined(
                f"{self.name}: {100 * (1 - alive.mean()):.0f}% of nodes have no simulated motion. "
                f"The decomposition asks how much is recovered by correcting one property, and "
                f"everything is recoverable from nothing -- it must not be scored for that.")
        _, d = _harm().loopscore_residual(_t(sim), _t(real), None)
        return d[self.key]


class ResidualSize(ResidualDimension):
    name, key, responds_to = "residual/size", "size", {"size"}
    definition = ("how much loopscore is recovered by scaling the simulated loops to the "
                  "recording's size, leaving everything else alone.")


class ResidualOrientation(ResidualDimension):
    name, key, responds_to = "residual/orientation", "orientation", {"orientation"}
    definition = "how much is recovered by rotating the simulated loops onto the recording's axis."
    undefined_on = ResidualDimension.undefined_on | {"mirror"}
    known_defects = [
        "the residual family measures RECOVERABILITY, not error, and mirroring is where the two "
        "part company: a reflected loop genuinely has the wrong axis -- orientation_error sees it -- "
        "but no ROTATION undoes a reflection, so nothing is recoverable and this reads zero. The "
        "battery reported that as a failure until the distinction was written down."]


class ResidualOpenness(ResidualDimension):
    name, key, responds_to = "residual/openness_aspect", "openness/aspect", {"openness"}
    definition = ("how much is recovered by matching the aspect ratio -- how fat the loop is "
                  "against how long -- while keeping its size.")


class ResidualChirality(ResidualDimension):
    name, key, responds_to = "residual/chirality", "chirality", {"chirality"}
    definition = ("how much is recovered by making each loop circulate the same way round as the "
                  "recording's.")


class ResidualShapeDetail(ResidualDimension):
    name, key = "residual/shape_detail", "shape-detail(k>=2)"
    # relative harmonic content, so scaling every loop changes nothing -- the battery showed that
    # listing `size` here was wrong
    responds_to = {"openness"}
    definition = ("how much is recovered by taking the higher harmonics from the recording -- "
                  "everything the ellipse does not describe.")


# =============================================================================================
# PER-LOOP QUANTITIES -- the July audit's remedy, and the two that were missing
# =============================================================================================
class PairedProperty(Metric):
    """A property of ONE set of loops, made into a test by reading it on BOTH and subtracting.

    WHY THESE THREE WERE NOT MEASUREMENTS OF ANYTHING.
    `openness`, `path_length` and `peak_excursion` all took two arguments and used only the first.
    They described the model -- how fat its loops are, how far a node travels, how far it reaches --
    and never compared it with the recording. A description has no right answer, so it has no score
    for knowing nothing, so its precision divides into nothing: all three were the most precise
    numbers in the registry and none of them could support a claim.

    Subtracting the two readings turns each into a distance, which does have a right answer (zero)
    and does have a value for a model that knows nothing. `reading()` keeps the raw property
    available, because a run record still wants to say WHICH SIDE the model is on -- too fat or too
    thin -- and the distance alone cannot.
    """
    higher_is_better = False        # it is now an error, so less of it is better

    def property(self, p):                                    # pragma: no cover - interface
        raise NotImplementedError

    def compute(self, sim, real):
        return abs(self.property(sim) - self.property(real))

    def scale(self, real):
        """The natural size of this quantity, for judging whether a change is a change.

        A paired metric reads exactly 0 when nothing is wrong, so a change "relative to its own
        value" is a division by zero and the battery was substituting 1.0 -- an arbitrary number
        that happens to be a thousand times the real path lengths and a third of the openness.
        """
        return abs(self.property(real))

    def reading(self, p, mask=None):
        """The raw property on one side, signed and uncompared. Never a score."""
        return float(self.property(_sel(p, mask)))


class Openness(PairedProperty):
    name = "openness"
    definition = ("how far the model's loops are from the recording's in fatness -- enclosed area "
                  "over the area of the box the loop fits in, read on both and subtracted. 0 is "
                  "perfect. A straight line reads 0 fatness, a circle about 0.79.")
    source = "cardio_mpm_train._openness (audit remedy, 4-5 July), paired here"
    responds_to = {"openness"}
    domain = "a closed path with non-zero extent"
    known_defects = [
        "IT DOES NOT READ HOW FAT A LOOP IS. On ellipses it returns pi/4 for EVERY aspect ratio -- "
        "a circle and a needle read the same to five decimals. What it separates is a loop from a "
        "degenerate back-and-forth line, and nothing finer. Aspect is residual/openness_aspect's "
        "job, not this one's.",
        "IT RESPONDS TO ORIENTATION, WHICH IT DOES NOT DECLARE. The normaliser is an AXIS-ALIGNED "
        "bounding box, which grows when the loop turns off-axis, so the same ellipse reads 0.785 "
        "aligned and 0.588 at 45 degrees -- a 25% swing from turning alone. The battery cannot see "
        "it: it turns every loop by one angle, and a population already pointing every which way "
        "is a symmetry of that. Found by calibrate.py on shapes with a closed form. Normalising by "
        "the loop's OWN axes, or by 4*pi*area/perimeter^2, removes it -- both change every number "
        "measured so far, so it is a decision and not a patch. UNTIL THEN: a claim from this metric "
        "must be accompanied by orientation_error, which is separately measured and does read the "
        "axis."]

    null = 0.3389                  # floors.py N0: what predicting nothing costs
    null_source = MEASURED

    def property(self, p):
        return np.median(np.abs(signed_area(p)) / (bbox_area(p) + 1e-12))


class PathLength(PairedProperty):
    name = "path_length"
    tier = CERTIFIED
    certified_because = (
        "CERTIFIED 2026-08-09. 6.5 steps, the narrowest margin of the four, so a claim resting on it alone should say so. Calibration: reproduces the closed-form perimeter and is invariant to turning and to rolling once the closing segment was restored. Declared defect: it is a COMPOSITE of size and openness and may not be quoted as a size measure on its own.")
    definition = ("how far the model's nodes travel over the beat against how far the recording's "
                  "do, summed frame to frame and subtracted. 0 is perfect. A loop and a line of "
                  "the same width differ here; the enclosed area alone does not say so.")
    source = "this file -- the one quantity the prototype named and never computed"
    responds_to = {"size", "openness"}
    domain = "any window"
    known_defects = [
        "NOT AN INDEPENDENT AXIS, and the battery is what showed it: flattening an ellipse onto its "
        "long axis changes the distance travelled (the perimeter goes from about pi*(a+b) to 4a), "
        "so this mixes how big the loop is with how open it is. It may not be quoted as a size "
        "measure on its own."]

    null = 0.0042                  # floors.py N0: what predicting nothing costs
    null_source = MEASURED

    def property(self, p):
        # the closing segment counts. np.diff over G frames returns G-1 of them and drops the one
        # from the last frame back to the first, so rolling a beat changed WHICH segment was missing
        # and the total shifted by about 1% -- enough for the battery to report that path length
        # responds to timing, which it must not. The beat is a closed path; sum all G segments.
        d = np.diff(np.concatenate([p, p[:1]], axis=0), axis=0)
        return np.median(np.linalg.norm(d, axis=-1).sum(0))


class PeakExcursion(PairedProperty):
    name = "peak_excursion"
    tier = CERTIFIED
    certified_because = (
        "CERTIFIED 2026-08-09. 8.5 steps. Paired (it compares, it does not merely describe) with a measured null from the do-nothing model. Calibration: returns the long semi-axis exactly and is invariant to turning. It is the AMPLITUDE channel and must be reported beside the amplitude-blind instruments, never in place of them.")
    definition = ("how far the model's nodes reach from the centre of their own path against how "
                  "far the recording's do, at furthest, subtracted. 0 is perfect. Centred, so "
                  "sliding a loop elsewhere does not change it.")
    source = "cardio_mpm_train.enclosure_row (peak), centred here"
    responds_to = {"size"}
    domain = "any window"
    known_defects = [
        "the first version measured distance from the ORIGIN, so merely translating a loop changed "
        "it -- the same uncentred defect the July audit withdrew morphology_row/size for. Caught by "
        "the battery, which is what the battery is for."]

    null = 0.0011                  # floors.py N0: what predicting nothing costs
    null_source = MEASURED

    def property(self, p):
        return np.median(np.linalg.norm(_centred(p), axis=-1).max(0))


class ChiralityMatch(Metric):
    name = "chirality_match"
    definition = "the fraction of nodes circulating the same way round as the recording's."
    source = "cardio_mpm_train.enclosure_row (chir_match)"
    responds_to = {"chirality"}
    domain = ("a closed path with non-zero area. A flattened loop has no handedness, so this is "
              "meaningless there and is expected to move when a loop is collapsed onto a line.")
    null = 0.5                     # a coin toss
    null_source = ANALYTIC

    def compute(self, sim, real):
        return (np.sign(signed_area(sim)) == np.sign(signed_area(real))).mean()


class OrientationError(Metric):
    """NEW -- and only because nothing reported it."""

    name = "orientation_error"
    tier = CERTIFIED
    certified_because = (
        "CERTIFIED 2026-08-09. 10.1 distinguishable steps between its measured null (pi/4, analytic: two unrelated axes) and the tissue's own beat-to-beat agreement (0.0582), against the declared threshold of 5. Battery: moves on orientation and chirality, holds on the other seven. Calibration: returns the planted rotation to three decimals from 0 to pi/2. Floors: beat 0.0232, same-seed and seed-to-seed both measured. Declared defect: two circles have no axis and it returns 1.4360 rad instead of refusing -- latent, since real loops are elongated.")
    definition = ("the angle between the simulated and the recorded long axis of each loop, median "
                  "over nodes, in radians. 0 is perfect; about 0.785 is what chance gives, because "
                  "an axis has no head or tail.")
    source = ("this file. The quantity lived inside the objective as the phase of a product of "
              "Fourier coefficients and was NEVER REPORTED, so this axis could move for a whole "
              "campaign with nobody able to see it.")
    responds_to = {"orientation"}
    domain = "a path with a distinguishable long axis; degenerate for a circle"
    null = float(np.pi / 4)
    null_source = ANALYTIC
    undefined_on = {"a loop with no long axis"}
    higher_is_better = False
    known_defects = [
        "EXACT WHERE IT IS DEFINED, SILENT WHERE IT IS NOT. calibrate.py sweeps a known rotation "
        "and it returns that angle to three decimals from 0 to pi/2, which is its maximum -- two "
        "axes can differ by at most 90 degrees, so perpendicular reads pi/2 and there is no wrap. "
        "But a CIRCLE has no long axis, and asked to compare two circles it returned 1.4360 rad "
        "instead of refusing: the covariance is isotropic and the eigenvector it picks is whichever "
        "way the arithmetic fell. Real loops are elongated, so this is latent rather than active, "
        "and it is now declared. If a claim ever rests on near-circular loops it must be checked."]

    def compute(self, sim, real):
        return np.median(axis_difference(major_axis_angle(sim), major_axis_angle(real)))

    def cross_check(self, sim, real, mask=None):
        """The same angle read a second way, from the Fourier phase rather than the covariance.

        Two readings of one quantity: if they disagree, one of them is broken, and that is worth
        knowing before either is trusted.
        """
        s, r = _sel(sim, mask), _sel(real, mask)

        def ang(p):
            z = p[..., 0] + 1j * p[..., 1]
            G = z.shape[0]
            Z = np.fft.fft(z, axis=0) / G
            return 0.5 * np.angle(Z[1] * Z[G - 1]) % np.pi

        cov = float(np.median(axis_difference(major_axis_angle(s), major_axis_angle(r))))
        fou = float(np.median(axis_difference(ang(s), ang(r))))
        return {"covariance": cov, "fourier": fou, "agree_to": abs(cov - fou)}


class Coordination(Metric):
    """NEW -- and it is the hole the whole registry was built around."""

    name = "coordination"
    tier = CERTIFIED
    certified_because = (
        "CERTIFIED 2026-08-09. 8.0 steps between its measured null (0.0778, the scrambled-timing row) and 0.9968. It is the only instrument that sees WHEN the tissue moves, which the objective scores at exactly 1.0000 for a sheet beating in random order. Battery clean; calibration confirms the declared antiphase blindness on a shape where the answer is known. Amplitude-blind: reads 1.0 at 1% amplitude, which is what makes it usable without a gauge.")
    definition = ("does the tissue contract TOGETHER the way the recording does? Per node, find the "
                  "time shift that best lines its motion up with the recording's; then measure how "
                  "tightly those shifts agree with each other. 1 means the pattern of "
                  "who-moves-when matches, up to one global offset; 0 means it is unrelated.")
    source = ("this file. Nothing measured it, which is why the objective scores a sheet whose "
              "points beat in random order at exactly 1.0000 -- whether the tissue contracts "
              "together was not a measurable property of any run in sixty batches.")
    responds_to = {"coordination"}
    domain = ("a periodic beat. Undefined for a node that does not move, so those must be masked "
              "out before this is read. Determined only up to HALF a beat -- see the defect below.")
    null = 0.0778
    null_source = MEASURED          # the scrambled-timing row of the battery, not a derivation
    undefined_on = {"a model with no motion to time"}
    known_defects = [
        "FIXED, and kept because the number it gave was a perfect one: a model predicting NO MOTION "
        "AT ALL scored 1.0000. A dead signal has no peak, so the cross-correlation is flat, argmax "
        "returns 0 for every node, and the lags agree perfectly because they are all the same "
        "nothing. Found by scoring the null bank through the registry -- the distortion battery "
        "could not have found it, because every distortion starts from a real loop. compute() now "
        "enforces its own domain and raises Undefined.",
        "its null is NOT 0. Predicting nothing scores 1.0 (above) and the analytic 0 that was "
        "declared here was simply wrong; the measured baseline for knowing nothing about timing is "
        "the scrambled-timing row, 0.0778.",
        "CANNOT TELL IN-PHASE FROM EXACTLY ANTIPHASE. The signal it uses is a distance from a "
        "centre, which peaks twice on a path traversed once per beat, so the alignment is only "
        "determined modulo half a beat. Two regions contracting in exact opposition therefore read "
        "as coordinated. Anything less than exact opposition is seen.",
        "reads a whole number of frames, so on a 24-frame beat it cannot resolve better than about "
        "15 degrees of phase"]

    #: a node whose simulated motion is below this fraction of the recording's typical motion is
    #: dead rather than quiet, and carries no timing information. Deliberately strict, so that a
    #: model which is merely too small keeps every node and this stays a measure of TIMING alone.
    DEAD_FRACTION = 0.01
    MIN_ALIVE = 0.25                # below this share of nodes the question has no answer

    def compute(self, sim, real):
        # ENFORCE THE DOMAIN, rather than describing it and trusting the caller.
        a_sim = np.ptp(motion_signal(sim), axis=0)
        a_real = np.ptp(motion_signal(real), axis=0)
        ref = np.median(a_real[a_real > 0]) if np.any(a_real > 0) else 0.0
        alive = a_sim > self.DEAD_FRACTION * ref
        if ref <= 0 or alive.mean() < self.MIN_ALIVE:
            raise Undefined(
                f"{self.name}: {100 * (1 - alive.mean()):.0f}% of nodes have no simulated motion to "
                f"time (threshold {self.DEAD_FRACTION:g} of the recording's typical excursion). "
                f"Coordination is a question about WHEN things move and has no answer for something "
                f"that does not move -- it must not be scored 1.0 for holding still.")
        sim, real = sim[:, alive], real[:, alive]

        lag, G = timing_lag(sim, real)
        # The lag is only determined MODULO HALF A BEAT, so it is mapped onto the half period before
        # the lags are combined. Why: the signal is a distance from a centre, and on a path traversed
        # once per beat that distance peaks TWICE -- once at each end of the long axis -- so the
        # cross-correlation has two equally good maxima half a beat apart and `argmax` chooses
        # between them arbitrarily. Combined naively those two answers CANCEL, which is exactly what
        # happened: identical inputs read 0.96 and a global shift read 0.32. Mapping onto the half
        # period makes the two indistinguishable, which is honest, because on this signal they are.
        #
        # The cost is declared in `known_defects`: this cannot tell in-phase from exactly antiphase.
        return np.abs(np.exp(4j * np.pi * lag / G).mean())


# =============================================================================================
# WITHDRAWN. Kept with the cause of death; asking for one raises.
# =============================================================================================
class _WithdrawnMetric(Metric):
    tier = WITHDRAWN
    domain = "none"

    def compute(self, sim, real):
        raise Withdrawn(f"{self.name}: {self.cause_of_death}")


class MorphologyRowSize(_WithdrawnMetric):
    name = "morphology_row/size"
    definition = ("the mean over the 10x10 grid of each node's largest displacement, computed on "
                  "the simulation alone and uncentred.")
    source = "cardio_mpm_train.morphology_row (~24 June)"
    cause_of_death = ("simulation-only, so it cannot see the residual at all; and 36 of its 100 "
                      "panels are pinned to the recording, so it largely reported the anchor. The "
                      "campaign read it as 'size is flat against every lever, therefore a "
                      "structural limit' and chased that for four rounds. (Audit, 4 July.)")


class MorphologyRowOpen(_WithdrawnMetric):
    name = "morphology_row/open"
    definition = "openness over the 10x10 grid, on the simulation alone."
    source = "cardio_mpm_train.morphology_row"
    cause_of_death = "the same defect as morphology_row/size"


class MorphologyRowChir(_WithdrawnMetric):
    name = "morphology_row/chir"
    definition = "the fraction of grid panels turning anticlockwise, on the simulation alone."
    source = "cardio_mpm_train.morphology_row"
    cause_of_death = "the same defect as morphology_row/size"


class AmpL(_WithdrawnMetric):
    name = "ampL"
    definition = ("the ratio of total motion energy, simulated against recorded, over the whole "
                  "field.")
    source = "cardio_mpm_train (23 June, Batch 1)"
    cause_of_death = ("a GLOBAL ratio dominated by a few large-motion nodes. The best run on record "
                      "read 0.002 -- 'the cleanest ever' -- while the median node was at 0.57 of the "
                      "recorded amplitude. It was read as a shape score and is not one. "
                      "(Audit, 4 July.)")


# =============================================================================================
# THE REGISTRY
# =============================================================================================
_ALL = [LoopScore, LoopScoreSpread, InteriorR2,
        ResidualSize, ResidualOrientation, ResidualOpenness, ResidualChirality,
        ResidualShapeDetail,
        Openness, PathLength, PeakExcursion, ChiralityMatch, OrientationError, Coordination,
        MorphologyRowSize, MorphologyRowOpen, MorphologyRowChir, AmpL]

REGISTRY = {c.name: c() for c in _ALL}


def admitted():
    """The only names a claim may cite. Empty until Phase 2 certifies something -- deliberately."""
    return {k: v for k, v in REGISTRY.items() if v.tier == CERTIFIED}


def withdrawn():
    return {k: v for k, v in REGISTRY.items() if v.tier == WITHDRAWN}


def live():
    return {k: v for k, v in REGISTRY.items() if v.tier != WITHDRAWN}


# Reconciliation with descriptors.py, which measures several of the same things under other names.
# The names here are canonical -- they are the campaign's, and they came first.
ALIASES = {"opening_loopiness": "openness", "magnitude_peak": "peak_excursion",
           "direction_chirality": "chirality_match"}


# =============================================================================================
# THE CERTIFICATION BATTERY. Distort the recording one axis at a time; every metric must move on
# the axes it declares and hold still on the rest. Both halves are required -- one that always
# moves measures nothing in particular, one that never moves measures nothing at all.
# =============================================================================================
def population(G=24, M=200, seed=0):
    """A population of ellipses standing in for the recording: random size, aspect, angle, phase."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, G, endpoint=False)
    a = rng.uniform(0.5, 1.5, M)
    b = a * rng.uniform(0.25, 0.9, M)
    th = rng.uniform(0, np.pi, M)
    ph = rng.integers(0, G, M)
    u, v = np.cos(t)[:, None] * a, np.sin(t)[:, None] * b
    p = np.stack([u * np.cos(th) - v * np.sin(th), u * np.sin(th) + v * np.cos(th)], -1)
    return np.stack([np.roll(p[:, j], int(ph[j]), 0) for j in range(M)], 1), th


def distortions(real, th):
    """(name, distorted, the axes it is supposed to touch)."""
    G, M = real.shape[0], real.shape[1]
    rng = np.random.default_rng(1)
    out = [("identity", real.copy(), set()),
           ("scale x2", real * 2.0, {"size"})]
    rot = np.pi / 5
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    out.append(("rotate by pi/5", real @ R.T, {"orientation"}))
    mir = real.copy(); mir[..., 1] *= -1
    out.append(("mirror", mir, {"chirality", "orientation"}))
    ca, sa = np.cos(th), np.sin(th)
    proj = real[..., 0] * ca[None] + real[..., 1] * sa[None]
    # Collapsing onto the long axis destroys the enclosed area and with it any handedness -- a
    # straight line has none. It does NOT shorten the reach along the surviving axis, so size
    # holds. Our first expectations had both of those backwards, and the battery blamed the
    # metrics for it.
    out.append(("collapse to a line", np.stack([proj * ca[None], proj * sa[None]], -1),
                {"openness", "chirality"}))
    out.append(("shift the whole beat",
                np.stack([np.roll(real[:, j], 7, 0) for j in range(M)], 1), {"phase"}))
    out.append(("translate", real + np.array([3.0, -2.0]), {"placement"}))
    # distort HALF the tissue and leave the rest alone: the only case that makes the error uneven,
    # and therefore the only thing loopscore_sd should respond to
    half = real.copy(); half[:, :M // 2] *= 2.0
    out.append(("double half the tissue", half, {"size", "heterogeneity"}))
    out.append(("scramble each node's timing",
                np.stack([np.roll(real[:, j], int(rng.integers(0, G)), 0) for j in range(M)], 1),
                {"coordination"}))
    return out


def certify(verbose=True, tol=0.02):
    real, th = population()
    names = list(live())
    rows = []
    for dname, sim, should in distortions(real, th):
        rec = {"distortion": dname, "should_move": sorted(should), "metrics": {}}
        for n in names:
            m = REGISTRY[n]
            if not getattr(m, "axis_separable", True) and "heterogeneity" not in should:
                rec["metrics"][n] = {"skipped": True, "why": "not axis-separable; held only to the "
                                                            "heterogeneity case"}
                continue
            if dname in getattr(m, "undefined_on", set()):
                rec["metrics"][n] = {"skipped": True, "why": "outside its declared domain"}
                continue
            try:
                base, got = m(real, real), m(sim, real)
            except Exception as e:
                rec["metrics"][n] = {"error": f"{type(e).__name__}: {e}"}
                continue
            own = getattr(m, "scale", None)
            cand = max(abs(base), abs(own(real)) if callable(own) else 0.0)
            scale = cand if cand > 1e-9 else 1.0        # 1.0 only when nothing better is declared
            rel = abs(got - base) / scale
            moved = rel > tol
            expect = bool(m.responds_to & should)
            rec["metrics"][n] = {"base": base, "value": got, "rel_change": rel,
                                 "moved": bool(moved), "expected_to_move": expect,
                                 "ok": bool(moved) == expect}
        rows.append(rec)
    bad = [(r["distortion"], n) for r in rows for n, a in r["metrics"].items()
           if "error" not in a and not a.get("skipped") and not a["ok"]]
    if verbose:
        show_names = [n for n in names if not n.startswith("residual/")]
        print(f"\n{'=' * 118}\n  CERTIFICATION -- each metric must move on the axes it declares and "
              f"hold still on the rest\n{'=' * 118}")
        print(f"  {'distortion':<28s} " + " ".join(f"{n[:12]:>13s}" for n in show_names))
        for r in rows:
            cells = []
            for n in show_names:
                a = r["metrics"][n]
                if "error" in a:
                    cells.append(f"{'ERR':>13s}")
                elif a.get("skipped"):
                    cells.append(f"{'n/a':>13s}")
                else:
                    cells.append(f"{('MOVE' if a['moved'] else 'hold') + ('' if a['ok'] else ' X'):>13s}")
            print(f"  {r['distortion']:<28s} " + " ".join(cells))
        print(f"\n  the five residual/* dimensions are checked too, and not shown for width.")
        print(f"  X marks a metric that moved when it should not, or held when it should have.")
        print(f"  {len(bad)} disagreements" + (f": {bad[:8]}" if bad else ""))
        print("=" * 118)
    return rows, bad


# =============================================================================================
# THE REGISTRY'S OWN CHECKS
# =============================================================================================
def check(verbose=True):
    rows = []

    def add(name, ok, detail=""):
        rows.append({"check": name, "pass": bool(ok), "detail": detail})
        return ok

    real, th = population()

    broken = []
    for n, m in live().items():
        try:
            m(real, real)
        except Exception as e:
            broken.append(f"{n}: {type(e).__name__}")
    add("every live metric computes", not broken,
        f"{len(live())} live, {len(withdrawn())} withdrawn" if not broken else str(broken))

    add("every metric has a written definition",
        all(m.definition.strip() for m in REGISTRY.values()), "all present")

    wd = set(withdrawn())
    bad = [n for n, m in live().items()
           for w in wd if w.split("/")[-1] in m.definition.split()]
    add("no live definition names a withdrawn metric", not bad, "clean" if not bad else str(bad))

    add("every withdrawn metric records why",
        all(m.cause_of_death for m in withdrawn().values()),
        f"{len(wd)} withdrawn, all with a cause")

    # A CERTIFICATION IS A JUDGEMENT AND MUST BE SIGNED, for the same reason a withdrawal is.
    # The failure this catches is not a metric that is wrongly certified; it is the four that sat
    # ELIGIBLE and uncertified for a day while every consumer, finding nothing citable, fell back
    # to the objective and then built a gauge to make the objective behave. A gate that refuses
    # everything does not prevent bad claims -- it redirects them to the ungated thing.
    cert = admitted()
    add("every certified metric records why",
        all(m.certified_because.strip() for m in cert.values()),
        f"{len(cert)} certified" + (f": {', '.join(cert)}" if cert else " -- NOTHING IS CITABLE"))

    declared = {n: m for n, m in live().items() if m.null is not None}
    add("every declared null says where it came from",
        all(m.null_source in (ANALYTIC, MEASURED) for m in declared.values()),
        f"{sum(m.null_source == MEASURED for m in declared.values())} measured, "
        f"{sum(m.null_source == ANALYTIC for m in declared.values())} analytic, "
        f"{len(live()) - len(declared)} with no null at all")

    # a copied digit drifts; this is the only thing that stops it
    fl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_metrology", "floors.json")
    bank = (json.load(open(fl)).get("models", {}).get("N0_zero", {})
            .get("windows", {}).get("fit", {}).get("registry", {}) if os.path.exists(fl) else {})
    drift = {n: (m.null, bank[n]) for n, m in declared.items()
             if m.null_source == MEASURED and isinstance(bank.get(n), (int, float))
             and abs(m.null - bank[n]) > 0.02}
    add("every MEASURED null still matches the null bank", not drift,
        (f"checked {sum(1 for n, m in declared.items() if m.null_source == MEASURED and n in bank)}"
         f" against floors.json N0" if not drift else
         f"drifted: {drift}") if bank else
        "floors.json has no registry row yet -- rerun floors.py --nulls")

    # THE GATE THAT CAUGHT COORDINATION, generalised.
    # The distortion battery starts from a real loop every time, so it can only ever ask "does this
    # respond correctly to a change?" It cannot ask "can this be fooled by a model that does
    # nothing?" -- and a metric whose best score in the whole null bank belongs to the do-nothing
    # model is not measuring the thing it claims. Scored on the bank rather than argued.
    bank_all = (json.load(open(fl)).get("models", {}) if os.path.exists(fl) else {})
    def _reg(k):
        return (bank_all.get(k, {}).get("windows", {}).get("fit", {}) or {}).get("registry", {})
    n0 = _reg("N0_zero")
    flattered, tested = [], 0
    for n, m in live().items():
        vals = {k: _reg(k).get(n) for k in bank_all}
        vals = {k: v for k, v in vals.items() if isinstance(v, (int, float))}
        if n not in n0 or not isinstance(n0.get(n), (int, float)) or len(vals) < 3:
            continue
        tested += 1
        best = max(vals.values()) if m.higher_is_better else min(vals.values())
        # N4 is the model with its muscle switched off and is the same nothing as N0; a tie with it
        # is not flattery
        if abs(n0[n] - best) < 1e-9 and not all(
                abs(v - best) < 1e-9 for k, v in vals.items() if k in ("N0_zero", "N4_passive")
        ) is None:
            ties = [k for k, v in vals.items() if abs(v - best) < 1e-9]
            if set(ties) <= {"N0_zero", "N4_passive"}:
                flattered.append(f"{n} ({n0[n]:+.4f}, the best of {len(vals)} trivial models)")
    add("no metric scores its best by predicting NOTHING", not flattered,
        f"{tested} scored on the null bank" if not flattered else "; ".join(flattered))

    objectives = {n: m for n, m in REGISTRY.items() if m.role == OBJECTIVE}
    add("every non-instrument records why it is not one",
        all(m.not_evidence_because.strip() for m in objectives.values()),
        f"{len(objectives)} objective ({', '.join(objectives) or 'none'}), "
        f"{len(live()) - len(objectives)} instruments")

    refused = []
    for n, m in objectives.items():
        try:
            m.cite(real, real)
        except NotEvidence:
            refused.append(n)
    add("an objective REFUSES to be cited", len(refused) == len(objectives),
        f"{refused} raise NotEvidence when asked to support a claim"
        if refused else "nothing refused -- the gate is not wired up")

    computed = []
    for n, m in withdrawn().items():
        try:
            m(real, real); computed.append(n)
        except Withdrawn:
            pass
    add("a withdrawn metric REFUSES to be computed", not computed,
        f"all {len(wd)} refuse" if not computed else f"computed anyway: {computed}")

    add("every live metric declares which axes it responds to",
        all(m.responds_to and m.responds_to <= set(AXES) for m in live().values()),
        f"axes: {', '.join(AXES)}")

    # This check used to read `not admitted()` -- it asserted the empty set, and passed for as long
    # as NOTHING was certified. It was a placeholder for Phase 2 that outlived Phase 2, and it is
    # the reason the failure went unseen: the note said seven instruments were closed while the
    # suite was actively confirming that none of them could be cited. Replaced by the real
    # invariant, which is that a certification must be AUDITABLE against the same evidence that
    # was supposed to justify it.
    bad = []
    for n, m in admitted().items():
        if m.role != EVIDENCE:
            bad.append(f"{n}: certified but role={m.role}")
        if m.null is None:
            bad.append(f"{n}: certified with no declared null, so its range is uninterpretable")
    try:                                       # floors live in json; absent on a fresh checkout
        import noise as _nz
        for r in _nz.resolving_power(verbose=False):
            if r["metric"] in admitted():
                if r["levels"] is None:
                    bad.append(f"{r['metric']}: certified with no measured floor")
                elif r["levels"] < MIN_LEVELS:
                    bad.append(f"{r['metric']}: certified at {r['levels']:.1f} steps, "
                               f"{MIN_LEVELS:g} required")
    except Exception as e:
        print(f"    (floors not audited: {type(e).__name__})")
    add("every certification is auditable against its own evidence", not bad,
        f"{len(admitted())} certified, {len(live()) - len(admitted())} provisional, "
        f"{len(wd)} withdrawn" + ("" if not bad else " -- " + "; ".join(bad)))

    # the two new ones must answer the question they were added for
    G, M = real.shape[0], real.shape[1]
    rng = np.random.default_rng(2)
    sc = np.stack([np.roll(real[:, j], int(rng.integers(0, G)), 0) for j in range(M)], 1)
    co, ls = REGISTRY["coordination"], REGISTRY["loopscore"]
    c_id, c_sc, l_sc = co(real, real), co(sc, real), ls(sc, real)
    add("Coordination sees what the objective cannot", c_id > 0.99 and c_sc < 0.3,
        f"identical {c_id:.4f} -> scrambled {c_sc:.4f}, where loopscore reads {l_sc:.4f}")

    shifted = np.stack([np.roll(real[:, j], 5, 0) for j in range(M)], 1)
    add("Coordination ignores a shift of the whole beat", abs(co(shifted, real) - 1.0) < 0.01,
        f"{co(shifted, real):.4f} -- one global offset is not a coordination failure")

    oe = REGISTRY["orientation_error"]
    rot = np.pi / 5
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    cc = oe.cross_check(real @ R.T, real)
    add("OrientationError agrees with itself read two ways", cc["agree_to"] < 0.02,
        f"covariance {cc['covariance']:.4f} vs Fourier {cc['fourier']:.4f} "
        f"(the rotation applied was {rot:.4f})")

    g = grid_report()
    add("the 10x10 reading surface is outside the pinned band", g["corrected"]["in_band"] == 0,
        f"inherited margin {g['inherited']['margin_nodes']}: {g['inherited']['in_band']}/100 INSIDE "
        f"the band; corrected margin {g['corrected']['margin_nodes']}: {g['corrected']['in_band']}/100")

    meas = os.path.join(HERE, "_metrology", "archive_metric_test.json")
    if os.path.exists(meas):
        m = json.load(open(meas))["dashboard_grid"]
        add("the grid defect is measured on a real fit, not just counted",
            m["overlap"] == m["pinned_to_the_recording"] == m["scoring_1.000"],
            f"all {m['overlap']} pinned panels score 1.000; the picture reads "
            f"{m['mean_all_panels']:+.3f} against {m['mean_unpinned']:+.3f} unpinned")

    if verbose:
        print(f"\n{'=' * 108}\n  METRICS -- one name, one quantity\n{'=' * 108}")
        for r in rows:
            print(f"  [{'  ok  ' if r['pass'] else ' FAIL '}] {r['check']:<52s} {r['detail']}")
        ok = all(r["pass"] for r in rows)
        print(f"\n  REGISTRY: {'PASS' if ok else 'FAIL'} ({sum(r['pass'] for r in rows)}/{len(rows)})")
        print("=" * 108)
    return all(r["pass"] for r in rows), rows


def show():
    print(f"\n{'=' * 112}\n  THE REGISTRY -- {len(REGISTRY)} measurements\n{'=' * 112}")
    for tier in (CERTIFIED, PROVISIONAL, WITHDRAWN):
        ms = [m for m in REGISTRY.values() if m.tier == tier]
        print(f"\n  {tier.upper()}" + ("" if ms else ": none"))
        for m in ms:
            print(f"    {m.name:<26s} [{','.join(sorted(m.responds_to)) or '-'}]"
                  + (f"   null {m.null:+.4f}" if m.null is not None else ""))
            print(f"      {m.definition[:98]}")
            for d in m.known_defects:
                print(f"      defect: {d[:94]}")
            if m.cause_of_death:
                print(f"      why:    {m.cause_of_death[:94]}")
    g = grid_report()
    print(f"\n  READING SURFACE -- the 10x10 grid the campaign looks at")
    for k, v in g.items():
        print(f"    {k:<11s} margin {v['margin_nodes']:>3d} -> {v['in_band']:>3d}/100 panels inside "
              f"the pinned band")
    print("=" * 112)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--certify", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(HERE, "_metrology"), exist_ok=True)
    if a.list:
        show(); return 0
    if a.certify:
        rows, bad = certify()
        json.dump({"rows": rows, "disagreements": bad},
                  open(os.path.join(HERE, "_metrology", "metrics_certify.json"), "w"),
                  indent=1, default=float)
        return 0 if not bad else 1
    ok, rows = check()
    json.dump({"pass": ok, "checks": rows,
               "registry": {k: v.as_dict() for k, v in REGISTRY.items()},
               "aliases": ALIASES, "grid": grid_report()},
              open(os.path.join(HERE, "_metrology", "metrics_registry.json"), "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
