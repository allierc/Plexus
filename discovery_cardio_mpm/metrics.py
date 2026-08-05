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
    undefined_on: set = set()       # distortions outside this metric's declared domain
    axis_separable = True           # False when its response is not attributable to single axes
    null = None                     # what a model that knows nothing scores
    higher_is_better = True

    def __call__(self, sim, real, mask=None):
        if self.tier == WITHDRAWN:
            raise Withdrawn(f"{self.name} may not be quoted: {self.cause_of_death}")
        return float(self.compute(_sel(sim, mask), _sel(real, mask)))

    def compute(self, sim, real):                             # pragma: no cover - interface
        raise NotImplementedError

    def as_dict(self):
        return {"name": self.name, "definition": self.definition, "source": self.source,
                "tier": self.tier, "responds_to": sorted(self.responds_to), "domain": self.domain,
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
    null = -0.875
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
              "battery skips it there rather than recording a failure.")
    undefined_on = {"collapse to a line"}

    def compute(self, sim, real):
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
class Openness(Metric):
    name = "openness"
    definition = ("enclosed area divided by the area of the box the loop fits in. A straight line "
                  "is 0; a circle is about 0.79.")
    source = "cardio_mpm_train._openness (audit remedy, 4-5 July)"
    responds_to = {"openness"}
    domain = "a closed path with non-zero extent"

    def compute(self, sim, real):
        return np.median(np.abs(signed_area(sim)) / (bbox_area(sim) + 1e-12))


class PathLength(Metric):
    name = "path_length"
    definition = ("how far a node travels over the beat, summed frame to frame. A loop and a line "
                  "of the same width differ here; the enclosed area alone does not say so.")
    source = "this file -- the one quantity the prototype named and never computed"
    responds_to = {"size", "openness"}
    domain = "any window"
    known_defects = [
        "NOT AN INDEPENDENT AXIS, and the battery is what showed it: flattening an ellipse onto its "
        "long axis changes the distance travelled (the perimeter goes from about pi*(a+b) to 4a), "
        "so this mixes how big the loop is with how open it is. It may not be quoted as a size "
        "measure on its own."]

    def compute(self, sim, real):
        return np.median(np.linalg.norm(np.diff(sim, axis=0), axis=-1).sum(0))


class PeakExcursion(Metric):
    name = "peak_excursion"
    definition = ("how far a node reaches from the centre of its own path, at its furthest. "
                  "Centred, so sliding the loop elsewhere does not change it.")
    source = "cardio_mpm_train.enclosure_row (peak), centred here"
    responds_to = {"size"}
    domain = "any window"
    known_defects = [
        "the first version measured distance from the ORIGIN, so merely translating a loop changed "
        "it -- the same uncentred defect the July audit withdrew morphology_row/size for. Caught by "
        "the battery, which is what the battery is for."]

    def compute(self, sim, real):
        return np.median(np.linalg.norm(_centred(sim), axis=-1).max(0))


class ChiralityMatch(Metric):
    name = "chirality_match"
    definition = "the fraction of nodes circulating the same way round as the recording's."
    source = "cardio_mpm_train.enclosure_row (chir_match)"
    responds_to = {"chirality"}
    domain = ("a closed path with non-zero area. A flattened loop has no handedness, so this is "
              "meaningless there and is expected to move when a loop is collapsed onto a line.")
    null = 0.5                     # a coin toss

    def compute(self, sim, real):
        return (np.sign(signed_area(sim)) == np.sign(signed_area(real))).mean()


class OrientationError(Metric):
    """NEW -- and only because nothing reported it."""

    name = "orientation_error"
    definition = ("the angle between the simulated and the recorded long axis of each loop, median "
                  "over nodes, in radians. 0 is perfect; about 0.785 is what chance gives, because "
                  "an axis has no head or tail.")
    source = ("this file. The quantity lived inside the objective as the phase of a product of "
              "Fourier coefficients and was NEVER REPORTED, so this axis could move for a whole "
              "campaign with nobody able to see it.")
    responds_to = {"orientation"}
    domain = "a path with a distinguishable long axis; degenerate for a circle"
    null = float(np.pi / 4)
    higher_is_better = False

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
    null = 0.0
    known_defects = [
        "CANNOT TELL IN-PHASE FROM EXACTLY ANTIPHASE. The signal it uses is a distance from a "
        "centre, which peaks twice on a path traversed once per beat, so the alignment is only "
        "determined modulo half a beat. Two regions contracting in exact opposition therefore read "
        "as coordinated. Anything less than exact opposition is seen.",
        "reads a whole number of frames, so on a 24-frame beat it cannot resolve better than about "
        "15 degrees of phase"]

    def compute(self, sim, real):
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
            scale = abs(base) if abs(base) > 1e-9 else 1.0
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

    add("nothing is cited as certified before Phase 2 certifies it", not admitted(),
        f"{len(admitted())} certified, {len(live()) - len(admitted())} provisional, "
        f"{len(wd)} withdrawn")

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
