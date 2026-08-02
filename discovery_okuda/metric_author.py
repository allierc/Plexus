#!/usr/bin/env python
"""metric_author -- when the eye sees something no number can measure, write the number.

THE GAP THIS CLOSES
------------------------------------------------------------------------------------------------
Every numeric measurement in this campaign has a closed vocabulary: elongation, tube count, spot
count, diameter. Each one presupposes the phenotype it is looking for. So the only component that
can name something nobody listed is the one that produces FREE TEXT -- the model that watches the
movie and describes it.

Cedric: "maybe the VLM will catch it?" -- yes, and it is the ONLY thing that can. Then: "best
scenario, the metrics or the analysis or another agent would write a code snippet to analyse
[it], if pointed as interesting from the VLM description." That is the missing link. Today the
description says "a ring-shaped shell" and the sentence goes nowhere: no metric covers it, so the
observation cannot be sweep-tested, cannot enter the map, and cannot be argued about.

    the eye names it  ->  an agent writes the metric  ->  THE METRIC MUST EARN ADMISSION
                                                          ->  then it can measure real runs

THE THIRD ARROW IS THE ONE THAT MATTERS
------------------------------------------------------------------------------------------------
A metric written on demand, by an agent, to measure a thing that agent just found interesting, is
the most dangerous object this campaign can produce. Its entire history is broken metrics:

    retention          rewarded STASIS -- a sphere that never moved scored 1.000
    protr_peak         rewarded DESTRUCTION -- the tearing spike became the score
    hollow_frac        counted CELL DIVISION and was read as damage
    Q                  relaxed a fresh sphere and returned 1.014 for every run
    ta_* radius        measured from the world origin, so drift read as elongation

Every one of those was plausible when written. So a candidate metric here is not admitted because
it looks reasonable; it is admitted because it PASSES A SUITE OF SHAPES WHOSE ANSWER IS ALREADY
KNOWN. The suite is synthetic on purpose -- analytic point sets, no simulator in the loop -- so it
tests the metric and nothing else. This is the same move that derived the tube threshold from
analytic capsules, and the same move that killed my curvature hypothesis with perfect spheres.

AND THE AUTHOR DOES NOT CERTIFY ITS OWN WORK. Writing the metric and admitting it are different
jobs, held apart for the same reason the Proposer does not referee its own batch: the Metrologist
owns admission, the author only proposes.

THE THREE THINGS A METRIC MUST DO -- the standing admissibility standard, unchanged
------------------------------------------------------------------------------------------------
  1 AGREE WITH GROUND TRUTH   rank the known cases in the known order
  2 SEPARATE                  distinct shapes must get distinguishable values, by a stated margin
  3 IGNORE NUISANCES          rotation, translation, scale, and POINT COUNT must not move it.
                              Point count is not a theoretical concern here: cell number is the
                              variable this substrate changes most, and it has confounded a
                              measurement before.
"""
from __future__ import annotations

import numpy as np

# ============================================================================ synthetic shapes
# Analytic surfaces sampled as point clouds -- the same thing a cell-centroid cloud is. No mesh,
# no simulator: a metric that cannot tell these apart cannot tell anything apart.


def sphere_shell(n=800, R=5.0, rng=None):
    rng = rng or np.random.default_rng(0)
    v = rng.normal(size=(n, 3))
    return R * v / np.linalg.norm(v, axis=1, keepdims=True)


def torus(n=800, R=5.0, r=1.6, rng=None):
    rng = rng or np.random.default_rng(0)
    u = rng.uniform(0, 2 * np.pi, n)
    w = rng.uniform(0, 2 * np.pi, n)
    return np.stack([(R + r * np.cos(w)) * np.cos(u),
                     (R + r * np.cos(w)) * np.sin(u),
                     r * np.sin(w)], 1)


def capsule(n=800, aspect=3.0, r=1.5, rng=None):
    """A tube with hemispherical caps -- the phenotype the campaign is chasing."""
    rng = rng or np.random.default_rng(0)
    L = max((aspect - 1.0) * 2 * r, 1e-6)
    a_cyl = 2 * np.pi * r * L
    a_cap = 4 * np.pi * r * r
    n_cyl = int(round(n * a_cyl / (a_cyl + a_cap)))
    th = rng.uniform(0, 2 * np.pi, n_cyl)
    z = rng.uniform(-L / 2, L / 2, n_cyl)
    cyl = np.stack([r * np.cos(th), r * np.sin(th), z], 1)
    v = rng.normal(size=(n - n_cyl, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= r
    v[:, 2] += np.sign(v[:, 2]) * L / 2
    return np.vstack([cyl, v])


def invagination(n=800, R=5.0, depth=0.85, rng=None):
    """A sphere with one deep dimple -- genus 0, but it can LOOK like a hole."""
    p = sphere_shell(n, R, rng)
    ax = np.array([0.0, 0.0, 1.0])
    c = p @ ax / R
    pull = np.clip((c - 0.55) / 0.45, 0, 1)
    return p - np.outer(pull * depth * R, ax)


SHAPES = {"sphere": sphere_shell, "torus": torus, "capsule": capsule,
          "invagination": invagination}


# ============================================================================ a candidate metric
def shape_anisotropy(pts):
    """Candidate: how the cloud's spread is distributed across its three principal axes.

    Returns (prolate, oblate), each in [0, 1]:
        prolate  one axis dominates            -- a tube
        oblate   one axis is much SHORTER      -- a flat ring
    A sphere is neither. A torus is oblate and not prolate; a capsule is prolate and not oblate.
    Scale-free by construction (the eigenvalues are normalised by their sum), and rotation- and
    translation-free because it is a centred PCA.
    """
    p = np.asarray(pts, float)
    p = p - p.mean(0)
    ev = np.linalg.eigvalsh(np.cov(p.T))
    ev = np.sort(np.maximum(ev, 0))[::-1]
    s = ev.sum()
    if s <= 0:
        return 0.0, 0.0
    l1, l2, l3 = ev / s
    return float((l1 - l2) / l1), float((l2 - l3) / l2)


def ring_score(pts):
    """Candidate: is there a HOLE through the middle?

    Project onto the best-fit plane (the two long principal axes) and look at how far the points
    sit from the axis. A ring keeps them in a narrow annulus well away from the centre; a sphere
    or a dimpled sphere fills every radius down to zero.
    """
    p = np.asarray(pts, float)
    p = p - p.mean(0)
    ev, evec = np.linalg.eigh(np.cov(p.T))
    plane = evec[:, np.argsort(ev)[::-1][:2]]          # the two widest directions
    q = p @ plane
    d = np.linalg.norm(q, axis=1)
    d /= (d.max() + 1e-12)
    return float(np.percentile(d, 5))                  # a hole pushes even the innermost points out


# ============================================================================ certification
def certify(fn, expect, name="candidate", margin=0.15, n_seeds=4, verbose=True):
    """Admit a metric only if it agrees with ground truth, separates, and ignores nuisances.

    `expect` maps shape name -> "high" | "low". Deterministic; no model is consulted.
    """
    rng_shapes = {}
    for s, gen in SHAPES.items():
        rng_shapes[s] = [gen(n=800, rng=np.random.default_rng(k)) for k in range(n_seeds)]

    vals = {s: np.array([float(fn(p)) for p in ps]) for s, ps in rng_shapes.items()}
    report = {"name": name, "values": {s: (float(v.mean()), float(v.std()))
                                       for s, v in vals.items()}}

    hi = [s for s, e in expect.items() if e == "high"]
    lo = [s for s, e in expect.items() if e == "low"]
    hmin = min(vals[s].mean() for s in hi) if hi else 0.0
    lmax = max(vals[s].mean() for s in lo) if lo else 0.0
    report["separation"] = float(hmin - lmax)
    ok_rank = hmin - lmax >= margin

    # nuisance invariance: rotate, translate, rescale, and CHANGE THE POINT COUNT
    base = rng_shapes[hi[0]][0] if hi else rng_shapes["sphere"][0]
    v0 = float(fn(base))
    Q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    nuis = {"rotated": float(fn(base @ Q)),
            "translated": float(fn(base + np.array([13.0, -7.0, 4.0]))),
            "rescaled": float(fn(base * 3.7)),
            "half_the_points": float(fn(SHAPES[hi[0] if hi else "sphere"](n=400))),
            "double_the_points": float(fn(SHAPES[hi[0] if hi else "sphere"](n=1600)))}
    drift = {k: abs(x - v0) for k, x in nuis.items()}
    ok_nuis = max(drift.values()) < 0.5 * max(abs(v0), 1e-9) + 0.05
    report["nuisance_drift"] = {k: round(x, 4) for k, x in drift.items()}

    report["admitted"] = bool(ok_rank and ok_nuis)
    report["why"] = ("admitted" if report["admitted"] else
                     ("fails to separate "
                      f"(margin {hmin - lmax:.3f} < {margin})" if not ok_rank else
                      f"not nuisance-invariant (worst drift {max(drift.values()):.3f})"))
    if verbose:
        print(f"\n  CANDIDATE `{name}`  expect high={hi} low={lo}")
        for s, (m, sd) in report["values"].items():
            print(f"    {s:14} {m:7.3f} +/- {sd:.3f}")
        print(f"    separation {report['separation']:+.3f}   "
              f"worst nuisance drift {max(drift.values()):.4f}")
        print(f"    -> {'ADMITTED' if report['admitted'] else 'REJECTED'}: {report['why']}")
    return report


# ============================================================================ the trigger
def unmeasured(description, admitted_metrics):
    """Does the eye's description name a feature no admitted metric can quantify?

    Deliberately crude and deliberately over-eager: it costs nothing to look twice at a run, and
    the failure this guards against -- a genuinely new structure passing unremarked -- is not
    recoverable later, because nobody re-reads old captions.
    """
    vocab = {
        "ring": ("torus", "ring", "donut", "doughnut", "annulus", "hole through"),
        "hole": ("hole", "perforat", "punctur", "opening"),
        "flat": ("flat", "disc", "disk", "pancake", "sheet"),
        "branch": ("branch", "bifurcat", "fork", "split into"),
        "invagination": ("invaginat", "dimple", "indent", "pit", "crater"),
        "multiple_lobes": ("lobe", "cauliflower", "cluster of bumps"),
        # CELL-LEVEL, not tissue-level. The first version of this vocabulary was entirely about
        # the shape of the whole tissue, so when Cedric looked at a cross-section and said the
        # tube cells were "very distorted, very thin", nothing here would have fired -- the
        # trigger built to catch what no metric measures had a tissue-shaped blind spot.
        "cell_distortion": ("stretch", "distort", "elongated cell", "thin cell", "squash",
                            "compress", "sliver", "flattened cell", "deformed cell"),
        "cell_heterogeneity": ("some cells", "uneven", "irregular size", "heterogen",
                               "mixture of", "two populations", "patchy"),
    }
    d = (description or "").lower()
    hits = [k for k, words in vocab.items() if any(w in d for w in words)]
    covered = {"branch"} if any("tube" in m or "branch" in m for m in admitted_metrics) else set()
    return [h for h in hits if h not in covered]


# ============================================================================ self-test
if __name__ == "__main__":
    print("=" * 78)
    print("Can a metric written on demand be trusted? Only if it passes shapes we know.")
    print("=" * 78)

    r1 = certify(lambda p: shape_anisotropy(p)[1], name="oblateness (a ring is flat)",
                 expect={"torus": "high", "sphere": "low", "capsule": "low",
                         "invagination": "low"})
    r2 = certify(ring_score, name="ring_score (is the middle empty?)",
                 expect={"torus": "high", "sphere": "low", "capsule": "low",
                         "invagination": "low"})
    r3 = certify(lambda p: shape_anisotropy(p)[0], name="prolateness (a tube is long)",
                 expect={"capsule": "high", "sphere": "low", "torus": "low",
                         "invagination": "low"})

    print("\n" + "=" * 78)
    print("A METRIC THAT SOUNDS RIGHT AND IS NOT -- the control for this whole exercise")
    print("=" * 78)
    bad = certify(lambda p: float(np.linalg.norm(np.asarray(p) - np.asarray(p).mean(0),
                                                 axis=1).max()),
                  name="max radius ('a torus is wide')",
                  expect={"torus": "high", "sphere": "low", "capsule": "low",
                          "invagination": "low"})
    assert not bad["admitted"], "a scale-dependent metric must be refused"
    print("\n  ^ refused, and for the right reason: it is not scale-invariant, so it would have")
    print("    measured how BIG a run got and reported it as how ring-LIKE it was.")

    print("\n" + "=" * 78)
    print("THE TRIGGER -- what the eye says, and whether a number exists for it")
    print("=" * 78)
    ADMITTED = ("elongation_peak", "elongation_at_end", "ta_n_tubes_final")
    for d in ("a single protrusion extends from the sphere and narrows at the tip",
              "the shell has deformed into a ring with a clear hole through the middle",
              "the protrusion splits into two branches near its end",
              "a deep dimple forms on one side, almost meeting the far wall"):
        print(f"\n  \"{d[:66]}...\"\n    unmeasured: {unmeasured(d, ADMITTED) or 'nothing new'}")

    ok = all(r["admitted"] for r in (r1, r2, r3)) and not bad["admitted"]
    print("\n" + ("metric_author OK" if ok else "SELF-TEST FAILED"))
    raise SystemExit(0 if ok else 1)
