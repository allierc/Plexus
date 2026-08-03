#!/usr/bin/env python
"""descriptors -- Track B's measurement: how far the simulated loop is from the recorded one.

WHAT THIS IS FOR
================================================================================================
Track A has no success rule; its product is a map that gets better by being filled in. Track B has
a sharp one, and this file is it: **the difference between the simulated and the recorded loop
trajectory, decomposed into named axes.**

Decomposed, never as one number. A single score collapses failures that have nothing to do with
each other -- a loop that is the right size but circulates the wrong way, and one that circulates
correctly but encloses nothing, are the same score and completely different problems.

WHY IT IS REAL-REFERENCED AND INTERIOR-ONLY, IN TWO SENTENCES
------------------------------------------------------------------------------------------------
The previous campaign's shape numbers (`size`, `open`, `chir+`) were computed on the SIMULATION
ALONE, over a hundred dashboard nodes of which thirty-six were Dirichlet points pinned to the real
data, and without centring. It therefore reported a "size" that was largely the boundary anchor,
concluded that size was flat against every lever and structurally limited, and spent four rounds
chasing that -- while a from-scratch recomputation showed size did move, and that the dominant
residual was enclosure. Every number here is sim-versus-real, per node, on the moving interior.

THE FIVE AXES
------------------------------------------------------------------------------------------------
    magnitude     is the tissue doing about the right AMOUNT of motion?
    opening       does the path ENCLOSE area, or is it a flattened sliver?
    direction     does it circulate the same way ROUND?
    orientation   is the long axis of the loop POINTING the same way?
    shape         everything the four above do not capture

The first four are computed here. `orientation` is the one the inherited apparatus never reported:
it lives inside the objective as the phase of the `c+ * c-` product and was never surfaced, so it
could have been moving the whole time with nobody able to see it.

`shape` is deliberately absent and is not faked. It needs a learned descriptor -- render both
paths, embed them, measure the distance -- which does not exist in this project yet and is a
Phase-2 candidate held to the same admission rule as everything else. A `None` in that slot is an
honest hole; a hand-rolled substitute would be a fifth axis that presupposes its own answer.

NOTHING HERE IS CERTIFIED
------------------------------------------------------------------------------------------------
This module computes; Phase 2 decides whether any of it may be believed. Every axis returned
carries `certified: False` until it has a measured null, a measured noise floor, and a
demonstrated ability to move when it should and stay still when it should not.
"""
from __future__ import annotations

import numpy as np

# The axes Track B is judged on. `shape` is listed with no implementation on purpose.
AXES = ("magnitude", "opening", "direction", "orientation", "shape")


def _as_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x, dtype=np.float64)


def _harmonics(path, K=4):
    """path [G, M, 2] -> (cp, cm) each [K, M] complex, the +k and -k Fourier coefficients.

    Same construction as the inherited LoopScore, so the two are comparable: the constant term is
    dropped (position invariance) and only k>=1 is kept.
    """
    z = path[..., 0] + 1j * path[..., 1]                       # [G, M]
    G = z.shape[0]
    Z = np.fft.fft(z, axis=0) / G
    K = min(K, (G - 1) // 2)
    cp = Z[1:K + 1]                                            # k = +1..+K
    cm = Z[G - K:G][::-1]                                      # k = -1..-K, ordered k=1..K
    return cp, cm


def _signed_area(path):
    """Shoelace area of each node's closed path. [G,M,2] -> [M]. Sign carries the handedness."""
    x, y = path[..., 0], path[..., 1]
    xn, yn = np.roll(x, -1, axis=0), np.roll(y, -1, axis=0)
    return 0.5 * (x * yn - xn * y).sum(axis=0)


def _bbox_area(path):
    return (np.ptp(path[..., 0], axis=0) * np.ptp(path[..., 1], axis=0))


def _major_axis_angle(path):
    """Principal-axis angle of each node's path, in [0, pi). [G,M,2] -> [M].

    Taken from the covariance rather than from the Fourier phase, so it is an INDEPENDENT reading
    of the same quantity: if the two disagree, one of them is wrong, and that is worth knowing
    before either is trusted.
    """
    p = path - path.mean(axis=0, keepdims=True)
    xx = (p[..., 0] ** 2).mean(0)
    yy = (p[..., 1] ** 2).mean(0)
    xy = (p[..., 0] * p[..., 1]).mean(0)
    return 0.5 * np.arctan2(2 * xy, xx - yy) % np.pi


def _minor_fraction(path):
    """lambda2 / (lambda1 + lambda2) per node: 0 = a straight line, 0.5 = a circle."""
    p = path - path.mean(axis=0, keepdims=True)
    xx = (p[..., 0] ** 2).mean(0); yy = (p[..., 1] ** 2).mean(0)
    xy = (p[..., 0] * p[..., 1]).mean(0)
    tr, det = xx + yy, xx * yy - xy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    return np.where(tr > 0, l2 / np.maximum(l1 + l2, 1e-30), 0.0)


def _circ_diff(a, b):
    """Smallest angular difference on the half-circle (an axis has no head or tail)."""
    d = np.abs(a - b) % np.pi
    return np.minimum(d, np.pi - d)


def loop_residual(sim, real, mov=None, K=4):
    """The Track B measurement.

    sim, real : [G, M, 2] displacement paths over ONE beat, per node, already referenced to the
                start of that beat.
    mov       : optional [M] bool mask of nodes to score. Pass the moving INTERIOR nodes; passing
                None scores everything, including any pinned band, which is what went wrong before.

    Returns a dict of axes; each axis carries `sim`, `real`, `ratio` (or a per-node error), the
    per-node spread, and `certified: False`.
    """
    s, r = _as_np(sim), _as_np(real)
    if s.shape != r.shape:
        raise ValueError(f"sim {s.shape} and real {r.shape} must have the same shape")
    if mov is not None:
        m = np.asarray(mov, bool)
        s, r = s[:, m], r[:, m]
    n = s.shape[1]

    out = {"n_nodes": int(n), "n_frames": int(s.shape[0]), "K": int(K)}

    def axis(name, sv, rv, ratio_of="median"):
        sv, rv = np.asarray(sv, float), np.asarray(rv, float)
        if ratio_of == "median":
            a, b = float(np.median(sv)), float(np.median(rv))
        else:
            a, b = float(np.mean(sv)), float(np.mean(rv))
        out[name] = {"sim": a, "real": b, "ratio": (a / b) if b else float("nan"),
                     "per_node_sd": float(np.std(sv)), "certified": False}

    # --- magnitude: how much motion -----------------------------------------------------------
    # The EUCLIDEAN peak, not max(|x|,|y|). The selftest caught the difference: an L-infinity peak
    # is not rotation-invariant, so merely turning a loop on the spot changed its "magnitude" and
    # the rotation check failed. A magnitude axis that moves when only the orientation axis should
    # is exactly the cross-talk this decomposition exists to prevent.
    axis("magnitude_peak", np.linalg.norm(s, axis=-1).max(axis=0), np.linalg.norm(r, axis=-1).max(axis=0))
    axis("magnitude_energy", np.sqrt((s ** 2).sum(axis=(0, 2))), np.sqrt((r ** 2).sum(axis=(0, 2))))

    # --- opening: does it enclose area --------------------------------------------------------
    sa, ra = _signed_area(s), _signed_area(r)
    axis("opening_area", np.abs(sa), np.abs(ra))
    sb, rb = _bbox_area(s), _bbox_area(r)
    axis("opening_loopiness", np.abs(sa) / np.maximum(sb, 1e-30),
         np.abs(ra) / np.maximum(rb, 1e-30))

    # --- direction: which way round -----------------------------------------------------------
    # A fraction, so it has no sim|real pair: the real data IS the reference, by definition 1.0.
    agree = np.sign(sa) == np.sign(ra)
    out["direction_chirality"] = {
        "sim": float(agree.mean()), "real": 1.0, "ratio": float(agree.mean()),
        "per_node_sd": float(agree.astype(float).std()), "certified": False,
        "note": "fraction of nodes circulating the same way as the recording"}

    # --- orientation: the axis the inherited apparatus never reported --------------------------
    # Two independent readings of the same angle. If they disagree, one is wrong.
    d_cov = _circ_diff(_major_axis_angle(s), _major_axis_angle(r))
    cps, cms = _harmonics(s, K); cpr, cmr = _harmonics(r, K)
    ang_s = 0.5 * np.angle((cps * cms)[0]) % np.pi                 # k=1 carries the ellipse axis
    ang_r = 0.5 * np.angle((cpr * cmr)[0]) % np.pi
    d_fft = _circ_diff(ang_s, ang_r)
    out["orientation_error_rad"] = {
        "sim": float(np.median(d_cov)), "real": 0.0, "ratio": float("nan"),
        "per_node_sd": float(np.std(d_cov)), "certified": False,
        "median_rad_covariance": float(np.median(d_cov)),
        "median_rad_fourier": float(np.median(d_fft)),
        "two_readings_agree_rad": float(np.median(_circ_diff(d_cov, d_fft))),
        "note": "median angular error between the sim and real major axes; 0 is perfect, pi/4 is "
                "the value random orientations would give. Read BOTH readings: they measure the "
                "same thing two ways and a disagreement means one of them is broken."}

    # --- shape: 2-D versus radial is all we have, and it is not enough ------------------------
    axis("shape_minor_fraction", _minor_fraction(s), _minor_fraction(r))
    out["shape_learned"] = {
        "sim": None, "real": None, "ratio": None, "certified": False,
        "note": "NOT IMPLEMENTED, on purpose. Needs a learned descriptor (render both paths, "
                "embed, measure the distance) so that shape disagreement can be seen without "
                "anyone naming the axis first. No image-embedding model exists in this project; "
                "building and CERTIFYING one is a Phase-2 candidate. A hand-rolled substitute "
                "here would be a fifth axis that presupposes its own answer."}
    return out


def random_phase_scramble(path, rng):
    """Give every node an independent random circular time shift.

    The coordination destroyer. A tissue whose points contract in random order is not a beating
    tissue by any biological reading -- so any measure that scores this as unchanged is blind to
    coordination, and cannot be used to judge a claim about waves, timing or rotation.

    LoopScore returns a PERFECT score on this input. That is what this function is for.
    """
    p = np.array(_as_np(path))
    G, M = p.shape[0], p.shape[1]
    shifts = rng.integers(0, G, size=M)
    for j in range(M):
        p[:, j] = np.roll(p[:, j], int(shifts[j]), axis=0)
    return p


def format_row(res):
    """One block of text, the way a run's progress file should carry it."""
    L = ["TRACK B -- loop residual (sim | real | ratio), interior nodes only",
         f"  nodes={res['n_nodes']}  frames={res['n_frames']}  K={res['K']}"]
    for k in ("magnitude_peak", "magnitude_energy", "opening_area", "opening_loopiness",
              "direction_chirality", "shape_minor_fraction"):
        a = res[k]
        L.append(f"  {k:<24s} {a['sim']:.4g} | {a['real']:.4g} | {a['ratio']:.4g}")
    o = res["orientation_error_rad"]
    L.append(f"  {'orientation_error_rad':<24s} {o['median_rad_covariance']:.4g} "
             f"(fourier {o['median_rad_fourier']:.4g}, agree {o['two_readings_agree_rad']:.4g})")
    L.append(f"  {'shape_learned':<24s} not implemented -- see METRICS.md")
    L.append("  NOTHING HERE IS CERTIFIED. Phase 2 decides what may be believed.")
    return "\n".join(L)


def selftest(verbose=True):
    """Every axis must move when it should, and hold still when it should not.

    A measure that has never been watched responding to a change it is supposed to see, and
    ignoring one it is supposed to ignore, is a measure nobody should rank on.
    """
    rng = np.random.default_rng(0)
    G, M = 53, 400
    t = np.linspace(0, 2 * np.pi, G, endpoint=False)
    # a base population of ellipses: random size, aspect, orientation, phase; all counter-clockwise
    a = rng.uniform(0.5, 1.5, M); b = a * rng.uniform(0.3, 0.9, M)
    th = rng.uniform(0, np.pi, M); ph = rng.uniform(0, 2 * np.pi, M)
    u = np.cos(t)[:, None] * a[None] ; v = np.sin(t)[:, None] * b[None]
    base = np.stack([u * np.cos(th) - v * np.sin(th), u * np.sin(th) + v * np.cos(th)], -1)
    base = np.stack([np.roll(base[:, j], int(ph[j] / (2 * np.pi) * G), 0) for j in range(M)], 1)

    def R(sim, real=base):
        return loop_residual(sim, real)

    checks = []

    def want(name, cond, detail=""):
        checks.append((name, bool(cond), detail)); return cond

    r0 = R(base)
    want("identical paths: every ratio is 1",
         abs(r0["magnitude_peak"]["ratio"] - 1) < 1e-9
         and abs(r0["opening_area"]["ratio"] - 1) < 1e-9
         and abs(r0["direction_chirality"]["ratio"] - 1) < 1e-9
         and r0["orientation_error_rad"]["median_rad_covariance"] < 1e-9,
         "and orientation error is 0")

    r = R(base * 2.0)
    want("scaling moves magnitude, not direction",
         abs(r["magnitude_peak"]["ratio"] - 2) < 1e-6 and abs(r["direction_chirality"]["ratio"] - 1) < 1e-9,
         f"peak ratio {r['magnitude_peak']['ratio']:.3f}")

    flipped = base.copy(); flipped[..., 1] *= -1
    r = R(flipped)
    want("mirroring flips direction", r["direction_chirality"]["ratio"] < 0.02,
         f"chirality {r['direction_chirality']['ratio']:.3f}")

    rot = np.pi / 6
    Rm = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    r = R(base @ Rm.T)
    want("rotation moves orientation and nothing else",
         abs(r["orientation_error_rad"]["median_rad_covariance"] - rot) < 1e-6
         and abs(r["magnitude_peak"]["ratio"] - 1) < 1e-6
         and abs(r["opening_area"]["ratio"] - 1) < 1e-6,
         f"orientation error {r['orientation_error_rad']['median_rad_covariance']:.4f} rad, wanted {rot:.4f}")

    r = R(np.stack([np.roll(base[:, j], 7, 0) for j in range(M)], 1))
    want("a GLOBAL time shift changes nothing",
         abs(r["magnitude_peak"]["ratio"] - 1) < 1e-9 and abs(r["opening_area"]["ratio"] - 1) < 1e-9,
         "as it must: where the beat starts is not a property of the loop")

    r = R(base + np.array([3.0, -2.0]))
    want("a translation changes nothing",
         abs(r["opening_area"]["ratio"] - 1) < 1e-9
         and r["orientation_error_rad"]["median_rad_covariance"] < 1e-9,
         "the loop is where it goes, not where it sits")

    squashed = base.copy()
    ca, sa_ = np.cos(th), np.sin(th)
    proj = base[..., 0] * ca[None] + base[..., 1] * sa_[None]
    squashed = np.stack([proj * ca[None], proj * sa_[None]], -1)          # collapse onto the major axis
    r = R(squashed)
    want("collapsing to a line kills opening",
         r["opening_area"]["ratio"] < 1e-6 and r["shape_minor_fraction"]["ratio"] < 1e-6,
         f"area ratio {r['opening_area']['ratio']:.2e}")

    # the one that matters: destroy coordination and see whether anything notices
    scr = random_phase_scramble(base, np.random.default_rng(1))
    r = R(scr)
    moved = (abs(r["magnitude_peak"]["ratio"] - 1) > 0.05
             or abs(r["opening_area"]["ratio"] - 1) > 0.05
             or r["orientation_error_rad"]["median_rad_covariance"] > 0.05)
    want("per-node time scramble is INVISIBLE to all five axes", not moved,
         "recorded, not fixed: none of these axes can see coordination, so no claim about "
         "waves, timing or rotation is scoreable until one can")

    if verbose:
        print("\n  descriptors selftest -- every axis watched moving and watched holding still")
        for name, ok, detail in checks:
            print(f"   [{'  ok  ' if ok else ' FAIL '}] {name:<48s} {detail}")
    return all(ok for _, ok, _ in checks)


if __name__ == "__main__":
    import sys
    ok = selftest()
    print(f"\n  descriptors: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
