"""validate -- is the structure in the axis field real, or is it my own smoothing kernel?

THE FAILURE MODE THIS EXISTS TO CATCH
================================================================================================
Take any noisy angle field, smooth it, and plot the gradient of exp(2i.theta): you get a maze of
closed ridges with a characteristic size set by the smoothing kernel. It looks exactly like a
tessellation of cells. The strain-based axis produced precisely that -- a maze at ~5 grid points
with a kernel of 1 grid point -- and no amount of looking at it would have told me whether it was
biology.

THE TEST THAT SETTLES IT NEEDS NO GROUND TRUTH. There are four beats. Build the descriptor from
beats {1,3} and, independently, from beats {2,4}. A real cell boundary is a property of the tissue
and is in both halves. A maze made of noise is different noise each time.

    reproducibility = circular correlation between the two halves' axis fields
    null             = the same, with one half spatially rolled -- destroys the pairing, keeps the
                       smoothness, so it measures what any smooth field would score by chance

Reported against smoothing scale, because the scale at which reproducibility PEAKS is the real
correlation length of the tissue, and it is a measurement rather than a knob I chose.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

import beat as B
import strain as ST


def half_beats(uv, which, onsets=B.ONSETS, n=B.BEAT):
    sel = [onsets[i] for i in which if onsets[i] + n <= uv.shape[0]]
    return np.mean([uv[o:o + n] - uv[o:o + n][0] for o in sel], 0)


def axis_disp(bb, sigma):
    ang, amp, aniso = B.axis_and_anisotropy(bb)
    w = np.clip((aniso - 1) / 6, 0, 1) * np.clip(amp / np.percentile(amp, 90), 0, 1)
    return _smooth(ang, w, sigma)


def axis_strain(bb, sigma):
    E = ST.strain_series(bb, sigma=1.0)
    ang, e2, aniso, _, _ = ST.contraction_axis(E)
    w = np.clip(e2 / np.percentile(e2, 90), 0, 1)
    return _smooth(ang, w, sigma)


def _smooth(ang, w, sigma):
    z = w * np.exp(2j * ang)
    if sigma > 0:
        z = ndi.gaussian_filter(z.real, sigma) + 1j * ndi.gaussian_filter(z.imag, sigma)
        w = np.maximum(ndi.gaussian_filter(w, sigma), 1e-9)
    return z / np.maximum(w, 1e-9), w


def circ_corr(za, zb, wa, wb):
    """Weighted agreement of two doubled-angle fields: 1 = identical axes everywhere, 0 = unrelated."""
    w = np.sqrt(np.abs(wa) * np.abs(wb))
    ua = za / np.maximum(np.abs(za), 1e-12)
    ub = zb / np.maximum(np.abs(zb), 1e-12)
    return float((w * (ua * np.conj(ub)).real).sum() / max(w.sum(), 1e-12))


def report(uv, sigmas=(0, 0.5, 1, 1.5, 2, 3, 4, 6, 8), rolls=(37, 61, 89)):
    out = {}
    for name, fn in (("displacement", axis_disp), ("strain", axis_strain)):
        b1, b2 = half_beats(uv, [0, 2]), half_beats(uv, [1, 3])
        print(f"\n  {name.upper()}-based axis")
        print(f"  {'sigma(grid)':>12s} {'sigma(px)':>10s} {'reproducible':>13s} "
              f"{'null(rolled)':>13s} {'above null':>11s}")
        out[name] = []
        for s in sigmas:
            za, wa = fn(b1, s)
            zb, wb = fn(b2, s)
            r = circ_corr(za, zb, wa, wb)
            nulls = [circ_corr(za, np.roll(np.roll(zb, k, 0), k, 1),
                               wa, np.roll(np.roll(wb, k, 0), k, 1)) for k in rolls]
            nl = float(np.mean(nulls))
            out[name].append({"sigma": s, "repro": r, "null": nl, "excess": r - nl})
            print(f"  {s:>12.1f} {s*15:>10.0f} {r:>13.4f} {nl:>13.4f} {r-nl:>11.4f}")
    return out
