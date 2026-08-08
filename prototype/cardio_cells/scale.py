"""scale -- how big is a domain of constant contraction axis? Measured, not chosen.

The split-half test says the axis field reproduces at 0.999 with NO smoothing, so its fine
structure is a property of the tissue and not of my kernel. That makes the next question
answerable from the data instead of guessed: over what distance does the axis stay the same?

    C(r) = < cos 2(theta(x) - theta(x+r)) >   weighted by how well-defined each axis is

C falls from 1 to 0 over the size of a domain. If that length is a few grid points the domains are
cell-sized; if it is tens, they are tissue-scale and this whole approach finds folds in the sheet
rather than cells. The number decides the method.
"""
import numpy as np


def corr_vs_r(ang, w, rmax=40):
    z = w * np.exp(2j * ang)
    W = w
    out = []
    for r in range(1, rmax + 1):
        acc = num = 0.0
        for dy, dx in ((0, r), (r, 0), (0, -r), (-r, 0)):
            a = z
            b = np.roll(np.roll(z, dy, 0), dx, 1)
            wb = np.roll(np.roll(W, dy, 0), dx, 1)
            # trim the wrapped band so periodicity does not fake a correlation
            sl = (slice(abs(dy), None) if dy > 0 else slice(None, -abs(dy)) if dy < 0 else slice(None),
                  slice(abs(dx), None) if dx > 0 else slice(None, -abs(dx)) if dx < 0 else slice(None))
            ua = a[sl] / np.maximum(np.abs(a[sl]), 1e-12)
            ub = b[sl] / np.maximum(np.abs(b[sl]), 1e-12)
            ww = np.sqrt(W[sl] * wb[sl])
            acc += (ww * (ua * np.conj(ub)).real).sum(); num += ww.sum()
        out.append(acc / max(num, 1e-12))
    return np.array(out)
