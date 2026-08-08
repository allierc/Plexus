"""The beat, isolated -- and the two per-point quantities cells should differ in.

WHY NOT THE RAW LAGRANGIAN FIELD. The provided u,v are displacements from frame 0, so they carry
the sheet's slow drift on top of the beat. A cell boundary is a discontinuity in the BEAT, not in
the drift, so the drift is removed first: each point's trajectory is band-passed around the beat.

TWO QUANTITIES, both per grid point, both computed over time and not over space:
  AXIS   the principal direction of that point's motion during a beat. A cardiomyocyte contracts
         along its own long axis, so the axis should be constant inside a cell and jump across a
         boundary.
  PHASE  when in the beat that point moves. Cells are electrically coupled but not identical, so
         the activation sweeps -- and a boundary is where the timing steps.
"""
import numpy as np

ONSETS = [2, 51, 101, 152, 204]
BEAT = 49


def load(path="/tmp/uv.npy"):
    return np.load(path)                                   # [T,H,W,2] px, Lagrangian from frame 0


def mean_beat(uv, onsets=ONSETS, n=BEAT):
    """The average beat: [n,H,W,2], each beat re-referenced to its own first frame.

    Averaging the four beats is the cheapest noise reduction available and it costs nothing we
    want: a cell that contracts differently on different beats is not what we are looking for.
    """
    bs = [uv[o:o + n] - uv[o:o + n][0] for o in onsets if o + n <= uv.shape[0]]
    return np.mean(bs, 0), len(bs)


def axis_and_anisotropy(b):
    """Per point: the principal angle of its beat trajectory, and how directional it is.

    PCA of the [n,2] path. lam1/lam2 says whether the motion is a line (a contraction along an
    axis) or a blob (no preferred direction) -- and an axis is only meaningful where it is a line.
    """
    n, H, W, _ = b.shape
    x = b - b.mean(0, keepdims=True)
    cxx = (x[..., 0] ** 2).mean(0); cyy = (x[..., 1] ** 2).mean(0)
    cxy = (x[..., 0] * x[..., 1]).mean(0)
    tr, det = cxx + cyy, cxx * cyy - cxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 1e-20)
    ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy)             # principal axis, mod pi
    return ang, np.sqrt(l1), l1 / l2


def phase(b):
    """Per point: the phase of the fundamental of its speed, i.e. WHEN it moves in the beat."""
    s = np.linalg.norm(b - b.mean(0, keepdims=True), axis=-1)     # [n,H,W]
    s = s - s.mean(0, keepdims=True)
    F = np.fft.rfft(s, axis=0)
    k = int(np.argmax(np.abs(F[1:6]).sum((1, 2))) + 1)            # the beat's own harmonic
    return np.angle(F[k]), np.abs(F[k]), k
