#!/usr/bin/env python
"""data -- the ONE path to the recording. No fallback, no search, no default.

WHY THE FALLBACK IS GONE
================================================================================================
The inherited loader resolved the recording by trying three locations and taking whichever
existed first. That was added after a batch was lost to a moved file, and it is the wrong cure:
it converts a LOUD failure ("that file is not there") into a SILENT one ("fitted a different
recording than you think"). Every number in a campaign rests on which array was opened, so this
is the last place in the project that may guess.

    a missing file raises, naming the path it wanted
    a file whose content does not match its declared identity raises
    there is no search order to reason about, because there is no search

IDENTITY IS CONTENT, NOT PATH
------------------------------------------------------------------------------------------------
The dataset holds TWO specimens under FIVE filenames -- the healthy sheet tracked twice, the
diseased sheet three times, under names that do not say so. `Cardio_0/derivatives.npy` sounds
like a third specimen and is a second tracking of the diseased one. So a seal on a filename
seals nothing, and `specimen_id()` fingerprints the DISPLACEMENT FIELD instead: two files
holding the same measurement get the same id however they are named.

Phase 1 uses that to seal the held-out specimen. Phase 0 uses it so a run's manifest records
what was actually opened.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# The MPM sheet domain -- the block the spec fills, [0.15,0.85]^2. The recording, normalised to
# ~[0,1] by the image width, is mapped onto it.
DOM_LO, DOM_HI = 0.15, 0.85
DOM = DOM_HI - DOM_LO

# The recording this project fits, named once. `ingest.py` rebuilds it bit-exactly from the
# microscope derivatives; there is no other admissible source.
DEFAULT_NPZ = os.path.abspath(os.path.join(HERE, "..", "prototype", "cardio_mpm", "cardio_real.npz"))

# sha256 of `pos`, established by ingest.py --verify on 2 August 2026. A file that does not
# match this is a different recording, whatever it is called.
HEALTHY_POS_SHA256 = "e5fb1774286545923e6dd97f62e9f64891686d8080c43fee09d9fc2ba8ecc725"


class DataRefusal(RuntimeError):
    """Raised instead of guessing. Carries the path it wanted."""


def open_npz(path=None, expect_sha256=None):
    """Open the recording. Raises DataRefusal rather than searching or falling back."""
    p = os.path.abspath(path or DEFAULT_NPZ)
    if not os.path.exists(p):
        raise DataRefusal(
            f"recording not found: {p}\n"
            f"  There is no fallback by design. Rebuild it with:\n"
            f"    python ingest.py --write {p}")
    z = np.load(p)
    if "pos" not in z.files:
        raise DataRefusal(f"recording has no 'pos' array: {p} (has {z.files})")
    if expect_sha256:
        got = hashlib.sha256(np.ascontiguousarray(z["pos"]).tobytes()).hexdigest()
        if got != expect_sha256:
            raise DataRefusal(
                f"recording content does not match its declared identity: {p}\n"
                f"  expected sha256(pos) {expect_sha256}\n"
                f"  got                  {got}")
    return z


def specimen_id(pos):
    """A content fingerprint of the MEASUREMENT, not the file.

    Hashes the displacement field referenced to frame 0 and rounded to 1e-6 of an image width
    (~0.002 px, far below the 0.017 px measurement jitter), so re-savings, dtype churn and
    filename changes do not move it, while a genuinely different specimen does.
    """
    p = np.asarray(pos, dtype=np.float64)
    d = np.round((p - p[0]) * 1e6).astype(np.int64)
    return hashlib.sha256(np.ascontiguousarray(d).tobytes()).hexdigest()


def beats(pos):
    """Beat onsets and period, from the mean nodal speed. Deterministic, no fitting.

    Reproduces the inherited detector exactly so the two are comparable, and additionally
    reports what the inherited one silently discarded: the onset spacings are 49/50/51/52, the
    mean is 50.5, and `int(round(50.5))` is 50 by banker's rounding -- so `period` is not the
    mean interval. Anything that needs a beat window should take it from `onsets`, not `period`.
    """
    P = np.asarray(pos, dtype=np.float32)
    from scipy.signal import find_peaks
    spd = np.linalg.norm(np.diff(P, axis=0), axis=2).mean(1)
    pk, _ = find_peaks(spd, height=spd.mean(), distance=20)
    onsets = [int(x) for x in pk]
    gaps = list(np.diff(onsets)) if len(onsets) > 1 else []
    period = int(round(float(np.mean(gaps)))) if gaps else 50
    return {"onsets": onsets, "gaps": [int(g) for g in gaps],
            "mean_gap": float(np.mean(gaps)) if gaps else float("nan"),
            "period": period, "n_frames": int(P.shape[0]), "n_nodes": int(P.shape[1])}


def load_real(rest_pos, bwidth=0.06, path=None, expect_sha256=HEALTHY_POS_SHA256):
    """Map the recording onto the MPM particles. One model frame = one real frame.

    Returns (real_disp [F,N,2] per particle, bnd [N] bool outer band, onsets, period).
    Same contract as the inherited loader, so the ported trainer is a drop-in -- but the path
    is explicit and the content is checked.
    """
    from scipy.spatial import cKDTree
    z = open_npz(path, expect_sha256)
    P = z["pos"].astype(np.float32)
    Pm = DOM_LO + DOM * P
    node = cKDTree(Pm[0]).query(np.asarray(rest_pos, np.float32))[1]
    real_disp = (Pm[:, node] - Pm[0, node]).astype(np.float32)
    b = beats(P)
    rp = np.asarray(rest_pos, np.float32)
    bnd = ((rp[:, 0] < DOM_LO + bwidth) | (rp[:, 0] > DOM_HI - bwidth)
           | (rp[:, 1] < DOM_LO + bwidth) | (rp[:, 1] > DOM_HI - bwidth))
    return real_disp, bnd, b["onsets"], b["period"]


if __name__ == "__main__":
    import json
    z = open_npz(expect_sha256=HEALTHY_POS_SHA256)
    p = z["pos"]
    print(json.dumps({"path": DEFAULT_NPZ, "specimen_id": specimen_id(p)[:16], **beats(p)}, indent=1))
