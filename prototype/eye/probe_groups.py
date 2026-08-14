"""probe_groups -- WHICH synergies to contract, and what they are supposed to do.

The operator itself lives in `probe_ops` -- `muscle_probe [groups]` drives a list of muscle
groups in turn, open loop -- so this module registers nothing. It holds only the eye's four
cardinal synergies and the readout that says whether each one did what the anatomy claims:

    SR + SO   the two DORSAL insertions      -> the eye should go UP
    IR + IO   the two VENTRAL insertions     -> the eye should go DOWN
    LR        the CAUDAL insertion           -> temporal (abduction)
    MR        the ROSTRAL insertion          -> nasal (adduction)

In the plant's colours (`eye_anatomy.MUSCLES`): SR red, SO violet, IR green, IO orange,
LR blue, MR yellow. Muscle indices follow `EA.MUSCLE_KEYS` = LR 0, SR 1, MR 2, IR 3, SO 4,
IO 5.

Nothing here reads the gaze, so `report` is a test OF THE GEOMETRY: on scanned anatomy the
insertions decide where the globe goes, and either they agree with the textbook or they do
not.
"""
from __future__ import annotations

import numpy as np

import eye_anatomy as EA
from probe_ops import groups_spec, MuscleProbeGroups        # noqa: F401  (re-exported)

DORSAL_PAIR = [EA.MUSCLE_KEYS.index("SR"), EA.MUSCLE_KEYS.index("SO")]
VENTRAL_PAIR = [EA.MUSCLE_KEYS.index("IR"), EA.MUSCLE_KEYS.index("IO")]
CAUDAL = [EA.MUSCLE_KEYS.index("LR")]
ROSTRAL = [EA.MUSCLE_KEYS.index("MR")]

PAIRS = [DORSAL_PAIR, VENTRAL_PAIR, CAUDAL, ROSTRAL]
PAIR_LABELS = ["up (SR+SO, the two dorsal insertions)",
               "down (IR+IO, the two ventral insertions)",
               "temporal (LR, the caudal insertion)",
               "nasal (MR, the rostral insertion)"]
# the component of (h, v, t) each synergy should move, and the sign it should have
PAIR_EXPECT = [(1, +1), (1, -1), (0, +1), (0, -1)]


def report(cap, probe, labels=None, expect=None) -> dict:
    """What each synergy actually did to the gaze, read off its own hold window.

    `gaze_excursion_deg` is the (h, v, t) change from the frame the step began to the end of
    the hold; `ok` is whether the LARGEST component of that excursion is the expected one,
    with the expected sign.
    """
    g = np.asarray(cap["gaze"], float)
    frames = np.asarray(cap["frame"])
    labels = labels or PAIR_LABELS
    expect = expect or PAIR_EXPECT
    out = {}
    for slot, grp in enumerate(probe.groups):
        t_on, t_off = probe.window(slot)
        sel = (frames >= t_on) & (frames <= t_off)
        if sel.sum() < 2:
            continue
        base = g[frames <= t_on][-1] if (frames <= t_on).any() else g[0]
        exc = g[sel][-1] - base
        comp, sign = (expect[slot] if slot < len(expect)
                      else (int(np.argmax(np.abs(exc))), int(np.sign(exc[np.argmax(np.abs(exc))]))))
        dominant = int(np.argmax(np.abs(exc)))
        out["+".join(EA.MUSCLE_KEYS[i] for i in grp)] = dict(
            label=labels[slot] if slot < len(labels) else "",
            window=[int(t_on), int(t_off)],
            gaze_excursion_deg=[round(float(v), 2) for v in exc],
            peak_abs_deg=[round(float(v), 2) for v in np.abs(g[sel] - base).max(0)],
            expected=["horizontal", "vertical", "torsion"][comp] + (" +" if sign > 0 else " -"),
            ok=bool(dominant == comp and np.sign(exc[comp]) == sign))
    return out
