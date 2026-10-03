"""exp20: the EXPERIMENTAL DESIGN of every fish of the deposit, read from its own files (no trace loaded):
experiments/exp20_gutbrain_graphcast/data/design.json -- per fish its condition, volumes, volume period, minutes,
segmented cells, UV pulses (onset volume, site, duration), the grating's on fraction and whether a swim channel has
bouts (tools/export_gutbrain_recording.py's own readers).

    PYTHONPATH=src:tools python tools/exp20_design.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
import export_gutbrain_recording as X  # noqa: E402

COND = ["glucose", "glutamate", "Lglucose", "fish_water", "blood_glucose"]


def main():
    import h5py
    out = []
    for c in COND:
        ks = sorted({int(m.group(1)) for f in glob.glob(os.path.join(X.DATA, f"{c}_fish*_cells0_clean.hdf5"))
                     for m in [re.search(rf"^{c}_fish(\d+)_", os.path.basename(f))] if m})
        for k in ks:
            fc, fv, fe = X.files(c, k)
            E = X.read_ephys(fe)
            st = X.frame_clock(E)
            h = h5py.File(fc, "r")
            T, N = int(h["t"][()]), int(h["n"][()])
            st = st[:T]
            S, tr, period, pre, post, live = X.forcings_and_trials(E, st)
            out.append({"condition": c, "fish": k, "volumes": T, "volume_s": period, "minutes": T * period / 60,
                        "cells": N, "pulses": [{"volume": int(r[0]), "site": int(r[2]), "ms": float(r[1])} for r in tr],
                        "grating_on": float(np.mean(S[:, 3] > 0)), "swim_live": live})
            print(f"{c:14s} fish {k}: {T:5d} vol x {period:.3f} s = {T * period / 60:5.1f} min, {N:,} cells, "
                  f"{len(tr)} pulses, sites {''.join(str(int(r[2])) for r in tr)}", flush=True)
    json.dump(out, open(os.path.join(X.DATA, "design.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
