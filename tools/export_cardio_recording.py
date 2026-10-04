#!/usr/bin/env python
"""Freeze a cardiomyocyte recording as a training REFERENCE: graphs_data/cardio/<specimen>_recording.npz.

    PYTHONPATH=src python tools/export_cardio_recording.py healthy hcm

The recording is read by the prototype's own loader (`prototype/cardio_mpm/strain/recording.load`),
which content-checks the healthy tracking against its SHA-256 and reads the HCM sheet's 137-grid
derivatives, and written once, with its provenance, where `plexus.tasks.recording` reads it. A
reference is data, not code: after this the codebase never imports the prototype to train on it.

    pos      [T, N, 2]  tracked node positions per frame, in world units (the sheet's [0.15, 0.85]^2)
    labels   [N]        the cell each node belongs to, 1..C
    onsets   [B]        frame of each beat's speed peak
    dt_s                seconds per frame of the recording
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "prototype", "cardio_mpm", "strain"))
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
import recording as R                                                  # noqa: E402
from plexus.paths import graphs_data_path                              # noqa: E402

for specimen in (sys.argv[1:] or ["healthy", "hcm"]):
    rec = R.load(device="cpu", specimen=specimen)
    pos = rec["pos"].numpy().astype(np.float32)
    lab = rec["labels"].numpy().astype(np.int64)
    out = graphs_data_path("cardio", f"{specimen}_recording.npz")
    np.savez_compressed(out, pos=pos, labels=lab, onsets=np.asarray(rec["onsets"], np.int64),
                        dt_s=np.float64(rec["dt_s"]), n_cells=np.int64(rec["n_cells"]))
    prov = dict(specimen=specimen, source="prototype/cardio_mpm/strain/recording.load",
                frames=int(pos.shape[0]), nodes=int(pos.shape[1]), cells=int(rec["n_cells"]),
                onsets=list(map(int, rec["onsets"])), dt_s=float(rec["dt_s"]),
                sha256_pos=hashlib.sha256(pos.tobytes()).hexdigest())
    json.dump(prov, open(out[:-4] + ".provenance.json", "w"), indent=1)
    print(f"{specimen}: {pos.shape[0]} frames x {pos.shape[1]:,} nodes, {rec['n_cells']} cells, "
          f"onsets {rec['onsets']} -> {out}")
