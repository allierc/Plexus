#!/usr/bin/env python
"""exp20: A TIME WINDOW OF A GUT-BRAIN RECORDING, as a recording of its own (batch 5: glucose fish 4 cut to 41 min,
the length of fish 1-3, to tell a long session's drift from its length).

    PYTHONPATH=src:tools python tools/exp20_window_recording.py gutbrain_glucose_f4 1586 3820 --name gutbrain_glucose_f4_w41

Every per-frame array (dff, stimulus, condition, t_s restarted at 0) is cut to the volumes [start, stop); the cells, their
positions and cell_index are kept, so the source's input mask (graphs_data/zebrafish/input_mask_<source>.npz) still
indexes the same cells. The trials inside the window are re-indexed; a trial's window is full when its 9 volumes
before and 50 after the onset (the export's window_volumes) lie inside; the held-out trials are drawn again by the
export's own rule (tools/export_gutbrain_recording.py hold_out: the last full-window pulse of every site). The
provenance JSON is the source's with the window written in. Writes graphs_data/zebrafish/<name>_recording.npz / .json.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
ZF = os.path.join(os.environ["GNN_OUTPUT_ROOT"], "graphs_data", "zebrafish")


def window(source, start, stop, name):
    from export_gutbrain_recording import hold_out
    z = np.load(os.path.join(ZF, f"{source}_recording.npz"))
    prov = json.load(open(os.path.join(ZF, f"{source}_recording.json")))
    pre, post = prov["window_volumes"]
    T = stop - start
    out = {k: z[k] for k in z.files}
    for k in ("dff", "stimulus", "condition"):
        out[k] = z[k][start:stop]
    out["t_s"] = z["t_s"][start:stop] - z["t_s"][start]
    c = out["condition"]
    cut = np.flatnonzero(np.diff(c)) + 1
    out["offsets"] = np.r_[0, cut, T].astype(np.int64)
    tr = z["trials"].copy()
    tr = tr[(tr[:, 0] >= start) & (tr[:, 0] < stop)]
    tr[:, 0] -= start
    tr[:, 5] = ((tr[:, 0] - pre >= 0) & (tr[:, 0] + post < T)).astype(float)       # a full window inside the cut
    tr[:, 6] = 0.0
    out["split"] = hold_out(tr, pre, post, T)
    out["trials"] = tr
    np.savez(os.path.join(ZF, f"{name}_recording.npz"), **out)
    prov.update({"name": name, "frames": T, "trials": tr.tolist(),
                 "window_of": {"source": source, "start_volume": start, "stop_volume": stop,
                               "minutes": float(T * prov["volume_s"] / 60), "tool": "tools/exp20_window_recording.py"}})
    json.dump(prov, open(os.path.join(ZF, f"{name}_recording.json"), "w"), indent=1)
    sites = {int(s): int(((tr[:, 2] == s) & (tr[:, 5] > 0)).sum()) for s in np.unique(tr[:, 2])}
    print(f"[window] {name}: volumes {start}-{stop} of {source} ({T} volumes, {T * prov['volume_s'] / 60:.1f} min); "
          f"full-window pulses per site {sites}; held out at {tr[tr[:, 6] > 0, 0].astype(int).tolist()} "
          f"({int((out['split'] == 2).sum())} held-out volumes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("start", type=int)
    ap.add_argument("stop", type=int)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    window(a.source, a.start, a.stop, a.name)
