#!/usr/bin/env python
"""Freeze ZAPBench as a training REFERENCE: graphs_data/zebrafish/zapbench_recording.npz (+ .provenance.json).

    PYTHONPATH=src python tools/export_zapbench_recording.py

ZAPBench (Lueckmann et al., ICLR 2025) is one larval zebrafish (6 dpf, Tg(elavl3:H2B-GCaMP7f)) imaged by
light sheet for 2 h, one whole-brain volume every 0.914 s, while nine visual stimulus conditions were played
in sequence. Google's release segmented 71,721 somata and extracted one dF/F trace per soma (the 8th
percentile of each voxel over 400 frames as F0, clipped to [-0.25, 1.5], App. B.3). This script only READS
the release mirror (tools/.../data/fetch_zapbench_release.py, MD5-checked against the bucket); it never
recomputes a trace. Runs in the devcontainer (INSTRUCTION.md, the cluster rule).

WHY THE RELEASE AND NOT zapbench_dff_full.npy. The local npy (7,870 frames) was assembled from fishFuncEM
files; on the first 512 neurons it correlates 0.980 with the release traces and differs by up to 0.144 dF/F
(2026-09-30). The benchmark's own series is the release's, 7,879 frames (`CONDITION_OFFSETS` ends there).

WHY COLUMN i IS SEGMENT LABEL i + 1. ZAPBench's `get_segmentation_dataframe` says the dataframe maps to "indices
in associated trace matrices through df.loc[trace_idx] (or label minus 1)", sorted by label; the centroid file's
rows are labels 1..71,721 in order, which is checked here.

WHY 22 STIMULUS NUMBERS. `stimuli_features` has 26 columns; App. B.6 says the last four track specimen identity
and "are not meaningful in the benchmark context", and they are exactly zero on every frame (checked here).

WHY THE TIMES ARE frame x 0.914 s. The release carries no per-frame clock; 0.914 s is the volume period measured
from the plane markers of the raw ephys file (zapbench.zarr .zattrs, `frame_interval_s`). Planes of one volume
are 12 ms apart over 0.851 s, so a neuron at depth z is sampled up to 0.85 s after one at the top: kept as a
known limitation, not corrected (it is constant per neuron).

    dff        [T, N] float32   dF/F, T = 7,879 frames, N = 71,721 neurons (column i = segment label i + 1)
    pos_um     [N, 3] float64   soma centroid x, y, z in um (the release's own centred coordinates)
    stimulus   [T, 22] float32  the known stimulus features (App. B.6), dims 0-21
    condition  [T] int8         the stimulus condition of each frame, 0..8 (CONDITION_NAMES order)
    offsets    [10] int64       CONDITION_OFFSETS: condition c spans offsets[c] .. offsets[c+1]
    names      [9] str          CONDITION_NAMES
    t_s        [T] float64      frame index x 0.914 s
"""
import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
from plexus.paths import graphs_data_path                              # noqa: E402

NAME = "zapbench"
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
REL = os.path.join(os.environ["GNN_OUTPUT_ROOT"], "graphs_data", "zebrafish", "zapbench", "release_20240930")
VOL = os.path.join(REL, "volumes", "20240930")
CONST = os.path.join(EXP, "papers", "zapbench_code", "zapbench", "constants.py")
T, N, CHUNK, N_STIM = 7879, 71721, 512, 22
FRAME_S = 0.914


def sha256_file(p, chunk=64 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_tree(d):
    """One hash over every file of a folder, in sorted relative-path order (path and bytes both hashed)."""
    h = hashlib.sha256()
    for dp, _, fs in sorted(os.walk(d)):
        for f in sorted(fs):
            p = os.path.join(dp, f)
            h.update(os.path.relpath(p, d).encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def constants():
    src = open(CONST).read()
    off = tuple(int(x) for x in re.search(r"CONDITION_OFFSETS = \(([^)]*)\)", src).group(1).split(","))
    names = tuple(re.findall(r"'([^']*)'", re.search(r"CONDITION_NAMES = \(([^)]*)\)", src, re.S).group(1)))
    return off, names


def read_traces():
    """The release traces zarr v3 [7879, 71721]: uncompressed little-endian float32 chunks of 512 x 512."""
    zj = json.load(open(os.path.join(VOL, "traces_unzipped", "traces", "zarr.json")))
    if zj["shape"] != [T, N] or zj["codecs"] != [{"configuration": {"endian": "little"}, "name": "bytes"}]:
        raise ValueError(f"unexpected traces layout: {zj['shape']}, {zj['codecs']}")
    root = os.path.join(VOL, "traces_unzipped", "traces", "c")
    out = np.empty((T, N), np.float32)
    for i in range(math.ceil(T / CHUNK)):
        r0, r1 = i * CHUNK, min(T, (i + 1) * CHUNK)
        for j in range(math.ceil(N / CHUNK)):
            c0, c1 = j * CHUNK, min(N, (j + 1) * CHUNK)
            blk = np.fromfile(os.path.join(root, str(i), str(j)), "<f4").reshape(CHUNK, CHUNK)
            out[r0:r1, c0:c1] = blk[:r1 - r0, :c1 - c0]
    return out


def main():
    t_all = time.time()
    off, names = constants()
    if off[-1] != T or len(names) != len(off) - 1:
        raise ValueError(f"CONDITION_OFFSETS {off} / names {names} do not describe {T} frames")
    dff = read_traces()
    if not np.isfinite(dff).all():
        raise ValueError("non-finite dF/F in the release traces")
    d = json.load(open(os.path.join(VOL, "segmentation", "dataframe_centroids.json")))
    lab = np.array([d["label"][str(i)] for i in range(N)])
    if not (lab == np.arange(1, N + 1)).all():
        raise ValueError("centroid rows are not labels 1..N in order")
    pos = np.array([[d[k][str(i)] for k in ("centroid_x", "centroid_y", "centroid_z")] for i in range(N)])
    import zarr
    S = zarr.open(os.path.join(VOL, "stimuli_features"), mode="r")[:]
    if S.shape != (T, 26) or np.abs(S[:, N_STIM:]).max() != 0:
        raise ValueError(f"stimulus {S.shape}: expected [T, 26] with dims 22-25 all zero")
    cond = np.zeros(T, np.int8)
    for c in range(len(names)):
        cond[off[c]:off[c + 1]] = c
    t_s = np.arange(T) * FRAME_S
    out = graphs_data_path("zebrafish", f"{NAME}_recording.npz")
    np.savez(out, dff=dff, pos_um=pos, stimulus=S[:, :N_STIM].astype(np.float32), condition=cond,
             offsets=np.array(off, np.int64), names=np.array(names), t_s=t_s)
    prov = {
        "name": NAME, "written": time.strftime("%Y-%m-%d %H:%M"), "script": "tools/export_zapbench_recording.py",
        "source": {
            "release": "gs://zapbench-release/volumes/20240930 (HTTPS mirror, MD5-checked per object)",
            "sha256_traces_zip": sha256_file(os.path.join(VOL, "traces.zip")),
            "sha256_stimuli_features": sha256_tree(os.path.join(VOL, "stimuli_features")),
            "sha256_centroids": sha256_file(os.path.join(VOL, "segmentation", "dataframe_centroids.json")),
            "sha256_constants_py": sha256_file(CONST),
        },
        "shape": {"frames": T, "neurons": N, "stimulus_dims": N_STIM},
        "frame_s": FRAME_S, "span_h": T * FRAME_S / 3600,
        "conditions": {names[c]: [off[c], off[c + 1]] for c in range(len(names))},
        "pos_extent_um": np.ptp(pos, 0).round(2).tolist(),
        "dff": {"min": float(dff.min()), "max": float(dff.max()), "mean": float(dff.mean()),
                "p1": float(np.percentile(dff[::8], 1)), "p99": float(np.percentile(dff[::8], 99))},
        "sha256_dff": hashlib.sha256(dff.tobytes()).hexdigest(),
        "sha256_pos": hashlib.sha256(pos.tobytes()).hexdigest(),
        "seconds": round(time.time() - t_all, 1),
    }
    json.dump(prov, open(out.replace(".npz", ".provenance.json"), "w"), indent=1)
    print(json.dumps(prov, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
