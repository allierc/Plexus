"""Fingerprints of working points: what a run looks like at a few checkpoints, small enough to commit.

A WORKING POINT is an archived run the group has looked at and accepted: `<root>/graphs_data/
<family>/<name>/` with `spec.yaml` and `trajectory.npz`. The trajectory is gigabytes and lives on a
mount; the FINGERPRINT is a few kilobytes of JSON in `tests/regression/fingerprints/` that says
what that run was at frames 0, N/4, N/2, 3N/4, N of a CUT of the first N frames -- cell count, shell
radius, roughness, area, deaths, the matrix's spread and penetration where there is one -- plus the
run's own frame time. `tests/regression/test_working_points.py` reruns the cut on the current code
and compares against the JSON within stated bands. tests/REGRESSION_PLAN.md says why each band is
the width it is.

Everything here is a function of a trajectory dict (the arrays in trajectory.npz) so the same code
fingerprints an archive and a fresh run, and the comparison is symmetric.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINGERPRINTS = os.path.join(ROOT, "tests", "regression", "fingerprints")
PYTHON = sys.executable

# THE BANDS. Relative unless stated. tests/REGRESSION_PLAN.md section 2 derives each one from a
# measured spread (two identical CUDA runs) and a measured regression (what it must not let through).
BANDS = {
    "cells": 0.03, "half_edges": 0.03, "deaths_or_net_loss": 0.03,
    # ndiv_sum (sum of per-cell generation numbers) is RECORDED, not asserted: 5-6% between two runs
    # of the same code on a dividing tissue; `cells` already guards division.
    "r_med": 0.01, "r_p10": 0.03, "r_p90": 0.03, "area_mean": 0.05,   # area: a batch of fresh daughters moves the mean 2-3%
    "roughness": 0.25, "z_sd": 0.25,
    "mpm_spread": 0.05, "mpm_bbox": 0.05, "mpm_inside": 0.05,
    "chem_max": 0.10, "chem_min": 0.10,
}
FRAME0_EXACT = ("cells", "half_edges")     # a seed that changed is a default that changed
# ABSOLUTE FLOORS for metrics whose reference can sit near zero. Roughness (std/mean of the vertex
# radius) is 0.006-0.02 on a healthy shell and swings 30-45% between an archive and a rerun that
# agree in everything else; the defect it guards is 0.0054 -> 0.077. A violation needs BOTH the
# relative band and this much absolute change.
ABS_FLOOR = {"roughness": 0.02, "z_sd": 0.05, "chem_min": 1e-3}   # 0.018 vs 0.026 between two reruns of mesh_mpm_spheroid_nominal at frame 150
TIME_RATCHET = 1.5                          # ms/frame on the SAME GPU model may grow this much


# ---------------------------------------------------------------------------------------------
# reading a trajectory
# ---------------------------------------------------------------------------------------------
def load_traj(path: str) -> dict:
    """The arrays of a trajectory.npz, lazily (np.load keeps the zip open)."""
    return np.load(path, allow_pickle=False)


def _occ(d, key, fr):
    o = d.get(key)
    return None if o is None else o[fr].astype(bool)


def checkpoint_metrics(d, fr: int) -> dict:
    """Everything the fingerprint records at one frame. Keys absent from the run are absent here."""
    m: dict = {}
    files = set(d.files)
    # ---- a vertex tissue --------------------------------------------------------------------
    if "vertex__pos" in files:
        occ = _occ(d, "vertex__occ", fr)
        P = d["vertex__pos"][fr]
        P = P[occ] if occ is not None else P
        if P.shape[0]:
            c = P.mean(0)
            r = np.linalg.norm(P - c, axis=1)
            m["r_med"] = float(np.median(r))
            m["r_p10"] = float(np.percentile(r, 10))
            m["r_p90"] = float(np.percentile(r, 90))
            m["roughness"] = float(r.std() / max(r.mean(), 1e-12))
            m["centroid"] = [float(v) for v in c]
            m["vertices"] = int(P.shape[0])
            # out-of-plane spread, the sheet's own number; on a shell it is just the radius
            m["z_sd"] = float(P[:, -1].std())
        if "cell__occ" in files:
            m["cells"] = int(d["cell__occ"][fr].sum())
        if "vertex__mesh_offsets" in files:
            off = d["vertex__mesh_offsets"]
            if fr + 1 < off.shape[0]:
                m["half_edges"] = int(off[fr + 1] - off[fr])   # live half-edges, from the mesh table
        if "cell__area" in files and "cell__occ" in files:
            a = d["cell__area"][fr].reshape(-1)
            co = d["cell__occ"][fr].astype(bool)
            a = a[co] if a.shape[0] == co.shape[0] else a
            if a.size:
                m["area_mean"] = float(a.mean())
        for k in ("cell__ndiv", "cell__age"):
            if k in files and "cell__occ" in files:
                v = d[k][fr].reshape(-1)
                co = d["cell__occ"][fr].astype(bool)
                v = v[co] if v.shape[0] == co.shape[0] else v
                if k == "cell__ndiv" and v.size:
                    m["ndiv_sum"] = int(v.sum())   # sum of per-cell generation counts, monotone
        for k in ("cell__chem", "vertex__chem"):
            if k in files:
                v = d[k][fr]
                oc = _occ(d, k.split("__")[0] + "__occ", fr)
                v = v[oc] if oc is not None and v.shape[0] == oc.shape[0] else v
                if v.size and float(np.abs(v).max()) > 0.0:   # an all-zero chem block is unused
                    m["chem_max"] = float(v[:, 0].max())
                    m["chem_min"] = float(v[:, 0].min())
    # ---- an MPM body -------------------------------------------------------------------------
    mp = [k[:-5] for k in files if k.endswith("__pos") and k != "vertex__pos"]
    if mp:
        mset = "mpm_particle" if "mpm_particle" in mp else sorted(mp)[0]
        m["mpm_set"] = mset
        occ = _occ(d, mset + "__occ", fr)
        X = d[mset + "__pos"][fr]
        X = X[occ] if occ is not None else X
        if X.shape[0]:
            m["mpm_n"] = int(X.shape[0])
            m["mpm_spread"] = float(X.std(0).mean())
            m["mpm_bbox"] = float((X.max(0) - X.min(0)).mean())
            m["mpm_centroid"] = [float(v) for v in X.mean(0)]
            if "vertex__pos" in files and "r_med" in m:
                c = np.asarray(m["centroid"])
                m["mpm_inside"] = int((np.linalg.norm(X - c, axis=1) < m["r_med"]).sum())
    return m


def deaths_by(d, fr: int) -> int | None:
    """Cells that have left the live set since frame 0, when the run has a cell set."""
    if "cell__occ" not in d.files:
        return None
    return int(max(0, int(d["cell__occ"][0].sum()) - int(d["cell__occ"][fr].sum())))


def fingerprint_traj(d, cut: int, spec: dict, meta: dict, checkpoints=None) -> dict:
    _posk = [k for k in d.files if k.endswith("__pos")]
    n_rec = int(d[_posk[0]].shape[0])
    N = min(cut, n_rec - 1)
    frames = sorted({0, N // 4, N // 2, (3 * N) // 4, N}) if checkpoints is None else sorted(int(f) for f in checkpoints if int(f) <= N)
    cps = {}
    for fr in frames:
        cm = checkpoint_metrics(d, fr)
        dd = deaths_by(d, fr)
        if dd is not None and "cells" in cm:
            cm["deaths_or_net_loss"] = dd
        cps[str(fr)] = cm
    fp = dict(meta)
    # RECORDED ROWS, NOT ENGINE FRAMES. The engine keeps <= record_cap set frames, thinning by
    # stride = (n_frames + cap) // cap; a 2000-frame run with record_cap 60 has rows 0, 34, 68 ...
    # A cut of C rows therefore means C * stride engine frames, and the rerun reproduces the
    # stride with record_cap = C + 1 (which gives the same stride whenever stride <= C + 1).
    g = spec.get("general", {})
    stride = max(1, (int(g.get("n_frames", N)) + int(g.get("record_cap", 10000))) // int(g.get("record_cap", 10000)))
    fp["cut"] = {"n_frames": N, "checkpoints": frames, "record_stride": stride, "engine_frames": N * stride}
    fp["spec_sha256"] = hashlib.sha256(yaml.safe_dump(spec, sort_keys=True).encode()).hexdigest()[:16]
    fp["checkpoints"] = cps
    if "frame_ms" in d.files:
        ms = np.asarray(d["frame_ms"], dtype=np.float64)
        lo, hi = min(20, max(N - 1, 1)), max(N, 2)
        fp["timing"] = {"ms_per_frame": float(np.median(ms[lo:hi])),
                        "frames": [lo, hi], "device": meta.get("device", "unknown")}
    return fp


def moves(fp: dict) -> bool:
    """A fingerprint whose checkpoints do not change over the cut cannot detect anything."""
    cps = [fp["checkpoints"][str(f)] for f in fp["cut"]["checkpoints"]]
    first, last = cps[0], cps[-1]
    for k in ("cells", "r_med", "deaths_or_net_loss", "mpm_spread", "area_mean", "chem_max"):
        if k in first and k in last and abs(float(last[k]) - float(first[k])) > 1e-3 * max(abs(float(first[k])), 1e-9):
            return True
    return False


# ---------------------------------------------------------------------------------------------
# an archive -> a fingerprint
# ---------------------------------------------------------------------------------------------
def fingerprint_archive(folder: str, cut: int, commit: str | None = None) -> dict:
    spec = yaml.safe_load(open(os.path.join(folder, "spec.yaml")))
    d = load_traj(os.path.join(folder, "trajectory.npz"))
    name = spec["general"]["name"]
    meta = {
        "name": name,
        "family": os.path.basename(os.path.dirname(os.path.abspath(folder))),
        "archive": os.path.abspath(folder),
        "archived_on": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(os.path.join(folder, "trajectory.npz")))),
        "generated_at_commit": commit or git_head(),
        "device": "archive",
        "because": "registered from the accepted archive",
    }
    return fingerprint_traj(d, cut, spec, meta)


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------------------------
# a fresh cut of a spec on the current code
# ---------------------------------------------------------------------------------------------
def _add_missing_blocks(spec_path: str, log_path: str) -> bool:
    """A spec archived before a block was introduced: add `{width: 1, record: false}` and retry.
    Returns True if something was added."""
    import re
    log = open(log_path, errors="ignore").read()
    m = re.search(r"per-cell block '([A-Za-z0-9_]+)', which the set '([a-z_]+)' does not declare", log)
    if not m:
        m2 = re.search(r"^KeyError: '([A-Za-z0-9_]+)'", log, re.M)
        if not m2:
            return False
        block, setname = m2.group(1), "cell"
    else:
        block, setname = m.group(1), m.group(2)
    s = yaml.safe_load(open(spec_path))
    st = s["sets"].get(setname, {}).setdefault("state", {})
    if block in st:
        return False
    st[block] = {"width": 1, "record": False}
    yaml.safe_dump(s, open(spec_path, "w"), sort_keys=False)
    return True


def run_cut(spec: dict, family: str, cut: int, device: str, keep: bool = False, stride: int = 1) -> tuple[str, str]:
    """Run `spec` for `cut` frames through Plexus_Main into a temporary output root.
    Returns (trajectory.npz path, output root). The spec is written under config/<family>/ with a
    `_regression_` prefix for the duration of the run and removed afterwards."""
    s = dict(spec)
    s["general"] = dict(spec["general"], n_frames=cut * stride, record_cap=(cut + 1) if stride > 1 else cut + 2)
    name = f"_regression_{s['general']['name']}"
    s["general"]["name"] = name
    spec_path = os.path.join(ROOT, "config", family, name + ".yaml")
    root = tempfile.mkdtemp(prefix="plexus_regression_")
    yaml.safe_dump(s, open(spec_path, "w"), sort_keys=False)
    log = os.path.join(root, "run.log")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), GNN_OUTPUT_ROOT=root)
    try:
        for _attempt in range(6):
            with open(log, "w") as lf:
                rc = subprocess.run([PYTHON, os.path.join(ROOT, "Plexus_Main.py"), "-o", "generate",
                                     f"{family}/{name}", "--device", device, "--force", "--no-viz",
                                     "--no-describe"], cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            traj = os.path.join(root, "graphs_data", family, name, "trajectory.npz")
            if rc == 0 and os.path.exists(traj):
                return traj, root
            if not _add_missing_blocks(spec_path, log):
                break
        tail = open(log, errors="ignore").read()[-1500:]
        raise RuntimeError(f"regression cut of {spec['general']['name']} did not produce a trajectory:\n{tail}")
    finally:
        if not keep and os.path.exists(spec_path):
            os.remove(spec_path)


def fingerprint_fresh(spec: dict, family: str, cut: int, device: str, because: str, checkpoints=None, stride: int = 1) -> dict:
    traj, root = run_cut(spec, family, cut, device, stride=stride)
    try:
        d = load_traj(traj)
        meta = {"name": spec["general"]["name"], "family": family, "archive": None,
                "archived_on": time.strftime("%Y-%m-%d %H:%M"), "generated_at_commit": git_head(),
                "device": gpu_name(device), "because": because}
        return fingerprint_traj(d, cut, spec, meta, checkpoints=checkpoints)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def gpu_name(device: str) -> str:
    try:
        import torch
        if str(device).startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.get_device_name(int(str(device).split(":")[-1]) if ":" in str(device) else 0)
    except Exception:                                    # noqa: BLE001
        pass
    return "cpu"


# ---------------------------------------------------------------------------------------------
# comparing
# ---------------------------------------------------------------------------------------------
def compare(ref: dict, got: dict) -> list[str]:
    """Every checkpoint metric of `got` against `ref`. Returns the list of violations (empty = pass).
    A metric present in the reference and absent in the run is a violation: a block that stopped
    being recorded is a change."""
    bad = []
    for fr in ref["cut"]["checkpoints"]:
        key = str(fr)
        r, g = ref["checkpoints"][key], got["checkpoints"].get(key)
        if g is None:
            bad.append(f"frame {fr}: missing from the run"); continue
        for k, rv in r.items():
            if k not in BANDS and k not in FRAME0_EXACT:
                continue
            if k not in g:
                bad.append(f"frame {fr}: {k} not recorded by the run (reference {rv})"); continue
            gv = g[k]
            if fr == 0 and k in FRAME0_EXACT:
                if int(gv) != int(rv):
                    bad.append(f"frame 0: {k} {gv} != {rv} (exact: the seed changed)")
                continue
            tol = BANDS[k]
            scale = max(abs(float(rv)), 1e-9)
            if k in ("deaths_or_net_loss", "ndiv_sum"):
                # COUNTS OF EVENTS ARE RELATIVE TO THE POPULATION, not to themselves: at death
                # onset 314 vs 325 deaths is 11 cells of 2000, and the onset frame is where two
                # identical CUDA runs differ most.
                scale = max(float(r.get("cells", ref["checkpoints"]["0"].get("cells", rv))), 1.0)   # the population at THIS frame
            err = abs(float(gv) - float(rv)) / scale
            if err > tol and abs(float(gv) - float(rv)) > ABS_FLOOR.get(k, 0.0):
                bad.append(f"frame {fr}: {k} {gv:.6g} vs {rv:.6g} ({err*100:.1f}% > {tol*100:.0f}%)")
    return bad


def compare_timing(ref: dict, got: dict) -> tuple[str | None, str]:
    """(violation or None, note). Asserted only on the same GPU model; otherwise reported."""
    rt, gt = ref.get("timing"), got.get("timing")
    if not rt or not gt:
        return None, "no timing on one side"
    note = f"ms/frame {gt['ms_per_frame']:.1f} on {gt['device']} vs {rt['ms_per_frame']:.1f} on {rt['device']}"
    if rt.get("device") in ("archive", "unknown", None) or rt["device"] != gt["device"]:
        return None, note + " (different device: not asserted)"
    if gt["ms_per_frame"] > TIME_RATCHET * rt["ms_per_frame"]:
        return f"frame time {gt['ms_per_frame']:.1f} ms > {TIME_RATCHET}x {rt['ms_per_frame']:.1f} ms on {gt['device']}", note
    return None, note


def read_fingerprint(name: str) -> dict:
    return json.load(open(os.path.join(FINGERPRINTS, name + ".json")))


def write_fingerprint(fp: dict) -> str:
    os.makedirs(FINGERPRINTS, exist_ok=True)
    p = os.path.join(FINGERPRINTS, fp["name"] + ".json")
    json.dump(fp, open(p, "w"), indent=1, sort_keys=True)
    return p


def registered() -> list[str]:
    if not os.path.isdir(FINGERPRINTS):
        return []
    return sorted(f[:-5] for f in os.listdir(FINGERPRINTS) if f.endswith(".json"))
