#!/usr/bin/env python
"""Score a rendered run against a stated target, 0-10, with an external agent and a ruler.

    PYTHONPATH=src python tools/judge.py <group>/<spec> --target "..." [--measure left_right_size]

    <5    out of scope
    5-8   the right direction, not acceptable
    8-10  acceptable

TWO VERDICTS, AND THE RULER IS THE ARBITER. The agent is the local VLM (`VLLM/describe_video.py`'s
model), shown the run's own stills and asked for a number and a reason. It is the right judge for
"does this LOOK like the target", which is the thing a measurement cannot ask -- a spec can have a
correct left/right size ratio and still render as an unreadable smear.

But a VLM score alone is soft, and this repository has already written down why: the VLM RECORDS
DISAGREEMENTS, IT NEVER VETOES, and the arbiter is a measurement. So `--measure` computes the
quantity the target is actually about, from the trajectory, and both go in the table. Where they
disagree that is the finding, not a tie to be broken.

`--measure left_right_size` is the one this experiment needs: the mean nearest-neighbour spacing of
the cells in the left half of the disc against the right half. Spacing is the honest proxy for
"how big is a cell" in a packed sheet -- it is what the eye reads as size, and unlike a declared
radius it is a property of the RESULT rather than of the request.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

RUBRIC = """You are scoring a scientific simulation against a stated target.

TARGET: {target}

Give your answer in exactly this form, nothing else:

SCORE: <integer 0-10>
REASON: <one sentence naming what you see that earns or costs the score>

The scale is:
  0-4   out of scope -- the image does not show the target's subject at all
  5-8   the right direction, but not acceptable -- the subject is there and the
        stated property is absent, too weak, or in the wrong place
  9-10  acceptable -- the stated property is clearly and unambiguously present

Judge ONLY the stated target. Do not reward a picture for being pretty, dense or
well rendered if the stated property is not visible."""


def measure_left_right_size(run_dir):
    """Mean nearest-neighbour spacing, left half against right half, from the last frame.

    SPACING AND NOT A DECLARED RADIUS. What the eye calls "a bigger cell" in a packed sheet is a
    larger distance to its neighbours; a radius written in the spec is the REQUEST, and the whole
    point of running the engine is to find out whether the request became the result.
    """
    tj = os.path.join(run_dir, "trajectory.npz")
    if not os.path.isfile(tj):
        return {}
    z = np.load(tj)
    key = next((k for k in z.files if k.endswith("__pos")), None)
    if key is None:
        return {}
    p = np.asarray(z[key], float)
    p = p[-1] if p.ndim == 3 else p                     # the last recorded frame
    live = np.isfinite(p).all(1)
    p = p[live][:, :2]
    if len(p) < 8:
        return {}
    mid = 0.5 * (p[:, 0].min() + p[:, 0].max())
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    left, right = p[:, 0] < mid, p[:, 0] >= mid
    l, r = float(np.median(nn[left])), float(np.median(nn[right]))
    return {"spacing_left": round(l, 4), "spacing_right": round(r, 4),
            "ratio_left_over_right": round(l / max(r, 1e-12), 3),
            "n_left": int(left.sum()), "n_right": int(right.sum())}


def measure_left_right_area(run_dir):
    """Median cell AREA, left half of the disc against the right half, at the last frame.

    THE VERTEX MODEL'S OWN SIZE. A cell here is a polygon, and `cell_geometry` writes its area
    into the `area` block every frame -- so size is measured, not inferred from how far apart
    points sit. The split is at the midpoint of the live centroids' x range, and only live cells
    count: the cell buffer is larger than the sheet, and its dormant slots carry area 0.

    Both ratios are reported because "twice as large" is ambiguous between area and linear size:
    `area_ratio` is the one the experiment is accepted on, `linear_ratio` its square root.
    """
    tj = os.path.join(run_dir, "trajectory.npz")
    if not os.path.isfile(tj):
        return {}
    z = np.load(tj)
    if "cell__area" not in z.files or "cell__centroid" not in z.files:
        return {"error": f"no cell__area / cell__centroid in {sorted(z.files)[:12]}"}
    a = np.asarray(z["cell__area"], float)
    c = np.asarray(z["cell__centroid"], float)
    a, c = a[-1, :, 0], c[-1]
    live = np.isfinite(a) & (a > 0)
    if "cell__occ" in z.files:
        live &= np.asarray(z["cell__occ"])[-1] > 0
    a, x = a[live], c[live, 0]
    if len(a) < 8:
        return {"error": f"only {len(a)} live cells"}
    mid = 0.5 * (x.min() + x.max())
    L, R = a[x < mid], a[x >= mid]
    al, ar = float(np.median(L)), float(np.median(R))
    return {"area_left": round(al, 5), "area_right": round(ar, 5),
            "area_ratio": round(al / max(ar, 1e-12), 3),
            "linear_ratio": round(float(np.sqrt(al / max(ar, 1e-12))), 3),
            "n_live": int(live.sum()), "n_left": int(len(L)), "n_right": int(len(R))}


def measure_area_smooth(run_dir):
    """Mean cell AREA in um^2 at the last frame, the spread, and how much each cell FLICKERS.

    AREA IN um^2, NOT BOX UNITS. The spec's `general.units.length_um` converts: 1 box unit of area
    is length_um^2 um^2 (100 um^2 at the demo sheets' 10 um). A spec without units is reported in
    box units, and `units` in the record says which.

    FLICKER IS A CELL'S AREA WOBBLING ABOUT ITS OWN TREND. Each cell's area is smoothed by an
    11-frame moving average; the jitter is the RMS of the residual, as a fraction of the cell's
    area, taken over the LAST HALF of the run -- when a relaxation should be over -- and the median
    over cells is reported. A smooth relaxation scores ~0 however large the change it makes; a
    sheet that settles in 20 frames and then shakes by 2% a frame scores 0.02. `reversal_frac` is
    the companion reading: of the frame steps larger than 0.1% of the area, the fraction that
    reverse the previous step's direction (~0.5 is noise, ~0 is a monotone trajectory).

    Only cells alive at frame 0 AND at the last frame are followed, by slot index, so a division
    or a death cannot splice two cells into one trace.

    PERIMETER beside area: `perim_mean_last` (um), `perim_cv_last` (cell-to-cell SD / mean), and
    `perim_mean_over_target` against the recorded target `P0` = p0 sqrt(A0). ROUNDNESS as the
    mean shape index P / sqrt(A) (circle 3.545, regular hexagon 3.722) and the share of six-sided
    cells, first and last frame.
    """
    tj = os.path.join(run_dir, "trajectory.npz")
    if not os.path.isfile(tj):
        return {}
    z = np.load(tj)
    if "cell__area" not in z.files:
        return {"error": f"no cell__area in {sorted(z.files)[:12]}"}
    a = np.asarray(z["cell__area"], float)[:, :, 0]
    occ = np.asarray(z["cell__occ"]) > 0 if "cell__occ" in z.files else np.isfinite(a) & (a > 0)
    k, units = 1.0, "box"
    try:
        import yaml
        _u = (yaml.safe_load(open(os.path.join(run_dir, "spec.yaml"))).get("general") or {}).get("units") or {}
        if _u.get("length_um"):
            k, units = float(_u["length_um"]) ** 2, "um2"
    except Exception:                                            # noqa: BLE001
        pass
    a = a * k
    last = a[-1][occ[-1]]
    keep = occ[0] & occ[-1] & np.all(np.isfinite(a), axis=0) & np.all(a > 0, axis=0)
    A = a[:, keep]
    T = A.shape[0]
    h = T // 2
    w = 11
    ker = np.ones(w) / w
    sm = np.stack([np.convolve(A[:, i], ker, mode="same") for i in range(A.shape[1])], 1)
    res = (A - sm)[h + w // 2: T - w // 2]
    jit = np.sqrt(np.mean(res ** 2, 0)) / np.mean(A[h:], 0)
    dA = np.diff(A[h:], axis=0)
    big = np.abs(dA) > 1e-3 * A[h:-1]
    both = big[1:] & big[:-1]
    rev = (np.sign(dA[1:]) != np.sign(dA[:-1])) & both
    # PERIMETER, FROM THE MESH -- it is not a recorded block. The half-edge table of each frame is
    # sliced out by `vertex__mesh_offsets`; a cell's perimeter is the summed length of the
    # half-edges whose face it is. `P0` IS recorded, so the target is read back, not re-derived.
    per = {}
    if all(k in z.files for k in ("vertex__pos", "vertex__mesh_offsets", "vertex__mesh_E_srce",
                                  "vertex__mesh_E_trgt", "vertex__mesh_E_face")):
        L = float(k ** 0.5) if units == "um2" else 1.0
        off = np.asarray(z["vertex__mesh_offsets"])
        es_, et_, ef_ = z["vertex__mesh_E_srce"], z["vertex__mesh_E_trgt"], z["vertex__mesh_E_face"]
        pos = z["vertex__pos"]

        def _perim(t):
            s_, e_, f_ = (np.asarray(v[off[t]:off[t + 1]]) for v in (es_, et_, ef_))
            x = np.asarray(pos[t], float)
            ok = f_ >= 0
            return np.bincount(f_[ok], weights=np.linalg.norm(x[e_[ok]] - x[s_[ok]], axis=1),
                               minlength=occ.shape[1])[:occ.shape[1]] * L
        p_first, p_last = _perim(0)[occ[0]], _perim(T - 1)[occ[-1]]

        # ROUNDNESS: the shape index q = P / sqrt(A), dimensionless (a circle is 3.545, a regular
        # hexagon 3.722, a regular pentagon 3.812), and the share of six-sided cells -- the number
        # of half-edges whose face a cell is. Both at the first and the last frame.
        def _sides(t):
            f_ = np.asarray(ef_[off[t]:off[t + 1]])
            return np.bincount(f_[f_ >= 0], minlength=occ.shape[1])[:occ.shape[1]]
        q_first = _perim(0)[occ[0]] / np.sqrt(a[0][occ[0]])
        q_last = p_last / np.sqrt(a[-1][occ[-1]])
        n_last, n_first = _sides(T - 1)[occ[-1]], _sides(0)[occ[0]]
        per.update({"shape_index_first": round(float(q_first.mean()), 4),
                    "shape_index_last": round(float(q_last.mean()), 4),
                    "shape_index_p90_last": round(float(np.percentile(q_last, 90)), 4),
                    "hex_frac_first": round(float(np.mean(n_first == 6)), 3),
                    "hex_frac_last": round(float(np.mean(n_last == 6)), 3)})
        per.update({"perim_mean_last": round(float(p_last.mean()), 3),
               "perim_cv_last": round(float(p_last.std() / p_last.mean()), 4),
               "perim_cv_first": round(float(p_first.std() / p_first.mean()), 4)})
        if "cell__P0" in z.files:
            p0t = float(np.mean(np.asarray(z["cell__P0"], float)[-1, :, 0][occ[-1]])) * L
            per["perim_target"] = round(p0t, 3)
            per["perim_mean_over_target"] = round(float(p_last.mean()) / max(p0t, 1e-12), 4)
    mean_t = [float(np.mean(a[t][occ[t]])) for t in range(T)]
    cv_t = [float(np.std(a[t][occ[t]]) / np.mean(a[t][occ[t]])) for t in range(T)]
    return {"units": units,
            "mean_area_last": round(float(last.mean()), 3),
            "sd_area_last": round(float(last.std()), 3),
            "cv_last": round(float(last.std() / last.mean()), 4),
            "cv_first": round(cv_t[0], 4),
            "mean_area_first": round(mean_t[0], 3),
            "jitter_median": round(float(np.median(jit)), 5),
            "jitter_p90": round(float(np.percentile(jit, 90)), 5),
            "reversal_frac": round(float(rev.sum() / max(both.sum(), 1)), 3),
            **per,
            "n_followed": int(keep.sum()), "n_last": int(occ[-1].sum()), "frames": int(T)}


def _movie_facts(mp4):
    """(duration s, fps, frames) of an mp4 read by the ffmpeg the env ships, or {} without one."""
    import re
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                            # noqa: BLE001
        return {}
    r = subprocess.run([exe, "-i", mp4], capture_output=True, text=True)
    txt = r.stderr or ""
    out = {}
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", txt)
    if m:
        out["duration_s"] = round(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)), 2)
    m = re.search(r"(\d+(?:\.\d+)?) fps", txt)
    if m:
        out["fps"] = float(m.group(1))
    if "duration_s" in out and "fps" in out:
        out["frames"] = int(round(out["duration_s"] * out["fps"]))
    return out


# the channel of a run, from the name of its first protein set (tools/channel_spec.py names them)
_CHANNEL_OF = {"MscS": "mscs", "MscL": "mscl", "Kv12": "kv12", "alphahemolysin": "ahl", "aHL": "ahl",
               # the ion-flow rig's set names (channel_spec.build_flow: the channel's name, six letters)
               "alphah": "ahl", "KcsA": "kcsa", "gramic": "gramicidin", "GLIC": "glic", "NavAb": "navab",
               "TRAAK": "traak", "OmpF": "ompf"}


def measure_channel(run_dir):
    """Experiment 4's ruler: the movie as a file, the render's soundness, and the channel's own
    quantities from the trajectory, each converted through the SPEC'S declared units and set
    against the channel's bands in the experiment's front matter.

    THE MOVIE: duration, frame rate and frame count read from the mp4 itself (not the spec's
    request), and the render log's last "shells closed N of M" (the job's cluster.out).

    THE CHANNEL (whichever the run records): the pore's diameter (2 x `cell.pore_r`) closed (the
    median of the first 5% of the recorded frames) and at its widest (the 95th percentile);
    the tension the membrane carries (`cell.tension`, mN/m) and the tension at which the pore
    first reaches half-way from closed to its widest on the way up -- the gating midpoint, read
    off the run, never fitted; the current (`cell.j_channel`, pA) and the conductance |I / psi|
    closed and at its largest; the bilayer's area modulus as the slope of tension on area strain
    over the ramp's first third (before anything opens).

    INVARIANTS: non-finite positions, and the largest single-record jump of any protein bead in
    nm. A blow-up is reported first.
    """
    import re
    import yaml
    tj = os.path.join(run_dir, "trajectory.npz")
    spec_p = os.path.join(run_dir, "spec.yaml")
    out = {}
    mp4 = os.path.join(run_dir, "movie.mp4")
    if os.path.isfile(mp4):
        out["movie"] = _movie_facts(mp4)
    name = os.path.basename(run_dir.rstrip("/"))
    co = os.path.join(ROOT, "log", "experiments", "exp04", name, "cluster.out")
    if os.path.isfile(co):
        txt = open(co, errors="ignore").read()
        sh = re.findall(r"shells closed: (\d+) of (\d+)", txt)
        if sh:
            out["shells_closed"] = f"{sh[-1][0]} of {sh[-1][1]}"
        w = re.search(r"Run time :\s+(\d+) sec", txt)
        if w:
            out["wall_s"] = int(w.group(1))
        if "Traceback" in txt:
            # A TRACEBACK IN THE CAPTION STEP IS NOT A CRASHED RUN. The generate job describes its
            # movie after the simulation has written it; that step can fail on its own (two describe
            # steps loading the captioner on one GPU ran out of memory on H1/H3, 2026-09-26) while
            # the trajectory and the movie are whole. Only a traceback outside it is the run's.
            tb = txt[txt.rfind("Traceback"):][:400]
            out["caption_error" if "describe_video.py" in tb else "error"] = tb
    if not os.path.isfile(tj) or not os.path.isfile(spec_p):
        return out
    spec = yaml.safe_load(open(spec_p))
    u = (spec.get("general") or {}).get("units") or {}
    L_um, T_s, F_nN = float(u["length_um"]), float(u["time_s"]), float(u["force_nN"])
    z = np.load(tj)
    prot = [s for s in spec["sets"] if "_" in s and s.split("_")[0] in _CHANNEL_OF and s.split("_")[-1].isdigit()]
    ch = _CHANNEL_OF.get(prot[0].split("_")[0]) if prot else None
    if ch is None and {"K", "Cl"} <= set(spec["sets"]):
        ch = "hole"                                                  # the ion-flow test: a membrane with a hole
    out["channel"] = ch

    def blk(b):
        k = f"cell__{b}"
        return np.asarray(z[k], float)[:, 0, 0] if k in z.files else None
    e = 1.602176634e-19
    volt_per_sim = F_nN * 1e-9 * L_um * 1e-6 / e
    psi = blk("psi"); ten = blk("tension"); pr = blk("pore_r"); I = blk("j_channel"); strain = blk("strain")
    n = len(pr) if pr is not None else 0
    if n:
        D = 2.0 * pr * L_um * 1e3                                   # nm
        k5 = max(1, n // 20)
        out["pore_diameter_closed_nm"] = round(float(np.median(D[:k5])), 3)
        out["pore_diameter_max_nm"] = round(float(np.percentile(D, 95)), 3)
        out["pore_opened_by_nm"] = round(out["pore_diameter_max_nm"] - out["pore_diameter_closed_nm"], 3)
    if ten is not None:
        tau = ten * F_nN / L_um                                     # nN/um = mN/m
        out["tension_max_mN_m"] = round(float(np.percentile(tau, 99)), 2)
        if n and out.get("pore_opened_by_nm", 0) > 0.2:
            half = 0.5 * (out["pore_diameter_closed_nm"] + out["pore_diameter_max_nm"])
            k = int(np.argmax(D >= half))
            w = max(1, n // 50)
            out["half_open_tension_mN_m"] = round(float(np.median(tau[max(0, k - w):k + w + 1])), 2)
            out["half_open_record"] = k
        if strain is not None and n > 10:
            sel = slice(n // 20, n // 3)
            if np.ptp(strain[sel]) > 1e-6:
                out["K_A_measured_mN_m"] = round(float(np.polyfit(strain[sel], tau[sel], 1)[0]), 1)
    if I is not None and psi is not None:
        I_pA = I * e / T_s * 1e12
        V = psi * volt_per_sim
        G = np.abs(I_pA * 1e-12 / np.where(np.abs(V) > 1e-9, V, np.nan)) * 1e9   # nS
        k5 = max(1, len(G) // 20)
        out["psi_mV"] = round(float(np.median(V)) * 1e3, 1)
        out["current_max_pA"] = round(float(np.nanpercentile(np.abs(I_pA), 95)), 2)
        out["conductance_closed_nS"] = round(float(np.nanmedian(G[:k5])), 4)
        out["conductance_max_nS"] = round(float(np.nanpercentile(G, 95)), 4)
    # KV1.2: the conductance from the NET count of ions across (flux_counter's cumulative `n_crossed`), not
    # from the running current, whose one-ion spikes read as 56 pA (round 4's judges); and the filter's
    # occupancy, the mean of `n_filter`
    ncr, nfl = blk("n_crossed"), blk("n_filter")
    if ncr is not None and psi is not None and len(ncr) > 1:
        g_ = spec.get("general") or {}
        T_run = float(g_.get("n_frames", 0)) * float(g_.get("dt", 0.0)) * T_s
        V_ = abs(float(np.median(psi)) * volt_per_sim)
        if T_run > 0 and V_ > 0:
            out["net_ions_across"] = round(float(ncr[-1]), 1)
            out["sim_time_ns"] = round(T_run * 1e9, 2)
            out["conductance_pS"] = round(abs(float(ncr[-1])) * e / T_run / V_ * 1e12, 2)
    if nfl is not None:
        out["filter_occupancy_mean"] = round(float(np.mean(nfl)), 2)
    # THE FLOW, COUNTED: ions started half on each side of a closed membrane; `compartment_count`'s charge
    # moved into the cell over the run is the net current, set against the continuum conductance the same
    # geometry has (the electrolyte solver's own, `g_channel` in pS)
    qin, nKin, nClin, gch = blk("q_in"), blk("nK_in"), blk("nCl_in"), blk("g_channel")
    if nKin is None and blk("nNa_in") is not None:                    # the sodium channel's rig runs Na+
        nKin = blk("nNa_in")
        out["cation"] = "Na"
    if qin is not None and psi is not None and len(qin) > 1:
        g_ = spec.get("general") or {}
        T_run = float(g_.get("n_frames", 0)) * float(g_.get("dt", 0.0)) * T_s
        V_ = abs(float(np.median(psi)) * volt_per_sim)
        out["charge_moved_in_e"] = round(float(qin[-1]), 1)
        if nKin is not None:
            out["K_net_in"] = int(round(float(nKin[-1] - nKin[0])))
        if nClin is not None:
            out["Cl_net_in"] = int(round(float(nClin[-1] - nClin[0])))
        if T_run > 0 and V_ > 0:
            out["sim_time_ns"] = round(T_run * 1e9, 2)
            out["conductance_counted_nS"] = round(abs(float(qin[-1])) * e / T_run / V_ * 1e9, 4)
        if gch is not None:
            out["conductance_continuum_nS"] = round(float(gch[0]) / 1000.0, 4)
            if out.get("conductance_continuum_nS", 0) > 0 and "conductance_counted_nS" in out:
                out["flow_ratio_counted_over_continuum"] = round(out["conductance_counted_nS"] / out["conductance_continuum_nS"], 3)
    # THE MOVING GATE (batch 11): the open basin's weight the elastic network publishes -- its largest value,
    # its value over the run's last tenth, and the record where it first passed one half
    wo = blk("w_open")
    if wo is not None and len(wo) > 5:
        out["gate_open_weight_max"] = round(float(np.nanmax(wo)), 3)
        out["gate_open_weight_end"] = round(float(np.nanmedian(wo[-max(1, len(wo) // 10):])), 3)
        k_ = np.argmax(wo > 0.5)
        out["gate_opened_at_record"] = int(k_) if wo[k_] > 0.5 else None
        if ten is not None and wo[k_] > 0.5:
            out["tension_at_opening_mN_m"] = round(float(ten[k_] * F_nN / L_um), 2)
    # THROUGH THE CORE, from the trajectory: each ion's side is set only when it is clear of the membrane's core
    # (below zc - h: inside; above zc + h: outside) and kept while it is in between, so an ion that wanders
    # into a pore's mouth and back is not a transit. Net charge carried from outside to inside over the run.
    # (compartment_count's mid-plane counted 10g's ions sitting in the prepore's stems as "inside".)
    sb = next((o for o in (spec.get("operators") or []) if o.get("op") == "slab_barrier"), None)
    if sb is not None and psi is not None:
        z0_, h_ = float(sb["z0"]), float(sb["half_thickness"])
        tot_q, per = 0.0, {}
        for sp_, qq_ in (("K", 1.0), ("Na", 1.0), ("Cl", -1.0)):
            k_ = f"{sp_}__pos"
            if k_ not in z.files:
                continue
            Z = np.asarray(z[k_], float)[:, :, 2]                     # [records, ions]
            side = np.where(Z < z0_ - h_, -1, np.where(Z > z0_ + h_, 1, 0))
            last = side[0].copy()
            n_in = 0
            for r_ in range(1, side.shape[0]):
                s_ = side[r_]
                moved = (s_ != 0) & (s_ != last) & (last != 0)
                n_in += int(((last == 1) & (s_ == -1) & moved).sum()) - int(((last == -1) & (s_ == 1) & moved).sum())
                last = np.where(s_ != 0, s_, last)
            per[sp_] = n_in
            tot_q += qq_ * n_in
            # WAS EACH TRANSIT THROUGH A PATH? An ion that crossed but, inside the core, stood further than a wall's
            # width (0.1 nm) outside every measured path's radius at its height went round the barrier: a LEAK
            # (batch 10: the barrier had no force along a path's slope). Counted apart, and out of the flow.
            pores_ = [np.asarray(po["profile"], float) for po in (sb.get("pores") or [])]
            if pores_:
                # EACH CROSSING ON ITS OWN RECORDS, and a leak only when the ion stood outside every path for TWO
                # records running. The test read every record an ion ever spent in the core, the whole run long, and
                # one sample off the path condemned it: 15e's three K+ walked single-file down Kv1.2's filter
                # (10.2 -> 9.9 -> 9.5 -> 9.2 nm) and were filed as leaks -- one had a record 0.36 nm off the axis where
                # the table allows 0.13, a place the hard core returns an ion from within a frame. A way round the
                # barrier takes more than one record; a sampled excursion does not.
                def _out(X_, r_):
                    zz = X_[r_, 2]
                    return min(np.hypot(X_[r_, 0] - p_[np.argmin(np.abs(p_[:, 0] - zz)), 4],
                                        X_[r_, 1] - p_[np.argmin(np.abs(p_[:, 0] - zz)), 5])
                               - p_[np.argmin(np.abs(p_[:, 0] - zz)), 1] for p_ in pores_) * L_um * 1e3 > 0.1
                lk, clean_net = 0, 0
                for i_ in range(Z.shape[1]):
                    s_i = side[:, i_]
                    nzr = np.where(s_i != 0)[0]
                    if len(nzr) < 2 or len(set(s_i[nzr])) < 2:
                        continue
                    X_ = np.asarray(z[k_], float)[:, i_, :]
                    for a_, b_ in zip(nzr[:-1], nzr[1:]):
                        if s_i[a_] == s_i[b_]:
                            continue                                  # in and back out the same face: no crossing
                        way = 1 if s_i[a_] > s_i[b_] else -1          # outside (above) -> inside (below) positive
                        flags = [_out(X_, r_) for r_ in range(a_ + 1, b_)]
                        if any(u_ and v_ for u_, v_ in zip(flags[:-1], flags[1:])):
                            lk += 1
                        else:
                            clean_net += way
                out.setdefault("leak_transits", {})[sp_] = lk
                out.setdefault("clean_net_in", {})[sp_] = clean_net
        if per:
            out["through_core_net_in"] = per
            out["charge_through_core_e"] = tot_q
            if "clean_net_in" in out:
                qc = sum((-1.0 if sp_ == "Cl" else 1.0) * v for sp_, v in out["clean_net_in"].items())
                out["clean_charge_e"] = qc
            # AN ION'S LARGEST JUMP BETWEEN RECORDS, against one grid cell (the rubric's invariant)
            jmax = 0.0
            for sp_ in per:
                Pz = np.asarray(z[f"{sp_}__pos"], float)
                if Pz.shape[0] > 1:
                    jmax = max(jmax, float(np.nanmax(np.linalg.norm(np.diff(Pz, axis=0), axis=2))))
            # A RECORD IS MANY FRAMES (300,000 frames kept as 3,001 records: 100 frames, 15.6 ps), so the largest
            # jump between records is diffusion's extreme, not a frame's: flagged only past six standard
            # deviations of a free ion's 3D displacement over one record (D 2.03e-9 m^2/s, the faster ion)
            g_ = spec.get("general") or {}
            nrec = max(1, Pz.shape[0] - 1)
            t_rec = float(g_.get("n_frames", 0)) * float(g_.get("dt", 0.0)) * T_s / nrec
            sig_nm = math.sqrt(2.0 * 2.03e-9 * t_rec) * 1e9 if t_rec > 0 else 0.0
            out["ion_max_jump_nm_per_record"] = round(jmax * L_um * 1e3, 3)
            out["ion_jump_diffusive_6sd_nm"] = round(6.0 * math.sqrt(3.0) * sig_nm, 3)
            out["ion_jump_beyond_diffusion"] = bool(jmax * L_um * 1e3 > 6.0 * math.sqrt(3.0) * sig_nm)
            g_ = spec.get("general") or {}
            T_run = float(g_.get("n_frames", 0)) * float(g_.get("dt", 0.0)) * T_s
            V_ = abs(float(np.median(psi)) * volt_per_sim)
            if T_run > 0 and V_ > 0:
                out["conductance_through_core_pS"] = round(abs(tot_q) * e / T_run / V_ * 1e12, 1)
                if "clean_charge_e" in out:
                    out["conductance_clean_pS"] = round(abs(out["clean_charge_e"]) * e / T_run / V_ * 1e12, 1)
                    out["conductance_clean_nS"] = round(out["conductance_clean_pS"] / 1000.0, 4)
    # THE ION-FLOW RIG (channel_spec.build_flow): the deposited state it holds and what its ion path is --
    # the narrowest radius in the core and whether a dry (hydrophobic, too narrow) gate sits on it -- from the
    # generator's own record of the spec; and the flow's make-up: K+ in against Cl- out, in pS as well
    pj = os.path.join(ROOT, "config", "channel", f"{name}.pred.json")
    if os.path.isfile(pj):
        try:
            pr_ = json.load(open(pj))
            if "state" in pr_:
                out["state"] = pr_["state"]
                out["narrowest_path_radius_nm"] = pr_.get("narrowest_R_in_core_nm")
                out["dry_gate_on_path"] = pr_.get("dry_plug_on_path")
                if "path_passable" in pr_:
                    out["path_passable"] = pr_["path_passable"]
                    out["path_free_radius_min_nm"] = pr_.get("path_free_radius_min_nm")
        except Exception:                                        # noqa: BLE001
            pass
    if "conductance_counted_nS" in out:
        out["conductance_counted_pS"] = round(out["conductance_counted_nS"] * 1000.0, 1)
    # THE FILTER'S OCCUPANCY, from the trajectory, where the rig declares a dehydrated stretch (the pair law's `slab`
    # window, batch 15): cations inside that z window and within 0.35 nm of the first ion path's centre line, averaged
    # over the records, and how many different ions were ever there. A K+ filter holds 2-3; batch 15's Kv1.2 held
    # 1.4-2.5 and KcsA ~2, with tens of ions coming and going -- read from an ad-hoc script until this was here.
    try:
        _ops = spec.get("operators") or []
        _win = next((o["slab"]["z"] for o in _ops if o.get("op") == "pair_potential" and isinstance(o.get("slab"), dict)), None)
        _sb = next((o for o in _ops if o.get("op") == "slab_barrier" and o.get("at") in ("K", "Na") and o.get("pores")), None)
        if _win is not None and _sb is not None and "filter_occupancy_mean" not in out:
            _ck = f'{_sb["at"]}__pos'
            if _ck in z.files:
                _P = np.asarray(_sb["pores"][0]["profile"], float)
                _X = np.asarray(z[_ck], float)
                _zz = _X[..., 2]
                _r = np.hypot(_X[..., 0] - np.interp(_zz, _P[:, 0], _P[:, 4]),
                              _X[..., 1] - np.interp(_zz, _P[:, 0], _P[:, 5])) * L_um * 1e3
                _in = (_zz > _win[0]) & (_zz < _win[1]) & (_r < 0.35)
                out["filter_occupancy_mean"] = round(float(_in.sum(1).mean()), 2)
                out["filter_ions_distinct"] = int(len(np.unique(np.where(_in)[1])))
    except Exception:                                            # noqa: BLE001
        pass
    if "K_net_in" in out and "Cl_net_in" in out:
        kin, clo = out["K_net_in"], -out["Cl_net_in"]
        out["ions_crossed"] = abs(kin) + abs(out["Cl_net_in"])
        if abs(kin) + abs(clo) > 0:
            out["cation_share_of_charge"] = round(kin / (kin + clo), 2) if (kin + clo) != 0 else None
    # ALPHA-HEMOLYSIN: the assembly's size at the end (the median over the last fifth of the records)
    nla, rng_ = blk("n_largest"), blk("ring")
    if nla is not None and len(nla) > 5:
        out["subunits_per_ring"] = int(round(float(np.median(nla[-max(1, len(nla) // 5):]))))
        if rng_ is not None:
            out["ring_closed_fraction"] = round(float(np.mean(rng_ > 0.5)), 3)
    # invariants
    bad, jump = 0, 0.0
    for s in prot:
        k = f"{s}__pos"
        if k in z.files:
            P = np.asarray(z[k], float)
            bad += int((~np.isfinite(P)).sum())
            if P.shape[0] > 1:
                jump = max(jump, float(np.nanmax(np.linalg.norm(np.diff(P, axis=0), axis=2))) * L_um * 1e3)
    out["nonfinite"] = bad
    out["max_bead_jump_nm_per_record"] = round(jump, 3)
    # the bands, from the experiment's own front matter
    try:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        from exp import load
        fm, _ = load(os.path.join(ROOT, "experiments", "exp04_membrane_channels.md"))
        acc = ((fm.get("channels") or {}).get(ch) or {}).get("accept") or {}
        key = {"conductance_nS": "conductance_max_nS", "half_open_tension_mN_per_m": "half_open_tension_mN_m",
               "open_pore_diameter_nm": "pore_diameter_max_nm", "conductance_pS": "conductance_pS",
               "filter_occupancy_ions": "filter_occupancy_mean", "subunits_per_ring": "subunits_per_ring",
               "constriction_diameter_nm": "pore_diameter_max_nm",
               "conductance_counted_nS": "conductance_counted_nS", "conductance_counted_pS": "conductance_counted_pS"}
        verdict = {}
        closed_ = any(w in str(out.get("state", "")) for w in ("closed", "non-conductive", "prepore", "apart"))
        # A COUNTED FLOW GRADES A CONDUCTANCE BAND, when there is one: the continuum solve's number (4.07 nS for 11h)
        # is a reference, not the channel's -- 11h counted 1.81 nS
        if "conductance_clean_nS" in out:
            key = {**key, "conductance_nS": "conductance_clean_nS", "conductance_pS": "conductance_clean_pS"}
        for band, (lo, hi) in acc.items():
            if closed_ and band.startswith(("conductance", "open_pore")):
                verdict[band] = "not graded (a closed state)"
                continue
            m = key.get(band)
            if m and m in out:
                verdict[band] = "in" if lo <= out[m] <= hi else f"OUT ({out[m]} vs {lo}-{hi})"
            elif m:
                verdict[band] = "not measured"
        out["bands"] = verdict
    except Exception as ex:                                      # noqa: BLE001
        out["bands_error"] = str(ex)[:120]
    return out


MEASURES = {"left_right_size": measure_left_right_size,
            "channel": measure_channel,
            "left_right_area": measure_left_right_area,
            "area_smooth": measure_area_smooth}


def _parse(txt):
    """(score, reason) from a `SCORE: n / REASON: ...` answer; score None when absent."""
    score, reason = None, txt.strip().replace("\n", " ")
    for line in txt.splitlines():
        if line.upper().startswith("SCORE:"):
            digits = "".join(c for c in line.split(":", 1)[1] if c.isdigit() or c == ".")
            score = int(float(digits)) if digits else None
        if line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return score, reason


def ask_agent(images, target, device="cuda:0"):
    """The VLM's score. Returns (score, reason) or (None, why it could not be asked).

    THE PERSISTENT SERVER FIRST (`VLLM/vlm_server.py`): the model is already loaded there, and a
    score is a few seconds instead of a ~70 s load. Without it, a one-off subprocess loads its own
    copy -- slower, same answer."""
    sys.path.insert(0, os.path.join(ROOT, "VLLM"))
    from vlm_client import post, server_url
    url = server_url()
    if url:
        try:
            txt = post(url, "/ask", {"images": [os.path.abspath(i) for i in images],
                                     "text": RUBRIC.format(target=target), "max_new_tokens": 120})["text"]
            return _parse(txt or "")
        except Exception as e:                                  # noqa: BLE001
            print(f"[agent]  server at {url} failed ({e}); loading a private copy", flush=True)
    gemma = os.environ.get("GEMMA_DIR", os.path.join(ROOT, "VLLM", "gemma-4-12B-it"))
    if not os.path.isdir(gemma):
        return None, f"no VLM weights at {gemma}"
    code = f'''
import sys, os, torch
sys.path.insert(0, {os.path.join(ROOT, "VLLM")!r})
os.environ["GEMMA_DIR"] = {gemma!r}
from transformers import AutoProcessor, AutoModelForImageTextToText
proc = AutoProcessor.from_pretrained({gemma!r})
model = AutoModelForImageTextToText.from_pretrained(
    {gemma!r}, torch_dtype=torch.bfloat16, device_map={device!r})
content = [{{"type": "image", "image": f}} for f in {images!r}]
content += [{{"type": "text", "text": {RUBRIC.format(target=target)!r}}}]
msgs = [{{"role": "system", "content": "You are a precise scientific assistant."}},
        {{"role": "user", "content": content}}]
inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                               return_dict=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inp, max_new_tokens=120, do_sample=False)
print("@@@" + proc.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True))
'''
    py = "/workspace/.conda_envs/neural-graph-linux/bin/python"
    r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=1800)
    txt = (r.stdout or "").split("@@@")[-1] if "@@@" in (r.stdout or "") else ""
    if not txt:
        return None, ((r.stderr or "").strip().splitlines() or ["no output"])[-1][:160]
    return _parse(txt)


def stills_of(run_dir, k=3):
    """The run's own stills, newest last. `3d.png` is written by every mpl2d/vtk render."""
    imgs = sorted(glob.glob(os.path.join(run_dir, "*.png")))
    imgs = [i for i in imgs if "fig_" not in os.path.basename(i)] or imgs
    return imgs[-k:]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="<group>/<name>, as Plexus_Main takes it")
    ap.add_argument("--target", required=True)
    ap.add_argument("--measure", default=None, choices=sorted(MEASURES))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="append one JSON line here")
    a = ap.parse_args()

    from plexus.paths import graphs_data_path
    group, name = a.spec.split("/", 1)
    run_dir = graphs_data_path(group, name)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"{run_dir} does not exist -- generate it first")

    rec = {"spec": a.spec, "target": a.target}
    if a.measure:
        rec["measured"] = MEASURES[a.measure](run_dir)
        print(f"[ruler]  {json.dumps(rec['measured'])}")
    imgs = stills_of(run_dir)
    if not imgs:
        print("[agent]  no stills in the run directory")
    else:
        score, reason = ask_agent(imgs, a.target, a.device)
        rec["score"], rec["reason"], rec["images"] = score, reason, [os.path.basename(i) for i in imgs]
        verdict = ("out of scope" if (score or 0) < 5 else
                   "right direction, not acceptable" if (score or 0) < 9 else "acceptable")
        print(f"[agent]  SCORE {score}  ({verdict})\n[agent]  {reason}")
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
