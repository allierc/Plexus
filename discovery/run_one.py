#!/usr/bin/env python
"""run_one -- execute ONE generated config and record it as campaign evidence.

This is the executable half of the validation gate. `validate_space.py` proves a composition is
expressible and compiles to a faithful spec; this proves the spec actually RUNS, produces the
phenotype, and yields a re-scorable record.

What it does that run_tyssue_round.do() does not:

  D3  ASSERTS the recording alignment instead of clamping it.
      do() contains  `mt = hist[min(t, len(hist) - 1)]`  -- when the position and topology
      series drift out of step this silently pairs one frame's coordinates with another frame's
      connectivity. That is the exact line that produced the phantom "97% hollow / global
      buckling" result believed for days. Here a mismatch is a hard error.

  D7  PERSISTS THE FULL TRAJECTORY + a per-frame metric table, so a new observable can re-score
      the archive without re-simulating. The SMG2 archive stored only the final frame.

  D4  RECORDS THE ACTED-LEDGER -- which operators actually did something. A run in which a
      scheduled operator never acted is not evidence.

  Q   Optionally runs the QUASI-STATIC TEST: continue with growth and any driver switched OFF,
      mechanics + reconnection only, and measure what fraction of the protrusion survives.
      Q -> 0 : the tube was FORCED.   Q -> 1 : the tube is an EQUILIBRIUM shape.
      This is the campaign's primary discriminator (round 41, made measurable).

    python run_one.py <config-name> [--frames N] [--device cuda:0] [--no-movie] [--q]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
TYSSUE = os.path.join(ROOT, "prototype", "Tyssue")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, TYSSUE)

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import yaml                                                          # noqa: E402

from run_record import RunArchive, RunRecord, comp_hash              # noqa: E402

CONFIG_DIR = os.path.join(ROOT, "config", "okuda")
LOG_DIR = os.path.join(ROOT, "log", "okuda")
ARCHIVE = os.path.join(HERE, "_archive")


def _lazy_engine():
    """Import the heavy stack only when we actually run (keeps validate_space fast)."""
    import plexus.operators                                          # noqa: F401
    import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d, tyssue_monolayer, ckpt  # noqa: F401
    import plexus.schema as S
    from plexus.engine import run as engine_run
    import instrument
    instrument.install()                       # D4: every operator now reports whether it acted
    return S, engine_run


# --------------------------------------------------------------------------- D3: assert alignment
def check_alignment(posf, hist, name=""):
    """The phantom-result guard. Positions and topology MUST be the same length."""
    T, H = len(posf), len(hist)
    if T != H:
        raise AssertionError(
            f"[D3] recording misalignment in {name}: positions={T} frames but topology={H}. "
            f"do() silently clamped this with hist[min(t, len(hist)-1)], which pairs each "
            f"frame's coordinates with ANOTHER frame's connectivity and fabricates inverted "
            f"cells. Fix the recording strides; do not clamp.")
    return True


# --------------------------------------------------------------------------- per-frame metrics
def frame_metrics(frames):
    """A cheap per-frame table computed on every run, so any temporal observable can be
    re-derived later without re-simulating (D7). `frames` = [(pos, mesh, act), ...]."""
    # NOTE the name. `protr` = percentile(r,95)/median(r) is EXACTLY tube_analysis.py:89's
    # definition. It is NOT the report's "aspect ~7.5" for round_40_mc8, which is
    # tube_len/tube_diam -- a different quantity. Calling this one "aspect" led me to compare
    # 1.73 against 7.5 and invent a discrepancy that does not exist. Measure the geometric thing
    # you mean, and NAME it the thing you measured.
    out = {"n_cells": [], "protr": [], "r95": [], "rmed": [], "act_max": [], "red_frac": []}
    for pos, mt, act in frames:
        r = np.linalg.norm(pos - pos.mean(0), axis=1)
        r95, rmed = float(np.percentile(r, 95)), float(np.median(r) + 1e-9)
        out["n_cells"].append(float(mt["nF"]))
        out["r95"].append(r95)
        out["rmed"].append(rmed)
        out["protr"].append(r95 / rmed)
        a = np.asarray(act, float)
        out["act_max"].append(float(a.max()) if a.size else 0.0)
        out["red_frac"].append(float((a > 0.5 * a.max()).mean()) if a.size and a.max() > 0 else 0.0)
    return out


def protr_of(pos):
    """percentile(r,95)/median(r) -- tube_analysis.py:89. NOT tube_len/tube_diam."""
    r = np.linalg.norm(pos - pos.mean(0), axis=1)
    return float(np.percentile(r, 95) / (np.median(r) + 1e-9))


# --------------------------------------------------------------------------- the run
def run_config(name, frames=None, device="cpu", movie=True, do_q=False, campaign="validation"):
    S, engine_run = _lazy_engine()
    cfg_path = os.path.join(CONFIG_DIR, f"{name}.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    disc = cfg.get("_discovery", {})
    # resolve repo-relative asset paths against THIS checkout, so one tracked config runs both in
    # the devcontainer (/workspace/...) and on the cluster (/groups/.../Graph/...).
    for o in cfg["operators"]:
        ck = o.get("ckpt")
        if ck and not os.path.isabs(ck):
            o["ckpt"] = os.path.join(ROOT, ck)
    out_dir = os.path.join(LOG_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    if frames is not None:                                            # smoke override
        cfg["general"]["n_frames"] = int(frames)
        cfg["general"]["record_cap"] = int(frames) + 2
        tmp = os.path.join(out_dir, "spec_run.yaml")
        yaml.safe_dump(cfg, open(tmp, "w"), sort_keys=False)
        cfg_path = tmp

    print(f"[{name}] comp={disc.get('comp_hash')} region={disc.get('region')!r} "
          f"frames={cfg['general']['n_frames']} dt={cfg['general']['dt']} device={device}",
          flush=True)

    t0 = time.time()
    sim = S.load(cfg_path)
    Hf, out = engine_run(sim, device=device)
    wall = time.time() - t0

    vlvl = Hf.level("vertex")
    emesh = getattr(vlvl, "_mesh", None) or {}
    hist = emesh.get("hist", [])
    posf = out["sets"]["vertex"]["pos"]
    chemf = out["sets"]["cell"]["state"]["chem"]
    T = posf.shape[0]

    check_alignment(posf, hist, name)                                 # D3

    def frame(t):
        mt = hist[t]
        return (posf[t][:mt["Nv"]].astype(np.float64), mt, chemf[t][:mt["nF"], 0])

    fr = [frame(t) for t in range(T)]
    fm = frame_metrics(fr)

    # D4: the acted-ledger. The engine does not yet report this (that fix lands in the operators);
    # until it does we record what we CAN observe and flag the rest as unknown.
    import instrument
    acted, inert = instrument.report(Hf, cfg["schedule"])
    n_unknown = 0

    # --------------------------------------------------------------- Q: the quasi-static test
    q = None
    if do_q:
        q = quasi_static_Q(cfg, cfg_path, device, protr_before=fm["protr"][-1], out_dir=out_dir)

    # --------------------------------------------------------------- the REAL tube metrics
    # tube_analysis is the archive's own metric bank. Comparing against archived numbers requires
    # ITS definitions, not look-alikes of our own.
    tube = {}
    try:
        from tube_analysis import analyze
        samp = np.unique(np.linspace(0, T - 1, min(40, T)).astype(int))
        tube = analyze([(int(t), fr[t][0], fr[t][1], fr[t][2]) for t in samp], out_dir) or {}
        keep = ("tube_len_final", "tube_diam_final", "n_tubes_final", "protr_final",
                "hollow_n_peak", "hollow_n_final", "area_cv_final", "vol_cv_final",
                "red_frac_final", "tip_act_final")
        raw = {k: v for k, v in tube.items() if k in keep}
        if raw.get("tube_diam_final", 0) > 1e-9:
            raw["aspect_len_over_diam"] = round(
                float(raw["tube_len_final"]) / float(raw["tube_diam_final"]), 3)
        # NAMESPACE THEM. tube_analysis computes on 40 SAMPLED frames with its own body-median
        # definition; our frame_metrics computes on all 901. Merging them unprefixed produced
        # `protr_final 3.124 > protr_peak 1.732`, which is impossible -- two different quantities
        # under one name. That is the SAME defect I had just diagnosed, committed again one
        # function later. Prefix `ta_` so provenance is visible in the summary itself.
        tube = {f"ta_{k}": v for k, v in raw.items()}
    except Exception as e:
        print(f"[{name}] tube_analysis unavailable: {type(e).__name__}: {str(e)[:80]}", flush=True)

    # --------------------------------------------------------------- persist
    arch = RunArchive(ARCHIVE)
    graph_struct = disc.get("structure") or {"operators": [], "connections": []}
    rec = RunRecord(graph_struct, params={}, seed=cfg["general"]["seed"],
                    backend="tyssue_avm_3d", ic="checkpoint",
                    campaign=campaign, wall_s=round(wall, 1))
    ref = arch.save_trajectory(rec.run_id, [p for p, _, _ in fr], fm,
                               meta={"config": name, "comp_hash": disc.get("comp_hash"),
                                     "region": disc.get("region"), "n_frames": T})
    rec.set_trajectory_ref(ref)
    rec.set_acted(acted)
    # --------------------------------------------------------------- saturation guard
    # "every high-division run pinned at exactly 890 cells -- a buffer ceiling, not physics."
    # A run that hits its cell buffer is not evidence about a mechanism; it is evidence about a
    # buffer. Flag it loudly so the ledger can never read it as a phenotype.
    cbuf = cfg["sets"]["cell"]["n"]
    saturated = fm["n_cells"][-1] >= 0.9 * cbuf
    if saturated:
        print(f"[{name}] 🔴 SATURATED: {int(fm['n_cells'][-1])} cells vs buffer {cbuf}. "
              f"This run is NOT evidence -- raise the buffer or bound proliferation.", flush=True)

    # retention = final/peak aspect. A FORCED protrusion peaks then collapses (low retention);
    # an EQUILIBRIUM one holds (high). Computable from the archived per-frame table for every
    # run without re-simulating -- the D7 payoff -- and a cheap proxy for the full Q test.
    _pk = max(fm["protr"]) if fm["protr"] else 0.0
    retention = (fm["protr"][-1] / _pk) if _pk > 1e-9 else 0.0

    summary = {"saturated": bool(saturated), "inert_operators": inert,
               "retention": round(retention, 3),
               "valid_evidence": bool(not inert and not saturated),
               "protr_final": round(fm["protr"][-1], 3),
               "protr_peak": round(max(fm["protr"]), 3),
               "n_cells_final": int(fm["n_cells"][-1]),
               "red_frac_final": round(fm["red_frac"][-1], 3),
               "act_max_final": round(fm["act_max"][-1], 3),
               "frames": T, "wall_s": round(wall, 1)}
    summary.update(tube)                       # the archive's own definitions, for comparison
    if q is not None:
        summary["Q"] = round(q, 3)
    rec.add_analysis("metric_v1", summary)
    arch.add(rec)

    print(f"[{name}] {json.dumps(summary)}", flush=True)
    if inert:
        print(f"[{name}] 🔴 INERT OPERATORS {inert} -- this run is NOT evidence. A scheduled "
              f"operator that never acted would be recorded as 'this mechanism cannot work' "
              f"when the mechanism never ran (D4).", flush=True)
    else:
        print(f"[{name}] D4 ok: all {len(acted)} scheduled operators acted "
              f"({ {k: v for k, v in sorted(acted.items(), key=lambda kv: kv[1])[:3]} } ...)",
              flush=True)

    # --------------------------------------------------------------- artefacts
    if movie:
        try:
            render(name, fr, out_dir)
        except Exception as e:
            print(f"[{name}] render failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    json.dump({"config": name, "comp_hash": disc.get("comp_hash"),
               "region": disc.get("region"), "summary": summary, "acted": acted,
               "run_id": rec.run_id},
              open(os.path.join(out_dir, "diag.json"), "w"), indent=1)
    return summary


# --------------------------------------------------------------------------- Q
def quasi_static_Q(cfg, cfg_path, device, protr_before, out_dir, relax_frames=60):
    """Continue from the end state with growth + driver OFF: mechanics and reconnection only.

    A FORCED protrusion collapses (Q -> 0). A GROWN one persists (Q -> 1). This is round 41's
    finding turned into a number, and it is the campaign's primary discriminator.

    NOTE: this requires a checkpoint of the end state. Until `ckpt.save` is wired into the run
    path, Q is computed by re-running the same composition with the growth/forcing operators
    removed for the tail -- which is a WEAKER test (it re-grows from the start). Flagged so the
    ledger never treats a weak Q as a strong one.
    """
    import copy
    S, engine_run = _lazy_engine()
    c2 = copy.deepcopy(cfg)
    drop = {"morphogen_growth_3d", "vesicle_growth", "rd_interface_tension", "cell_rd_seed",
            "divide_3d"}
    c2["operators"] = [o for o in c2["operators"] if o["op"] not in drop]
    c2["schedule"] = [s for s in c2["schedule"] if s not in drop]
    c2["general"]["n_frames"] = relax_frames
    c2["general"]["record_cap"] = relax_frames + 2
    p2 = os.path.join(out_dir, "spec_q.yaml")
    yaml.safe_dump(c2, open(p2, "w"), sort_keys=False)
    try:
        Hq, oq = engine_run(S.load(p2), device=device)
        pos = oq["sets"]["vertex"]["pos"]
        m = getattr(Hq.level("vertex"), "_mesh", {}) or {}
        hq = m.get("hist", [])
        nv = hq[-1]["Nv"] if hq else pos.shape[1]
        a_after = protr_of(pos[-1][:nv].astype(np.float64))
        return a_after / max(protr_before, 1e-9)
    except Exception as e:
        print(f"  [Q] failed: {type(e).__name__}: {str(e)[:90]}", flush=True)
        return None


# --------------------------------------------------------------------------- artefacts
def render(name, fr, out_dir, n_strip=8, movie_frames=60):
    """Strip + movie, matching the minisite convention (black bg, activator LUT)."""
    from run_tyssue_vesicle import _draw, make_movie_axes
    # the minisite / archive convention is 3D shell + a CROSS-SECTION inset taken in the plane
    # of the tubing, so a protrusion reads correctly against the 3D view.
    from run_tyssue_round import _cross_screen, _cross_axis
    from matplotlib.animation import FFMpegWriter
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    T = len(fr)
    asamp = np.concatenate([fr[t][2] for t in np.unique(np.linspace(0, T - 1, 12).astype(int))])
    lo, hi = float(np.percentile(asamp, 5)), float(np.percentile(asamp, 99) + 1e-6)
    col = lambda a: np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    lbox = lambda pt: float(np.abs(pt).max()) * 1.12

    # two rows: 3D shell on top, cross-section beneath, as the archive strips are drawn
    fig = plt.figure(figsize=(4.4 * n_strip, 9.0))
    fig.patch.set_facecolor("black")
    for i, t in enumerate([int(round(f * (T - 1))) for f in np.linspace(0, 1, n_strip)]):
        pt, mt, a = fr[t]
        ax = fig.add_subplot(2, n_strip, i + 1, projection="3d")
        _draw(ax, pt, mt, 3.90, azim=30, act=col(a), Lbox=lbox(pt))
        ax2 = fig.add_subplot(2, n_strip, n_strip + i + 1)
        # NO try/except here. Swallowing the error is exactly the silent-no-op pattern this
        # project keeps being bitten by: the first version caught a TypeError from a wrong
        # signature and rendered a blank row that looked deliberate.
        _cross_screen(ax2, pt, mt, col(a), seed_dir=_cross_axis(pt, None), Lbox=lbox(pt) * 2.05)
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995, wspace=0.02, hspace=0.02)
    fig.savefig(os.path.join(out_dir, "strip.png"), dpi=100, facecolor="black")
    plt.close(fig)

    figm = plt.figure(figsize=(5.0, 5.2))
    figm.patch.set_facecolor("black")
    axm, axin = make_movie_axes(figm)
    keep = np.unique(np.linspace(0, T - 1, min(movie_frames, T)).astype(int))
    wri = FFMpegWriter(fps=10, metadata={"title": name})
    with wri.saving(figm, os.path.join(out_dir, "movie.mp4"), dpi=85):
        for t in keep:
            pt, mt, a = fr[int(t)]
            _draw(axm, pt, mt, 3.90, azim=30, act=col(a), Lbox=lbox(pt))
            _cross_screen(axin, pt, mt, col(a), seed_dir=_cross_axis(pt, None),
                          Lbox=lbox(pt) * 2.05)      # cross-section, minisite convention
            wri.grab_frame()
    plt.close(figm)
    print(f"[{name}] artefacts -> {os.path.relpath(out_dir, ROOT)}/{{strip.png,movie.mp4}}",
          flush=True)


# --------------------------------------------------------------------------- cli
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--q", action="store_true", help="run the quasi-static test")
    ap.add_argument("--campaign", default="validation")
    a = ap.parse_args()
    run_config(a.name, frames=a.frames, device=a.device, movie=not a.no_movie,
               do_q=a.q, campaign=a.campaign)
