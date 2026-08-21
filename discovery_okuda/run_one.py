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
import signal
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
TYSSUE = os.path.join(ROOT, "discovery_okuda", "ops")
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


# OFF BY DEFAULT, ON BY REQUEST. The bar earns its place in a figure and gets in the way of a
# gallery clip, where the page writes the caption and the frame should carry no printing at all --
# every card on the minisite is cropped to remove it. A module flag rather than an argument
# threaded through nine call sites: `_scalebar` is called from run_one, kburns_render, restyle3d
# and edge_test, and the question "does this picture carry a bar" is a property of the RUN, not of
# each axes. Set it with `--scalebar` on any of those entry points, or `run_one.SCALEBAR = True`.
SCALEBAR = False


def _scalebar(ax, Lbox, color="w", frac=0.25, on=None):
    """A scale bar in the bottom-left, with the world length written on it.

    Drawn only when `on` -- or, if `on` is None, the module's `SCALEBAR` -- is true.

    Cedric, 8 August: "the mp4 should have a scale bar bottom left with a number, and the eye
    agent should be aware of the scale bar through passing the camera zoom value."

    WHY IT IS NEEDED HERE SPECIFICALLY. This project renders with a camera that is FIXED for the
    whole run, deliberately -- run_one prints "per-frame autofit would have run 5.467 -> 7.649,
    x1.40: that rescaling is what hid growth". A fixed camera makes growth visible WITHIN a run
    and invisible BETWEEN runs: a 2,000-cell sphere and a 53,000-cell one are drawn the same size,
    each filling its own box. Anyone comparing two movies -- the eye agent above all -- is reading
    shape while believing they are reading size. The number on the bar is the only thing in the
    frame that distinguishes them.

    The bar is drawn in AXES coordinates and its length is computed from the axis limits, so it is
    exact for the 2D panels. On the 3D panels matplotlib's default projection is perspective, so
    the bar is exact only at the depth of the view centre; `_draw` sets a cube box with equal
    aspect, so that is the sphere's own centre and the error at its silhouette is a few percent.
    It is a scale bar, not a caliper -- and a few percent is the difference between reading 53,000
    cells and reading 2,000.
    """
    if not (SCALEBAR if on is None else on):
        return
    span = 2.0 * float(Lbox)
    # a round number near `frac` of the view, so the label reads 1 / 2 / 5 x 10^n
    raw = span * frac
    import math
    mag = 10.0 ** math.floor(math.log10(max(raw, 1e-12)))
    nice = min((1.0, 2.0, 5.0, 10.0), key=lambda m: abs(m * mag - raw)) * mag
    f = nice / span                                    # bar length as a fraction of the axis
    x0, y0 = 0.04, 0.055
    # 3D AXES NEED DIFFERENT CALLS, and this is what broke all sixteen runs of round 1:
    # `Axes3D.text` is text(x, y, z, s) -- four arguments -- so the 2D call raised TypeError inside
    # `render`, whose caller catches, and every run lost strip.png, movie.mp4 AND 3d.png at once.
    # The bar had been checked on a 2D axis only. `text2D` is the axes-fraction version, and a
    # plain Line2D carries its own transform onto a 3D axes where `plot` would not.
    from matplotlib.lines import Line2D
    ax.add_artist(Line2D([x0, x0 + f], [y0, y0], transform=ax.transAxes, color=color, lw=2.6,
                         solid_capstyle="butt", zorder=10_000, clip_on=False))
    _text = getattr(ax, "text2D", ax.text)
    _text(x0 + f / 2.0, y0 + 0.018, f"{nice:g}", transform=ax.transAxes, color=color,
          fontsize=9, ha="center", va="bottom", zorder=10_000, clip_on=False)


def _lazy_engine():
    """Import the heavy stack only when we actually run (keeps validate_space fast)."""
    import plexus.operators                                          # noqa: F401
    import mesh_ops, chem_ops, t1_ops, monolayer_ops, ckpt  # noqa: F401
    import shape_chem_ops                                       # noqa: F401
    # AN OPERATOR THAT IS NOT IMPORTED IS NOT REGISTERED, and the spec naming it dies at compile
    # with no hint that a missing import is the cause. Three of the four shape-gate runs failed
    # here having written nothing but spec_run.yaml, while the fourth -- the only one without the
    # probe -- ran fine.
    import shape_probe_ops                                          # noqa: F401
    import plexus.schema as S
    from plexus.engine import run as engine_run
    import instrument
    instrument.install()                       # D4: every operator now reports whether it acted
    return S, engine_run


HEARTBEAT_SECONDS = 30      # write at least this often, however slow the frames are


def _heartbeat(name, t0, every=10):
    """Write `progress.json` while the run is ALIVE, so a watcher can tell working from stuck.

    Nothing else on disk can make that distinction. `metrics.json` is written by analyze() only
    after the loop finishes; LSF copies the job's `.out` at TERMINATION, not continuously; and the
    run directory holds nothing but `spec_run.yaml` until the end. So a healthy 46-minute run and
    a wedged one are byte-for-byte identical to anyone looking from outside, and the straggler
    killer that looked at them killed five productive runs on a median set by rigid,
    non-dividing chemistry probes that finish in seven minutes.

    fsync'd, because the reader is on a different machine across a network filesystem, and a
    heartbeat sitting in a page cache is not a heartbeat.
    """
    d = os.path.join(LOG_DIR, name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "progress.json")

    last = [0.0]

    def beat(H, tick, phase="simulating", force=False):
        # TIME-BASED AS WELL AS FRAME-BASED. A run that grows 2000 -> 55,521 cells takes ~30x
        # longer per frame than it did at the start, so a beat every 10 frames became a beat
        # every ~2200s -- and a watcher polling each minute saw an unchanged frame number for
        # thirty-six consecutive polls. Writing on a clock as well means the file's mtime tracks
        # LIVENESS, which is the thing the watcher actually needs to know.
        now = time.time()
        if not force and (tick % every and tick) and (now - last[0] < HEARTBEAT_SECONDS):
            return
        last[0] = now
        try:
            # THE LIVE COUNT, from the mesh -- `H.level("cell").n` is the RESERVOIR SIZE, so the
            # first heartbeats reported 138888 cells for a run holding a few thousand. A progress
            # line that reports the array it was given, instead of what it has built, says
            # nothing and looks like everything.
            nc = nv = None
            try:
                m = getattr(H.level("vertex"), "_mesh", None) or {}
                nc, nv = int(m["nF"]), int(m["Nv"])
            except Exception:
                pass
            with open(path, "w") as fh:
                json.dump({"frame": int(tick), "n_cells": nc, "n_vertices": nv,
                           "phase": phase,
                           "elapsed_s": round(time.time() - t0, 1)}, fh)
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            pass                                  # a heartbeat must never take the run down
    return beat


# --------------------------------------------------------------------------- D3: assert alignment
def check_alignment(posf, hist, name=""):
    """The phantom-result guard. Positions and topology MUST be the same length."""
    if not hist:
        raise AssertionError(
            f"[D3] no topology history recorded for {name}: every frame would fall back to the "
            f"SEED mesh, so late-frame coordinates would be read against frame-0 connectivity. "
            f"Schedule topo_record every=1; do not fall back.")
    T, H = len(posf), len(hist)
    if T != H:
        raise AssertionError(
            f"[D3] recording misalignment in {name}: positions={T} frames but topology={H}. "
            f"do() silently clamped this with hist[min(t, len(hist)-1)], which pairs each "
            f"frame's coordinates with ANOTHER frame's connectivity and fabricates inverted "
            f"cells. Fix the recording strides; do not clamp.")
    return True


# --------------------------------------------------------------------------- per-frame metrics
# The growth operator's OWN switch, read from the spec by run_config before the table is
# built. A module-level cell because frame_metrics takes only the frames -- and thresholding
# red_frac at anything else measures a number the growth operator does not act on.
GROWTH_SWITCH = [1.5]


def frame_metrics(frames):
    """The EVERY-FRAME table: chemistry and centroid geometry, on all 901 frames.

    THIS IS TIER 1 AND TIER 2 OF THE SAMPLING, and the split is set by measurement, not taste.
    Timed on a real 3,975-cell mesh:

        the whole mesh table   1300 ms/frame    (hollow_flags 583, face_polygons 301)
        cell centroids         2.1 ms/frame     (vectorised; was 29.7)
        chemistry alone        0.12 ms/frame

    So the mesh metrics are sampled every 25 frames and everything here is measured on EVERY
    frame, for about two seconds a run. That is not a refinement. `okuda_route`'s activator is a
    LIMIT CYCLE OF PERIOD 53 FRAMES -- 17 cycles, constant amplitude, measured -- and at a stride
    of 25 that is 2.1 samples per cycle, below Nyquist. Sampled coarsely it looked like fourteen
    unrelated flashes whose spacings were all exact multiples of the sampling interval: a beat,
    not a signal. The loop could not have found the period at any threshold, on any statistic.

    ONE DEFINITION PER NAME. `protr` was computed here from VERTEX positions and in
    tissue_analysis from CELL CENTROIDS, and both were lifted into the same diag.json -- so one
    summary carried two different numbers under one word. It is centroid-referenced in both
    places now, which changes what `protr_peak` means against the archive; that is acceptable
    because the archive is not treated as reliable evidence, and carrying two definitions of one
    word forever is not.
    """
    from tissue_analysis import _cell_centroids, protrusion_ratio
    keys = ("n_cells", "protr", "protr_p99", "r_cv", "gyr_prolate", "gyr_oblate",
            "act_mean", "act_sd", "act_cv", "act_occupancy", "act_max", "act_min",
            "red_frac", "act_alive", "corr_act_rad", "act_at_tip",
            # species B's own statistics, so a two-species run reports both maps
            "b_act_mean", "b_act_cv", "b_act_max", "b_act_min", "b_act_occupancy")
    out = {k: [] for k in keys}
    for pos, mt, act, act_b in frames:
        cen, rad, live = _cell_centroids(pos, mt)
        out["n_cells"].append(float(mt["nF"]))
        r = rad[live]
        med = float(np.median(r)) if r.size else 0.0
        if r.size > 2 and med > 1e-9:
            out["protr"].append(float(np.percentile(r, 95) / med))
            out["protr_p99"].append(float(np.percentile(r, 99) / med))
            out["r_cv"].append(float(r.std() / (r.mean() + 1e-12)))
            try:
                w = np.linalg.eigvalsh(np.cov(cen[live].T))[::-1]
                tr = float(w.sum())
                out["gyr_prolate"].append(float(w[0] / (0.5 * (w[1] + w[2]) + 1e-12)) if tr > 1e-12 else np.nan)
                out["gyr_oblate"].append(float(1.5 * (w[1] - w[2]) / tr) if tr > 1e-12 else np.nan)
            except Exception:
                out["gyr_prolate"].append(np.nan); out["gyr_oblate"].append(np.nan)
        else:
            for k in ("protr", "protr_p99", "r_cv", "gyr_prolate", "gyr_oblate"):
                out[k].append(np.nan)

        a = np.asarray(act, float)
        mu, sd = (float(a.mean()), float(a.std())) if a.size else (0.0, 0.0)
        lo, hi = (float(a.min()), float(a.max())) if a.size else (0.0, 0.0)
        cv = sd / abs(mu) if abs(mu) > 1e-12 else 0.0
        occ = float((a > lo + 0.5 * (hi - lo)).mean()) if hi > lo + 1e-12 else 0.0
        out["act_mean"].append(mu); out["act_sd"].append(sd); out["act_cv"].append(cv)
        out["act_min"].append(lo); out["act_max"].append(hi); out["act_occupancy"].append(occ)
        out["red_frac"].append(float((a > GROWTH_SWITCH[0]).mean()) if a.size else 0.0)
        # BOTH conditions, so a blow-up is not a pattern: one cell of 4,000 gives cv ~10 at
        # occupancy 0.01, and that is a singularity.
        out["act_alive"].append(float(cv > 0.05 and occ > 0.01))

        # THE SECOND SPECIES, the same reduction on its own channel. Zeros on a single-species run
        # rather than a ragged column, and `b_act_max` is then 0 -- which the registry's `requires`
        # reads as "no second map", so every other b_ quantity is UNDEFINED rather than reported.
        if act_b is None:
            for k in ("b_act_mean", "b_act_cv", "b_act_max", "b_act_min", "b_act_occupancy"):
                out[k].append(0.0)
        else:
            b = np.asarray(act_b, float)
            bmu, bsd = (float(b.mean()), float(b.std())) if b.size else (0.0, 0.0)
            blo, bhi = (float(b.min()), float(b.max())) if b.size else (0.0, 0.0)
            bcv = bsd / abs(bmu) if abs(bmu) > 1e-12 else 0.0
            bocc = float((b > blo + 0.5 * (bhi - blo)).mean()) if bhi > blo + 1e-12 else 0.0
            out["b_act_mean"].append(bmu); out["b_act_cv"].append(bcv)
            out["b_act_max"].append(bhi); out["b_act_min"].append(blo)
            out["b_act_occupancy"].append(bocc)

        # THE COUPLING, REFUSED ON A DEAD FIELD. Pearson is scale-free by construction and
        # returns a confident number for noise: 0.294 measured on an activator whose entire
        # spread across 3,975 cells was 8.4e-05. NaN here means "no pattern to correlate with",
        # which is the honest reading, and _measured_frac reports how often that happened.
        cr = at = np.nan
        if a.size == rad.size and live.sum() > 8 and cv > 0.05:
            al, rl = a[live], rad[live]
            if al.std() > 1e-12 and rl.std() > 1e-12:
                cr = float(np.corrcoef(al, rl)[0, 1])
                tip = rl >= np.percentile(rl, 90)
                if tip.any() and abs(al.mean()) > 1e-12:
                    at = float(al[tip].mean() / al.mean())
        out["corr_act_rad"].append(cr); out["act_at_tip"].append(at)
    return out


def protr_of(pos):
    """percentile(r,95)/median(r) about the TISSUE CENTROID. NOT tube_len/tube_diam.

    The formula now lives in tissue_analysis.protrusion_ratio and is shared with `ta_protr`, so the
    two cannot silently become different quantities again (they did: ta_* measured radius from the
    world origin, this one from the centroid, and both were called `protr`).
    """
    from tissue_analysis import protrusion_ratio
    return protrusion_ratio(np.linalg.norm(pos - pos.mean(0), axis=1))


# --------------------------------------------------------------------------- the run
def _probe_device(device, name):
    """Touch the GPU before anything else, and say WHERE and WHAT if it refuses.

    Twelve of sixteen jobs in the first claim round died in nine seconds each with

        torch.AcceleratorError: CUDA error: unknown error
          at torch.Generator(device=device).manual_seed(sim.seed)   [engine.build]

    and the traceback said nothing about which machine, which card, or whether the driver was
    reachable at all -- so "the cluster is flaky" was as far as the diagnosis could go. That is the
    same defect as a silent fallback: the failure reported, but not the one fact that would let
    anyone act on it.

    This forces CUDA initialisation HERE, where the message can carry the execution host, the
    visible devices, the card and the driver's own version of events. It runs before the spec is
    parsed, so a bad node costs seconds and names itself.

    `cudaErrorUnknown` at generator creation is almost always the context failing to initialise --
    a wedged or contended card, an ECC error, a driver/library mismatch on that host -- and not
    anything the spec or this program did. The point of the probe is to make that visible rather
    than to fix it: the job still dies, but it dies saying which host to drain.
    """
    import socket
    host = socket.gethostname()
    if not str(device).startswith("cuda"):
        return
    tag = f"[{name}] device probe on {host}"
    try:
        import torch
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
        avail = torch.cuda.is_available()
        cnt = torch.cuda.device_count() if avail else 0
        print(f"{tag}: torch {torch.__version__}, cuda avail={avail}, count={cnt}, "
              f"CUDA_VISIBLE_DEVICES={vis}, asked for {device}", flush=True)
        if not avail or cnt == 0:
            raise RuntimeError(f"no CUDA device visible on {host} (count={cnt}, "
                               f"CUDA_VISIBLE_DEVICES={vis})")
        idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
        p = torch.cuda.get_device_properties(idx)
        print(f"{tag}: {p.name}, {p.total_memory / 1e9:.0f} GB, capability {p.major}.{p.minor}, "
              f"driver-reported free {torch.cuda.mem_get_info(idx)[0] / 1e9:.1f} GB", flush=True)
        # THE TWO CALLS THAT ACTUALLY FAILED, in the order the engine makes them: a generator, then
        # a real allocation and a kernel. Doing them here means the error surfaces with the lines
        # above already printed.
        torch.Generator(device=device).manual_seed(0)
        _t = torch.ones(1024, device=device)
        _ = float((_t * 2).sum())
        del _t
        torch.cuda.synchronize(idx)
        print(f"{tag}: OK -- generator, allocation and kernel all succeeded", flush=True)
    except Exception as e:
        print(f"{tag}: FAILED {type(e).__name__}: {e}", flush=True)
        try:
            import subprocess
            smi = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,driver_version,memory.used,memory.total,"
                 "utilization.gpu,ecc.errors.uncorrected.volatile.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30)
            print(f"{tag}: nvidia-smi -> {(smi.stdout or smi.stderr or '(no output)').strip()}",
                  flush=True)
        except Exception as _s:
            print(f"{tag}: nvidia-smi unavailable ({type(_s).__name__})", flush=True)
        raise


def run_config(name, frames=None, device="cpu", movie=True, do_q=False, campaign="validation"):
    _probe_device(device, name)
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
    # ALWAYS WRITE THE SPEC THIS RUN ACTUALLY RAN. This used to live INSIDE the `--frames` branch,
    # so a run only recorded its own spec when a frame count was passed on the command line -- and
    # the round's job script does not pass one. The consequence is not cosmetic: `graph_from_run`
    # rebuilds a parent from `spec_run.yaml`, so EVERY run the loop produced was unusable as a
    # parent. Measured on the first resumed round: all sixteen r001 runs finished, all sixteen
    # lacked the file, `menu` and `coverage` raised AttributeError on the None, `build` followed,
    # and the round launched nothing -- a complete round of GPU spent and nothing able to inherit
    # from it. The basis members escaped it only because they were launched by hand WITH --frames.
    #
    # A run's spec is its primary record. It is written unconditionally, and after the override so
    # it says what ran rather than what was asked for.
    tmp = os.path.join(out_dir, "spec_run.yaml")
    yaml.safe_dump(cfg, open(tmp, "w"), sort_keys=False)
    cfg_path = tmp

    print(f"[{name}] comp={disc.get('comp_hash')} region={disc.get('region')!r} "
          f"frames={cfg['general']['n_frames']} dt={cfg['general']['dt']} device={device}",
          flush=True)

    # PREMISES, STATIC TIER -- before the GPU is touched. These read only the spec, so a
    # composition that cannot possibly work is rejected in milliseconds instead of after ten
    # minutes of simulation followed by a plausible-looking null result. This is a HARD gate: a
    # growth ceiling below the division trigger, or chemistry on the mechanics clock, makes every
    # downstream number a statement about the configuration rather than about the tissue.
    # A deliberate violation is legal -- declare it in the spec under _premises.waive.
    import biologist as _pc
    _static = _pc.check(cfg)
    if _pc.report(_static, f"{name} (static, pre-run)"):
        raise SystemExit(f"[{name}] refusing to run: a premise is broken before the simulation "
                         f"starts. Fix it, or declare it under _premises.waive with a reason.")

    t0 = time.time()
    # Carry the composition into the RUN's folder, next to its spec: that is where the Archivist
    # reads from, and a graph it cannot rebuild is a starting point it cannot use.
    try:
        import shutil as _sh
        _c = cfg_path.replace(".yaml", ".composition.json")
        if os.path.exists(_c):
            _sh.copyfile(_c, os.path.join(out_dir, "composition.json"))
    except Exception:
        pass
    sim = S.load(cfg_path)
    _beat = _heartbeat(name, t0)

    # A KILLED RUN IS STILL EVIDENCE ABOUT THE FRAMES IT REACHED. Cedric, 8 August: "if 1 hour
    # expires, the last job is killed but can still be used."
    #
    # Until now it could not be. `r005_12` was killed at frame 690 of 900 holding 95,755 cells --
    # the largest tissue this project has produced -- and left ROUTE_A.md, progress.json and
    # spec_run.yaml. Nothing measurable. The trajectory lives in the engine's recording buffers
    # and was assembled into `out` only on return, so an interrupt discarded all of it.
    #
    # LSF sends SIGTERM before SIGKILL, so the handler has a window: raise inside `on_frame`, let
    # the engine's loop unwind, then assemble the SAME structure `run` would have returned from
    # the buffers it stashed on H. Everything downstream -- tissue_analysis, morphology, the
    # movie -- then runs unchanged on a shorter trajectory, and the record carries
    # `stopped_early` so no one mistakes 690 frames for 900.
    class _Stopped(Exception):
        pass

    _stop = {"hit": False, "why": None}

    def _on_signal(_sig, _frm):
        _stop["hit"] = True                      # do not raise here -- unwinding C code is unsafe

    for _s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1, signal.SIGUSR2):
        try:
            signal.signal(_s, _on_signal)
        except (ValueError, OSError):
            pass

    _live = {}

    # THE RUNTIME CELL CEILING. A run that crosses it stops ITSELF, cleanly, through the same
    # salvage path a SIGTERM uses -- so it lands as a complete diag.json over a shorter trajectory
    # instead of as nothing.
    #
    # WHY A CEILING AND NOT A PREDICTION. The obvious fix was to refuse these compositions before
    # they run, and it does not work: projecting wall time from `cell_grow.rate` gives a median of 26
    # minutes for the runs that DIED and 26 for the runs that finished -- no discriminating power
    # at all. The strongest single correlate of the final cell count is rate at 0.50 and K_V at
    # -0.49, which is not enough to refuse on. What makes a run reach 66,000 cells is the
    # interaction of growth, the activator gate and division, and the honest way to bound an
    # interaction you cannot predict is to measure it while it runs.
    #
    # 25,000 IS WHERE THE EVIDENCE IS. Measured over r013-r021: the 91 runs that finished have a
    # median of 6,194 cells and a maximum of 50,532; the 42 that died sit at a median of 65,684.
    # A body past 25,000 has already answered "does this composition overgrow" -- everything after
    # that is spent measuring a reservoir, and it is what pushes a run past the 90-minute round cap
    # into producing nothing whatsoever. `_run.cell_ceiling: 0` in a spec turns it off for a run
    # that genuinely wants to be enormous.
    _ceiling = int(os.environ.get("OKUDA_CELL_CEILING",
                                  (cfg.get("_run") or {}).get("cell_ceiling", 25000)))

    def _beat_or_stop(H, tick, phase="simulating", force=False):
        _live["H"] = H                            # on_frame is handed H; keep it for the salvage
        _beat(H, tick, phase=phase, force=force)
        if _stop["hit"]:
            _stop["why"] = _stop.get("why") or "signal"
            raise _Stopped(tick)
        if _ceiling > 0:
            try:
                _m = getattr(H.level("vertex"), "_mesh", None) or {}
                if int(_m["nF"]) >= _ceiling:
                    _stop["why"] = f"cell ceiling {_ceiling}"
                    print(f"[{name}] CELL CEILING {_ceiling} reached at frame {tick} -- stopping "
                          f"cleanly and keeping the trajectory so far. This run is evidence about "
                          f"a tissue that grew to {_ceiling}, not a run that failed.", flush=True)
                    raise _Stopped(tick)
            except _Stopped:
                raise
            except Exception:
                pass                              # never let the ceiling check take a run down

    stopped_early = None
    try:
        Hf, out = engine_run(sim, device=device, on_frame=_beat_or_stop)
    except _Stopped as e:
        import plexus.engine as _E
        Hf = _live["H"]
        stopped_early = int(getattr(e, "args", [0])[0] or 0)
        # rows actually written: the recorded ticks at or before the stop
        _rows = sum(1 for _t in getattr(Hf, '_rec_index', {}) if _t <= stopped_early) or 1
        out = _E._assemble(Hf, *Hf._rec, n_rows=_rows)
        print(f"[{name}] STOPPED EARLY at frame {stopped_early} ({_stop.get('why') or 'signal'}) "
              f"-- assembling the trajectory reached so far; this run is evidence about those "
              f"frames", flush=True)
    # THE LAST BEAT. The heartbeat fires from on_frame, so it stopped the instant the loop ended
    # -- and everything after this line (tissue_analysis, morphology, the movie, the strip) takes
    # minutes. A run that had FINISHED its 900 frames therefore went silent and was killed as
    # "wedged, not slow" while it was writing its own results. Marking the phase means staleness
    # can mean wedged again, because a run in `analysing` is never a straggler.
    try:
        _beat(Hf, sim.n_frames, phase="analysing", force=True)
    except Exception:
        pass
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
        # CHANNEL 2 TRAVELS WITH THE FRAME. A two-species run carries species A in chem columns
        # (0,1) and species B in (2,3); this tuple carried column 0 only, so every per-frame
        # quantity described one species and was silent about the other. None on a single-species
        # run, and the loop below then measures nothing extra.
        b = chemf[t][:mt["nF"], 2] if chemf[t].shape[1] >= 3 else None
        return (posf[t][:mt["Nv"]].astype(np.float64), mt, chemf[t][:mt["nF"], 0], b)

    fr = [frame(t) for t in range(T)]
    # THE SWITCH BEFORE THE TABLE. `red_frac` is the fraction of cells the growth operator
    # actually acts on, so it must be thresholded at that operator's OWN a_sw -- and the table is
    # built here, before the block below that used to read it. Read it first or every red_frac in
    # the every-frame table is measured against a default the simulation never used.
    GROWTH_SWITCH[0] = next((float(o["a_sw"]) for o in cfg.get("operators", [])
                    if o.get("op") == "cell_grow" and "a_sw" in o), 1.5)
    per_frame = frame_metrics(fr)
    # THE EVERY-FRAME RECORD, ON DISK. `time_analysis.report` reads it beside metrics.npz, so the
    # trajectories an agent is shown carry the chemistry at full resolution while the mesh
    # columns stay at their coarse stride.
    try:
        np.savez(os.path.join(out_dir, "frames.npz"),
                 frame=np.arange(len(per_frame["n_cells"]), dtype=float),
                 **{k: np.asarray(v, float) for k, v in per_frame.items()})
    except Exception as _e:
        print(f"[{name}] frames.npz not written: {type(_e).__name__}", flush=True)

    # D4: the acted-ledger. The engine does not yet report this (that fix lands in the operators);
    # until it does we record what we CAN observe and flag the rest as unknown.
    import instrument
    acted, inert = instrument.report(Hf, cfg["schedule"])
    n_unknown = 0

    # --------------------------------------------------------------- Q: the quasi-static test
    q = None
    if do_q:
        q = quasi_static_Q(cfg, cfg_path, device, protr_before=per_frame["protr"][-1], out_dir=out_dir,
                           Hf=Hf)

    # A FIXED STRIDE, NOT A FIXED COUNT. `min(40, T)` gave EVERY run forty samples, so a
    # 900-frame run was sampled every 23 frames and a 300-frame run every 8 -- and "frame 12 of
    # the series" meant a different moment in each. Two runs' trajectories were therefore not
    # comparable point by point, which is the one thing a trajectory is for; and the resolution of
    # "when did the pattern die" silently depended on how long the run happened to be.
    #
    # Every 25 frames, always. A 900-frame run yields 37 samples, a 300-frame run 13, and sample k
    # is frame 25k in both. The cost scales with run LENGTH, which is correct: a longer run has
    # more to say. One number, here, so the frequency is a decision rather than an accident.
    ANALYSIS_STRIDE = 25

    # --------------------------------------------------------------- the REAL tube metrics
    # tissue_analysis is the archive's own metric bank. Comparing against archived numbers requires
    # ITS definitions, not look-alikes of our own.
    tube = {}
    try:
        from tissue_analysis import analyze
        samp = np.unique(np.append(np.arange(0, T, ANALYSIS_STRIDE), T - 1))
        # red_frac must be thresholded at the GROWTH OPERATOR'S OWN switch, not at the midpoint of
        # the activator's current range. The relative version is scale-free and therefore blind --
        # it read exactly 0.070 on every one of 40 frames while the pattern changed under it.
        growth_switch = GROWTH_SWITCH[0]
        tube = analyze([(int(t), fr[t][0], fr[t][1], fr[t][2]) for t in samp], out_dir,
                       a_sw=growth_switch) or {}
        keep = ("tube_len_final", "tube_diam_final", "n_tubes_final", "protr_final",
                "hollow_n_peak", "hollow_n_final", "area_cv_final", "vol_cv_final",
                "red_frac_final", "tip_act_final")
        raw = {k: v for k, v in tube.items() if k in keep}
        if raw.get("tube_diam_final", 0) > 1e-9:
            raw["aspect_len_over_diam"] = round(
                float(raw["tube_len_final"]) / float(raw["tube_diam_final"]), 3)
        # NAMESPACE THEM. tissue_analysis computes on 40 SAMPLED frames with its own body-median
        # definition; our frame_metrics computes on all 901. Merging them unprefixed produced
        # `protr_final 3.124 > protr_peak 1.732`, which is impossible -- two different quantities
        # under one name. That is the SAME defect I had just diagnosed, committed again one
        # function later. Prefix `ta_` so provenance is visible in the summary itself.
        tube = {f"ta_{k}": v for k, v in raw.items()}
    except Exception as e:
        print(f"[{name}] tissue_analysis unavailable: {type(e).__name__}: {str(e)[:80]}", flush=True)

    # ------------------------------------------------------------- WHICH OF OKUDA'S SHAPES IS IT
    # THE ARBITER, as of 2026-08-01. The instrument gate used to certify metrics by making them
    # reproduce labels a person wrote from the rendered movies, and the eye-check could drop a run
    # out of the ranking on a caption. Cedric's call: movies inform, they do not decide. So the
    # ground truth is COMPUTED -- and it is stronger than an eye label, not merely cheaper,
    # because morphology.classify is certified against shapes whose answer is known BY
    # CONSTRUCTION (a built sphere is a sphere; a self-intersecting shell gets no label at all).
    #
    # The whole PATH is recorded, not just the endpoint: Okuda's tubes begin as bumps, and the
    # transition is the part that says when the mechanism acted.
    morph = {}
    try:
        import morphology as MP
        from tissue_analysis import _cell_centroids
        series = []
        for t in np.unique(np.append(np.arange(0, T, ANALYSIS_STRIDE), T - 1)):
            pos, mt, _ = fr[int(t)][:3]
            cen, rad, live = _cell_centroids(pos, mt)
            if live.sum() < 8:
                continue
            series.append(MP.classify(cen, rad, live,
                                      protr=float(np.percentile(rad[live], 95)
                                                  / (np.median(rad[live]) + 1e-9))))
        if series:
            morph = MP.classify_series(series)
            print(f"[{name}] morphology: {morph.get('morphology')}   path "
                  f"{' -> '.join(morph.get('path', []))}", flush=True)
    except Exception as e:
        # NOT swallowed into a default: a missing label must read as missing, never as "sphere".
        print(f"[{name}] MORPHOLOGY UNAVAILABLE: {type(e).__name__}: {str(e)[:90]}", flush=True)

    # --------------------------------------------------------------- persist
    arch = RunArchive(ARCHIVE)
    graph_struct = disc.get("structure") or {"operators": [], "connections": []}
    rec = RunRecord(graph_struct, params={}, seed=cfg["general"]["seed"],
                    backend="tyssue_avm_3d", ic="checkpoint",
                    campaign=campaign, wall_s=round(wall, 1))
    # INDEXED, NOT UNPACKED. The frame tuple gained a fourth member (species B's channel) and a
    # fixed-arity unpack here raised "too many values to unpack (expected 3)" -- after the run had
    # finished simulating, so the whole run was lost at the archiving step.
    ref = arch.save_trajectory(rec.run_id, [f[0] for f in fr], per_frame,
                               meta={"config": name, "comp_hash": disc.get("comp_hash"),
                                     "region": disc.get("region"), "n_frames": T})
    rec.set_trajectory_ref(ref)
    rec.set_acted(acted)
    # --------------------------------------------------------------- saturation guard
    # "every high-division run pinned at exactly 890 cells -- a buffer ceiling, not physics."
    # A run that hits its cell buffer is not evidence about a mechanism; it is evidence about a
    # buffer. Flag it loudly so the ledger can never read it as a phenotype.
    cbuf = cfg["sets"]["cell"]["n"]
    saturated = per_frame["n_cells"][-1] >= 0.9 * cbuf
    # WHEN it saturated, not merely THAT it did -- because that is the whole difference between a
    # censored measurement and a void one. A run that met the array at 60% of its length grew,
    # patterned and was measured for six hundred frames first, and discarding it threw away five
    # of twelve slots on 3 August. A run that met it at 10% never had a chance to say anything.
    # The Critic uses this fraction to tell those two apart; without it, both look identical.
    _sat_frac = None
    if saturated:
        _n = per_frame["n_cells"]
        _hit = next((i for i, v in enumerate(_n) if v >= 0.9 * cbuf), None)
        if _hit is not None and len(_n) > 1:
            _sat_frac = _hit / (len(_n) - 1)
        print(f"[{name}] SATURATED: {int(_n[-1])} cells vs buffer {cbuf}"
              + (f", first reached at {_sat_frac:.0%} of the run" if _sat_frac is not None else "")
              + ". n_cells_final is a LOWER BOUND from here on -- growth readings after that "
                "frame describe the array, not the tissue.", flush=True)

    # --------------------------------------------------------------- THE EVIDENCE HORIZON
    # 0A-7. `protr_peak` used to be max() over EVERY recorded frame with no validity filter. When
    # a mesh tears apart, cells fly outward and that number spikes -- and the spike BECAME THE
    # SCORE (score_run weights protr_peak). So the search was not merely tolerating broken meshes,
    # it was PAID to produce them: the most elongated run in the overnight study read 32.7 with
    # 84% of its cells flagged. Run long enough, any search under that rule discovers that the
    # cheapest way to score well is to blow the tissue apart, and reports it as the best mechanism
    # found.
    #
    # The rule is NOT "a broken mesh is not evidence" -- that was too strong (Cedric): if the mesh
    # fails at frame 380 of 400 the first 379 frames are sound physics. Instead every measurement
    # is taken strictly BEFORE the first frame at which the mesh stops being valid, and the
    # horizon is recorded so a reader can see how much of the run counted.
    #
    # It keys on `broken_n` ONLY -- under-connected cells and rings that are not polygons. NOT the
    # legacy blend: that is dominated by just-divided slivers (r=+0.94 with the tip-cell count,
    # while broken stayed at 0 for 300 frames), so thresholding it would penalise the tissue for
    # dividing, i.e. for doing the thing we want.
    horizon = {"horizon": None, "why": "not computed"}
    last_valid_i = len(per_frame["protr"]) - 1
    try:
        import time_analysis as _CS
        _mn = os.path.join(out_dir, "metrics.npz")
        if os.path.exists(_mn):
            _z = np.load(_mn)
            if "broken_n" in _z.files:
                # `cells` is passed so the 10% ceiling can apply: without it the horizon has no
                # denominator and one broken face in 3,250 ends the evidence (r001_06,
                # valid_frac 0.167 on damage that peaks at 5.9%).
                horizon = _CS.evidence_horizon(
                    {}, {"broken_n": _z["broken_n"]},
                    _z["frame"] if "frame" in _z.files else None,
                    cells=(_z["cells"] if "cells" in _z.files else None))
                if horizon.get("horizon") is not None and not horizon.get("complete", True):
                    last_valid_i = min(last_valid_i, max(0, int(horizon["horizon"])))
            else:
                horizon = {"horizon": None, "why": "metrics.npz carries no broken_n"}
            # AND IT MUST ASK THE OTHER WITNESS. broken_n is one signature of a run that stopped
            # being tissue; SELF-INTERSECTION is another, and the horizon was deaf to it.
            # Measured on r003c_04_6d1b25: P11 reports the surface folds through itself AT FRAME
            # 530 -- median 18 ray crossings -- and the record still said horizon_frame 900,
            # valid_frac 1.0, valid_evidence True, with protr_peak 2.266 computed across the
            # break. That run sat at the top of the leaderboard on a number measured from a
            # folded mesh, while the premise that detected the fold was written down beside it.
            #
            # ray_single_frac is the same quantity P11 keys on, so the horizon and the premise
            # can no longer disagree about when the run ended.
            if "ray_single_frac" in _z.files:
                ray_single = np.asarray(_z["ray_single_frac"], float)
                folded_samples = np.where(np.isfinite(ray_single) & (ray_single < 0.5))[0]
                if folded_samples.size:
                    # A SAMPLE INDEX IS NOT A FRAME NUMBER, and this branch used one as the other.
                    # metrics.npz has ONE ROW PER SAMPLED FRAME -- 37 rows at stride 25 -- while
                    # `per_frame` and `last_valid_i` are per-FRAME over all 901. So a fold first seen at sample 8
                    # (frame 200) was recorded as `horizon_frame 8` and truncated the evidence to
                    # `per_frame["protr"][:8]`: eight frames of a nine-hundred-frame run, with protr_peak
                    # and protr_final then computed over the seed sphere alone. Every run whose
                    # mesh ever folds was scored on its first few frames.
                    #
                    # The sibling branch above never had this: it hands `_z["frame"]` to
                    # evidence_horizon and gets a frame number back. This one read the column and
                    # forgot the axis. Found by an external review of the metric work, not by me.
                    frame_numbers = np.asarray(_z["frame"], float) if "frame" in _z.files else None
                    fold_sample = int(folded_samples[0])
                    fold_frame = int(frame_numbers[fold_sample]) if (frame_numbers is not None and fold_sample < frame_numbers.size) else fold_sample
                    if horizon.get("horizon") is None or fold_frame < int(horizon["horizon"]):
                        horizon = {"horizon": fold_frame, "complete": False,
                                   "why": (f"the surface stops being singly covered at index "
                                           f"{fold_frame} (ray_single_frac < 0.5) -- the same fold P11 "
                                           f"reports; everything after it measures a folded mesh")}
                    # THE FOLDED FRAME IS NOT EVIDENCE. `valid_protr = protr[:last_valid_i + 1]` is
                    # inclusive, so passing the fold index would keep the first frame at which
                    # the surface is already through itself -- and that frame is often exactly
                    # where the spurious peak sits. The last admissible frame is the one before.
                    last_valid_i = min(last_valid_i, max(0, fold_frame - 1))
        else:
            horizon = {"horizon": None, "why": "no metrics.npz (tissue_analysis did not run)"}
    except Exception as e:
        # Loud, not silent: if the horizon cannot be computed we must not quietly fall back to
        # scoring the whole run, because that is the behaviour being fixed.
        horizon = {"horizon": None, "why": f"{type(e).__name__}: {str(e)[:90]}"}
    if horizon.get("horizon") is None:
        print(f"[{name}] ⚠ no evidence horizon ({horizon['why']}) -- peak/final are taken over "
              f"ALL {len(per_frame['protr'])} frames, which is the un-truncated behaviour", flush=True)
    elif last_valid_i < len(per_frame["protr"]) - 1:
        print(f"[{name}] evidence horizon at frame {horizon['horizon']}: peak/final taken over "
              f"the first {last_valid_i + 1} of {len(per_frame['protr'])} frames", flush=True)

    valid_protr = per_frame["protr"][:last_valid_i + 1] or per_frame["protr"]

    # retention = final/peak aspect. A FORCED protrusion peaks then collapses (low retention);
    # an EQUILIBRIUM one holds (high). Computable from the archived per-frame table for every
    # run without re-simulating -- the D7 payoff -- and a cheap proxy for the full Q test.
    _pk = max(valid_protr) if valid_protr else 0.0
    retention = (valid_protr[-1] / _pk) if _pk > 1e-9 else 0.0

    # THE PATTERN'S LIFETIME, over the whole run. A per-frame number tells you the field is dead
    # NOW; these say whether it ever lived and when it stopped -- which is the difference between
    # "chemistry is inert for shape" and "the chemistry died at frame 350 and the rest of the run
    # grew on a corpse".
    _al = per_frame["act_alive"]
    _alive_frac = round(sum(_al) / len(_al), 4) if _al else None
    # The LAST time it was alive and stopped being alive -- not the first dip, so a field that
    # flickers early and recovers is not recorded as extinct.
    _extinct = None
    if _al and _al[-1] < 0.5 and any(v > 0.5 for v in _al):
        for _k in range(len(_al) - 1, 0, -1):
            if _al[_k - 1] > 0.5 and _al[_k] < 0.5:
                _extinct = _k
                break
    _peak_f = int(max(range(len(per_frame["act_max"])), key=lambda i: per_frame["act_max"][i])) if _al else None

    summary = {"saturated": bool(saturated), "saturated_frac_of_run": _sat_frac,
               "act_alive_frac": _alive_frac, "act_extinct_frame": _extinct,
               "act_peak_frame": _peak_f,
               "inert_operators": inert,
               "retention": round(retention, 3),
               "valid_evidence": bool(not inert and not saturated),
               "protr_final": round(valid_protr[-1], 3),          # last VALID frame, not last frame
               "protr_peak": round(max(valid_protr), 3),           # over VALID frames only
               # THE EXPLICIT VOCABULARY (Phase 0, item 20). `protr` reads as a length and is a
               # RATIO -- 1.0 is a sphere -- and reading 1.62 as an amount of protrusion is the
               # mistake the old name invited. Both names are written: the archive is never
               # rewritten, and 92 references cannot be cut over in one step without a window
               # where half the code reads a key the other half stopped writing. New readers
               # take `elongation*`; `vocab.canonical()` resolves either.
               "elongation_at_end": round(valid_protr[-1], 3),
               "elongation_peak": round(max(valid_protr), 3),
               "horizon_frame": horizon.get("horizon"),
               "horizon_why": horizon.get("why"),
               "first_damage_frame": horizon.get("first_damage"),
               "valid_frac": horizon.get("valid_frac", 1.0),
               # kept so the truncation is auditable and the change is visible in the record
               "protr_peak_untruncated": round(max(per_frame["protr"]), 3),
               "protr_final_untruncated": round(per_frame["protr"][-1], 3),
               "n_cells_final": int(per_frame["n_cells"][-1]),
               # ONE PRODUCER PER NAME. These used to be written here from `per_frame` AND again by the
               # lift below from tissue_analysis's coarse series -- the same word, two numbers, one
               # diag.json. They come from the every-frame table only, and the lift skips any name
               # this table already owns.
               "red_frac_final": round(per_frame["red_frac"][-1], 3),
               "act_max_final": round(per_frame["act_max"][-1], 3),
               "frames": T, "wall_s": round(wall, 1)}
    summary.update(tube)                       # the archive's own definitions, for comparison
    # THE PATTERN, LIFTED INTO THE SUMMARY. pattern_scale has existed and been certified for
    # weeks -- weekend.py records it: "n_spots exact at 3/5/12, spacing within 13% of
    # R sqrt(4pi/k)" -- and tissue_analysis computes it every frame. It reached metrics and stopped
    # there, so the summary the agents read carried no pattern number at all. The consequence is
    # the phase's headline finding: the run with the finest Turing field in the campaign is
    # recorded as `morphology=sphere, protr 1.003`, a null, because shape was measured and
    # pattern was not. Identical in shape to the reservoir counters, which cell_divide also
    # computed and nobody carried.
    # READ FROM THE SERIES ON DISK, not from `per_frame`. The first wiring looked in `per_frame`, which does
    # not carry them: pattern_metrics is merged into the per-frame record by tissue_analysis and
    # reaches metrics.json, so the series is where they live. Measured after that wiring shipped:
    # nine finished runs, "pattern keys: NONE" in every summary while n_spots, spot_cells_med,
    # spot_cells_max, spot_frac and spot_spacing_cells sat in metrics.json all along. The same
    # defect as before -- computed, written, and never lifted to the one structure agents read.
    try:
        _pser = json.load(open(os.path.join(out_dir, "metrics.json"))).get("series") or []
        if _pser:
            _last = _pser[-1]
            # EVERY ADMITTED METRIC IN THE SERIES, not a hand-kept list of six. The list was
            # written when the pattern metrics were lifted and then had to be edited by hand for
            # every metric added afterwards -- which is the same defect one level up: computed,
            # written to metrics.json, and never carried to the one structure the agents read.
            # corr_act_rad has been ADMITTED since the Turing x vertex study and reached no
            # summary ever, so every prediction naming it scored `not measured`.
            from predict import KNOWN_METRICS as _KM
            # EVERYTHING THE EVERY-FRAME TABLE OWNS. The coarse mesh series still carries these
            # columns for metrics.png, but they must not reach the summary: the every-frame table
            # is the producer, and a second value under the same key is the twin defect.
            _AMBIGUOUS_BARE = tuple(per_frame.keys()) + ("protr", "act_max", "red_frac", "cells")
            _want = set(_KM) | {"n_spots", "spot_cells_med", "spot_cells_max", "spot_frac",
                                "spot_spacing_cells", "wavelength_cells"}
            for _k, _v in _last.items():
                if _v is None or not isinstance(_v, (int, float)) or isinstance(_v, bool):
                    continue
                if _k not in _want and f"{_k}_final" not in _want:
                    continue
                _rv = round(float(_v), 4) if isinstance(_v, float) else _v
                # FILL, NEVER OVERWRITE. `red_frac_final` and `act_max_final` are already in the
                # summary from `per_frame`, measured with run_one's own definitions -- per_frame thresholds
                # red_frac at half the activator's current RANGE, tissue_analysis at the growth
                # operator's absolute switch `a_sw`. tissue_analysis's is the better definition and
                # replacing it here would silently redefine a metric mid-campaign, so every
                # archived run would carry a number that no longer means what the new ones mean.
                # Changing a definition is a deliberate act with a version bump, not a side
                # effect of generalising a loop.
                if f"{_k}_final" not in summary:
                    summary[f"{_k}_final"] = _rv
                # ...and under its BARE name too when that is what is admitted, because
                # `predict.Clause.check` looks the metric up by exact key: an admitted `r_cv`
                # against a summary holding only `r_cv_final` is unmeasured, silently.
                # Never overwrite: run_one's own `per_frame` keys are measured from VERTEX positions and
                # tissue_analysis's from CELL CENTROIDS, and letting one quietly replace the other
                # is exactly the protr/ta_protr divergence again.
                # ...EXCEPT where run_one already defines a family under that name. `protr`,
                # `act_max` and `red_frac` are measured by BOTH: run_one's `per_frame` from VERTEX
                # positions and an activator thresholded at half its own range, tissue_analysis's
                # from CELL CENTROIDS and the growth operator's absolute switch. The summary
                # already carries protr_peak/protr_final, act_max_final and red_frac_final from
                # the first; writing the bare name from the second puts two definitions of one
                # word in one dict, which is the protr/ta_protr divergence that produced a real
                # wrong conclusion. The `ta_` prefix exists precisely for these.
                if _k in _KM and _k not in summary and _k not in _AMBIGUOUS_BARE:
                    summary[_k] = _rv
    except Exception as _e:
        print(f"[{name}] pattern metrics not lifted: {type(_e).__name__}", flush=True)
    if morph:
        summary["morphology"] = morph.get("morphology")
        summary["morphology_path"] = " -> ".join(morph.get("path", []))
        summary["morphology_why"] = str(morph.get("why", ""))[:200]
    try:
        summary.update(mechanics(name, fr, cfg, out_dir))   # force / stress / tension / migration
    except Exception as e:
        print(f"[{name}] mechanics FAILED: {type(e).__name__}: {str(e)[:110]}", flush=True)
    if q is not None:
        summary["Q_protr_after_relax"] = round(q, 3)          # ABSOLUTE, not a ratio (M4)
        summary["Q_drop"] = round(per_frame["protr"][-1] - q, 3)     # how much did NOT survive
    # THE REDUCTIONS, INTO THE ONE STRUCTURE THE SCORER READS. Everything above this line was
    # built today -- three sampling tiers, an every-frame chemistry record, six temporal
    # reductions in time_analysis, 152 admitted names in predict -- and NONE of it was reaching a
    # prediction. Measured on this very run: `act_cv_peak <= 0.3`, `act_max_span >= 100` and
    # `corr_act_rad_measured_frac <= 0.1` -- three claims that state exactly what okuda_route
    # does -- all scored `not measured` -> inconclusive, because `reduce_all` was called by the
    # tests and by nothing else.
    #
    # That is the defect this entire phase is about, reproduced one level up while fixing it: a
    # producer with no consumer. The instrument existed, was documented, was admitted, and was
    # not plugged in.
    try:
        import time_analysis as _TA
        _cols, _fb = {}, {}
        for _f in ("frames.npz", "metrics.npz"):
            _pth = os.path.join(out_dir, _f)
            if not os.path.exists(_pth):
                continue
            _zz = np.load(_pth)
            _fr = np.asarray(_zz["frame"], float) if "frame" in _zz.files else None
            for _k in _zz.files:
                if _k == "frame" or _k in _cols:      # frames.npz first: never overwritten by
                    continue                          # the coarse table
                if _zz[_k].dtype.kind not in "fiub" or _zz[_k].ndim != 1:
                    continue
                _cols[_k] = _zz[_k]
                _fb[_k] = _fr if _fr is not None else np.arange(_zz[_k].size, dtype=float)
        # ONLY THE ADMITTED QUANTITIES. Reducing every column wrote 390 keys, including six
        # suffixes of `autocorr_hops_uncalibrated` -- a metric the bank REJECTS as uncalibrated.
        # A record that carries a withdrawn instrument under six new names has re-admitted it.
        from predict import SERIES_QUANTITIES as _SQ
        _red = _TA.reduce_all(_cols, _fb, horizon_frame=horizon.get("horizon"),
                              keys=[k for k in _cols if k in set(_SQ)])
        # FILL, NEVER OVERWRITE. protr_peak and protr_final are written above from the
        # every-frame table with the evidence horizon already applied; a second value under the
        # same key is the twin defect that was closed this morning.
        _new = {k: v for k, v in _red.items() if k not in summary and v is not None}
        summary.update(_new)
        print(f"[{name}] {len(_new)} temporal reduction(s) written to the summary "
              f"({len(_cols)} series x 6)", flush=True)
    except Exception as _e:
        print(f"[{name}] ⚠ reductions NOT written: {type(_e).__name__}: {str(_e)[:90]}", flush=True)

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
    if True:
        try:
            _cam = render(name, fr, out_dir, movie=movie)
            if isinstance(_cam, dict):
                summary.update(_cam)
            # Captioning is NOT done here. The cluster environment has no `transformers`, so an
            # in-job caption fails on exactly the runs a long campaign produces -- leaving the
            # Watcher blind where it matters most. caption_wave.py does it on the devcontainer
            # side with ONE model load per wave, as part of closing the round.
        except Exception as e:
            print(f"[{name}] render failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        # RECORDED IN THE SUMMARY, NOT ONLY PRINTED.
        #
        # The print above and its traceback go to the job's stdout, which lives on the cluster and
        # is read by nobody -- so when a frame-tuple change broke `render`, every run in the batch
        # finished green with a full diag.json, a full metrics.png and no movie, and the only thing
        # that ever noticed was Cedric opening a folder and asking whether that was normal. A
        # failure that reports somewhere unread has not reported.
        #
        # `missing` is empty on a healthy run, so it costs one key and reads as a fact rather than
        # an alarm. It goes in the summary because the summary is what the loop, the collector and
        # every montage script already open.
        _missing = [f for f in ("strip.png", "3d.png") + (("movie.mp4",) if movie else ())
                    if not os.path.exists(os.path.join(out_dir, f))]
        summary["artefacts_missing"] = _missing
        if _missing:
            print(f"[{name}] ARTEFACTS MISSING: {', '.join(_missing)} -- recorded in diag.json. "
                  f"traj.npz is written, so `python rerender.py {name}` rebuilds them without "
                  f"re-simulating.", flush=True)

        # ---- THE VTK CLIPS, added 12 August, ALONGSIDE the matplotlib artefacts and not instead
        # of them. `mplot3d` has no depth buffer -- it paints polygons back to front by mean z, so
        # on a closed body half the faces point away from the camera and are drawn anyway, and
        # which of them wins a tie changes with the angle. VTK is z-buffered per pixel and 29x
        # faster (0.32 s a frame against 9.33); sequence 3 is four clips and 1,020 frames in about
        # 21 s, against a run that costs 30-40 minutes of GPU.
        #
        # WHY THE MATPLOTLIB PATH STAYS. `strip.png` is what the Eye agent actually reads -- it
        # takes a PNG and not an mp4 -- and `3d.png` is what every montage tiles. Dropping the old
        # renderer before those two have a VTK source would blind a role and break the sheets, so
        # the migration is additive until they do.
        #
        # IT CANNOT KILL A RUN. Everything above this point is already written; a missing clip is
        # recorded like any other missing artefact and rebuilt later from traj.npz.
        try:
            sys.path.insert(0, HERE) if HERE not in sys.path else None
            import vtk_render
            _t0 = time.time()
            _took = vtk_render.render_all(name, seq=vtk_render.LOOP_SEQ, quiet=True)
            if _took:
                summary["vtk_clips"] = sorted(_took)
                print(f"[{name}] VTK seq {vtk_render.LOOP_SEQ}: {len(_took)} clips in "
                      f"{time.time() - _t0:.1f} s ({', '.join(sorted(_took))})", flush=True)
            else:
                summary["vtk_clips"] = []
                print(f"[{name}] VTK: nothing rendered (no traj.npz)", flush=True)
        except Exception as _e:
            summary["vtk_clips"] = []
            summary.setdefault("artefacts_missing", []).append(f"vtk ({type(_e).__name__})")
            print(f"[{name}] VTK clips skipped: {type(_e).__name__}: {str(_e)[:90]} -- "
                  f"`python vtk_render.py {name}` rebuilds them from traj.npz", flush=True)

        # THE SHAPE FRAMES -- what the Eye now looks at, and what an encoder can read unmodified.
        # Cedric, 13 August: *"we should use this strip.png from now on, put it in the loop."*
        #
        # It REPLACES `strip.png` as the eye's picture and does not delete it: the old sheet is
        # still what `montage.py` tiles and what the 399 runs already on disk carry, and the eye
        # falls back to it when these are absent. What it fixes is measured -- on b_star, lit pixels
        # go 4.5% -> 28.9% of the frame, because three of the old sheet's four rows were a second
        # viewpoint, a per-frame contrast stretch of cell radius, and a cross-section, and because
        # the black per-cell stroke ate the body at 12,000 cells.
        #
        # 4.5 s on top of a 30-40 minute run, measured on b_star.
        try:
            import shape_frames
            _t0 = time.time()
            _sm = shape_frames.render_one(name, quiet=True)
            if _sm:
                summary["shape_frames"] = _sm["n_frames"]
                print(f"[{name}] shape: {_sm['n_frames']} x {_sm['size']}px in "
                      f"{time.time() - _t0:.1f} s -> shape/ + shape_strip.png", flush=True)
            else:
                summary["shape_frames"] = 0
                summary.setdefault("artefacts_missing", []).append("shape (no traj.npz)")
        except Exception as _e:
            summary["shape_frames"] = 0
            summary.setdefault("artefacts_missing", []).append(f"shape ({type(_e).__name__})")
            print(f"[{name}] shape frames skipped: {type(_e).__name__}: {str(_e)[:90]} -- "
                  f"`python shape_frames.py {name}` rebuilds them from traj.npz", flush=True)

    # PREMISES, PASSIVE TIER -- on the run's own recorded series. Unlike the static gate this
    # cannot abort anything (the simulation already happened), so it goes LOUDLY into the record:
    # an Analyst that reads a summary without knowing the chemistry was extinct, or that the cells
    # are stretched 2:1, is interpreting a broken specimen as a result.
    prem = []
    try:
        # read the series back off disk: analyze() has just written metrics.json, and its RETURN
        # value is the summary, not the per-frame table. (First wiring passed `tube` and the
        # passive tier silently reported nothing at all -- a check that quietly does not run is
        # worse than no check, because the record then says the premises held.)
        _ser = _pc._series(name)
        if _ser is None:
            print(f"[{name}] premise check: no series on disk, PASSIVE TIER DID NOT RUN", flush=True)
        prem = _pc.check(cfg, _ser, mech=_pc._mech(name))
        _pc.report(prem, f"{name} (post-run)")
    except Exception as e:
        print(f"[{name}] premise check unavailable: {type(e).__name__}: {e}", flush=True)

    # THE RESERVOIR'S OWN REPORT. cell_divide counts the divisions it REFUSED for want of vertex
    # buffer and flags when the array is full, and until now both died on the mesh: the only way
    # anyone learned a run was capped was the Critic inferring it from n_cells afterwards, or a
    # human noticing the green stop two seconds into a movie.
    # OVER THE WHOLE RUN, not the last frame. Reading hist[-1] reported div_blocked 0 and
    # buf_full False for a run that plateaued at 98.5% of its array at frame 507 of 900 -- by the
    # last frame nothing was even trying to divide, so the frame that would have said so was
    # hundreds of frames back. A flag sampled at the end cannot see an event in the middle.
    summary["div_blocked"] = int(max((h.get("div_blocked") or 0) for h in hist)) if hist else 0
    summary["buf_full"] = bool(any(h.get("buf_full") for h in hist))
    summary["div_blocked_first_frame"] = next(
        (i for i, h in enumerate(hist) if (h.get("div_blocked") or 0) > 0), None)
    # CUMULATIVE DEATHS, LIFTED THE SAME WAY, and it needs lifting for the same reason: a counter
    # on the mesh reaches nobody. It is a MAXIMUM over the run rather than the last frame's value
    # because the array is compacted on every death, so a late frame is not guaranteed to carry
    # the earlier total. The cell COUNT cannot substitute -- r019_02_apop_small went 2,000 -> 3,089
    # with death running, and r019_02_apop_low held at 2,000 with death running and nothing dying.
    summary["n_apop"] = int(max((h.get("n_apop") or 0) for h in hist)) if hist else 0
    # THE CONSERVATION ERROR, SURFACED. `apop_spill` accumulates material a dying cell could not
    # give away without pushing a neighbour outside the integrator's basin. It is monotone, so the
    # last frame carries the total. Recorded because a conservation law nobody can read the error
    # of is indistinguishable from one that works -- which is exactly how the -7.3e11 activator
    # went unnoticed for a whole series.
    summary["apop_spill"] = float(hist[-1].get("apop_spill") or 0.0) if hist else 0.0

    # UNDEFINED IS NOT ZERO, AND THE CONDITION IS DECLARED BY THE METRIC. This used to be a
    # hardcoded list of activator-derived names here, which is the same defect one level up: a rule
    # about a metric, kept somewhere the metric cannot see. `metrics.undefined_in` evaluates each
    # metric's own `requires` predicate against this summary, so a metric declines by being ABSENT
    # rather than by reporting a misleading number, and adding a metric cannot forget to do it.
    #
    # None rather than 0 is the load-bearing choice: `None > 1.3` raises and predict.score turns
    # that into `inconclusive`, while `0 > 1.3` is silently False and scores REFUTED. The language
    # refuses the conclusion for us, but only if we never write the zero.
    import metrics as _M
    _undef = _M.undefined_in(summary)
    for _k in _undef:
        summary[_k] = None
    summary["undefined_metrics"] = sorted(_undef)
    if _undef:
        summary["undefined_why"] = "; ".join(sorted(set(_undef.values())))
        print(f"[{name}] ⚠ {len(_undef)} metric(s) UNDEFINED on this run (not zero): "
              f"{', '.join(sorted(_undef))}", flush=True)
    if summary["buf_full"] or summary["div_blocked"]:
        print(f"[{name}] RESERVOIR FULL at {summary.get('n_cells_final')} cells "
              f"(first refused division at frame {summary.get('div_blocked_first_frame')}) -- "
              f"this run is capped by its array, not by its biology. Everything measured after "
              f"that frame describes the reservoir.", flush=True)
    json.dump({"config": name, "comp_hash": disc.get("comp_hash"),
               "region": disc.get("region"), "summary": summary, "acted": acted,
               "premises": [p.as_dict() for p in prem],
               "premises_broken": [p.pid for p in prem if p.status in ("fail", "error")],
               # CENSORED IS ITS OWN COLUMN. Not broken -- the run is admissible -- but not clean
               # either: a reader that takes n_cells_final at face value from one of these is
               # reading the reservoir. Recorded so the claim checker can refuse a conclusion that
               # rests on the censored quantity, which is the whole reason for keeping the run.
               "premises_censored": [p.pid for p in prem if p.status == "censored"],
               "premises_ablated": [p.pid for p in prem if p.status == "ablation"],
               # HOW FAR IT ACTUALLY GOT. None for a complete run; the frame it was stopped at
               # otherwise. Without this a salvaged 690-frame run and a finished 900-frame one are
               # indistinguishable in the record, and every `_final` metric would silently mean
               # two different things.
               "stopped_early": stopped_early,
               # AND WHY. `stopped_early` alone says a run is short; it does not say whether the
               # clock ran out, a signal arrived, or the tissue hit its own ceiling -- and those
               # are three different findings. A ceiling stop is a RESULT about an overgrowing
               # composition; a signal stop is a fact about the queue.
               "stopped_reason": (_stop.get("why") if stopped_early is not None else None),
               "run_id": rec.run_id},
              open(os.path.join(out_dir, "diag.json"), "w"), indent=1)
    return summary


# --------------------------------------------------------------------------- Q
def quasi_static_Q(cfg, cfg_path, device, protr_before, out_dir, Hf, relax_frames=60):
    """Continue from the END STATE with growth + driver OFF: mechanics and reconnection only.

    A FORCED protrusion collapses (Q -> 0). A GROWN one persists (Q -> 1). This is round 41's
    finding turned into a number, and it is the campaign's primary discriminator.

    THE DEFECT THIS REPLACES. The previous version deleted the growth/driver operators from the
    spec and then called `S.load(spec_q.yaml)` -- which builds a NEW simulation from the seed
    sphere. It therefore relaxed a fresh sphere for 60 frames and reported its elongation, every
    time: Q was 1.014 in 14 of the 16 runs that computed it, whose real `protr_final` spanned
    1.02--2.81. Q carries weight 1.0 in `score_run` and gates `meets_success` at Q >= 2.0, so the
    campaign's own success criterion was unreachable and every forced-vs-grown verdict on record
    was drawn over a constant. The end state is now checkpointed and reloaded.

    AND IT IS VERIFIED. The failure above was silent for weeks because nothing ever asked whether
    the relaxation had started where the run finished. It is asked here, on frame 0, and a
    mismatch returns None rather than a number.
    """
    import copy
    import ckpt
    S, engine_run = _lazy_engine()

    ck = os.path.join(out_dir, "ckpt_end.npz")
    ckpt.save_state(Hf, ck)                                # the end state, positions + topology

    c2 = copy.deepcopy(cfg)
    drop = {"cell_grow", "interface_tension", "cell_chem_seed",
            "cell_divide"}
    seeders = {"mesh_seed", "load_mesh_3d"}             # replaced by the end-state checkpoint
    c2["operators"] = [o for o in c2["operators"] if o["op"] not in drop | seeders]
    c2["schedule"] = [s for s in c2["schedule"] if s not in drop | seeders]
    c2["operators"].insert(0, {"op": "load_mesh_3d", "at": "vertex", "cell_set": "cell",
                               "ckpt": ck, "before_frame": 1})
    c2["schedule"].insert(0, "load_mesh_3d")
    c2["general"]["n_frames"] = relax_frames
    c2["general"]["record_cap"] = relax_frames + 2
    p2 = os.path.join(out_dir, "spec_q.yaml")
    yaml.safe_dump(c2, open(p2, "w"), sort_keys=False)
    try:
        Hq, oq = engine_run(S.load(p2), device=device)
        pos = oq["sets"]["vertex"]["pos"]
        m = getattr(Hq.level("vertex"), "_mesh", {}) or {}
        hq = m.get("hist", [])
        nv0 = hq[0]["Nv"] if hq else pos.shape[1]
        nv = hq[-1]["Nv"] if hq else pos.shape[1]
        # THE GUARD: frame 0 of the relaxation must BE the end of the run. If the checkpoint did
        # not take, this is the seed sphere again and Q is meaningless -- say so, never return it.
        p0 = protr_of(pos[0][:nv0].astype(np.float64))
        if protr_before > 1.05 and abs(p0 - protr_before) > 0.15 * max(protr_before, 1.0):
            print(f"  [Q] REFUSED: relaxation started at protr {p0:.3f} but the run ended at "
                  f"{protr_before:.3f} -- the checkpoint did not take, so this would be the seed "
                  f"sphere relaxing, not the end state. Recording no Q.", flush=True)
            return None
        # M4: Q was final/before -- a RATIO, which the instrument gate showed is perfectly
        # ANTI-correlated with elongation (tau=-1.00): a sphere that never moved scores 1.0.
        # Q must be the ABSOLUTE elongation that SURVIVES relaxation, with the pre-relaxation
        # value reported alongside so the drop is still visible.
        q = protr_of(pos[-1][:nv].astype(np.float64))
        # PRINT the handover, always. The old Q was a constant 1.014 for weeks precisely because
        # nobody could see where the relaxation started; a log line is what makes the guard
        # auditable after the fact instead of only at the moment it fires.
        print(f"  [Q] continued from the end state: protr {p0:.3f} (run ended {protr_before:.3f}, "
              f"{hq[0]['nF'] if hq else '?'} cells) -> relaxed {relax_frames} frames -> Q={q:.3f}",
              flush=True)
        return q
    except Exception as e:
        print(f"  [Q] failed: {type(e).__name__}: {str(e)[:90]}", flush=True)
        return None


# --------------------------------------------------------------------------- artefacts
# The two camera angles every artefact is drawn from. SIDE is the archive/minisite convention;
# TOP exists because a single viewpoint cannot be trusted -- see `render`.
CAM_SIDE = dict(elev=18, azim=30)          # _draw's own default elevation, stated explicitly here
CAM_TOP = dict(elev=88, azim=30)           # near-polar: a tube pointing at the side camera is
#                                            side-on here, so it cannot hide behind the body.


def run_box(fr, pad=1.12):
    """ONE half-width for the WHOLE run: max |coordinate| over every frame, plus headroom.

    THE DEFECT THIS REPLACES. The box used to be `lambda pt: |pt|.max() * 1.12`, evaluated
    INSIDE the per-frame loop -- so every frame was re-fitted to its own extent and the tissue
    filled the same fraction of the axes from first frame to last. A vesicle that doubles in
    radius rendered as a ball of constant apparent size: GROWTH, the quantity these runs exist
    to show, was invisible in every strip and movie in the archive, and the eye had no way to
    tell an inflating sphere from a static one. Okuda's figures use a fixed frame; so do we.

    Measured on the whole trajectory (not just the last frame) because a run can peak and then
    retract -- fitting to the end state would clip the peak straight out of the picture.
    """
    return float(max(np.abs(f[0]).max() for f in fr)) * pad


def render(name, fr, out_dir, n_strip=8, movie_frames=60, movie=True):
    """Strip + movie, matching the minisite convention (black bg, activator LUT).

    THREE rows / TWO 3D panels, not one. A protrusion growing along the side camera's view
    direction projects to a few pixels of foreshortening and reads as "no tube at all" -- the
    single fixed `azim=30` view could therefore hide the exact phenotype the campaign is
    scoring. The near-polar TOP view is 70 degrees off the side view, so a tube invisible in one
    is broadside in the other. Both panels share the same fixed box, so the two views are also
    directly comparable to each other.
    """
    from run_tyssue_vesicle import _draw
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
    # ONE COLOUR SCALE FOR THE RUN, FROM FINITE VALUES ONLY.
    # The scale is computed once and applied to every frame, which is right -- a per-frame
    # rescale makes a decaying pattern and a stable one look identical, and that is how the
    # coral movie hid an extinct field for months. But it was computed with np.percentile over
    # samples INCLUDING any NaN, and a single NaN anywhere makes both bounds NaN, so col()
    # returns NaN for every cell in EVERY frame. Frame 0 is perfectly good and is drawn black
    # because frame 800 diverged. The movie then reports "nothing happened" for a run whose
    # first half was fine -- the artefact says the opposite of the truth, which is worse than
    # having no artefact.
    asamp = np.concatenate([fr[t][2] for t in np.unique(np.linspace(0, T - 1, 12).astype(int))])
    finite = asamp[np.isfinite(asamp)]
    if finite.size:
        lo, hi = float(np.percentile(finite, 5)), float(np.percentile(finite, 99) + 1e-6)
    else:
        lo, hi = 0.0, 1.0
    n_bad = int(asamp.size - finite.size)
    if n_bad:
        # Say it on the artefact itself. A viewer must not have to know the run diverged to
        # read the movie correctly, and a caption is the only place that survives being looked at.
        print(f"[{name}] ⚠ {n_bad}/{asamp.size} sampled activator values are non-finite -- the "
              f"colour scale is taken from the finite ones and NON-FINITE CELLS ARE DRAWN GREY. "
              f"Frames after divergence are not a morphology.", flush=True)

    def col(a):
        a = np.asarray(a, float)
        out = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
        # NaN is not zero. Zero is white (no activator) and would read as a healthy resting
        # cell; the truth is "this cell left the model", so it gets its own value and the
        # renderer paints it grey rather than silently white.
        out[~np.isfinite(a)] = np.nan
        return out

    # ---- ONE box, computed once, held for every frame of both artefacts and both viewpoints.
    # THE GEOMETRY LEAVES THIS FUNCTION, and until now it never did. `frames.npz` holds 17
    # per-frame SCALARS -- 482 KB for a 900-frame run -- and the vertex positions and mesh existed
    # only inside `render`. So when a one-line bug in the scale bar killed strip.png, movie.mp4 and
    # 3d.png across four rounds and 64 runs, there was nothing to re-render FROM: the only way to
    # get a picture of an already-measured run was to spend the GPU again and re-run it.
    #
    # This writes exactly the frames the strip and the movie draw -- the same subsample, not the
    # full trajectory -- so a plot fix costs a re-render instead of a re-simulation. Measured on
    # b_gs_gated_plain: ~60 frames of a 17k-cell mesh is a few tens of MB, against 900 frames of a
    # 50k-cell mesh which would be tens of GB. The cheap half of the choice is the useful half.
    try:
        _keep = np.unique(np.linspace(0, T - 1, min(max(movie_frames, n_strip), T)).astype(int))
        np.savez_compressed(
            os.path.join(out_dir, "traj.npz"),
            ticks=_keep.astype(np.int32),
            **{f"pos_{i}": np.asarray(fr[int(t)][0], np.float32) for i, t in enumerate(_keep)},
            **{f"mesh_{i}": np.asarray(fr[int(t)][1]) for i, t in enumerate(_keep)},
            **{f"act_{i}": np.asarray(fr[int(t)][2], np.float32) for i, t in enumerate(_keep)})
    except Exception as _e:
        print(f"[{name}] traj.npz not written ({type(_e).__name__}: {str(_e)[:70]}) -- this run "
              f"cannot be re-rendered without re-running it", flush=True)

    L3 = run_box(fr)
    L2 = L3 * 2.05
    l_first, l_last = np.abs(fr[0][0]).max() * 1.12, np.abs(fr[-1][0]).max() * 1.12
    print(f"[{name}] camera: FIXED Lbox={L3:.3f} for all {T} frames and both views "
          f"(per-frame autofit would have run {l_first:.3f} -> {l_last:.3f}, "
          f"x{l_last / max(l_first, 1e-9):.2f}: that rescaling is what hid growth)", flush=True)

    # A cell counts as "just divided" for this many division-calls. The movie samples ~60 of
    # ~260 frames, so a 1-2 frame window would be missed more often than not; 4 is short enough to
    # mean "recent" and long enough to actually be seen.
    DIVIDED_WINDOW = 4

    def faults_of(pt, mt):
        """Per-cell `divided` / `broken` masks for the render, computed ONCE per frame.

        Grey used to be the only non-red colour in these movies, and it meant nothing -- it was
        the shaded side of a pale cell, and it was reasonably mistaken for phantom cells. Now the
        two states that actually matter are drawn explicitly: green for a just-divided cell
        (benign; these dominate the "damage" count) and magenta for a genuinely broken one (an
        alarm; zero in every frame measured so far, which is the point of showing it).
        """
        try:
            from diag_tools import mesh_faults
            f = mesh_faults(pt, mt)
            # GREEN = RECENTLY DIVIDED, taken from the division event itself.
            # It used to come from the `sliver` mask (area far below the local mean) and was
            # simply wrong: on a 260-frame run with 101 divisions the sliver count was 0 in every
            # sampled frame, so nothing was ever green. A division makes two roughly equal halves
            # -- a daughter is ~50-70% of its neighbours -- while the sliver test looks below 15%,
            # so it finds DEGENERATE cells, not new ones. `age` is reset to 0 by cell_divide, which
            # is the event we actually mean.
            age = mt.get("age")
            if age is None:
                print(f"[{name}] no per-cell age recorded -- movie drawn WITHOUT the divided "
                      f"colour (older run, or topo_snapshot predates it)", flush=True)
                div = None
            else:
                div = np.asarray(age)[:mt["nF"]] <= DIVIDED_WINDOW
                # ... AND the cell must actually have divided at least once. `age` starts at 0 for
                # every seeded cell, so on its own it paints the whole untouched tissue green in the
                # opening frames -- which is what p1_ph_rd_only's movie showed, in a run where
                # division never fires. Spotted by Cedric watching the movie, not by any check.
                nd = mt.get("ndiv")
                if nd is not None:
                    div = div & (np.asarray(nd)[:mt["nF"]] > 0)
            return div, f["broken"]
        except Exception as e:
            # Not swallowed silently: a missing overlay must announce itself, or a movie with no
            # green looks like a tissue that never divided.
            print(f"[{name}] fault overlay unavailable ({type(e).__name__}: {str(e)[:70]}) -- "
                  f"movie drawn WITHOUT the divided/broken colours", flush=True)
            return None, None

    def classes_of(pt, mt):
        """Per-cell structural class for the cell-type row. Loud if unavailable."""
        try:
            from tissue_analysis import cell_classes
            return cell_classes(pt, mt)
        except Exception as e:
            print(f"[{name}] cell-type row unavailable ({type(e).__name__}: {str(e)[:60]})",
                  flush=True)
            return None

    def inhib_of(mt):
        """Cells whose growth a second morphogen has switched off -- recorded per frame by
        topo_record. None on any run without an inhibitor, which is every run before
        11 August, so the renderer draws nothing extra."""
        d = mt.get("inhib")
        return None if d is None else np.asarray(d)[:mt["nF"]]

    def dying_of(mt):
        """Cells marked to die and not yet extruded -- recorded per frame by topo_record.
        None on any run without apoptosis, which is every run before 9 August."""
        d = mt.get("apop")
        return None if d is None else (np.asarray(d)[:mt["nF"]] > 0)

    def draw3d(ax, pt, mt, a, cam, div=None, brk=None, classes=None):
        _draw(ax, pt, mt, 3.90, azim=cam["azim"], act=col(a), Lbox=L3,
              divided=div, broken=brk, classes=classes, dying=dying_of(mt),
              inhib=inhib_of(mt))
        # _draw hardwires elev=18 as its last statement; re-aim afterwards to get the 2nd view.
        ax.view_init(elev=cam["elev"], azim=cam["azim"])

    # FOUR rows: 3D side, 3D top-down, cell TYPE (blue body / amber branch / yellow tip), and the
    # cross-section. The type row answers a question the activator colouring cannot: is the
    # protrusion a COHERENT structure, or the same number of raised cells scattered about? With
    # several seeded spots it also shows at a glance whether every tube is developing alike.
    fig = plt.figure(figsize=(4.4 * n_strip, 18.0))
    fig.patch.set_facecolor("black")
    for i, t in enumerate([int(round(f * (T - 1))) for f in np.linspace(0, 1, n_strip)]):
        pt, mt, a = fr[t][:3]
        div, brk = faults_of(pt, mt)
        cls = classes_of(pt, mt)
        _a1 = fig.add_subplot(4, n_strip, i + 1, projection="3d")
        draw3d(_a1, pt, mt, a, CAM_SIDE, div, brk)
        _a2 = fig.add_subplot(4, n_strip, n_strip + i + 1, projection="3d")
        draw3d(_a2, pt, mt, a, CAM_TOP, div, brk)
        axc = fig.add_subplot(4, n_strip, 2 * n_strip + i + 1, projection="3d")
        draw3d(axc, pt, mt, a, CAM_SIDE, None, brk, classes=cls)
        # THE STRIP CARRIES THE BAR TOO, and this is the panel that matters most for it: `Read`
        # takes PNG and not mp4, so strip.png -- not the movie -- is what the eye agent actually
        # looks at. A scale bar only on the movie would put the number in front of the one reader
        # who cannot open it.
        for _ax in (_a1, _a2, axc):
            _scalebar(_ax, L3)
        ax3 = fig.add_subplot(4, n_strip, 3 * n_strip + i + 1)
        # NO try/except here. Swallowing the error is exactly the silent-no-op pattern this
        # project keeps being bitten by: the first version caught a TypeError from a wrong
        # signature and rendered a blank row that looked deliberate.
        _cross_screen(ax3, pt, mt, col(a), seed_dir=_cross_axis(pt, None), Lbox=L2)
        _scalebar(ax3, L2)
    # STAMP THE STRIP WHEN THE FIELD IS NOT FINITE. A NaN activator paints magenta per cell now,
    # but a reader still has to infer WHY the picture changed. Say it: the frame the chemistry
    # died, on the image, so the strip accuses the run instead of the plotter.
    try:
        _first_nan = next((t for t in range(T)
                           if not np.all(np.isfinite(np.asarray(fr[t][2], dtype=float)))), None)
    except Exception:
        _first_nan = None
    if _first_nan is not None:
        fig.text(0.5, 0.985, f"CHEMISTRY NOT FINITE FROM FRAME {_first_nan} "
                             f"-- magenta cells are NaN, not measurements",
                 color="#ff19d9", fontsize=13, ha="center", va="top", weight="bold")
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995, wspace=0.02, hspace=0.02)
    fig.savefig(os.path.join(out_dir, "strip.png"), dpi=100, facecolor="black")
    plt.close(fig)

    # THE STRIP IS NOT OPTIONAL. `--no-movie` used to gate this whole function, so skipping the
    # expensive mp4 also threw away the cheap contact sheet -- and the contact sheet is the only
    # detector in this campaign with a perfect record. Every one of the ten defects found on
    # 31 July was found by a human scanning a picture: a ball that shrank, a tissue that went
    # green all at once, a coral that began uniformly red. Saving a minute by suppressing the
    # image removes the one instrument that has never missed. The mp4 may be skipped; the strip
    # may not.
    if not movie:
        print(f"[{name}] artefacts -> {os.path.relpath(out_dir, ROOT)}/strip.png (mp4 skipped)",
              flush=True)
        return

    # ONE PANEL, VTK, WITH THE CELL OUTLINES. Cedric, 21 August: *"change the render of the
    # movie.mp4, use one panel and use vtk mesh"*.
    #
    # The layout below is two matplotlib 3D viewpoints plus a cross-section inset, drawn by the
    # painter's algorithm -- which sorts faces by depth and paints back-facing ones anyway, so on a
    # star's end frame thousands of hidden faces are painted and which one wins a tie changes with
    # the angle. `vtk_render.evolve` is the z-buffered single-panel successor and already exists:
    # same fixed camera for the whole run, so growth within the run is real rather than a camera
    # move. `mesh` rather than `nomesh` because the outlines are what make cell-scale events --
    # who divided, how many cells across a tube -- legible in a movie.
    #
    # The matplotlib layout stays as the fallback: it needs no VTK and no GPU-side render, so a
    # node where pyvista cannot open a window still produces a movie rather than nothing.
    _mp4 = os.path.join(out_dir, "movie.mp4")
    _movie_done = False
    try:
        import vtk_render as _V
        _V.evolve(name, "mesh", _mp4)
        print(f"[{name}] movie.mp4 <- vtk evolve, one panel, mesh", flush=True)
        _movie_done = True
    except Exception as _e:
        print(f"[{name}] movie: VTK failed ({type(_e).__name__}: {str(_e)[:60]}), "
              f"falling back to the matplotlib layout", flush=True)

    # movie: the two 3D viewpoints side by side + the cross-section inset (bottom-right).
    # Laid out here rather than via make_movie_axes, which only makes the single-view layout.
    figm = plt.figure(figsize=(10.0, 5.2))
    figm.patch.set_facecolor("black")
    axs = figm.add_subplot(1, 2, 1, projection="3d")
    axt = figm.add_subplot(1, 2, 2, projection="3d")
    figm.subplots_adjust(0, 0, 1, 1, wspace=0.0)
    axin = figm.add_axes([0.83, 0.0, 0.17, 0.34])
    axin.set_facecolor("none")
    axin.patch.set_alpha(0.0)
    keep = np.unique(np.linspace(0, T - 1, min(movie_frames, T)).astype(int))
    if not _movie_done:
        wri = FFMpegWriter(fps=10, metadata={"title": name})
        with wri.saving(figm, os.path.join(out_dir, "movie.mp4"), dpi=85):
            for t in keep:
                pt, mt, a = fr[int(t)][:3]
                div, brk = faults_of(pt, mt)
                draw3d(axs, pt, mt, a, CAM_SIDE, div, brk)
                draw3d(axt, pt, mt, a, CAM_TOP, div, brk)
                # _draw calls ax.clear(), which drops the label -- re-stamp it every frame.
                axs.text2D(0.02, 0.96, "side  elev 18", transform=axs.transAxes, color="w",
                           fontsize=9)
                axt.text2D(0.02, 0.96, "top  elev 88", transform=axt.transAxes, color="w", fontsize=9)
                # THE SCALE. `_draw` clears the axes every frame, so this is re-stamped like the
                # labels above. Both 3D panels share L3, the one box held for the whole run.
                _scalebar(axs, L3)
                _scalebar(axt, L3)
                _cross_screen(axin, pt, mt, col(a), seed_dir=_cross_axis(pt, None),
                              Lbox=L2)              # cross-section, minisite convention
                axin.axis("off")
                _scalebar(axin, L2)                 # its own box: the inset is 2.05x the 3D view
                wri.grab_frame()
    plt.close(figm)
    # 3d.png -- THE LAST FRAME, ALONE, AT FULL SIZE. Cedric, 8 August: "add one 3d.png end frame
    # of mp4". The strip is eight thumbnails across a 4-row grid, so the final state is one small
    # panel among 32; the movie ends on it but cannot be opened by `Read`, which is how every
    # automated reader in this loop looks at a picture. This is the run's outcome as a single
    # image, drawn with the same fixed camera and the same scale bar as everything else.
    # DRAWN BY VTK SINCE 20 AUGUST, matplotlib only if that fails. This was the last picture in the
    # project still drawn by the painter's-algorithm renderer, and it is the MOST READ one: every
    # montage tiles it, the forecast graph puts it in each node, and `Read` opens it. matplotlib
    # sorts faces by depth and draws back-facing ones anyway, so on a star's end frame thousands of
    # hidden faces are painted and which one wins a tie depends on the angle; a z-buffer cannot have
    # that argument. Measured on the same mesh: 0.5 s a frame against 9.3, and 28.9% of pixels lit
    # against 4.5%.
    #
    # FLAT SHADING HERE AND NOWHERE ELSE. Cedric refused flat for the movies on 12 August -- a
    # curved arm reads as a faceted cone -- and asked for it here, which is the case it fits: a
    # still that will be tiled at ~190 px in a montage, where a 0.4 px cell outline is a grey wash
    # and a smooth body is a featureless blob, while facets scale with the cells they belong to.
    _vtk_ok = False
    try:
        import vtk_render as _V
        _V.still(name, style="flat", out=os.path.join(out_dir, "3d.png"))
        _vtk_ok = True
    except Exception as _e:
        print(f"[{name}] 3d.png: VTK failed ({type(_e).__name__}: {str(_e)[:60]}), "
              f"falling back to matplotlib", flush=True)
    try:
        if _vtk_ok:
            raise StopIteration                       # the picture is written; skip the fallback
        figE = plt.figure(figsize=(7.0, 7.0)); figE.patch.set_facecolor("black")
        axE = figE.add_subplot(111, projection="3d")
        ptE, mtE, aE = fr[-1][:3]
        divE, brkE = faults_of(ptE, mtE)
        draw3d(axE, ptE, mtE, aE, CAM_SIDE, divE, brkE)
        axE.text2D(0.02, 0.96, f"{name}  frame {T - 1}", transform=axE.transAxes,
                   color="w", fontsize=10)
        _scalebar(axE, L3)
        figE.savefig(os.path.join(out_dir, "3d.png"), dpi=110, facecolor="black",
                     bbox_inches="tight")
        plt.close(figE)
    except StopIteration:
        pass
    except Exception as _e:
        # SAID, NOT SWALLOWED: a missing 3d.png reads downstream as "the run had no end state".
        print(f"[{name}] 3d.png not written: {type(_e).__name__}: {str(_e)[:80]}", flush=True)

    print(f"[{name}] artefacts -> {os.path.relpath(out_dir, ROOT)}/"
          f"{{strip.png,movie.mp4,3d.png}}", flush=True)
    # THE CAMERA IS EVIDENCE, so it leaves this function. The eye is shown a picture drawn with a
    # box held fixed for the whole run and identical in structure for every run in the batch; the
    # only thing separating a 2,000-cell sphere from a 53,000-cell one in that picture is the
    # number on the scale bar. Returning it lets the summary carry it and the eye be told it.
    return {"camera_lbox": round(float(L3), 3), "camera_lbox_cross": round(float(L2), 3)}


# --------------------------------------------------------------------------- mechanics
def mechanics(name, fr, cfg, out_dir, n=24):
    """Per-cell FORCE / PRESSURE / TENSION / MIGRATION, from the trajectory we already have.

    `analyze_forces.run()` re-runs the simulation to get these; we do not need to. `cell_mechanics`
    is a pure function of (positions, half-edge table, per-cell targets), and topo_record
    already stores A0/P0/V0f in the mesh history -- so the fields come for free from the frames
    on disk. No doubled compute, and every job carries its own mechanical analysis.

    The campaign-critical output is `p_ratio`: mean pressure in the PROTRUDING cells divided by
    mean pressure in the BODY. Round 41 diagnosed our tube as forced rather than grown precisely
    from this signature -- pressure ~3 concentrated in the tube while the body sat idle. A
    growth-driven equilibrium tube should approach 1. It is the direct measurement of the
    campaign objective, and it is differentiable, so it is also the Loop-II objective.
    """
    import torch
    from analyze_forces import cell_mechanics
    se = next((o for o in cfg["operators"] if o["op"] == "cell_mechanics"), {})
    kA, kP = se.get("K_A", 1.0), se.get("K_P", 1.0)
    kV = se.get("K_V", se.get("k_v", 4.0))
    Lam, Gam = se.get("Lambda", 0.2), se.get("Gamma", se.get("gamma", 0.05))

    T = len(fr)
    idx = np.unique(np.linspace(0, T - 1, min(n, T)).astype(int))
    rows, prev_cen = [], None
    for t in idx:
        pos, mt, act = fr[int(t)][:3]
        if mt.get("A0") is None or mt.get("V0f") is None:
            continue
        x = torch.tensor(pos, dtype=torch.float32)
        es = torch.as_tensor(mt["E_srce"], dtype=torch.long)
        et = torch.as_tensor(mt["E_trgt"], dtype=torch.long)
        ef = torch.as_tensor(mt["E_face"], dtype=torch.long)
        nF = int(mt["nF"])
        A0 = torch.as_tensor(mt["A0"], dtype=torch.float32)[:nF]
        P0 = torch.as_tensor(mt["P0"], dtype=torch.float32)[:nF]
        V0 = torch.as_tensor(mt["V0f"], dtype=torch.float32)[:nF]
        f, area, perim, vf, pres, tens, cen = cell_mechanics(
            x, es, et, ef, nF, A0, P0, V0, kA, kP, kV, Lam, Gam)
        cen_np = cen.numpy()
        r = np.linalg.norm(cen_np - cen_np.mean(0), axis=1)
        prot = r > 1.3 * np.median(r)                      # the protruding cells (tissue_analysis defn)
        fmag = np.linalg.norm(f.numpy(), axis=1)
        # `min` OF BOTH, and this sliced only the new array before. It assumed the cell count can
        # only GROW -- true while division was the sole topology change -- so introducing the Die
        # family broke it immediately: 1,999 cells against the previous 2,000 is not broadcastable.
        # Across a death the rows also no longer name the same cells (`keep` renumbers), so this is
        # a mean over a population whose membership shifted, not a per-cell displacement. It is
        # reported as migration and should not be read as one on a run where cells die.
        _n = min(len(cen_np), len(prev_cen)) if prev_cen is not None else 0
        vel = (np.linalg.norm(cen_np[:_n] - prev_cen[:_n], axis=1).mean() if _n else 0.0)
        prev_cen = cen_np
        pr = pres.numpy()
        rows.append(dict(t=int(t), n_cells=nF,
                         force_mean=float(fmag.mean()),
                         p_body=float(np.abs(pr[~prot]).mean()) if (~prot).any() else 0.0,
                         p_tube=float(np.abs(pr[prot]).mean()) if prot.any() else 0.0,
                         tension_mean=float(tens.numpy().mean()),
                         migration=float(vel), n_protruding=int(prot.sum())))
    if not rows:
        print(f"[{name}] mechanics: mesh history carries no per-cell targets -- skipped",
              flush=True)
        return {}
    last = rows[-1]
    pb = max(last["p_body"], 1e-9)
    summ = {"mech_force_mean": round(last["force_mean"], 4),
            "mech_p_body": round(last["p_body"], 4),
            "mech_p_tube": round(last["p_tube"], 4),
            "mech_p_ratio": round(last["p_tube"] / pb, 3),     # ~3 forced, ~1 grown (R41)
            "mech_tension_mean": round(last["tension_mean"], 4),
            "mech_migration": round(float(np.mean([r["migration"] for r in rows[1:]] or [0])), 5)}
    np.savez(os.path.join(out_dir, "mechanics.npz"),
             **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    _plot_mechanics(rows, name, out_dir)
    print(f"[{name}] mechanics: p_tube/p_body = {summ['mech_p_ratio']} "
          f"(~3 = forced protrusion, ~1 = growth-driven equilibrium)", flush=True)
    return summ


def _plot_mechanics(rows, name, out_dir):
    t = [r["t"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(9, 6), facecolor="black")
    for a in ax.ravel():
        a.set_facecolor("black")
        for sp in a.spines.values():
            sp.set_color("0.5")
        a.tick_params(colors="0.7", labelsize=7)
    ax[0, 0].plot(t, [r["force_mean"] for r in rows], color="#4da6ff")
    ax[0, 0].set_title("force  $\\|-\\nabla U\\|$", color="w", fontsize=9)
    ax[0, 1].plot(t, [r["p_body"] for r in rows], color="w", label="body")
    ax[0, 1].plot(t, [r["p_tube"] for r in rows], color="#ff4d4d", label="protruding")
    ax[0, 1].legend(fontsize=6, facecolor="black", labelcolor="w", edgecolor="0.4")
    ax[0, 1].set_title("pressure  $2K_V(V_0-v)$", color="w", fontsize=9)
    ax[1, 0].plot(t, [r["tension_mean"] for r in rows], color="#ffd24d")
    ax[1, 0].set_title("cortical tension", color="w", fontsize=9)
    ax[1, 1].plot(t, [r["migration"] for r in rows], color="#7bd67b")
    ax[1, 1].set_title("migration (centroid displacement)", color="w", fontsize=9)
    fig.suptitle(name, color="w", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mechanics.png"), dpi=110, facecolor="black")
    plt.close(fig)


# --------------------------------------------------------------------------- VLM caption
def describe(name, out_dir, n_frames=8):
    """Caption the run's movie with the local VLM and write it INTO the job's log folder.

    Every job therefore carries its own evidence triple: movie.mp4 + strip.png +
    description.txt. Captioning is never deferred to an end pass and never disabled -- the
    caption is simultaneously documentation, the Watcher's input, and a semantic regression
    test (validation ladder L7: the numbers pass but the picture is wrong).
    """
    mp4 = os.path.join(out_dir, "movie.mp4")
    dst = os.path.join(out_dir, "description.txt")
    if not os.path.exists(mp4):
        print(f"[{name}] no movie to describe", flush=True)
        return None
    if os.path.exists(dst):
        return dst
    vlm = os.path.join(ROOT, "VLLM")
    sys.path.insert(0, vlm)
    try:
        import describe_video as DV
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        dev = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
        proc = AutoProcessor.from_pretrained(DV.GEMMA)
        model = AutoModelForMultimodalLM.from_pretrained(DV.GEMMA, dtype="bfloat16",
                                                         device_map=dev)
        txt = DV.describe_one(proc, model, mp4, n_frames)
    except Exception as e:
        # NOT silent: a missing caption is recorded as such, so the ledger can tell "no caption"
        # from "caption agreed". A silent skip would let a run look Watcher-approved.
        txt = None
        print(f"[{name}] VLM caption UNAVAILABLE: {type(e).__name__}: {str(e)[:110]}", flush=True)
    with open(dst, "w") as f:
        f.write(txt if txt else "UNAVAILABLE -- no caption was produced for this run.\n")
    print(f"[{name}] description -> {os.path.relpath(dst, ROOT)}"
          + ("" if txt else "  (UNAVAILABLE)"), flush=True)
    return dst


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
