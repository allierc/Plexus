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
    import tyssue_shape_to_chem                                       # noqa: F401
    import plexus.schema as S
    from plexus.engine import run as engine_run
    import instrument
    instrument.install()                       # D4: every operator now reports whether it acted
    return S, engine_run


# --------------------------------------------------------------------------- D3: assert alignment
def check_alignment(posf, hist, name=""):
    """The phantom-result guard. Positions and topology MUST be the same length."""
    if not hist:
        raise AssertionError(
            f"[D3] no topology history recorded for {name}: every frame would fall back to the "
            f"SEED mesh, so late-frame coordinates would be read against frame-0 connectivity. "
            f"Schedule topo_snapshot_3d every=1; do not fall back.")
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
    """percentile(r,95)/median(r) about the TISSUE CENTROID. NOT tube_len/tube_diam.

    The formula now lives in tube_analysis.protrusion_ratio and is shared with `ta_protr`, so the
    two cannot silently become different quantities again (they did: ta_* measured radius from the
    world origin, this one from the centroid, and both were called `protr`).
    """
    from tube_analysis import protrusion_ratio
    return protrusion_ratio(np.linalg.norm(pos - pos.mean(0), axis=1))


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
        q = quasi_static_Q(cfg, cfg_path, device, protr_before=fm["protr"][-1], out_dir=out_dir,
                           Hf=Hf)

    # --------------------------------------------------------------- the REAL tube metrics
    # tube_analysis is the archive's own metric bank. Comparing against archived numbers requires
    # ITS definitions, not look-alikes of our own.
    tube = {}
    try:
        from tube_analysis import analyze
        samp = np.unique(np.linspace(0, T - 1, min(40, T)).astype(int))
        # red_frac must be thresholded at the GROWTH OPERATOR'S OWN switch, not at the midpoint of
        # the activator's current range. The relative version is scale-free and therefore blind --
        # it read exactly 0.070 on every one of 40 frames while the pattern changed under it.
        a_sw = next((float(o["a_sw"]) for o in cfg.get("operators", [])
                     if o.get("op") == "morphogen_growth_3d" and "a_sw" in o), None)
        tube = analyze([(int(t), fr[t][0], fr[t][1], fr[t][2]) for t in samp], out_dir,
                       a_sw=a_sw) or {}
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
    hz_i = len(fm["protr"]) - 1
    try:
        import curve_shape as _CS
        _mn = os.path.join(out_dir, "metrics.npz")
        if os.path.exists(_mn):
            _z = np.load(_mn)
            if "broken_n" in _z.files:
                horizon = _CS.evidence_horizon({}, {"broken_n": _z["broken_n"]},
                                               _z["frame"] if "frame" in _z.files else None)
                if horizon.get("horizon") is not None and not horizon.get("complete", True):
                    hz_i = min(hz_i, max(0, int(horizon["horizon"])))
            else:
                horizon = {"horizon": None, "why": "metrics.npz carries no broken_n"}
        else:
            horizon = {"horizon": None, "why": "no metrics.npz (tube_analysis did not run)"}
    except Exception as e:
        # Loud, not silent: if the horizon cannot be computed we must not quietly fall back to
        # scoring the whole run, because that is the behaviour being fixed.
        horizon = {"horizon": None, "why": f"{type(e).__name__}: {str(e)[:90]}"}
    if horizon.get("horizon") is None:
        print(f"[{name}] ⚠ no evidence horizon ({horizon['why']}) -- peak/final are taken over "
              f"ALL {len(fm['protr'])} frames, which is the un-truncated behaviour", flush=True)
    elif hz_i < len(fm["protr"]) - 1:
        print(f"[{name}] evidence horizon at frame {horizon['horizon']}: peak/final taken over "
              f"the first {hz_i + 1} of {len(fm['protr'])} frames", flush=True)

    _valid = fm["protr"][:hz_i + 1] or fm["protr"]

    # retention = final/peak aspect. A FORCED protrusion peaks then collapses (low retention);
    # an EQUILIBRIUM one holds (high). Computable from the archived per-frame table for every
    # run without re-simulating -- the D7 payoff -- and a cheap proxy for the full Q test.
    _pk = max(_valid) if _valid else 0.0
    retention = (_valid[-1] / _pk) if _pk > 1e-9 else 0.0

    summary = {"saturated": bool(saturated), "inert_operators": inert,
               "retention": round(retention, 3),
               "valid_evidence": bool(not inert and not saturated),
               "protr_final": round(_valid[-1], 3),          # last VALID frame, not last frame
               "protr_peak": round(max(_valid), 3),           # over VALID frames only
               "horizon_frame": horizon.get("horizon"),
               "horizon_why": horizon.get("why"),
               "first_damage_frame": horizon.get("first_damage"),
               "valid_frac": horizon.get("valid_frac", 1.0),
               # kept so the truncation is auditable and the change is visible in the record
               "protr_peak_untruncated": round(max(fm["protr"]), 3),
               "protr_final_untruncated": round(fm["protr"][-1], 3),
               "n_cells_final": int(fm["n_cells"][-1]),
               "red_frac_final": round(fm["red_frac"][-1], 3),
               "act_max_final": round(fm["act_max"][-1], 3),
               "frames": T, "wall_s": round(wall, 1)}
    summary.update(tube)                       # the archive's own definitions, for comparison
    try:
        summary.update(mechanics(name, fr, cfg, out_dir))   # force / stress / tension / migration
    except Exception as e:
        print(f"[{name}] mechanics FAILED: {type(e).__name__}: {str(e)[:110]}", flush=True)
    if q is not None:
        summary["Q_protr_after_relax"] = round(q, 3)          # ABSOLUTE, not a ratio (M4)
        summary["Q_drop"] = round(fm["protr"][-1] - q, 3)     # how much did NOT survive
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
            render(name, fr, out_dir, movie=movie)
            # Captioning is NOT done here. The cluster environment has no `transformers`, so an
            # in-job caption fails on exactly the runs a long campaign produces -- leaving the
            # Watcher blind where it matters most. caption_wave.py does it on the devcontainer
            # side with ONE model load per wave, as part of closing the round.
        except Exception as e:
            print(f"[{name}] render failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

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

    json.dump({"config": name, "comp_hash": disc.get("comp_hash"),
               "region": disc.get("region"), "summary": summary, "acted": acted,
               "premises": [p.as_dict() for p in prem],
               "premises_broken": [p.pid for p in prem if p.status in ("fail", "error")],
               "premises_ablated": [p.pid for p in prem if p.status == "ablation"],
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
    drop = {"morphogen_growth_3d", "vesicle_growth", "rd_interface_tension", "cell_rd_seed",
            "divide_3d"}
    seeders = {"seed_mesh_3d", "load_mesh_3d"}             # replaced by the end-state checkpoint
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
    return float(max(np.abs(pt).max() for pt, _, _ in fr)) * pad


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
    asamp = np.concatenate([fr[t][2] for t in np.unique(np.linspace(0, T - 1, 12).astype(int))])
    lo, hi = float(np.percentile(asamp, 5)), float(np.percentile(asamp, 99) + 1e-6)
    col = lambda a: np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)

    # ---- ONE box, computed once, held for every frame of both artefacts and both viewpoints.
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
            from tyssue_diag import mesh_faults
            f = mesh_faults(pt, mt)
            # GREEN = RECENTLY DIVIDED, taken from the division event itself.
            # It used to come from the `sliver` mask (area far below the local mean) and was
            # simply wrong: on a 260-frame run with 101 divisions the sliver count was 0 in every
            # sampled frame, so nothing was ever green. A division makes two roughly equal halves
            # -- a daughter is ~50-70% of its neighbours -- while the sliver test looks below 15%,
            # so it finds DEGENERATE cells, not new ones. `age` is reset to 0 by divide_3d, which
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
            from tube_analysis import cell_classes
            return cell_classes(pt, mt)
        except Exception as e:
            print(f"[{name}] cell-type row unavailable ({type(e).__name__}: {str(e)[:60]})",
                  flush=True)
            return None

    def draw3d(ax, pt, mt, a, cam, div=None, brk=None, classes=None):
        _draw(ax, pt, mt, 3.90, azim=cam["azim"], act=col(a), Lbox=L3,
              divided=div, broken=brk, classes=classes)
        # _draw hardwires elev=18 as its last statement; re-aim afterwards to get the 2nd view.
        ax.view_init(elev=cam["elev"], azim=cam["azim"])

    # FOUR rows: 3D side, 3D top-down, cell TYPE (blue body / amber branch / yellow tip), and the
    # cross-section. The type row answers a question the activator colouring cannot: is the
    # protrusion a COHERENT structure, or the same number of raised cells scattered about? With
    # several seeded spots it also shows at a glance whether every tube is developing alike.
    fig = plt.figure(figsize=(4.4 * n_strip, 18.0))
    fig.patch.set_facecolor("black")
    for i, t in enumerate([int(round(f * (T - 1))) for f in np.linspace(0, 1, n_strip)]):
        pt, mt, a = fr[t]
        div, brk = faults_of(pt, mt)
        cls = classes_of(pt, mt)
        draw3d(fig.add_subplot(4, n_strip, i + 1, projection="3d"), pt, mt, a, CAM_SIDE, div, brk)
        draw3d(fig.add_subplot(4, n_strip, n_strip + i + 1, projection="3d"), pt, mt, a, CAM_TOP, div, brk)
        axc = fig.add_subplot(4, n_strip, 2 * n_strip + i + 1, projection="3d")
        draw3d(axc, pt, mt, a, CAM_SIDE, None, brk, classes=cls)
        ax3 = fig.add_subplot(4, n_strip, 3 * n_strip + i + 1)
        # NO try/except here. Swallowing the error is exactly the silent-no-op pattern this
        # project keeps being bitten by: the first version caught a TypeError from a wrong
        # signature and rendered a blank row that looked deliberate.
        _cross_screen(ax3, pt, mt, col(a), seed_dir=_cross_axis(pt, None), Lbox=L2)
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
    wri = FFMpegWriter(fps=10, metadata={"title": name})
    with wri.saving(figm, os.path.join(out_dir, "movie.mp4"), dpi=85):
        for t in keep:
            pt, mt, a = fr[int(t)]
            div, brk = faults_of(pt, mt)
            draw3d(axs, pt, mt, a, CAM_SIDE, div, brk)
            draw3d(axt, pt, mt, a, CAM_TOP, div, brk)
            # _draw calls ax.clear(), which drops the label -- re-stamp it every frame.
            axs.text2D(0.02, 0.96, "side  elev 18", transform=axs.transAxes, color="w",
                       fontsize=9)
            axt.text2D(0.02, 0.96, "top  elev 88", transform=axt.transAxes, color="w", fontsize=9)
            _cross_screen(axin, pt, mt, col(a), seed_dir=_cross_axis(pt, None),
                          Lbox=L2)              # cross-section, minisite convention
            axin.axis("off")
            wri.grab_frame()
    plt.close(figm)
    print(f"[{name}] artefacts -> {os.path.relpath(out_dir, ROOT)}/{{strip.png,movie.mp4}}",
          flush=True)


# --------------------------------------------------------------------------- mechanics
def mechanics(name, fr, cfg, out_dir, n=24):
    """Per-cell FORCE / PRESSURE / TENSION / MIGRATION, from the trajectory we already have.

    `analyze_forces.run()` re-runs the simulation to get these; we do not need to. `cell_mechanics`
    is a pure function of (positions, half-edge table, per-cell targets), and topo_snapshot_3d
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
    se = next((o for o in cfg["operators"] if o["op"] == "shape_energy_3d"), {})
    kA, kP = se.get("K_A", 1.0), se.get("K_P", 1.0)
    kV = se.get("K_V", se.get("k_v", 4.0))
    Lam, Gam = se.get("Lambda", 0.2), se.get("Gamma", se.get("gamma", 0.05))

    T = len(fr)
    idx = np.unique(np.linspace(0, T - 1, min(n, T)).astype(int))
    rows, prev_cen = [], None
    for t in idx:
        pos, mt, act = fr[int(t)]
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
        prot = r > 1.3 * np.median(r)                      # the protruding cells (tube_analysis defn)
        fmag = np.linalg.norm(f.numpy(), axis=1)
        vel = (np.linalg.norm(cen_np[:len(prev_cen)] - prev_cen, axis=1).mean()
               if prev_cen is not None and len(prev_cen) else 0.0)
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
