#!/usr/bin/env python
"""p1v_verify.py -- ADVERSARIAL RE-RUN of probe A-percell.

Three things probe A asserted that its own artefacts do not establish:

  1. "the single most favourable cell of 100 is worth only 2.56 certified steps".
     A scored 5 of 100 cells, selected by ONE-FRAME max|dx| at tick 165. Its own ceiling table
     shows the beat score is nearly uncorrelated with displacement (cell55 gain 0.2519 px rms ->
     0.14 steps; cell66 gain 0.2233 px rms -> 2.56 steps), so that proxy cannot be trusted to find
     the maximum. Here: EVERY one of the 100 cells, gain and E, full 150-frame beat, same tick,
     same tracers, same floors.

  2. "2.56 steps vs a 1.288-step divergence null = 1.99x". The null is ONE kick seed. If the null
     at fixed displacement has the same scatter the signal does, the ratio is noise. Here: the
     same kick amplitude at 8 seeds.

  3. the ladder ratio gain/E = 0.200 (substep) / 0.432 (frame). Re-run verbatim.

Everything reuses probe A's own helpers (p1a_percell.run / theta_vectors / disp_stats) so the
reading surface, the rollout order and the floors are identical -- only the set of cells changes.

usage: PYTHONPATH=/workspace/Plexus/src python p1v_verify.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
DISC = "/workspace/Plexus/discovery_cardio_mpm"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/workspace/Plexus/src", ALG, HERE, DISC):
    sys.path.insert(0, p)

import crash_test as CT                                               # noqa: E402
import accept as AC                                                   # noqa: E402
import metrics as MET                                                 # noqa: E402
from p1a_percell import clock_of, disp_stats, run, theta_vectors      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=180)
    ap.add_argument("--pulse-tick", type=int, default=165)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--e-lo", type=float, default=40.0)
    ap.add_argument("--e-hi", type=float, default=220.0)
    ap.add_argument("--g-lo", type=float, default=0.5)
    ap.add_argument("--g-hi", type=float, default=1.5)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--beat-frames", type=int, default=150)
    ap.add_argument("--plant-seed", type=int, default=2026)
    ap.add_argument("--stages", default="ladder,null,gain,E")
    ap.add_argument("--tag", default="p1v")
    args = ap.parse_args()
    stages = set(args.stages.split(","))

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    out_path = os.path.join(HERE, f"{args.tag}_verify.json")
    R = {"argv": vars(args)}

    def dump():
        json.dump(R, open(out_path, "w"), indent=1, default=str)
        open(os.path.join(HERE, f"{args.tag}_verify.log"), "w").write("\n".join(lines) + "\n")

    floors = AC.working_floors()
    R["floors"] = {n: floors[n] for n in floors}
    G = args.beat_frames
    pe = MET.REGISTRY["peak_excursion"]
    t_start = time.time()

    with torch.no_grad():
        # ---------------------------------------------------------------- the ladder, tick 165 --
        if "ladder" in stages:
            a2 = argparse.Namespace(**vars(args))
            a2.warmup = args.pulse_tick
            syP, _ = CT.plant_and_warm(a2, log, seed=args.plant_seed)
            C = syP.C
            t_l = time.time()
            base_sub, _ = run(syP, syP.E_true, syP.gain_true, args.pulse_tick, 0, n_sub_extra=1)
            base_frm, _ = run(syP, syP.E_true, syP.gain_true, args.pulse_tick, 1)
            lad = {}
            for c in range(1, C + 1):
                lad[c] = {}
                for param in ("E", "gain"):
                    E, g = theta_vectors(syP, cell=c, param=param, frac=args.frac)
                    xs, _ = run(syP, E, g, args.pulse_tick, 0, n_sub_extra=1)
                    xf, _ = run(syP, E, g, args.pulse_tick, 1)
                    lad[c][param] = {"substep_max_px": disp_stats(xs, base_sub)["max_px"],
                                     "frame_max_px": disp_stats(xf, base_frm)["max_px"]}
            L = {"tick": args.pulse_tick, "clock": clock_of(args.pulse_tick), "per_cell": lad}
            for cad in ("substep", "frame"):
                e = np.array([lad[c]["E"][f"{cad}_max_px"] for c in range(1, C + 1)])
                g = np.array([lad[c]["gain"][f"{cad}_max_px"] for c in range(1, C + 1)])
                L[cad] = {"median_E_px": float(np.median(e)), "median_gain_px": float(np.median(g)),
                          "ratio_of_medians": float(np.median(g) / np.median(e)),
                          "max_E_px": float(e.max()), "max_gain_px": float(g.max()),
                          "n_E_above_0.1px": int((e > 0.1).sum()),
                          "n_gain_above_0.1px": int((g > 0.1).sum())}
                log(f"[LADDER {cad:<8s}] E med {np.median(e):.4e} max {e.max():.4e}   "
                    f"gain med {np.median(g):.4e} max {g.max():.4e}   "
                    f"RATIO of medians {np.median(g)/np.median(e):.4f}")
            R["ladder_pulse"] = L
            log(f"[ladder] {time.time()-t_l:.0f} s")
            dump()
            del syP
            torch.cuda.empty_cache()

        # ---------------------------------------------------------- the campaign tick, tick 180 --
        a2 = argparse.Namespace(**vars(args))
        a2.warmup = args.warmup
        sy, _ = CT.plant_and_warm(a2, log, seed=args.plant_seed)
        C = sy.C
        W = args.warmup
        tracers = {m: CT.tracer_indices(sy.x0, CT.probe_points(m))
                   for m in (MET.MARGIN_SAFE, MET.MARGIN_INHERITED)}
        t_b = time.time()
        x_base, tr_base = run(sy, sy.E_true, sy.gain_true, W, G, tracers=tracers)
        real = tr_base[MET.MARGIN_SAFE].cpu().numpy()
        log(f"[baseline tick {W}] {G} frames in {time.time()-t_b:.0f} s; "
            f"peak_excursion {float(np.median(pe.reading(real))):.6g}")
        R["baseline"] = {"tick": W, "clock": clock_of(W), "frames": G,
                         "reference_peak_excursion": float(np.median(pe.reading(real))),
                         "seconds": time.time() - t_b}
        dump()

        def score(sim):
            one = AC.score_one(sim, real, floors)
            st = {n: one[n]["steps"] for n in AC.CERTIFIED}
            live = [s for s in st.values() if s is not None]
            return st, (max(live) if live else None)

        # ---------------------------------------------------------------- 2. the null, 8 seeds --
        if "null" in stages:
            NU = {}
            for js in range(1, 9):
                t_p, _, _ = CT.rollout(sy, sy.theta_true, W, G, tracers, jitter=1e-4,
                                       jitter_seed=js)
                sim = t_p[MET.MARGIN_SAFE].cpu().numpy()
                d = disp_stats(sy.p.get("pos").clone(), x_base)
                st, w = score(sim)
                NU[js] = {"kick_px": 1e-4 * 1024, "final_rms_px": d["rms_px"],
                          "final_max_px": d["max_px"], "steps": st, "worst": w}
                log(f"  [null seed {js}] rms {d['rms_px']:.4f} px  peak_exc "
                    f"{st['peak_excursion']:.4f}  WORST {w:.4f} steps")
            w = np.array([NU[j]["worst"] for j in NU])
            R["null_seeds"] = {"jitter": 1e-4, "rows": NU,
                               "worst_min": float(w.min()), "worst_median": float(np.median(w)),
                               "worst_max": float(w.max()),
                               "spread_ratio": float(w.max() / max(w.min(), 1e-30))}
            log(f"  NULL over 8 kick seeds: min {w.min():.4f}  med {np.median(w):.4f}  "
                f"max {w.max():.4f}  spread {w.max()/max(w.min(),1e-30):.2f}x")
            dump()

        # ------------------------------------------------- 1. EVERY cell, full beat, both params -
        for param in ("gain", "E"):
            if param not in stages:
                continue
            rows = {}
            t_p0 = time.time()
            for c in range(1, C + 1):
                E, g = theta_vectors(sy, cell=c, param=param, frac=args.frac)
                x_p, tr_p = run(sy, E, g, W, G, tracers=tracers)
                sim = tr_p[MET.MARGIN_SAFE].cpu().numpy()
                st, w = score(sim)
                d = disp_stats(x_p, x_base)
                rows[c] = {"steps": st, "worst": w, "final_rms_px": d["rms_px"],
                           "final_max_px": d["max_px"],
                           "peak_excursion_sim": float(np.median(pe.reading(sim)))}
                if c % 10 == 0 or c == 1:
                    el = time.time() - t_p0
                    log(f"  [{param} cell {c:>3d}] worst {w:.4f} steps  rms {d['rms_px']:.4f} px "
                        f"  [{el:.0f} s, eta {el/c*(C-c):.0f} s]")
                    R[f"allcell_{param}"] = {"partial_through": c, "rows": rows}
                    dump()
            w = np.array([rows[c]["worst"] for c in rows])
            order = np.argsort(-w) + 1
            R[f"allcell_{param}"] = {
                "tick": W, "rows": rows, "n": int(C),
                "worst_min": float(w.min()), "worst_median": float(np.median(w)),
                "worst_max": float(w.max()),
                "argmax_cell": int(order[0]),
                "top10": [{"cell": int(c), "worst": float(rows[int(c)]["worst"]),
                           "rms_px": rows[int(c)]["final_rms_px"]} for c in order[:10]],
                "n_above_1_step": int((w >= 1).sum()), "n_above_5_steps": int((w >= 5).sum()),
                "seconds": time.time() - t_p0}
            log(f"\n  ALL {C} CELLS, +{100*args.frac:.0f}% {param} at tick {W}: "
                f"min {w.min():.4f}  med {np.median(w):.4f}  MAX {w.max():.4f} steps "
                f"(cell {order[0]});  {int((w>=1).sum())}/{C} above 1 step, "
                f"{int((w>=5).sum())}/{C} above 5")
            log("  top 10: " + ", ".join(f"c{int(c)}={rows[int(c)]['worst']:.3f}"
                                         for c in order[:10]))
            dump()

    R["wall_seconds"] = time.time() - t_start
    dump()
    log(f"\nwrote {out_path}  [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()
