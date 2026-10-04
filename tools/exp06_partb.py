#!/usr/bin/env python
"""exp06 Part B -- does the cardio fit survive its shared clock being driven by a gap-junction wave?

    PYTHONPATH=src python tools/exp06_partb.py [--coupling 12010] [--sheets healthy,hcm]
                                                [--out experiments/exp06_excitation_wave/data/partB_score.json]

G-no-regression's ruler (`experiments/exp06_excitation_wave/gates.yaml`). For each Utrecht sheet:

  1. RUN the declared real-sheet excitation spec `config/tissue/exp06_real_<sheet>.yaml` -- Aliev-Panfilov
     on the 472 (healthy) / 434 (HCM) cells of the segmentation, coupled along their measured junctions
     (`cell_neighbours[model: label_image]`) -- fresh, at `--coupling`. Nothing is replayed.
  2. READ each cell's arrival time (first upstroke through u = 0.5), in frames of the 24 fps recording.
  3. WRITE a copy of the fit whose per-cell delay is the fitted delay PLUS the wave's arrival relative to
     the first cell to fire: gamma_j(t) = gamma(t - delay_j - (t_arr,j - min t_arr)). The fitted delay is
     KEPT -- it is each cell's own excitation-to-contraction delay (md, Stage 0 finding) -- and the wave
     adds the conduction. A cell the wave never reaches does not contract (g = g2 = 0).
  4. SCORE the fit and its wave-driven copy with `prototype/cardio_mpm/strain/s4_score.py`, run unchanged
     as a script, with the settings its own `s4_score.json` records for that sheet (120 particles per cell,
     128 grid, per-cell anchor 1e4, drag 150, band 0.03, beats 1-3 healthy / 1-4 HCM).

Writes, per sheet, the mean over beats of R^2 of the strain maps (`r2_A`, the 0.870 / 0.894 of
strain/SUMMARY.md section 4) and of the displacements (`r2_u`), for the fit and the wave-driven fit, the
arrival spread in ms, and `shortfall` = the largest (fit - wave-driven) r2_A over the sheets.

WHY THE FIT IS RE-SCORED HERE AND NOT READ FROM ITS JSON: the same rollout on another GPU disagrees with
itself at the float32 atomic floor (spec_parity.py), so the baseline is measured beside the variant, on
the same device, in the same job.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRAIN = os.path.join(ROOT, "prototype", "cardio_mpm", "strain")
PY = sys.executable
FRAME_MS = 1000.0 / 24.0
S4 = {"healthy": dict(beats="1,2,3", fit_beats="1,2,3"), "hcm": dict(beats="1,2,3,4", fit_beats="1,2,3,4")}


def run_excitation(sheet, coupling, device, work):
    """Generate the declared real-sheet spec at `coupling` into `work`; return the run folder."""
    sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
    from plexus.paths import graphs_data_path
    raw = yaml.safe_load(open(os.path.join(ROOT, "config", "tissue", f"exp06_real_{sheet}.yaml")))
    name = f"exp06_partb_{sheet}_D{int(round(coupling))}"
    raw["general"]["name"] = name
    # THE RUN FOLLOWS THE COUPLING: dt inside the explicit limit (D dt <= 0.48), and three crossing times
    # of the field (22 cells at ~0.64 sqrt(D) cells per time unit on this graph, Phase 1 / design estimate)
    # so that every cell's arrival is inside the run; below D = 1000 the stimulus is the brief pulse
    # (20 for 0.05 time units), which keeps u under 1 + a and v >= 0 (finding P2-1c).
    import math
    dt = min(4.0e-5 if coupling > 1000 else 0.008, 0.48 / float(coupling))
    t_end = max(3.0, 3.0 * 22.0 / (0.64 * math.sqrt(float(coupling))))
    raw["general"]["dt"] = dt
    raw["general"]["n_frames"] = int(math.ceil(t_end / dt))
    raw["general"]["record_cap"] = 3000
    for o in raw["operators"]:
        if o["op"] == "cell_chem_diffuse":
            o["d"] = [float(coupling), 0.0]
        if o["op"] == "cell_chem_react" and coupling <= 1000:
            o["stim"]["duration"] = 0.05
    spec = os.path.join(ROOT, "config", "tissue", f"{name}.yaml")
    yaml.safe_dump(raw, open(spec, "w"), sort_keys=False)
    os.makedirs(os.path.join(work, "graphs_data"), exist_ok=True)
    link = os.path.join(work, "graphs_data", "cardio")
    if not os.path.exists(link):
        os.symlink(graphs_data_path("cardio"), link)
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    subprocess.run([PY, "-u", "Plexus_Main.py", "-o", "generate", f"tissue/{name}", "--device", device,
                    "--output_root", work, "--no-viz", "--force"], cwd=ROOT, env=env, check=True)
    return os.path.join(work, "graphs_data", "tissue", name)


def arrivals_frames(run):
    """[C] arrival (first upstroke) of every cell, frames of 1/24 s after the first cell; NaN = never."""
    sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
    from exp_measures.common import open_run
    from exp_measures import exp06 as E
    U = E._on_set(open_run(run), "cell")
    tt = E._row_times(U)
    X, _, _ = E._series(U, 0)
    fir = E._firings(np.nan_to_num(X, nan=-np.inf), 0.5, 0.25)
    first = np.array([tt[f[0]] if f else np.nan for f in fir])
    ms = first * 1e3 * float(U.spec["general"]["units"]["time_s"])
    return (ms - np.nanmin(ms)) / FRAME_MS, ms


def wave_params(z, dfr):
    """The fit with the wave in its clock: delay_j + arrival_j (frames); never-excited cells silent."""
    z = dict(z)
    never = ~np.isfinite(dfr)
    z["delay"] = np.asarray(z["delay"], float) + np.nan_to_num(dfr, nan=0.0)
    z["g"] = np.where(never, 0.0, z["g"]); z["g2"] = np.where(never, 0.0, z["g2"])
    return z, never


def s4(params, sheet, device, out):
    cfg = S4[sheet]
    cmd = [PY, "-u", "s4_score.py", "--params", params, "--device", device, "--per-parent", "120",
           "--n-grid", "128", "--anchor", "10000", "--anchor-percell", "--drag", "150", "--band", "0.03",
           "--beats", cfg["beats"], "--fit-beats", cfg["fit_beats"], "--specimen", sheet, "--out", out]
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    subprocess.run(cmd, cwd=STRAIN, env=env, check=True)
    r = json.load(open(out))["results"]
    return (float(np.mean([v["model"]["r2_A"] for v in r.values()])),
            float(np.mean([v["model"]["r2_u"] for v in r.values()])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coupling", type=float, default=12010.0,
                    help="D per model time unit; 12,010 = D_normal (md, Stage 0 step 5)")
    ap.add_argument("--sheets", default="healthy,hcm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "exp06_excitation_wave", "data", "partB_score.json"))
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="exp06_partb_")
    res = {"coupling": a.coupling, "sheets": {}}
    try:
        for sheet in a.sheets.split(","):
            run = run_excitation(sheet, a.coupling, a.device, work)
            dfr, ms = arrivals_frames(run)
            fit = os.path.join(STRAIN, "out", "fits", f"{sheet}_allbeats", "params.npz")
            z, never = wave_params(dict(np.load(fit)), dfr)
            wave_fit = os.path.join(work, f"{sheet}_wave_params.npz")
            np.savez(wave_fit, **z)
            b_A, b_u = s4(fit, sheet, a.device, os.path.join(work, f"{sheet}_fit_s4.json"))
            w_A, w_u = s4(wave_fit, sheet, a.device, os.path.join(work, f"{sheet}_wave_s4.json"))
            interior = np.asarray(np.load(fit)["interior"], bool)
            res["sheets"][sheet] = dict(
                fit_r2_A=b_A, wave_r2_A=w_A, fit_r2_u=b_u, wave_r2_u=w_u,
                arrival_spread_ms=float(np.nanmax(ms[interior]) - np.nanmin(ms[interior])),
                arrival_spread_frames=float(np.nanmax(dfr[interior]) - np.nanmin(dfr[interior])),
                n_never_excited=int(never.sum()), n_cells=int(len(ms)))
            print(f"[partB] {sheet}: fit r2_A {b_A:.4f}  wave-driven {w_A:.4f}  "
                  f"arrival spread {res['sheets'][sheet]['arrival_spread_ms']:.2f} ms", flush=True)
        res["shortfall"] = max(v["fit_r2_A"] - v["wave_r2_A"] for v in res["sheets"].values())
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"[partB] shortfall {res['shortfall']:+.4f} -> {a.out}")
    finally:
        for sheet in a.sheets.split(","):
            p = os.path.join(ROOT, "config", "tissue", f"exp06_partb_{sheet}_D{int(round(a.coupling))}.yaml")
            if os.path.exists(p):
                os.remove(p)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
