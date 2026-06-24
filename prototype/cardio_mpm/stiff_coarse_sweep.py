#!/usr/bin/env python
"""stiff_coarse_sweep.py -- does COARSE stiffness structure influence the loops?

The atlas found stiff_wl=8 INERT -- but wl=8 is very fine (averages out over the loop scale). This
sweeps the stiffness from fine stripes -> coarser wavelengths -> a TWO-REGION step (soft|stiff split,
like material_directional_cardio's map_two_regions), at FULL contrast (youngs 50..150), to test whether
a coarse stiffness gradient finally moves the loop morphology. Appends to archive/stiffcoarse_<tag>/.

  PYTHONPATH=../../src python stiff_coarse_sweep.py
"""
from __future__ import annotations
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import yaml
import cardio_mpm_atlas as A

# (tag, builder) -- stiffness field in [0,1]; mapped to youngs [50,150] by apply_material_map
def _stripes(wl):
    return lambda: A.aniso_field([wl, 26], 0.0, 0.7)
def _two_region(softness=4.0):
    # smooth left(soft)|right(stiff) split across x at the domain centre
    xx = np.arange(A.N)[:, None] * np.ones((1, A.N))
    return lambda: (0.5 * (1 + np.tanh((xx - A.N / 2) / softness))).astype("float32")
def _two_region_v(softness=4.0):
    yy = np.ones((A.N, 1)) * np.arange(A.N)[None, :]
    return lambda: (0.5 * (1 + np.tanh((yy - A.N / 2) / softness))).astype("float32")

LEVELS = [("wl8_fine", _stripes(8)), ("wl16", _stripes(16)), ("wl32", _stripes(32)),
          ("wl64", _stripes(64)), ("two_region_x", _two_region()), ("two_region_y", _two_region_v())]
BASE = dict(stiff_wl=8.0, gain_wl=26.0, fibre_wl=40.0, fibre_angle=0.6,
            stiff_phase=0.7, amplitude=10.0, drag_k=30.0, n_frames=105)


def write_maps(tag, p, stiff_builder):
    import tifffile
    s = np.asarray(stiff_builder(), np.float32)
    g = (A.aniso_field([26, p["gain_wl"]], 0.0, 0.0) / 5.0).astype("float32")
    fib = (A.fiber_field(p["fibre_wl"], p["fibre_angle"]) / (2 * np.pi)).astype("float32")
    names = {}
    for nm, arr in [("stiff", s), ("gain", g), ("fiber", fib)]:
        fn = f"stiffcoarse_{tag}_{nm}.tif"; tifffile.imwrite(os.path.join(A.MAT, fn), arr); names[nm] = f"material/{fn}"
    return names


def write_spec(name, maps, p):
    s = {
        "fields": {"activation": {"frame": "grid", "res": 128}, "mpm_grid": {"frame": "mpm_grid", "n_grid": 128},
                   "direction": {"frame": "vector_grid", "source": maps["fiber"]},
                   "stiffness_map": {"frame": "image", "source": maps["stiff"]},          # full-contrast, normalize default
                   "gain_map": {"frame": "image", "source": maps["gain"]}},
        "general": {"boundary": "wall", "dt": 1.0, "n_frames": int(p["n_frames"]), "name": name, "seed": 0},
        "operators": [
            {"at": "cell", "op": "aggregate"},
            {"op": "apply_material_map", "at": "mpm_particle", "from": "stiffness_map", "target": "youngs", "min": 50, "max": 150},
            {"op": "apply_material_map", "at": "mpm_particle", "from": "gain_map", "target": "gain", "min": 0.4, "max": 1.6},
            {"at": "activation", "duration": 30, "name": "beat", "op": "pacemaker", "period": 50, "phase": 0},
            {"at": "activation", "center": [0.5, 0.5], "clock": "beat", "op": "pulse_stimulus", "radius": 0.12, "profile": "uniform"},
            {"amplitude": float(p["amplitude"]), "at": "mpm_particle", "from": "activation", "op": "pulse_to_active_stress", "direction_from": "direction"},
            {"at": "mpm_particle", "k": float(p["drag_k"]), "op": "mpm_drag"},
            {"at": "mpm_particle", "dt_sub": 0.0002, "op": "mpm_strain"},
            {"a_max": 200, "at": "mpm_particle", "drag": 0.0, "dt_sub": 0.0002, "op": "p2g", "to": "mpm_grid"},
            {"at": "mpm_grid", "dt_sub": 0.0002, "op": "mpm_grid_update", "wall_contact": 0.06, "wall_damp": 0.5},
            {"at": "mpm_particle", "dt_sub": 0.0002, "from": "mpm_grid", "op": "g2p", "wall_contact": 0.06, "wall_damp": 0.5}],
        "schedule": ["aggregate", "apply_material_map", "pacemaker", "pulse_stimulus", "pulse_to_active_stress", "mpm_drag",
                     {"dt": 0.0002, "steps": ["mpm_strain", "p2g", "mpm_grid_update", "g2p"], "substep": 10}],
        "sets": {"cell": {"n": 1, "start": [[0.5, 0.5]],
                          "types": {"sheet": {"block": [0.15, 0.15, 0.85, 0.85], "fraction": 1.0, "youngs": 80}}},
                 "mpm_particle": {"density": 1.0, "parent": "cell", "per_parent": 16384}},
    }
    out = os.path.join(A.REPO, "config", "material", f"{name}.yaml")
    yaml.safe_dump(s, open(out, "w"), sort_keys=False, default_flow_style=None)
    return out


def main():
    rows = []
    for tag, builder in LEVELS:
        outdir = os.path.join(A.HERE, "archive", f"stiffcoarse_{tag}")
        os.makedirs(outdir, exist_ok=True)
        name = f"material_stiffcoarse_{tag}"
        p = dict(BASE)
        print(f"=== stiffness coarseness {tag} -> {name} ===", flush=True)
        maps = write_maps(tag, p, builder); write_spec(name, maps, p)
        cmd = [sys.executable, "Plexus_Main.py", "-o", "generate", name, "--force", "--no-describe", "--device", "cuda:0"]
        r = subprocess.run(cmd, cwd=A.REPO, capture_output=True, text=True)
        if "done:" not in r.stdout and "done:" not in r.stderr:
            print(r.stdout[-1500:], r.stderr[-1500:], flush=True); print(f"  FAILED {tag}", flush=True); continue
        m, rest, disp, resid, idx = A.morphology(name)
        A.dashboard(outdir, name, m, rest, disp, resid, idx, maps, p)
        row = (f"{tag:14s} open={m['openness_nonaffine']:.3f} open_raw={m['openness_raw']:.3f} "
               f"aspect={m['aspect_mean']:.2f} angle={m['major_axis_angle_mean']:.2f} size={m['loop_size_mean']:.2e} "
               f"chir+={m['chirality_pos_fraction']:.2f}")
        yaml.safe_dump({"name": name, "stiff_variant": tag, "params": p, **m},
                       open(os.path.join(outdir, "metrics.yaml"), "w"), sort_keys=False)
        open(os.path.join(outdir, "progress.txt"), "w").write(row)
        print(f"  {row}\n  done -> {outdir}", flush=True)
        rows.append(row)
    print("\n=== STIFFNESS COARSENESS SWEEP SUMMARY ===", flush=True)
    for r in rows:
        print("  " + r, flush=True)


if __name__ == "__main__":
    main()
