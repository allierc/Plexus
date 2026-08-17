"""Pilot sweep for the two-galaxy encounter: does the pair survive its own passage?

The first pilot dissolved -- half-mass radius 0.84 -> 7.3 and the two centres SEPARATING after
pericentre. That looked like energy injected by the integrator, and it was NOT: halving dt to
0.002 reproduced the same trajectory to three digits (A_soft15 vs D_dt002 below), and a single
disc held its size for 32 time units (E_single). The pair was simply UNBOUND -- the launch speed
was above the parabolic sqrt(2 G M_tot / d) -- and the passage that did survive was far too deep,
because an impact parameter with the launch aimed at the companion carries no angular momentum at
all. Hence `swirl`, and hence designing the orbit backwards from a target pericentre.

Diagnostic per run (all in the run's own dimensionless units):
  sep      distance between the red and blue centres of mass
  r50      median stellar radius about its own centre -- the disc's size; a disc that HOLDS
           stays near its spawn radius 1.2, a dissolving one grows without bound
  r95      95th-percentile radius about the pair's centre -- how far the debris reaches
  bound    fraction of stars inside 4x the spawn radius of the pair centre

Run:  python prototype/galaxy_collision/sweep.py [variant ...]
"""
from __future__ import annotations
import os
import subprocess
import sys

import numpy as np
import yaml

ROOT = "/workspace/Plexus"
PY = "/workspace/.conda_envs/neural-graph-linux/bin/python"
CFG = os.path.join(ROOT, "config/inverse_square")
GD = "/groups/saalfeld/home/allierc/GraphData/graphs_data/inverse_square"
BASE = os.path.join(CFG, "galaxy_collision_3d.yaml")

# name -> patch. `_single` drops the companion (the control: does ONE disc hold at all?).
VARIANTS = {
    # the two demos: same discs, same operator, ONE knob -- the encounter's angular momentum.
    # Mass CONCENTRATED in the cores (1.2 of the 1.5 per galaxy): stars are then deeply bound to
    # their own core, so the passage strips the outskirts into tails instead of unbinding half the
    # disc -- the Toomre & Toomre (1972) setup, whose galaxies were point masses with test-particle
    # discs. With the mass in the disc instead, the pilot lost 50% of its stars at first pericentre.
    # pericentre 2.0 (1.7 disc radii): the cores never meet, so nothing hard scatters, and the
    # passage still raises tails. approach/swirl are the radial/tangential halves of v_rel at
    # frame 0 for the orbit a=5.0, e=0.6 -> pericentre 2.0, period 40.6, first passage t=8.7.
    "graze3":  dict(softening=0.15, separation=6.0, spawn_offset=0.0, approach=0.18, swirl=0.26,
                    n_frames=10000),
    "graze2":  dict(softening=0.15, separation=6.0, spawn_offset=0.0, approach=0.25, swirl=0.20,
                    central_mass=1.2, disc=0.3, n_frames=10000),
    "headon2": dict(softening=0.15, separation=6.0, spawn_offset=0.0, approach=0.25, swirl=0.0,
                    central_mass=1.2, disc=0.3, n_frames=10000),
    "graze":  dict(softening=0.15, separation=6.0, spawn_offset=0.0, approach=0.25, swirl=0.20,
                   n_frames=10000),
    "headon": dict(softening=0.15, separation=6.0, spawn_offset=0.0, approach=0.25, swirl=0.0,
                   n_frames=10000),
    # earlier probes, kept so the reasoning is re-runnable
    "A_soft15":  dict(softening=0.15),
    "B_nocore":  dict(softening=0.15, central_mass=0.0),
    "C_graze":   dict(softening=0.15, central_mass=0.30, spawn_offset=4.5),
    "D_dt002":   dict(softening=0.15, central_mass=0.30, spawn_offset=4.5, dt=0.002, n_frames=16000),
    "E_single":  dict(softening=0.15, central_mass=0.30, single=True),
}


def _orbit(v: dict, spec: dict) -> str:
    """The two-body orbit the IC asks for, from the spec's own numbers -- printed BEFORE the run
    so the measurement has something to disagree with. `approach`/`swirl` are half-speeds, so
    v_rel = 2 sqrt(a^2 + s^2); M_tot = 2 x (disc mass + central_mass)."""
    s = spec["sets"]["star"]
    X = float(v.get("separation", s.get("spawn_separation", 0.0)))
    b = float(v.get("spawn_offset", s.get("spawn_offset", 0.0)))
    ap = float(v.get("approach", s["vel_init"].get("approach", 0.0)))
    sw = float(v.get("swirl", s["vel_init"].get("swirl", 0.0)))
    cm = float(v.get("central_mass", s["vel_init"].get("central_mass", 0.0)))
    disc = float(v.get("disc", 0.75))             # per-galaxy DISC mass (the core is `cm`)
    GM = 2.0 * (disc + cm)
    d = (X ** 2 + b ** 2) ** 0.5
    vrel = 2.0 * (ap ** 2 + sw ** 2) ** 0.5
    E = 0.5 * vrel ** 2 - GM / d
    L = abs(X * 2.0 * sw + b * 2.0 * ap)          # |sep x v_rel|
    if E >= 0:
        return f"UNBOUND: v_rel={vrel:.2f} vs parabolic {(2 * GM / d) ** 0.5:.2f}"
    a = -GM / (2 * E)
    e = max(0.0, 1.0 + 2 * E * L ** 2 / GM ** 2) ** 0.5
    P = 2 * 3.14159265 * (a ** 3 / GM) ** 0.5
    return (f"bound: v_rel={vrel:.2f} (parabolic {(2 * GM / d) ** 0.5:.2f}), a={a:.2f}, e={e:.3f}, "
            f"pericentre={a * (1 - e):.2f} (disc radius 1.2), period={P:.1f} time units")


def _patch(name: str, v: dict) -> str:
    """Write a pilot config for one variant: 6000 stars (4x the mass each, same disc mass)."""
    with open(BASE) as f:
        spec = yaml.safe_load(f)
    spec["general"]["name"] = f"gcol_{name}"
    spec["general"]["n_frames"] = int(v.get("n_frames", spec["general"]["n_frames"]))
    spec["general"]["dt"] = float(v.get("dt", spec["general"]["dt"]))
    spec["general"]["record_cap"] = 240
    s = spec["sets"]["star"]
    s["n"] = 6000
    for t in s["types"].values():                # 3000 stars per disc carry `disc` between them
        t["mass"] = float(v.get("disc", 0.75)) / 3000.0
    if "spawn_offset" in v:
        s["spawn_offset"] = float(v["spawn_offset"])
    if "separation" in v:
        s["spawn_separation"] = float(v["separation"])
    if "swirl" in v:
        s["vel_init"]["swirl"] = float(v["swirl"])
    if "approach" in v:
        s["vel_init"]["approach"] = float(v["approach"])
    if "central_mass" in v:
        s["vel_init"]["central_mass"] = float(v["central_mass"])
    if "softening" in v:
        s["vel_init"]["softening"] = float(v["softening"])
        spec["operators"][0]["softening"] = float(v["softening"])
    if v.get("single"):                          # control: ONE disc, nothing to collide with
        s["spawn"] = "disk"
        s["n"] = 3000
        s.pop("spawn_separation", None); s.pop("spawn_offset", None); s.pop("spawn_tilt", None)
        s["vel_init"].pop("approach", None)
        s["types"] = {"red": {"fraction": 1.0, "mass": float(v.get("disc", 0.75)) / 3000.0}}
    spec["plotting"]["movie_max_frames"] = 240
    spec["plotting"]["splat_res"] = 520
    out = os.path.join(CFG, f"gcol_{name}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    return out


def _diag(name: str) -> None:
    d = np.load(os.path.join(GD, f"gcol_{name}", "trajectory.npz"))
    pos = d["star__pos"]; nt = d["star__node_type"]; T = pos.shape[0]
    groups = [nt == k for k in np.unique(nt)]
    print(f"--- {name}: {T} recorded frames, N={pos.shape[1]}")
    for i in range(0, T, max(1, T // 8)):
        p = pos[i]
        cs = [p[g].mean(0) for g in groups]
        r50 = [float(np.median(np.linalg.norm(p[g] - c, axis=1))) for g, c in zip(groups, cs)]
        c_all = p.mean(0); rad = np.linalg.norm(p - c_all, axis=1)
        sep = float(np.linalg.norm(cs[0] - cs[1])) if len(cs) > 1 else 0.0
        print(f"   row{i:4d}  sep={sep:5.2f}  r50={' '.join(f'{x:5.2f}' for x in r50)}"
              f"  r95={np.percentile(rad, 95):6.2f}  bound={float((rad < 4.8).mean()):.2f}")


def main():
    want = sys.argv[1:] or list(VARIANTS)
    for name in want:
        cfg = _patch(name, VARIANTS[name])
        import yaml as _y
        print(f"=== {name} -> {cfg}\n    predicted {_orbit(VARIANTS[name], _y.safe_load(open(BASE)))}",
              flush=True)
        subprocess.run([PY, "Plexus_Main.py", "-o", "generate", cfg, "--device", "cuda:0",
                        "--force", "--no-describe"], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        _diag(name)


if __name__ == "__main__":
    main()
