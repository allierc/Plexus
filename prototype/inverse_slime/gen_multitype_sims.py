"""gen_multitype_sims.py -- generate GT + position-loss recovery TRAJECTORIES for the
2-type and 4-type slime, so build_pdf.py can show GT-vs-recovery pages (incl. cross)."""
import os, sys, subprocess
import numpy as np
import yaml
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RL"))
import plexus.operators  # noqa
from plexus.engine import run
from spec_space import SLIME2, SLIME4
from sense_trainer import fit_sense_pos_mc, DEFAULT
from exp_multitype import recover_field_mc, gen_gt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def write_spec(M, u, name):
    p = M.lo + np.clip(u, 0, 1) * (M.hi - M.lo)
    raw = yaml.safe_load(open(os.path.join(REPO, M.config)))
    raw.pop("descriptions", None)
    raw["general"].update(name=name, seed=0, n_frames=200)
    raw["sets"]["cell"]["n"] = 6000; raw["fields"]["chemical"]["res"] = 140
    for o in raw["operators"]:
        for op, key, val in [("deposit", "amount", p[0]), ("diffuse", "rate", p[1]),
                             ("decay", "rate", p[2]), ("sense", "cross", p[3])]:
            if o["op"] == op:
                o[key] = round(float(val), 6)
    for td in raw["sets"]["cell"]["types"].values():
        td["move_speed"] = round(float(p[4]), 6); td["turn_speed"] = round(float(p[5]), 4)
        td["sensor_angle"] = round(float(p[6]), 3); td["sensor_dist"] = round(float(p[7]), 5)
    yaml.safe_dump(raw, open(os.path.join(REPO, "config/slime", name + ".yaml"), "w"),
                   sort_keys=False, default_flow_style=None)


def generate(name):
    subprocess.run([sys.executable, os.path.join(REPO, "Plexus_Main.py"), "-o", "generate",
                    name, "--force", "--device", "cpu"], cwd=REPO,
                   env={**os.environ, "PYTHONPATH": os.path.join(REPO, "src")})


def main():
    for M, tag in [(SLIME2, "slime2"), (SLIME4, "slime4")]:
        rng = np.random.default_rng(7)
        u_true = M.sample_u(rng)
        write_spec(M, u_true, f"{tag}_gt"); generate(f"{tag}_gt")
        grid, pos, nt, p = gen_gt(M, u_true)
        a, d, k, move = recover_field_mc(grid, pos, nt, grid.shape[-1])
        slo = np.array([M.lo[5], M.lo[6], M.lo[7]]); shi = np.array([M.hi[5], M.hi[6], M.hi[7]])
        r = fit_sense_pos_mc(grid, pos, nt, p, grid.shape[-1], {**DEFAULT}, slo, shi,
                             float(M.lo[3]), float(M.hi[3]), seed=0, schedule=(1,), iters_per=800)
        rec = np.array([a, d, k, r["cross"], move, r["turn_speed"], r["sensor_angle"], r["sensor_dist"]])
        u_rec = (rec - M.lo) / (M.hi - M.lo)
        write_spec(M, u_rec, f"{tag}_pos"); generate(f"{tag}_pos")
        print(f"{tag}: GT + recovery generated (cross u true {u_true[3]:.2f} rec {u_rec[3]:.2f})", flush=True)


if __name__ == "__main__":
    main()
