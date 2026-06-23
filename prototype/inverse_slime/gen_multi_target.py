"""gen_multi_target.py -- generate GT + one-step-position-loss recovery TRAJECTORIES for
several targets, so build_pdf.py can show GT-vs-recovery on more pages (one per target).

Trajectory-only (-o generate, no movie/caption) -> fast. Names: slime_gt_t<i> / slime_pos_t<i>.
"""
import os, sys, subprocess
import numpy as np
import yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RL"))
import autorun as A
from spec_space import SLIME
from sense_trainer import fit_sense_curriculum_pos, DEFAULT

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGETS = [1, 2, 3]                                            # target0 already has its pages


def write_spec(u, name):
    p = SLIME.lo + np.clip(u, 0, 1) * (SLIME.hi - SLIME.lo)
    raw = yaml.safe_load(open(os.path.join(REPO, "config/slime/slime_random_full.yaml")))
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
    path = os.path.join(REPO, "config/slime", name + ".yaml")
    yaml.safe_dump(raw, open(path, "w"), sort_keys=False, default_flow_style=None)


def generate(name):                                            # trajectory only -- fast
    subprocess.run([sys.executable, os.path.join(REPO, "Plexus_Main.py"), "-o", "generate",
                    name, "--force", "--device", "cpu"], cwd=REPO,
                   env={**os.environ, "PYTHONPATH": os.path.join(REPO, "src")})


def main():
    targets = A.load_targets()
    for ti in TARGETS:
        T = targets[ti]
        write_spec(T["u_true"], f"slime_gt_t{ti}"); generate(f"slime_gt_t{ti}")
        a, d, k, move = A.recover_field(T)
        r = fit_sense_curriculum_pos(T["grid"], T["pos"], T["p"], T["R"], {**DEFAULT}, A.SENSE_LO, A.SENSE_HI,
                                     seed=0, schedule=(1,), iters_per=800)
        u = T["u_true"].copy()
        for i, v in {0: a, 1: d, 2: k, 4: move, 5: r["turn_speed"], 6: r["sensor_angle"], 7: r["sensor_dist"]}.items():
            u[i] = A.u_of(v, i)
        ae = abs(A.u_of(r["sensor_angle"], 6) - T["u_true"][6])
        print(f"target{ti}: sensor_angle u-err={ae:.3f}", flush=True)
        write_spec(u, f"slime_pos_t{ti}"); generate(f"slime_pos_t{ti}")
        print(f"target{ti}: GT + recovery generated", flush=True)


if __name__ == "__main__":
    main()
