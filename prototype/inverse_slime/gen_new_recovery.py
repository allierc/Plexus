"""gen_new_recovery.py -- recover the showcase target's 8D spec with the CURRENT method
(std3 steering + curriculum sense + exact field/move), write the spec, and generate its
simulation so build_pdf.py can show it as a new visual test (slime_gd8d_v2)."""
import os, sys, json, subprocess
import numpy as np
import yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RL"))
import autorun as A
from spec_space import SLIME
from sense_trainer import fit_sense_curriculum_pos, DEFAULT

# prototype/inverse_slime/<file> -> repo root = /workspace/Plexus
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAME = "slime_gd8d_v2"


def main():
    T = A.load_targets()[0]                                    # the showcase target (best_dist)
    a, d, k, move = A.recover_field(T)
    # ONE-STEP position loss -- the ablation winner (de-biased, beats recurrent)
    r = fit_sense_curriculum_pos(T["grid"], T["pos"], T["p"], T["R"], {**DEFAULT}, A.SENSE_LO, A.SENSE_HI,
                                 seed=0, schedule=(1,), iters_per=800)
    u = T["u_true"].copy()
    for i, v in {0: a, 1: d, 2: k, 4: move, 5: r["turn_speed"],
                 6: r["sensor_angle"], 7: r["sensor_dist"]}.items():
        u[i] = A.u_of(v, i)                                    # cross (3) held @GT
    p = SLIME.lo + np.clip(u, 0, 1) * (SLIME.hi - SLIME.lo)
    print("recovered u:", [round(float(x), 3) for x in u], "| true:", [round(float(x), 3) for x in T["u_true"]])

    raw = yaml.safe_load(open(os.path.join(REPO, "config/slime/slime_random_full.yaml")))
    raw.pop("descriptions", None)
    raw["general"].update(name=NAME, seed=0, n_frames=200)
    raw["sets"]["cell"]["n"] = 6000; raw["fields"]["chemical"]["res"] = 140
    for o in raw["operators"]:
        for op, key, val in [("deposit", "amount", p[0]), ("diffuse", "rate", p[1]),
                             ("decay", "rate", p[2]), ("sense", "cross", p[3])]:
            if o["op"] == op:
                o[key] = round(float(val), 6)
    for td in raw["sets"]["cell"]["types"].values():
        td["move_speed"] = round(float(p[4]), 6); td["turn_speed"] = round(float(p[5]), 4)
        td["sensor_angle"] = round(float(p[6]), 3); td["sensor_dist"] = round(float(p[7]), 5)
    cfgpath = os.path.join(REPO, "config/slime", NAME + ".yaml")
    yaml.safe_dump(raw, open(cfgpath, "w"), sort_keys=False, default_flow_style=None)
    print("wrote", cfgpath)
    subprocess.run([sys.executable, os.path.join(REPO, "Plexus_Main.py"),
                    "-o", "generate_plot", NAME, "--movie", "--force", "--device", "cpu"],
                   cwd=os.path.join(REPO),
                   env={**os.environ, "PYTHONPATH": os.path.join(REPO, "src")})
    print("generated", NAME)


if __name__ == "__main__":
    main()
