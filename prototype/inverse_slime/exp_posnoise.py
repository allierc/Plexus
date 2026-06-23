"""exp_posnoise.py -- improve POSITION-only training: recurrent state noise (with/without)
and curriculum length, per the connectome-cx recipe (noise_recurrent_level=0.03)."""
import os, sys, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autorun as A
from sense_trainer import fit_sense_curriculum_pos, DEFAULT

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "logbook2.md")


def logw(s):
    open(LOG, "a").write(s + "\n"); print(s, flush=True)


def run(cfg_over, sched, targets):
    e = {"sensor_angle": [], "turn_speed": [], "sensor_dist": []}
    for T in targets:
        r = fit_sense_curriculum_pos(T["grid"], T["pos"], T["p"], T["R"], {**DEFAULT, **cfg_over},
                                     A.SENSE_LO, A.SENSE_HI, seed=0, schedule=sched, iters_per=400)
        for k, i in [("sensor_angle", 6), ("turn_speed", 5), ("sensor_dist", 7)]:
            e[k].append(abs(A.u_of(r[k], i) - T["u_true"][i]))
    return {k: float(np.mean(v)) for k, v in e.items()}


def main():
    targets = A.load_targets()
    logw(f"\n## POSITION-LOSS + NOISE [{datetime.datetime.now().strftime('%H:%M:%S')}] "
         f"(recurrent state noise with/without; {len(targets)} targets, mean u-err)")
    configs = [
        ("pos no-noise", dict(rec_noise=0.0), (10, 20, 30)),
        ("pos noise=0.03", dict(rec_noise=0.03), (10, 20, 30)),
        ("pos noise=0.08", dict(rec_noise=0.08), (10, 20, 30)),
        ("pos noise=0.03 long", dict(rec_noise=0.03), (10, 30, 60, 100)),
    ]
    best = None
    for name, ov, sched in configs:
        r = run(ov, sched, targets)
        logw(f"- {name:22s}: angle={r['sensor_angle']:.3f} turn={r['turn_speed']:.3f} dist={r['sensor_dist']:.3f}")
        sc = r["sensor_angle"] + r["turn_speed"] + r["sensor_dist"]
        if best is None or sc < best[1]:
            best = (name, sc, r)
    logw(f"- BEST (sum of 3 u-err): {best[0]}  angle={best[2]['sensor_angle']:.3f} "
         f"turn={best[2]['turn_speed']:.3f} dist={best[2]['sensor_dist']:.3f}")


if __name__ == "__main__":
    main()
