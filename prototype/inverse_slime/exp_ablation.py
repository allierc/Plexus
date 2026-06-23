"""exp_ablation.py -- position-loss training: RECURRENT vs one-step  x  NOISE vs none.

The two ablations the user flagged as key. one-step = schedule (1,) (predict next position,
no rollout); recurrent = free-run a horizon. noise = connectome recurrent state noise.
2 contrasting targets (t0 identifiable large-angle, t1 hard small-angle), mean u-err.
"""
import os, sys, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autorun as A
from sense_trainer import fit_sense_curriculum_pos, DEFAULT

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "logbook2.md")


def logw(s):
    open(LOG, "a").write(s + "\n"); print(s, flush=True)


def run(sched, noise, targets):
    e = {"sensor_angle": [], "turn_speed": [], "sensor_dist": []}
    for T in targets:
        r = fit_sense_curriculum_pos(T["grid"], T["pos"], T["p"], T["R"],
                                     {**DEFAULT, "rec_noise": noise}, A.SENSE_LO, A.SENSE_HI,
                                     seed=0, schedule=sched, iters_per=300)
        for k, i in [("sensor_angle", 6), ("turn_speed", 5), ("sensor_dist", 7)]:
            e[k].append(abs(A.u_of(r[k], i) - T["u_true"][i]))
    return {k: float(np.mean(v)) for k, v in e.items()}


def main():
    targets = A.load_targets()[:2]
    logw(f"\n## ABLATION recurrent x noise [{datetime.datetime.now().strftime('%H:%M:%S')}] "
         f"(position loss; 2 targets t0,t1; mean u-err)")
    cells = [
        ("one-step,  no-noise", (1,), 0.0),
        ("one-step,  noise.03", (1,), 0.03),
        ("recurrent, no-noise", (10, 20, 30), 0.0),
        ("recurrent, noise.03", (10, 20, 30), 0.03),
    ]
    rows = []
    for name, sched, noise in cells:
        r = run(sched, noise, targets)
        rows.append((name, r))
        logw(f"- {name:22s}: angle={r['sensor_angle']:.3f} turn={r['turn_speed']:.3f} dist={r['sensor_dist']:.3f}")
    best = min(rows, key=lambda x: x[1]["sensor_angle"] + x[1]["turn_speed"] + x[1]["sensor_dist"])
    logw(f"- BEST (sum 3): {best[0]}")


if __name__ == "__main__":
    main()
