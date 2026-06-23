"""exp_curriculum.py -- H7: the connectome-cx CURRICULUM scheme for slime sense.

Free-run the heading as (cos,sin), score the PER-FRAME (cos,sin) error over a horizon
(growing curriculum), from random start frames, with bounce-aware resync. The key
difference from my earlier recurrent (H2/H5): per-frame loss over the whole horizon,
not just the final net change. Compares short vs growing horizons.
"""
import sys, os, time, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autorun as A
from sense_trainer import fit_sense_curriculum, DEFAULT

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "logbook2.md")


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")


def logw(s):
    open(LOG, "a").write(s + "\n"); print(s, flush=True)


def main():
    targets = A.load_targets()
    cfg = {**DEFAULT}
    logw(f"\n## H7 [{now()}] CURRICULUM (per-frame (cos,sin) loss, free-run heading, random starts, "
         f"bounce resync) -- the connectome-cx scheme. My prior recurrent matched only the FINAL net "
         f"change; this scores every frame over the horizon.")
    for sched in [(10,), (10, 20, 30), (10, 30, 60, 100, 140)]:
        e = {"sensor_angle": [], "turn_speed": [], "sensor_dist": []}
        t0 = time.time()
        for T in targets:
            r = fit_sense_curriculum(T["grid"], T["pos"], T["p"], T["R"], cfg, A.SENSE_LO, A.SENSE_HI,
                                     seed=0, schedule=sched, iters_per=400)
            for k, i in [("sensor_angle", 6), ("turn_speed", 5), ("sensor_dist", 7)]:
                e[k].append(abs(A.u_of(r[k], i) - T["u_true"][i]))
        m = {k: float(np.mean(v)) for k, v in e.items()}
        a = m["sensor_angle"]
        v = ("BREAKTHROUGH (<0.11 floor)" if a < 0.11 else "improved" if a < 0.18 else "at floor")
        logw(f"- schedule={sched}: angle={a:.3f} turn={m['turn_speed']:.3f} dist={m['sensor_dist']:.3f}  "
             f"[{v}; prior floor 0.12-0.18]  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
