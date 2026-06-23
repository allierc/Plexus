"""exp_recurrent_schemes.py -- sweep different RECURRENT training schemes for sense.

Compares, on the same targets, how the recurrent scheme affects sensor_angle/turn/dist
recovery. Saves runs/recurrent_schemes.json (consumed by build_pdf.py) and logs a
verdict block to logbook2.md. Schemes:
  one_step          : fit_sense (mixture NLL, no rollout)
  rec_final_K6      : recurrent, match only the FINAL net heading change (cos,sin)
  curric_K10        : curriculum, per-frame loss, single short horizon
  curric_10_30_60   : curriculum, growing horizon
  curric_10_30_60_100: curriculum, longer growing horizon
"""
import os, sys, json, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autorun as A
from sense_trainer import (fit_sense, fit_sense_recurrent_cossin, fit_sense_curriculum, DEFAULT)

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "logbook2.md")
OUTJSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "recurrent_schemes.json")


def logw(s):
    open(LOG, "a").write(s + "\n"); print(s, flush=True)


def uerr(r, T):
    return {k: abs(A.u_of(r[k], i) - T["u_true"][i])
            for k, i in [("sensor_angle", 6), ("turn_speed", 5), ("sensor_dist", 7)]}


def run_scheme(fn, targets, **kw):
    accs = {"sensor_angle": [], "turn_speed": [], "sensor_dist": []}
    for T in targets:
        r = fn(T["grid"], T["pos"], T["p"], T["R"], {**DEFAULT}, A.SENSE_LO, A.SENSE_HI, seed=0, **kw)
        e = uerr(r, T)
        for k in accs:
            accs[k].append(e[k])
    return {k: float(np.mean(v)) for k, v in accs.items()}


def main():
    targets = A.load_targets()[:3]
    schemes = {
        "one_step": lambda t: run_scheme(fit_sense, t),
        "rec_final_K6": lambda t: run_scheme(fit_sense_recurrent_cossin, t, K=6),
        "curric_K10": lambda t: run_scheme(fit_sense_curriculum, t, schedule=(10,), iters_per=400),
        "curric_10_30_60": lambda t: run_scheme(fit_sense_curriculum, t, schedule=(10, 30, 60), iters_per=300),
        "curric_10_30_60_100": lambda t: run_scheme(fit_sense_curriculum, t, schedule=(10, 30, 60, 100), iters_per=250),
    }
    logw(f"\n## RECURRENT-SCHEME SWEEP [{datetime.datetime.now().strftime('%H:%M:%S')}] "
         f"(std3 steering; {len(targets)} targets, mean u-err)")
    out = {}
    for name, fn in schemes.items():
        res = fn(targets)
        out[name] = res
        logw(f"- {name:22s}: angle={res['sensor_angle']:.3f} turn={res['turn_speed']:.3f} dist={res['sensor_dist']:.3f}")
        json.dump(out, open(OUTJSON, "w"), indent=2)            # incremental save
    best = min(out, key=lambda k: out[k]["sensor_angle"])
    logw(f"- best sensor_angle: {best} ({out[best]['sensor_angle']:.3f})")


if __name__ == "__main__":
    main()
