"""exp_margin.py -- H1: does a no-Gumbel MARGIN loss beat the softmax-mixture NLL
for recovering sensor geometry (esp. sensor_angle)?

Controlled comparison on the same K targets: same reconstruction, same iters/lr/restarts;
only the sense MODEL + LOSS differ (mixture-(mu,sigma)-NLL vs margin-hinge). Logs to
logbook2.md with the hypothesis, per-target results (mean+/-std = replication), and a
validate/falsify verdict vs the grid-search ceiling (sensor_angle 0.121).
"""
from __future__ import annotations

import os, sys, json, datetime
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "RL"))
sys.path.insert(0, _HERE)

import autorun as A
from sense_trainer import fit_sense, fit_sense_margin, DEFAULT

LOG2 = os.path.join(_HERE, "runs", "logbook2.md")


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")


def logw(s):
    with open(LOG2, "a") as f:
        f.write(s + "\n")
    print(s, flush=True)


def evalfit(fitfn, targets, cfg):
    e = {"turn_speed": [], "sensor_angle": [], "sensor_dist": []}
    for T in targets:
        r = fitfn(T["grid"], T["pos"], T["p"], T["R"], cfg, A.SENSE_LO, A.SENSE_HI, seed=0)
        e["turn_speed"].append(abs(A.u_of(r["turn_speed"], 5) - T["u_true"][5]))
        e["sensor_angle"].append(abs(A.u_of(r["sensor_angle"], 6) - T["u_true"][6]))
        e["sensor_dist"].append(abs(A.u_of(r["sensor_dist"], 7) - T["u_true"][7]))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in e.items()}


def main():
    if not os.path.exists(LOG2):
        logw(f"# Model/trainer improvement logbook (beyond hyperparameter grid)\n\n"
             f"Started {now()}. The grid search ceilings the *hyperparameters*; here we improve "
             f"the *model* and *trainer*. Benchmark to beat = grid-search best (sensor_angle u-err "
             f"0.121). Each experiment: a hypothesis, a controlled comparison on the same targets "
             f"(mean+/-std), a verdict.\n")
    targets = A.load_targets()
    cfg = {**DEFAULT, "iters": 800, "restarts": 3}
    logw(f"\n## H1 [{now()}]: a no-Gumbel margin/hinge loss recovers sensor_angle better than "
         f"the softmax-mixture NLL.\nControlled: same reconstruction + iters=800 restarts=3; only "
         f"the sense model/loss differ.")
    mix = evalfit(fit_sense, targets, cfg)
    mar = evalfit(fit_sense_margin, targets, cfg)
    logw(f"- mixture-NLL : angle={mix['sensor_angle'][0]:.3f}+/-{mix['sensor_angle'][1]:.3f} "
         f"turn={mix['turn_speed'][0]:.3f} dist={mix['sensor_dist'][0]:.3f}")
    logw(f"- margin-hinge: angle={mar['sensor_angle'][0]:.3f}+/-{mar['sensor_angle'][1]:.3f} "
         f"turn={mar['turn_speed'][0]:.3f} dist={mar['sensor_dist'][0]:.3f}")
    a_mix, a_mar = mix["sensor_angle"][0], mar["sensor_angle"][0]
    verdict = ("VALIDATED (margin better)" if a_mar < a_mix - 0.01
               else "FALSIFIED (margin not better)" if a_mar > a_mix + 0.01 else "INCONCLUSIVE (tie)")
    logw(f"- verdict: {verdict}  [mixture {a_mix:.3f} vs margin {a_mar:.3f}; grid ceiling 0.121]")
    json.dump(dict(mixture=mix, margin=mar), open(os.path.join(_HERE, "runs", "exp_margin.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
