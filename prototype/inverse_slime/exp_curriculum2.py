"""H7b: curriculum with per-iter random-start DIVERSITY + per-target breakdown."""
import sys, os, datetime
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
    cfg = {**DEFAULT, "resample": 25}
    logw(f"\n## H7b [{now()}] CURRICULUM + per-iter random-start DIVERSITY (resample=25), "
         f"per-target breakdown. schedule=(10,30,60), iters=400.")
    per = []
    for ti, T in enumerate(targets):
        r = fit_sense_curriculum(T["grid"], T["pos"], T["p"], T["R"], cfg, A.SENSE_LO, A.SENSE_HI,
                                 seed=0, schedule=(10, 30, 60), iters_per=400)
        ae = abs(A.u_of(r["sensor_angle"], 6) - T["u_true"][6])
        de = abs(A.u_of(r["sensor_dist"], 7) - T["u_true"][7])
        te = abs(A.u_of(r["turn_speed"], 5) - T["u_true"][5])
        per.append(ae)
        logw(f"- target{ti}: angle_uerr={ae:.3f}  turn={te:.3f} dist={de:.3f}")
    per = np.array(per)
    tag = "target-DEPENDENT identifiability" if per.max() - per.min() > 0.1 else "uniform"
    logw(f"- MEAN angle={per.mean():.3f} (min {per.min():.3f}, max {per.max():.3f}) -> {tag}")


if __name__ == "__main__":
    main()
