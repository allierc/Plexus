"""autorun.py -- autonomous scientific optimisation of the `sense` surrogate fit.

Goal: drive the 3 sensing params (turn_speed, sensor_angle, sensor_dist) -- esp.
sensor_angle, which the first 8D fit missed -- to low recovery error by iterating on
the LEARNABLE OPERATOR + TRAINER + LEARNING SCHEME. Method = coordinate ascent over
the trainer hyperparameters, then random refinement, evaluated across K held-out
targets (replication, with mean +/- std), every trial logged to logbook.md with a
hypothesis / result / verdict (validate or falsify). Best fits archive a spec + mp4
into archive/ for inspection. Field + move_speed are recovered exactly (separate,
already solved) so the search focuses where it is hard.

    PYTHONPATH=../../src nohup python autorun.py --hours 12 > runs/autorun.out 2>&1 &
"""
from __future__ import annotations

import os, sys, json, time, math, shutil, subprocess, datetime
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "RL"))
sys.path.insert(0, _HERE)

import plexus.operators  # noqa
from plexus.engine import run
from spec_space import SLIME
import operators as OP
from operators import LearnableDeposit, LearnableDiffuse, LearnableDecay
from sense_trainer import fit_sense, DEFAULT

DEV = "cpu"                                    # engine rollout (slime many small ops -> cpu)
FIT_DEV = "cuda" if torch.cuda.is_available() else "cpu"   # sense fit (batched -> gpu)
K_TARGETS = 4
SENSE_LO = np.array([SLIME.lo[5], SLIME.lo[6], SLIME.lo[7]])
SENSE_HI = np.array([SLIME.hi[5], SLIME.hi[6], SLIME.hi[7]])
RUNS = os.path.join(_HERE, "runs"); os.makedirs(RUNS, exist_ok=True)
ARCH = os.path.join(_HERE, "archive"); os.makedirs(ARCH, exist_ok=True)
LOG = os.path.join(RUNS, "logbook.md")

SWEEP = {
    "iters": [200, 400, 800, 1500], "restarts": [1, 3, 6],
    "lr": [0.02, 0.05, 0.1, 0.2], "beta0": [6.0, 12.0, 30.0, 60.0],
    "temp0": [0.02, 0.05, 0.1, 0.2], "anneal": [False, True],
    "field_offset": [0, 1, 2], "max_turn": [0.6, 1.0, 1.5],
    "n_per_frame": [200, 400, 800], "wd": [0.0, 1e-4, 1e-3], "sched": ["cos", "none"],
}


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")


def logw(s):
    with open(LOG, "a") as f:
        f.write(s + "\n")
    print(s, flush=True)


# ---- data: cache GT transitions for K targets ----
def load_targets():
    rs = [json.loads(l) for l in open(os.path.join(_HERE, "..", "RL", "runs", "multi", "slime", "results.jsonl"))]
    rs.sort(key=lambda r: r["best_dist"])
    out = []
    for r in rs[:K_TARGETS]:
        u_true = np.array(r["u_true"])
        SLIME.frames = 150
        spec, p = SLIME.apply_u(u_true, seed=0)
        _, o = run(spec, out_path=None, device=DEV)
        grid = torch.tensor(np.asarray(o["fields"]["chemical"]["grid"]), dtype=torch.float32).to(FIT_DEV)
        pos = torch.tensor(np.asarray(o["sets"]["cell"]["pos"]), dtype=torch.float32).to(FIT_DEV)
        out.append(dict(u_true=u_true, ucb_u=np.array(r["best_u"]), p=p,
                        grid=grid, pos=pos, R=grid.shape[-1], target=r["target"]))
    return out


def u_of(phys, idx):
    return (phys - SLIME.lo[idx]) / (SLIME.hi[idx] - SLIME.lo[idx])


def evaluate(cfg, targets):
    """Fit sense on each target; return per-sensing-param u-errors (mean,std) + score."""
    errs = {"turn_speed": [], "sensor_angle": [], "sensor_dist": []}
    for T in targets:
        res = fit_sense(T["grid"], T["pos"], T["p"], T["R"], cfg, SENSE_LO, SENSE_HI, seed=0)
        if res is None:
            return None
        errs["turn_speed"].append(abs(u_of(res["turn_speed"], 5) - T["u_true"][5]))
        errs["sensor_angle"].append(abs(u_of(res["sensor_angle"], 6) - T["u_true"][6]))
        errs["sensor_dist"].append(abs(u_of(res["sensor_dist"], 7) - T["u_true"][7]))
    out = {k: (float(np.mean(v)), float(np.std(v))) for k, v in errs.items()}
    out["score"] = float(np.mean([out[k][0] for k in errs]))          # mean sensing u-error
    out["angle"] = out["sensor_angle"][0]
    return out


# ---- archive: full 8D recovery for the showcase target -> spec + mp4 ----
def recover_field(T):
    grid, pos, R = T["grid"], T["pos"], T["R"]
    dev = grid.device
    nt = torch.zeros(pos.shape[1], dtype=torch.long, device=dev)
    dep = LearnableDeposit(0.5, SLIME.lo[0], SLIME.hi[0]).to(dev)
    dif = LearnableDiffuse(0.5, SLIME.lo[1], SLIME.hi[1]).to(dev)
    dec = LearnableDecay(0.5, SLIME.lo[2], SLIME.hi[2]).to(dev)
    opt = torch.optim.Adam(list(dep.parameters()) + list(dif.parameters()) + list(dec.parameters()), lr=0.1)
    ts = list(range(2, min(grid.shape[0], pos.shape[0]) - 1))
    for _ in range(300):
        opt.zero_grad(); loss = 0.0
        for t in ts:
            loss = loss + ((dec(dif(dep(grid[t], pos[t], nt, R))) - grid[t + 1]) ** 2).mean()
        (loss / len(ts)).backward(); opt.step()
    move = float((pos[3:120] - pos[2:119]).norm(dim=-1).median())
    return float(dep.amount()), float(dif.rate()), float(dec.rate()), move


def archive(cfg, T, tag):
    import yaml
    a, d, k, move = recover_field(T)
    res = fit_sense(T["grid"], T["pos"], T["p"], T["R"], cfg, SENSE_LO, SENSE_HI, seed=0)
    u = T["u_true"].copy()
    for i, v in {0: a, 1: d, 2: k, 4: move, 5: res["turn_speed"],
                 6: res["sensor_angle"], 7: res["sensor_dist"]}.items():
        u[i] = u_of(v, i)                                            # cross (3) held @GT
    p = SLIME.lo + np.clip(u, 0, 1) * (SLIME.hi - SLIME.lo)
    raw = yaml.safe_load(open(os.path.join(_HERE, "..", "..", "config/slime/slime_random_full.yaml")))
    raw.pop("descriptions", None)
    raw["general"].update(name="slime_gd8d", seed=0, n_frames=200)
    raw["sets"]["cell"]["n"] = 6000; raw["fields"]["chemical"]["res"] = 140
    for o in raw["operators"]:
        for op, key, val in [("deposit", "amount", p[0]), ("diffuse", "rate", p[1]),
                             ("decay", "rate", p[2]), ("sense", "cross", p[3])]:
            if o["op"] == op: o[key] = round(float(val), 6)
    for td in raw["sets"]["cell"]["types"].values():
        td["move_speed"] = round(float(p[4]), 6); td["turn_speed"] = round(float(p[5]), 4)
        td["sensor_angle"] = round(float(p[6]), 3); td["sensor_dist"] = round(float(p[7]), 5)
    cfgyaml = os.path.join(_HERE, "..", "..", "config/slime/slime_gd8d.yaml")
    yaml.safe_dump(raw, open(cfgyaml, "w"), sort_keys=False, default_flow_style=None)
    adir = os.path.join(ARCH, tag); os.makedirs(adir, exist_ok=True)
    shutil.copy(cfgyaml, os.path.join(adir, "slime_gd8d.yaml"))
    json.dump(dict(cfg=cfg, u_recovered=u.tolist(), u_true=T["u_true"].tolist()),
              open(os.path.join(adir, "recovery.json"), "w"), indent=2)
    try:
        subprocess.run([sys.executable, os.path.join(_HERE, "..", "..", "Plexus_Main.py"),
                        "-o", "generate_plot", "slime_gd8d", "--movie", "--force", "--device", "cpu"],
                       cwd=os.path.join(_HERE, "..", ".."), timeout=600,
                       env={**os.environ, "PYTHONPATH": os.path.join(_HERE, "..", "..", "src")})
        src = "/groups/saalfeld/home/allierc/GraphData/graphs_data/slime/slime_gd8d/movie_cell.mp4"
        if os.path.exists(src):
            shutil.copy(src, os.path.join(adir, "movie_cell.mp4"))
    except Exception as e:
        logw(f"  (archive render failed: {e})")
    logw(f"  -> archived spec+mp4 to archive/{tag}/")


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--hours", type=float, default=12.0)
    args = ap.parse_args()
    deadline = time.time() + args.hours * 3600
    OP.run_all_tests(DEV)
    targets = load_targets()
    show = targets[0]                                               # showcase target for mp4

    if not os.path.exists(LOG):
        logw(f"# Sense-surrogate optimisation logbook\n\nStarted {now()}. "
             f"Objective: minimise sensing u-error (esp. sensor_angle) across "
             f"{K_TARGETS} held-out slime targets. Method: coordinate ascent over trainer "
             f"hyperparameters then random refinement; each trial evaluated on all targets "
             f"(mean+/-std = replication); verdict logged vs running best. Field+move are "
             f"recovered exactly elsewhere.\n")

    best_cfg = dict(DEFAULT); base = evaluate(best_cfg, targets)
    best = base
    logw(f"\n## Baseline [{now()}]\ncfg={best_cfg}\n"
         f"result: score={base['score']:.3f} angle={base['sensor_angle'][0]:.3f}"
         f"+/-{base['sensor_angle'][1]:.3f} turn={base['turn_speed'][0]:.3f} "
         f"dist={base['sensor_dist'][0]:.3f}\n")
    archive(best_cfg, show, "baseline")
    last_arch = time.time(); trial = 0

    # ---- coordinate ascent: sweep each hyperparameter, adopt the best option ----
    sweep_round = 0
    while time.time() < deadline:
        sweep_round += 1
        improved_round = False
        logw(f"\n## Coordinate-ascent sweep {sweep_round} [{now()}]  (best score {best['score']:.3f})")
        for key, options in SWEEP.items():
            if time.time() > deadline: break
            trial_results = []
            for opt in options:
                if opt == best_cfg.get(key):
                    trial_results.append((opt, best)); continue
                cfg = dict(best_cfg); cfg[key] = opt
                r = evaluate(cfg, targets); trial += 1
                if r is None: continue
                trial_results.append((opt, r))
                verdict = "IMPROVED" if r["score"] < best["score"] - 1e-4 else "falsified"
                logw(f"- T{trial} [{now()}] {key}={opt}: score={r['score']:.3f} "
                     f"angle={r['sensor_angle'][0]:.3f}+/-{r['sensor_angle'][1]:.3f} "
                     f"turn={r['turn_speed'][0]:.3f} dist={r['sensor_dist'][0]:.3f}  [{verdict}]")
            if trial_results:
                bopt, br = min(trial_results, key=lambda x: x[1]["score"])
                if br["score"] < best["score"] - 1e-4:
                    best, best_cfg[key], improved_round = br, bopt, True
                    logw(f"  => adopt {key}={bopt} (score {best['score']:.3f}, "
                         f"angle {best['sensor_angle'][0]:.3f})")
                    if time.time() - last_arch > 2400:             # archive at most ~40 min
                        archive(best_cfg, show, f"r{sweep_round}_{key}"); last_arch = time.time()
        if not improved_round:
            logw(f"  sweep {sweep_round}: no improvement -> switch to random refinement")
            break

    # ---- random refinement around the best until the deadline ----
    logw(f"\n## Random refinement [{now()}]  best cfg={best_cfg} score={best['score']:.3f}")
    rng = np.random.default_rng(0)
    while time.time() < deadline:
        cfg = dict(best_cfg)
        for key in rng.choice(list(SWEEP), size=int(rng.integers(1, 4)), replace=False):
            cfg[key] = SWEEP[key][int(rng.integers(len(SWEEP[key])))]
        r = evaluate(cfg, targets); trial += 1
        if r and r["score"] < best["score"] - 1e-4:
            best, best_cfg = r, cfg
            logw(f"- T{trial} [{now()}] RANDOM IMPROVED: score={r['score']:.3f} "
                 f"angle={r['sensor_angle'][0]:.3f} cfg-delta logged. New best.")
            logw(f"  cfg={cfg}")
            if time.time() - last_arch > 2400:
                archive(best_cfg, show, f"rand_T{trial}"); last_arch = time.time()
        elif trial % 25 == 0:
            logw(f"- T{trial} [{now()}] (random, best still {best['score']:.3f} "
                 f"angle {best['sensor_angle'][0]:.3f})")

    logw(f"\n## DONE [{now()}]  best score={best['score']:.3f} "
         f"angle={best['sensor_angle'][0]:.3f} cfg={best_cfg}")
    archive(best_cfg, show, "final")
    json.dump(dict(best_cfg=best_cfg, best=best), open(os.path.join(RUNS, "best.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
