"""s3_gate -- S3, the STOP: what may a fit to this recording claim per cell?

Planted truths at the recording's own spreads (g and phi as measured, log E ~ N(log 80, 0.3)),
the recording's own rest-frame noise added to the target (data/noise_rest.npz), fits from a flat
start, three seeds per free-set. A per-cell family SHIPS only if, over the three seeds, the
median of  median_cells |estimate - truth| / spread(truth)  is below 0.5. The diagnostic rows
(`--init-truth`) start the other families at the truth and say whether a family is unidentifiable
in itself or only entangled with the others.

    python s3_gate.py run          # queue the fits over both GPUs (resumable: done fits are skipped)
    python s3_gate.py table        # the verdict table from out/fits/*/fit.json
"""
import glob, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/workspace/.conda_envs/neural-graph-linux/bin/python"
ENV = dict(os.environ, PYTHONPATH="/workspace/Plexus/src")
ITERS = int(os.environ.get("S3_ITERS", "200"))
LR = "g=2e-3,phi=0.03,logE=0.06,clock=0.05"
GPUS = ["cuda:0", "cuda:1"]

JOBS = []
for free in ("g,clock", "g,phi,clock", "g,phi,logE,clock"):
    for seed in (0, 1, 2):
        JOBS.append(dict(free=free, seed=seed, noise="rest", scale=1.0, init_truth=""))
for free in ("g,clock", "g,phi,clock", "g,phi,logE,clock"):
    JOBS.append(dict(free=free, seed=0, noise="none", scale=1.0, init_truth=""))
JOBS.append(dict(free="logE", seed=0, noise="rest", scale=1.0, init_truth="g,phi,clock"))
JOBS.append(dict(free="logE", seed=1, noise="rest", scale=1.0, init_truth="g,phi,clock"))
JOBS.append(dict(free="phi", seed=0, noise="rest", scale=1.0, init_truth="g,logE,clock"))
JOBS.append(dict(free="g,phi,logE,clock", seed=0, noise="rest", scale=3.0, init_truth=""))


def tag(j):
    return (f"s3_{j['free'].replace(',', '-')}_n{j['noise']}{j['scale']:g}_s{j['seed']}"
            + (f"_truth-{j['init_truth'].replace(',', '-')}" if j["init_truth"] else ""))


def done(j):
    f = os.path.join(HERE, "out", "fits", tag(j), "fit.json")
    return os.path.exists(f) and len(json.load(open(f))["log"]) >= ITERS


def run():
    queue = [j for j in JOBS if not done(j)]
    print(f"{len(queue)} of {len(JOBS)} fits to run, {ITERS} iterations each", flush=True)
    running = {}
    while queue or running:
        for g in GPUS:
            if g not in running and queue:
                j = queue.pop(0)
                cmd = [PY, os.path.join(HERE, "fit.py"), "--device", g, "--target", "planted",
                       "--iters", str(ITERS), "--free", j["free"], "--seed", str(j["seed"]),
                       "--noise", j["noise"], "--noise-scale", str(j["scale"]), "--lr", LR,
                       "--tag", tag(j)] + (["--init-truth", j["init_truth"]] if j["init_truth"] else [])
                log = open(os.path.join(HERE, "out", f"{tag(j)}.log"), "w")
                running[g] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV), j, time.time())
                print(f"  [{g}] start {tag(j)}", flush=True)
        for g, (p, j, t0) in list(running.items()):
            if p.poll() is not None:
                print(f"  [{g}] {'done' if p.returncode == 0 else 'FAILED'} {tag(j)} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
                del running[g]
        time.sleep(10)


def table():
    rows = []
    for j in JOBS:
        f = os.path.join(HERE, "out", "fits", tag(j), "fit.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f)); last = d["log"][-1]; rc = last["recovery"]
        rows.append(dict(tag=tag(j), **{k: j[k] for k in ("free", "seed", "noise", "scale", "init_truth")},
                         iters=len(d["log"]), loss=last["loss"],
                         g_rel=rc["g"]["rel_err"], g_r=rc["g"]["corr"],
                         E_rel=rc["logE"]["rel_err"], E_r=rc["logE"]["corr"],
                         phi_deg=rc["phi"]["weighted_median_deg"], phi_agree=rc["phi"]["axis_agreement"],
                         t0=rc["clock"]["t0"], t0_true=rc["clock"]["truth_t0"],
                         s_per_it=sum(r["seconds"] for r in d["log"]) / len(d["log"])))
    print(f"{'free':<20s}{'noise':>8s}{'sd':>3s}{'truth-init':<14s}{'it':>4s}{'loss':>9s}{'g rel':>7s}{'g r':>6s}"
          f"{'E rel':>7s}{'E r':>6s}{'phi deg':>8s}{'agree':>7s}{'t0/true':>12s}{'s/it':>6s}")
    for r in rows:
        print(f"{r['free']:<20s}{r['noise']+str(r['scale']):>8s}{r['seed']:>3d}{r['init_truth']:<14s}{r['iters']:>4d}"
              f"{r['loss']:>9.5f}{r['g_rel']:>7.3f}{r['g_r']:>6.3f}{r['E_rel']:>7.3f}{r['E_r']:>6.3f}"
              f"{r['phi_deg']:>8.1f}{r['phi_agree']:>7.3f}{r['t0']:>6.2f}/{r['t0_true']:<5.2f}{r['s_per_it']:>6.1f}")
    # the rule, per family, over the three noisy seeds of the full free-set
    import statistics as st
    full = [r for r in rows if r["free"] == "g,phi,logE,clock" and r["noise"] == "rest" and r["scale"] == 1.0
            and not r["init_truth"]]
    if full:
        v = dict(g=st.median(r["g_rel"] for r in full), logE=st.median(r["E_rel"] for r in full),
                 phi_deg=st.median(r["phi_deg"] for r in full))
        print(f"\nRULE (median over {len(full)} seeds, full free-set, rest noise x1): g rel {v['g']:.3f} "
              f"{'SHIPS' if v['g'] < 0.5 else 'does NOT ship'}; log E rel {v['logE']:.3f} "
              f"{'SHIPS' if v['logE'] < 0.5 else 'does NOT ship'}; phi {v['phi_deg']:.1f} deg")
    json.dump(rows, open(os.path.join(HERE, "out", "s3_table.json"), "w"), indent=1)


if __name__ == "__main__":
    {"run": run, "table": table}[sys.argv[1]]()
