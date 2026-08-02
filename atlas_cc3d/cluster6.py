"""cluster6 -- run the Phase 6 variance sweep on the Janelia L4 partition.

The question Phase 6 has to settle before Figure 5 is worth attempting is not "does a gradient
flow" (it does, measured) but "is the gradient WORTH ANYTHING": the division draw is discrete, so a
straight-through gradient taken through one sampled rollout is a gradient through that sample's
luck. `phase6_fit.py` turns that into a measurement -- fit the same target with K seeds averaged
per Adam step, across independent replicates, and look at the spread.

That workload is exactly what a cluster is for and exactly what one machine is bad at: each fit is
small and mostly Python (an A6000 beats this container's CPU by only 1.6x on a 40-frame rollout,
because 146 cells is not enough work to hide kernel-launch overhead), while the replicates are
completely independent. Wall-clock comes from running 32 of them at once, not from making one
fast.

THE MACHINERY IS NOT NEW. `discovery/cluster.py` already carries the hardened primitives this
campaign paid for -- detached submission (a bsub whose ssh reply is truncated reads as "0 jobs
submitted", and we created duplicate jobs that way), job IDs as the only ground truth (`bjobs -a`
returns historical jobs, so a previous EXIT with the same name is indistinguishable from a new
PEND), an ssh wall-timeout, and completion only on POSITIVE evidence. This module reuses all of
it and supplies only the atlas's own job script, under its own job-name prefix so the two
campaigns' queues never alias.

    python cluster6.py submit --ks 1 2 4 8 --replicates 4
    python cluster6.py status
    python cluster6.py collect            # -> _state/phase6_variance.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PLEXUS, "discovery"))

import cluster as C  # noqa: E402  the hardened primitives; see the module docstring

# Our own prefix, so `status`/`kill` here never touch a discovery job (and vice versa). C's queue
# helpers filter on the module-level PREFIX, so it is set once, here, for every call we make.
C.PREFIX = "at6_"
LOGDIR = os.path.join(PLEXUS, "log", "atlas_cc3d", "_cluster6")
RUNS = os.path.join(PLEXUS, "log", "atlas_cc3d", "_phase6")
TAG = ""


def use_tag(tag):
    """Point one sweep's artefacts at their own directories and job-name prefix.

    The 20-step sweep is a RESULT, not scratch: a convergence rerun that wrote into the same
    place would silently overwrite the very numbers it is being compared against.
    """
    global LOGDIR, RUNS, TAG
    if not tag:
        return
    TAG = tag
    LOGDIR = os.path.join(PLEXUS, "log", "atlas_cc3d", f"_cluster6_{tag}")
    RUNS = os.path.join(PLEXUS, "log", "atlas_cc3d", f"_phase6_{tag}")
    # the WHOLE tag, not its first letter: two tags starting with the same letter
    # would share a job-name prefix and each would see the other's jobs in `status`.
    C.PREFIX = f"at6{tag}_"
    C.LOGDIR = LOGDIR


C.LOGDIR = LOGDIR


def job_name(k, rep):
    return f"k{k}_r{rep}"


def _job_script(k, rep, frames, steps, params):
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(RUNS, exist_ok=True)
    name = job_name(k, rep)
    script = os.path.join(LOGDIR, f"{name}.sh")
    out = C.cpath(os.path.join(RUNS, f"{name}.json"))
    with open(script, "w") as f:
        f.write("\n".join([
            "#!/bin/bash -l",
            f"cd {C.cpath(HERE)}",
            f"export PYTHONPATH={C.cpath(os.path.join(PLEXUS, 'src'))}:{C.cpath(HERE)}",
            "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",
            f"conda run -n {C.ENV} python phase6_fit.py --seeds {k} --replicate {rep} "
            f"--frames {frames} --steps {steps} --params {' '.join(params)} "
            f"--device cuda:0 --out {out}",
        ]) + "\n")
    os.chmod(script, 0o755)
    return script


def _bsub_cmd(k, rep, frames, steps, params):
    script = _job_script(k, rep, frames, steps, params)
    name = job_name(k, rep)
    out = C.cpath(os.path.join(LOGDIR, f"{name}.out"))
    gpu = "-gpu num=1 " if C.GPU != "0" else ""
    return (f"cd {C.cpath(HERE)} && bsub -n {C.NCPUS} {gpu}-q {C.QUEUE} -W {C.WALL} "
            f"-J {C.PREFIX}{name} -o {out} -e {out[:-4]}.err bash -l {C.cpath(script)}")


def submit(ks, replicates, frames, steps, params):
    os.makedirs(LOGDIR, exist_ok=True)
    jobs = [(k, r) for k in ks for r in range(replicates)]
    runner = os.path.join(LOGDIR, "_submit.sh")
    with open(runner, "w") as f:
        f.write("#!/bin/bash -l\n"
                + "\n".join(_bsub_cmd(k, r, frames, steps, params) for k, r in jobs) + "\n")
    os.chmod(runner, 0o755)
    logl = os.path.join(LOGDIR, "_submit.log")
    if os.path.exists(logl):
        os.replace(logl, logl + ".prev")
    C._ssh(f"nohup bash {C.cpath(runner)} > {C.cpath(logl)} 2>&1 < /dev/null &", timeout=30)
    print(f"[cluster6] fired {len(jobs)} bsub(s) DETACHED on {C.SSH}")
    print("  the ssh returns BEFORE bsub lands -- the queue is the only ground truth")
    ids = C.submitted_ids(wait_s=30, expect=len(jobs))
    print(f"[cluster6] job ids: {' '.join(ids) if ids else '(none yet)'}")
    return ids


def collect(tail=0):
    """Gather the finished fits, and report the spread that is the whole point.

    `tail` turns on Polyak averaging: the mean of the parameter over the last `tail` Adam steps
    rather than its last value. With a noisy gradient the iterate never settles -- it random-walks
    around the optimum -- so the final value is one draw from that walk and the tail mean is the
    estimate. It is also what makes "did it converge, or did we just stop?" answerable.
    """
    import statistics
    rows = []
    if os.path.isdir(RUNS):
        for fn in sorted(os.listdir(RUNS)):
            if fn.endswith(".json"):
                with open(os.path.join(RUNS, fn)) as f:
                    rows.append(json.load(f))
    by_k = {}
    for r in rows:
        if tail and r.get("history"):
            key = list(r["final"])[0]
            hist = [h["params"][key] for h in r["history"]][-tail:]
            r["final"] = {key: statistics.fmean(hist)}
            r["tail"] = tail
        by_k.setdefault(r["K"], []).append(r)
    print(f"{'K':>3}  {'n':>3}  {'mean':>10}  {'stdev':>10}  {'spread':>10}   fitted values")
    print("-" * 84)
    summary = []
    for k in sorted(by_k):
        vals = [r["final"][list(r["final"])[0]] for r in by_k[k]]
        sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        rng = (max(vals) - min(vals)) if vals else float("nan")
        print(f"{k:>3}  {len(vals):>3}  {statistics.fmean(vals):>10.4f}  {sd:>10.4f}  "
              f"{rng:>10.4f}   {', '.join(f'{v:.4f}' for v in vals)}")
        summary.append({"K": k, "n": len(vals), "mean": statistics.fmean(vals),
                        "stdev": sd, "range": rng, "values": vals})
    if not rows:
        print("  (no finished fits yet)")
        return summary
    out = os.path.join(HERE, "_state", f"phase6_variance{'_' + TAG if TAG else ''}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": summary, "runs": rows}, f, indent=2)
    print(f"\n-> {os.path.relpath(out, HERE)}")
    print("READ IT AS: if stdev shrinks with K, the estimator is noisy but consistent and the "
          "fix is more seeds.\n            If it does not, the pathwise gradient is biased and "
          "the trace/replay/score contract is required.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "status", "wait", "kill", "collect"])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--replicates", type=int, default=4)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--params", nargs="+", default=["max_radius"])
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--tag", default=None, help="name this sweep; keeps its artefacts separate")
    ap.add_argument("--tail", type=int, default=0,
                    help="Polyak-average the last N steps instead of taking the final value")
    # A queue query is only as good as the prefix it filters on: changing the naming scheme
    # between a submit and a status made a healthy 32-job batch read as "queue empty", which is
    # the same false-completion this machinery was hardened against -- just self-inflicted.
    ap.add_argument("--prefix", default=None,
                    help="override the job-name prefix (for a batch submitted under an old one)")
    a = ap.parse_args()
    use_tag(a.tag)
    if a.prefix:
        C.PREFIX = a.prefix

    if a.cmd == "submit":
        if not C.preflight(verbose=True):
            return 1
        submit(a.ks, a.replicates, a.frames, a.steps, a.params)
    elif a.cmd == "status":
        C.status(verbose=True)
    elif a.cmd == "wait":
        C.wait(poll=a.poll)
    elif a.cmd == "kill":
        C.kill()
    elif a.cmd == "collect":
        collect(tail=a.tail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
