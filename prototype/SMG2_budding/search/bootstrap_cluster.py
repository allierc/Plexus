"""bootstrap_cluster -- run the biased bootstrap sharded across N L4 jobs (parallel), then merge.

Submits NSHARDS bsub jobs to gpu_l4 over ssh (each an independent biased bootstrap of N_PER specs
with its own seed -> its own dataset shard), polls until done, then merges the shards into
search/_bootstrap/{dataset.jsonl, encodings.npy}. Pre-flight: the reward calibration_gate() must pass.

  cd prototype/SMG2_budding
  python search/bootstrap_cluster.py                 # 16 shards x 20 specs = 320, gpu_l4
  python search/bootstrap_cluster.py --shards 16 --per 20 --frames 600 --wall 60 [--skip-gate]
"""
import os, sys, re, json, time, glob, argparse, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

# --- cluster config (devcontainer /workspace/Plexus  ==  cluster /groups/.../Graph/Plexus) ---
SSH = os.environ.get("SMG_CLUSTER_SSH", "allierc@login1")
CPY = os.environ.get("SMG_CLUSTER_PY", "/groups/saalfeld/home/allierc/miniforge3/envs/neural-graph/bin/python")
CROOT = "/groups/saalfeld/home/allierc/Graph/Plexus/prototype/SMG2_budding"
CSRC = "/groups/saalfeld/home/allierc/Graph/Plexus/src"
CAM2 = "/groups/saalfeld/home/allierc/Graph/Plexus/prototype/active_matter2"
LSF = "/etc/profile.d/profile.lsf.sh"
QUEUE = os.environ.get("SMG_QUEUE", "gpu_l4")
OUT = os.path.join(HERE, "_bootstrap")


def _ssh(cmd, timeout=90, retries=3):
    for _ in range(retries):
        try:
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", SSH,
                                f"source {LSF} 2>/dev/null; {cmd}"],
                               capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return r
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)
    return r if 'r' in dir() else None


def submit(shards, per, frames, stride, wall):
    logs = os.path.join(HERE, "_bootstrap", "logs"); os.makedirs(logs, exist_ok=True)
    clogs = f"{CROOT}/search/_bootstrap/logs"
    ids = {}
    for i in range(shards):
        tag = f"boot{i:02d}"
        script = os.path.join(logs, f"{tag}.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash -l\n"
                    f"cd {CROOT}\n"
                    f"export PYTHONPATH={CSRC}:{CAM2}:{CROOT}/search\n"
                    f"echo START $(date +%s) $(hostname)\n"
                    f"{CPY} search/bootstrap.py --n {per} --seed {i} --frames {frames} "
                    f"--stride {stride} --skip-gate --out search/_bootstrap/shard_{i:02d}\n"
                    f"echo END $(date +%s)\n")
        bsub = (f"cd {CROOT} && bsub -n 4 -gpu num=1 -q {QUEUE} -W {wall} -J {tag} "
                f"-o {clogs}/{tag}.out -e {clogs}/{tag}.err bash -l {clogs}/{tag}.sh")
        r = _ssh(bsub)
        m = re.search(r"Job <(\d+)>", r.stdout if r else "")
        if m:
            ids[tag] = m.group(1); print(f"[submit] L4 job {m.group(1)}  {tag}", flush=True)
        else:
            print(f"[submit] FAILED {tag}: {(r.stdout if r else '')}{(r.stderr if r else '')}", flush=True)
    return ids


def poll(ids, every=60):
    live = set(ids.values())
    while live:
        time.sleep(every)
        r = _ssh("bjobs -noheader -o 'id stat' " + " ".join(sorted(live)))
        if r and r.returncode == 0:
            st = {p.split()[0]: p.split()[1] for p in r.stdout.splitlines() if len(p.split()) >= 2}
            live = {j for j in live if st.get(j) in ("PEND", "RUN", "PROV", "WAIT")}
        done = sum(len(open(p).readlines()) for p in glob.glob(os.path.join(OUT, "shard_*", "dataset.jsonl")))
        print(f"[poll] {len(live)} L4 jobs running; {done} specs written", flush=True)


def merge():
    rows, encs = [], []
    for d in sorted(glob.glob(os.path.join(OUT, "shard_*"))):
        dj = os.path.join(d, "dataset.jsonl")
        if os.path.isfile(dj):
            rows += [l for l in open(dj) if l.strip()]
        ep = os.path.join(d, "encodings.npy")
        if os.path.isfile(ep):
            encs.append(np.load(ep))
    with open(os.path.join(OUT, "dataset.jsonl"), "w") as f:
        f.writelines(rows)
    if encs:
        np.save(os.path.join(OUT, "encodings.npy"), np.concatenate(encs))
    from collections import Counter
    hist = Counter(json.loads(l)["failure"] for l in rows)
    onpath = sum(1 for l in rows if json.loads(l).get("value", {}).get("duct_score", 0) > 0.4)
    print(f"\n=== MERGED {len(rows)} specs -> {OUT}/dataset.jsonl + encodings.npy ===")
    print("failure manifold:", dict(hist))
    print(f"on-path (duct>0.4): {onpath}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--wall", type=int, default=60)
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.merge_only:
        merge(); return

    if not args.skip_gate:
        print("=== pre-flight: reward calibration gate ===", flush=True)
        import smg_reward as R
        if not R.calibration_gate():
            print("CALIBRATION GATE FAILED -> not launching"); return
        print()

    print(f"=== submitting {args.shards} L4 shards x {args.per} specs = {args.shards*args.per} total "
          f"(frames={args.frames}, queue={QUEUE}) ===", flush=True)
    ids = submit(args.shards, args.per, args.frames, args.stride, args.wall)
    if not ids:
        print("no jobs submitted (cluster/ssh?) -> aborting"); return
    poll(ids)
    merge()


if __name__ == "__main__":
    main()
