#!/usr/bin/env python
"""smg_loop -- agentic scientific loop for SMG branching morphogenesis.

Each batch: the agent (Claude CLI) reads instruction.md + knowledge.md (variance ledger) +
the previous montage, appends a dated analysis entry, updates the ledger, and DESIGNS 8 slots
into smg_slots.md answering ONE causal question (Q1..Q7). The loop runs those slots (2 local
GPUs by default; --cluster submits L4 via bsub), each producing scorecard.json + montage.png
via smg_showcase.py, then tiles a batch montage. Mechanism -> observable, not observable -> fit.

  cd prototype/SMG2_budding
  python smg_loop.py 30            # 30 batches, local 2-GPU, RESUMES
  python smg_loop.py 30 --fresh    # restart at batch 1
  python smg_loop.py 1  --manual   # skip design; run whatever is in smg_slots.md
  python smg_loop.py 30 --cluster  # submit slots to L4 (needs live ssh/kerberos)
"""
import os, sys, re, json, time, glob, shutil, subprocess, threading
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PY = os.environ.get("SMG_PYBIN", "/workspace/.conda_envs/neural-graph-linux/bin/python")
CLAUDE = os.environ.get("CLAUDE_BIN", "claude")
GPUS = [g for g in os.environ.get("SMG_GPUS", "0,1").split(",") if g != ""]
FRAMES = int(os.environ.get("SMG_FRAMES", "1200"))
STRIDE = int(os.environ.get("SMG_STRIDE", "8"))
TIMEOUT_MIN = float(os.environ.get("SMG_CLAUDE_TIMEOUT_MIN", "25"))
WORKER = os.path.join(HERE, "smg_showcase.py")
ARCHIVE = "archive"; SLOTS = "smg_slots.md"; STATE = "smg_loop_state.json"
INSTR = "instruction.md"; LEDGER = "knowledge.md"; ANALYSIS = "analysis.md"
USERIN = "user_input.md"; TRANSCRIPT = "smg_cli_transcript.md"; STAGE = "current_stage.txt"
PYENV = dict(os.environ, PYTHONPATH="/workspace/Plexus/src")

# --- L4 cluster (optional) ---
CLUSTER_SSH = os.environ.get("SMG_CLUSTER_SSH", "allierc@login1")
CLUSTER_PY = os.environ.get("SMG_CLUSTER_PY", "/groups/saalfeld/home/allierc/miniforge3/envs/neural-graph/bin/python")
CLUSTER_SRC = os.environ.get("SMG_CLUSTER_SRC", "/groups/saalfeld/home/allierc/Graph/Plexus/src")
LSF_PROFILE = "/etc/profile.d/profile.lsf.sh"
QUEUE = os.environ.get("SMG_QUEUE", "gpu_l4"); WALL_MIN = int(os.environ.get("SMG_WALL_MIN", "30"))
POLL_SEC = int(os.environ.get("SMG_POLL_SEC", "45"))
_MAP = ("/workspace", "/groups/saalfeld/home/allierc/Graph")


def _cpath(p):
    ap = os.path.abspath(p)
    return _MAP[1] + ap[len(_MAP[0]):] if ap.startswith(_MAP[0]) else ap


CLUSTER_HERE = _cpath(HERE)


# --------------------------------------------------------------- design (Claude CLI)
def design_prompt(batch, n):
    stage = open(STAGE).read().strip() if os.path.isfile(STAGE) else "Q1"
    prev = f"montages/smg_b{batch-1:02d}.png"
    obs = (f"Read the previous montage {prev}, metrics_summary.md, and EACH slot's archive/b{batch-1:02d}_s*/scorecard.json "
           f"+ metrics.json (target_score, topology trajectory, migration polar_order, growth_ratio). "
           f"Decide on the NUMBERS + the variance ledger, not the movie."
           if batch > 1 else
           "FIRST batch: read instruction.md (Q-ladder, observables vs mechanisms, variance ledger, "
           "DELAY-THE-SIREN expressiveness ladder) + knowledge.md (mechanism->operator map, smooth target) "
           "+ specs/smg_base.yaml. Target Q1 (pairwise+polarity, NO growth/program/SIREN) to measure how "
           "much velocity/topology variance bare mechanics explains.")
    return f"""SMG BRANCHING MORPHOGENESIS -- BATCH {batch}/{n}, causal question {stage}.
You are a SCIENTIST discovering which MECHANISMS (composed Plexus operators) produce which OBSERVABLE
(topology / velocity / growth) -- NOT a parameter search, NOT a per-cell fit.

MEMORY (read every batch): method+rules+Q-ladder+expressiveness-ladder = {INSTR};
variance ledger + mechanism->operator map + smooth target = {LEDGER}; analysis log = {ANALYSIS};
user input = {USERIN} (acknowledge if non-empty).

{obs}

Do ALL, in order, AUTO-UPDATING files:
1. OBSERVE last batch (numbers + montage): which mechanism moved which observable?
2. EDIT {ANALYSIS}: append "## Batch {batch}"; every claim paired with an observable number +
   variance share vs its ablation (quantitative report protocol). Update the VARIANCE LEDGER in {LEDGER}.
3. DISTILL {LEDGER}: tag findings [established]/[open]/[rejected]/[engineering].
4. STATE the causal question {stage} + ONE predictive hypothesis; write {stage} to {STAGE}.
5. DESIGN 8 slots into {SLOTS} (~4 exploit, 3 explore, 1 ABLATION control), ONE mechanism lever per
   slot so each yields a clean variance-share. Author per-slot specs (copy specs/smg_base.yaml, edit
   operators) when you change mechanism; dotted overrides for scalars. DELAY expressiveness: only climb
   to a static->slow->SIREN program when the simpler class is exhausted.
Slot line: `name : SPEC specs/<file>.yaml [key val ...]`. Keep 8 non-comment lines (~3-min L4 jobs)."""


def run_claude(prompt, label):
    print(f"[loop] Claude: {label}", flush=True)
    cmd = [CLAUDE, "-p", prompt, "--output-format", "text", "--max-turns", "250",
           "--allowedTools", "Read", "Edit", "Write"]
    try:
        with open(TRANSCRIPT, "a") as tf:
            tf.write(f"\n\n===== {label} =====\n")
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            timer = threading.Timer(TIMEOUT_MIN * 60, p.kill); timer.start()
            try:
                for line in p.stdout:
                    tf.write(line); tf.flush()
            finally:
                p.wait(); timer.cancel()
    except Exception as e:
        print(f"[loop] Claude design FAILED ({e}); falling back to existing {SLOTS}", flush=True)


def parse_slots(batch):
    out = []
    try:
        lines = open(SLOTS).read().splitlines()
    except OSError:
        return out
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, rest = line.split(":", 1)
        toks = rest.split(); spec = "specs/smg_base.yaml"; ov = []; i = 0
        while i < len(toks):
            if toks[i] == "SPEC" and i + 1 < len(toks):
                spec = toks[i + 1]; i += 2
            elif i + 1 < len(toks):
                ov.append(f"{toks[i]}={toks[i+1]}"); i += 2
            else:
                i += 1
        out.append({"name": f"b{batch:02d}_s{len(out)}_{name.strip().replace(' ', '_')}",
                    "spec": spec, "ov": ov})
    return out[:8]


# --------------------------------------------------------------- local run
def run_slot_local(slot, gpu):
    cmd = [PY, WORKER, slot["spec"], f"tag={slot['name']}", f"frames={FRAMES}",
           f"stride={STRIDE}", *slot["ov"]]
    env = dict(PYENV, CUDA_VISIBLE_DEVICES=str(gpu))
    os.makedirs("loop_logs", exist_ok=True)
    lf = open(os.path.join("loop_logs", f"{slot['name']}.log"), "w")
    print(f"[loop] slot {slot['name']} gpu{gpu}: {' '.join(slot['ov'])}", flush=True)
    return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)


def run_batch_local(slots):
    running = []; queue = list(slots)
    while queue or running:
        while queue and len(running) < len(GPUS):
            gpu = GPUS[len(running) % len(GPUS)]
            running.append(run_slot_local(queue.pop(0), gpu))
        time.sleep(3)
        running = [p for p in running if p.poll() is None]
    print("[loop] local batch complete", flush=True)


# --------------------------------------------------------------- cluster run (optional)
def _ssh(cmd, retries=3, timeout=90):
    for _ in range(retries):
        try:
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
                                CLUSTER_SSH, f"source {LSF_PROFILE} 2>/dev/null; {cmd}"],
                               capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return r
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)
    return None


def run_batch_cluster(slots):
    ids = {}
    os.makedirs("loop_logs", exist_ok=True)
    for slot in slots:
        tag = slot["name"]; script = os.path.join("loop_logs", f"{tag}.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash -l\n" + f"cd {CLUSTER_HERE}\nexport PYTHONPATH={CLUSTER_SRC}\n"
                    f"echo START $(date +%s)\n{CLUSTER_PY} smg_showcase.py {slot['spec']} tag={tag} "
                    f"frames={FRAMES} stride={STRIDE} {' '.join(slot['ov'])}\necho END $(date +%s)\n")
        out = _cpath(os.path.join(HERE, "loop_logs", f"{tag}.out"))
        bsub = (f"cd {CLUSTER_HERE} && bsub -n 4 -gpu num=1 -q {QUEUE} -W {WALL_MIN} -J {tag} "
                f"-o {out} -e {out[:-4]}.err bash -l {_cpath(os.path.join(HERE, script))}")
        r = _ssh(bsub); m = re.search(r"Job <(\d+)>", r.stdout if r else "")
        if m:
            ids[tag] = m.group(1); print(f"[loop] L4 job {m.group(1)} {tag}", flush=True)
    live = set(ids.values())
    while live:
        time.sleep(POLL_SEC)
        r = _ssh("bjobs -noheader -o 'id stat' " + " ".join(sorted(live)))
        if r and r.returncode == 0:
            st = {p.split()[0]: p.split()[1] for p in r.stdout.splitlines() if len(p.split()) >= 2}
            live = {j for j in live if st.get(j) in ("PEND", "RUN", "PROV", "WAIT")}
        print(f"[loop] {len(live)} L4 jobs running", flush=True)


# --------------------------------------------------------------- montage (tile slot montages)
def montage(batch):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dirs = sorted(glob.glob(os.path.join(ARCHIVE, f"b{batch:02d}_s*")))
    imgs = [(os.path.join(d, "montage.png"), d) for d in dirs if os.path.isfile(os.path.join(d, "montage.png"))]
    if not imgs:
        print("[loop] no slot montages to tile", flush=True); return
    os.makedirs("montages", exist_ok=True)
    ncol = 2; nrow = int(np.ceil(len(imgs) / ncol)) if (np := __import__("numpy")) else 4
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 9, nrow * 4.2)); fig.patch.set_facecolor("black")
    axs = np.atleast_1d(axs).ravel()
    for ax, (img, d) in zip(axs, imgs):
        ax.imshow(plt.imread(img)); ax.set_facecolor("black")
        sc = {}
        try:
            sc = json.load(open(os.path.join(d, "metrics.json")))
        except Exception:
            pass
        ax.text(0.01, 0.99, f"{os.path.basename(d)[:38]}\nscore {sc.get('target_score','?')} "
                f"buds {sc.get('n_bud','?')} br {sc.get('n_branch','?')} tube {sc.get('n_tube','?')} "
                f"polar {sc.get('polar_order','?')}", transform=ax.transAxes, color="white",
                fontsize=8, va="top", ha="left")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axs[len(imgs):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002, wspace=0.02, hspace=0.06)
    fig.savefig(f"montages/smg_b{batch:02d}.png", dpi=80, facecolor="black"); plt.close(fig)
    print(f"[loop] montage -> montages/smg_b{batch:02d}.png", flush=True)


def load_state():
    return json.load(open(STATE)).get("batch", 1) if os.path.isfile(STATE) else 1


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 20
    fresh = "--fresh" in sys.argv; manual = "--manual" in sys.argv
    cluster = "--cluster" in sys.argv
    print(f"[loop] mode={'L4' if cluster else f'local GPUs {GPUS}'} frames={FRAMES} stride={STRIDE}", flush=True)
    start = 1 if fresh else load_state()
    for b in range(start, n + 1):
        if not manual:
            run_claude(design_prompt(b, n), f"DESIGN batch {b}")
        slots = parse_slots(b)
        if not slots:
            print(f"[loop] no slots for batch {b}; stopping", flush=True); break
        (run_batch_cluster if cluster else run_batch_local)(slots)
        montage(b)
        json.dump({"batch": b + 1}, open(STATE, "w"))
        print(f"[loop] === batch {b} done ({len(slots)} slots) ===", flush=True)


if __name__ == "__main__":
    main()
