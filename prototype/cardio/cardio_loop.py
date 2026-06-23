"""cardio_loop.py -- agent-in-the-loop model discovery for the cardio sheet (knowledge over score).

Each BATCH launches 6 training jobs IN PARALLEL on the LSF cluster (one GPU each), each a variant
of cardio_train08_09.py driven by a config (env overrides) from cardio_plan.json. When all 6
finish, the Claude CLI acts as the SCIENTIST: it reads each job's dashboard + run.log, updates the
knowledge ledger (knowledge.md) and the running analysis (analysis.md), and rewrites
cardio_plan.json for the next 6 jobs (and may edit the model code). 6 cluster jobs amortize the
agent call.

The agent maintains: instruction_cardio.md (read), knowledge.md (ledger -- THE deliverable),
analysis.md (per-batch), cardio_plan.json (the next 6 configs), user_input.md (pending notes).
The loop writes (human-only, NOT read by the agent): cardio_cli_transcript.md -- raw CLI output.

  cd prototype/cardio   # run from an LSF SUBMIT HOST for cluster mode (needs bsub/bjobs)
  python cardio_loop.py 10                 # 10 batches (6 cluster jobs each); RESUMES saved state
  python cardio_loop.py 10 --fresh         # restart from CARDIO START (ignore saved state)
  python cardio_loop.py 4 --local          # run on local GPUs instead of the cluster (testing)
  CARDIO_QUEUE=gpu_h100 CLAUDE_TIMEOUT_MIN=40 python cardio_loop.py 10
"""
import os, sys, re, json, time, glob, shlex, shutil, threading, subprocess, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# --- environment / cluster config (override via env) ---
PYBIN = os.environ.get("CARDIO_PYBIN", "/home/allierc@hhmi.org/miniforge3/envs/neural-graph-linux/bin/python")
CONDA_ENV = os.environ.get("CARDIO_CONDA_ENV", "neural-graph")  # conda env ON THE CLUSTER NODE (`conda run -n`)
SRC = os.environ.get("CARDIO_SRC", os.path.abspath(os.path.join(HERE, "..", "..", "src")))
CLAUDE = os.environ.get("CLAUDE_BIN", "claude")
# LSF is reached by SSH to a submit host (bsub/bjobs are NOT on this workstation) -- same convention
# as connectome-gnn's GNN_LLM (ssh login1 -> source the LSF profile -> bsub).
CLUSTER_SSH = os.environ.get("CARDIO_CLUSTER_SSH", "allierc@login1")
LSF_PROFILE = os.environ.get("CARDIO_LSF_PROFILE", "/etc/profile.d/profile.lsf.sh")
NODE = os.environ.get("CARDIO_NODE", "a100")                # GPU node type -> queue gpu_<NODE> ("cluster 100")
QUEUE = os.environ.get("CARDIO_QUEUE", f"gpu_{NODE}")        # LSF GPU queue (override wins)
NCPUS = os.environ.get("CARDIO_NCPUS", "2")
WALL_MIN = int(os.environ.get("CARDIO_WALL_MIN", "360"))     # bsub wall-clock minutes per job
POLL_SEC = int(os.environ.get("CARDIO_POLL_SEC", "300"))    # cluster status+metrics poll interval (s)
TIMEOUT_MIN = float(os.environ.get("CLAUDE_TIMEOUT_MIN", "40"))
LOCAL_GPUS = [g for g in os.environ.get("CARDIO_LOCAL_GPUS", "0,1").split(",") if g != ""]

STATE = "cardio_loop_state.json"
TRANSCRIPT = "cardio_cli_transcript.md"                      # human-only raw CLI output
PLAN = "cardio_plan.json"
INSTR, LEDGER, ANALYSIS, USERIN = "instruction_cardio.md", "knowledge.md", "analysis.md", "user_input.md"
LOGDIR = "loop_logs"


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------- #
#  Claude CLI (the scientist)
# --------------------------------------------------------------------------- #
def run_claude(prompt, label):
    """Stream the Claude CLI, tee stdout into the human-only transcript, enforce a wall timeout."""
    with open(TRANSCRIPT, "a") as f:
        f.write(f"\n\n{'='*80}\n## {label} -- {_now()}\n{'='*80}\n")
    cmd = [CLAUDE, "-p", prompt, "--output-format", "text",
           "--max-turns", "250", "--allowedTools", "Read", "Edit", "Write"]
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    killed = {"v": False}
    timer = threading.Timer(TIMEOUT_MIN * 60, lambda: (killed.__setitem__("v", True), proc.kill()))
    timer.start()
    try:
        with open(TRANSCRIPT, "a") as f:
            for line in proc.stdout:
                print(line, end="", flush=True); f.write(line)
        proc.wait()
    finally:
        timer.cancel()
    if killed["v"]:
        msg = f"\n[loop] Claude TIMEOUT after {TIMEOUT_MIN} min -- killed (partial edits kept)\n"
        with open(TRANSCRIPT, "a") as f:
            f.write(msg)
        print(msg, flush=True)


# --------------------------------------------------------------------------- #
#  Job launching -- one slot per config
# --------------------------------------------------------------------------- #
def _slots(batch):
    """Resolve cardio_plan.json into per-slot dicts: name, arch_name, archive dir, env, log file."""
    plan = json.load(open(PLAN))
    train = plan.get("train_script", "cardio_train08_09.py")
    out = []
    for s, cfg in enumerate(plan["configs"][:6]):
        arch = f"loop_b{batch:02d}_s{s}_{cfg['name']}"
        out.append({"slot": s, "name": cfg["name"], "train": train, "arch": arch,
                    "dir": os.path.join("archive", arch), "env": dict(cfg.get("env", {})),
                    "log": os.path.join(LOGDIR, f"{arch}.out")})
    return out


def _job_env(job):
    """Base env for a training job: PYTHONPATH + the script knobs + ARCH_NAME (loop owns these)."""
    env = {**os.environ, "PYTHONPATH": SRC, "CARDIO_ARCH_NAME": job["arch"], **job["env"]}
    return env


def _write_manifest(job):
    os.makedirs(job["dir"], exist_ok=True)
    json.dump({"name": job["name"], "arch": job["arch"], "env": job["env"], "launched": _now()},
              open(os.path.join(job["dir"], "config.json"), "w"), indent=2)


def _ssh(remote_cmd, retries=1):
    """Run a command on the LSF submit host over SSH under a login shell (sources the LSF profile).
    remote_cmd must contain NO single quotes (we shlex-quote it into `bash -l -c '...'`)."""
    payload = f"bash -l -c {shlex.quote(remote_cmd)}"
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", CLUSTER_SSH, payload]
    last = None
    for attempt in range(retries):
        last = subprocess.run(cmd, capture_output=True, text=True)
        if last.returncode == 0 or last.stdout.strip():
            return last
        time.sleep(min(30, 5 * (2 ** attempt)))
    return last


def _job_script(job):
    """Write a per-job launch script (runs on the compute node under a login shell)."""
    path = os.path.join(HERE, LOGDIR, f"{job['arch']}.sh")
    lines = ["#!/bin/bash -l", f"cd {HERE}", f"export PYTHONPATH={SRC}",
             "export CARDIO_DEVICE=cuda", f"export CARDIO_ARCH_NAME={job['arch']}"]
    lines += [f"export {k}={v}" for k, v in job["env"].items()]
    lines.append(f"conda run -n {CONDA_ENV} python {job['train']}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    return path


def submit_cluster(jobs):
    """bsub each job (one GPU each) on the submit host via SSH; return {slot: lsf_job_id}."""
    ids = {}
    for j in jobs:
        _write_manifest(j)
        script = _job_script(j)
        out = os.path.join(HERE, j["log"]); err = out[:-4] + ".err"
        bsub = (f"cd {HERE} && bsub -n {NCPUS} -gpu num=1 -q {QUEUE} -W {WALL_MIN} "
                f"-J {j['arch']} -o {out} -e {err} bash -l {script}")
        res = _ssh(bsub, retries=3)
        m = re.search(r"Job <(\d+)>", res.stdout if res else "")
        if m:
            ids[j["slot"]] = m.group(1)
            print(f"[loop] job {m.group(1)}  slot {j['slot']}  {j['name']:26s} -> {QUEUE} (submitted)", flush=True)
        else:
            so = (res.stdout if res else "").strip(); se = (res.stderr if res else "").strip()
            print(f"[loop] job -            slot {j['slot']}  {j['name']:26s} -> SUBMIT FAILED: {so} {se}", flush=True)
    return ids


RUNNING_STATES = ("PEND", "RUN", "PROV", "WAIT")


def _bjobs_states(jids):
    """Map job id -> LSF status via `bjobs -a` (includes finished DONE/EXIT for a grace period)."""
    if not jids:
        return {}
    res = _ssh(f"source {LSF_PROFILE} && bjobs -a " + " ".join(jids), retries=6)
    states = {}
    for line in (res.stdout if res else "").splitlines():
        p = line.split()
        if len(p) >= 3 and p[0].isdigit():
            states[p[0]] = p[2]
    return states


def _read_progress(job):
    """Latest live metric line the training wrote (progress.txt); '' until the first eval."""
    p = os.path.join(job["dir"], "progress.txt")
    try:
        return open(p).read().strip() if os.path.exists(p) else ""
    except OSError:
        return ""


def wait_cluster(ids, jobs):
    """Poll bjobs over SSH every POLL_SEC until all jobs leave PEND/RUN; each cycle print a
    per-job status+metrics table (job id leading each row)."""
    pending = set(ids.values())
    while pending:
        states = _bjobs_states(list(ids.values()))               # all ids, so finished show DONE/EXIT too
        print(f"[loop] status {_now()}  ({len([j for j in ids.values() if states.get(j) in RUNNING_STATES])}"
              f"/{len(ids)} active):", flush=True)
        for j in jobs:
            jid = ids.get(j["slot"], "-")
            st = states.get(jid, "?")
            prog = _read_progress(j) or "(no progress yet)"
            print(f"[loop]   job {jid}  s{j['slot']} {j['name']:24s} {st:5s}  {prog}", flush=True)
        pending = {jid for jid in ids.values() if states.get(jid) in RUNNING_STATES}
        if pending:
            time.sleep(POLL_SEC)
    print(f"[loop] all {len(ids)} cluster jobs left the queue {_now()}", flush=True)


def check_completion(jobs, ids):
    """Did each job's TRAINING finish? `run.log` is written only at the end of main() (after the
    loop + final renders), so its presence == ran to completion. Also report the LSF exit state and
    flag a non-empty stderr. Returns True iff every submitted slot completed."""
    states = _bjobs_states(list(ids.values()))
    print(f"[loop] --- batch completion check ({_now()}) ---", flush=True)
    all_ok = True
    for j in jobs:
        jid = ids.get(j["slot"])
        done = os.path.exists(os.path.join(j["dir"], "run.log"))
        lsf = states.get(jid, "?") if jid else "NOT-SUBMITTED"
        errf = os.path.join(HERE, j["log"][:-4] + ".err")
        stderr_nonempty = os.path.exists(errf) and os.path.getsize(errf) > 0
        flag = "OK" if (done and jid) else "INCOMPLETE"
        if flag != "OK":
            all_ok = False
        note = "  (stderr non-empty)" if stderr_nonempty and not done else ""
        print(f"[loop]   slot {j['slot']} {j['name']:26s} LSF={lsf:5s} run.log={'yes' if done else 'NO '} -> {flag}{note}",
              flush=True)
    if not all_ok:
        print("[loop] WARNING: some slots did NOT complete — the agent is told to treat those as FAILED.", flush=True)
    return all_ok


def run_local(jobs):
    """Fallback: run the jobs on local GPUs (round-robin, <= len(LOCAL_GPUS) concurrent)."""
    procs = []  # (proc, slot, gpu)
    free = list(LOCAL_GPUS) or ["0"]
    queue = list(jobs)
    running = []
    while queue or running:
        while queue and free:
            j = queue.pop(0); gpu = free.pop(0)
            _write_manifest(j)
            env = _job_env(j); env["CARDIO_DEVICE"] = f"cuda:{gpu}"   # pin by device id (no CUDA_VISIBLE reindex)
            lf = open(os.path.join(HERE, j["log"]), "w")
            p = subprocess.Popen([PYBIN, j["train"]], cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)
            print(f"[loop] local slot {j['slot']} ({j['name']}) on cuda:{gpu} pid {p.pid}", flush=True)
            running.append((p, j["slot"], gpu, lf))
        for tup in list(running):
            p, slot, gpu, lf = tup
            if p.poll() is not None:
                lf.close(); free.append(gpu); running.remove(tup)
                print(f"[loop] local slot {slot} done (exit {p.returncode}) {_now()}", flush=True)
        if queue or running:
            time.sleep(15)
    print(f"[loop] all local jobs finished {_now()}", flush=True)


# --------------------------------------------------------------------------- #
#  Prompts
# --------------------------------------------------------------------------- #
def start_prompt():
    return f"""CARDIO START. You are the scientist in a hypothesis-driven model-discovery loop on
the real cardiomyocyte sheet. Goal: KNOWLEDGE about what mechanics open the per-node trajectory
loops, superpose learned (red) on real (green), and stay periodic across cycles -- not merely a
low fit-RMSE.

Read (follow ALL of the instruction):
  instruction: {INSTR}
  knowledge ledger (read + keep updating; THE deliverable): {LEDGER}
  running analysis (append): {ANALYSIS}
  user input (read + acknowledge pending, if present): {USERIN}

A seed {PLAN} already exists (6 configs probing 08_09 limit-cycle stability). Review it against the
ledger and, if you would design batch 1 differently, REWRITE {PLAN} (exactly 6 configs, schema in
the instruction). Otherwise leave it. Do not launch anything; the loop launches the 6 cluster jobs."""


def analysis_prompt(batch, n, jobs):
    listing = "\n".join(
        f"  - slot {j['slot']} [{j['name']}]: {j['dir']}/  "
        f"(config.json, run.log, true_vs_learned.png, checkpoints/dashboard_*.png)"
        for j in jobs)
    return f"""CARDIO BATCH {batch}/{n}. The 6 parallel cluster jobs you designed have finished.

Read (follow ALL of the instruction):
  instruction: {INSTR}
  knowledge ledger (read + UPDATE -- the deliverable): {LEDGER}
  running analysis (append a dated batch section): {ANALYSIS}
  user input (read + acknowledge pending, if present): {USERIN}

The 6 result directories (Read the images -- morphology is the primary evidence):
{listing}

Steps (do ALL):
1. For each slot: Read its last `checkpoints/dashboard_*.png` AND `true_vs_learned.png`; read
   `run.log` (fit-RMSE / drift / trans|max| / scalars) and `config.json` (the env it ran with).
   Note in {ANALYSIS}: does red superpose on green? are the loops OPEN or collapsed? are there
   STREAKS (cross-cycle drift) and where? the numbers.
2. Update {LEDGER}: Established Principles / Falsified Hypotheses / Open Questions (each tied to the
   slot(s) that show it). A clean falsification is a success.
3. Append a dated batch section to {ANALYSIS}: the 6 configs, numbers, the winner, the reasoning.
4. Rewrite {PLAN} for the next 6 jobs: refine around the current best, drop saturated knobs, add a
   new hypothesis (you MAY edit cardio_train08_09.py / cardio_real_fit.py to add a mechanism) and
   keep an ABLATION slot. Reflect the current best as the parent/control.
If a slot's directory is missing run.log or dashboards, treat that job as FAILED -- say so and design
around it (e.g. fix the config or the code) rather than inventing results."""


# --------------------------------------------------------------------------- #
def load_state():
    return json.load(open(STATE))["batch"] if os.path.exists(STATE) else 0


def save_state(b):
    json.dump({"batch": b}, open(STATE, "w"))


def _preflight(local):
    """Fail fast with actionable guidance if the chosen backend isn't usable from this host."""
    if not shutil.which(CLAUDE):
        sys.exit(f"[loop] ERROR: claude CLI ('{CLAUDE}') not found on PATH. Set CLAUDE_BIN.")
    if local:
        if not os.path.exists(PYBIN):
            sys.exit(f"[loop] ERROR: training python ('{PYBIN}') not found. Set CARDIO_PYBIN.")
        return
    # Cluster mode: LSF lives on a submit host reached by SSH (bsub is not on this workstation).
    if not shutil.which("ssh"):
        sys.exit("[loop] ERROR: 'ssh' not found; cannot reach the LSF submit host. Use --local.")
    probe = _ssh(f"source {LSF_PROFILE} && command -v bsub", retries=2)
    if not (probe and "bsub" in (probe.stdout or "")):
        se = (probe.stderr if probe else "").strip()
        sys.exit(f"[loop] ERROR: cannot reach bsub on {CLUSTER_SSH} via SSH.\n"
                 f"        ssh probe said: {se or '(no output)'}\n"
                 f"        Check passwordless SSH to {CLUSTER_SSH} (CARDIO_CLUSTER_SSH) and the LSF\n"
                 f"        profile ({LSF_PROFILE}); or use --local to run on local GPUs.")
    # The compute nodes see a DIFFERENT conda base than this workstation -- verify the env exists
    # there and has torch BEFORE burning a batch of submissions on EnvironmentLocationNotFound.
    envck = _ssh(f"conda run -n {CONDA_ENV} python -c 'import torch'", retries=2)
    if not envck or envck.returncode != 0:
        se = (envck.stderr if envck else "").strip().splitlines()
        sys.exit(f"[loop] ERROR: conda env '{CONDA_ENV}' not usable on the cluster (import torch failed).\n"
                 f"        {se[-1] if se else '(no output)'}\n"
                 f"        Set CARDIO_CONDA_ENV to a cluster env with torch "
                 f"(node conda base differs from this workstation).")
    print(f"[loop] cluster preflight OK: {CLUSTER_SSH} -> {QUEUE}, env '{CONDA_ENV}' (bsub + torch reachable)",
          flush=True)


def main(n_batches, fresh, local):
    _preflight(local)
    os.makedirs(LOGDIR, exist_ok=True)
    start = 0 if fresh else load_state()      # resume from saved state by default; --fresh restarts
    if start == 0:
        print("[loop] FRESH -- CARDIO START (agent reviews seed plan)", flush=True)
        run_claude(start_prompt(), "CARDIO START"); start = 1; save_state(1)
    for b in range(start, n_batches + 1):
        print(f"\n\033[94m[loop] BATCH {b}/{n_batches}  ({'local' if local else QUEUE})\033[0m", flush=True)
        jobs = _slots(b)
        if local:
            run_local(jobs)
            ids = {j["slot"]: "local" for j in jobs}              # no LSF id; run.log is the completion signal
        else:
            ids = submit_cluster(jobs)
            if ids:
                wait_cluster(ids, jobs)
            else:
                print("[loop] no jobs submitted -- aborting batch (check bsub/queue)", flush=True)
        check_completion(jobs, ids)                               # report per-slot OK/INCOMPLETE before analysis
        run_claude(analysis_prompt(b, n_batches, jobs), f"BATCH {b}")
        save_state(b + 1)
    print(f"[loop] DONE through batch {n_batches}. Ledger: {LEDGER}  Analysis: {ANALYSIS}", flush=True)


if __name__ == "__main__":
    fresh = "--fresh" in sys.argv             # restart from CARDIO START (default: resume saved state)
    local = "--local" in sys.argv
    nb = next((int(a) for a in sys.argv[1:] if a.isdigit()), 10)
    main(nb, fresh, local)
