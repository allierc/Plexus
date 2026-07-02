"""am2_cluster.py -- shared LSF-cluster + Claude-CLI machinery for the active_matter2
figure-reproduction loops (Fig. 1 / 2 / 3). A generic sibling of
prototype/cardio_mpm/cardio_mpm_cluster.py, adapted for this project:

  * the compute node is an **L4** (queue gpu_l4), not an A100;
  * up to **4 jobs run in parallel** per batch (MAX_SLOTS), not 8;
  * each slot runs `am2_job.py` (a forward simulation + a diagnostic panel), not a
    trainer -- the loops are about UNDERSTANDING which mechanisms reproduce the paper's
    figures, so there is no scalar objective to minimize, only agreement to inspect.

The three loop drivers (`am2_fig{1,2,3}_loop.py`) import this module as `C`, set the
per-figure file constants (INSTR/LEDGER/ANALYSIS/PLAN/STATE/TRANSCRIPT) and the batch
prefix, then call the same submit/wait/claude helpers. Runs land in `archive/<arch>/`.

  cd prototype/active_matter2
  python am2_fig1_loop.py 20               # 20 batches on the cluster; RESUMES
  python am2_fig1_loop.py 3 --local        # local GPUs (testing)
  AM2_QUEUE=gpu_l4 CLAUDE_TIMEOUT_MIN=30 python am2_fig1_loop.py 20
"""
import os, sys, re, time, shlex, shutil, subprocess, threading, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

PYBIN = os.environ.get("AM2_PYBIN", "/home/allierc@hhmi.org/miniforge3/envs/neural-graph-linux/bin/python")
CONDA_ENV = os.environ.get("AM2_CONDA_ENV", "neural-graph")           # conda env ON the cluster node
# invoke the env python DIRECTLY on the cluster (avoids `conda run` mis-parsing job flags
# like `--n`, which it reads as its own --name/--no-capture-output).
CLUSTER_PY = os.environ.get("AM2_CLUSTER_PY",
                            "/groups/saalfeld/home/allierc/miniforge3/envs/neural-graph/bin/python")
SRC = os.environ.get("AM2_SRC", os.path.abspath(os.path.join(HERE, "..", "..", "src")))
CLAUDE = os.environ.get("CLAUDE_BIN", "claude")
CLUSTER_SSH = os.environ.get("AM2_CLUSTER_SSH", "allierc@login1")
LSF_PROFILE = os.environ.get("AM2_LSF_PROFILE", "/etc/profile.d/profile.lsf.sh")
NODE = os.environ.get("AM2_NODE", "l4")                               # L4 (not A100)
QUEUE = os.environ.get("AM2_QUEUE", f"gpu_{NODE}")
NCPUS = os.environ.get("AM2_NCPUS", "4")
WALL_MIN = int(os.environ.get("AM2_WALL_MIN", "180"))
POLL_SEC = int(os.environ.get("AM2_POLL_SEC", "120"))
TIMEOUT_MIN = float(os.environ.get("CLAUDE_TIMEOUT_MIN", "30"))
LOCAL_GPUS = [g for g in os.environ.get("AM2_LOCAL_GPUS", "0,1").split(",") if g != ""]
MAX_SLOTS = int(os.environ.get("AM2_MAX_SLOTS", "8"))                 # 8 parallel jobs per batch

WORKER = "am2_job.py"                                                 # the per-slot forward-sim worker
ARCHIVE = "archive"
LOGDIR = "loop_logs"

# The devcontainer mounts the shared NFS export /groups/saalfeld/home/allierc/Graph at
# /workspace, so the SAME files are /workspace/... here but /groups/.../Graph/... on the
# cluster. The loop driver (run_claude / montage / file writes) uses the local /workspace
# path; every CLUSTER-SIDE path (bsub cd, job script, -o/-e logs, PYTHONPATH) is translated.
_MAP = os.environ.get("AM2_CLUSTER_ROOT_MAP", "/workspace:/groups/saalfeld/home/allierc/Graph").split(":")


def _cpath(p):
    ap = os.path.abspath(p)
    return (_MAP[1] + ap[len(_MAP[0]):]) if (len(_MAP) == 2 and ap.startswith(_MAP[0])) else ap


CLUSTER_HERE = _cpath(HERE)
CLUSTER_SRC = _cpath(SRC)

# --- per-figure files: the loop driver overrides these before calling main() ------ #
INSTR = "instruction_am2.md"
LEDGER = "knowledge_am2.md"
ANALYSIS = "analysis_am2.md"
USERIN = "user_input.md"
PLAN = "am2_slots.md"
STATE = "am2_loop_state.json"
TRANSCRIPT = "am2_cli_transcript.md"
ARCH_PREFIX = "f0"


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------- #
#  the in-loop agent (Claude CLI)
# --------------------------------------------------------------------------- #
def run_claude(prompt, label):
    """Stream the Claude CLI, tee into the human-only transcript, enforce a wall timeout.
    The agent is allowed to Read the run dirs / paper figure and Edit the memory + slots."""
    with open(TRANSCRIPT, "a") as f:
        f.write(f"\n\n{'='*80}\n## {label} -- {_now()}\n{'='*80}\n")
    cmd = [CLAUDE, "-p", prompt, "--output-format", "text", "--max-turns", "250",
           "--allowedTools", "Read", "Edit", "Write"]
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
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
        print(f"\n[loop] Claude TIMEOUT after {TIMEOUT_MIN} min -- killed (partial edits kept)\n", flush=True)


# --------------------------------------------------------------------------- #
#  slots -> jobs   (agent writes PLAN as `name : --flag val ...` lines)
# --------------------------------------------------------------------------- #
def parse_slots(batch):
    """Parse the agent-written slots file: each non-comment `name : --flag val ...` line."""
    try:
        lines = open(PLAN).read().splitlines()
    except OSError:
        lines = []
    out = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, argstr = line.split(":", 1)
        name = name.strip().replace(" ", "_")
        toks, args, i = argstr.split(), {}, 0
        while i < len(toks):
            if toks[i].startswith("--"):
                if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                    args[toks[i]] = toks[i + 1]; i += 2
                else:
                    args[toks[i]] = ""; i += 1
            else:
                i += 1
        s = len(out)
        arch = f"{ARCH_PREFIX}_b{batch:02d}_s{s}_{name}"
        out.append({"slot": s, "name": name, "args": args, "arch": arch,
                    "dir": os.path.join(ARCHIVE, arch),
                    "log": os.path.join(LOGDIR, f"{arch}.out")})
        if len(out) >= MAX_SLOTS:
            break
    return out


def _argstr(job):
    return " ".join(f"{k} {v}".rstrip() for k, v in job["args"].items())


def _write_manifest(job):
    import json
    os.makedirs(job["dir"], exist_ok=True)
    json.dump({"name": job["name"], "args": job["args"], "launched": _now()},
              open(os.path.join(job["dir"], "config.json"), "w"), indent=2)


def _job_cmd(job, device):
    return f"{WORKER} --outdir {job['dir']} --device {device} {_argstr(job)}"


# --------------------------------------------------------------------------- #
#  LSF cluster submission (bsub over ssh)
# --------------------------------------------------------------------------- #
def _ssh(remote_cmd, retries=1):
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
    """Write the job script on the devcontainer side; its CONTENT uses cluster paths
    (the compute node sees the same file under /groups/.../Graph). Returns (local, cluster)."""
    path = os.path.join(HERE, LOGDIR, f"{job['arch']}.sh")
    with open(path, "w") as f:
        f.write("\n".join(["#!/bin/bash -l", f"cd {CLUSTER_HERE}", f"export PYTHONPATH={CLUSTER_SRC}",
                           f"{CLUSTER_PY} {_job_cmd(job, 'cuda')}"]) + "\n")
    os.chmod(path, 0o755)
    return path, _cpath(path)


def submit_cluster(jobs):
    ids = {}
    for j in jobs:
        _write_manifest(j); _local_script, script = _job_script(j)
        out = _cpath(os.path.join(HERE, j["log"])); err = out[:-4] + ".err"
        bsub = (f"cd {CLUSTER_HERE} && bsub -n {NCPUS} -gpu num=1 -q {QUEUE} -W {WALL_MIN} "
                f"-J {j['arch']} -o {out} -e {err} bash -l {script}")
        res = _ssh(bsub, retries=3)
        m = re.search(r"Job <(\d+)>", res.stdout if res else "")
        if m:
            ids[j["slot"]] = m.group(1)
            print(f"[loop] job {m.group(1)}  slot {j['slot']}  {j['name']:24s} -> {QUEUE}", flush=True)
        else:
            so = (res.stdout if res else "").strip(); se = (res.stderr if res else "").strip()
            print(f"[loop] job -            slot {j['slot']}  {j['name']:24s} -> SUBMIT FAILED: {so} {se}", flush=True)
    return ids


RUNNING_STATES = ("PEND", "RUN", "PROV", "WAIT")


def _bjobs_states(jids):
    if not jids:
        return {}
    res = _ssh(f"source {LSF_PROFILE} && bjobs -a " + " ".join(jids), retries=6)
    states = {}
    for line in (res.stdout if res else "").splitlines():
        p = line.split()
        if len(p) >= 3 and p[0].isdigit():
            states[p[0]] = p[2]
    return states


def _short_progress(job):
    p = os.path.join(job["dir"], "progress.txt")
    try:
        return open(p).read().strip().replace("\n", " | ") if os.path.exists(p) else "(no progress yet)"
    except OSError:
        return "(no progress yet)"


def wait_cluster(ids, jobs):
    pending = set(ids.values())
    while pending:
        states = _bjobs_states(list(ids.values()))
        active = len([j for j in ids.values() if states.get(j) in RUNNING_STATES])
        print(f"[loop] status {_now()}  ({active}/{len(ids)} active):", flush=True)
        for j in jobs:
            jid = ids.get(j["slot"], "-"); st = states.get(jid, "?")
            print(f"[loop]   s{j['slot']} {j['name']:20s} {st:5s}  {_short_progress(j)}", flush=True)
        print(flush=True)
        pending = {jid for jid in ids.values() if states.get(jid) in RUNNING_STATES}
        if pending:
            time.sleep(POLL_SEC)
    print(f"[loop] all {len(ids)} cluster jobs left the queue {_now()}", flush=True)


def check_completion(jobs, ids):
    numeric = [v for v in ids.values() if str(v).isdigit()]
    states = _bjobs_states(numeric) if numeric else {}      # skip SSH bjobs for --local runs
    print(f"[loop] --- batch completion check ({_now()}) ---", flush=True)
    all_ok = True
    for j in jobs:
        jid = ids.get(j["slot"])
        logf = os.path.join(HERE, j["log"])
        done = os.path.exists(logf) and "done ->" in open(logf, errors="ignore").read()
        panel = os.path.exists(os.path.join(j["dir"], "panel.png"))
        flag = "OK" if (panel and (done or jid == "local")) else "INCOMPLETE"
        if flag != "OK":
            all_ok = False
        print(f"[loop]   slot {j['slot']} {j['name']:24s} done={'yes' if done else 'NO '} "
              f"panel={'yes' if panel else 'NO '} -> {flag}", flush=True)
    if not all_ok:
        print("[loop] WARNING: some slots did NOT complete -- the agent treats those as FAILED.", flush=True)
    return all_ok


def run_local(jobs):
    """Run the batch on local GPUs, at most len(LOCAL_GPUS) at once (the --local path)."""
    os.makedirs(LOGDIR, exist_ok=True)
    pybin = PYBIN if os.path.exists(PYBIN) else sys.executable   # local: the interpreter running the loop
    free = list(LOCAL_GPUS) or ["0"]; queue = list(jobs); running = []
    while queue or running:
        while queue and free:
            j = queue.pop(0); gpu = free.pop(0); _write_manifest(j)
            lf = open(os.path.join(HERE, j["log"]), "w")
            cmd = [pybin] + _job_cmd(j, f"cuda:{gpu}").split()
            env = {**os.environ, "PYTHONPATH": SRC}
            p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)
            print(f"[loop] local slot {j['slot']} ({j['name']}) on cuda:{gpu} pid {p.pid}", flush=True)
            running.append((p, j["slot"], gpu, lf))
        for tup in list(running):
            p, slot, gpu, lf = tup
            if p.poll() is not None:
                lf.close(); free.append(gpu); running.remove(tup)
                print(f"[loop] local slot {slot} done (exit {p.returncode}) {_now()}", flush=True)
        if queue or running:
            time.sleep(10)
    print(f"[loop] all local jobs finished {_now()}", flush=True)


# --------------------------------------------------------------------------- #
def load_state():
    import json
    return json.load(open(STATE))["batch"] if os.path.exists(STATE) else 0


def save_state(b):
    import json
    json.dump({"batch": b}, open(STATE, "w"))


def _preflight(local):
    if not shutil.which(CLAUDE):
        sys.exit(f"[loop] ERROR: claude CLI ('{CLAUDE}') not found on PATH. Set CLAUDE_BIN.")
    if local:
        if not os.path.exists(PYBIN) and not shutil.which("python"):
            sys.exit(f"[loop] ERROR: local python ('{PYBIN}') not found. Set AM2_PYBIN.")
        return
    if not shutil.which("ssh"):
        sys.exit("[loop] ERROR: 'ssh' not found; cannot reach the LSF submit host. Use --local.")
    probe = _ssh(f"source {LSF_PROFILE} && command -v bsub", retries=2)
    if not (probe and "bsub" in (probe.stdout or "")):
        sys.exit(f"[loop] ERROR: cannot reach bsub on {CLUSTER_SSH} (check passwordless SSH + "
                 f"{LSF_PROFILE}); or use --local.")
    print(f"[loop] cluster preflight OK: {CLUSTER_SSH} -> {QUEUE}, env '{CONDA_ENV}'", flush=True)


def run_batch(jobs, local):
    """Submit + wait + check one batch (cluster or local)."""
    if local:
        run_local(jobs); ids = {j["slot"]: "local" for j in jobs}
    else:
        ids = submit_cluster(jobs)
        if ids:
            wait_cluster(ids, jobs)
        else:
            print("[loop] no jobs submitted -- aborting batch (check bsub/queue)", flush=True)
    check_completion(jobs, ids)
    return ids
