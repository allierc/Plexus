"""cardio_mpm_loop.py -- agent-in-the-loop model discovery for the MLS-MPM cardio inverse fit.

Each BATCH launches up to 6 `cardio_mpm_train.py` jobs IN PARALLEL on the LSF cluster (one GPU
each), each a config (spec + CLI knobs) from `cardio_mpm_plan.json`. When all finish, the Claude
CLI acts as the SCIENTIST: it reads each job's dashboard (sim-red / real-green trajectories +
learned stiffness + direction dx/dy) and its final interior R2, updates the knowledge ledger
(`knowledge_cardio_mpm.md`) + the running analysis (`analysis_cardio_mpm.md`), and rewrites
`cardio_mpm_plan.json` for the next batch (and MAY edit `cardio_mpm_train.py` / a spec to add a
mechanism). It RANKS on the interior R2 (motion-normalised, boundary excluded).

The knobs are the new MLS-MPM ones: contraction mode/spec family (directional_* | map_*), amplitude,
drag k, pulse duration init (dur0), lr, the differentiable-window cost (substeps / grad / warmup),
fit_beat, bwidth, and the PHASE SWEEP max_delay (>0 => a learnable phase-delay field). The learned
objects are the interpretable fields: stiffness + direction (+ phase-delay tau when max_delay>0).

  cd prototype/cardio_mpm   # run from an LSF SUBMIT HOST for cluster mode
  python cardio_mpm_loop.py 10                 # 10 batches; RESUMES saved state
  python cardio_mpm_loop.py 10 --fresh         # restart from START
  python cardio_mpm_loop.py 4 --local          # run on local GPUs (testing)
  CARDIO_QUEUE=gpu_h100 CLAUDE_TIMEOUT_MIN=40 python cardio_mpm_loop.py 10
"""
import os, sys, re, json, time, shlex, shutil, threading, subprocess, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

PYBIN = os.environ.get("CARDIO_PYBIN", "/home/allierc@hhmi.org/miniforge3/envs/neural-graph-linux/bin/python")
CONDA_ENV = os.environ.get("CARDIO_CONDA_ENV", "neural-graph")     # conda env ON THE CLUSTER NODE
SRC = os.environ.get("CARDIO_SRC", os.path.abspath(os.path.join(HERE, "..", "..", "src")))
CLAUDE = os.environ.get("CLAUDE_BIN", "claude")
CLUSTER_SSH = os.environ.get("CARDIO_CLUSTER_SSH", "allierc@login1")
LSF_PROFILE = os.environ.get("CARDIO_LSF_PROFILE", "/etc/profile.d/profile.lsf.sh")
NODE = os.environ.get("CARDIO_NODE", "a100")
QUEUE = os.environ.get("CARDIO_QUEUE", f"gpu_{NODE}")
NCPUS = os.environ.get("CARDIO_NCPUS", "8")
WALL_MIN = int(os.environ.get("CARDIO_WALL_MIN", "600"))
POLL_SEC = int(os.environ.get("CARDIO_POLL_SEC", "300"))          # status every 5 min
TIMEOUT_MIN = float(os.environ.get("CLAUDE_TIMEOUT_MIN", "40"))
LOCAL_GPUS = [g for g in os.environ.get("CARDIO_LOCAL_GPUS", "0,1").split(",") if g != ""]

TRAIN = "cardio_mpm_train.py"
STATE = "cardio_mpm_loop_state.json"
TRANSCRIPT = "cardio_mpm_cli_transcript.md"                        # human-only raw CLI output
PLAN = "cardio_mpm_plan.json"
INSTR, LEDGER, ANALYSIS, USERIN = "instruction_cardio_mpm.md", "knowledge_cardio_mpm.md", "analysis_cardio_mpm.md", "user_input.md"
LOGDIR = "loop_logs"


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_claude(prompt, label):
    """Stream the Claude CLI, tee stdout into the human-only transcript, enforce a wall timeout."""
    with open(TRANSCRIPT, "a") as f:
        f.write(f"\n\n{'='*80}\n## {label} -- {_now()}\n{'='*80}\n")
    cmd = [CLAUDE, "-p", prompt, "--output-format", "text", "--max-turns", "250",
           "--allowedTools", "Read", "Edit", "Write"]
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    killed = {"v": False}
    timer = threading.Timer(TIMEOUT_MIN * 60, lambda: (killed.__setitem__("v", True), proc.kill())); timer.start()
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
#  jobs
# --------------------------------------------------------------------------- #
def _slots(batch):
    """cardio_mpm_plan.json -> per-slot dicts. Each config: {name, spec, args:{--flag: val}}."""
    plan = json.load(open(PLAN))
    train_script = plan.get("train_script", TRAIN)          # morphology pivot: atlas runner via the plan
    out = []
    for s, cfg in enumerate(plan["configs"][:6]):
        arch = f"mpm_b{batch:02d}_s{s}_{cfg['name']}"
        out.append({"slot": s, "name": cfg["name"], "spec": cfg["spec"], "args": dict(cfg.get("args", {})),
                    "train_script": cfg.get("train_script", train_script),
                    "arch": arch, "dir": os.path.join("archive", arch),
                    "log": os.path.join(LOGDIR, f"{arch}.out")})
    return out


def _argstr(job):
    return " ".join(f"{k} {v}" for k, v in job["args"].items())


def _write_manifest(job):
    os.makedirs(job["dir"], exist_ok=True)
    json.dump({"name": job["name"], "spec": job["spec"], "args": job["args"], "launched": _now()},
              open(os.path.join(job["dir"], "config.json"), "w"), indent=2)


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


def _train_cmd(job, device):
    return (f"{job.get('train_script', TRAIN)} {job['spec']} --device {device} --outdir {job['dir']} {_argstr(job)}")


def _job_script(job):
    path = os.path.join(HERE, LOGDIR, f"{job['arch']}.sh")
    with open(path, "w") as f:
        f.write("\n".join(["#!/bin/bash -l", f"cd {HERE}", f"export PYTHONPATH={SRC}",
                           f"conda run -n {CONDA_ENV} python {_train_cmd(job, 'cuda')}"]) + "\n")
    os.chmod(path, 0o755)
    return path


def submit_cluster(jobs):
    ids = {}
    for j in jobs:
        _write_manifest(j); script = _job_script(j)
        out = os.path.join(HERE, j["log"]); err = out[:-4] + ".err"
        bsub = (f"cd {HERE} && bsub -n {NCPUS} -gpu num=1 -q {QUEUE} -W {WALL_MIN} "
                f"-J {j['arch']} -o {out} -e {err} bash -l {script}")
        res = _ssh(bsub, retries=3)
        m = re.search(r"Job <(\d+)>", res.stdout if res else "")
        if m:
            ids[j["slot"]] = m.group(1)
            print(f"[loop] job {m.group(1)}  slot {j['slot']}  {j['name']:24s} {j['spec'].split('/')[-1]:28s} -> {QUEUE}", flush=True)
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


def _colorize_hrm(text):
    """Color the LS=<score> token green/yellow/red (morphology score, 1=perfect)."""
    def repl(m):
        try:
            v = float(m.group(1))
        except ValueError:
            return f"\033[31m{m.group(0)}\033[0m"                 # nan -> red
        c = "\033[32m" if v >= 0.40 else "\033[33m" if v >= 0.0 else "\033[31m"  # green good / yellow partial / red overshoot(<0)
        return f"{c}{m.group(0)}\033[0m"
    return re.sub(r"LS=([+-]?(?:[0-9.]+|nan))", repl, text)


def _read_progress(job):
    p = os.path.join(job["dir"], "progress.txt")
    try:
        return _colorize_hrm(open(p).read().strip()) if os.path.exists(p) else ""
    except OSError:
        return ""


def _short_progress(job):
    """Compact status: LoopScore (mean+/-sd, colored) FIRST, then R2 and iteration."""
    p = os.path.join(job["dir"], "progress.txt")
    try:
        t = open(p).read() if os.path.exists(p) else ""
    except OSError:
        t = ""
    if not t:
        return "(no progress yet)"
    g = lambda k: (re.search(rf"{k}=([+-]?[0-9.eE+-]+|nan)", t) or [None, "?"])[1]
    it = (re.search(r"it=([0-9]+/[0-9]+)", t) or [None, "?"])[1]
    return _colorize_hrm(f"LS={g('LS')}+/-{g('LS_SD')}  R2={g('R2')}  it={it}")


def _final_r2(job):
    """Parse the FINAL interior R2 from the job stdout ('done -> ... (R2=...)') or progress.txt."""
    for src in (os.path.join(HERE, job["log"]), os.path.join(job["dir"], "progress.txt")):
        try:
            m = re.findall(r"R2=([+-]?[0-9.]+)", open(src).read())
            if m:
                return float(m[-1])
        except OSError:
            pass
    return None


def wait_cluster(ids, jobs):
    pending = set(ids.values())
    while pending:
        states = _bjobs_states(list(ids.values()))
        print(f"[loop] status {_now()}  ({len([j for j in ids.values() if states.get(j) in RUNNING_STATES])}"
              f"/{len(ids)} active):", flush=True)
        for j in jobs:
            jid = ids.get(j["slot"], "-"); st = states.get(jid, "?")
            print(f"[loop]   s{j['slot']} {j['name']:20s} {st:5s}  {_short_progress(j)}", flush=True)
        print(flush=True)                                            # blank line between status blocks
        pending = {jid for jid in ids.values() if states.get(jid) in RUNNING_STATES}
        if pending:
            time.sleep(POLL_SEC)
    print(f"[loop] all {len(ids)} cluster jobs left the queue {_now()}", flush=True)


def check_completion(jobs, ids):
    """Completion signal: the trainer prints 'done ->' at the very end -> ran to completion.
    Also report the final interior R2 (the ranking metric) and the LSF state."""
    states = _bjobs_states(list(ids.values()))
    print(f"[loop] --- batch completion check ({_now()}) ---", flush=True)
    all_ok = True
    for j in jobs:
        jid = ids.get(j["slot"])
        logf = os.path.join(HERE, j["log"])
        done = os.path.exists(logf) and "done ->" in open(logf, errors="ignore").read()
        r2 = _final_r2(j); lsf = states.get(jid, "?") if jid else "NOT-SUBMITTED"
        flag = "OK" if (done and jid) else "INCOMPLETE"
        if flag != "OK":
            all_ok = False
        print(f"[loop]   slot {j['slot']} {j['name']:24s} LSF={lsf:5s} done={'yes' if done else 'NO '} "
              f"R2={r2 if r2 is not None else 'na':>8} -> {flag}", flush=True)
    if not all_ok:
        print("[loop] WARNING: some slots did NOT complete -- the agent treats those as FAILED.", flush=True)
    return all_ok


def run_local(jobs):
    free = list(LOCAL_GPUS) or ["0"]; queue = list(jobs); running = []
    while queue or running:
        while queue and free:
            j = queue.pop(0); gpu = free.pop(0); _write_manifest(j)
            lf = open(os.path.join(HERE, j["log"]), "w")
            cmd = [PYBIN] + _train_cmd(j, f"cuda:{gpu}").split()
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
            time.sleep(15)
    print(f"[loop] all local jobs finished {_now()}", flush=True)


# --------------------------------------------------------------------------- #
def start_prompt():
    return f"""CARDIO-MPM START. You are the scientist in a hypothesis-driven inverse-modelling loop:
a UNet predicts interpretable fields -- STIFFNESS, active-stress DIRECTION, and (when --max_delay>0) a
learnable PHASE-DELAY map tau(x,y) -- that drive the real MLS-MPM contraction forward, fit to the real
cardiomyocyte beat (interior predicted, outer band anchored). tau makes the activation a TRAVELLING wave
a(x,y,t)=pulse(t-tau(x,y)) so neighbouring regions fire in sequence (the substrate for curved / rotary
trajectories); --max_delay=0 recovers a single global beat. Goal is KNOWLEDGE about which fields/knobs make
the learned (red) per-node loops match the real (green) beat -- ranked by the interior R2 (motion-normalised,
boundary excluded; 1=perfect, <=0=worse than no motion). The seed plan is a PHASE SWEEP (--max_delay).

Read (follow ALL of the instruction):
  instruction: {INSTR}
  knowledge ledger (read + keep updating; THE deliverable): {LEDGER}
  running analysis (append): {ANALYSIS}
  user input (read + acknowledge pending, if present): {USERIN}

A seed {PLAN} already exists. Review it against the ledger; if you would design batch 1 differently,
REWRITE {PLAN} (<=6 configs, schema in the instruction). Otherwise leave it. Do not launch anything."""


def analysis_prompt(batch, n, jobs):
    listing = "\n".join(f"  - slot {j['slot']} [{j['name']}] spec={j['spec']}: {j['dir']}/  "
                        f"(config.json, progress.txt, checkpoints/dashboard_*.png + model_*.pt)" for j in jobs)
    return f"""CARDIO-MPM BATCH {batch}/{n}. The parallel cluster jobs you designed have finished.

Read (follow ALL of the instruction):
  instruction: {INSTR}
  knowledge ledger (read + UPDATE -- the deliverable): {LEDGER}
  running analysis (append a dated batch section): {ANALYSIS}
  user input (read + acknowledge pending, if present): {USERIN}

The result directories (Read the images -- trajectory morphology is the primary evidence):
{listing}

CRITICAL: You MUST automatically update {LEDGER} and {ANALYSIS} and rewrite {PLAN} (do not wait for user input).

Steps (do ALL, AUTO-UPDATE the files as you go):
1. Per slot: Read its last `checkpoints/dashboard_*.png` (panels: sim-red/real-green trajectories,
   learned stiffness, direction dx, direction dy, and -- when --max_delay>0 -- the learned phase-delay
   tau(x,y) map in frames) and its final interior R2 (progress.txt / the job
   log `done -> (R2=...)`) + `config.json`. Note in {ANALYSIS}: does red superpose on green? what
   stiffness/direction STRUCTURE did the learned maps show? the R2. RANK on R2.
2. EDIT {LEDGER}: update Established Principles / Falsified Hypotheses / Open Questions, each tied to the
   slot(s) that show it. A clean falsification is a success.
3. EDIT {ANALYSIS}: append a dated batch {batch} section: the configs, R2s, the winner, the reasoning.
4. EDIT {PLAN}: rewrite for the next batch (<=6 one-knob-from-parent configs incl. an ABLATION; you MAY
   edit cardio_mpm_atlas.py / a spec to add a mechanism). Reflect the current best as the parent.
A slot with `done=NO` / R2=na FAILED -- say so and design around it, do not invent results."""


def load_state():
    return json.load(open(STATE))["batch"] if os.path.exists(STATE) else 0


def save_state(b):
    json.dump({"batch": b}, open(STATE, "w"))


def _preflight(local):
    if not shutil.which(CLAUDE):
        sys.exit(f"[loop] ERROR: claude CLI ('{CLAUDE}') not found on PATH. Set CLAUDE_BIN.")
    if local:
        if not os.path.exists(PYBIN):
            sys.exit(f"[loop] ERROR: training python ('{PYBIN}') not found. Set CARDIO_PYBIN.")
        return
    if not shutil.which("ssh"):
        sys.exit("[loop] ERROR: 'ssh' not found; cannot reach the LSF submit host. Use --local.")
    probe = _ssh(f"source {LSF_PROFILE} && command -v bsub", retries=2)
    if not (probe and "bsub" in (probe.stdout or "")):
        sys.exit(f"[loop] ERROR: cannot reach bsub on {CLUSTER_SSH} via SSH (check passwordless SSH + "
                 f"{LSF_PROFILE}); or use --local.")
    envck = _ssh(f"conda run -n {CONDA_ENV} python -c 'import torch'", retries=2)
    if not envck or envck.returncode != 0:
        sys.exit(f"[loop] ERROR: conda env '{CONDA_ENV}' not usable on the cluster (import torch failed). "
                 f"Set CARDIO_CONDA_ENV.")
    print(f"[loop] cluster preflight OK: {CLUSTER_SSH} -> {QUEUE}, env '{CONDA_ENV}'", flush=True)


def main(n_batches, fresh, local):
    _preflight(local); os.makedirs(LOGDIR, exist_ok=True)
    start = 0 if fresh else load_state()
    if start == 0:
        print("[loop] FRESH -- CARDIO-MPM START (agent reviews seed plan)", flush=True)
        run_claude(start_prompt(), "START"); start = 1; save_state(1)
    for b in range(start, n_batches + 1):
        print(f"\n\033[94m[loop] BATCH {b}/{n_batches}  ({'local' if local else QUEUE})\033[0m", flush=True)
        jobs = _slots(b)
        if local:
            run_local(jobs); ids = {j["slot"]: "local" for j in jobs}
        else:
            ids = submit_cluster(jobs)
            if ids:
                wait_cluster(ids, jobs)
            else:
                print("[loop] no jobs submitted -- aborting batch (check bsub/queue)", flush=True)
        check_completion(jobs, ids)
        run_claude(analysis_prompt(b, n_batches, jobs), f"BATCH {b}")
        save_state(b + 1)
    print(f"[loop] DONE through batch {n_batches}. Ledger: {LEDGER}  Analysis: {ANALYSIS}", flush=True)


if __name__ == "__main__":
    fresh = "--fresh" in sys.argv
    local = "--local" in sys.argv
    nb = next((int(a) for a in sys.argv[1:] if a.isdigit()), 10)
    main(nb, fresh, local)
