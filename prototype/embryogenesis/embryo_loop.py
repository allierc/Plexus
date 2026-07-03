#!/usr/bin/env python
"""embryo_loop -- the agentic scientific loop for the embryogenesis blastula (Phase 1).

Design-first, hypothesis-driven (modelled on prototype/cardio_mpm). Each batch the agent (Claude
CLI) READS its memory + the previous montage, writes a dated analysis entry, distils the knowledge
ledger, and DESIGNS 8 slots into `embryo_slots.md` -- each slot a spec it authored (composing
operators from the whole codebase) plus optional dotted overrides. The loop then runs those slots
(2 local GPUs, or an L4 cluster), each delivering the 2x2 mp4 + Phase-1 metrics, and tiles a
montage. Objective: understand which operators/couplings produce flow-driven membrane deformation,
non-collapsing coverage, division-driven shape change, collective migration, and two-type partition.

  cd prototype/embryogenesis
  python embryo_loop.py 20                 # 20 batches (Claude designs each); RESUMES
  python embryo_loop.py 20 --fresh         # restart from batch 1
  python embryo_loop.py 1 --manual         # skip Claude: just run whatever is in embryo_slots.md
"""
import os, sys, re, json, time, shutil, subprocess, threading

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PYBIN = os.environ.get("EMBRYO_PYBIN", sys.executable)
CLAUDE = os.environ.get("CLAUDE_BIN", "claude")
LOCAL_GPUS = [g for g in os.environ.get("EMBRYO_GPUS", "0,1").split(",") if g != ""]
FRAMES = int(os.environ.get("EMBRYO_FRAMES", "3000"))
STRIDE = int(os.environ.get("EMBRYO_STRIDE", "3"))
TIMEOUT_MIN = float(os.environ.get("CLAUDE_TIMEOUT_MIN", "30"))
WORKER = os.path.join(HERE, "showcase.py")
ARCHIVE = "archive"; SLOTS = "embryo_slots.md"; STATE = "embryo_loop_state.json"
INSTR = "instruction_embryo.md"; LEDGER = "knowledge_embryo.md"; ANALYSIS = "analysis_embryo.md"
USERIN = "user_input.md"; TRANSCRIPT = "embryo_cli_transcript.md"

# --- L4 cluster (bsub over ssh); design runs locally, each slot's forward-sim runs on an L4 ---
CLUSTER_SSH = os.environ.get("EMBRYO_CLUSTER_SSH", "allierc@login1")
CLUSTER_PY = os.environ.get("EMBRYO_CLUSTER_PY", "/groups/saalfeld/home/allierc/miniforge3/envs/neural-graph/bin/python")
CLUSTER_SRC = os.environ.get("EMBRYO_CLUSTER_SRC", "/groups/saalfeld/home/allierc/Graph/Plexus/src")
LSF_PROFILE = os.environ.get("EMBRYO_LSF_PROFILE", "/etc/profile.d/profile.lsf.sh")
QUEUE = os.environ.get("EMBRYO_QUEUE", "gpu_l4")
NCPUS = os.environ.get("EMBRYO_NCPUS", "4")
WALL_MIN = int(os.environ.get("EMBRYO_WALL_MIN", "30"))
POLL_SEC = int(os.environ.get("EMBRYO_POLL_SEC", "60"))
MAX_SLOTS = int(os.environ.get("EMBRYO_MAX_SLOTS", "8"))
_MAP = os.environ.get("EMBRYO_ROOT_MAP", "/workspace:/groups/saalfeld/home/allierc/Graph").split(":")


def _cpath(p):
    ap = os.path.abspath(p)
    return (_MAP[1] + ap[len(_MAP[0]):]) if (len(_MAP) == 2 and ap.startswith(_MAP[0])) else ap


CLUSTER_HERE = _cpath(HERE)


def _ssh(cmd, retries=3):
    for _ in range(retries):
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", CLUSTER_SSH,
                            f"source {LSF_PROFILE} 2>/dev/null; {cmd}"], capture_output=True, text=True)
        if r.returncode == 0:
            return r
        time.sleep(5)
    return r


def submit_cluster(slots, batch):
    ids = {}
    for slot in slots:
        tag = slot["name"]
        script = os.path.join("loop_logs", f"{tag}.sh"); os.makedirs("loop_logs", exist_ok=True)
        ov = " ".join(slot["ov"])
        with open(script, "w") as f:
            f.write("#!/bin/bash -l\n"
                    f"cd {CLUSTER_HERE}\n"
                    f"export PYTHONPATH={CLUSTER_SRC}\n"
                    f"echo START $(date +%s) $(hostname)\n"
                    f"{CLUSTER_PY} showcase.py {slot['spec']} tag={tag} frames={FRAMES} stride={STRIDE} {ov}\n"
                    f"echo END $(date +%s)\n")
        out = _cpath(os.path.join(HERE, "loop_logs", f"{tag}.out")); err = out[:-4] + ".err"
        bsub = (f"cd {CLUSTER_HERE} && bsub -n {NCPUS} -gpu num=1 -q {QUEUE} -W {WALL_MIN} "
                f"-J {tag} -o {out} -e {err} bash -l {_cpath(os.path.join(HERE, script))}")
        r = _ssh(bsub)
        m = re.search(r"Job <(\d+)>", r.stdout if r else "")
        if m:
            ids[tag] = m.group(1); print(f"[loop] L4 job {m.group(1)}  {tag}", flush=True)
        else:
            print(f"[loop] SUBMIT FAILED {tag}: {(r.stdout if r else '')} {(r.stderr if r else '')}", flush=True)
    return ids


def poll_cluster(ids):
    if not ids:
        return
    live = set(ids.values())
    while live:
        time.sleep(POLL_SEC)
        r = _ssh("bjobs -noheader -o 'id stat' " + " ".join(sorted(live)))
        txt = (r.stdout if r else "") or ""
        done = set()
        for jid in list(live):
            st = next((ln.split()[1] for ln in txt.splitlines() if ln.split() and ln.split()[0] == jid), "DONE")
            if st not in ("PEND", "RUN", "PROV", "WAIT"):
                done.add(jid)
        live -= done
        if done:
            print(f"[loop] {len(live)} L4 jobs still running", flush=True)


BATCH_JOBS = "embryo_batch_jobs.json"


def save_batch_jobs(batch, ids):
    json.dump({"batch": batch, "ids": ids}, open(BATCH_JOBS, "w"))


def load_batch_jobs():
    return json.load(open(BATCH_JOBS)) if os.path.isfile(BATCH_JOBS) else None


def jobs_live(ids):
    """Which of these job ids are still PEND/RUN on the cluster (survives driver restarts)."""
    if not ids:
        return set()
    r = _ssh("bjobs -noheader -o 'id stat' " + " ".join(sorted(set(str(v) for v in ids.values()))))
    txt = (r.stdout if r else "") or ""
    live = set()
    for ln in txt.splitlines():
        p = ln.split()
        if len(p) >= 2 and p[1] in ("PEND", "RUN", "PROV", "WAIT"):
            live.add(p[0])
    return live


def run_batch_cluster(slots, batch):
    ids = submit_cluster(slots, batch)
    save_batch_jobs(batch, ids)                     # persist so a restart RESUMES these, not resubmits
    poll_cluster(ids)
    print("[loop] L4 batch complete", flush=True)


PHASE_HOURS = float(os.environ.get("EMBRYO_PHASE_HOURS", "24"))   # max wall-clock per sub-phase
PHASE_TIMER = "phase_timer.json"


def phase_elapsed_h():
    """(stage, hours) the CURRENT stage (current_stage.txt) has run; clock starts on first sight."""
    stage = ""
    if os.path.isfile("current_stage.txt"):
        raw = open("current_stage.txt").read().strip()
        stage = raw.split()[0] if raw else ""
    if not stage:
        return "", 0.0
    t = {}
    if os.path.isfile(PHASE_TIMER):
        try:
            t = json.load(open(PHASE_TIMER))
        except Exception:
            t = {}
    now = time.time()
    if stage not in t:
        t[stage] = now
        json.dump(t, open(PHASE_TIMER, "w"))
    return stage, (now - t[stage]) / 3600.0


def design_prompt(batch, n):
    stage, eh = phase_elapsed_h()
    timecap = ""
    if stage and eh >= PHASE_HOURS:
        timecap = (f"\n\n>>> TIME CAP HIT: sub-phase {stage} has run {eh:.1f}h (>= {PHASE_HOURS:.0f}h budget). "
                   f"THIS BATCH you MUST: adopt the best clean (escape-free) point as {stage}'s operating spec, "
                   f"log the blocker as [open] in the ledger, ADVANCE to the next sub-phase, and write the new "
                   f"stage to current_stage.txt (which resets the phase clock). Do not run more {stage} experiments.")
    prev = f"montages/embryo_b{batch-1:02d}.png"
    obs = (f"Read the previous batch montage {prev}, and for EACH slot read archive/eb_b{batch-1:02d}_*/"
           f"scorecard.json (5-family metrics at 5/25/50/75/100%) + metrics.json (hard-failure gate) + scorecard.png. "
           f"Decide on the NUMBERS + their trajectory, not the movie."
           if batch > 1 else
           "FIRST batch (fresh restart): read specs/embryo_base.yaml, knowledge_embryo.md (system + scorecard "
           "+ the zebrafish quantitative reference + provisional pilot hints to RE-VERIFY). Target Stage 1A "
           "(stable, no collapse) and design slots that establish the scorecard baseline.")
    return f"""EMBRYOGENESIS BLASTULA / PHASE 1 -- BATCH {batch}/{n}. You are a SCIENTIST discovering
which operators + couplings make a flowing, dividing, self-partitioning blastula (NOT a param search).

Your MEMORY (read EVERY batch):
  method + RULES + operator catalog + slot schema : {INSTR}
  knowledge ledger (CUMULATIVE -- read + curate, never erase) : {LEDGER}
  analysis log (append a dated batch section) : {ANALYSIS}
  user input (read + acknowledge if non-empty) : {USERIN}

{obs}

Do ALL, in order, AUTO-UPDATING the files:
1. OBSERVE: from the montage/metrics, what happened vs last batch's predictions? (collapse? deform? flow? migration? partition?)
2. EDIT {ANALYSIS}: append "## Batch {batch}". EVERY claim = QUANTITATIVE REPORT PROTOCOL: pair each
   visual observation with scorecard support (e.g. "lobed" -> "shape.fourier_m3 0.006->0.013 (2.1x); circularity 0.92->0.78").
   A claim with no scorecard number is an opinion, not a finding.
3. DISTILL {LEDGER}: merge findings tagged [established]/[open]/[rejected]/[engineering]. Promote to
   [established] ONLY with >=3 seeds AND |delta|>2*SD vs control (report mean+/-SD); else it is [open]. Keep compact.
4. STATE ONE predictive hypothesis for this batch.
5. DESIGN 8 slots into {SLOTS}: one variable/operator change per slot (~4 exploit, 3 explore, 1 control).
   Each line: `name : SPEC specs/<file>.yaml [key val ...]`. AUTHOR the per-slot spec YAML (copy embryo_base, edit operators) when you change the mechanism; use dotted overrides for scalar tweaks.
Keep every job within the ~20-min-on-L4 budget."""


def run_claude(prompt, label):
    print(f"[loop] Claude: {label}", flush=True)
    cmd = [CLAUDE, "-p", prompt, "--output-format", "text", "--max-turns", "250",
           "--allowedTools", "Read", "Edit", "Write"]
    with open(TRANSCRIPT, "a") as tf:
        tf.write(f"\n\n===== {label} =====\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        timer = threading.Timer(TIMEOUT_MIN * 60, p.kill); timer.start()
        try:
            for line in p.stdout:
                tf.write(line); tf.flush()
        finally:
            p.wait(); timer.cancel()


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
        name = name.strip().replace(" ", "_")
        toks = rest.split()
        spec = "specs/embryo_base.yaml"; ov = []
        i = 0
        while i < len(toks):
            if toks[i] == "SPEC" and i + 1 < len(toks):
                spec = toks[i + 1]; i += 2
            elif i + 1 < len(toks):
                ov.append(f"{toks[i]}={toks[i+1]}"); i += 2
            else:
                i += 1
        arch = f"eb_b{batch:02d}_s{len(out)}_{name}"
        out.append({"name": arch, "spec": spec, "ov": ov})
    return out


def run_slot(slot, gpu):
    tag = slot["name"]
    cmd = [PYBIN, WORKER, slot["spec"], f"tag={tag}", f"frames={FRAMES}", f"stride={STRIDE}", *slot["ov"]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log = os.path.join("loop_logs", f"{tag}.log"); os.makedirs("loop_logs", exist_ok=True)
    print(f"[loop] slot {tag} on gpu{gpu}: {' '.join(slot['ov'])}", flush=True)
    with open(log, "w") as lf:
        return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)


def run_batch_local(slots, batch):
    running = []
    queue = list(slots)
    while queue or running:
        while queue and len(running) < len(LOCAL_GPUS):
            gpu = LOCAL_GPUS[len(running) % len(LOCAL_GPUS)]
            running.append((run_slot(queue.pop(0), gpu), gpu))
        time.sleep(3)
        running = [(p, g) for (p, g) in running if p.poll() is None]
    # showcase tags runs "<specname>_<tag>"; metrics land in archive/<tag-name-as-set-in-showcase>
    print("[loop] batch complete", flush=True)


def rename_batch_dirs(batch):
    """After montage, rename this batch's archive dirs `*eb_b<NN>_*` ->
    `embryo_<stage>_b<NN>_<rest>` using the stage Claude wrote to current_stage.txt."""
    stage = ""
    if os.path.isfile("current_stage.txt"):
        stage = open("current_stage.txt").read().strip().split()[0][:3] if open("current_stage.txt").read().strip() else ""
    if not stage:
        return
    import glob
    for d in glob.glob(os.path.join(ARCHIVE, f"*eb_b{batch:02d}_*")):
        base = os.path.basename(d)
        rest = re.sub(r".*eb_b\d+_", "", base)
        new = os.path.join(ARCHIVE, f"embryo_{stage}_b{batch:02d}_{rest}")
        if os.path.abspath(d) != os.path.abspath(new) and not os.path.exists(new):
            try:
                os.rename(d, new)
            except OSError:
                pass


def montage(batch):
    try:
        subprocess.run([PYBIN, "montage.py", "--out", f"embryo_b{batch:02d}.png", f"eb_b{batch:02d}"], check=False)
    except Exception as e:
        print("[loop] montage failed:", e)


def load_state():
    return json.load(open(STATE)).get("batch", 1) if os.path.isfile(STATE) else 1


def save_state(b):
    json.dump({"batch": b}, open(STATE, "w"))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
    fresh = "--fresh" in sys.argv; manual = "--manual" in sys.argv
    cluster = ("--cluster" in sys.argv) or bool(os.environ.get("EMBRYO_CLUSTER"))
    run_batch = run_batch_cluster if cluster else run_batch_local
    print(f"[loop] mode = {'L4 cluster' if cluster else 'local GPUs'}  frames={FRAMES} stride={STRIDE}", flush=True)
    start = 1 if fresh else load_state()
    for b in range(start, n + 1):
        bj = load_batch_jobs() if cluster else None
        live = jobs_live(bj["ids"]) if (bj and bj.get("batch") == b) else set()
        if live:                                    # RESUME: this batch's jobs are already on the cluster
            print(f"[loop] RESUME batch {b}: {len(live)} jobs still on L4 — polling (no redesign/resubmit)", flush=True)
            poll_cluster(bj["ids"])
        else:
            if not manual:
                run_claude(design_prompt(b, n), f"DESIGN batch {b}")
            slots = parse_slots(b)
            if not slots:
                print(f"[loop] no slots for batch {b}; stopping"); break
            run_batch(slots, b)
        montage(b)
        rename_batch_dirs(b)                         # -> embryo_<stage>_b<NN>_... (consistent archive naming)
        save_state(b + 1)
        print(f"[loop] === batch {b} done ({len(slots)} slots) -> montages/embryo_b{b:02d}.png ===", flush=True)


if __name__ == "__main__":
    main()
