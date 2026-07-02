#!/usr/bin/env python
"""embryo_loop -- the agentic scientific loop for the embryogenesis blastula (Phase 1).

Design-first, hypothesis-driven (modelled on prototype/cardio_mpm). Each batch the agent (Claude
CLI) READS its memory + the previous montage, writes a dated analysis entry, distils the knowledge
ledger, and DESIGNS <=6 slots into `embryo_slots.md` -- each slot a spec it authored (composing
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


def design_prompt(batch, n):
    prev = f"montages/embryo_b{batch-1:02d}.png"
    obs = (f"Read the previous batch montage {prev} and each archive/eb_b{batch-1:02d}_*/metrics.json."
           if batch > 1 else
           "FIRST batch: read specs/embryo_base.yaml; design slots that probe the Phase-1 targets "
           "(start by finding a WEAK-coupling regime that deforms the membrane without collapsing).")
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
2. EDIT {ANALYSIS}: append "## Batch {batch}" with observations, per-slot verdicts (supported/falsified/inconclusive), and levers.
3. DISTILL {LEDGER}: merge new causal findings, tagged [established]/[open]/[rejected]/[engineering]; keep compact.
4. STATE ONE predictive hypothesis for this batch.
5. DESIGN <=6 slots into {SLOTS}: one variable/operator change per slot (~3 exploit, 2 explore, 1 control).
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


def run_batch(slots):
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
    start = 1 if fresh else load_state()
    for b in range(start, n + 1):
        if not manual:
            run_claude(design_prompt(b, n), f"DESIGN batch {b}")
        slots = parse_slots(b)
        if not slots:
            print(f"[loop] no slots for batch {b}; stopping"); break
        run_batch(slots)
        montage(b)
        save_state(b + 1)
        print(f"[loop] === batch {b} done ({len(slots)} slots) -> montages/embryo_b{b:02d}.png ===", flush=True)


if __name__ == "__main__":
    main()
