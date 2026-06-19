"""dicty_loop.py -- agent-in-the-loop model discovery for dicty (knowledge over score).

Each batch runs a STRUCTURED SWEEP set: 16 single-parameter sweeps x 16 values = 256 simulations
(eval_sweeps.py, fresh process so operator/engine CODE edits are picked up). The Claude CLI then
acts as the SCIENTIST: it reads the per-sweep response figures + the best montage, updates the
knowledge ledger, and rewrites sweep_plan.json (and may edit operator/engine code to test a new
mechanism). 256 cheap sims amortize the agent call.

The agent maintains: dicty_knowledge.md (ledger), dicty_loop_log.md (append), user_input.md,
sweep_plan.json (the experiment design), specs/dicty_loop_base.yaml (the parent/control config).
The loop writes (human-only, NOT read by the agent): dicty_cli_transcript.md -- raw CLI output.

  cd prototype/dicty
  PYTHONPATH=../../src python dicty_loop.py 30            # 30 batches (256 sims each), fresh
  PYTHONPATH=../../src python dicty_loop.py 30 --resume   # continue from saved batch
  EVAL_DEVICE=cuda:0 CLAUDE_TIMEOUT_MIN=30 python dicty_loop.py 30
"""
import os, sys, json, glob, threading, subprocess, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
STATE = "dicty_loop_state.json"
TRANSCRIPT = "dicty_cli_transcript.md"          # human-only raw CLI output (agent never reads this)
EVAL_DEVICE = os.environ.get("EVAL_DEVICE", "cuda:0")
TIMEOUT_MIN = float(os.environ.get("CLAUDE_TIMEOUT_MIN", "30"))
PYBIN = "/workspace/.conda_envs/neural-graph-linux/bin/python"
EP = "/workspace/Plexus/src:/workspace/Plexus/prototype"

INSTR, LEDGER, LOG, USERIN = "instruction_dicty.md", "dicty_knowledge.md", "dicty_loop_log.md", "user_input.md"
PLAN, BASE, RESULTS = "sweep_plan.json", "specs/dicty_loop_base.yaml", "sweep_results.json"


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_claude(prompt, label):
    """Stream the Claude CLI, tee stdout into the human-only transcript, enforce a wall timeout."""
    with open(TRANSCRIPT, "a") as f:
        f.write(f"\n\n{'='*80}\n## {label} — {_now()}\n{'='*80}\n")
    cmd = ["claude", "-p", prompt, "--output-format", "text",
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
        with open(TRANSCRIPT, "a") as f:
            f.write(f"\n[loop] TIMEOUT after {TIMEOUT_MIN} min — killed (partial edits kept)\n")
        print(f"\n[loop] TIMEOUT after {TIMEOUT_MIN} min", flush=True)


def _engine_from_spec():
    """Auto-pick the engine: peek at the base spec; if it schedules any MPM-only operator
    (e.g. `inflow_mpm`, `mpm`), select `dicty_engine_mpm`. Env override `DICTY_ENGINE` wins
    if set. Fixes the B32 silent-failure where the MPM base spec was launched against the
    point-cell engine (operator 'inflow_mpm' not in registry) and all 256 sims NaN-ed."""
    forced = os.environ.get("DICTY_ENGINE")
    if forced:
        return forced
    try:
        with open(os.path.join(HERE, BASE)) as f:
            blob = f.read()
        # cheap textual sniff -- avoids importing yaml here
        if "inflow_mpm" in blob or "op: mpm" in blob or "particle:" in blob:
            return "dicty_engine_mpm"
    except Exception:
        pass
    return "dicty_engine"


def eval_sweeps():
    engine = _engine_from_spec()
    print(f"[loop] running 256 sims (16 sweeps x 16) on {EVAL_DEVICE}, engine={engine} ...", flush=True)
    env = {**os.environ, "PYTHONPATH": EP, "DICTY_DEVICE": EVAL_DEVICE, "DICTY_ENGINE": engine}
    subprocess.run([PYBIN, "eval_sweeps.py"], cwd=HERE, env=env, check=False)


def _figs():
    figs = sorted(glob.glob("sweep_*.png"))
    lst = "\n".join(f"  - {f}" for f in figs)
    return lst + "\n  - best_montage.png"


def start_prompt():
    return f"""DICTY START. You are the scientist in a hypothesis-driven model-discovery loop.
Goal: discover which mechanisms are NECESSARY and SUFFICIENT for dicty aggregation — KNOWLEDGE,
not a low score. Each batch you run 16 single-parameter sweeps x 16 values = 256 simulations.

Read (follow ALL of the instruction):
  instruction: {INSTR}
  knowledge ledger (read + update; THE deliverable): {LEDGER}
  full log (append-only): {LOG}
  user input (read + acknowledge pending): {USERIN}
  parent/control config: {BASE}

Set up the FIRST sweep batch by editing {PLAN} (schema: {{"base_spec":"{BASE}","sweeps":[{{"param":"<path>","values":[16 numbers]}}, ... up to 16]}}).
param paths: cell.n, dt, vmax, camp.diffusion, camp.decay, or "<op>.<param>" (sense.gain, inflow.rate,
spring.kadh, ...). You MAY also edit operator/engine code (dicty_ops.py, dicty_engine.py) to ADD a
mechanism (e.g. a cAMP relay-wave field) and then sweep its strength as one of the 16 sweeps.
Keep {BASE} as the current best/parent (control). Write the planned hypotheses to {LEDGER} first."""


def analysis_prompt(batch, n):
    return f"""DICTY BATCH {batch}/{n}. The 256 simulations (16 sweeps x 16) you designed have run.

Read (follow ALL of the instruction):
  instruction: {INSTR}
  knowledge ledger (read + UPDATE — the deliverable): {LEDGER}
  full log (append-only): {LOG}
  user input (read + acknowledge pending): {USERIN}
  sweep metrics: {RESULTS}

LOOK AT the figures (mandatory — morphology is primary evidence; Read each image). Each sweep figure
is a response curve (inner_mass & loss vs the swept value; REAL inner-mass=0.61 dashed) + a strip of
final SIM density per value with REAL at the end:
{_figs()}

Steps:
1. For each sweep: read its curve AND its morphology strip; write a log entry to {LOG} with the
   response shape, the best value, and a one-line MORPHOLOGICAL observation from the strip.
2. Update {LEDGER} (Established Principles / Falsified Hypotheses / Open Questions).
3. Set {BASE} to the best config found so far (the new parent/control).
4. Rewrite {PLAN} for the next batch: 16 sweeps, each varying ONE parameter around the new parent
   (refine ranges around optima, drop saturated params, add new ones). To test a MECHANISM, edit
   dicty_ops.py/dicty_engine.py and add a sweep over its strength + keep an ablation (strength=0).
Remember: a clean falsification is a success. The score adjudicates; the ledger is the result."""


def load_state():
    return json.load(open(STATE))["batch"] if os.path.exists(STATE) else 0


def save_state(b):
    json.dump({"batch": b}, open(STATE, "w"))


def main(n_batches, resume):
    start = load_state() if resume else 0
    if start == 0:
        print("[loop] FRESH — DICTY START (agent designs sweep batch 1)", flush=True)
        run_claude(start_prompt(), "DICTY START"); start = 1; save_state(1)
    for b in range(start, n_batches + 1):
        print(f"\n\033[94m[loop] BATCH {b}/{n_batches}\033[0m", flush=True)
        eval_sweeps()
        run_claude(analysis_prompt(b, n_batches), f"BATCH {b}")
        save_state(b + 1)
    print(f"[loop] DONE through batch {n_batches}. Ledger: {LEDGER}", flush=True)


if __name__ == "__main__":
    resume = "--resume" in sys.argv
    nb = next((int(a) for a in sys.argv[1:] if a.isdigit()), 20)
    main(nb, resume)
