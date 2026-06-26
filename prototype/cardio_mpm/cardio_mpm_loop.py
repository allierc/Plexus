"""cardio_mpm_loop.py -- agentic scientific loop (LoopScore objective), design-first.

Design-first flow (no JSON plan). At each iteration the agent (Claude CLI) does ONE step that:
  1. reads the previous batch's results (the dashboards / progress in archive/) -- begins from OBSERVATIONS,
  2. reads its MEMORY -- instruction (method/RULES) + knowledge_cardio_mpm.md (CUMULATIVE working memory) +
     analysis_cardio_mpm.md (human per-batch narrative log),
  3. writes/updates analysis (append) + knowledge (curate, never erase),
  4. designs the next <=6 slots into `cardio_mpm_slots.md` (one line per slot: `name : --flag val ...`).
Then the loop runs whatever slots the agent wrote. Objective = the LoopScore `LS` loop-morphology metric (the
trainer's --loss default, `harmonic`); rank on LS (mean). R² is a diagnostic only.

The cluster machinery lives in `cardio_mpm_cluster.py` (imported as L); this file is the design-first
DRIVER. New runs go to archive/p3_b* so the kept p2_b14 dirs stay as historical reference. Knowledge is
CUMULATIVE -- prior R²-objective conclusions are carried forward as provisional hypotheses, not discarded.

  cd prototype/cardio_mpm
  python cardio_mpm_loop.py 40                 # 40 Phase-3 batches; RESUMES saved state
  python cardio_mpm_loop.py 40 --fresh         # restart from batch 1
  python cardio_mpm_loop.py 3 --local          # local GPUs (testing)
"""
import os, sys, glob
import cardio_mpm_cluster as L                                         # reuse the proven cluster machinery

# --- Phase-3 file overrides (read by our main() below) ---
L.TRAIN = "cardio_mpm_train.py"
L.PLAN = "cardio_mpm_slots.md"                                         # the agent-written slots file (NOT json)
L.STATE = "cardio_mpm_loop_state.json"
L.TRANSCRIPT = "cardio_mpm_cli_transcript.md"
L.INSTR = "instruction_cardio_mpm.md"
ARCH_PREFIX = "p3"
SPEC = "material/material_aniso_cardio"


def _slots(batch):
    """Parse the agent-written `cardio_mpm_slots.md`: each non-comment line `name : --flag val ...`."""
    out = []
    try:
        lines = open(L.PLAN).read().splitlines()
    except OSError:
        lines = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, argstr = line.split(":", 1)
        name = name.strip().replace(" ", "_")
        toks = argstr.split()
        args, i = {}, 0
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
        out.append({"slot": s, "name": name, "spec": SPEC, "args": args, "train_script": L.TRAIN,
                    "arch": arch, "dir": os.path.join("archive", arch),
                    "log": os.path.join(L.LOGDIR, f"{arch}.out")})
        if len(out) >= 6:
            break
    return out


def design_prompt(batch, n_batches):
    """ONE agent step per iteration: read prev results + memory, update analysis/knowledge, design slots."""
    prev = batch - 1
    prev_dirs = sorted(glob.glob(os.path.join("archive", f"{ARCH_PREFIX}_b{prev:02d}_s*")))
    if prev_dirs:
        results = "The PREVIOUS batch finished -- read its results (the dashboard PNGs are the primary evidence):\n" + \
                  "\n".join(f"  - {d}/  (checkpoints/dashboard_*.png, progress.txt with LS/LS_SD/R2, config.json)"
                            for d in prev_dirs)
        step1 = ("1. READ the previous results above and BEGIN FROM OBSERVATIONS. Per slot: read its last "
                 "`checkpoints/dashboard_*.png` (zoom 3x3 shows per-node `LS=`; red sim vs green real) and its "
                 "final `LS`/`LS_SD`/`R2` from progress.txt or the job log `done -> (... LS=...)`. Does red superpose "
                 "on green? what is the SYSTEMATIC failure (which nodes wrong, how: size / chirality / axis / "
                 "overshoot)? what structure did the learned field converge to? RANK on LoopScore (mean).")
    else:
        results = ("This is the FIRST batch -- no previous results yet. Read knowledge_cardio_mpm.md (cumulative; "
                   "prior R²-objective conclusions are PROVISIONAL hypotheses under LoopScore -- do NOT discard them) "
                   "and design Batch 1 from the open frontier it identifies.")
        step1 = "1. (First batch -- no previous results; begin from the cumulative knowledge ledger.)"
    return f"""CARDIO-MPM -- BATCH {batch}/{n_batches}. You are a SCIENTIST discovering which physical mechanisms produce
the real cardiomyocyte trajectories -- NOT a hyperparameter optimizer. Training is your experimental instrument; this
batch is ONE experiment answering ONE question.

OBJECTIVE: maximize the LoopScore (LS) -- per-node loop-morphology (mean = objective, LS_SD = uniformity). It is the
trainer's DEFAULT `--loss` (omit the flag). R² is now a DIAGNOSTIC ONLY: a mechanism that improves LoopScore while
degrading R² is a SUCCESS. Rank slots by LS, then LS_SD, then morphology, then R².

AUTONOMY: there is NO predetermined roadmap of levers. Use the current knowledge to decide the next experiment.
Begin from OBSERVATIONS (the systematic failure), form ONE predictive hypothesis, then design the SMALLEST experiment
that distinguishes the live explanations. You may revisit ANY lever if evidence suggests an earlier conclusion was
optimization-limited or regime-bound. Keep stiffness/direction fields COARSE (low --siren_omega, larger --fibre_wl):
a too-short wavelength is inert -- coarsen before concluding a lever doesn't matter.

Your MEMORY (read EVERY batch):
  instruction (the RULES / method): {L.INSTR}
  knowledge ledger (CUMULATIVE working memory -- read + UPDATE, never erase; reclassify R² conclusions as provisional): {L.LEDGER}
  analysis log (human narrative; append a dated section): {L.ANALYSIS}
  user input (read + acknowledge if non-empty): {L.USERIN}

{results}

Do ALL of the following, in order, AUTO-UPDATING the files:
{step1}
2. EDIT {L.LEDGER}: append this batch's slot rows to the comparison table (sorted best-LS-first); update Established /
   Falsified / Open, each tied to slot(s) and tagged by CLASS ([engineering] stable / [mechanism] / [optimization@regime])
   AND regime (loss=LoopScore, @<N>it, dur/amp/parent). Never a bare FALSIFIED/CLOSED. Reclassify any prior R² conclusion
   you re-tested. A clean falsification OR a clean overturn is a success.
3. EDIT {L.ANALYSIS} (human log): append a dated Batch {batch} section (observation, hypothesis, slots, LS/LS_SD/R2, winner, verdict).
4. DESIGN this batch: write <=6 slots to `{L.PLAN}` (one line per slot: `name : --flag val ...`). spec is always
   {SPEC} (omit it); objective defaults to LoopScore (omit --loss). Each slot changes EXACTLY ONE variable from the
   parent (causal inference); include an ablation when it sharpens the inference. Keep amplitude in [10,15].
You MAY edit cardio_mpm_train.py to add a mechanism. A slot with done=NO / LS=na FAILED -- design around it."""


def main(n_batches, fresh, local):
    """Design-FIRST loop: agent designs each batch's slots, then we run them."""
    L._preflight(local); os.makedirs(L.LOGDIR, exist_ok=True)
    start = L.load_state() or 1
    for b in range(start, n_batches + 1):
        print(f"\n\033[94m[loop] BATCH {b}/{n_batches}  ({'local' if local else L.QUEUE})  -- agent designing slots...\033[0m", flush=True)
        L.run_claude(design_prompt(b, n_batches), f"DESIGN {b}")      # agent reads prev + memory, writes analysis/knowledge + slots
        jobs = _slots(b)
        if not jobs:
            print(f"[loop] no slots in {L.PLAN} -- agent wrote none; skipping batch {b}.", flush=True)
            L.save_state(b + 1); continue
        if local:
            L.run_local(jobs); ids = {j["slot"]: "local" for j in jobs}
        else:
            ids = L.submit_cluster(jobs)
            if ids:
                L.wait_cluster(ids, jobs)
            else:
                print("[loop] no jobs submitted -- aborting batch (check bsub/queue)", flush=True)
        L.check_completion(jobs, ids if not local else {j["slot"]: "local" for j in jobs})
        L.save_state(b + 1)
    print(f"[loop] DONE through batch {n_batches}. Ledger: {L.LEDGER}  Analysis: {L.ANALYSIS}", flush=True)


if __name__ == "__main__":
    fresh = "--fresh" in sys.argv
    local = "--local" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(pos[0]) if pos else 40
    if fresh:
        L.save_state(1)
    main(n, fresh, local)
