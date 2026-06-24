"""cardio_mpm_loop2.py -- Phase 2 agentic loop: PARAMETRIC inverse fit (cardio_mpm_train2.py).

Reuses ALL the cluster submit / poll / Claude-scientist machinery from cardio_mpm_loop.py, but
points at the Phase-2 files and prompts:

  plan      cardio_mpm_plan2.json        (train_script: cardio_mpm_train2.py, 15-param configs)
  instr     instruction_cardio_mpm_phase2.md
  ledger    knowledge_cardio_mpm.md      (SHARED -- the running deliverable)
  analysis  analysis_cardio_mpm.md       (SHARED -- append Phase-2 batch sections)
  state     cardio_mpm_loop2_state.json  (separate Phase-2 batch counter)
  archive   p2_b<NN>_s<k>_<name>/        (distinct prefix so Phase-1 dirs aren't clobbered)

Phase 2 RANKS ON INTERIOR R2 (real-trajectory match); morphology metrics are secondary validators.

  cd prototype/cardio_mpm
  python cardio_mpm_loop2.py 5                  # 5 Phase-2 batches; RESUMES saved state
  python cardio_mpm_loop2.py 5 --fresh          # restart from START (agent reviews seed plan2)
  python cardio_mpm_loop2.py 3 --local          # local GPUs (testing)
"""
import os, sys, json
import cardio_mpm_loop as L                                            # reuse the proven machinery

# --- Phase-2 file overrides (monkeypatched onto the loop module; main() reads these at runtime) ---
L.TRAIN = "cardio_mpm_train2.py"
L.PLAN = "cardio_mpm_plan2.json"
L.STATE = "cardio_mpm_loop2_state.json"
L.TRANSCRIPT = "cardio_mpm_cli2_transcript.md"
L.INSTR = "instruction_cardio_mpm_phase2.md"
# LEDGER + ANALYSIS stay shared (knowledge_cardio_mpm.md / analysis_cardio_mpm.md)
ARCH_PREFIX = "p2"


def _slots(batch):
    """Phase-2 slots: distinct archive prefix `p2_b<NN>_s<k>_<name>` (Phase-1 dirs untouched)."""
    plan = json.load(open(L.PLAN))
    train_script = plan.get("train_script", L.TRAIN)
    out = []
    for s, cfg in enumerate(plan["configs"][:6]):
        arch = f"{ARCH_PREFIX}_b{batch:02d}_s{s}_{cfg['name']}"
        out.append({"slot": s, "name": cfg["name"], "spec": cfg["spec"], "args": dict(cfg.get("args", {})),
                    "train_script": cfg.get("train_script", train_script),
                    "arch": arch, "dir": os.path.join("archive", arch),
                    "log": os.path.join(L.LOGDIR, f"{arch}.out")})
    return out


def start_prompt():
    return f"""CARDIO-MPM PHASE 2 START. You are the scientist in a hypothesis-driven PARAMETRIC inverse-fit
loop. Phase 1 (the SHAPE ATLAS) is DONE; the deliverable ledger records that the anisotropic active-stress
pattern family spans the real loop morphology, with fibre_wl~40 leading. Phase 2 INVERTS that family:
`cardio_mpm_train2.py` learns 12 PARAMETRIC PATTERN scalars (fibre wl/angle/amp/phase, gain wl/phase/lo/hi,
stiff wl/phase/lo/hi) + a learnable pulse duration, NOT free pixels. The outer band is Dirichlet-anchored to
the real beat; the interior is fit. AMPLITUDE + DRAG are FIXED per-slot knobs (the plan sweeps them; amplitude
is CONSTRAINED to 10-15). Goal = KNOWLEDGE about which parametric patterns make the learned (red) per-node
loops match the real (green) beat -- ranked by interior R2 (motion-normalised, boundary excluded; 1=perfect,
<=0=worse than no motion), with loop-morphology (openness/chirality/size) as a secondary validator.

Read (follow ALL of the instruction):
  instruction: {L.INSTR}
  knowledge ledger (read + keep updating; THE deliverable): {L.LEDGER}
  running analysis (append Phase-2 batch sections): {L.ANALYSIS}
  user input (read + acknowledge pending, if present): {L.USERIN}

A seed {L.PLAN} already exists (parent fibre_wl40 + fibre_angle/gain_wl sweep + an amplitude15 slot). Review
it against the ledger; if you would design batch 1 differently, REWRITE {L.PLAN} (<=6 configs, schema in the
instruction). Otherwise leave it. Do not launch anything."""


def analysis_prompt(batch, n, jobs):
    listing = "\n".join(f"  - slot {j['slot']} [{j['name']}] spec={j['spec']}: {j['dir']}/  "
                        f"(config.json, progress.txt, checkpoints/dashboard_*.png + model_*.pt)" for j in jobs)
    return f"""CARDIO-MPM PHASE 2 BATCH {batch}/{n}. The parallel cluster jobs you designed (cardio_mpm_train2.py,
PARAMETRIC active-stress inverse fit) have finished.

Read (follow ALL of the instruction):
  instruction: {L.INSTR}
  knowledge ledger (read + UPDATE -- the deliverable): {L.LEDGER}
  running analysis (append a dated Phase-2 batch section): {L.ANALYSIS}
  user input (read + acknowledge pending, if present): {L.USERIN}

The result directories (Read the images -- sim-red vs real-green trajectory match is the primary evidence):
{listing}

CRITICAL: You MUST automatically update {L.LEDGER} and {L.ANALYSIS} and rewrite {L.PLAN} (do not wait for user input).

Steps (do ALL, AUTO-UPDATE the files as you go):
1. Per slot: Read its last `checkpoints/dashboard_*.png` (panels: sim-red/real-green trajectories, learned
   stiffness, gain, fibre-angle, fibre dx/dy) and its final interior R2 (progress.txt / the job log
   `done -> (R2=...)`) + `config.json`. Note in {L.ANALYSIS}: does red superpose on green? what parametric
   STRUCTURE did the 12 params converge to (fibre wl/angle, gain lo/hi, stiff)? the R2 AND the morphology
   (openness/chirality/size). RANK on interior R2.
2. EDIT {L.LEDGER}: update Established Principles / Falsified Hypotheses / Open Questions for the PARAMETRIC
   inverse, each tied to the slot(s) that show it. A clean falsification is a success.
3. EDIT {L.ANALYSIS}: append a dated Phase-2 batch {batch} section: the configs, R2s, morphology, the winner,
   the reasoning.
4. EDIT {L.PLAN}: rewrite for the next batch (<=6 one-knob-from-parent configs incl. an ABLATION). Keep
   amplitude in [10,15]; prioritise fibre params, then gain, then stiffness. You MAY edit cardio_mpm_train2.py
   to add a mechanism (then sweep it + keep its ablation).
A slot with `done=NO` / R2=na FAILED -- say so and design around it, do not invent results."""


# rebind onto the loop module so main() (defined there) uses the Phase-2 versions at runtime
L._slots = _slots
L.start_prompt = start_prompt
L.analysis_prompt = analysis_prompt


if __name__ == "__main__":
    fresh = "--fresh" in sys.argv
    local = "--local" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_batches = int(pos[0]) if pos else 5
    L.main(n_batches, fresh, local)
