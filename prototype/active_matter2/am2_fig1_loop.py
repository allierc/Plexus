"""am2_fig1_loop.py -- agentic loop to reproduce FIG. 1 of Ziepke et al. (2022):
the six collective dynamic states of the AGENT-BASED communicating-active-matter model
(directed streams, ring streams, active droplets, vortices, polar bands, aggregation).

Philosophy (like cardio_mpm_loop.py): this is about UNDERSTANDING which mechanisms /
parameters produce which state, and getting the reproduced montage to AGREE with the
paper -- not minimizing a scalar loss. Each batch the Claude CLI acts as the scientist:
it Reads the paper figure (paper_fig1.png) + the previous batch montage, updates the
knowledge + analysis ledgers, and designs <=8 experiment slots (one variable changed
from a parent) into am2_slots_fig1.md. Each slot runs am2_job.py --kind agent on the
cluster (8 in parallel, L4). A montage of the batch vs the paper is then assembled.

  cd prototype/active_matter2
  python am2_fig1_loop.py 20             # 20 batches on the cluster; RESUMES
  python am2_fig1_loop.py 20 --fresh     # restart from batch 1
  python am2_fig1_loop.py 2 --local      # local GPUs (testing)
"""
import os, sys, glob
import am2_cluster as C
import am2_montage as M

C.INSTR = "instruction_fig1.md"
C.LEDGER = "knowledge_fig1.md"
C.ANALYSIS = "analysis_fig1.md"
C.USERIN = "user_input.md"
C.PLAN = "am2_slots_fig1.md"
C.STATE = "am2_fig1_state.json"
C.TRANSCRIPT = "am2_fig1_transcript.md"
C.ARCH_PREFIX = "f1"
PAPER = "paper_fig1.png"
STATES = "streams, ring-streams, active-droplets, vortices, polar-bands, aggregation"


def design_prompt(batch, n):
    prev = batch - 1
    prev_mont = f"fig1_b{prev:02d}_montage.png"
    if os.path.exists(prev_mont):
        obs = (f"READ the previous batch montage `{prev_mont}` (our reproduced panels + the PAPER REFERENCE "
               f"tile) and the individual slot dirs `archive/f1_b{prev:02d}_s*/` (panel.png = particles coloured "
               f"by orientation over the chemical field; movie_*.mp4; progress.txt = P/Nc/contrast/signal). "
               f"BEGIN FROM OBSERVATIONS: for each targeted state, does our panel MATCH the paper's corresponding "
               f"panel (same morphology of aggregates AND of the chemical field)? What is the SYSTEMATIC mismatch "
               f"(too gas-like? no closed rings? bands not coherent? chemical fragmented vs wave-like?).")
    else:
        obs = (f"FIRST batch -- no montage yet. Read the paper figure `{PAPER}` and knowledge ledger, and design "
               f"batch 1 to place each of the six states with the base operator set (see am2_job.py AGENT_DEFAULTS "
               f"and the specs in specs/ for known-good starting points).")
    return f"""ACTIVE-MATTER2 / FIG.1 -- BATCH {batch}/{n}. You are a SCIENTIST reproducing the six collective
dynamic states of communicating active matter (Ziepke et al. 2022, Fig. 1): {STATES}. The agent-based model
is glide + polar_align + chemotax + relay(excitable) + adapt + repel on a shared chemical field (am2_ops.py).
GOAL: make our reproduced montage AGREE with the paper figure `{PAPER}` -- a QUALITATIVE morphology match of BOTH
the particle orientation field (top) and the chemical field (bottom), per state. This is understanding, NOT loss
minimization: map parameter -> state, and explain WHY each knob moves the morphology.

Your MEMORY (read EVERY batch):
  instruction / method + slot schema: {C.INSTR}
  knowledge ledger (CUMULATIVE -- read + UPDATE, never erase): {C.LEDGER}
  analysis log (append a dated batch section): {C.ANALYSIS}
  user input (read + acknowledge if non-empty): {C.USERIN}
  paper figure to match: {PAPER}

{obs}

Do ALL of the following, in order, AUTO-UPDATING the files:
1. Per targeted state: judge the morphology match (paper vs ours) from the montage + panels. Name the biggest
   SURPRISE and the systematic mismatch. Which knob is the lever?
2. EDIT {C.ANALYSIS}: append a dated Batch {batch} section (per-slot: state, key params, P/Nc/contrast, match verdict, the lever).
3. DISTILL {C.LEDGER} (compact, causal): for each state record the parameter regime that produces it and the
   mechanism (e.g. "rings need strong Gamma + fast Dc + slow decay so closed wave fronts form"). Tag Established /
   Open / Rejected. Never erase; reclassify if overturned.
4. DESIGN <=8 slots into {C.PLAN}, one per line: `name : --kind agent --state <target> --<param> <val> ...`.
   Tunable params (defaults in am2_job.py AGENT_DEFAULTS): --n --move_speed --radius --res --frames --seed
   --omega(chemotaxis) --gamma(alignment) --align_noise --beta(emission) --sigma(source width) --eps(adaptation)
   --diffuse --decay --repel --r0 --marker(triangle|dot). Each slot changes ONE variable from its parent (causal
   inference). Cover the states that still MISMATCH; keep 1-2 slots as controls/ablations. A slot with no panel FAILED.
You MAY edit am2_ops.py / am2_job.py to add a mechanism if a state is unreachable with current knobs."""


def montage(batch, jobs):
    out = os.path.join(C.HERE, f"fig1_b{batch:02d}_montage.png")
    M.batch_montage(jobs, PAPER, out, title=f"Fig.1 reproduction -- batch {batch}")


def main(n, fresh, local):
    C._preflight(local); os.makedirs(C.LOGDIR, exist_ok=True)
    if fresh:
        C.save_state(1)
    start = C.load_state() or 1
    for b in range(start, n + 1):
        print(f"\n\033[94m[fig1] BATCH {b}/{n}  ({'local' if local else C.QUEUE}) -- agent designing slots...\033[0m", flush=True)
        C.run_claude(design_prompt(b, n), f"FIG1 DESIGN {b}")
        jobs = C.parse_slots(b)
        if not jobs:
            print(f"[fig1] no slots in {C.PLAN}; skipping batch {b}.", flush=True); C.save_state(b + 1); continue
        C.run_batch(jobs, local)
        montage(b, jobs)
        C.save_state(b + 1)
    print(f"[fig1] DONE through batch {n}. Ledger: {C.LEDGER}  Analysis: {C.ANALYSIS}", flush=True)


if __name__ == "__main__":
    fresh = "--fresh" in sys.argv
    local = "--local" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(int(pos[0]) if pos else 20, fresh, local)
