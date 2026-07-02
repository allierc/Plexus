"""am2_fig2_loop.py -- agentic loop to reproduce FIG. 2 of Ziepke et al. (2022):
the principal collective states of the HYDRODYNAMIC model across the (v0, omega) plane,
and the phase diagram. Each slot runs am2_job.py --kind hydro --mode snapshot at one
(v0, omega) point (or a coefficient variant); the batch montage places our orientation
snapshots beside paper_fig2.png. Understanding-first: map (v0, omega) -> state.

NOTE our nondimensionalization puts the aggregation threshold near omega~1 (the paper's
axis is ~0.05); the agent works in OUR units and reports the mapping.

  python am2_fig2_loop.py 20            # cluster; RESUMES
  python am2_fig2_loop.py 2 --local
"""
import os, sys
import am2_cluster as C
import am2_montage as M

C.INSTR = "instruction_fig2.md"
C.LEDGER = "knowledge_fig2.md"
C.ANALYSIS = "analysis_fig2.md"
C.USERIN = "user_input.md"
C.PLAN = "am2_slots_fig2.md"
C.STATE = "am2_fig2_state.json"
C.TRANSCRIPT = "am2_fig2_transcript.md"
C.ARCH_PREFIX = "f2"
PAPER = "paper_fig2.png"


def design_prompt(batch, n):
    prev = batch - 1
    prev_mont = f"fig2_b{prev:02d}_montage.png"
    if os.path.exists(prev_mont):
        obs = (f"READ `{prev_mont}` (our (v0,omega) snapshots + PAPER REFERENCE) and `archive/f2_b{prev:02d}_s*/` "
               f"(panel.png = polarization orientation over c; progress.txt = P/Nc/contrast/signal). Per slot: which "
               f"paper state (droplets/vortices/rings/silent-bands/streams/polar-bands) does it resemble, and what "
               f"(v0,omega) region does that put it in? What SURPRISED you (a state appearing where you didn't expect)?")
    else:
        obs = (f"FIRST batch -- read the paper figure `{PAPER}` (panels a-f at their (v0,omega); panel g the phase "
               f"diagram) and the ledger. Base preset is am2_hydro PRESETS['fig'] (aggregating vortex regime); sweep "
               f"v0 and omega around it. Known: strong omega + low v0 -> droplets/vortices; raise v0 -> rings; weak "
               f"omega -> streams; high v0 -> bands.")
    return f"""ACTIVE-MATTER2 / FIG.2 -- BATCH {batch}/{n}. You are a SCIENTIST mapping the HYDRODYNAMIC phase diagram
(Ziepke et al. 2022, Fig. 2) in the (v0=motility, omega=signal susceptibility) plane. Solver: am2_hydro.py integrates
rho, p, s, c (Eqs 6-9). GOAL: find the (v0,omega) points that reproduce EACH state (active droplets, vortices, rings,
silent polar bands, streams, polar bands with signalling) so a snapshot montage matches `{PAPER}`, AND chart the phase
boundaries. Understanding, not loss: explain WHY each region gives its state (motility vs signalling balance).

MEMORY (read every batch): method+schema {C.INSTR} ; ledger {C.LEDGER} (UPDATE) ; analysis {C.ANALYSIS} (append) ;
user input {C.USERIN} ; paper figure {PAPER}.

{obs}

Do ALL, auto-updating files:
1. Per slot: classify the state and its (v0,omega); judge match to the paper panel. Biggest surprise + the lever.
2. EDIT {C.ANALYSIS}: dated Batch {batch} section (per-slot v0/omega/P/Nc/contrast/signal, state, verdict).
3. DISTILL {C.LEDGER}: causal (v0,omega)->state boundaries + any coefficient (sigma/alpha/beta/eps/Dc) that shifts them.
4. DESIGN <=8 slots into {C.PLAN}: `name : --kind hydro --v0 <val> --omega <val> [--sigma --alpha --beta --eps --Dc
   --chi --Q --delta --Drho --Dp --N --nsteps --seed] --mode snapshot`. One variable changed per slot from its parent;
   cover the states still MISSING or MISPLACED; include a boundary-probe slot. A slot with no panel FAILED.
You MAY edit am2_hydro.py (PRESETS['fig'] or the integrator) to reach a state that is currently unreachable."""


def montage(batch, jobs):
    out = os.path.join(C.HERE, f"fig2_b{batch:02d}_montage.png")
    M.batch_montage(jobs, PAPER, out, title=f"Fig.2 (v0,omega) states -- batch {batch}")


def main(n, fresh, local):
    C._preflight(local); os.makedirs(C.LOGDIR, exist_ok=True)
    if fresh:
        C.save_state(1)
    start = C.load_state() or 1
    for b in range(start, n + 1):
        print(f"\n\033[94m[fig2] BATCH {b}/{n}  ({'local' if local else C.QUEUE}) -- agent designing...\033[0m", flush=True)
        C.run_claude(design_prompt(b, n), f"FIG2 DESIGN {b}")
        jobs = C.parse_slots(b)
        if not jobs:
            print(f"[fig2] no slots; skipping batch {b}.", flush=True); C.save_state(b + 1); continue
        C.run_batch(jobs, local)
        montage(b, jobs)
        C.save_state(b + 1)
    print(f"[fig2] DONE through batch {n}.", flush=True)


if __name__ == "__main__":
    fresh = "--fresh" in sys.argv
    local = "--local" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(int(pos[0]) if pos else 20, fresh, local)
