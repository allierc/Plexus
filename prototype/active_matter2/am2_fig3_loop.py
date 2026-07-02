"""am2_fig3_loop.py -- agentic loop to reproduce FIG. 3 of Ziepke et al. (2022):
hierarchical self-organization + information processing in the HYDRODYNAMIC model.
Each slot runs am2_job.py --kind hydro --mode coarsen at one v0 (a long run), producing
a panel with the orientation cascade snapshots, the cluster-number Nc(t), the per-field
information content I(t) (PNG/LZW file-size proxy) and the processing rate R(t). The batch
montage places these beside paper_fig3.png.

Targets to match: (a) Nc(t) coarsening (~1/t early, faster once streams form), (b) the
droplet->stream->vortex cascade in the snapshots, (f) information content decaying as mass
condenses into few vortices, (d) processing rate R(t).

  python am2_fig3_loop.py 20            # cluster; RESUMES
  python am2_fig3_loop.py 2 --local
"""
import os, sys
import am2_cluster as C
import am2_montage as M

C.INSTR = "instruction_fig3.md"
C.LEDGER = "knowledge_fig3.md"
C.ANALYSIS = "analysis_fig3.md"
C.USERIN = "user_input.md"
C.PLAN = "am2_slots_fig3.md"
C.STATE = "am2_fig3_state.json"
C.TRANSCRIPT = "am2_fig3_transcript.md"
C.ARCH_PREFIX = "f3"
PAPER = "paper_fig3.png"


def design_prompt(batch, n):
    prev = batch - 1
    prev_mont = f"fig3_b{prev:02d}_montage.png"
    if os.path.exists(prev_mont):
        obs = (f"READ `{prev_mont}` (our coarsening panels + PAPER REFERENCE) and `archive/f3_b{prev:02d}_s*/` "
               f"(panel.png = orientation cascade + Nc(t) + info + R(t); coarsen.npz = the raw series; progress.txt "
               f"= Nc_final/Nc_max/R_final). Per slot: does Nc(t) show the multi-scale decay (early ~1/t then faster)? "
               f"do the snapshots show droplets -> streams -> few vortices? does information decay as mass condenses? "
               f"Biggest surprise vs the paper?")
    else:
        obs = (f"FIRST batch -- read the paper figure `{PAPER}` (a: Nc(t) for several v0; b: t=1200/5000/20000 "
               f"snapshots orientation+c; f: information per field). Run --mode coarsen at a few v0 (the paper uses "
               f"v0=0.2..0.7, snapshots at v0=0.5). Base = am2_hydro PRESETS['fig']. Longer nsteps -> more coarsening.")
    return f"""ACTIVE-MATTER2 / FIG.3 -- BATCH {batch}/{n}. You are a SCIENTIST reproducing the hierarchical
coarsening + information dynamics (Ziepke et al. 2022, Fig. 3) of the HYDRODYNAMIC model. Solver: am2_hydro.py,
--mode coarsen records Nc(t), per-field information I(t) (PNG-compressed size) and processing rate R(t). GOAL: match
the paper's PHENOMENA -- the cluster-number decay with multi-scale exponents, the droplet->stream->vortex cascade,
and the information content decreasing as mass gathers into few vortices. Understanding, not loss.

MEMORY (read every batch): method+schema {C.INSTR} ; ledger {C.LEDGER} (UPDATE) ; analysis {C.ANALYSIS} (append) ;
user input {C.USERIN} ; paper figure {PAPER}.

{obs}

Do ALL, auto-updating files:
1. Per slot: read Nc(t) shape (plateau -> ~1/t -> faster), the cascade snapshots, I(t) and R(t). Match to paper? surprise?
2. EDIT {C.ANALYSIS}: dated Batch {batch} section (per-slot v0/nsteps, Nc_final, Nc curve shape, cascade verdict).
3. DISTILL {C.LEDGER}: causal statements about what controls the coarsening rate + the exponent crossover +
   the information decay (which fields lose information first).
4. DESIGN <=8 slots into {C.PLAN}: `name : --kind hydro --mode coarsen --v0 <val> [--nsteps --N --omega --sigma
   --alpha --beta --eps --Dc --seed]`. Vary v0 to reproduce Fig.3a's family; longer nsteps to reach the vortex
   endgame; one variable per slot from parent. A slot with no panel FAILED.
You MAY edit am2_job.py's coarsen diagnostics or am2_hydro.py to measure/reach the paper's regime."""


def montage(batch, jobs):
    out = os.path.join(C.HERE, f"fig3_b{batch:02d}_montage.png")
    M.batch_montage(jobs, PAPER, out, title=f"Fig.3 coarsening + info -- batch {batch}", panel_h=360)


def main(n, fresh, local):
    C._preflight(local); os.makedirs(C.LOGDIR, exist_ok=True)
    if fresh:
        C.save_state(1)
    start = C.load_state() or 1
    for b in range(start, n + 1):
        print(f"\n\033[94m[fig3] BATCH {b}/{n}  ({'local' if local else C.QUEUE}) -- agent designing...\033[0m", flush=True)
        C.run_claude(design_prompt(b, n), f"FIG3 DESIGN {b}")
        jobs = C.parse_slots(b)
        if not jobs:
            print(f"[fig3] no slots; skipping batch {b}.", flush=True); C.save_state(b + 1); continue
        C.run_batch(jobs, local)
        montage(b, jobs)
        C.save_state(b + 1)
    print(f"[fig3] DONE through batch {n}.", flush=True)


if __name__ == "__main__":
    fresh = "--fresh" in sys.argv
    local = "--local" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(int(pos[0]) if pos else 20, fresh, local)
