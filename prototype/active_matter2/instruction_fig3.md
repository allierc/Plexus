# instruction -- Fig. 3 loop (hydrodynamic coarsening + information)

Reproduce the PHENOMENA of Ziepke et al. Fig. 3 with am2_hydro --mode coarsen: (a) the
cluster-number Nc(t) decay (multi-scale: ~1/t early, faster once streams form), (b) the
droplet->stream->vortex cascade in orientation+c snapshots, (f) per-field information
content I(t) (PNG/LZW file-size proxy) decaying as mass condenses, (d) processing rate R(t).
Match `paper_fig3.png`. Understanding, not loss.

Slot schema (am2_slots_fig3.md, <=8 lines):
`name : --kind hydro --mode coarsen --v0 <val> [--nsteps --N --omega --sigma --alpha
 --beta --eps --Dc --seed]`. Vary v0 for Fig.3a's family; longer nsteps for the vortex
endgame. One variable per slot. A slot with no panel.png FAILED.

Success: a batch that reproduces the Nc(t) shape for a v0, the cascade, or the info decay.
