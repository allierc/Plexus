# instruction -- Fig. 2 loop (hydrodynamic phase diagram)

Map the HYDRODYNAMIC states across (v0, omega) and match `paper_fig2.png` (panels a-f
snapshots coloured by polarization orientation; panel g the phase diagram). No scalar
loss -- the deliverable is the (v0,omega)->state map + boundaries in knowledge_fig2.md.

Solver: am2_hydro.py (rho,p,s,c, Eqs 6-9). Base = PRESETS['fig'] (aggregating vortex
regime, c_th<0 excitable). OUR omega axis is rescaled vs the paper (aggregation near
omega~1, not 0.05) -- report the mapping, don't chase the paper's literal numbers.

Slot schema (am2_slots_fig2.md, <=8 lines):
`name : --kind hydro --v0 <val> --omega <val> [--sigma --alpha --beta --eps --Dc --chi
 --Q --delta --Drho --Dp --N --nsteps --seed] --mode snapshot`
One variable changed per slot from its parent. Cover missing/misplaced states + a
boundary probe. A slot with no panel.png FAILED.

Success: a batch that places a new state correctly OR sharpens a phase boundary.
