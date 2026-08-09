"""69-75: the spring-graph membrane at work, and the configurations that change its dynamics.

CHOSEN FROM WHAT THE SWEEPS RULED OUT. `k_bond` moves nothing across three orders (the 4x4 grid), and
the secretion threshold at 0.006 is already mapped, so neither is a useful axis for showing different
behaviour. What the sweeps pointed AT is remodelling: strain sits at growth_rate x tau regardless of
stiffness or supply, so `tau` is the parameter that decides whether the sheet ever approaches its break
threshold. These seven vary the things that actually move it.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "prototype", "Tyssue"),
          os.path.join(_ROOT, "discovery_okuda")):
    sys.path.insert(0, p)
import json

import aniso
import combine as C
import membrane_ops
import run_ecm as R
import tissue as TIS

GATE = os.path.join(_ROOT, "log", "okuda_ECM", "49_aniso_i0_fibres", "load.npz")

# k = 5000: inside the graph-mode ceiling of 8,220 measured at k = 200..40,000.
BASE = dict(membrane_particles=45000, membrane_cutoff=0.008, membrane_break=0.35,
            membrane_bond_k=5.0e3, membrane_adhesion=1.0e4, membrane_tau=60.0,
            membrane_jitter=0.35, membrane_reserve=12.5, membrane_secrete_rate=0.012,
            membrane_impl="graph", membrane_drag=40.0)

SERIES = {
    "69_graph_reference": dict(),
    # tau = 0 removes `basement_membrane_remodel` entirely: rest lengths are frozen, so stretch
    # ACCUMULATES instead of being forgotten. This is the one configuration in which the 0.35 break
    # threshold is reachable at all -- every run so far sat at 0.08 because remodelling absorbed it.
    "70_no_remodelling": dict(membrane_tau=0.0),
    "71_fast_remodelling": dict(membrane_tau=10.0),
    "72_starved": dict(membrane_secrete_rate=0.002),
    # the measured null: without an anchor the sheet SLIDES over the epithelium and never stretches,
    # mean bond strain 0.0000 at every stiffness. Kept in the series because it is the control that
    # says the strain in the others is real.
    "73_no_adhesion": dict(membrane_adhesion=0.0),
    "74_brittle": dict(membrane_break=0.08),
    "75_on_ovoid": dict(_gated=True),

    # --- 76 onward: THE FIXED SEED ------------------------------------------------------------------
    # Everything above 75 was deleted. The sheet those runs started from was not packed: the relaxation
    # ran over the whole 45,000-particle reservoir and the sheet then kept every 13.5th of them, and a
    # SUBSET of a blue-noise set is not blue noise -- thinning randomises it back to Poisson. Measured on
    # the seeded sheet, nearest-neighbour distance was 0.505 of a hexagonal packing with cv 0.411,
    # against 0.461 / 0.535 for points thrown at random. So it began 8% better than random, which is why
    # holes were visible in the first frames and grew from there. Relaxing the laid-down count directly
    # gives 0.885 with cv 0.046.
    #
    # 76 is the new nominal: same configuration as 69, on a sheet that starts packed.
    "76_reference_fixed_seed": dict(),

    # --- 77: THE BIOLOGICAL VERSION -----------------------------------------------------------------
    # Two rules replaced, both because the versions they replace are not things a cell could do.
    #
    # DEPOSITION. Cells secrete basement membrane basally, into the patch of surface directly beneath
    # themselves -- laminin polymerises where integrin and dystroglycan nucleate it, collagen IV
    # assembles onto that. So deposition is local to each cell and UNIFORM PER UNIT BASAL AREA. Placing
    # new material beside an existing node is a random walk and clumps; placing it in the largest gap
    # packs better but is worse biology, because it needs a global view of where the holes are and a cell
    # has none. Uniform fills a hole at the same rate it adds anywhere else, which is the honest
    # mechanism, and leaves the evening-out to the network.
    #
    # ANCHORS. `u0` was frozen at seeding, which says a patch of membrane remembers where a cell was
    # four hundred frames ago. Focal adhesions turn over in seconds to minutes, and the cells themselves
    # divide and swap neighbours, so the tissue a patch is attached to MOVES. Frozen anchors are also
    # what pins the holes: material that relaxes into a gap is dragged back out by its own tether.
    "77_biological": dict(membrane_deposit="uniform", membrane_tau_adh=40.0),
    # the two controls that say which of the two changes did the work
    "78_uniform_only": dict(membrane_deposit="uniform", membrane_tau_adh=0.0),
    "79_turnover_only": dict(membrane_deposit="parent", membrane_tau_adh=40.0),

    # --- 80: WITH ONGOING CROSSLINKING --------------------------------------------------------------
    # The 2x2 of 76-79 settled two things and left the real defect untouched. Anchor turnover does
    # essentially nothing (d/hex moves 0.013 and 0.001, inside run-to-run scatter), so frozen anchors
    # were NOT what stopped relaxation working. Deposition is a straight trade: parent packs well
    # (0.79) with large gaps (1.64), uniform has the smallest gaps (1.41) and packing that reverts to
    # random (0.47).
    #
    # Neither is the problem. The bond list was built once and extended only for newly secreted nodes,
    # so the topology was frozen: 29% of pairs within the cutoff had no bond, and there were more bonds
    # than close pairs -- a third of the network held together by history. The black patches are
    # plausibly regions with nodes and no crosslinks, which renders identically to an empty patch and
    # which no amount of relaxation could repair, because relaxation moves nodes and the defect is in
    # the edges. `basement_membrane_crosslink` (rewire) is the missing half of fragmentation.
    "80_crosslinking": dict(membrane_deposit="uniform", membrane_rebond_every=20),

    # --- 81, 82: WHY THE NETWORK DOES NOT EVEN OUT --------------------------------------------------
    # Cedric: a spring network, overdamped, should reach equilibrium -- so why are the nodes not
    # equidistant? It has reached equilibrium. Measured on 80, the cv of the REST LENGTHS is 0.521:
    # every spring wants a different length, so a disordered configuration is exactly what the network
    # relaxes to. That is why anchors, crosslinking and deposition all did nothing -- none of them
    # touches where the disorder is stored.
    #
    # 81 is his test: remove the anchors entirely, so the sheet is a pure spring network with nothing
    # holding it to the surface. It should slide (73 lost coverage doing this), but it isolates whether
    # the tethers constrain the packing at all.
    # 82 is what the rest-length measurement points to: relax rest lengths toward the spacing the
    # sheet's own density implies, instead of toward each bond's own length. A real collagen IV network
    # has a characteristic mesh size; giving the springs a COMMON target is what lets a relaxed network
    # be a uniform one.
    "81_no_anchor": dict(membrane_deposit="uniform", membrane_adhesion=0.0),
    "82_mesh_restlength": dict(membrane_deposit="uniform", membrane_remodel_target="mesh"),

    # 83-85: the holes, taken to the bottom. 82 helped (cv of the rest lengths 0.527 -> 0.191) but only
    # halfway, and the relaxation bench said why: fitting cv = A*exp(-t/T) + C over 6000 iterations gives
    # C = 0.18-0.22, a PLATEAU, not a slow decay. About 85% of the disorder never relaxes at all.
    #
    # Two changes, and neither is a tuning knob.
    #
    # `fixed` holds l* at the frame-0 spacing instead of tracking the sheet's own mean. `mesh` is
    # self-referential: if secretion under-delivers, the mean grows and the springs bless the sparser
    # sheet. Fixed, N = 4*pi*R^2/l*^2 is the node count the sheet NEEDS, secretion is the only thing that
    # can supply it, and a shortfall shows up as tension rather than being absorbed into the target.
    #
    # `repel_w` adds excluded volume beside the springs. The bench isolated the cause: it is not a
    # missing repulsion (a spring pushes when compressed) but an unopposed attraction -- the rewire
    # throws crosslinks clear across the holes and those long bonds haul the rim into knots. Deleting
    # the pull fixes it (cv 0.173 -> 0.049) but leaves a sheet that bears no tension, so the pull stays
    # and repulsion is added: spacing from one, load from the other.
    #
    # THREE RUNS BECAUSE THE BENCH BRACKETS w RATHER THAN FIXING IT, and the bracket is stiffness
    # dependent. Swept at k = 2e5 the answer was w = 8 (cv 0.020) with w = 20 unstable (0.179); swept at
    # 100x softer springs it was the reverse, w = 20 the better of the two. THIS SERIES RUNS k = 5e3, so
    # h*z*k*(1+w)/gamma = 0.06 / 0.51 / 1.20 for w = 0 / 8 / 20 -- all under the limit of 2, and none of
    # them is the case the bench calibrated. So the bracket is run, not predicted.
    #
    # 83 separates the target change from the repulsion: if `fixed` alone closes the holes, the excluded
    # volume is not needed and 84/85 should not be kept. (The bond range needs no change here: cutoff =
    # 0.008 against l* = 0.00575 is 1.4 spacings already, not the 3.7 the spec default would have given.)
    "83_fixed_target": dict(membrane_deposit="uniform", membrane_remodel_target="fixed"),
    "84_fixed_repel8": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                            membrane_repel_w=8.0),
    "85_fixed_repel20": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                             membrane_repel_w=20.0),

    # 86-88. What 83-85 settled, measured at the middle frame:
    #
    #   82 mesh target      d/hex 0.543   cv 0.442
    #   83 fixed target     d/hex 0.540   cv 0.443     <- a NULL. Fixing l* alone changes nothing.
    #   84 w = 8            d/hex 0.604   cv 0.339
    #   85 w = 20           d/hex 0.677   cv 0.243     <- monotone, no plateau: the optimum is above 20
    #
    # And the supply question is closed: N_live/N_needed at l* is 1.14 at EVERY frame from R = 0.088 to
    # 0.298, so secretion tracks growth exactly and the sheet always carries a 14% surplus of nodes. The
    # holes are not a material shortage, they are enough nodes badly arranged -- which is precisely the
    # case repulsion addresses, so the lever is right and only too weak.
    #
    # 86/87 push w, since 85 was still climbing. Frame-level h*z*k*(1+w)/gamma is 2.9 at w = 50 and 6.9
    # at w = 120, both past the overdamped limit of 2 -- but the membrane is on the MPM path, integrated
    # at the substep, which is why w = 20 improved instead of diverging. So these two also LOCATE THE
    # CEILING: if 87 explodes, the substep headroom is smaller than the MPM path implies.
    #
    # 88 is the other error 83-85 exposed. The sheet's actual mean spacing is 0.94 l*, BELOW the
    # repulsion range, so nearly every pair is inside it and the operator acts as a uniform outward
    # pressure against the anchors rather than selectively separating the over-close pairs. Setting the
    # range to the density-implied spacing restores the selectivity at a stiffness already shown stable.
    "86_repel50": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                       membrane_repel_w=50.0),
    "87_repel120": dict(membrane_deposit="uniform", membrane_remodel_target="fixed",
                        membrane_repel_w=120.0),
    # ====================================================================================
    # THE CONTINUUM LADDER, from 88. One step per folder, one change per step.
    #
    # WHY THE SPRING LINE STOPPED. Runs 82-87 and the deleted 88-97 chased the holes with crosslink
    # springs, excluded volume and the attraction_repulsion law. The best of them (86) reached d/hex
    # 0.733 -- and across all 16, corr(d/hex, mean_degree_z) = -0.68: EVERY mechanism that improved the
    # packing did so by breaking crosslinks, with no counterexample. Best packing at z >= 6 was 0.581;
    # to beat it the network had to be dismantled.
    #
    # That is a defect of the OBJECT, not of the tuning. Holes and coordination are properties of a bond
    # network. An MPM continuum has none: particles are quadrature points, the response comes from the
    # deformation gradient through the grid, and a gap between particles is not a gap in the material.
    # So d/hex, gap, lcc and z stop being meaningful rather than becoming good.
    #
    # THE ONE RISK, AND IT IS REAL. The sheet is 0.002 thick and the grid cell is 1/64 = 0.0156, so the
    # BM is 1/8 of a cell through-thickness and the grid smears it over 8x its thickness. In-plane it is
    # well resolved (spacing 0.0015 at reserve 0, ~10 particles per cell edge); through-thickness it is
    # not resolved at all. Step 89 refines the grid precisely to find out how much of step 88 is that.
    #
    # 88  elastic continuum, NOTHING else: no bonds, no anchor, no secretion, no remodelling.
    #     Does an MPM shell survive 402 frames and get pushed outward by the growing epithelium at all?
    #     `membrane_impl="mpm"` matters: BASE runs "graph", which STRIPS mpm_strain/scatter/gather from
    #     the membrane. With springs off as well that would leave a set with no mechanics whatsoever.
    #     reserve=0 lays all 45,000 particles down at frame 0, since there is no secretion to release them.
    "88_mpm_bare": dict(membrane_springs=False, membrane_impl="mpm", membrane_adhesion=0.0,
                        membrane_secrete_rate=0.0, membrane_tau=0.0, membrane_reserve=0.0),
    # 89: 88 said the sheet GROWS with the spheroid (R 0.0875 -> 0.2985, coverage 1.0) but carries no
    #     strain -- sigma_max(F) - 1 reaches 7e-4 against a true stretch of 3.4x. The log says why:
    #     18,134 particles per frame are PROJECTED out of the tissue by cell_exclude_3d, a positional
    #     constraint that repositions without touching F. The sheet is a decal. A body at zero strain
    #     cannot tear, cannot resist growth and cannot load the stroma.
    #
    #     The membrane receives no force at all: cell_to_ecm is an analytic growing sphere bound to
    #     mpm_particle and its geometry is not the vertex tissue's. integrin_adhesion is the only
    #     operator that couples the membrane to the REAL surface with a force -- and a force is what
    #     becomes grid momentum, which is what makes C non-zero and F integrate. So the anchor turns
    #     out to be the prerequisite for milestone 1, not a milestone 3 refinement.
    #     Cedric's call, and it is the better one: inject the growth into the MPM GRID rather than
    #     anchoring particles. `mpm_grid_update` already zeroes grid velocity inside obstacles -- right
    #     for a wall, wrong for a growing tissue, which is an obstacle with a velocity. Setting the grid
    #     velocity inside the surface to the surface's own Rdot makes growth arrive as momentum, so C is
    #     non-zero and F integrates the stretch the projection was discarding. Standard collision-object
    #     treatment, and it reads the same `smap` the membrane is seeded on, so boundary and sheet cannot
    #     disagree about where the surface is. NO ANCHOR: this tests the grid route alone.
    "89_mpm_gridbc": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                          membrane_adhesion=0.0, membrane_secrete_rate=0.0, membrane_tau=0.0,
                          membrane_reserve=0.0),
    # 90: 89 changed NOTHING -- strain 3.8e-4, p99 5.9e-4, identical to 88 to every digit. Because the
    #     grid BC was only half the fix: `cell_exclude_3d` still ran on the membrane and is applied
    #     AFTER the substep, so it rewrote the positions the grid had just moved and laundered the
    #     deformation straight back out of F. A hard projection always beats a body force that ran
    #     before it. 90 removes it, which is the actual test of the grid route.
    "90_gridbc_noexcl": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                             membrane_exclude=False, membrane_adhesion=0.0,
                             membrane_secrete_rate=0.0, membrane_tau=0.0, membrane_reserve=0.0),
    # 91: 90 ended at R = 0.0875 -- the membrane did not move at all in 402 frames, so the projection
    #     had been the ONLY thing carrying it outward. The boundary condition was imposed on nodes
    #     INSIDE the surface, and THE LUMEN IS EMPTY: no MPM particles live there, so those nodes carry
    #     no mass, `mv = v*m` writes nothing and G2P gathers nothing back. The constraint was applied
    #     exactly where there is no material to apply it to.
    #
    #     91 imposes it where the material actually is: on MASSED nodes in a band around the surface,
    #     correcting only the outward normal component and only when material is moving slower than the
    #     surface. That is a separating collision object rather than no-slip -- the tissue pushes
    #     material out of its way, and never pulls it back or drags it tangentially. Welding the sheet
    #     to the epithelium is what the integrin anchor is for, and is a different experiment.
    "91_gridbc_band": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                           membrane_exclude=False, membrane_adhesion=0.0, membrane_secrete_rate=0.0,
                           membrane_tau=0.0, membrane_reserve=0.0),

    # ---- M1 PASSES. The bug in 89-91 was one line and not physics: `mpm_grid_update` writes the solved
    # velocity into `g.v` and `mpm_gather` reads `g.v`, while the boundary operator was editing `g.mv`.
    # It ran every substep (140 calls in 6 frames, counted) and was invisible, which reads exactly like a
    # physical null -- so it was misdiagnosed three times, as the empty lumen and then as the band.
    #
    # With `g.v` written, the continuum registers the stretch the geometry implies:
    #     frame   100     200     300     402
    #     R/R0-1  0.481   1.019   1.708   2.600
    #     strain  0.494   1.029   1.682   2.365
    # and coverage falls to 0.934 -- the sheet begins to FAIL with no material being added, which is the
    # tear the ladder was built to produce, arriving without being asked for.
    #
    # ---- M2: does secreted material stop it? -----------------------------------------------------
    # 92 turns secretion on (reserve 12.5, nominal rate): 3,333 particles laid down at frame 0 and the
    #    rest released as the surface grows. If coverage holds at ~1 while 91 falls to 0.93, added
    #    material is what keeps a basement membrane intact under a tripling radius.
    # 93 starves it (rate/6) -- the control that says the effect is supply and not the reserve merely
    #    existing. Expect a WORSE tear than 91, since it starts with a sparser sheet.
    "92_mpm_secrete": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                           membrane_exclude=False, membrane_adhesion=0.0, membrane_tau=0.0,
                           membrane_reserve=12.5, membrane_secrete_rate=0.012),
    # 93 is now the CONTROL, not a starved variant: the same sparse seeded sheet with secretion off, so
    # 92 minus 93 is the effect of added material and nothing else. (A rate sweep is worthless until the
    # rate is read at all -- 92 vs 93 at 0.012 and 0.002 returned identical numbers because the operator
    # returned early on the missing bond list.)
    "_probe_occ_centre": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                              membrane_exclude=False, membrane_adhesion=0.0, membrane_tau=0.0,
                              membrane_reserve=12.5, membrane_secrete_rate=0.0, membrane_park=(0.5,0.5,0.5)),
    "93_mpm_nosecrete": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                             membrane_exclude=False, membrane_adhesion=0.0, membrane_tau=0.0,
                             membrane_reserve=12.5, membrane_secrete_rate=0.0),
    # 94: 92 and 93 both froze at R = 0.0876 while 91 tracked to 0.3151, and the difference is DENSITY,
    #     not secretion: `reserve=12.5` lays down only 3,333 of the 45,000 particles, and a sheet that
    #     sparse does not register as material on the grid, so the boundary has nothing to push. Two
    #     things were changed from the working point at once and the first one broke it.
    #
    #     94 changes exactly ONE thing from 91: the reserve is added ON TOP of the working sheet rather
    #     than carved out of it -- 90,000 particles at reserve 1.0 is the same 45,000 laid down at frame
    #     0, with 45,000 held back to secrete. 91 is its own control.
    # 94/95: THE GAP between sheet and epithelium. The boundary corrects every massed node within
    #     `band` grid cells of the surface, so material is pushed out to about R + band*dx. At band 2.0
    #     that is R + 0.042 against an intended standoff of membrane_offset = 0.004 -- ten times too far,
    #     and the visible space in the movies is that width rather than physics. The band cannot go to
    #     zero: it has to be at least the B-spline stencil's reach or the constraint cannot see the
    #     material it is meant to push. 1.0 and 0.5 bracket that limit.
    "94_band1": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                     membrane_exclude=False, membrane_adhesion=0.0, membrane_secrete_rate=0.0,
                     membrane_tau=0.0, membrane_reserve=0.0, membrane_band=1.0),
    "95_band05": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                      membrane_exclude=False, membrane_adhesion=0.0, membrane_secrete_rate=0.0,
                      membrane_tau=0.0, membrane_reserve=0.0, membrane_band=0.5),
    # 96: band 1.0 with the surface read by BILINEAR interpolation instead of nearest bin -- the
    #     staircase fix. 94 is its control (same run, nearest bin).
    "96_bilinear": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                        membrane_exclude=False, membrane_adhesion=0.0, membrane_secrete_rate=0.0,
                        membrane_tau=0.0, membrane_reserve=0.0, membrane_band=1.0),
    # 97: non-penetration. 96 showed the bilinear read changes nothing (47.9% inside vs 94's 46.6%), so
    #     the staircase was not the cause -- the constraint simply never pushed back material the surface
    #     had already overtaken.  clears an overlap over 2 frames.
    "97_recover": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                       membrane_exclude=False, membrane_adhesion=0.0, membrane_secrete_rate=0.0,
                       membrane_tau=0.0, membrane_reserve=0.0, membrane_band=1.0, membrane_recover=2.0),
    # 98: 97 fixed the sinking (46.6% inside -> 3.8%) but opened a standoff visible from frame 0,
    #     because  was a BALL, not a shell: every node inside the tissue qualified, and with the
    #     penetration term a node near the centre carries pen = R and an enormous outward velocity.
    "98_shell": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                     membrane_exclude=False, membrane_adhesion=0.0, membrane_secrete_rate=0.0,
                     membrane_tau=0.0, membrane_reserve=0.0, membrane_band=1.0, membrane_recover=2.0),
    # M3: is the integrin anchor still needed now that the grid boundary carries the sheet? 91 is the
    #     control (no anchor). 99 tethers it, 100 tethers it stiffly -- a stiff tether on a sheet the
    #     boundary is already moving is a way to tear it, not to hold it.
    "99_anchor": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                      membrane_exclude=False, membrane_adhesion=1.0e4, membrane_secrete_rate=0.0,
                      membrane_tau=0.0, membrane_reserve=0.0, membrane_band=1.0, membrane_recover=2.0),
    "100_anchor_stiff": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                             membrane_exclude=False, membrane_adhesion=5.0e4, membrane_secrete_rate=0.0,
                             membrane_tau=0.0, membrane_reserve=0.0, membrane_band=1.0,
                             membrane_recover=2.0),
    # THE GAP, against what biology asks for. A basement membrane sits ON the basal plasma membrane --
    # integrin a6b4 and dystroglycan bind laminin directly, and the lamina lucida of classical TEM is
    # read today as a fixation artefact. So the standoff should be zero to within the sheet thickness
    # (0.002 here), and +0.0123 is ~6x that: numerical, not physiological.  clears an overlap
    # over N frames, so a LARGER N is gentler and should overshoot less.
    "108_recover6": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                         membrane_exclude=False, membrane_adhesion=0.0, membrane_secrete_rate=0.0,
                         membrane_tau=0.0, membrane_reserve=0.0, membrane_band=1.0, membrane_recover=6.0),
    "109_recover20": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                          membrane_exclude=False, membrane_adhesion=0.0, membrane_secrete_rate=0.0,
                          membrane_tau=0.0, membrane_reserve=0.0, membrane_band=1.0, membrane_recover=20.0),
    # 110/111: THE HACK REMOVED. The grid boundary condition is off; non-penetration is a per-particle
    #     PENALTY FORCE against the surface instead. The standoff should then emerge from stiffness
    #     rather than from , and a real penalty contact has standoff proportional to 1/k --
    #     which is what 111 checks. If it does not scale, the force is just another dialled number.
    "110_contact": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=False,
                        membrane_contact_k=1.0e4, membrane_exclude=False, membrane_adhesion=0.0,
                        membrane_secrete_rate=0.0, membrane_tau=0.0, membrane_reserve=0.0),
    "111_contact_x5": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=False,
                           membrane_contact_k=5.0e4, membrane_exclude=False, membrane_adhesion=0.0,
                           membrane_secrete_rate=0.0, membrane_tau=0.0, membrane_reserve=0.0),
    # 112/113: 110 and 111 showed a one-sided contact CANNOT set a standoff -- with nothing pulling back
    #     there is no equilibrium, so the sheet is pushed off the surface (+0.1238) and, having left it,
    #     stops being stretched at all (strain 0.05 against 2.37). Biology already said this: the sheet
    #     is HELD by integrin a6b4 and dystroglycan binding laminin, not merely prevented from entering.
    #     112 adds that adhesion to the contact; 113 is adhesion alone, the control that says which of
    #     the two does the holding.
    "112_contact_adhesion": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=False,
                                 membrane_contact_k=1.0e4, membrane_exclude=False,
                                 membrane_adhesion=1.0e4, membrane_secrete_rate=0.0,
                                 membrane_tau=0.0, membrane_reserve=0.0),
    "113_adhesion_only": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=False,
                              membrane_contact_k=0.0, membrane_exclude=False,
                              membrane_adhesion=1.0e4, membrane_secrete_rate=0.0,
                              membrane_tau=0.0, membrane_reserve=0.0),
    # 114/115: PREDICTED, not swept. With critical damping the contact behaves as a penalty contact
    #     should -- standoff scaled 5.4x for a 5x stiffness (110 -0.0421, 111 -0.0078), which is the
    #     1/k law and the check that it is a force rather than a knob. Fitting standoff = -C/k gives
    #     C = 421, so k = 4e5 should sit within one sheet thickness (0.002) of the surface, and 2e6
    #     should be indistinguishable from it. If 115 differs from 114 by more than that, the load is
    #     not what the fit assumed.
    "114_contact_k4e5": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=False,
                             membrane_contact_k=4.0e5, membrane_exclude=False, membrane_adhesion=0.0,
                             membrane_secrete_rate=0.0, membrane_tau=0.0, membrane_reserve=0.0),
    "115_contact_k2e6": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=False,
                             membrane_contact_k=2.0e6, membrane_exclude=False, membrane_adhesion=0.0,
                             membrane_secrete_rate=0.0, membrane_tau=0.0, membrane_reserve=0.0),
    # 116-118: THE HYBRID. A continuum sheet (which homogenises perfectly -- coverage 1.000, no holes)
    #     held by PUNCTATE adhesion (which is what made the spring network track the surface), because
    #     hemidesmosomes are discrete plaques with membrane spanning between them. The contact drops to
    #     5e4 so it only stops the sheet entering the cells rather than dictating every particle -- that
    #     dictation is what made the sheet wear the surface map's texture, 5.2x more variation between
    #     map bins than within one. 118 anchors everything, as the control.
    "116_punctate5": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=False,
                          membrane_contact_k=5.0e4, membrane_exclude=False, membrane_adhesion=1.0e4,
                          membrane_adhesion_fraction=0.05, membrane_secrete_rate=0.0,
                          membrane_tau=0.0, membrane_reserve=0.0),
    "117_punctate20": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=False,
                           membrane_contact_k=5.0e4, membrane_exclude=False, membrane_adhesion=1.0e4,
                           membrane_adhesion_fraction=0.20, membrane_secrete_rate=0.0,
                           membrane_tau=0.0, membrane_reserve=0.0),
    "118_punctate100": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=False,
                            membrane_contact_k=5.0e4, membrane_exclude=False, membrane_adhesion=1.0e4,
                            membrane_adhesion_fraction=1.0, membrane_secrete_rate=0.0,
                            membrane_tau=0.0, membrane_reserve=0.0),
    # 119: hemidesmosomes as a SET. 2,000 discrete plaques on the basal surface, each bound to one
    #     membrane patch, pulling OVERDAMPED (F/gamma, no dashpot -- the dashpot in integrin_adhesion is
    #     an inertial fix for a mass the sheet should not have at Re ~ 1e-10). The contact stays weak so
    #     it only stops the sheet entering the cells; the sheet spans between anchors under its own
    #     elasticity, which is what should stop it wearing the surface map's texture.
    "119_adhesion_set": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=False,
                             membrane_contact_k=5.0e4, membrane_exclude=False, membrane_adhesion=0.0,
                             n_adhesions=2000, adhesion_k=1.0e4, adhesion_gamma=1.0,
                             membrane_secrete_rate=0.0, membrane_tau=0.0, membrane_reserve=0.0),
    # 120-123: THE HYBRID, tuned. MPM keeps the sheet's material response (a continuum has no bond
    #     network, so no holes -- coverage 1.000); adhesion and contact act DIRECTLY on the particle,
    #     engine-integrated and overdamped, as they do in graph mode. Run 70 is the evidence that this
    #     works at THIS grid and THIS sheet thickness, which is what refutes the resolution story.
    #     Overdamped, k and gamma enter only as the RATIO k/gamma -- a relaxation rate -- so that is the
    #     axis swept: 5, 25, 125. 123 adds the contact on top of the best-guess adhesion.
    "120_direct_r5": dict(membrane_springs=False, membrane_impl="mpm", membrane_direct_forces=True,
                          membrane_grid_bc=False, membrane_contact_k=0.0, membrane_exclude=False,
                          membrane_adhesion=1.0e4, membrane_gamma=2.0e3,
                          membrane_secrete_rate=0.0, membrane_tau=0.0, membrane_reserve=0.0),
    "121_direct_r25": dict(membrane_springs=False, membrane_impl="mpm", membrane_direct_forces=True,
                           membrane_grid_bc=False, membrane_contact_k=0.0, membrane_exclude=False,
                           membrane_adhesion=5.0e4, membrane_gamma=2.0e3,
                           membrane_secrete_rate=0.0, membrane_tau=0.0, membrane_reserve=0.0),
    "122_direct_r125": dict(membrane_springs=False, membrane_impl="mpm", membrane_direct_forces=True,
                            membrane_grid_bc=False, membrane_contact_k=0.0, membrane_exclude=False,
                            membrane_adhesion=2.5e5, membrane_gamma=2.0e3,
                            membrane_secrete_rate=0.0, membrane_tau=0.0, membrane_reserve=0.0),
    "123_direct_r25_contact": dict(membrane_springs=False, membrane_impl="mpm",
                                   membrane_direct_forces=True, membrane_grid_bc=False,
                                   membrane_contact_k=5.0e4, membrane_exclude=False,
                                   membrane_adhesion=5.0e4, membrane_gamma=2.0e3,
                                   membrane_secrete_rate=0.0, membrane_tau=0.0, membrane_reserve=0.0),
    # 124-128: TOWARD ADHESION BIOLOGY, each one change from 121 (k/gamma = 25, the stable point:
    #     coverage 1.000, strain 2.25, standoff -0.0082). 121 is biologically a continuous glue that
    #     tethers all 45,000 particles to positions frozen at frame 0, on a sheet that is never rebuilt.
    #     These five relax that, one property at a time.
    #     NOTE the known flaw they inherit: MPM gather and the engine delta both write particle
    #     positions in the direct-force path. It is benign at k/gamma = 25 and destroyed 122 at 125, so
    #     none of these raises the ratio.
    "124_punctate20": dict(membrane_springs=False, membrane_impl="mpm", membrane_direct_forces=True,
                     membrane_grid_bc=False, membrane_contact_k=0.0, membrane_exclude=False,
                     membrane_adhesion=5.0e4, membrane_gamma=2.0e3, membrane_tau=0.0, membrane_adhesion_fraction=0.20,
                          membrane_secrete_rate=0.0, membrane_reserve=0.0),
    "125_punctate5": dict(membrane_springs=False, membrane_impl="mpm", membrane_direct_forces=True,
                     membrane_grid_bc=False, membrane_contact_k=0.0, membrane_exclude=False,
                     membrane_adhesion=5.0e4, membrane_gamma=2.0e3, membrane_tau=0.0, membrane_adhesion_fraction=0.05,
                         membrane_secrete_rate=0.0, membrane_reserve=0.0),
    "126_turnover": dict(membrane_springs=False, membrane_impl="mpm", membrane_direct_forces=True,
                     membrane_grid_bc=False, membrane_contact_k=0.0, membrane_exclude=False,
                     membrane_adhesion=5.0e4, membrane_gamma=2.0e3, membrane_tau=0.0, membrane_tau_adh=40.0,
                        membrane_secrete_rate=0.0, membrane_reserve=0.0),
    "127_rupture": dict(membrane_springs=False, membrane_impl="mpm", membrane_direct_forces=True,
                     membrane_grid_bc=False, membrane_contact_k=0.0, membrane_exclude=False,
                     membrane_adhesion=5.0e4, membrane_gamma=2.0e3, membrane_tau=0.0, membrane_detach=0.02,
                       membrane_secrete_rate=0.0, membrane_reserve=0.0),
    "128_secreting": dict(membrane_springs=False, membrane_impl="mpm", membrane_direct_forces=True,
                     membrane_grid_bc=False, membrane_contact_k=0.0, membrane_exclude=False,
                     membrane_adhesion=5.0e4, membrane_gamma=2.0e3, membrane_tau=0.0, membrane_secrete_rate=0.012, membrane_reserve=1.0,
                         membrane_particles=90000),
    # 129-130: THE DIRECT-FORCE PATH, THIS TIME ACTUALLY APPLIED. 120-128 all set
    #     `membrane_direct_forces=True` and none of them ran it: the branch was nested inside
    #     `if membrane_springs:` and every one of those runs sets springs off. So `membrane_gamma` never
    #     reached the operator, 120/121/122 were a bare stiffness sweep (1e4 / 5e4 / 2.5e5) rather than
    #     the k/gamma = 5 / 25 / 125 they are named for, and 122 collapsed at the frame-level stability
    #     limit dt_frame*sqrt(k) = 2.0 rather than from two integrators disagreeing.
    #
    #     WHAT CHANGES, AND THE PREDICTION IT MAKES. `emit: velocity` integrates x += dt*(k/gamma)*(a-x),
    #     first order and overdamped, so stability needs dt*(k/gamma) < 2 instead of dt*sqrt(k) < 1 --
    #     and the standoff stops being a force balance against the sheet's resistance to inflation and
    #     becomes a TRACKING LAG behind a moving anchor. The anchor moves 5.32e-4 box units per frame
    #     (R: 0.0835 -> 0.2973 over 402), so the lag is 5.32e-4 / (dt*k/gamma):
    #         r =  25   dt*r = 0.10   lag 0.0053   standoff = offset - lag = -0.0013
    #         r = 125   dt*r = 0.50   lag 0.0011   standoff              = +0.0029
    #     against 121's -0.0082. Both land near the 0 to +0.002 the biology asks for, and the second
    #     prediction is the sharper one: as r rises the standoff approaches +offset, i.e. the sheet sits
    #     at the fibre's rest length -- which is the flat rig's result arriving on the spheroid. If that
    #     holds, `membrane_offset` becomes the standoff and is a material property rather than a knob.
    "129_direct_r25_fixed": dict(membrane_springs=False, membrane_impl="mpm",
                                 membrane_direct_forces=True, membrane_grid_bc=False,
                                 membrane_contact_k=0.0, membrane_exclude=False,
                                 membrane_adhesion=5.0e4, membrane_gamma=2.0e3,
                                 membrane_secrete_rate=0.0, membrane_tau=0.0, membrane_reserve=0.0),
    "130_direct_r125_fixed": dict(membrane_springs=False, membrane_impl="mpm",
                                  membrane_direct_forces=True, membrane_grid_bc=False,
                                  membrane_contact_k=0.0, membrane_exclude=False,
                                  membrane_adhesion=2.5e5, membrane_gamma=2.0e3,
                                  membrane_secrete_rate=0.0, membrane_tau=0.0, membrane_reserve=0.0),

    # 131: THE FIBRE THAT CANNOT BE SQUASHED -- 126 again (turnover on, k/gamma = 25) with the radial
    #     compression branch stiffened 10x, `k_compress = 5e5` against `k = 5e4`. The question is
    #     whether a length is enough on its own: an integrin is a molecule of finite length, so the
    #     sheet should not be able to approach the surface closer than that length, and if the fibre
    #     enforces it then no separate repulsion is needed to keep the BM out of the epithelium --
    #     `basement_membrane_contact` becomes redundant rather than badly tuned.
    #     Measured on 121, every integrin in the run ends SQUASHED to a third of its 0.004 length
    #     (the sheet sits 0.0126 inside its rest position) and nothing objects, which is what a single
    #     stiffness both ways buys. Overdamped the compression branch is stable to dt*(k/gamma) < 2,
    #     i.e. k/gamma < 500, so 250 is inside it with room.
    "131_fibre_stiff_compress": dict(membrane_springs=False, membrane_impl="mpm",
                                     membrane_direct_forces=True, membrane_grid_bc=False,
                                     membrane_contact_k=0.0, membrane_exclude=False,
                                     membrane_adhesion=5.0e4, membrane_adhesion_compress=5.0e5,
                                     membrane_gamma=2.0e3, membrane_tau=0.0, membrane_tau_adh=40.0,
                                     membrane_secrete_rate=0.0, membrane_reserve=0.0),

    "_unused_secrete_dense": dict(membrane_springs=False, membrane_impl="mpm", membrane_grid_bc=True,
                             membrane_exclude=False, membrane_adhesion=0.0, membrane_tau=0.0,
                             membrane_particles=90000, membrane_reserve=1.0,
                             membrane_secrete_rate=0.012),

    # 89-91: Plexus's own attraction_repulsion law instead of the one-sided spring.
    #
    # WHAT 86-88 LEFT. 86 (w = 50) is the best packing so far, d/hex 0.746 -- but at z = 3.82, under the
    # 2D rigidity threshold, with lcc 0.64: some of that gain is bought by BREAKING crosslinks, which is
    # the same process that destroyed 87 (w = 120: lcc 0.04, z = 2.06, packing back down to 0.568). 88
    # was a null (0.679 against 85's 0.677), so the range-selectivity argument was simply wrong.
    #
    # THE ARCHIVED `blue` PARAMETERS DO NOT DO THIS, and that is worth recording because it is the
    # opposite of what the name suggests. f(0) = p0 - p2, and blue's is +0.022: an attractive core. Any
    # pair closing below r = 0.0011 welds, so the set CLUMPS -- in 2D, d/hex 0.471 -> 0.242 over 250
    # frames, worse than random (0.465). Every archived type in the two configs has an attractive core
    # except rand_2t's t0, and that one never decays to zero at the neighbour radius (f = -0.63 there),
    # so it churns instead. Dropping p0 leaves one decaying repulsion, which is the CGI recipe for
    # scattering points evenly over a surface, and it reaches blue noise in 2D (d/hex 0.887, cv 0.037).
    #
    # ON THE REAL SHEET, run 85's middle frame relaxed 300 iterations:
    #     start                          d/hex 0.677   cv 0.243   gap 1.14
    #     attraction_repulsion, 3 l*     d/hex 0.895   cv 0.052   gap 0.65
    # Better than anything the spring reached, and the likely reason is RANGE, not strength: the AR law
    # acts over ~3 spacings and dies smoothly, the one-sided spring acts over 1 and is truncated.
    #
    # k is bracketed, not derived. Matching total force against 85 (w = 20) over ~28 neighbours instead
    # of 6 gives k ~ 1.7e4, i.e. w ~ 3.5, so 89/90 straddle it. 91 is the literal reading of "only
    # attraction-repulsion": the crosslink springs off entirely, absolute k, no tension-bearing network
    # left -- expected to pack well and to have nothing holding the sheet against growth.

    # 92-94: 89-91 again with k set correctly. THE ERROR WAS UNITS, NOT THE LAW. In the ar form `k` is
    # not a stiffness: amp is O(1), the step is dt*k*amp*d, so the scale is fixed by k*dt ~ 1, meaning
    # k ~ 250 at dt = 4e-3 -- not the 1.75e4 I derived by matching spring forces, which is 70x past it.
    # Above the edge the sheet scrambles to a fixed point that sits near where it started, which is why
    # 89-91 read as a null (0.564 / 0.591 / 0.535) instead of as an overshoot. Swept on 85's middle
    # frame: k = 25 -> 0.785, 100 -> 0.830, 250 -> 0.846, 600 -> 0.855, 1500 -> 0.872, 4000 -> 0.730.
    # Mean aggregation, as the attraction_repulsion operator itself uses.

    # 95-97. 92-94 came back a flat null (0.542/0.540/0.544 against 82's 0.543 with no repulsion at
    # all), and that finally located the real error -- which is NOT the one 92-94 were correcting.
    #
    # THE BENCH CANNOT CALIBRATE THIS OPERATOR. relax_bench integrates x += dt*F, a position update. The
    # operator emits an ACCELERATION into MPM, so displacement goes as dt^2*F, smaller by ~1.6e-5 at
    # dt = 4e-3. The bench's k ~ 250-1500 is therefore about 1e5 too small in the run, and 89's faint
    # effect at k = 1.75e4 (0.564 against a 0.543 baseline) is consistent with being under-powered, not
    # with the overshoot I read it as. Two brackets went to calibration because I kept trusting a
    # testbed that does not share the run's equation of motion.
    #
    # The only sound anchor is the LINEAR repel, which is the same operator on the same emission path:
    # k = 1e5 there (85, w = 20) moved d/hex 0.543 -> 0.677 and k = 2.5e5 (86, w = 50) -> 0.746. Matching
    # the ar law's typical force against those gives k ~ 2e5, so this brackets it across a decade and a
    # half. Sum aggregation, deliberately -- same as the linear operator, so the ONLY thing that differs
    # is the force shape, which is the actual hypothesis: range ~3 l* and a smooth decay instead of
    # range 1 l* and a hard truncation.
}


def main():
    name = sys.argv[1]
    dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 402
    over = dict(SERIES[name])
    gated = over.pop("_gated", False)
    myo = over.pop("_myo", None)          # junction-level knobs: they belong to PASS 1, not the spec
    # RECORDED BEFORE THE POP CONSUMES IT. `_gated` is the only change 75 makes, so popping it left an
    # empty dict and the run archived itself as `{"reference": true}` -- identical to 69's label, for a
    # run on a tissue of aspect 1.332 against 69's 1.021.
    label = dict(over)
    if myo:
        label.update(myo)
    if gated:
        label["tissue"] = "gated ovoid (aspect 1.33)"
    for t in (membrane_ops.BOND_TRACE, membrane_ops.BOUNDARY_REACTION, membrane_ops.MEMBRANE_STRAIN,
              membrane_ops.SECRETE_TRACE, membrane_ops.BOND_SNAPSHOTS, membrane_ops.HOOP_TRACE):
        t.clear()
    tk = dict(frames=401, device=dev, buffer_x=4, myosin=1.0)
    if myo:
        tk.update(myo)
        # `:g` on a str raises; `myo_keyed_on` is a string ("tension"/"strain_rate"), so the tag has to
        # format by type. The cache key is built from this, so getting it wrong either crashes or -- worse
        # -- collides two different configurations onto one cached tissue.
        tk["tag_extra"] = "_" + "_".join(
            f"{k}{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k}{v}"
            for k, v in sorted(myo.items()))
    if gated:
        tk.update(gate_npz=GATE, gate_p_half="auto", gate_hill=6.0, gate_floor=0.08,
                  gate_smooth_frames=25, gate_smooth_phi=360.0, tag_extra="_gated_myo")
    npz = TIS.load_or_build(**tk)
    cfg = dict(aniso.BASE)
    cfg.update(BASE)
    cfg.update(over)
    cfg["membrane"] = npz
    spec, info = C.build(name, npz, **cfg)
    spec["general"]["n_frames"] = frames
    # A SHORT RUN IS A SMOKE TEST, AND IT GETS ITS OWN FOLDER. Running one under the real name writes
    # into the folder a cluster job may be using for the same name -- which is how 77 ended up with a
    # 24-frame movie from a 25-frame test sitting beside a 402-frame trajectory, and could as easily have
    # corrupted the trajectory instead of just the movie. The underscore prefix also keeps it off the
    # numbered ledger.
    if frames < 100:
        name = "_smoke_" + name
        print(f"[series] {frames} frames -> writing to {name}, not the numbered folder", flush=True)
    # WIPED BEFORE WRITING. A relaunch that leaves the previous attempt's files behind gives a folder
    # whose contents came from two different runs, and no way to tell which is which -- 79 held a
    # 24-frame movie from a smoke test beside nothing else, and 77 held one beside a 402-frame
    # trajectory. Either the folder is this run's output or it is empty.
    d = os.path.join(R.LOG, name)
    if os.path.isdir(d):
        import shutil
        shutil.rmtree(d, ignore_errors=True)
        print(f"[series] cleared {d} before writing", flush=True)
    os.makedirs(d, exist_ok=True)
    info["varied"] = label or {"reference": True}
    json.dump(info, open(os.path.join(d, "pass1.json"), "w"), indent=1)
    R.run(name, spec, device=dev, movie=True, render_kw={"movie_frames": 150, "fps": 15})

    # THE REACTION THE TISSUE WOULD FEEL, saved whenever the grid boundary condition ran. Pass 2 replays
    # a tissue it cannot push, so this is the coupling's other half measured rather than applied.
    br = np.asarray(membrane_ops.BOUNDARY_REACTION, float)
    if br.size:
        np.savez_compressed(os.path.join(d, "reaction.npz"), reaction=br)
    bt = np.asarray(membrane_ops.BOND_TRACE, float)
    z = np.load(os.path.join(d, "traj.npz"))
    al = np.asarray(z["malive"]) if "malive" in z.files else None
    ms = np.asarray(membrane_ops.MEMBRANE_STRAIN[-1], float)
    if al is not None:
        ms = ms[al]
    ms = np.nan_to_num(ms, nan=-1, posinf=-1, neginf=-1)
    P = np.asarray(z["mpos"])[-1][al] if al is not None else np.asarray(z["mpos"])[-1]
    u = P - P.mean(0)
    u /= np.linalg.norm(u, axis=1)[:, None]
    th = np.arccos(np.clip(u[:, 2], -1, 1))
    ph = np.arctan2(u[:, 1], u[:, 0])
    bi = (np.clip((th / np.pi * 16).astype(int), 0, 15) * 32
          + np.clip(((ph + np.pi) / (2 * np.pi) * 32).astype(int), 0, 31))
    # FROM THE TISSUE FILE, not the module global: the global is empty whenever pass 1 came from cache.
    _tz = np.load(npz)
    t1 = np.asarray(_tz["t1_trace"], float) if "t1_trace" in _tz.files else np.zeros((1, 4))
    if t1.size == 0:
        t1 = np.zeros((1, 4))
    # A CONTINUUM MEMBRANE HAS NO BONDS, so bonds_end / lcc / z are not poor measurements of it -- they
    # do not exist. What replaces them is the thing those numbers were standing in for: whether the sheet
    # still COVERS the epithelium. `coverage` (fraction of 16x32 solid-angle bins holding a particle) is
    # already computed below and is the honest measure for both representations, which is why it is the
    # one to compare a continuum run against a spring run on.
    has_bonds = bt.ndim == 2 and bt.shape[0] > 0
    info["result"] = dict(bonds_start=int(bt[0, 0]) if has_bonds else None,
                          bonds_end=int(bt[-1, 0]) if has_bonds else None,
                          lcc_end=float(bt[-1, 3]) if has_bonds and bt.shape[1] > 3 else None,
                          mean_degree_z=float(bt[-1, 4]) if has_bonds and bt.shape[1] > 4 else None,
                          t1_total=int(t1[-1, 2]) if len(t1) else 0,
                          t1_per_cell_per_frame=float(t1[:, 1].sum() / max(t1[:, 3].mean(), 1)
                                                      / max(len(t1), 1)),
                          n_alive=int(al.sum()) if al is not None else None,
                          strain_end=float(ms.mean()),
                          strain_p99=float(np.percentile(ms, 99)),
                          coverage=len(np.unique(bi)) / (16 * 32),
                          reaction_radial_end=(float(br[br[:, 0] == br[:, 0].max(), 2].sum())
                                               if br.size else None))
    json.dump(info, open(os.path.join(d, "pass1.json"), "w"), indent=1)
    print("SERIES " + json.dumps({"name": name, **info["result"]}), flush=True)

    # THE SECTION, ALWAYS. `movie.mp4` frames the whole tissue, where the sheet is a one-dot rim and the
    # integrin holding it is sub-pixel -- so every question this prototype is now asking (where the BM
    # sits, whether the fibre is stretched or squashed, which anchors are still bound) was invisible in
    # the artefact each run produced. It costs ~3 minutes off `traj.npz`, no GPU, and a run whose spec
    # has no `integrin_adhesion` simply says so and skips.
    try:
        import bm_section
        bm_section.render(name)
    except SystemExit as e:
        print(f"[series] no section.mp4: {e}", flush=True)
    except Exception as e:
        print(f"[series] section.mp4 FAILED ({type(e).__name__}: {e}) -- traj.npz is on disk, so "
              f"`bm_section.render('{name}')` redraws it without re-running", flush=True)


if __name__ == "__main__":
    main()
