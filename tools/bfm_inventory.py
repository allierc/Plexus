#!/usr/bin/env python
"""Step 0001 of exp_02_bacterium: the AUDIT, as a figure in the record.

    PLEXUS_BUILDER=/workspace/Plexus/builder/exp_02_bacterium python tools/bfm_inventory.py

WHY THIS IS THE FIRST ENTRY. builder/instruction.md section 2: "you are not starting from
scratch, and finding that out is the first task" -- audit what exists, what it achieved, what it
abandoned and why, and put it in the record BEFORE writing a spec. A table in a chat message is
the one part of the work the human cannot see, so the inventory is a picture with its reasons
beside it, like every other step.

Every line below was read in the source or in exp_01's `why` texts today; nothing is from
memory. The `why` text carries the full inventory; the figure is the same content laid out so
it can be read at a glance on the watch page.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

# ---- what exists, one row each: (component, where, what it gives the bacterium) -------------
EXISTS = [
    ("registry", "142 operators / 154 contracts, 27 entities, 6 fields, 1,701 specs in 22 areas",
     "census run today; helix / rotary / torque appear nowhere"),
    ("rod_seed", "rod_ops.py  lays a STRAIGHT chain; writes the 1D mesh (segment table, no face)",
     "the filament set; needs a helix layout"),
    ("rod_elastic", "stretch + bend on the table, force- AND torque-free, overdamped limit zeta h^4/B",
     "rest curvature is ZERO (straight); no torsion term"),
    ("rod_base", "pin + direction clamp (no drive; the beat is rod_motor's); anchor: <set> returns the reaction",
     "reaction onto a body proven (0411); no axial torque"),
    ("rod_motor", "distributed travelling curvature wave, as couples; `ramp` for startup",
     "a cilium/sperm drive -- not a bacterium's"),
    ("drag", "motion_ops.py  RFT anisotropy (along: chain, ratio 2), field: advection, react: two-way,\n"
             "coupling 0..1, zeta in sim-mass/s, residual gate ~1e-8",
     "the filament-water coupling; anisotropy NOT in the overdamped branch"),
    ("water", "cil_t01_c0p1: 155,374 MPM points, n_grid 32 (dx 1.56 um), warp + CUDA graph, dt 1e-5 s",
     "quiescent uncoupled (0.0 nm / 15,000 frames); nu_eff is the GRID's, 1e4x water"),
    ("units", "length_um 50, time_s 1, force_nN 6.25e-9 (1 sim mass = 1.25e-13 kg, from the water)",
     "inherit; re-derive if the box changes"),
    ("mpm_spin", "mpm_ops.py  P-controller to solid-body rotation of MPM points",
     "the only rotary thing; kinematic, deformable body, not a torque motor"),
    ("mesh_contact", "contact_ops.py  live triangulated surface <-> MPM (Chen 2015)",
     "a body in water -- but a 1 um body is sub-grid at dx 1.56 um"),
    ("renderer", "plotting.chain draws any segment table; vtk_points water, cutaway, radial colour",
     "a helix is a chain with other positions: nothing to add"),
    ("tools", "gui_drive opencycle; cilia_row gates; rod_probe; builder_figure; cilium_units; nu_effective",
     "gates transfer: box, arc, strain, Hz, drift-not-excursion"),
]

# ---- what exp_01 tried and rejected, and the measurement that decided it --------------------
REJECTED = [
    ("MPM cilium (cilia_ops, deleted)", "0.25 um body in a 4.08 um cell is advected, not swum; widened to a paddle = not a cilium"),
    ("rod as MPM particles", "node is 0.53% of its stencil's mass; gather REPLACES its velocity: base 60.3 -> 1.0 deg"),
    ("inertial rod", "real cilium overdamped by 6.1e6; inertial rod rings and spins (0331-0342, t/b 1.0-1.05)"),
    ("motor as k(c - C0)", "force-free but NOT torque-free: net torque 8e-8 vs 2e-17 -> couples"),
    ("basal moment only", "Machin 1958: 15x decay over 20 um (10.2 -> 0.67 deg) -> distributed drive"),
    ("standing-wave control", "0416: a flexible filament makes its own travelling wave -> RIGID control instead"),
    ("full-amplitude start", "54x startup velocity spike, 6.5e6 kick to the water -> ramp"),
    ("constants tuned to pictures", "`force NOT DECLARED`; deriving from Sartori/Gray-Hancock/Poon moved the working point by decades"),
    ("Re quoted from eta", "shear-wave bench: nu_eff = 0.016 at n_grid 32, eta invisible; Re <= cells across the body"),
]

# ---- the gaps a rotary-motor bacterium opens, each with the smallest extension --------------
GAPS = [
    ("1. helical rest shape", "rod_seed `shape: helix` (radius, pitch, hand); rod_elastic rest turning angle + a\n"
                              "   DIHEDRAL (torsion) term -- a point chain has no material frame, so torsion is a dihedral"),
    ("2. anisotropic overdamped drag", "the overdamped branch of `drag` discards the tangent it computed; rotation with\n"
                                       "   ISOTROPIC drag gives zero thrust, so this is the whole propulsion mechanism"),
    ("3. the rotary motor", "a TORQUE about the filament axis at the base, equal and opposite on the body,\n"
                            "   with Berg's torque-speed curve; no operator does this (rod_base bends, it does not twist)"),
    ("4. a body with orientation", "no set carries a pose; a body is a point cloud. Cluster of drag nodes vs a pose\n"
                                   "   state -- to decide at the decomposition, not before"),
    ("5. resolution", "helix radius 0.2 um and pitch 2.3 um sit at or below dx 1.56 um: RFT is local and\n"
                      "   survives; the far field and body-filament interaction do not"),
]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from builder_figure import record

    fig = plt.figure(figsize=(17, 12.5))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], width_ratios=[1.0, 1.0],
                          hspace=0.18, wspace=0.06, left=0.02, right=0.98, top=0.95, bottom=0.02)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    for ax in (ax_a, ax_b, ax_c):
        ax.set_axis_off()

    def table(ax, title, rows, x0, x1, x2, fs=8.6, gap=1.0):
        ax.text(0.0, 1.0, title, transform=ax.transAxes, fontsize=11.5, va="top", ha="left")
        y = 0.90
        step = 0.90 / (sum(2 if "\n" in (r[1] if len(r) == 2 else r[1] + r[2]) else 1 for r in rows) * gap + 1.5)
        for r in rows:
            if len(r) == 3:
                name, where, gives = r
                ax.text(x0, y, name, transform=ax.transAxes, fontsize=fs, va="top",
                        fontweight="bold", family="monospace", color="#1f4e79")
                ax.text(x1, y, where, transform=ax.transAxes, fontsize=fs, va="top", color="#222222")
                ax.text(x2, y, gives, transform=ax.transAxes, fontsize=fs, va="top", color="#7a2e1d")
                n = 2 if "\n" in where else 1
            else:
                name, what = r
                ax.text(x0, y, name, transform=ax.transAxes, fontsize=fs, va="top",
                        fontweight="bold", family="monospace", color="#1f4e79")
                ax.text(x1, y, what, transform=ax.transAxes, fontsize=fs, va="top", color="#222222")
                n = 2 if "\n" in what else 1
            y -= step * n * gap
        ax.plot([0, 1], [0.93, 0.93], transform=ax.transAxes, color="#bbbbbb", lw=0.8)

    table(ax_a, "A. WHAT EXISTS (read in the source today)  --  component | where and what it does | what it gives the bacterium",
          EXISTS, 0.0, 0.085, 0.66, fs=8.6)
    table(ax_b, "B. WHAT exp_01 TRIED AND REJECTED, with the number that decided it",
          REJECTED, 0.0, 0.34, None, fs=8.2)
    table(ax_c, "C. THE GAPS A ROTARY-MOTOR BACTERIUM OPENS  --  each with the smallest extension",
          GAPS, 0.0, 0.0, None, fs=8.4, gap=1.15)
    # C's second column starts under the heading, so the two-line entries read as one paragraph
    for t in ax_c.texts[1:]:
        if t.get_fontweight() != "bold":
            t.set_position((0.0, t.get_position()[1] - 0.035))

    why = (
        "AUDIT, not a run -- the first entry of exp_02_bacterium, as builder/instruction.md section 2 asks: what "
        "exists, what it achieved, what it abandoned and why, written BEFORE any spec. Every line was read in the "
        "source or in exp_01's own `why` texts today. "
        "THE FRAMEWORK: a census of the live registry gives 142 canonical operators (154 contracts counting "
        "model/implementation variants), 27 entities, 6 fields and 1,701 specs in 22 config areas; the words "
        "helix, helical, rotary and torque occur in no operator except a docstring of rod_base. The language: "
        "sets, maps (containment pi plus declared `maps:`, so a relation is a set with functions out of it -- the "
        "1D mesh is a half-edge set with `face` absent), state blocks with per-block integration (first_order = "
        "overdamped, second_order = inertial), fields, and operators that return deltas the ENGINE integrates; the "
        "operator's EMIT decides whether a delta is a velocity or an acceleration, which is how an overdamped body "
        "is expressed at all. Seed is its own lifecycle phase. Units are declared once (length_um, time_s, force_nN) "
        "and never inferred. "
        "PANEL A, DIRECTLY REUSABLE: rod_seed lays a straight chain and writes the segment table (E_srce/E_trgt, "
        "rod_arc, rod_axis, one anchor per rod, n_rod + spacing); rod_elastic is stretch plus bend on that table, "
        "force- and torque-free to machine precision, with stated explicit limits (dt < 2/sqrt(k_s), "
        "dt < 0.5/sqrt(k_b), overdamped limit zeta h^4/B); rod_base pins the base, clamps its direction, and with "
        "`anchor: <set>` returns the equal and opposite onto a body set -- proven to reproduce the fixed-base beat "
        "to three digits at exp_01 step 0411; rod_motor is a distributed prescribed curvature wave applied as "
        "torque-free couples -- a cilium's drive, which a bacterium does not have, though its `ramp` startup fix "
        "carries over. `drag` is the filament-water coupling: resistive-force anisotropy (`along: chain`, `ratio` "
        "about 2 after Gray & Hancock 1955), advection by the B-spline-sampled MPM grid velocity (`field:`), "
        "two-way momentum exchange (`react: true`, `react_set`, `zeta` in sim-mass per second), a `coupling` dial "
        "from 0 to 1 that scales advection and reaction by the same factor so momentum balances at any setting, "
        "and a printed residual gate |sum f_rod + sum f_water| / sum |f_rod| that sat near 1e-8 with thousands of "
        "fluid particles touched. The water block of cil_t01_c0p1 -- 155,374 MPM points, n_grid 32 so dx = 1.56 um "
        "in a 50 um box, warp kernels captured as one CUDA graph, substep 1e-5 s -- was proven quiescent when "
        "uncoupled (0 of 15,001 frames differ, 0.0 nm displacement, step 0356). Its effective viscosity is the "
        "grid's: the shear-wave bench (commit 4e4fcd6d) measured nu_eff = 0.016 world^2/s at n_grid 32 against a "
        "physical 1e-6, and a 1000x change of eta moved nu_eff by 8 percent, so eta is inert and Re is capped near "
        "L/dx, the cells across the body. The units contract that goes with that water: 1 length unit = 50 um, "
        "1 time unit = 1 s, 1 force unit = 6.25e-9 nN, derived from the water's own particle mass (1 sim mass = "
        "1.25e-13 kg, step 0349). mpm_spin is the only rotary mechanism in the registry -- a proportional "
        "controller driving MPM points toward solid-body rotation -- kinematic and for a deformable continuum, "
        "not a torque motor and not applicable to a rod. mesh_contact couples a live triangulated surface to MPM "
        "(Chen et al. 2015) and is the natural candidate for a cell body in water, but a 1 um bacterium is below "
        "the 1.56 um grid cell, the same floor that killed the MPM cilium, so it is rejected for the body at this "
        "grid. The renderer's `plotting.chain` draws whatever the segment table says, so a helix costs nothing; "
        "the tools -- gui_drive opencycle, cilia_row's gate table (box, arc, strain, Hz, and DRIFT as the mean "
        "displacement vector over whole beats rather than excursion), rod_probe, builder_figure, cilium_units, "
        "nu_effective -- transfer as they are. "
        "PANEL B, WHAT WAS REJECTED AND WHY, each by a measurement: the MPM cilium (deleted): a 0.25 um body in a "
        "4.08 um cell is advected rather than swum, and widening it made a paddle, not a cilium. The rod immersed "
        "as MPM particles: a node is 0.53 percent of its stencil's mass and mpm_gather REPLACES its velocity, so "
        "99 percent of each substep's drive was discarded (base 60.3 -> 1.0 degrees; density x100 moved the tip "
        "0.00 -> 4.79 degrees, which identified the cause). The inertial rod: a real cilium is overdamped by "
        "6.1e6 and an inertial one rings and spins whatever the drive (0331-0342, tip/base 1.00-1.05 = rigid "
        "rotation through a fivefold curvature drop, a 19x stiffer clamp and a pinned stroke plane). The motor as "
        "k(c - C0): force-free but not torque-free, net torque 8e-8 against rod_elastic's 2e-17, so it spun the "
        "filament with no external agent -> couples. The basal moment alone: Machin 1958, 15x decay over 20 um. "
        "The standing-wave reciprocity control (0416): a flexible filament driven in phase makes its own travelling "
        "wave (lag 7 -> 101 degrees along it) and pumps perfectly well -> the control has to be a RIGID filament. "
        "The full-amplitude start: 54x velocity spike -> ramp. Constants tuned to pictures under `force NOT "
        "DECLARED`: deriving them from Sartori 2016, Gray & Hancock 1955 and Poon 2025 moved the working point by "
        "orders of magnitude. Every Reynolds number quoted from eta: fiction, the grid sets the dissipation. "
        "PANEL C, THE GAPS THIS TASK OPENS, each with the smallest extension I can see: (1) a helical rest shape -- "
        "rod_seed lays straight lines and rod_elastic drives curvature to ZERO, so a helix would straighten; it "
        "needs a `shape: helix` layout (radius, pitch, handedness) and a rest turning angle per joint plus a "
        "DIHEDRAL term per four consecutive nodes for the torsion, because a point chain has no material frame and "
        "a helix is exactly constant curvature plus constant torsion (the bead-spring helices of Reichert & Stark "
        "2005 and Vogel & Stark 2010). (2) Anisotropic drag in the OVERDAMPED branch: `drag` computes the tangent "
        "and the ratio, then overwrites the result with the advection velocity and hands the fluid the isotropic "
        "-zeta (v - u) -- step 0349 says so in one sentence. A rotating helix with isotropic drag produces exactly "
        "zero thrust, so this is not a refinement: it is the propulsion mechanism, and it belongs in the existing "
        "operator as the parameter it already declares. (3) The rotary motor: a torque about the filament's own "
        "axis at its base, with the equal and opposite torque on the body, following Berg's torque-speed curve; "
        "no operator does this -- rod_motor applies a BENDING moment about the beat-plane normal, not an axial "
        "torque. This rotary drive is the one genuinely new mechanism. (4) A body with an "
        "orientation: no set carries a pose, a body is a point cloud; the choice between a rigid cluster of drag "
        "nodes and a pose state is the decomposition's to make, after the anatomy. (5) Resolution: a helix of "
        "radius 0.2 um and pitch 2.3 um sits at or below dx 1.56 um; the RFT coupling is local and survives, the "
        "far field and the body-filament hydrodynamic interaction do not, and that has to be stated in every "
        "claim about swimming. "
        "THE ORDER FROM HERE is the instruction's: literature (Berg 2003, Purcell 1977, an in-situ cryo-ET motor "
        "with verified accessions) -> anatomy -> decomposition into sets, maps and operators -> the smallest thing "
        "that can be wrong: one helix, one motor, no water, does it rotate at the commanded rate and stay helical."
    )
    record(fig, why, name="audit: what exists, what was rejected, what a rotary-motor bacterium opens")


if __name__ == "__main__":
    main()
