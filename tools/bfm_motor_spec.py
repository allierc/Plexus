#!/usr/bin/env python
"""Write the flagellar-motor anatomy spec from its measured dimensions, and print what it predicts.

    python tools/bfm_motor_spec.py [name] [--cj] [--scaffold] [--cryo] [--emerge [--stochastic] [--thermal]] [--ions [--pump X] [--shunt X N]] [--membrane] [--scaffold-material] [--protons] [--cut] [--volume-curves | --diag-curves] [--grid N] [--color-stators x|psi]
        # -> config/bacterium/<name>.yaml (default bfm_a03_rotor_stator); --cj the C. jejuni table
        #    (CJ_PARTS), --scaffold the static 9HMF scaffold around it, --cryo the figure look,
        #    --emerge the stepping stator on its elastic linkage instead of the constant push

WHY A SCRIPT WRITES THE SPEC. Every number in the spec is DERIVED: a ring's `aspect` and `hollow`
from its radii and height, a part's particle count from its volume and one particle volume, the
stator force from the torque the load and the target rate demand, the units from the mapping of
that force and that rate onto the animal's. Written by hand those are twelve chances to disagree;
written here they are one function of the anatomy table below, and the `why` of the record step
can quote the predictions this prints (the rotor's moment of inertia, the torque, the rotation
rate the torque/load balance implies) so the run has a number to be wrong against.

THE ANATOMY, an E. coli / Salmonella-type motor, in nanometres from the inner-membrane plane
(z up = towards the outside of the cell). Every dimension is from the literature read today and is
a stand-in until the deposited map lands in the record (builder/exp_02_bacterium/data/):

    part      r_out  r_in   z0    z1   copies  source
    c_posts   22.5   15.0  -22    -6     34    the C-ring's 34 FliM/FliN posts: ~45 nm diameter,
                                               34-fold (Thomas et al. 2006 J Bacteriol 188:7039;
                                               Singh/Johnson 2024 Nat Microbiol)
    c_flig    22.5   13.0   -6    -2      1    the continuous FliG ring on top of the posts, its
                                               inner lip under the MS-ring's rim; stators act on it
                                               at r = 20 nm: 1,606 pN nm = 11 x 7.3 pN x r
                                               (Drobnic et al. 2025, Nat Microbiol 10:1723)
    ms_ring   14.0    5.5   -2    10      1    34 FliF, ~28 nm across (Tan et al. 2021 Cell 184:2665,
                                               PDB 7CGO); beta-collar radius 51 A (Drobnic 2025)
    rod        5.5    0      2    46      1    ~11 nm wide, from inside the MS-ring's bore up
                                               through the periplasm into the hook's base (Tan 2021)
    lp_ring   13.0   10.0   26    40      1    the L/P bushing, ~26 nm across (Salmonella LP-rings,
                                               EMD-12183 cited by Drobnic 2025); the rod-ring gap is
                                               set to 4.5 nm = 4.1 grid cells so the grid cannot glue
                                               them -- the real gap is ~1-2 nm and that is STATED
    hook       9.0    0     44    84      1    a 40 nm stub of the 55 nm hook, 18 nm wide (Berg 2003),
                                               starting 4 nm above the bushing (see PARTS below)
    stator    r 4.5 spheres at radius 27.5, z -4, 11 units  MotA5B2, ~9 nm (Santiveri 2020, Deme
                                               2020); >= 11 units in E. coli (Reid et al. 2006)

Each part is its OWN SET of material points with its own one-element parent that carries the
part's material -- the language's point: an MS-ring, a C-ring and a rod are distinct kinds of
entity, not one cloud with three colours. Parts that touch are glued by the shared MPM grid, which
is what makes the rotor one body; the bushing keeps a gap and is anchored; the stator units are a
separate set of eleven elements that act on the rotor through `stator_push` and never through the
grid.
"""
from __future__ import annotations

import math
import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- the box and the grid ----------------------------------------------------------------------
BOX_NM = 140.0                 # world unit = 140 nm; the motor stack spans 98 nm of it
N_GRID = 128                   # dx = 1.094 nm; a 4.5 nm gap is 4.1 cells, beyond the 3-cell stencil
PPC = 8                        # material points per grid cell
Z_MEMBRANE = 0.30              # the inner-membrane plane, in world units
DENSITY = 1.0                  # sim density; one value for every protein part
YOUNGS = 2000.0                # sim stress; c = sqrt(E/rho) = 44.7 world/s, CFL 0.29 at the substep
SUBSTEP = 5.0e-5               # s (sim)
FRAME_DT = 2.0e-3              # s (sim): 40 substeps a frame
N_FRAMES = 1500                # 3 s sim = 1.5 turns at the target rate
LOAD_DRAG = 50.0               # 1/s: the placeholder load, `drag` in mpm_scatter, until the filament exists
TARGET_TURNS_PER_S = 0.5       # the design rate: one turn every 2 s sim <-> 100 Hz in the animal
N_STATOR = 11
F_UNIT_PN = 7.3                # pN per stator unit (Ryu, Berry & Berg 2000; Drobnic 2025)
REACH_NM = 9.0
STATOR_R_NM, STATOR_Z_NM, STATOR_RADIUS_NM = 27.5, -4.0, 4.5

# part: (r_out, r_in, z0, z1, copies)
#
# PARTS THAT ARE ONE BODY MUST SHARE GRID NODES, and the first draft's did not: an MS-ring 7.8-12.2 nm
# from the axis stacked on C-ring pillars 15.9-21.6 nm out meet on a plane with no radial overlap,
# 3.7 nm = 3.4 cells apart, beyond the 3-cell stencil -- so the C-ring turned and nothing above it
# moved (measured headless before A02: the hook at exactly 0.000 degrees; quoted in the `why` of
# record step 0005). The C-ring is therefore its
# two real layers: the continuous FliG ring on top (r 13-22.5 nm), whose inner lip lies under the
# MS-ring's rim (r 13-14 nm, the FliF-FliG contact of Johnson et al. 2024), and the 34 FliM/FliN
# posts beneath it; the rod passes THROUGH the MS-ring's hole from z 2 (the FliE/FlgB proximal rod
# sits inside FliF) rather than starting on its top face; and the hook's base overlaps the rod's top.
# The stators push the FliG ring, which is what MotA actually steps against.
# THE 34 POSTS ARE TWO SETS OF 17, ALTERNATING, and that is a rendering fact rather than an
# anatomical one: the movie renderer paints a set one colour, so a single ring of 34 identical
# posts turning is a picture of nothing turning (A02's caption saw the turn only on the FliG ring's
# copy colours, which the compartments renderer that draws ALL the parts does not carry). Two
# interleaved sets, one red and one pale, make the rotation legible; they are the same material at
# the same radius and height, one phase apart, and the grid glues them into the same ring.
# (r_out, r_in, z0, z1, copies, phase_deg, posts_in_the_whole_ring): a set holding 17 of the ring's 34
# posts gets 17/34 of the annulus's volume, and each post the size a 34-post ring gives it.
PARTS = {
    "c_posts_a": (22.5, 15.0, -22.0, -6.0, 17, 0.0, 34),
    "c_posts_b": (22.5, 15.0, -22.0, -6.0, 17, 360.0 / 34.0, 34),
    "c_flig":  (22.5, 13.0, -6.0, -2.0, 1),
    "ms_ring": (14.0, 5.5, -2.0, 10.0, 1),
    "rod":     (5.5, 0.0, 2.0, 46.0, 1),
    "lp_ring": (13.0, 10.0, 26.0, 40.0, 1),
    "hook":    (9.0, 0.0, 44.0, 84.0, 1),
}
# THE HOOK STARTS 4 nm ABOVE THE BUSHING, NOT ON IT. With its base at z 34, inside the L-ring's
# bore, the 9 nm hook sat 1 nm from the 10 nm ring and the grid glued them: the anchored bushing
# turned with the hook, 2.62 against 2.63 degrees at frame 40. In the animal the hook begins at the
# outer membrane, level with the L-ring's top; here it begins one stencil (4 nm = 3.7 cells) above
# it, and the rod runs 2 nm up into its base so the two overlap in volume. Same compromise as the
# 4.5 nm rod-bushing gap, stated for the same reason.
ROTOR = ("c_posts_a", "c_posts_b", "c_flig", "ms_ring", "rod", "hook")
PUSHED = "c_flig"

# THE C. JEJUNI ROTOR, `--cj`: the same parts widened to the stator ring the scaffold of PDB 9HMF
# fixes (record steps 0010-0014). Nothing here is measured yet -- the rotor is NOT in 9HMF and the
# whole-motor map EMD-16723 is what will replace this table -- so every number is the E. coli
# stand-in scaled to the one thing that is known: the stators push the C-ring at r = 26.5 nm
# (3,288 pN nm = 17 x 7.3 pN x 26.5 nm, Drobnic et al. 2025), and the MS-ring's beta-collar is
# 6.2 nm in radius against Salmonella's 5.1. The FliG ring therefore reaches r 29 nm with 34 posts
# beneath it (two sets of 17, the scaffold's own symmetry doubled -- a guess, stated); the MS-ring
# keeps r_out 14 because PflA's inner end sits at r 15.4 with the E-ring between them; rod, bushing
# and hook are unchanged. The scaffold's own atoms are placed as static sets around it
# (`--scaffold`), 8.5 nm above the membrane plane as in the scaffold scene.
CJ_PARTS = {
    "c_posts_a": (29.0, 19.0, -22.0, -6.0, 17, 0.0, 34),
    "c_posts_b": (29.0, 19.0, -22.0, -6.0, 17, 360.0 / 34.0, 34),
    "c_flig":  (29.0, 13.0, -6.0, -2.0, 1),
    "ms_ring": (14.0, 6.6, -2.0, 10.0, 1),
    "rod":     (5.5, 0.0, 2.0, 46.0, 1),
    "lp_ring": (13.0, 10.0, 26.0, 40.0, 1),
    "hook":    (9.0, 0.0, 44.0, 84.0, 1),
}
CJ_N_STATOR, CJ_STATOR_R_NM, CJ_STATOR_Z_NM, CJ_R_PUSH_NM = 17, 26.5, -1.5, 26.5


def nm(x):
    return x / BOX_NM


def main():
    global PARTS, N_STATOR, STATOR_R_NM, STATOR_Z_NM
    cj = "--cj" in sys.argv
    with_scaffold = "--scaffold" in sys.argv
    look = "cryo" if "--cryo" in sys.argv else "dots"
    emerge = "--emerge" in sys.argv            # the stepping stator on its linkage, not the constant push
    stochastic = "--stochastic" in sys.argv
    ions = "--ions" in sys.argv                # the membrane potential as a state of the cell (implies --emerge)
    cut = "--cut" in sys.argv                  # `near_side`: every surface cut at its centre, normal to the view, to look inside
    volume_curves = "--volume-curves" in sys.argv   # the axis parts' volume (det F) over time on the movie, instead of the ion panels
    color_stators = next((sys.argv[i + 1] for i, t in enumerate(sys.argv) if t == "--color-stators" and i + 1 < len(sys.argv)), None)
    thermal = "--thermal" in sys.argv               # the random torque of the load's own drag at kT (with --stochastic for the shot noise)
    membrane = "--membrane" in sys.argv             # E1: the inner membrane as a set of points around the motor (implies --ions)
    ions = ions or membrane
    scaffold_material = "--scaffold-material" in sys.argv   # E2: the 9HMF proteins as MPM bodies, the stators anchored in MotB
    protons = "--protons" in sys.argv                # E3: the proton pool's depletion at each channel's mouth (a Green's function, not a field)
    with_scaffold = with_scaffold or scaffold_material
    emerge = emerge or ions
    # THE LOAD SWEEP: `--drag X` sets the placeholder drag (1/s) on every rotor point, `--frames N`
    # the run length -- a torque-speed curve is one run per load, and a steady state arrives
    # within a few hundred frames.
    global LOAD_DRAG, N_FRAMES, N_GRID, SUBSTEP, YOUNGS
    diag_curves = "--diag-curves" in sys.argv       # rotor rate, rod volume, rod strain on the movie
    for i, tok in enumerate(sys.argv):
        if tok == "--drag":
            LOAD_DRAG = float(sys.argv[i + 1])
        if tok == "--frames":
            N_FRAMES = int(sys.argv[i + 1])
        if tok == "--youngs":                           # `--youngs Y --substep s`: a stiffer material (protein ~1 GPa = 50,000 sim) at the CFL it needs
            YOUNGS = float(sys.argv[i + 1])
        if tok == "--substep":
            SUBSTEP = float(sys.argv[i + 1])
        if tok == "--grid":
            # THE GRID TEST: the same motor on a coarser or finer grid. dx follows N, the substep
            # follows dx so the CFL number is the one the 128 grid ran at, and the particle count
            # follows dx^3 at the same PPC. The 4.5 nm rod-bushing gap is 4.1 cells at 128, 3.1 at
            # 96 (inside the 3-cell stencil: the bushing would grip the rod) and 5.1 at 160.
            N_GRID = int(sys.argv[i + 1])
            SUBSTEP = 5.0e-5 * 128.0 / N_GRID
    if cj:
        PARTS, N_STATOR, STATOR_R_NM, STATOR_Z_NM = CJ_PARTS, CJ_N_STATOR, CJ_STATOR_R_NM, CJ_STATOR_Z_NM
    dx = 1.0 / N_GRID
    p_vol = (dx ** 3) / PPC                      # world^3 per material point
    p_mass = p_vol * DENSITY
    sets, ops_frame = {}, []
    strains, scatters, gathers = [], [], []
    I_rotor, counts = 0.0, {}
    for name, spec_ in PARTS.items():
        ro, ri, z0, z1, k = spec_[:5]
        phase = float(spec_[5]) if len(spec_) > 5 else 0.0
        n_full = int(spec_[6]) if len(spec_) > 6 else k        # posts in the whole ring this set is part of
        h = z1 - z0
        V_nm3 = math.pi * (ro ** 2 - ri ** 2) * h * (k / n_full)
        V = V_nm3 / BOX_NM ** 3
        n = int(round(V / p_vol))
        counts[name] = n
        zc = Z_MEMBRANE + nm(0.5 * (z0 + z1))
        t = {"count": 1, "material": "elastic", "youngs": YOUNGS, "density": DENSITY,
             "shape": "cylinder", "axis": "z"}
        if k > 1:
            # k protomers, each a solid pillar of V/k and the ring's height, on a ring at r_mid
            r_mid = 0.5 * (ro + ri)
            r_copy = math.sqrt((V_nm3 / k) / (math.pi * h))     # = the annulus / n_full posts
            t["aspect"] = h / (2.0 * r_copy)
            t["hollow"] = 0.0
            t["ring"] = {"n": k, "radius": round(nm(r_mid), 6), "axis": "z", "phase": phase}
        else:
            t["aspect"] = h / (2.0 * ro)
            t["hollow"] = ri / ro
        sets[name] = {"n": 1, "start": [[0.5, 0.5, round(zc, 6)]], "types": {name: t}}
        sets[name + "_pt"] = {
            "parent": name, "entity": "mpm_particle", "per_parent": n, "radius": round(nm(ro), 6),
            "density": DENSITY, "particle_mass": p_mass,
            "state": {"pos": {"width": 3, "role": "coordinate", "integration": "second_order_coordinate",
                              "boundary": "world"},
                      "vel": {"width": 3, "role": "rate", "integration": "second_order_rate",
                              "record": False},
                      "copy": {"width": 1}}}
        if name in ROTOR:
            # I = rho V (r_out^2 + r_in^2) / 2 for an annulus about its own axis, in world units
            I_rotor += DENSITY * V * (nm(ro) ** 2 + nm(ri) ** 2) / 2.0
        # THE CYCLE'S ORDER IS NOT PER SET. Every set's strain, then every set's scatter into the ONE
        # grid, then ONE grid solve, then every set's gather -- the eye scene's schedule. The first
        # draft of this script interleaved strain/scatter/gather per set and never named
        # `mpm_grid_update` at all, and the engine ran it without a word: mass and momentum were
        # deposited on the grid forty times a frame and never divided by mass or read back, so the
        # rotor stood still under a torque the log reported as delivered (record step 0003).
        strains.append({"op": "mpm_strain", "at": name + "_pt", "implementation": "warp"})
        scatters.append({"op": "mpm_scatter", "at": name + "_pt", "to": "mpm_grid", "drag": LOAD_DRAG,
                         "a_max": 200.0, "implementation": "warp", "polar": "higham"})
        gathers.append({"op": "mpm_gather", "at": name + "_pt", "from": "mpm_grid", "vmax": 1.0e9,
                        "wall_damp": 0.9, "implementation": "warp"})

    # ---- the stator units: a set of eleven elements on a ring, drawn as spheres -------------------
    phase0 = math.radians(9.6) if cj else 0.0            # the first MotB centroid of 9HMF (step 0010)
    th = [phase0 + 2.0 * math.pi * i / N_STATOR for i in range(N_STATOR)]
    stator_pts = [[round(0.5 + nm(STATOR_R_NM) * math.cos(a), 6),
                   round(0.5 + nm(STATOR_R_NM) * math.sin(a), 6),
                   round(Z_MEMBRANE + nm(STATOR_Z_NM), 6)] for a in th]
    sets["stator_unit"] = {"n": N_STATOR, "start": stator_pts}
    if emerge:
        # THE STATOR CARRIES STATE: the stretch of its linkage `x` (radians) that stator_step feeds
        # and stator_contact turns into torque, and the ions it has passed (signed transits).
        sets["stator_unit"]["state"] = {
            "pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"},
            "x": {"width": 1, "integration": "first_order", "unit": "1"},
            "ions": {"width": 1, "integration": "first_order", "unit": "count"}}
    seeds, scaf_plot = [], {}
    if ions:
        # THE CELL OWNS THE STATORS. `cell` is one element carrying the membrane potential `psi`
        # (e psi, sim energy per charge), the net protons pumped out and in, and the motor current
        # the stators' transits sum to; `stator_unit` becomes its child (the containment map pi the
        # two hierarchy operators walk), so its positions come from a seed rather than `start:`.
        sets["cell"] = {"n": 1, "start": [[0.5, 0.5, Z_MEMBRANE]], "state": {
            "pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"},
            "psi": {"width": 1, "integration": "first_order", "unit": "voltage"},
            "h_peri": {"width": 1, "integration": "first_order", "unit": "count"},
            "h_cyto": {"width": 1, "integration": "first_order", "unit": "count"},
            "j_motor": {"width": 1, "integration": "none", "unit": "current"}}}
        sets["stator_unit"] = {"parent": "cell", "per_parent": N_STATOR, "radius": round(nm(STATOR_R_NM), 6),
                               "state": {**sets["stator_unit"]["state"],
                                         "psi": {"width": 1, "integration": "none", "unit": "voltage"},
                                         "flux": {"width": 1, "integration": "none", "unit": "current"}}}
        import numpy as _np
        from plexus.paths import graphs_data_path
        _folder = os.path.join(graphs_data_path(), "shapes", "bfm_cj")
        os.makedirs(_folder, exist_ok=True)
        _origin = [0.5, 0.5, round(Z_MEMBRANE + nm(STATOR_Z_NM), 6)]
        _scale = 1.0 / (BOX_NM * 1e-9)
        _np.savez_compressed(os.path.join(_folder, "points.npz"),
                             stators=((_np.asarray(stator_pts) - _np.asarray(_origin)) / _scale).astype(_np.float32))
        with open(os.path.join(_folder, "PROVENANCE.md"), "w") as f:
            f.write(f"# points.npz\n- stators: {N_STATOR} stator positions, metres about the motor axis at the membrane, "
                    f"r {STATOR_R_NM} nm at the MotB angles of PDB 9HMF (record step 0010); written by tools/bfm_motor_spec.py --ions\n")
        seeds.append({"op": "cloud_seed", "at": "stator_unit", "cloud": "bfm_cj/stators", "origin": _origin, "scale": _scale})
        if membrane:
            # E1 -- THE MEMBRANE AS A SET. A hexagonal lattice of points on the inner-membrane plane
            # (world z Z_MEMBRANE), from just outside the MS-ring to near the box wall, 3 nm apart;
            # children of the cell so pi carries psi down (uniform, by physics -- see
            # stator_membrane) and the local currents up. Not material: it carries state and
            # place, no mass and no grid, so it cannot lock the rotor that passes through it.
            MEM_SPACING_NM, MEM_R_IN_NM, MEM_R_OUT_NM = 3.0, 16.0, 66.0
            pts = []
            a_ = MEM_SPACING_NM; hgt = a_ * math.sqrt(3.0) / 2.0
            ny = int(2 * MEM_R_OUT_NM / hgt) + 2
            for iy in range(-ny, ny + 1):
                y = iy * hgt; x0 = 0.5 * a_ if iy % 2 else 0.0
                for ix in range(-int(MEM_R_OUT_NM / a_) - 2, int(MEM_R_OUT_NM / a_) + 3):
                    x = x0 + ix * a_
                    r = math.hypot(x, y)
                    if MEM_R_IN_NM <= r <= MEM_R_OUT_NM:
                        pts.append((x * 1e-9, y * 1e-9, 0.0))
            mem_pts = _np.asarray(pts, _np.float32)
            _prev = dict(_np.load(os.path.join(_folder, "points.npz")))
            _prev["membrane"] = mem_pts
            _np.savez_compressed(os.path.join(_folder, "points.npz"), **_prev)
            with open(os.path.join(_folder, "PROVENANCE.md"), "a") as f:
                f.write(f"- membrane: {len(mem_pts)} points, a hexagonal lattice {MEM_SPACING_NM} nm apart on the inner-membrane "
                        f"plane, r {MEM_R_IN_NM}-{MEM_R_OUT_NM} nm about the axis; written by tools/bfm_motor_spec.py --membrane\n")
            _m_origin = [0.5, 0.5, Z_MEMBRANE]
            sets["membrane"] = {"parent": "cell", "per_parent": int(len(mem_pts)), "radius": round(nm(MEM_R_OUT_NM), 6),
                                "state": {"pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"},
                                          "psi": {"width": 1, "integration": "none", "unit": "voltage"},
                                          "j_in": {"width": 1, "integration": "none", "unit": "current"},
                                          "j_avg": {"width": 1, "integration": "none", "unit": "current"}}}
            seeds.append({"op": "cloud_seed", "at": "membrane", "cloud": "bfm_cj/membrane", "origin": _m_origin, "scale": _scale})
            print(f"  MEMBRANE: {len(mem_pts)} points, {MEM_SPACING_NM} nm hex lattice, r {MEM_R_IN_NM}-{MEM_R_OUT_NM} nm at world z {Z_MEMBRANE}; "
                  f"a stator's footprint (reach 6 nm) covers ~{math.pi * 36 / (a_ * a_ * math.sqrt(3) / 2):.0f} points")
    if with_scaffold and not scaffold_material:
        sys.path.insert(0, os.path.join(REPO, "tools"))
        from bfm_scaffold_spec import scaffold, LOOKS
        scaf_sets, scaf_seeds, scaf_plot, _ = scaffold(look, z_membrane=Z_MEMBRANE, box_nm=BOX_NM)
        sets.update(scaf_sets); seeds += scaf_seeds
    elif scaffold_material:
        # E2 -- THE SCAFFOLD IS MATERIAL. The six 9HMF proteins as MPM elastic bodies, filled from
        # the shape library's parts (isosurfaces of each protein's alpha carbons, cut by
        # tools/bfm_anatomy_from_structure.py --parts 9hmf), placed by the library's own frame:
        # `origin: Lipo` pins the lipoprotein ring's centroid at `at`, and every other part lands at
        # its measured offset from it, scaled from metres to the box. Each protein is its own set,
        # as the rotor's parts are, so the grid fuses what touches (PflA spokes into the PflB rim
        # and the PDZ disk, FliL and MotB into the spokes) and the colours stay per protein. The
        # rim, PflB, is anchored -- the peptidoglycan it hangs from is not in the box -- and the
        # rest hangs from it through the grid. The stators are held by MotB's own periplasmic
        # domain (`anchor` on stator_contact), so the reaction of their torque enters the scaffold
        # and the units ride on whatever it gives.
        import json as _json
        sys.path.insert(0, os.path.join(REPO, "tools"))
        from bfm_scaffold_spec import LOOKS, COLORS as SCAF_COLORS, Z_OFFSET_NM
        from plexus.paths import graphs_data_path
        _pj = _json.load(open(os.path.join(graphs_data_path(), "shapes", "9hmf", "parts.json")))
        _scale = 1.0 / (BOX_NM * 1e-9)                          # world units per metre
        _lipo_c = _pj["lipo"]["centroid"]                        # metres, in the scaffold's own frame (library keys are lowercase)
        _at = [0.5 + _lipo_c[0] * _scale, 0.5 + _lipo_c[1] * _scale, Z_MEMBRANE + nm(Z_OFFSET_NM) + _lipo_c[2] * _scale]
        _frame = {"shape": "9hmf", "origin": "lipo", "at": [round(v, 6) for v in _at], "scale": _scale}
        scaf_plot = {"compartment_sets": [], "surface": {}, "opacity": {}, "colors": {}}
        for part, col in SCAF_COLORS.items():
            V_world = float(_pj[part.lower()]["volume"]) * _scale ** 3
            n_pt = int(round(V_world / p_vol))
            counts[part] = n_pt
            sets[part] = {"n": 1, "frame": _frame,
                          "types": {part: {"count": 1, "material": "elastic", "youngs": YOUNGS, "density": DENSITY,
                                           "shape": f"mesh:9hmf/{part.lower()}", "place": "measured"}}}
            sets[part + "_pt"] = {
                "parent": part, "entity": "mpm_particle", "per_parent": n_pt,
                "radius": round(0.5 * max(_pj[part.lower()]["extent"]) * _scale, 6), "density": DENSITY, "particle_mass": p_mass,
                "state": {"pos": {"width": 3, "role": "coordinate", "integration": "second_order_coordinate", "boundary": "world"},
                          "vel": {"width": 3, "role": "rate", "integration": "second_order_rate", "record": False},
                          "copy": {"width": 1}}}
            # ITS OWN GRID. On the rotor's grid the scaffold FUSED to the rotor: the E-ring gap
            # (lipoprotein ring to MS-ring, 1.4 nm) and FliL/MotB to the FliG ring's top (~1 nm) are
            # inside the 3-cell stencil at dx 1.09 nm, so the anchored scaffold welded the rotor and
            # the motor stalled at one full step of stretch (E2's first run, 2026-09-23 20:45: ring
            # 0.0025 rad/s at frame 20). In the animal those gaps are lipid and water, and the only
            # mechanical link from scaffold to rotor is the stator units themselves -- which is what
            # a second MPM grid says: the six proteins scatter and gather on `mpm_grid_scaffold`,
            # share no node with the rotor, and touch it only through `stator_contact`.
            strains.append({"op": "mpm_strain", "at": part + "_pt", "implementation": "warp"})
            scatters.append({"op": "mpm_scatter", "at": part + "_pt", "to": "mpm_grid_scaffold", "drag": LOAD_DRAG,
                             "a_max": 200.0, "implementation": "warp", "polar": "higham"})
            gathers.append({"op": "mpm_gather", "at": part + "_pt", "from": "mpm_grid_scaffold", "vmax": 1.0e9,
                            "wall_damp": 0.9, "implementation": "warp"})
            scaf_plot["compartment_sets"].append(part + "_pt")
            scaf_plot["surface"][part + "_pt"] = ({"render": "dots", "point_size": 2.4} if look == "dots" else
                                                  {"render": "surface", "spacing": round(dx, 6), "blur": 1.5, "iso_frac": 0.3,
                                                   "smooth": 30, "specular": 0.0, "ambient": 0.32, "diffuse": 0.72})
            scaf_plot["opacity"][part + "_pt"] = 1.0
            scaf_plot["colors"][part + "_pt"] = col
        print(f"  SCAFFOLD AS MATERIAL: frame at {_frame['at']} (Lipo's centroid), scale {_scale:.4g} world/m; points "
              + ", ".join(f"{k} {counts[k]:,}" for k in SCAF_COLORS) + f"; total {sum(counts[k] for k in SCAF_COLORS):,}; PflB anchored; stators anchored in MotB_pt")
    else:
        from bfm_scaffold_spec import LOOKS

    # ---- the torque the load and the target rate demand, and what one unit must then push --------
    omega = 2.0 * math.pi * TARGET_TURNS_PER_S            # rad/s sim
    torque = LOAD_DRAG * I_rotor * omega                  # sim force x world: T = k_d I omega
    r_push = nm(CJ_R_PUSH_NM if cj else 20.0)             # the stator's lever arm on the C-ring rim
    F_unit = torque / (N_STATOR * r_push)
    # units: the animal's 100 Hz is this rate, and one unit's 7.3 pN is this force
    time_s = omega / (2.0 * math.pi * 100.0)              # seconds per sim second: omega_sim <-> 2 pi 100 Hz
    force_nN = (F_UNIT_PN * 1e-3) / F_unit
    if emerge:
        # UNITS DO NOT FOLLOW THE LOAD. The force unit above is tied to the constant push the design
        # torque asks of one unit at THIS drag, which made a load sweep a sweep of the force unit --
        # and with it of every stiffness declared in sim units (Young's modulus, the linkage). The
        # stepping stator has no design force: its units are anchored once, at the reference load of
        # 50/s (C02, D01), and every sweep run shares them.
        force_nN = (F_UNIT_PN * 1e-3) / ((50.0 * I_rotor * omega) / (N_STATOR * r_push))
    length_um = BOX_NM * 1e-3

    # ---- the stepping stator, in the run's own units ---------------------------------------------
    # Energies are sim force x world length. The free energy of one transit is q e Delta-psi: two
    # protons across the 150 mV of the inner membrane are 48 pN nm (1 eV = 160.2 pN nm); kT is
    # 4.1 pN nm at 23 C. Stall torque per unit is then eps / delta = 48 / (2 pi / 26) = 199 pN nm,
    # which is 7.3 pN at 27 nm -- the constant-force baseline is this machine at stall. The
    # linkage stiffness is set so that stall stretches it by one step, kappa = eps / delta^2 =
    # 823 pN nm/rad; the ion-binding rate k_c and the stroke attempt rate r0 set the zero-load
    # speed, ~300 Hz x 26 steps = 7,800 steps/s (Chen & Berg 2000's intercept), and are the
    # baseline's two fitted rates; theta = 0.1 is Meacci & Tu's transition-state position.
    E_unit = (force_nN * 1e3) * (length_um * 1e3)        # pN nm per sim energy unit
    KT_PNNM, Q_IONS, DPSI_MV = 4.1, 2.0, 150.0
    eps_pnnm = Q_IONS * DPSI_MV * 1e-3 * 160.2
    N_STEPS = 26
    delta = 2.0 * math.pi / N_STEPS
    kappa_pnnm = eps_pnnm / delta ** 2
    R0_HZ, KC_HZ, THETA = 11000.0, 10000.0, 0.1
    stepper = {"eps": eps_pnnm / E_unit, "kT": KT_PNNM / E_unit, "delta": delta,
               "kappa": kappa_pnnm / E_unit, "r0": R0_HZ * time_s, "k_c": KC_HZ * time_s, "theta": THETA}

    # ---- the ion economy, in the run's own units ----------------------------------------------
    # psi is carried as e psi in sim energy per charge. The membrane is a capacitor of 1 uF/cm^2
    # over a C. jejuni's ~3 um^2, C/e = 1.87e5 charges per volt; respiration pumps up to J_MAX
    # protons/s and stalls at 200 mV; the leak is set so the cell without its motor rests at
    # 150 mV (g psi = J_pump(150 mV)). At 100 Hz the 17 units draw 88,000 protons/s, a few per
    # cent of the pump: the potential should sag by ~7 mV when the motor runs, and recover if it
    # stops -- the coupling the fixed potential could not show.
    E_V_PNNM = 160.2                                    # pN nm per eV (one elementary charge across one volt)
    PSI0_MV, PSI_REV_MV, J_MAX_HZ, CELL_AREA_UM2, C_M_UF_CM2 = 150.0, 200.0, 2.0e6, 3.0, 1.0
    SHUNT_X, SHUNT_FROM = 0.0, 0
    for i, tok in enumerate(sys.argv):                   # `--pump X`: respiration's ceiling, protons/s (the pump sweep)
        if tok == "--pump":
            J_MAX_HZ = float(sys.argv[i + 1])
        if tok == "--shunt":                             # `--shunt X N`: a resistor of X times the rest leak, from frame N
            SHUNT_X, SHUNT_FROM = float(sys.argv[i + 1]), int(sys.argv[i + 2])
    psi_sim = lambda mv: mv * 1e-3 * E_V_PNNM / E_unit            # mV -> sim e psi
    charges_per_volt = C_M_UF_CM2 * 1e-6 * CELL_AREA_UM2 * 1e-8 / 1.602e-19
    cap_sim = charges_per_volt / E_V_PNNM * E_unit               # charges per sim psi unit
    pump_max_sim = J_MAX_HZ * time_s
    leak_sim = (1.0 - PSI0_MV / PSI_REV_MV) * pump_max_sim / psi_sim(PSI0_MV)
    mv_per_sim_psi = E_unit / E_V_PNNM * 1e3

    if emerge:
        ops_frame = ([
            {"op": "membrane_potential", "at": "cell", "capacitance": cap_sim, "pump_max": pump_max_sim,
             "pump_rev": psi_sim(PSI_REV_MV), "leak": leak_sim, "psi0": psi_sim(PSI0_MV), "h0": [0.0, 0.0],
             "flux": "j_motor", **({"shunt": SHUNT_X * leak_sim, "shunt_from": SHUNT_FROM} if SHUNT_X > 0 else {})},
            {"op": "broadcast", "at": "stator_unit", "block": "psi", "source": "psi"},
        ] if ions else []) + [
            {"op": "stator_contact", "at": PUSHED + "_pt", "stator": "stator_unit", "kappa": stepper["kappa"],
             "reach": round(nm(REACH_NM), 6), "axis": [0.0, 0.0, 1.0], "sign": 1.0,
             **({"anchor": "MotB_pt", "anchor_reach": round(nm(8.0), 6)} if scaffold_material else {}),
             # THE LOAD'S ROTATIONAL DRAG, zeta = k_d I, in sim torque x time per radian: the
             # fluctuation partner of the placeholder drag on every rotor point.
             **({"kT": stepper["kT"], "zeta_rot": LOAD_DRAG * I_rotor, "seed": 0} if thermal else {})},
            {"op": "stator_step", "at": "stator_unit", "r0": stepper["r0"], "k_c": stepper["k_c"],
             "eps": stepper["eps"], "kT": stepper["kT"], "delta": delta, "theta": THETA, "ions": Q_IONS,
             "stochastic": stochastic, "seed": 0,
             # E3: alpha = 1 / (4 pi D r c0) in s per proton, times time_s so that nu (per sim s) gives the
             # same dimensionless depletion: D 1e9 nm^2/s (buffered protons), c0 1 mM = 6.0e-4 per nm^3,
             # r 2 nm, the mouth of the channel.
             **({"pool_alpha": (1.0 / (4.0 * math.pi * 1.0e9 * 2.0 * 6.02e-4)) / time_s} if protons else {})},
        ] + ([{"op": "aggregate", "at": "stator_unit", "block": "flux", "target": "j_motor", "reduce": "sum"}] if ions else []) + ([
            {"op": "broadcast", "at": "membrane", "block": "psi", "source": "psi"},
            {"op": "stator_membrane", "at": "membrane", "stator": "stator_unit", "reach": round(nm(6.0), 6), "block": "j_in",
             "tau": 50.0, "avg": "j_avg"},
        ] if membrane else []) + [
            {"op": "mpm_anchor", "at": "lp_ring_pt", "k": 4000.0, "applies_to": "substrate"},
        ] + ([{"op": "mpm_anchor", "at": "PflB_pt", "k": 4000.0, "applies_to": "substrate"}] if scaffold_material else [])
    else:
        ops_frame = [
            {"op": "stator_push", "at": PUSHED + "_pt", "stator": "stator_unit", "force": F_unit,
             "reach": round(nm(REACH_NM), 6), "axis": [0.0, 0.0, 1.0], "sign": 1.0},
            {"op": "mpm_anchor", "at": "lp_ring_pt", "k": 4000.0, "applies_to": "substrate"},
        ]
    ops_sub = strains + scatters + [{"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": 0.9,
                                     "implementation": "warp"}] + ([{"op": "mpm_grid_update", "at": "mpm_grid_scaffold",
                                     "wall_damp": 0.9, "implementation": "warp"}] if scaffold_material else []) + gathers
    _vals = {sys.argv[i + 1] for i, t in enumerate(sys.argv) if t in ("--drag", "--frames", "--pump", "--grid", "--shunt", "--youngs", "--substep") and i + 1 < len(sys.argv)}
    _vals |= {sys.argv[i + 2] for i, t in enumerate(sys.argv) if t == "--shunt" and i + 2 < len(sys.argv)}
    _vals |= {sys.argv[i + 1] for i, t in enumerate(sys.argv) if t == "--color-stators" and i + 1 < len(sys.argv)}
    args_ = [x for x in sys.argv[1:] if not x.startswith("--") and x not in _vals]
    name_spec = args_[0] if args_ else "bfm_a03_rotor_stator"
    spec = {
        "general": {"name": name_spec, "seed": 0, "n_frames": N_FRAMES, "dt": FRAME_DT,
                    "boundary": "wall", "dim": 3, "world": [1.0, 1.0, 1.0], "save_data": True,
                    "record_cap": 301,
                    "units": {"length_um": length_um, "time_s": time_s, "force_nN": force_nN}},
        "sets": sets,
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": N_GRID},
                   **({"mpm_grid_scaffold": {"frame": "mpm_grid", "n_grid": N_GRID}} if scaffold_material else {})},
        **({"seed": seeds} if seeds else {}),
        "operators": ops_frame + ops_sub,
        "schedule": [o["op"] for o in ops_frame] + [{"substep_dt": SUBSTEP, "steps": [o["op"] for o in ops_sub]}],
        "plotting": {
            "renderer": "vtk_points", "background": "black", "up_axis": 2, "box_frame": True,
            "hide_sets": list(PARTS) + ["stator_unit"],
            # ALL SEVEN PARTS, NOT THE SUBJECT ALONE. `render_3d: dots` draws one set -- the subject --
            # and A02's movie showed the C-ring under a black void where the MS-ring, rod, bushing and
            # hook were turning unseen. `compartments` with `compartment_sets` draws each named set
            # as its own cloud in its own colour (live_movie._skins_build, `render: dots` per set).
            "render_3d": "compartments",
            "compartment_sets": [n + "_pt" for n in PARTS] + scaf_plot.get("compartment_sets", []) + (["membrane"] if membrane else []),
            # THE ROTOR'S LOOK FOLLOWS THE SCAFFOLD'S: dots on black, or the cryo figure -- a matte
            # surface contoured at the grid's own spacing with a black silhouette on white.
            "surface": {**({"membrane": {"render": "dots", "point_size": 13.0}} if membrane else {}),
                        **{n + "_pt": ({"render": "dots", "point_size": 2.4} if look == "dots" else
                                       {"render": "surface", "spacing": round(dx, 6), "blur": 1.5, "iso_frac": 0.3,
                                        "smooth": 30, "specular": 0.0, "ambient": 0.32, "diffuse": 0.72})
                           for n in PARTS}, **scaf_plot.get("surface", {})},
            # WITH THE MEMBRANE ON, the scaffold goes translucent (0.3): the sheet's hot spots sit
            # under its lobes (r 21-32 nm, the stator ring), and an opaque scaffold hides them.
            # ... the roof (PflA/B/CD, the lipoprotein) at 0.12 and the stator bodies (FliL, MotB) at
            # 0.2, because the spots sit INSIDE the stator bodies (the channel is in MotA/B) and are
            # seen only through them: step 0073, where every skin was also 5 nm too fat until the
            # renderer's bulk-density fix of 2026-09-24.
            "opacity": {**{n + "_pt": 1.0 for n in PARTS},
                        **{k: ((0.2 if k in ("FliL", "MotB") else 0.12) if membrane else v)
                           for k, v in scaf_plot.get("opacity", {}).items()},
                        **({"membrane": 1.0} if membrane else {})},
            "colors": {**{"c_posts_a_pt": [0.88, 0.22, 0.2], "c_posts_b_pt": [0.92, 0.9, 0.82],
                          "c_flig_pt": [0.62, 0.35, 0.82], "ms_ring_pt": [0.28, 0.52, 0.9],
                          "rod_pt": [0.32, 0.76, 0.42], "lp_ring_pt": [0.6, 0.6, 0.62],
                          "hook_pt": [0.95, 0.62, 0.2]}, **scaf_plot.get("colors", {})},
            # WITH THE MEMBRANE ON, the unit markers shrink to 2 nm at half opacity: at their 4.5 nm
            # they sit exactly on the sheet's hot spots (r 21-32 nm) and hide them (step 0073).
            "spheres": {"set": "stator_unit", "radius": round(nm(2.0 if membrane else STATOR_RADIUS_NM), 6),
                        **({"opacity": 0.5} if membrane else {}),
                        "color": "#e8b04a", "specular": 0.0, "ambient": 0.25, "diffuse": 0.8},
            **LOOKS[look],
            # `--cut`: LOOK INSIDE. `near_side: true` keeps, on every surface and sphere, the faces
            # behind the plane through the body's centre normal to the view, so the camera looks
            # into the motor's section; the cut follows the camera.
            **({"near_side": True, "near_side_centre": [0.5, 0.5, 0.5]} if cut else {}),
            # `--color-stators x|psi`: THE 17 BALLS COLOURED BY THEIR OWN STATE (plotting.color_block):
            # the stretch in steps (the torque each unit carries) or the potential each unit sees.
            **({"color_block": {**({"stator_unit": ({"block": "x", "range": [0.0, 1.0], "cmap": "plasma"} if color_stators == "x"
                                                 else {"block": "psi", "unit": "voltage", "range": [60.0, 160.0], "cmap": "viridis"})}
                                   if color_stators else {}),
                                # THE MEMBRANE COLOURED BY THE CURRENT ENTERING IT (fA per point): the 17 spots of
                                # inflow under the stators are the picture of where the circuit meets the sheet.
                                # plasma, not magma: magma's zero is black, and a membrane carrying no current yet
                                # vanished into the black ground (E1's first still).
                                # the range from the data (no `range:`), the running mean when the strokes are random
                                **({"membrane": {"block": ("j_avg" if stochastic else "j_in"), "unit": "current", "cmap": "plasma"}} if membrane else {})}}
               if (color_stators or membrane) else {}),
            # THE ION LEVELS OVER TIME, drawn on the movie: the cell's potential in mV, the net
            # protons pumped out, and the stators' stretch in steps (`block:<set>:<block>` curves).
            # THE RATE AXIS FITS THE RUN: the plateau speed at this load is ~100 Hz x 50/drag, capped
            # at the 300 Hz zero-load speed, and the axis tops out 25% above it (the human found a
            # 320 Hz axis far too large for a 100 Hz motor).
            **({"curve": [
                {"quantity": "rate:c_flig_pt", "factor": 1.0 / time_s, "ymin": 0.0,
                 "ymax": round(1.25 * min(300.0, 100.0 * 50.0 / LOAD_DRAG), 0), "ylabel": "C-ring rotation rate (Hz)"},
                {"quantity": "jacobian:rod_pt", "ymin": 0.95, "ymax": 1.05, "ylabel": "rod volume / rest (mean det F)"},
                {"quantity": "strain:rod_pt", "ymin": 0.0, "ymax": 0.02, "ylabel": "rod Green strain, rms (rotation-free)"}]}
               if diag_curves and not cut else
               {"curve": [
                {"quantity": "rate:c_flig_pt", "factor": 1.0 / time_s, "ymin": 0.0,
                 "ymax": round(1.25 * min(300.0, 100.0 * 50.0 / LOAD_DRAG), 0), "ylabel": "C-ring rotation rate (Hz)"},
                {"quantity": "block:cell:psi", "unit": "voltage", "ymin": 120.0, "ymax": 160.0,
                 "ylabel": "membrane potential psi"},
                {"quantity": "strain:rod_pt", "ymin": 0.0, "ymax": 0.02, "ylabel": "rod Green strain, rms (rotation-free)"}]}
               if cut and ions else
               {"curve": [
                {"quantity": f"jacobian:{n}_pt", "ymin": 0.9, "ymax": 1.1,
                 "ylabel": f"{n} volume / rest (mean det F)"} for n in ("c_flig", "rod", "hook")]} if volume_curves else
               {"curve": [
                # RANGES THAT SHOW THE CHANGE: D03's 4.4 mV sag on a 0-200 mV axis and its stretch on
                # 0-1.2 read as flat lines (the human's word), so the panels frame what moves.
                # THE THREE PANELS OF AN ION RUN (the human, 2026-09-24): the potential, the motor's
                # current (the 0.5 ms mean of the summed unit fluxes, in fA -- 884 protons per turn
                # here, 17 units x 26 steps x 2 protons, so 11 fA at 80 Hz), and the rotation rate
                # in Hz. Short titles at the header's font size; the unit is appended by the renderer.
                {"quantity": "block:cell:psi", "unit": "voltage", "ymin": 120.0, "ymax": 160.0,
                 "ylabel": "potential"},
                {"quantity": "block:cell:j_motor", "unit": "current", "smooth": 50.0, "ymin": 0.0, "ymax": 20.0,
                 "ylabel": "motor current"},
                {"quantity": "rate:c_flig_pt", "factor": 1.0 / time_s, "ymin": 0.0, "ymax": 320.0,
                 "ylabel": "rotation (Hz)"}]} if ions else {}),
            "subject": "c_flig_pt", "keep_stills": True, "stills": 8, "max_frames": 300,
            # AN 8 s MOVIE, FIXED: real time off and the framerate set from the frame count, 300 / 8 s.
            # The renderer writes real time once units are declared, and this motor's real time is
            # 7.5 ms a run -- a 0.01 s film (record steps 0003-0009 played as stills).
            "real_time": False, "duration_s": 10.0,
        },
    }
    for _cv in (spec.get("plotting", {}).get("curve") or []):
        _cv.setdefault("font_size", 21); _cv.setdefault("tick_font_size", 16)
    out = os.path.join(REPO, "config", "bacterium", name_spec + ".yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=None, width=100)

    c = math.sqrt(YOUNGS / DENSITY)
    print(f"wrote {os.path.relpath(out, REPO)}  (species {'C. jejuni' if cj else 'E. coli'}, "
          f"{'with' if with_scaffold else 'no'} scaffold, look {look})")
    print(f"  box {BOX_NM:g} nm, n_grid {N_GRID}, dx {dx * BOX_NM:.3f} nm, p_vol {p_vol:.3e} world^3 "
          f"({p_vol * BOX_NM ** 3:.3f} nm^3), particle_mass {p_mass:.3e}")
    print("  counts: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()) + f"  total {sum(counts.values()):,}")
    print(f"  sound speed {c:.1f} world/s, CFL c dt/dx = {c * SUBSTEP / dx:.2f} at substep {SUBSTEP:g}")
    print(f"  rotor moment of inertia I = {I_rotor:.4e} (density x world^5)")
    print(f"  design: omega {omega:.4f} rad/s ({TARGET_TURNS_PER_S:g} turn/s), load drag {LOAD_DRAG:g} /s "
          f"-> torque {torque:.4e}, force per unit {F_unit:.4e} at lever arm {r_push:.4f} "
          f"({N_STATOR} units)")
    print(f"  spin-up time 1/k_d = {1.0 / LOAD_DRAG:.3f} s; rim speed {omega * nm(22.5):.3f} world/s "
          f"= {omega * nm(22.5) / c * 100:.2f}% of the sound speed")
    if protons:
        alpha = 1.0 / (4.0 * math.pi * 1.0e9 * 2.0 * 6.02e-4)
        print(f"  PROTON POOL (E3): alpha {alpha:.3e} s per proton; at 5,200 protons/s per unit the depletion at the mouth is "
              f"{alpha * 5200:.2e} -- the spatial term, as a number; a grid field cannot hold it (a 10 nm^3 voxel holds 0.006 protons)")
    if thermal:
        zeta = LOAD_DRAG * I_rotor
        D_th = stepper["kT"] / zeta
        print(f"  THERMAL: random torque of variance 2 zeta kT / dt with zeta = k_d I = {zeta:.4e} (sim torque s/rad), "
              f"kT {stepper['kT']:.3e} -> angle diffusion D = kT/zeta = {D_th:.3e} rad^2 per sim s "
              f"({D_th / time_s:.3f} rad^2/s real); over the run's {N_FRAMES * FRAME_DT:.2f} sim s the rms angle wander is "
              f"{math.degrees(math.sqrt(2 * D_th * N_FRAMES * FRAME_DT)):.1f} deg against {N_FRAMES * FRAME_DT * 180:.0f} deg turned; "
              f"mean rate unchanged is the prediction")
    if ions and SHUNT_X > 0:
        g_mv = (1.0 - PSI0_MV / PSI_REV_MV) * J_MAX_HZ / PSI0_MV            # the rest leak, protons/s per mV
        slope = g_mv * (1.0 + SHUNT_X) + J_MAX_HZ / PSI_REV_MV + N_STATOR * Q_IONS * 26 * 100 / PSI0_MV
        psi_ss = J_MAX_HZ / slope
        print(f"  SHUNT: {SHUNT_X:g} x the rest leak from frame {SHUNT_FROM}; with it the potential must settle at "
              f"~{psi_ss:.0f} mV (time constant ~{charges_per_volt * 1e-3 / slope * 1e3:.1f} ms) and the motor at ~{100.0 * psi_ss / PSI0_MV:.0f} Hz")
    if ions:
        print(f"  ION ECONOMY: psi0 {PSI0_MV:g} mV (sim {psi_sim(PSI0_MV):.4e}), pump {J_MAX_HZ:g}/s stalling at {PSI_REV_MV:g} mV, "
              f"C/e {charges_per_volt:.3e} charges/V ({cap_sim:.3e} per sim psi), leak {leak_sim:.3e}/sim s per sim psi; "
              f"motor at 100 Hz draws {N_STATOR * Q_IONS * 26 * 100:.0f} protons/s = {N_STATOR * Q_IONS * 26 * 100 / (J_MAX_HZ * (1 - PSI0_MV / PSI_REV_MV)) * 100:.1f}% of the pump at rest")
    if emerge:
        print(f"  STEPPING STATOR ({'stochastic' if stochastic else 'mean rate'}): eps {eps_pnnm:.1f} pN nm "
              f"({Q_IONS:g} ions x {DPSI_MV:g} mV), kT {KT_PNNM} pN nm, {N_STEPS} steps/turn (delta {delta:.4f} rad), "
              f"kappa {kappa_pnnm:.0f} pN nm/rad, r0 {R0_HZ:g}/s, k_c {KC_HZ:g}/s, theta {THETA}")
        print(f"    -> stall torque per unit eps/delta = {eps_pnnm / delta:.1f} pN nm, x {N_STATOR} = "
              f"{N_STATOR * eps_pnnm / delta:.0f} pN nm; zero-load {(R0_HZ * (math.exp(THETA * eps_pnnm / KT_PNNM) - math.exp(-(1 - THETA) * eps_pnnm / KT_PNNM))) / (1.0 + R0_HZ * (math.exp(THETA * eps_pnnm / KT_PNNM) + math.exp(-(1 - THETA) * eps_pnnm / KT_PNNM)) / KC_HZ) / N_STEPS:.0f} Hz (the chain's mean); "
              f"sim: eps {stepper['eps']:.4e}, kappa {stepper['kappa']:.4e}, r0 {stepper['r0']:.2f}, k_c {stepper['k_c']:.2f} per sim s")
    print(f"  units: 1 length = {length_um:g} um, 1 time = {time_s:.4e} s, 1 force = {force_nN:.4e} nN "
          f"-> torque {torque * force_nN * 1e3 * length_um * 1e3:.0f} pN nm, "
          f"E = {YOUNGS * 1e3 * force_nN / length_um ** 2:.3e} Pa")


if __name__ == "__main__":
    main()
