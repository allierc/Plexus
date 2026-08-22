#!/usr/bin/env python
"""Move okuda's operator files into `src/plexus/operators/` as THEMATIC MODULES, verbatim.

WHY A SCRIPT AND NOT AN EDIT. The promotion's gate is bit-equality, so the code that lands in core
has to be the code that ran in okuda -- character for character inside every function body. A hand
move across 3,800 lines is where a stray edit hides, and it would show up as a digest mismatch with
no clue which of six files caused it. This script does exactly three things to the text it moves:

    1. hoists the top-of-file imports of each source into one deduplicated block;
    2. rewrites the imports that named a sibling by bare module name -- `from mesh_ops import ...`,
       `from topology_ops import ...` -- to the absolute path of their new home, or drops them
       where the two files are now ONE file;
    3. writes a banner above each source's body carrying that source's own module docstring.

Nothing inside a function body is touched. `tools/group_operators.py --verify` re-imports both the
old and the new module and compares every registered class's source text.

WHY THESE GROUPS. `src/plexus/operators/` is 45 files holding 43 names -- a directory in which the
two implementations of `cell_mechanics` cannot be read side by side because they are never on the
screen together. The okuda side is the same problem at 16 files. The grouping is by MECHANISM, which
is the axis a reader actually searches on:

    plexus/models/topology.py       the half-edge ALGORITHMS (rings, split, divide) -- not operators
    plexus/operators/vertex_ops.py  the 3D vertex model: seed, geometry, mechanics, divide, die,
                                    T1, and the monolayer implementation of the same contract
    plexus/operators/diffusion_reaction.py
                                    chemistry on the cell graph: seed, diffuse, react, the shape
                                    couplings, the interface terms

THE SHIM IS NOT OPTIONAL. Thirty files import `mesh_ops` / `chem_ops` / `topology_ops` by bare name
-- `run_one.py`, `instrument.py`, `vtk_render.py`, `metrics.py`, twenty archive and analysis scripts
-- and the campaign is still running. Each moved file is replaced by a re-export that keeps every one
of them working, PRIVATE NAMES INCLUDED: `t1_ops` calls `_mesh_ops._carry_face_state` and
`analyze_forces` reaches for `_engine_owns_clock`, so a shim that exported only the public surface
would break them at the first T1.

    python tools/group_operators.py --dry      what would move, and the rewrites
    python tools/group_operators.py            do it
    python tools/group_operators.py --verify   every registered class's source is unchanged
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPS = os.path.join(ROOT, "discovery_okuda", "ops")

# ------------------------------------------------------------------------------------- the groups
GROUPS = [
    dict(
        target="src/plexus/models/topology.py",
        sources=["topology_ops.py"],
        module="plexus.models.topology",
        doc='''"""Half-edge ALGORITHMS on the closed 3D surface: rings, edge split, face division.

Moved out of `discovery_okuda/ops/topology_ops.py`, and moved to `models/` rather than `operators/`
because none of it is an operator: these are pure functions over the flat table that
`plexus.models.mesh.MeshTable` holds, and `cell_divide` / `edge_flip` are the operators that drive
them. Keeping them beside the table is what makes the table's central invariant checkable in one
place -- `rings_from_flat_3d` walks `E_face` IN TABLE ORDER and never sorts, so the ordering of the
flat table is the geometry.
"""''',
    ),
    dict(
        target="src/plexus/operators/vertex_ops.py",
        sources=["mesh_ops.py", "t1_ops.py", "monolayer_ops.py"],
        module="plexus.operators.vertex_ops",
        doc='''"""The 3D vertex model, as one module: seed, geometry, mechanics, growth, division, death, T1.

    seed_mesh (alias mesh_seed)  build the closed spherical half-edge surface, once, at frame 0
    cell_mechanics               the AVM shape energy -- `default` (3D AVM) and `monolayer`
    cell_divide                  a septum through a face -> two daughters   (default/doubler/timer)
    cell_die                     shrink to a triangle, then extrude
    edge_flip                    the T1 / reversible network reconnection
    topo_record                  one recorded frame of topology per tick

THE THREE FILES THAT BECAME THIS ONE were `mesh_ops.py` (the operators), `t1_ops.py` (the T1, which
imported `mesh_ops` for the shared carry and the clock helper) and `monolayer_ops.py` (the second
implementation of the `cell_mechanics` contract, which imported `mesh_ops` for `face_geometry_3d`).
Every cross-import between them is now an ordinary reference inside one file, and the two
implementations of `cell_mechanics` can finally be read one after the other.

MODEL PROVENANCE. Okuda, S., Inoue, Y., Eiraku, M., Sasai, Y., Adachi, T. (2013)
Biomech. Model. Mechanobiol. 12(4):627 -- the reversible network reconnection `edge_flip`
implements; Okuda, S., Miura, T., Inoue, Y., Adachi, T., Eiraku, M. (2018) Sci. Rep. 8:2386;
ancestor Honda, H., Tanemura, M., Nagai, T. (2004) J. Theor. Biol. 226(4):439. The shape energy is
Farhadifar, R. et al. (2007) Curr. Biol. 17:2095. The mesh representation is Tyssue
(github.com/DamCB/tyssue).
"""''',
    ),
    dict(
        target="src/plexus/operators/diffusion_reaction.py",
        sources=["chem_ops.py", "shape_chem_ops.py", "shape_probe_ops.py"],
        module="plexus.operators.diffusion_reaction",
        doc='''"""Reaction-diffusion ON THE CELL GRAPH, and the two couplings between chemistry and shape.

    seed_cell_chem (alias cell_chem_seed)  the initial morphogen field
    cell_chem_diffuse                      graph_laplacian | interface_weighted
    cell_chem_react                        gray_scott | brusselator | gierer_meinhardt
    cell_neighbours                        the cell adjacency the Laplacian runs on
    cell_geometry                          per-cell area / perimeter / centroid / volume
    cell_grow                              default | balance | sizer | timer
    cell_chem_from_shape                   shape -> chemistry: apical_area | curvature | pressure | tension
    cell_shape_probe                       aspect | shape_index, published for a discriminator
    interface_tension                      a purse-string line tension on the red/white interface
    interface_push                         and the term that is NOT physics -- kept separate on purpose

THE DIFFUSION IS NOT ON A GRID. The cells are the nodes, `cell_neighbours` is the graph, and the
Laplacian is over shared faces -- so the domain grows and rewires as the tissue divides, which a
fixed lattice cannot do. That is why these are `set=cell` operators rather than `field` ones.

INTERFACE_TENSION AND INTERFACE_PUSH ARE TWO OPERATORS AND MUST STAY TWO. They were one,
`rd_interface_tension`, carrying `K_purse * sum l_e` (ordinary vertex-model physics) MINUS
`K_extrude * sum a*r` (an energy that falls as red cells move outward -- it pays the tissue to
produce the morphology the search was looking for). One name over both cost four campaign rounds of
verdicts about a term that measured 0.0 in all 78 specs that ever carried it. See OKUDA_PROMOTION.md.
"""''',
    ),
    dict(
        target="src/plexus/operators/junction_ops.py",
        sources=["junction_ops.py", "medioapical_ops.py"],
        module="plexus.operators.junction_ops",
        doc='''"""Myosin, and the two places a cell can put it: on its junctions or across its apex.

    junction_myosin      per-junction myosin -- `default` (one pool) and `two_pool`
    junction_sync        keep the per-junction store aligned with the half-edge table
    medioapical_myosin   the APICAL pool: an areal density on the face, not on its edges
    cytokinetic_ring     the contractile ring a dividing cell closes on its own septum

WHY THE TWO FILES ARE ONE. `medioapical_ops` imported `junction_ops` as `JO` and called six of its
private helpers -- `_live_edges`, `_lookup`, `_scatter_full`, `edge_tension` -- because the two-pool
model is not a different mechanism, it is the SAME junction bookkeeping with a second reservoir on
the face. Splitting them across files meant the shared half of the model was private to one of them.

WHICH POOL A SPEC GETS is chosen by `tissue.py`'s `myo_model`, through the `implementation` axis on
one contract, which is the paper's rule and not a switch statement: `junction_myosin[default]` and
`junction_myosin[two_pool]` are two bodies under one name.
"""''',
    ),
    dict(
        target="src/plexus/operators/ecm_ops.py",
        sources=["ecm_ops.py", "block_ops.py"],
        module="plexus.operators.ecm_ops",
        doc='''"""The extracellular matrix as MPM material, and the stiff blocks that confine it.

    ecm_seed        the box MINUS a cavity, as aligned fibres, once at frame 0
    ecm_stress      |J-1| (or the deviatoric / von Mises variant), banded, so the front is visible
    ecm_from_cell   the epithelium's surface as a moving boundary -- `replay` and `sphere`
    cell_exclude    the hard backstop: no matrix particle inside the lumen
    block_seed      two slabs beyond a free gap, as a SECOND MPM set ~130x stiffer
    block_stress    the block's own strain, at its own full scale

`ecm_seed` AND `block_seed` ARE NOT REDUNDANT and are not merged: one fills the complement of a
cavity with aligned fibres, the other fills two slabs with a jittered lattice. Same family, same
module, different geometry. `ecm_stress` and `block_stress` ARE the same body and are marked for
merging -- see OKUDA_PROMOTION.md; what keeps them apart today is a MODULE-LEVEL history list per
set, and moving that onto the Level is what lets one operator serve both.
"""''',
    ),
    dict(
        target="src/plexus/operators/membrane_ops.py",
        sources=["membrane_ops.py", "integrin_ops.py"],
        module="plexus.operators.membrane_ops",
        exclude=["MPMTissueBoundary", "BasementMembraneContinuumStrain"],
        doc='''"""The basement membrane: a crosslinked shell between the epithelium and the stroma.

    bm_seed / bm_bond / bm_crosslink / bm_unbond / bm_remodel / bm_secrete
                        the sheet, its network, and how the network turns over
    bm_contact / bm_repel                   it does not pass through what it rests on
    adhesion_seed / adhesion_pull / adhesion_turnover
                        the sheet's grip on the epithelium, and how that grip renews
    integrin_adhesion   MEMBRANE -> EPITHELIUM: each particle is pulled back to the angular
                        position it was seeded on, so a surface whose radius triples stretches its
                        bonds by ~R -- the loading a real basement membrane feels under growth
    integrin_seed / integrin_pull / integrin_track
                        MATRIX -> MEMBRANE: fibres seeded outward, each bound at its tip to the
                        nearest membrane particle, with the cell end prescribed

THE TWO INTEGRIN FAMILIES ARE ONE HOP APART IN THE SAME CHAIN AND ARE NOT THE SAME THING. The shared
prefix is what invites the confusion, so they are here together with that sentence at the top rather
than in two files where nobody compares them.

TWO OPERATORS DID NOT COME. `mpm_boundary` and `bm_strain` stay in `discovery_okuda/ops/membrane_ops.py`
and are registered only there, so archived specs still run and no new spec can reach them from core.
`mpm_boundary` overwrites grid-node velocity -- kinematic, momentum not conserved, the reaction
discarded -- and its standoff is set by the B-spline stencil width, measured across `recover`
0/2/6/20 as 46.6%/3.8%/11.5%/13.9% of the sheet inside the tissue against standoffs
+0.0006/+0.0124/+0.0088/+0.0069, never reaching the 0 -> +0.002 that would mean "just touching".
`integrin_track` is the constraint it should have been. `bm_strain` is, in AUDIT's words, "not a
mechanism".

THE RESOLUTION LIMIT TRAVELS WITH THE COUPLING. At `n_grid 48`, `dx = 0.021` against a 0.002-thick
sheet: one grid cell holds ~16 membrane particles, so the coupling strength here was set by grid
resolution and not by a measured adhesion.
"""''',
    ),
    dict(
        target="src/plexus/operators/contact_ops.py",
        sources=["mesh_contact_ops.py", "bm_sense_ops.py", "plate_ops.py", "surface_ops.py",
                 "load_ops.py"],
        module="plexus.operators.contact_ops",
        doc='''"""Where a triangulated surface meets a continuum, and what each tells the other.

    mesh_contact      the vertex mesh pushes MPM particles out of itself, and feels the reaction
    mesh_inside       which particles are inside the closed surface -- the test the contact needs
    surface_track     the surface's own moving frame, kept across division and death
    plate_confine     a rigid half-space (a projection; `block_seed` is the material version)
    bm_sense          the epithelium reads the membrane it is resting on
    ecm_load          the load the matrix puts back on the tissue
    ecm_gate_growth   and what that load does to growth -- entry condition `'mg_scale' in m`

WHY NOT GRID-BASED CONTACT. CFEMP (Lian et al. 2011, CMAME 200:3482) resolves contact by comparing
the two bodies' velocities at shared grid nodes, and needs mesh and grid to be comparable in size.
Ours are not: a cell is 0.73 dx and the basement membrane 0.1 dx, so both bodies live inside one
grid cell and the grid hands them ONE velocity -- the weld that `test_03_mesh_contact` measured.
"""''',
    ),
    # ---------------------------------------------------------------------------------------
    # PHASE 0b / E: regrouping the CORE's own one-operator files. Same rule, same script, and the
    # same twin gate -- a regrouping that moves a byte is a regrouping that broke something. The
    # sources here live in src/plexus/operators/ rather than discovery_okuda/ops/, so `src_dir`
    # says where to read them.
    dict(
        target="src/plexus/operators/mpm_ops.py",
        src_dir="src/plexus/operators",
        sources=["mpm_grid.py", "mpm_scatter.py", "mpm_grid_update.py", "mpm_gather.py",
                 "mpm_strain.py", "mpm_anchor.py", "mpm_spin.py", "material_map.py", "mpm.py"],
        module="plexus.operators.mpm_ops",
        doc='''"""MLS-MPM, as one module: the grid, the four-step cycle, and the two forces on it.

    mpm_grid            the background FIELD and the quadratic B-spline kernel (not an operator)
    mpm_scatter (p2g)   particle -> grid: mass, momentum, and the internal stress impulse
    mpm_grid_update     grid -> grid: the solve, gravity, and the wall conditions
    mpm_gather (g2p)    grid -> particle: velocity, the affine C, and advection
    mpm_strain          particle -> particle: F update and the material's response
    mpm_anchor          a spring to a rest position, for a body that must not drift
    mpm_spin            a prescribed angular velocity
    apply_material_map  a per-particle material assignment from a map
    mls_mpm_mechanics   the FENCED transitional oracle: the whole cycle in one operator

THE ORACLE IS STILL HERE AND IS STILL FENCED. `mls_mpm_mechanics` does in one operator what the
four above do in four, and it exists so the decomposition can be checked against something. It is
not the recommended path and it is not what a new spec should schedule.

WHY THE GRID IS IN THE SAME FILE. `stencil_offsets`, `bspline` and `sub_dt` were imported from
`mpm_grid` by seven other files, so the kernel that defines the discretisation was a private
detail of one of nine siblings. Every MPM operator's substep -- and the CFL ceiling that bounds it,
dt < dx / sqrt(E/rho) -- is now readable in one place.

TWO REJECTED NEIGHBOURS ARE NOT HERE. `mpm_boundary` (kinematic, momentum not conserved, standoff
set by the stencil width) and `bm_strain` stay in discovery_okuda; see membrane_ops and AUDIT.md.
"""''',
    ),
    dict(
        target="src/plexus/operators/motion_ops.py",
        src_dir="src/plexus/operators",
        sources=["drag.py", "glide.py", "sediment.py", "attractor_flow.py", "velocity_cruise.py",
                 "bounce.py", "gravity.py"],
        module="plexus.operators.motion_ops",
        doc='''"""How a body moves when nothing else is acting on it: damping, drift, walls, gravity.

    drag            velocity-proportional damping -- the overdamped limit's other half
    glide           move along the heading at a fixed speed
    velocity_cruise (cruise) relax the speed toward a target without touching the direction
    sediment        a settling velocity
    attractor_flow  a prescribed vector field (Lorenz, Rossler, ... ) as the velocity
    gravity         a uniform body force
    bounce          the wall and obstacle response

These are the operators with no INTERACTION in them: each reads one element's own state and writes
one element's own delta. Grouping them is what makes that visible -- and makes it obvious when a
new operator does not belong.
"""''',
    ),
    dict(
        target="src/plexus/operators/interaction_ops.py",
        src_dir="src/plexus/operators",
        sources=["attraction_repulsion.py", "squared_law.py", "cohesion.py", "separation.py",
                 "velocity_align.py", "stillinger_weber.py", "graph.py"],
        module="plexus.operators.interaction_ops",
        doc='''"""Pairwise laws, and the relation they act over.

    radius_graph          the relation: who is near whom (a rewire, and it comes first)
    attraction_repulsion  the two-term law the prototype scenarios are built on
    squared_law           inverse-square: gravity between bodies, Coulomb between charges
    cohesion / separation / velocity_align    the three boids terms, one operator each
    stillinger_weber      a three-body potential with an angular term (autograd)

THE GRAPH IS IN THIS FILE ON PURPOSE. Every law below it reads `edge_index`, and which relation
they read is the single most consequential thing about a spec that uses them -- a cutoff radius
changes a flock into a crystal. It is not a utility that happens to live elsewhere.
"""''',
    ),
    dict(
        target="src/plexus/operators/field_ops.py",
        src_dir="src/plexus/operators",
        sources=["scalar_field.py", "deposit.py", "diffuse.py", "decay.py", "sense.py",
                 "chemotax.py", "prescribed_field.py", "pacemaker.py", "activation_pulse.py",
                 "signal.py"],
        module="plexus.operators.field_ops",
        doc='''"""A continuum bound to a set: what writes into it, what happens inside it, what reads it.

    deposit        set -> field   (a cell lays a trail)
    diffuse        field -> field (finite_difference | spectral)
    decay          field -> field
    sense          field -> set   (a cell reads the value under it)
    chemotax       field -> set   (and moves up the gradient)
    playback       a PRESCRIBED field: a video or a measured stack, not a solved one
    pacemaker / activation_pulse   an excitable field's source terms
    signal         set -> set along an edge-set (the synapse case)

THE FOUR-STEP SHAPE IS THE POINT OF THE GROUPING. deposit / diffuse / decay / sense is one
mechanism written as four operators so that each can be swapped, and reading them apart is how a
spec ends up depositing into a field nothing senses.
"""''',
    ),
    dict(
        target="src/plexus/operators/agent_ops.py",
        src_dir="src/plexus/operators",
        sources=["agent_divide.py", "agent_grow.py", "agent_scatter.py", "agent_gather.py",
                 "agent_remodel.py", "polarity_align.py", "polarity_flow_align.py",
                 "active_force.py", "active_stress.py", "aggregate.py", "broadcast.py",
                 "segmentation_seed.py"],
        module="plexus.operators.agent_ops",
        doc='''"""Agents in a material: the two-way coupling, the population, and the scale maps.

    agent_scatter (agent_to_mpm)  agent -> grid: the agent deforms the material
    agent_gather  (mpm_to_agent)  grid -> agent: the material drags and confines the agent
    agent_remodel                 agent -> material stiffness: cells soften or rigidify tissue
    agent_divide / agent_grow     the population, on a fixed buffer with occupancy
    polarity_align (heading_align) / polarity_flow_align (flow_align)   where an agent points
    active_force / active_stress  a pulse becomes a contraction or a stress
    aggregate / broadcast         the two SCALE maps: child -> parent, parent -> child
    seed_from_segmentation        a measured instance segmentation becomes the cell level

THE PAIR THAT MATTERS is `agent_scatter` / `agent_gather`: they are the same coupling read in two
directions, and they must be scheduled together. Splitting them across files is how a spec comes to
push on a material that never pushes back.
"""''',
    ),
]

# ------------------------------------------------------------------- import rewrites, applied in order
# (pattern, replacement, which groups it applies to -- None = all)
REWRITES = [
    (r"^from topology_ops import ", "from plexus.models.topology import ", None),
    (r"^(\s+)from topology_ops import ", r"\1from plexus.models.topology import ", None),
    # inside vertex_ops the three files are now one, so a sibling import is a no-op
    (r"^import mesh_ops as _mesh_ops\n", "", "vertex_ops"),
    (r"^(\s*)from mesh_ops import ([^\n]+)\n",
     r"\1# (was `from mesh_ops import \2`) -- same module now\n", "vertex_ops"),
    (r"_mesh_ops\.", "", "vertex_ops"),
    # diffusion_reaction still needs the vertex module, by its new absolute name
    (r"^(\s*)from mesh_ops import ", r"\1from plexus.operators.vertex_ops import ", "diffusion_reaction"),
    # junction_ops + medioapical_ops become one file: `JO` was the sibling, now it is `self`
    (r"^import junction_ops as JO\n", "", "junction_ops"),
    (r"\bJO\.", "", "junction_ops"),
    (r"^(\s*)from mesh_ops import ", r"\1from plexus.operators.vertex_ops import ", "junction_ops"),
    (r"^(\s*)from mesh_ops import ", r"\1from plexus.operators.vertex_ops import ", "ecm_ops"),
    (r"^(\s*)from mesh_ops import ", r"\1from plexus.operators.vertex_ops import ", "membrane_ops"),
    (r"^(\s*)from mesh_ops import ", r"\1from plexus.operators.vertex_ops import ", "contact_ops"),
    # THE GRID MOVES INTO THE MODULE THAT USES IT. Inside `mpm_ops` the import is now a self-import
    # -- and not merely redundant: `mpm_grid.py` becomes a shim that re-exports FROM `mpm_ops`, so
    # leaving the line in would make the module import itself through its own shim and deadlock at
    # first import.
    (r"^from plexus\.operators\.mpm_grid import ([^\n]+)\n",
     r"# (was `from plexus.operators.mpm_grid import \1`) -- same module now\n", "mpm_ops"),
    (r"plexus\.operators\.mpm_grid", "plexus.operators.mpm_ops", "agent_ops"),
]


def _start(node):
    """The node's FIRST line, decorators included.

    `ClassDef.lineno` is the line of the `class` keyword, NOT of `@register_operator(...)` above it.
    Taking it as the start of the body dropped the decorator off the first class in every file --
    and a class whose decorator is gone is not registered, so `cell_geometry` vanished from the
    registry and 461 specs stopped loading. The failure was loud, which is the only reason it took
    one run to find; a decorator that merely tagged metadata would have gone through silently.
    """
    d = getattr(node, "decorator_list", None)
    return min([node.lineno] + [x.lineno for x in (d or [])])


def _split(path):
    """(docstring, [import lines], body) for one source file, by AST line numbers."""
    src = open(path).read()
    lines = src.splitlines(keepends=True)
    t = ast.parse(src)
    b = t.body
    doc_end = (b[0].end_lineno if b and isinstance(b[0], ast.Expr)
               and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str) else 0)
    imports, first_body = [], None
    for n in b:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            imports.extend(lines[n.lineno - 1:n.end_lineno])
        elif _start(n) > doc_end:
            first_body = _start(n) if first_body is None else min(first_body, _start(n))
    docstring = "".join(lines[:doc_end])
    body = "".join(lines[(first_body or doc_end + 1) - 1:])
    return docstring, imports, body


def _rewrite(text, tag):
    for pat, rep, only in REWRITES:
        if only is None or only == tag:
            text = re.sub(pat, rep, text, flags=re.M)
    return text


# A duplicate name whose two definitions are EQUIVALENT BUT NOT IDENTICAL, resolved by hand and
# recorded here rather than resolved silently by concatenation order.
#   `_np`  shape_chem_ops writes it as an `if`, shape_probe_ops as a ternary; both are
#          "detach/cpu/numpy if it is a tensor, else np.asarray". The `if` form is kept because its
#          docstring records WHY the helper exists -- the first end-to-end cuda launch died on
#          `can't convert cuda:0 device type tensor to numpy` after every CPU test had passed.
ALLOW_DUPLICATE = {"_np": "shape_chem_ops.py"}


def _toplevel_defs(body):
    """{name: (start_line, end_line, normalized_ast)} for the module-level defs of one body."""
    t = ast.parse(body)
    out = {}
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            m = ast.parse(body).body[t.body.index(n)]
            if (m.body and isinstance(m.body[0], ast.Expr)
                    and isinstance(m.body[0].value, ast.Constant)):
                m.body = m.body[1:]                       # ignore the docstring when comparing
            out[n.name] = (_start(n), n.end_lineno, ast.dump(m))
    return out


def _hold_back(body, exclude, src):
    """Cut the named top-level definitions out of `body`. Returns (body, {name: source_text})."""
    if not exclude:
        return body, {}
    lines = body.splitlines(keepends=True)
    defs = _toplevel_defs(body)
    held, cut = {}, []
    for name in exclude:
        if name not in defs:
            continue
        a, b, _dump = defs[name]
        held[name] = "".join(lines[a - 1:b])
        cut.append((a, b, name))
    for a, b, name in sorted(cut, reverse=True):
        lines[a - 1:b] = [f"# `{name}` is NOT PROMOTED -- see AUDIT.md. It stays in "
                          f"discovery_okuda/ops/{src}.\n"]
        print(f"    held back `{name}` from {src} -- registered in okuda only")
    return "".join(lines), held


def _drop_identical_duplicates(bodies):
    seen, out = {}, []
    for s, body in bodies:
        defs = _toplevel_defs(body)
        drop = []
        for name, (a, b, dump) in defs.items():
            if name in seen:
                src_first, dump_first = seen[name]
                if dump != dump_first and ALLOW_DUPLICATE.get(name) != src_first:
                    raise SystemExit(
                        f"  REFUSED: `{name}` is defined in both {src_first} and {s} and the two "
                        f"differ. Concatenating would silently shadow one with the other; decide "
                        f"which is right, add it to ALLOW_DUPLICATE with the reason, and record it "
                        f"in OKUDA_PROMOTION.md.")
                drop.append((a, b, name, src_first))
            else:
                seen[name] = (s, dump)
        if drop:
            lines = body.splitlines(keepends=True)
            for a, b, name, src_first in sorted(drop, reverse=True):
                lines[a - 1:b] = [f"# `{name}` is defined identically in {src_first} above; the "
                                  f"duplicate from {s} is dropped.\n"]
                how = "identical to" if ALLOW_DUPLICATE.get(name) != src_first else "equivalent to (allowlisted)"
                print(f"    dropped duplicate `{name}` from {s} -- {how} {src_first}'s")
            body = "".join(lines)
        out.append((s, body))
    return out


def build(group, dry=False):
    tag = os.path.basename(group["target"])[:-3]
    docs, imports, bodies = [], [], []
    # OPERATORS THAT ARE NOT PROMOTED. `AUDIT.md` rejects `mpm_boundary` and `bm_strain`, and a
    # rejection has to be visible in the code rather than only in a document: their classes are cut
    # out of the module that moves and left in the okuda file, so archived specs still run and no
    # new spec can reach them from core. `--verify` counts them as "held back", not as a difference.
    group["_held"] = {}
    for s in group["sources"]:
        d, imp, body = _split(os.path.join(ROOT, group.get("src_dir", ""), s)
                              if group.get("src_dir") else os.path.join(OPS, s))
        body, held = _hold_back(body, group.get("exclude") or [], s)
        # KEYED BY SOURCE. A group-wide dict wrote `bm_strain` into `integrin_ops.py`'s shim as well
        # as `membrane_ops.py`'s, and the second registration raised
        # `operator 'bm_strain' already has variant 'default'` -- so importing okuda died outright.
        if held:
            group["_held"][s] = held
        docs.append((s, d))
        imports.extend(imp)
        bodies.append((s, _rewrite(body, tag)))
    # dedup imports, keeping first appearance; `from __future__` must lead
    seen, keep = set(), []
    for line in imports:
        k = line.strip()
        if k and k not in seen:
            seen.add(k); keep.append(line)
    keep.sort(key=lambda l: (0 if "__future__" in l else 1,))
    keep = [_rewrite(l, tag) for l in keep]
    keep = [l for l in keep if l.strip()]

    # A NAME DEFINED IN TWO OF THE SOURCES would silently shadow, and concatenation is exactly where
    # that happens. `_np` is defined identically in `shape_chem_ops` and `shape_probe_ops`; the
    # second copy is dropped, and only when the two parse to the same tree with the docstring
    # ignored. A duplicate that is NOT identical stops the move -- that is a merge decision, not a
    # mechanical one, and it belongs in OKUDA_PROMOTION.md.
    bodies = _drop_identical_duplicates(bodies)

    out = [group["doc"], "\n"]
    out += keep
    for s, body in bodies:
        src_doc = dict(docs)[s].strip()
        head = src_doc.strip('"').strip().splitlines()[0] if src_doc else s
        out.append(f"\n\n# {'=' * 106}\n"
                   f"# FROM `discovery_okuda/ops/{s}` -- {head}\n"
                   f"# {'=' * 106}\n")
        out.append(body if body.endswith("\n") else body + "\n")
    text = "".join(out)
    dst = os.path.join(ROOT, group["target"])
    if dry:
        print(f"  {group['target']}: {len(text.splitlines())} lines from "
              f"{', '.join(group['sources'])}")
        return text
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        f.write(text)
    print(f"  wrote {group['target']}  ({len(text.splitlines())} lines)")
    return text


SHIM = '''"""{name} -- MOVED to `{module}`.

Kept as a re-export because thirty files import it by bare module name -- `run_one.py`,
`instrument.py`, `vtk_render.py`, `metrics.py` and twenty archive/analysis scripts -- and the
campaign is still running against them. PRIVATE NAMES ARE RE-EXPORTED TOO: `_carry_face_state`,
`_engine_owns_clock` and friends are called across module boundaries in okuda, so a shim that
exported only the public surface would break at the first T1.

New code should import from `{module}`.
"""
from {module} import *          # noqa: F401,F403
{privates}'''

# The private re-export block, emitted only when there ARE private names: a `from X import (\n#
# (none))` swallows its own closing paren inside the comment and the shim will not parse -- which
# would break every one of the thirty importers at once, loudly but for the silliest reason.
HELD_HEADER = '''

# =============================================================================================
# NOT PROMOTED: {names}. `AUDIT.md` rejects them, so they were cut out of the module that moved to
# `src/plexus/operators/` and left here. They are still registered -- an archived spec that names
# one still runs -- but no spec can reach them from the core registry, and there is no alias to
# find them by. A rejection that lives only in a markdown file is a rejection that the next reader
# re-promotes by accident.
# ============================================================================================='''

PRIVATE_BLOCK = '''from {module} import (          # noqa: F401  the underscored names okuda reaches for
{names})
'''


def shim(group):
    for s in group["sources"]:
        path = os.path.join(ROOT, group.get("src_dir", ""), s) if group.get("src_dir") \
            else os.path.join(OPS, s)
        _d, _i, body = _split(path)
        priv = sorted({n.name for n in ast.parse(body).body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name.startswith("_")}
                      | {t.id for n in ast.parse(body).body if isinstance(n, ast.Assign)
                         for t in n.targets if isinstance(t, ast.Name) and t.id.startswith("_")})
        block = (PRIVATE_BLOCK.format(module=group["module"],
                                      names=",\n".join(f"    {p}" for p in priv)) if priv else "")
        text = SHIM.format(name=s[:-3], module=group["module"], privates=block)
        held = (group.get("_held") or {}).get(s, {})
        if held:
            text += (HELD_HEADER.format(names=", ".join(f"`{k}`" for k in sorted(held)))
                     + "\n\n" + "\n\n".join(held[k] for k in sorted(held)) + "\n")
        with open(path, "w") as f:
            f.write(text)
        print(f"    shim {s} -> {group['module']}"
              + (f"  (+{len(priv)} private name(s))" if priv else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        return verify()
    for g in GROUPS:
        # IDEMPOTENT. Once a group has moved, its sources are re-export shims -- re-running would
        # concatenate the shims and overwrite the real module with 24 lines of imports. The tool
        # refuses rather than "succeeding": a move that silently empties the module it moved is the
        # worst possible failure here, because every spec would still load and register nothing.
        _sdir = os.path.join(ROOT, g["src_dir"]) if g.get("src_dir") else OPS
        already = [x for x in g["sources"]
                   if not os.path.exists(os.path.join(_sdir, x))
                   or "-- MOVED to `" in open(os.path.join(_sdir, x)).read()[:400]]
        if already:
            print(f"  {g['target']}: already moved ({', '.join(already)}) -- skipped")
            continue
        build(g, dry=a.dry)
        if not a.dry:
            shim(g)
    if not a.dry:
        print("\n  now: add the modules to src/plexus/operators/__init__.py, then run\n"
              "       python tools/group_operators.py --verify")
    return 0


def _pre_move_text(src, rel_dir="discovery_okuda/ops", max_back=40):
    """The last committed version of `discovery_okuda/ops/<src>` that was NOT already a shim.

    `HEAD:` is the wrong thing to read once the move is committed: the file there IS the shim, and
    comparing a shim against the module it points at reports everything as missing. Walking back to
    the last real version is what makes `--verify` still mean something a week later.
    """
    rel = f"{rel_dir}/{src}"
    revs = subprocess.run(["git", "-C", ROOT, "rev-list", f"-{max_back}", "HEAD", "--", rel],
                          capture_output=True, text=True, timeout=60).stdout.split()
    for rev in revs:
        t = subprocess.run(["git", "-C", ROOT, "show", f"{rev}:{rel}"],
                           capture_output=True, text=True, timeout=60).stdout
        if t and "-- MOVED to `" not in t[:400]:
            return t
    return ""


def verify():
    """Every top-level definition of every source must appear VERBATIM in the module it moved to.

    Not "the registry has the right number of names" -- that check passed while the decorator of the
    first class in each file was being silently dropped. This one compares TEXT: the exact lines the
    source file held, decorators included, searched for as a substring of the target. If the move
    changed one character inside one function body, this says which function and which file.
    """
    ok, bad, dropped, heldb = 0, [], [], []
    for g in GROUPS:
        if not os.path.exists(os.path.join(ROOT, g["target"])):
            print(f"  {g['target']}: not built yet -- skipped")
            continue
        target = open(os.path.join(ROOT, g["target"])).read()
        for src in g["sources"]:
            path = (os.path.join(ROOT, g["src_dir"], src) if g.get("src_dir")
                    else os.path.join(OPS, src))
            text = (_pre_move_text(src, g.get("src_dir", "discovery_okuda/ops"))
                    or (open(path).read() if os.path.exists(path) else ""))
            if not text or "-- MOVED to `" in text[:400]:
                print(f"    {src}: no pre-move source found in history -- cannot verify")
                continue
            lines = text.splitlines(keepends=True)
            for n in ast.parse(text).body:
                if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                chunk = "".join(lines[_start(n) - 1:n.end_lineno])
                chunk = _rewrite(chunk, os.path.basename(g["target"])[:-3])
                if chunk in target:
                    ok += 1
                elif n.name in (g.get("exclude") or []):
                    heldb.append(f"{src}:{n.name}")       # deliberately not promoted (AUDIT.md)
                elif n.name in ALLOW_DUPLICATE and ALLOW_DUPLICATE[n.name] != src:
                    dropped.append(f"{src}:{n.name}")     # a duplicate resolved by ALLOW_DUPLICATE
                else:
                    bad.append(f"{src}:{n.name}")
    print(f"  {ok} definition(s) moved verbatim"
          + (f"; {len(dropped)} dropped by decision ({', '.join(dropped)})" if dropped else "")
          + (f"; {len(heldb)} held back by AUDIT ({', '.join(heldb)})" if heldb else "")
          + (f"; {len(bad)} DIFFER: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
