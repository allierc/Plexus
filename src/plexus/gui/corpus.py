"""What the page's Claude session is primed with: the Plexus framework, its operator atlas, its
entities, and two reference specs for the page's domain.

The first task of a session carries this text; every later task resumes the session, so it is
paid once. Generated from the registries, not written by hand, except the framework digest
below, so a new operator is in the atlas the day it is registered.
"""
from __future__ import annotations

import os

from plexus.gui import studio

REPO = studio.REPO
_CACHE: dict = {}

FRAMEWORK = """\
PLEXUS, IN ONE PAGE.

A simulation is ONE YAML SPEC with six blocks: `general` (name, seed, n_frames, dt, boundary
free|wall, dim, world box, units), `sets`, `fields`, `operators`, `schedule`, `plotting`, plus
`seed:` (the seed operators, run once before frame 0).

SETS are the objects. Each set is a LEVEL of the hierarchy: `n` rows, a `state:` of named blocks
(pos, vel, sep, area, ...), an optional `entity:` (a registered kind that supplies the default state
schema, render and reserve), and optional `types:` -- species or bodies, each with `fraction:` or
`count:` and per-type properties the operators read (radius, region, material, youngs, ...).
Two ways sets relate:
  containment  `parent: <set>` with `per_parent: k` (k rows per parent row, blocks contiguous per
               parent; `parent_pos: centroid` places children on the parent's centroid). A cell
               contains its proteins and organelles; a body contains its material points.
  relation     `maps: {srce: vertex, trgt: vertex, face: cell}` on a half_edge set: a relation
               table between sets. `mesh: half_edge` on a vertex set says it carries a surface.
The tissue idiom: `cell` (per-cell scalars: area, A0, P0, V0f, age, ndiv, ...), `vertex`
(pos, vel, sep; `mesh: half_edge`; `cell_set: cell`), `half_edge` (srce/trgt/face), then contained
sets: `protein` (entity protein_cluster, parent cell, types = species with region/density/s/tau)
and `organelle` (entity organelle, parent cell, types = species with count/radius/region/on_divide).
The material idiom: `cell` (n bodies, `start:` centres, types = bodies with material/youngs or
bulk_modulus/density and `shape: ball` or `block: [x0,y0,z0,x1,y1,z1]`) and `mpm_particle`
(parent cell, per_parent points per body, radius of a ball).
An apico-basal cell is a prism: apical cap = pos + sep, basal cap = pos - sep (`seed_mesh` with
model apicobasal, h0 thickness, `apical: in|out`); regions apical | basal | mid | interior for
proteins, interior | apical_side | basal_side for organelles.

FIELDS are continua on their own grid: `mpm_grid` (frame mpm_grid, n_grid) is the MPM background.

OPERATORS act on one set (`at:`), some on a field (`to:`/`from:`). Kinds: `seed` (once, before
the run, in `seed:`), `dynamics` (every scheduled tick, produce per-node rates), `structural`
(change membership: divide, spawn, kill, re-project), `lateral` (pairwise), `aggregate`
(child -> parent). Every operator has a registered name, a family, the set kind it acts on,
required parameters and per-type properties it reads (listed in the atlas below). A `model:`
selects a variant (cell_divide model adder|timer|doubler, cell_mechanics model apicobasal|...).

SCHEDULE is the order per frame, a list of operator names; `{substep_dt, steps: [...]}` is a
micro-loop run many times per frame (the MPM step). `every: k` on an operator runs it every k
frames; `engine_clock: true` counts frames rather than time.

PLOTTING is the movie and the page's picture: `colors` per type, `dot_radius` per type (drawn as
spheres of that world radius), `mesh_surface` apical|basal|mid, `mesh_opacity`, `cross_section`
(axis, at, span, spheres: true cuts the spheres into discs), `curve` (up to three panels: cells,
count:<set>:<species>, area, volume, ...), `up_axis` (2 for tissues, 1 for material boxes),
`box_frame`, `dot_size`, `dot_shading: body`.

UNITS: `general.units` {length_um, time_s, force_nN} give one simulation length/time unit in
micrometres/seconds; the scale bar and the movie clock read them. Tissue specs use 10 um and 600 s
per unit; material specs use metres (1e6 um) and seconds.

WHAT THE ENGINE DOES: build (allocate every set from the spec, assign types, size reserves),
seed (run `seed:` once), run (for each frame, run the schedule, integrate, record). A run from the
page starts from the seed; there is no continue-from-here.
"""


def operator_atlas() -> str:
    """One line per registered operator: name, family/kind/set, parameters, type properties, doc."""
    import plexus.operators                                        # noqa: F401  registers the atlas
    from plexus.models.registry import _OPERATOR_REGISTRY as REG
    seen, lines = set(), []
    for name in sorted(REG):
        cls = REG[name]
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        names = getattr(cls, "REGISTERED_NAMES", [name])
        fam = getattr(cls, "FAMILY", None); kind = getattr(cls, "KIND", None); st = getattr(cls, "SET", None)
        req = list(getattr(cls, "REQUIRES_PARAMS", []) or [])
        roles = getattr(cls, "PARAM_ROLES", {}) or {}
        params = sorted(set(req) | set(roles.keys()))
        tprops = list(getattr(cls, "REQUIRES_TYPE_PROPS", []) or []) + list(getattr(cls, "OPTIONAL_TYPE_PROPS", []) or [])
        doc = (cls.__doc__ or "").strip().splitlines()
        doc = doc[0].strip() if doc else ""
        model = getattr(cls, "MODEL", None)
        head = "/".join(names) + (f" [model {model}]" if model else "")
        lines.append(f"- {head}: {fam}/{kind} at {st}"
                     + (f"; params {', '.join(params)}" if params else "")
                     + (f"; type props {', '.join(tprops)}" if tprops else "")
                     + (f" -- {doc[:140]}" if doc else ""))
    return "OPERATOR ATLAS (every registered operator; use these names and no others):\n" + "\n".join(lines)


def entities_text() -> str:
    import plexus.models.entities                                  # noqa: F401
    from plexus.models.registry import _ENTITY_REGISTRY as E
    seen, lines = set(), []
    for name in sorted(E):
        cls = E[name]
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        names = getattr(cls, "REGISTERED_NAMES", [name])
        doc = (cls.__doc__ or "").strip().splitlines()
        doc = doc[0].strip() if doc else ""
        lines.append(f"- {'/'.join(names)}: depth {getattr(cls, 'DEPTH', None)}"
                     + (f", reserve_factor {getattr(cls, 'RESERVE_FACTOR')}" if getattr(cls, "RESERVE_FACTOR", None) else "")
                     + (f" -- {doc[:120]}" if doc else ""))
    return "ENTITIES (`entity:` values a set may declare):\n" + "\n".join(lines)


def _module_doc(rel: str, limit: int = 4000) -> str:
    f = os.path.join(REPO, rel)
    if not os.path.exists(f):
        return ""
    src = open(f, errors="replace").read()
    if not src.startswith('"""'):
        return ""
    end = src.find('"""', 3)
    return f"FROM {rel} (design note):\n" + src[3:end].strip()[:limit]


def _spec(rel: str) -> str:
    f = os.path.join(REPO, rel)
    if not os.path.exists(f):
        return ""
    return f"REFERENCE SPEC {os.path.basename(rel)} -- copy this SHAPE:\n```yaml\n" + open(f).read() + "```"


REFERENCES = {
    "bio": ["config/tissue/spheroid_organelles_polar.yaml", "config/tissue/spheroid_proteins.yaml"],
    "material": ["config/si_material/si_three_balls.yaml", "config/si_material/si_waterfall.yaml"],
    "neurons": ["config/neural/ctrnn_gui.yaml"],
    "metabolism": ["config/metabolism/massaction_toy.yaml"],
}
DESIGN_NOTES = {
    "bio": ["src/plexus/operators/protein_ops.py", "src/plexus/operators/organelle_ops.py"],
    "material": ["src/plexus/operators/mpm_ops.py"],
    "neurons": ["src/plexus/operators/neural.py"],
    "metabolism": ["src/plexus/operators/metabolism.py"],
}


def corpus(mode: str = "bio") -> str:
    """The priming text for a page session, cached per mode."""
    mode = mode if mode in REFERENCES else "bio"
    if mode in _CACHE:
        return _CACHE[mode]
    parts = [FRAMEWORK, operator_atlas(), entities_text()]
    parts += [t for t in (_module_doc(r) for r in DESIGN_NOTES[mode]) if t]
    parts += [t for t in (_spec(r) for r in REFERENCES[mode]) if t]
    text = "\n\n".join(parts)
    _CACHE[mode] = text
    return text
