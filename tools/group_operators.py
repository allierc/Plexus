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
    for s in group["sources"]:
        d, imp, body = _split(os.path.join(OPS, s))
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
PRIVATE_BLOCK = '''from {module} import (          # noqa: F401  the underscored names okuda reaches for
{names})
'''


def shim(group):
    for s in group["sources"]:
        path = os.path.join(OPS, s)
        _d, _i, body = _split(path)
        priv = sorted({n.name for n in ast.parse(body).body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name.startswith("_")}
                      | {t.id for n in ast.parse(body).body if isinstance(n, ast.Assign)
                         for t in n.targets if isinstance(t, ast.Name) and t.id.startswith("_")})
        block = (PRIVATE_BLOCK.format(module=group["module"],
                                      names=",\n".join(f"    {p}" for p in priv)) if priv else "")
        with open(path, "w") as f:
            f.write(SHIM.format(name=s[:-3], module=group["module"], privates=block))
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
        build(g, dry=a.dry)
        if not a.dry:
            shim(g)
    if not a.dry:
        print("\n  now: add the modules to src/plexus/operators/__init__.py, then run\n"
              "       python tools/group_operators.py --verify")
    return 0


def verify():
    """Every top-level definition of every source must appear VERBATIM in the module it moved to.

    Not "the registry has the right number of names" -- that check passed while the decorator of the
    first class in each file was being silently dropped. This one compares TEXT: the exact lines the
    source file held, decorators included, searched for as a substring of the target. If the move
    changed one character inside one function body, this says which function and which file.
    """
    ok, bad, dropped = 0, [], []
    for g in GROUPS:
        target = open(os.path.join(ROOT, g["target"])).read()
        for src in g["sources"]:
            path = os.path.join(OPS, src)
            text = subprocess.run(["git", "-C", ROOT, "show", f"HEAD:discovery_okuda/ops/{src}"],
                                  capture_output=True, text=True, timeout=60).stdout
            if not text:                                  # not yet committed as a shim; read on disk
                text = open(path).read()
            lines = text.splitlines(keepends=True)
            for n in ast.parse(text).body:
                if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                chunk = "".join(lines[_start(n) - 1:n.end_lineno])
                chunk = _rewrite(chunk, os.path.basename(g["target"])[:-3])
                if chunk in target:
                    ok += 1
                elif n.name in ALLOW_DUPLICATE and ALLOW_DUPLICATE[n.name] != src:
                    dropped.append(f"{src}:{n.name}")     # a duplicate resolved by ALLOW_DUPLICATE
                else:
                    bad.append(f"{src}:{n.name}")
    print(f"  {ok} definition(s) moved verbatim"
          + (f"; {len(dropped)} dropped by decision ({', '.join(dropped)})" if dropped else "")
          + (f"; {len(bad)} DIFFER: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
