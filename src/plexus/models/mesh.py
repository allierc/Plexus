"""The half-edge mesh, as an engine concept rather than an attribute an operator invents.

WHAT IT IS. A closed cellular surface stored as a flat half-edge table: three parallel int64 arrays
`E_srce`, `E_trgt`, `E_face`, plus the counts `nF` (faces = cells) and `Nv` (vertices). Face *f*'s
ring is recovered by walking the table and appending `E_srce[k]` for every `k` with `E_face[k] == f`
-- **in table order**. That ordering IS the geometry: `topology_ops.rings_from_flat_3d` never reads
`E_trgt`, never sorts and never validates, so a sort, a compaction or a scatter-into-reservoir by any
future writer produces garbage rings in silence. Nothing here reorders the table.

WHY IT IS PROMOTED. It has been a private attribute (`lvl._mesh`) written by one operator and read by
roughly forty consumers across six surfaces -- the chemistry, the mechanics, the T1s, the cross-scale
contact operators, the recorder, the salvage, the instrument's fingerprint, the renderers. The engine
had no idea it existed: `grep -rn "_mesh" src/plexus/` returned nothing at all. A structure that
central cannot be a convention.

WHY IT SUBCLASSES `dict`, and this is not a style choice.

    discovery_okuda/instrument.py:43     if isinstance(m, dict):

That is the D4 acted-ledger -- the thing that decides whether an operator DID ANYTHING, wrapped in a
bare `except Exception: return ()`. A `MutableMapping` that is not a `dict` subclass fails that test
silently, the fingerprint collapses to `()`, and every operator whose only effect is on the mesh --
`cell_grow` writing `V0f`, `cell_divide` changing `nF`, `junction_myosin` writing `myo` -- scores
zero acts. The campaign then stamps `valid_evidence: False` on runs that were fine. The module's own
comment calls that reading "the strongest and least recoverable kind of error the campaign can
make". Six more consumers discover the contents by iteration or `get`/`setdefault`/`__setitem__`
with no key list. So: a dict, with methods.

THE NAMESPACE IS OPEN, deliberately. `RESERVED` is the five keys the engine owns. Everything else --
the per-face targets, the myosin, the chemistry's caches, a probe's published field -- is an
extension namespace, because three live mechanisms depend on a name being *absent*:
`cell_shape_probe` publishes and then deletes its field (absent means "no cell dies", zero-filled
would mean "kill everything"), `ecm_gate_growth`'s entire entry condition is `'mg_scale' in m`, and
the apoptosis discriminator resolves whatever name its spec gives it. A closed schema breaks all
three.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:                                     # the schema is readable without torch
    torch = None


# The engine owns these; everything else is the open extension namespace above.
RESERVED = ("E_srce", "E_trgt", "E_face", "nF", "Nv")

# What `mesh_seed` lays down. Recorded here so a checkpoint loader can be checked against it: a
# reload that produces fewer keys drops straight into the silent-zero paths (`age`/`ndiv` become
# zeros on a length mismatch; `divjit` falls back to a fixed-seed draw).
SEED_KEYS = RESERVED + ("A0", "P0", "alive", "divjit", "V0f", "Vbirth", "V0", "v_ref",
                        "R0", "verts0", "Nv_max", "nF_max", "Ebuf")

MESH_KINDS = ("half_edge",)


class MeshTable(dict):
    """The half-edge table. A `dict` first (see above) and a typed object second.

    Construct it exactly where the bare dict was constructed and with the same keys; every existing
    reader is unaffected by design, which is what makes the promotion checkable against a
    bit-for-bit twin run rather than against an opinion.
    """

    KIND = "half_edge"

    # ---------------------------------------------------------------- introspection
    @property
    def n_faces(self) -> int:
        return int(self.get("nF", 0))

    @property
    def n_vertices(self) -> int:
        """The vertex HIGH-WATER MARK, not the live count.

        `cell_die` rewrites `E_srce`/`E_trgt`/`E_face`/`nF` and never touches `Nv`, so vertices
        orphaned by a death stay inside every `state[:Nv]` slice and keep feeling the radial spring.
        That is today's behaviour and the promotion does not change it -- fixing it would move the
        energy, and the gate is bit-equality.
        """
        return int(self.get("Nv", 0))

    def describe(self) -> str:
        extra = sorted(k for k in self if k not in SEED_KEYS)
        return (f"half_edge: {self.n_faces} faces, {self.n_vertices} vertices, "
                f"{len(self.get('E_srce', ())) if 'E_srce' in self else 0} half-edges"
                + (f"; + {len(extra)} operator key(s): {', '.join(extra[:8])}" if extra else ""))

    # ---------------------------------------------------------------- the one carry
    def reindex_faces(self, keep, dt=None, dev=None, names=None):
        """Permute every per-face array through `keep` (new face -> old face).

        THE CARRY IS ONE OPERATION AND IT WAS WRITTEN FOUR TIMES -- in `cell_divide`, in `cell_die`,
        in `_carry_face_state` and in `edge_flip` -- which is why `edge_flip`'s copy silently omits
        the medioapical myosin: a per-face state added by one operator has to be known to every
        topology operator, and the fourth one did not get the memo.

        SEMANTICS, unchanged from `_carry_face_state`: `keep` COPIES the parent's value onto both
        daughters, so what travels this way must be INTENSIVE -- a density, a concentration, an age.
        Carrying an extensive quantity doubles it at every division.

        CLAMPS RATHER THAN RAISES on a short array, which is what `_carry_face_state` does today.
        `medioapical_myosin` raises on the same condition; the two disagree and this keeps the
        permissive behaviour, because changing it is a behaviour change and belongs in its own step.
        """
        if torch is None:
            raise RuntimeError("reindex_faces needs torch")
        idx = torch.as_tensor(np.asarray(keep, np.int64), device=dev)
        for nm in sorted(set(names or ()) | set(self.get("face_carry") or ())):
            a = self.get(nm)
            if a is None:
                continue
            a = a if torch.is_tensor(a) else torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
            self[nm] = a.to(dev)[idx.clamp(max=max(a.shape[0] - 1, 0))].to(dt)

    def carry_vertices(self, births, dt=None, dev=None, names=None, level=None):
        """Give every vertex BORN this tick a value, by averaging its parents'.

        THE PER-VERTEX HALF OF `reindex_faces`, and it did not exist. Faces are permuted by a
        topology edit and every per-face array is carried through `keep`; vertices are APPENDED by
        `divide_face_3d` and MERGED by `face_collapse_3d`, and nothing carried anything. That is
        invisible while `pos` is the only per-vertex quantity, because both functions write `pos`
        themselves -- the midpoint, the centroid. It stops being invisible the moment a second one
        exists: a monolayer's apico-basal separation would be 0 at a vertex born on a septum, which
        is a cell of zero height along the seam it just grew, and would keep `r[0]`'s value at an
        extrusion, which makes the site jump.

        `births` is `[(new_vertex, (parent, ...)), ...]`, exactly what `divide_face_3d` and
        `face_collapse_3d` now append to. A division reports two parents (the endpoints of the split
        edge); a collapse reports three (the survivor and the two it absorbed).

        THE BLEND IS A MEAN, WHICH MAKES THIS AN INTENSIVE-ONLY CARRY, the same restriction
        `reindex_faces` documents: a density, a thickness, a concentration. An extensive quantity
        averaged over two parents is halved at every division rather than conserved, and the failure
        is silent. Anything extensive needs its own operator, not this.

        ORDER MATTERS AND IS THE CALLER'S. `births` is applied in sequence, so a vertex that is
        itself a parent of a later birth contributes its NEW value -- which is what a collapse
        following a division in the same tick should see.

        `level` -- AND A DECLARED NAME MAY LIVE THERE RATHER THAN IN THE TABLE, which the first
        version of this missed and R2 walked straight into. A per-vertex quantity can be a MESH
        COLUMN (`self[name]`) or a STATE BLOCK on the Level (`level.state[:, c0:c1]`), and the
        apico-basal separation is the second kind -- deliberately, because a state block reaches the
        trajectory through the generic per-set recording path and therefore does not touch
        `FACE_RECORD`/`EDGE_RECORD`/`snapshot()` at all. Without this branch
        `declare_vertex_carry(m, "sep")` succeeded, `self.get("sep")` returned None, the carry
        skipped it silently, and every vertex born by division held the buffer's ZERO: measured on
        `ab_sphere` at 60 frames, all 66 newly born vertices had |sep| = 0.0000 against a seeded
        0.2000 -- a cell of zero height along the seam it had just grown, which is the exact failure
        the carry was written to prevent.
        """
        if torch is None:
            raise RuntimeError("carry_vertices needs torch")
        want = sorted(set(names or ()) | set(self.get("vertex_carry") or ()))
        if not want or not births:
            return
        for nm in want:
            a = self.get(nm)
            sl = None
            if a is None and level is not None and nm in getattr(level, "state_schema", ()):
                sl = level.state_schema[nm]                # (c0, c1) into the level's state
                a = level.state[:, sl[0]:sl[1]]
            if a is None:
                continue
            a = a if torch.is_tensor(a) else torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
            a = a.to(dev) if dev is not None else a
            for new_i, parents in births:
                ps = [int(q) for q in parents if 0 <= int(q) < a.shape[0]]
                if not ps or int(new_i) >= a.shape[0]:
                    # A VERTEX PAST THE END IS NOT AN ERROR HERE. `reindex_faces` clamps rather than
                    # raises on a short array and this keeps that contract: the reservoir is sized
                    # by the spec, and an operator that has not grown its own array yet must not
                    # take the run down.
                    continue
                a[int(new_i)] = a[ps].mean(dim=0)
            if sl is None:
                self[nm] = a.to(dt) if dt is not None else a
            else:
                st = level.state.clone()
                st[:, sl[0]:sl[1]] = a
                level.state = st

    # ---------------------------------------------------------------- recording
    # THE PER-FACE NAMES A PICTURE NEEDS, and this list is why `snapshot` is not topology-only.
    #
    # A renderer cannot colour by a quantity that only existed inside one frame's forward pass. The
    # VTK renderer's four colours are the activator (from the CELL SET's `chem`, which the recorder
    # already keeps), plus three marks that live ON THE MESH and nowhere else: `age`+`ndiv` say a
    # cell has just divided, `apop` says the second field marked it to die, `inhib` says the second
    # field switched its growth off. Record topology alone and every core-side movie is a plain
    # white surface -- which is exactly the difference between okuda's `movie.mp4` and the
    # matplotlib fallback the core wrote before this.
    #
    # `A0`/`P0`/`V0f` come too because `analyze_forces` reconstructs the energy from them offline,
    # and re-deriving a target from a position is not the same number.
    #
    # EVERY NAME IS OPTIONAL. A run without an inhibitor has no `inhib`; the key is simply absent
    # from the snapshot, and the reader draws nothing -- which is the same None-safe contract
    # `topo_record`'s `cp()` has always had.
    FACE_RECORD = ("A0", "P0", "V0f", "age", "ndiv", "apop", "inhib", "myo_med")

    # THE RECORDED NAME IS NOT ALWAYS THE LIVE ONE, and two of the four colours above were lost to
    # exactly that. The operators write `m["apop_flag"]` (`cell_die`) and `m["inhib_frac"]`
    # (`cell_chem_react`); `FACE_RECORD` calls them `apop` and `inhib`, which is what the renderer
    # and every offline reader ask for. `topo_record` bridges the two by hand -- `apop=cp(...)`,
    # `inhib=cp("inhib_frac")` -- but `snapshot` did a bare `self.get(nm)`, got None for both, and
    # skipped them. So the core recorded NEITHER the dying-cell flag NOR the growth inhibitor,
    # ever: `log/promotion/MINISITE_apop_patch_big` is an apoptosis scene whose trajectory contains
    # no record that any cell was sentenced, and its movie is a plain ball with the one mechanism
    # it exists to show invisible.
    #
    # An alias rather than a rename because both live names are load-bearing: `apop_flag` is
    # carried across renumbering by `cell_die` and `edge_flip`, and `ecm_gate_growth`'s entry
    # condition is a key TEST on this table, so the namespace is not free to be tidied.
    FACE_ALIAS = {"apop": "apop_flag", "inhib": "inhib_frac"}

    # THE OPERATORS' OWN COUNTERS, which are scalars on the table and not per-face arrays -- so
    # `FACE_RECORD` cannot carry them, and without them a gate has to INFER what an operator already
    # knows.
    #
    # THAT INFERENCE IS WHERE GATE 00 WENT WRONG. `t1_total` was reconstructed from the topology as
    # "edges newly formed between two pre-existing vertices" and returned 2,890 against `edge_flip`'s
    # own 1,499: about two new such edges per accepted reconnection, because a 3D RNR rewires more
    # than the one edge it is named for. Both numbers are correct about different quantities, which
    # is the worst kind of disagreement -- neither side is wrong and the row is meaningless.
    #
    #   n_t1         accepted reconnections, cumulative      (edge_flip)
    #   n_apop       extrusions, cumulative                  (cell_die)
    #   div_blocked  divisions refused for want of buffer    (cell_divide)
    #   apop_spill   material a death could not bequeath     (cell_die) -- must stay ~0
    #   renumber_failed  a renumber that did not act -- MUST be 0; see Hierarchy.renumber_set
    # `mono_h` IS A SCALAR AND NOT A FACE COLUMN, and that is the difference between a recorded
    # thickness and a zero. v1 of the monolayer is a UNIFORM thickness, so one number says all of
    # it -- and a per-face array here goes stale: `cell_mechanics` publishes it sized to the nF it
    # saw, `cell_divide` runs AFTER it and grows nF, and `snapshot` drops a short array rather
    # than padding it. Recorded per-face, the thickness came back 0.000 on every frame where a
    # division had fired and 1.200 only on the frames where none had -- a thickness that blinks.
    SCALAR_RECORD = ("n_t1", "n_apop", "div_blocked", "apop_spill", "renumber_failed", "mono_h")

    # PER-HALF-EDGE STATE, and it is a THIRD ragged length. `myo` has one entry per half-edge, not
    # per face and not per row, so it cannot ride in `FACE_RECORD` (which drops anything shorter
    # than `nF` and truncates anything longer) and it cannot share the face offsets. Without it,
    # every myosin row of gate 01 -- the dispersion, the hot fraction, the pinning to `activity` --
    # is uncomputable from a core run, which is eight of that gate's fifteen rows.
    #
    # IT IS RECORDED AFTER `junction_sync` FOR A REASON. `edge_flip` rewires the arrays and
    # `cell_divide` lengthens them, both AFTER `junction_myosin` wrote `myo` for the arrays as they
    # were: on the 401-frame nominal, 56 of 200 snapshots carried a myosin array 6 to 1,356 entries
    # short of the edge arrays, and every reader indexes it positionally. The engine records at the
    # END of the tick, so what lands here is the re-keyed array -- and
    # `myosin_array_aligned_with_half_edges` is the bookkeeping row that says so.
    EDGE_RECORD = ("myo", "myo_amount")

    def snapshot(self, face_record=None):
        """One recorded frame: the three half-edge arrays, the two counts, and the per-face state.

        The topology part is exactly what `topo_record` appends to `hist`, so every offline reader
        written against that shape is unaffected; the per-face part is a SUBSET of what `hist`
        carries, chosen by `FACE_RECORD` rather than by "whatever happens to be in the table",
        because the table's namespace is open and a snapshot of all of it would grow without
        bound the moment an operator cached something there.
        """
        def _np(x):
            if torch is not None and torch.is_tensor(x):
                return x.detach().cpu().numpy()
            return np.asarray(x)
        out = dict(E_srce=_np(self["E_srce"]), E_trgt=_np(self["E_trgt"]),
                   E_face=_np(self["E_face"]), nF=int(self["nF"]), Nv=int(self["Nv"]))
        nF = out["nF"]
        for nm in (face_record if face_record is not None else self.FACE_RECORD):
            v = self.get(self.FACE_ALIAS.get(nm, nm), self.get(nm))
            if v is None:
                continue
            a = _np(v).ravel()
            if a.shape[0] >= nF:                    # a stale short array is dropped, not padded
                out[nm] = a[:nF].astype(np.float32)
        nE = len(out["E_srce"])
        for nm in (self.EDGE_RECORD):
            v = self.get(nm)
            if v is None:
                continue
            a = _np(v).ravel()
            # A SHORT ARRAY IS RECORDED AS SHORT, NOT PADDED. Padding would make the alignment row
            # pass on exactly the frames it exists to catch.
            out["e_" + nm] = a.astype(np.float32) if a.shape[0] == nE else a.astype(np.float32)
        for nm in self.SCALAR_RECORD:
            v = self.get(nm)
            if v is not None and np.ndim(v) == 0:
                out["scalar_" + nm] = float(v)
        return out


def mesh_row_columns(ms):
    """Every mesh column ANY recorded row carries, with a filler for the rows that do not.

    Returns `(scalars, edge_cols, face_cols, fill)` -- three sorted name lists and
    `fill(m, col)` giving that row's values, zero-filled when the row lacks the name.

    THE UNION, NOT THE INTERSECTION, AND THIS IS A BUG FIX. Both writers used to take

        cols = set.intersection(*[{k for k in m if k not in RESERVED} for m in ms])

    so a column that did not exist in EVERY row was deleted from the whole trajectory. Almost
    nothing on this mesh exists at row 0: `cell_divide` is `after_frame`-gated in most specs, so
    `age` and `ndiv` are created a few frames in; `apop` is created when the first cell is
    sentenced; `scalar_n_apop` when the first one is extruded. Measured on the promotion's own runs:
    `apop_patch_big` and `grow_divide` recorded NONE of `apop`, `age`, `ndiv` -- an apoptosis scene
    with no record that anything died, and a division scene with no record that anything divided.
    The renderer colours dying cells from `apop` and the mother/daughter pair from `age`, so both
    movies came out a plain blue ball, and the mechanism each clip exists to show was invisible.

    IT ALSO BLINDED THE GATES, which is worse than a dull movie. `tools/gate_measures.py` returns
    None for a missing key and substitutes 0.0, so a run that extruded hundreds of cells reported
    `n_apop = 0.0` on every frame and the row asserting `apop_spill` stays near zero could not fail.
    `renumber_failed` -- added so a silent renumber failure would be assertable rather than merely
    printed -- would have been deleted the same way, in exactly the case it exists to catch, since
    it is created only WHEN a renumber fails.

    ZERO IS THE TRUTHFUL FILL, not a placeholder. A row before the first division genuinely has age
    0 and ndiv 0; a row before anything is sentenced genuinely has apop 0; the counters are
    genuinely 0 before they count anything. `_marks` already documents that an unmarked new cell is
    the truthful default, and this makes the record agree with it.

    PER-FACE COLUMNS ARE FILLED TO `nF` because they ride the shared `mesh_face_offsets` and a short
    row would slide every later frame. PER-HALF-EDGE columns keep their OWN offsets and a missing
    row contributes nothing: `snapshot` deliberately records a short myosin array as short, so that
    the alignment check can see it, and padding here would defeat that on the frames it exists for.
    """
    names = set()
    for m in ms:
        names |= {k for k in m if k not in RESERVED}
    scal = sorted(c for c in names if c.startswith("scalar_"))
    edge = sorted(c for c in names if c.startswith("e_"))
    face = sorted(c for c in names if not c.startswith(("scalar_", "e_")))

    def fill(m, col):
        v = m.get(col)
        if col.startswith("scalar_"):
            return 0.0 if v is None else float(v)
        if v is not None:
            return np.asarray(v).ravel()
        if col.startswith("e_"):
            return np.zeros(0, np.float32)
        return np.zeros(int(m["nF"]), np.float32)

    return scal, edge, face, fill


def declare_vertex_carry(m, name):
    """Ask the topology operators to give this per-vertex array a value at every new vertex.

    The per-vertex twin of `junction_ops._face_carry`, and open for the same reason: the set of
    names a topology operator carries was a literal tuple, so a state added by a new operator was
    silently dropped. An operator declares its own array once and `cell_divide` / `cell_die` still
    know nothing about what is in it.
    """
    m.setdefault("vertex_carry", set()).add(name)


def is_mesh(obj) -> bool:
    """A mesh table, or the bare dict that predates it. Both are legal for now: archived runs and
    four operator self-tests build the table as a plain dict, and they must keep working."""
    return isinstance(obj, dict) and "E_face" in obj
