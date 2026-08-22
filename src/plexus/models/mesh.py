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
    SCALAR_RECORD = ("n_t1", "n_apop", "div_blocked", "apop_spill")

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
            v = self.get(nm)
            if v is None:
                continue
            a = _np(v).ravel()
            if a.shape[0] >= nF:                    # a stale short array is dropped, not padded
                out[nm] = a[:nF].astype(np.float32)
        for nm in self.SCALAR_RECORD:
            v = self.get(nm)
            if v is not None and np.ndim(v) == 0:
                out["scalar_" + nm] = float(v)
        return out


def is_mesh(obj) -> bool:
    """A mesh table, or the bare dict that predates it. Both are legal for now: archived runs and
    four operator self-tests build the table as a plain dict, and they must keep working."""
    return isinstance(obj, dict) and "E_face" in obj
