"""Per-cell state lives on the CELL SET, and a rung that moved it there cannot quietly move back.

WHAT THIS GUARDS. plexus2 says a face of the mesh IS a cell and that per-cell state belongs to the
set whose members are cells, where `Hierarchy.renumber_set` carries it through every topology edit.
The tree grew a second home for it -- an open namespace of columns on the `vertex` set's mesh table,
carried by `MeshTable.reindex_faces` through a name list -- and that second home is where every
defect this campaign has found actually lived: a daughter handed a stranger's cell-cycle phase
because `reindex_faces` CLAMPS an out-of-range index, a growth baseline silently re-taken on most
frames because the reset test was `shape[0] != nF`, a replay that dropped a face column because it
listed four names instead of asking.

So the arrays that have been moved are asserted to be ON the cell set and NOT on the mesh table.
The negative half is the half that matters: an operator can put a column back on the table with one
line and nothing else in the tree would notice.

THE TEST IS A RUN, NOT A GREP, because the mesh table is a `dict` and its contents are whatever the
operators put there. `tools/consistency.py` takes the same census as a campaign progress bar over
every spec in a group; this pins the part that is already done.
"""
from __future__ import annotations

import os
import numpy as np
import pytest

import torch

from plexus import schema                                     # noqa: E402
from plexus.engine import run                                 # noqa: E402

# Moved by S2a (`phase_t`, `cyc_inhib`, `cyc_vprev`), S2b (`phase`) and S2c-1 (`Vbirth`, `divjit`).
# `mg_scale` and the three `*_init` baselines are NOT here: that rung is parked on
# `s2c-2-growth-baseline` because it moves three of `gate_00_spheroid`'s reference pins.
MOVED = {
    "cycle_phases": ["phase", "phase_t", "cyc_inhib", "cyc_vprev", "Vbirth", "divjit"],
    "divide_growing_ball": ["Vbirth", "divjit"],
    "mech_shell_free": ["Vbirth", "divjit"],
}

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def _run(name, frames=8):
    sim = schema.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "tissue", f"{name}.yaml"))
    sim.n_frames = frames
    H, _ = run(sim, out_path=None, device=DEV)
    return H


@pytest.mark.parametrize("name,moved", sorted(MOVED.items()))
def test_moved_state_is_on_the_cell_set_and_not_on_the_mesh(name, moved):
    H = _run(name)
    lvl = H.level("vertex")
    m = getattr(lvl, "mesh", None) or getattr(lvl, "_mesh", None)
    assert m is not None, f"{name}: no mesh table"
    clvl = H.level(lvl.mesh_cell_set)
    for b in moved:
        assert b in clvl.state_schema, (
            f"{name}: {b!r} is not a declared block on {lvl.mesh_cell_set!r}. Per-cell state "
            f"belongs to the cell set -- see this file's header.")
        # STORED, NOT REACHABLE. `MeshTable` SERVES a moved block -- `m["A0"]` still works, and
        # deliberately, because routing the name instead of editing 130 call sites is what made the
        # move testable by byte-identity. What must not exist is a COPY in the table's own dict:
        # that is the thing `reindex_faces` carries by a name list and `snapshot` records, and the
        # thing every defect in this campaign came out of. `dict.__contains__` asks the storage
        # directly, under the proxy.
        assert not dict.__contains__(m, b), (
            f"{name}: {b!r} is stored ON the mesh table again. It belongs to the cell set, where "
            f"`renumber_set` permutes it with everything else; a column here is carried by a name "
            f"list instead, and every name list in this tree has been wrong at least once.")
        assert b in getattr(m, "_proxied", ()), (
            f"{name}: {b!r} is declared on the cell set but the mesh table is not bound to it, so "
            f"a reader asking `m[{b!r}]` gets nothing. See `MeshTable.bind_cell_state`.")


def test_the_census_names_what_is_left_rather_than_claiming_none():
    """The mesh table still holds per-cell arrays, and the count is asserted so it cannot GROW.

    NOT ASSERTED TO BE ZERO, because it is not: `A0`, `P0`, `V0f`, `age`, `alive`, `ndiv` and the
    growth baselines are still columns. A test that demanded zero would have to be skipped, and a
    skipped test guards nothing. It is asserted to be no worse than it is, so a new operator that
    invents a per-cell column has to come here and say so.
    """
    H = _run("divide_growing_ball")
    lvl = H.level("vertex")
    m = getattr(lvl, "mesh", None) or getattr(lvl, "_mesh", None)
    nF = int(m["nF"])
    own = {"E_srce", "E_trgt", "E_face", "nF", "Nv", "n_t1", "n_apop", "n_div", "div_blocked",
           "apop_spill", "renumber_failed", "mono_h", "mono_k", "mono_delta", "buf_full",
           "v_ref", "v_ref_poly", "R0", "face_carry", "vertex_carry", "mech", "centroid_np",
           "apop_marked_once"}
    per_cell = []
    for k in sorted(dict.keys(m)):          # the table's OWN storage, not what it proxies
        if k in own:
            continue
        v = m.get(k)
        a = v.detach().cpu().numpy() if hasattr(v, "detach") else (
            np.asarray(v) if not isinstance(v, (set, dict, str, bool)) else None)
        if a is not None and a.ndim and a.shape[0] == nF:
            per_cell.append(k)
    assert len(per_cell) <= 6, (
        f"the mesh table gained a per-cell column: {per_cell}. Declare it on the cell set "
        f"instead, or raise this bound in the same commit that explains why.")
