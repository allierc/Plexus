#!/usr/bin/env python
"""The carry, tested where the gated run cannot reach.

`edge_flip`'s face-drop branch needs `nF2 != nF` -- a flip that leaves a cell with fewer than three
sides -- and `_ring_ok` (t1_ops.py:71) refuses to commit exactly that, so the branch does not fire in
any run the twin gate covers. Its comment records three real deaths at ticks 86/164/236, so it IS
reachable on a mesh that arrives with `nF` already disagreeing with `E_face`; it is simply not
reachable by asking the operator nicely.

So the fix -- that the OPEN per-face names follow a lost face, which this branch has never done --
is tested by constructing the condition directly. Everything else about the carry is covered by the
bit-for-bit twin run.

    python tools/test_mesh_carry.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "discovery_okuda", "ops"),
          os.path.join(ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

from plexus.models.mesh import MeshTable          # noqa: E402
import mesh_ops                                   # noqa: E402

FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def _mesh(nF=4):
    """A table with three faces' worth of closed list and one open, declared name."""
    ef = torch.repeat_interleave(torch.arange(nF), 3)
    return MeshTable(
        E_srce=torch.arange(3 * nF), E_trgt=torch.arange(3 * nF), E_face=ef, nF=nF, Nv=3 * nF,
        A0=torch.arange(nF, dtype=torch.float32) + 10.0,
        alive=torch.ones(nF), face_carry=["myo_med"],
        myo_med=torch.arange(nF, dtype=torch.float32) + 100.0,
        apop_flag=np.arange(nF, dtype=np.float64) + 1000.0)


def test_open_names_follow_keep():
    print("\nthe open names follow `keep`")
    m = _mesh(4)
    keep = [3, 0, 2]                              # face 1 lost; the rest permuted
    mesh_ops._carry_face_state(m, keep, torch.float32, "cpu")
    check(m["myo_med"].tolist() == [103.0, 100.0, 102.0],
          f"myo_med carried: {m['myo_med'].tolist()}")
    check(m["A0"].tolist() == [10.0, 11.0, 12.0, 13.0],
          "the CLOSED list is untouched by this helper -- its callers carry it themselves")


def test_bare_dict_still_works():
    """Four operator self-tests build fake meshes as plain dicts, and every archived run predates
    the table. A carry that only worked on a `MeshTable` would break all of them."""
    print("\na bare dict is still carried")
    m = dict(face_carry=["myo_med"], myo_med=torch.tensor([1.0, 2.0, 3.0, 4.0]))
    mesh_ops._carry_face_state(m, [3, 1], torch.float32, "cpu")
    check(m["myo_med"].tolist() == [4.0, 2.0], f"plain dict carried: {m['myo_med'].tolist()}")


def test_clamp_not_raise():
    """A short array CLAMPS rather than raising, which is what the helper has always done.
    `medioapical_myosin` raises on the same condition; the two disagree, and changing that is a
    behaviour change that belongs in its own step, not smuggled into a refactor."""
    print("\na short array clamps rather than raising")
    m = _mesh(4)
    m["myo_med"] = torch.tensor([7.0, 8.0])       # shorter than nF
    mesh_ops._carry_face_state(m, [3, 0], torch.float32, "cpu")
    check(m["myo_med"].tolist() == [8.0, 7.0], f"clamped to the last index: {m['myo_med'].tolist()}")


def test_edge_flip_branch_carries_open_names():
    """THE DEFECT THIS STEP FIXES, exercised by hand because the operator will not produce it.

    Before: the branch carried the eight closed names and `apop_flag`, and left `face_carry` behind.
    After: it calls the same carry every other topology operator calls.
    """
    print("\nthe face-drop branch carries the open names")
    src = open(os.path.join(ROOT, "discovery_okuda", "ops", "t1_ops.py")).read()
    i = src.find("a flip left")
    j = src.find("THE CELL STATE AND THE PENDING DELTAS", i)
    branch = src[i:j]
    check("_carry_face_state" in branch,
          "edge_flip's face-drop branch routes through the shared carry")
    check("apop_flag" in branch, "and still carries apop_flag as numpy, as before")

    # and the carry it now calls does the right thing on the shape that branch produces
    m = _mesh(4)
    keep = [0, 2, 3]
    mesh_ops._carry_face_state(m, keep, m["A0"].dtype, m["E_srce"].device)
    check(m["myo_med"].tolist() == [100.0, 102.0, 103.0],
          f"myo_med follows a lost face: {m['myo_med'].tolist()}")


def test_snapshot_is_topology_only():
    print("\nsnapshot records topology and nothing else")
    m = _mesh(3)
    s = m.snapshot()
    check(sorted(s) == ["E_face", "E_srce", "E_trgt", "Nv", "nF"], f"keys: {sorted(s)}")
    check(isinstance(s["nF"], int) and isinstance(s["E_srce"], np.ndarray),
          "counts are ints, arrays are numpy -- the shape every offline reader expects")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("\n" + "=" * 62)
    print(f"  {len(FAIL)} failure(s)" if FAIL else "  all checks passed")
    for f in FAIL:
        print("   - " + f)
    raise SystemExit(1 if FAIL else 0)
