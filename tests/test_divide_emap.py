"""`divide_face_3d` maintains the edge->face map instead of rebuilding it, and the two agree exactly.

WHY THIS MATTERS. The function needs two lookups out of an edge->face dict and used to rebuild the
whole O(E) map for them, once per dividing cell. Both the edge count and the number of cells
ripening on a tick grow with the tissue, so `cell_divide` was quadratic in the thing the run grows:
profiled on `mesh_mpm_spheroid_nominal` at frame 380, `_edge_face_map` was 19.4 s of the operator's
24.2 s across 5,480 rebuilds in 96 frames. That is the term behind 246 ms/frame at frame 50 becoming
85 s/frame at frame 700, and behind four gpu_l4 jobs dying on the wall clock at frame 736 of 801.

WHAT IS ASSERTED. Not "the map looks right" -- that a MAINTAINED map and a REBUILT one produce the
same topology, division for division. `_edge_face_map` is a pure function of `rings`, so the
maintained dict must equal a fresh rebuild after every single division; and two runs of the same
division sequence, one down each path, must end with identical rings and identical vertices. The
second is the property the trajectory depends on: division is discrete, so a single disagreement
about which neighbour owns an edge sends the two runs to different tissues, not to nearby ones.
"""
import numpy as np
import pytest

from plexus.models.topology import _edge_face_map, divide_face_3d
from plexus.operators.vertex_ops import build_sphere_mesh


def _mesh(n=80, seed=0):
    """A closed sphere mesh as (rings, pos) -- the representation `cell_divide` works in."""
    from plexus.models.topology import rings_from_flat_3d
    pos_np, es, et, ef, nF = _flat(n, seed)
    return rings_from_flat_3d(es, et, ef, nF), [p for p in pos_np]


def _flat(n, seed):
    out = build_sphere_mesh(n, r=1.0, jitter=0.05, seed=seed)
    if isinstance(out, dict):
        return (np.asarray(out["pos"], np.float64), np.asarray(out["E_srce"]),
                np.asarray(out["E_trgt"]), np.asarray(out["E_face"]), int(out["nF"]))
    pos, es, et, ef, nF = out[:5]
    return np.asarray(pos, np.float64), np.asarray(es), np.asarray(et), np.asarray(ef), int(nF)


def _divide_sequence(rings, pos, faces, maintained):
    """Divide `faces` in order. `maintained` picks the carried map or the per-call rebuild."""
    emap = _edge_face_map(rings) if maintained else None
    done = []
    for f in faces:
        if f >= len(rings) or rings[f] is None or len(rings[f]) < 4:
            continue
        res = divide_face_3d(rings, pos, f, emap=emap)
        done.append(None if res is None else res[0])
        if maintained and res is not None:
            # THE INVARIANT, CHECKED AFTER EVERY DIVISION rather than once at the end: a map that
            # goes stale on division 3 and is repaired by luck on division 4 would pass a final
            # comparison and still have handed division 3 the wrong neighbour.
            assert emap == _edge_face_map(rings), f"map diverged from a rebuild at face {f}"
    return done


def test_maintained_map_equals_rebuild_every_step():
    rings, pos = _mesh()
    faces = list(range(0, len(rings), 3))
    _divide_sequence(rings, pos, faces, maintained=True)     # the assert lives inside the loop


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_same_topology_either_path(seed):
    """The whole point: carrying the map does not change a single division outcome."""
    r_old, p_old = _mesh(seed=seed)
    r_new, p_new = _mesh(seed=seed)
    faces = list(range(0, len(r_old), 3))
    got_old = _divide_sequence(r_old, p_old, faces, maintained=False)
    got_new = _divide_sequence(r_new, p_new, faces, maintained=True)

    assert got_old == got_new, "a division succeeded on one path and not the other"
    assert len(r_old) == len(r_new), f"{len(r_old)} faces rebuilt vs {len(r_new)} maintained"
    assert r_old == r_new, "the rings differ"
    assert len(p_old) == len(p_new)
    np.testing.assert_allclose(np.asarray(p_old), np.asarray(p_new), rtol=0, atol=0)
    # the sequence has to actually DO something, or all three assertions above are vacuous
    assert sum(x is not None for x in got_new) > 10, "too few divisions to be a test"
