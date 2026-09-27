"""The growth auditor's jump test reads EVERY frame (2026-09-27): a vertex that jumps between two
sampled frames wrecks the run; sampled every 20th frame it was never seen (exp07, finding 54)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import growth_audit as GA  # noqa: E402


def _lattice(T=60, n=6, jump_at=None, nan_at=None):
    """An n x n vertex lattice (edge length 1) for T frames; optionally one vertex displaced by
    3 edge lengths at frame `jump_at` only, or every position NaN from `nan_at` on."""
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2).astype(float)
    P0 = np.concatenate([g, np.zeros((len(g), 1))], 1)
    idx = np.arange(n * n).reshape(n, n)
    E = np.concatenate([np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1),
                        np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], 1)])
    pos = np.repeat(P0[None], T, 0)
    if jump_at is not None:
        pos[jump_at, 7, 0] += 3.0
    if nan_at is not None:
        pos[nan_at:] = np.nan
    return {"vertex__pos": pos, "vertex__mesh_Nv": np.full(T, n * n),
            "vertex__mesh_offsets": np.arange(T + 1) * len(E),
            "vertex__mesh_E_srce": np.tile(E[:, 0], T), "vertex__mesh_E_trgt": np.tile(E[:, 1], T)}


def test_a_clean_run_has_no_jump():
    bad, why, jmax, jat, n = GA._every_frame(_lattice(), 1, 60)
    assert bad is None and jmax == 0.0 and n == 0


def test_a_jump_between_samples_is_caught():
    z = _lattice(jump_at=13)                      # frames 1, 21, 41 are the every-20th samples
    assert max(GA._jump(z, t) for t in range(1, 60, 20)) == 0.0
    bad, why, jmax, jat, n = GA._every_frame(z, 1, 60)
    assert bad == 13 and abs(jmax - 3.0) < 1e-9 and n == 2    # out at 13, back at 14
    assert "3.00 edge lengths" in why


def test_non_finite_is_caught_on_its_frame():
    bad, why, *_ = GA._every_frame(_lattice(nan_at=33), 1, 60)
    assert bad == 33 and why == "non-finite positions"


def test_frames_before_the_settle_window_are_not_read():
    bad, *_ = GA._every_frame(_lattice(jump_at=5), 10, 60)
    assert bad is None


def test_a_strided_record_divides_the_jump_by_its_stride():
    z = _lattice(jump_at=13)                      # 3 edge lengths between two rows 6 frames apart
    bad, why, jmax, jat, n = GA._every_frame(z, 1, 60, stride=6)
    assert bad is None and abs(jmax - 0.5) < 1e-9


def test_a_slot_whose_neighbours_changed_is_not_a_jump():
    z = _lattice(jump_at=13)
    E = np.stack([z["vertex__mesh_E_srce"], z["vertex__mesh_E_trgt"]], 1).copy()
    ne = len(E) // 60
    rows = E.reshape(60, ne, 2)
    rows[13:, 0, 1] = 7                           # from row 13 on, vertex 7 has a new neighbour (a slot re-used)
    z["vertex__mesh_E_srce"], z["vertex__mesh_E_trgt"] = rows[..., 0].ravel(), rows[..., 1].ravel()
    bad, why, jmax, jat, n = GA._every_frame(z, 1, 60)
    assert jat != 13 or jmax < GA.X_JUMP          # the move into row 13 is a topology change, not read
