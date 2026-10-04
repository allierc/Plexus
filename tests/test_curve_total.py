"""`total:<set>:<block>` and `block:<set>:<block>` curve quantities (plexus.measures.curve_row), exp 11 2026-09-29:
the sum over live elements on a level, and on the mesh's cell set, which on a replay is not a level and is read
through the mesh's cell columns."""
import numpy as np

from plexus.measures import curve_row


class _Replay:
    """A replay level: [T, n, w] behind `get`, `_occ` [T, n], `t` picks the row."""
    def __init__(self, val, occ):
        self._val, self._occ, self._pos, self.t = val, occ, np.zeros(val.shape[:2] + (3,)), 0

    def get(self, block):
        return self._val


class _H:
    def __init__(self, levels):
        self.levels = levels

    def level(self, name):
        return self.levels[name]


def test_total_sums_live_elements_of_a_level():
    val = np.zeros((2, 6, 1)); val[1, :, 0] = [1, 2, 3, 4, 5, 6]
    occ = np.zeros((2, 6), bool); occ[1, :4] = True
    rl = _Replay(val, occ); rl.t = 1
    H = _H({"bm_node": rl})
    assert tuple(curve_row(H, rl, "total:bm_node:M", None, 1, None)[0]) == (10.0, 0.0)
    mu, sd = curve_row(H, rl, "block:bm_node:M", None, 1, None)[0]
    assert np.isclose(mu, 2.5) and np.isclose(sd, np.std([1, 2, 3, 4]))


def test_total_on_the_cell_set_reads_the_mesh_cell_columns():
    class Mesh:
        mesh = {"nF": 3}
    lvl = Mesh()
    cols = {"itg_A": np.array([0.5, 1.5, 2.0, 99.0])}                 # the 4th slot is past nF: not live

    def cell_cols(H, lvl_, nF):
        return {k: v[:nF] for k, v in cols.items()}
    H = _H({})                                                          # no `cell` level, as on a replay
    assert tuple(curve_row(H, lvl, "total:cell:itg_A", None, 1, cell_cols)[0]) == (4.0, 0.0)
    assert np.isclose(curve_row(H, lvl, "block:cell:itg_A", None, 1, cell_cols)[0, 0], 4.0 / 3)
    assert np.isnan(curve_row(H, lvl, "total:cell:missing", None, 1, cell_cols)[0, 0])


def test_total_weighted_by_a_second_block_is_an_amount():
    class Mesh:
        mesh = {"nF": 2}
    cols = {"itg_A": np.array([0.5, 2.0]), "area": np.array([4.0, 1.0])}
    H = _H({})
    row = curve_row(H, Mesh(), "total:cell:itg_A@area", None, 1, lambda H_, l_, nF: {k: v[:nF] for k, v in cols.items()})
    assert tuple(row[0]) == (4.0, 0.0)
