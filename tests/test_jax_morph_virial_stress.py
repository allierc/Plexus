"""Property tests for the `mechanosense` operator (jax-morph VirialStress / virial pressure).

These assert properties statable WITHOUT the reference -- they follow from the definition of the
Irving-Kirkwood virial pressure and the calculus of the pair energy, not from the oracle's numbers:

* COMPRESSION-POSITIVE SIGN (the headline convention) -- two cells overlapping inside contact are
  compressed, so a purely repulsive law (soft_sphere) writes a STRICTLY POSITIVE stress on both.
  This is the minus sign in p_i = -(1/2 d V_i) sum_j r_ij dU/dr: repulsion has dU/dr < 0, so -sum > 0.
  Drop the sign and the mechanosensing signal would invert.
* ANALYTIC VALUE -- for soft_sphere U = 0.5 eps (1 - r/sigma)^2, dU/dr = -eps (1 - r/sigma)/sigma
  (done by hand, NOT read from the source), so the exact per-cell pressure is
  p = eps r (1 - r/sigma) / (sigma * 2 d V) for one neighbour. Matching it verifies the whole
  reduction: the radial projection r*dU/dr, the 1/(2 d) factor, and the d-ball volume V.
* COMPACT-SUPPORT LIMIT -- at and beyond contact (r >= sigma) soft_sphere has dU/dr = 0, so far
  cells contribute nothing: the stress is exactly zero.
* SYMMETRY -- two identical cells feel identical load; a translation/reflection of the pair leaves
  each cell's stress unchanged (it depends only on the separation).
* SIZE NORMALIZATION -- V_i is the cell's OWN d-ball volume: at the same overlap fraction and pair
  force, a larger cell reads a SMALLER pressure (load per unit volume).
* PURE SENSING -- the operator returns {} (moves nothing) and writes only the `stress` field; the
  integrated pos/vel state is untouched.
* DEAD-CELL MASKING -- a dead cell reads 0 stress and does not perturb the live pair's readout.

None of these check agreement with the oracle -- they test the operator's contract.
"""
import math
import types

import torch

from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator

import plexus.operators.candidates.jax_morph_virial_stress  # noqa: F401  (registers mechanosense)


def _world(pos, radius=0.5, occ=None, dim=2, with_stress_block=True):
    """A one-set world: cells at `pos` [N, dim], a per-cell `radius` buffer, pos+vel(+stress) state.

    `with_stress_block=True` declares `stress` as a schema block (the realistic path -- exercises the
    in-place state write + the frame-0-guard exemption); False leaves it to a lazily-made buffer."""
    pos = torch.as_tensor(pos, dtype=torch.float32)
    n = pos.shape[0]
    if with_stress_block:
        state = torch.zeros(n, 2 * dim + 1)
        schema = {"pos": (0, dim), "vel": (dim, 2 * dim), "stress": (2 * dim, 2 * dim + 1)}
    else:
        state = torch.zeros(n, 2 * dim)
        schema = {"pos": (0, dim), "vel": (dim, 2 * dim)}
    state[:, :dim] = pos
    occ = torch.ones(n) if occ is None else torch.as_tensor(occ, dtype=torch.float32)
    lvl = Level("cell", state=state, state_schema=schema, occ=occ)
    rr = torch.full((n,), float(radius)) if isinstance(radius, (int, float)) else torch.as_tensor(radius, dtype=torch.float32)
    lvl.register_buffer("radius", rr)
    H = Hierarchy()
    H.add_level(lvl)
    H.dim = dim
    H.config = types.SimpleNamespace(dt=1.0)
    return H, lvl


def _op(params=None):
    return get_operator("mechanosense")((params or {}), "cpu")


def _read_stress(lvl):
    if "stress" in lvl.state_schema:
        return lvl.get("stress")[:, 0].detach().clone()
    return lvl.stress.detach().clone()


def _soft_sphere_pressure(r, sigma, r_cell, eps, d=2):
    """The exact per-cell virial pressure for ONE overlapping neighbour under the soft_sphere law,
    derived BY HAND (not read from the source): dU/dr = -eps (1 - r/sigma)/sigma for r < sigma, so
    p = -(1/(2 d V)) * r * dU/dr = eps r (1 - r/sigma) / (sigma * 2 d V), with V the d-ball volume."""
    if r >= sigma:
        return 0.0
    du_dr = -eps * (1.0 - r / sigma) / sigma
    vol = {1: 2.0 * r_cell, 2: math.pi * r_cell ** 2, 3: 4.0 / 3.0 * math.pi * r_cell ** 3}[d]
    return -(1.0 / (2.0 * d * vol)) * (r * du_dr)


def test_overlap_reads_positive_stress_with_analytic_value():
    """Two overlapping cells are COMPRESSED -> strictly positive stress, matching the hand-derived
    soft_sphere virial pressure. This pins the minus sign, the 1/(2 d) factor, and the d-ball V."""
    eps, r_cell, s = 1.0, 0.5, 0.5                          # sigma = 1.0, separation 0.5 (overlap)
    H, lvl = _world([[0.0, 0.0], [s, 0.0]], radius=r_cell)
    out = _op({"potential": "soft_sphere", "epsilon": eps})(H, lvl.active)
    assert out == {}                                        # pure sensing: returns no delta
    stress = _read_stress(lvl)
    p = _soft_sphere_pressure(s, 2 * r_cell, r_cell, eps, d=2)
    assert p > 0.0                                          # compression reads POSITIVE
    assert torch.allclose(stress, torch.tensor([p, p]), atol=1e-6)


def test_beyond_contact_reads_zero_stress():
    """At and beyond contact (r >= sigma) the compact soft_sphere law contributes nothing: 0 stress."""
    H, lvl = _world([[0.0, 0.0], [1.2, 0.0]], radius=0.5)   # s = 1.2 > sigma = 1.0
    _op({"potential": "soft_sphere", "epsilon": 1.0})(H, lvl.active)
    assert torch.allclose(_read_stress(lvl), torch.zeros(2), atol=1e-7)
    # exactly at contact (r = sigma) is also zero (compact, continuous):
    H2, lvl2 = _world([[0.0, 0.0], [1.0, 0.0]], radius=0.5)
    _op({"potential": "soft_sphere", "epsilon": 1.0})(H2, lvl2.active)
    assert torch.allclose(_read_stress(lvl2), torch.zeros(2), atol=1e-7)


def test_stress_is_translation_and_reflection_invariant():
    """Stress depends only on the separation: translating or reflecting the pair leaves it unchanged,
    and two identical cells read identical load."""
    eps, r_cell, s = 2.0, 0.5, 0.6
    H, lvl = _world([[0.0, 0.0], [s, 0.0]], radius=r_cell)
    _op({"potential": "soft_sphere", "epsilon": eps})(H, lvl.active)
    base = _read_stress(lvl)
    assert torch.allclose(base[0], base[1], atol=1e-6)      # identical cells -> identical stress
    # translate both by (3, -2) and reflect x -> -x: same separation, same stress.
    H2, lvl2 = _world([[3.0, -2.0], [3.0 - s, -2.0]], radius=r_cell)
    _op({"potential": "soft_sphere", "epsilon": eps})(H2, lvl2.active)
    assert torch.allclose(_read_stress(lvl2), base, atol=1e-6)


def test_larger_cell_reads_smaller_pressure_same_force():
    """V_i is the cell's own volume: at the same overlap FRACTION (same pair force scaled by sigma),
    a bigger cell divides by a bigger volume -> a smaller pressure. Load is per unit volume."""
    eps = 1.0
    # small pair: r_cell 0.5 (sigma 1.0) at r = 0.6  -> fraction 0.6
    Hs, ls = _world([[0.0, 0.0], [0.6, 0.0]], radius=0.5)
    _op({"potential": "soft_sphere", "epsilon": eps})(Hs, ls.active)
    ps = float(_read_stress(ls)[0])
    # big pair: r_cell 1.0 (sigma 2.0) at r = 1.2 -> same fraction 0.6, force ~1/sigma smaller, V 4x bigger
    Hb, lb = _world([[0.0, 0.0], [1.2, 0.0]], radius=1.0)
    _op({"potential": "soft_sphere", "epsilon": eps})(Hb, lb.active)
    pb = float(_read_stress(lb)[0])
    assert ps > 0.0 and pb > 0.0
    assert pb < ps                                          # bigger cell -> smaller pressure
    # matches the hand-derived values in each geometry:
    assert math.isclose(ps, _soft_sphere_pressure(0.6, 1.0, 0.5, eps, d=2), rel_tol=1e-5)
    assert math.isclose(pb, _soft_sphere_pressure(1.2, 2.0, 1.0, eps, d=2), rel_tol=1e-5)


def test_moves_nothing_integrated_state_untouched():
    """Pure sensing: the operator writes only `stress`; pos and vel are byte-identical afterwards."""
    H, lvl = _world([[0.0, 0.0], [0.5, 0.0]], radius=0.5)
    pos_before = lvl.get("pos").clone()
    vel_before = lvl.get("vel").clone()
    _op({"potential": "soft_sphere", "epsilon": 1.0})(H, lvl.active)
    assert torch.equal(lvl.get("pos"), pos_before)
    assert torch.equal(lvl.get("vel"), vel_before)
    assert _read_stress(lvl).abs().sum() > 0.0             # but the stress readout did change


def test_dead_cell_reads_zero_and_does_not_perturb_live_pair():
    """A dead cell reads 0 stress and is excluded as a neighbour source, so the live pair's readout
    equals what it reads with the dead cell absent -- even when the dead cell sits on top of a live one."""
    eps, r_cell, s = 1.0, 0.5, 0.5
    # cells 0,1 overlap and are alive; cell 2 sits exactly on cell 0 but is DEAD.
    H, lvl = _world([[0.0, 0.0], [s, 0.0], [0.0, 0.0]], radius=r_cell, occ=[1, 1, 0])
    _op({"potential": "soft_sphere", "epsilon": eps})(H, lvl.active)
    stress = _read_stress(lvl)
    assert float(stress[2]) == 0.0                          # dead cell reads nothing
    p = _soft_sphere_pressure(s, 2 * r_cell, r_cell, eps, d=2)
    assert torch.allclose(stress[:2], torch.tensor([p, p]), atol=1e-6)   # live pair unperturbed by the phantom


def test_buffer_write_path_when_no_stress_block():
    """When the schema declares no `stress` block, the readout provisions a per-cell buffer and fills
    it -- the same values as the block path."""
    eps, r_cell, s = 1.0, 0.5, 0.5
    H, lvl = _world([[0.0, 0.0], [s, 0.0]], radius=r_cell, with_stress_block=False)
    assert getattr(lvl, "stress", None) is None
    _op({"potential": "soft_sphere", "epsilon": eps})(H, lvl.active)
    p = _soft_sphere_pressure(s, 2 * r_cell, r_cell, eps, d=2)
    assert torch.allclose(lvl.stress, torch.tensor([p, p]), atol=1e-6)
