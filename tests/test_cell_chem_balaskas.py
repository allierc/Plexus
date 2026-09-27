"""exp 7's two `cell_chem_react` models and one `seed_cell_chem` model, against numbers read off the paper.

    cell_chem_react[balaskas]      the Pax6-Olig2-Nkx2.2 circuit (Balaskas et al. 2012, eqs. 1-3, Table S2):
                                   one cell integrated to t = 20 from (P, O, N) = (3, 0, 0) switches fate at
                                   the G of the paper's Fig. 4B, wild type and the two one-line mutants
    cell_chem_react[source_decay]  the source mask is the declared fraction of the extent, and with
                                   cell_chem_diffuse on a chain of cells the steady decay length is
                                   sqrt(D_eff / k), D_eff = d chi a^2 / 2 on a 1D chain of spacing a
    seed_cell_chem[uniform]        every live cell gets the declared values, other columns untouched

    PYTHONPATH=src python -m pytest tests/test_cell_chem_balaskas.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.operators.diffusion_reaction import (CellDiffuse, CellRDSeedUniform,  # noqa: E402
                                                 CellReactBalaskas, CellReactBalaskasAdapt, CellReactBalaskasCommit,
                                                 CellReactBalaskasSchedule,
                                                 CellReactSourceDecay)


class Lvl:
    """A cell level: named blocks, an occupancy mask, an edge index."""

    def __init__(self, blocks, occ=None, edge_index=None):
        self.blocks, self.occ, self.edge_index = blocks, occ, edge_index

    def get(self, name):
        return self.blocks[name]


class H:
    def __init__(self, lvl):
        self.lvl, self.levels = lvl, {}

    def level(self, name):
        return self.lvl


def switches(G, **params):
    """Fate switches (the G where the highest factor changes) after t = 20 from (3, 0, 0), Euler dt 1e-3."""
    m = CellReactBalaskas(dict(g_col=3, **params))
    G = torch.as_tensor(G, dtype=torch.float64)
    P, O, N = torch.full_like(G, 3.0), torch.zeros_like(G), torch.zeros_like(G)
    for _ in range(20000):
        dP, dO, dN = m.rates(P, O, N, G)
        P, O, N = P + 1e-3 * dP, O + 1e-3 * dO, N + 1e-3 * dN
    f = torch.stack([P, O, N], 1).argmax(1).numpy()
    g = G.numpy()
    return [(float(g[i]), int(f[i - 1]), int(f[i])) for i in range(1, len(g)) if f[i] != f[i - 1]], (P, O, N)


GRID = np.arange(0.05, 3.01, 0.05)


def test_wild_type_switches_where_fig4Bi_does():
    sw, (P, O, N) = switches(GRID)
    assert [(a, b) for _, a, b in sw] == [(0, 1), (1, 2)]              # P -> O -> N, nested
    assert abs(sw[0][0] - 0.3) <= 0.1                                  # Fig. 4Bi: P / O cross at G ~ 0.3
    assert abs(sw[1][0] - 2.2) <= 0.1                                  # Fig. 4Bi: N takes over at G ~ 2.2
    assert abs(float(O.max()) - 2.2) < 0.15                            # Fig. 4Bi: O peaks ~ 2.2 near G 1.3


def test_pax6_mutant_alpha_zero_switches_where_fig4Biii_does():
    sw, _ = switches(GRID, alpha=0.0)
    assert sw[-1][2] == 2 and abs(sw[-1][0] - 0.8) <= 0.1               # Fig. 4Biii: N at G ~ 0.8


def test_olig2_mutant_beta_zero_switches_where_fig4Bii_does():
    sw, _ = switches(GRID, beta=0.0)
    assert [(a, b) for _, a, b in sw] == [(0, 2)]                       # no Olig2 domain at all
    assert abs(sw[0][0] - 1.33) <= 0.1                                 # Fig. 4Bii: N at G ~ 1.33


def test_signal_column_must_be_outside_the_span():
    with pytest.raises(ValueError):
        CellReactBalaskas(dict(g_col=1, chan=0))


def test_balaskas_writes_only_its_own_columns():
    chem = torch.rand(10, 4, dtype=torch.float64) * 3
    out = CellReactBalaskas(dict(g_col=3)).forward(H(Lvl({"chem": chem})))["cell"]
    assert torch.all(out[:, 3] == 0) and torch.any(out[:, :3] != 0)


def test_source_mask_is_the_declared_fraction_of_the_extent():
    x = torch.linspace(-3.0, 7.0, 101, dtype=torch.float64)
    cen = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], 1)
    occ = torch.ones(101, dtype=torch.bool)
    occ[0] = False                                                      # a dead slot does not set the extent
    chem = torch.zeros(101, 4, dtype=torch.float64)
    chem[:, 3] = 1.0
    lvl = Lvl({"chem": chem, "centroid": cen}, occ=occ.to(torch.float64))
    m = CellReactSourceDecay(dict(production=2.0, decay=0.5, chan=3, source={"axis": 0, "side": "low", "frac": 0.1}))
    src = m.source_mask(lvl)
    live_x = x[occ]
    L = float(live_x.max() - live_x.min())
    assert torch.equal(src, occ & (x - live_x.min() < 0.1 * L))
    out = m.forward(H(lvl))["cell"]
    assert torch.allclose(out[src, 3], torch.full_like(out[src, 3], 2.0 - 0.5))
    assert torch.allclose(out[occ & ~src, 3], torch.full_like(out[occ & ~src, 3], -0.5))
    assert torch.all(out[:, :3] == 0) and out[0, 3] == 0                 # other columns, dead slot


def test_source_and_diffusion_give_sqrt_D_over_k_on_a_chain():
    n, a, d, k = 300, 0.5, 0.5, 0.01
    x = torch.arange(n, dtype=torch.float64) * a
    cen = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], 1)
    i = torch.arange(n - 1)
    ei = torch.stack([torch.cat([i, i + 1]), torch.cat([i + 1, i])])
    chem = torch.zeros(n, 1, dtype=torch.float64)
    lvl = Lvl({"chem": chem, "centroid": cen}, edge_index=ei)
    src = CellReactSourceDecay(dict(production=1.0, decay=k, source={"axis": 0, "side": "low", "frac": 0.03}))
    dif = CellDiffuse(dict(d=[d], chi=1.0))
    for _ in range(3000):                                               # 30 decay times
        lvl.blocks["chem"] = lvl.blocks["chem"] + src.forward(H(lvl))["cell"] + dif.forward(H(lvl))["cell"]
    c = lvl.blocks["chem"][:, 0].numpy()
    far = (x.numpy() > 0.1 * n * a) & (x.numpy() < 0.5 * n * a)
    lam = -1.0 / np.polyfit(x.numpy()[far], np.log(c[far]), 1)[0]
    assert abs(lam / np.sqrt(d * a ** 2 / 2 / k) - 1) < 0.03


def test_uniform_seed_writes_every_live_cell_and_nothing_else():
    class SeedLvl:
        state_schema = {"area": (0, 1), "chem": (1, 5)}

        def __init__(self):
            self.state = torch.full((6, 5), 7.0)
            self.occ = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.float64)

    lvl = SeedLvl()
    s = CellRDSeedUniform(dict(values=[3.0, 0.0, 0.0, 0.0]))
    s.forward(H(lvl))
    assert torch.all(lvl.state[:4, 1] == 3.0) and torch.all(lvl.state[:4, 2:5] == 0.0)
    assert torch.all(lvl.state[:, 0] == 7.0) and torch.all(lvl.state[4:, 1:] == 7.0)
    with pytest.raises(ValueError):
        CellRDSeedUniform(dict())


def _adapt_cell(peak_G, b, T):
    """One cell under the paper's G(t) = a t^2 e^(-b t), written as peak x f(t); Euler dt 1e-3."""
    m = CellReactBalaskasAdapt(dict(g_col=3, t_peak=2.0 / b))
    P, O, N = (torch.tensor([v], dtype=torch.float64) for v in (3.0, 0.0, 0.0))
    t = 0.0
    for _ in range(int(T / 1e-3)):
        G = torch.tensor([peak_G * m.profile(t)], dtype=torch.float64)
        dP, dO, dN = m.rates(P, O, N, G)
        P, O, N = P + 1e-3 * dP, O + 1e-3 * dO, N + 1e-3 * dN
        t += 1e-3
    return float(P), float(O), float(N)


@pytest.mark.parametrize("a,b,fate", [(0.16, 0.16, 2), (0.05, 0.16, 1), (0.1, 0.4, 0)])
def test_adapting_input_reproduces_fig_S7A_at_t20(a, b, fate):
    """Fig. S7A: high (a 0.16, b 0.16) -> p3, Nkx2.2 ~3.4; medium (0.05, 0.16) -> pMN, Olig2 ~2.1 with
    Pax6 ~0.6; low (0.1, 0.4) -> p2, Pax6 back to ~3. The peak of a t^2 e^(-b t) is a (2 / b)^2 e^-2."""
    P, O, N = _adapt_cell(a * (2.0 / b) ** 2 * np.exp(-2.0), b, 20.0)
    assert int(np.argmax([P, O, N])) == fate
    if fate == 2:
        assert abs(N - 3.4) < 0.3
    if fate == 1:
        assert abs(O - 2.1) < 0.2 and abs(P - 0.6) < 0.2


def test_adapting_input_continued_erases_the_pattern():
    """Beyond the paper's window the same profile falls below Nkx2.2's maintenance level: by t = 60 the
    p3 cell is back to Pax6 -- the hysteresis holds only while G stays up."""
    P, O, N = _adapt_cell(0.16 * 12.5 ** 2 * np.exp(-2.0), 0.16, 60.0)
    assert P > 2.5 and N < 0.2


def test_adapt_profile_peaks_at_one():
    m = CellReactBalaskasAdapt(dict(g_col=3, t_peak=12.5))
    assert abs(m.profile(12.5) - 1.0) < 1e-12 and m.profile(0.0) == 0.0 and m.profile(25.0) < 1.0


class _Clock:
    def __init__(self, frame, lvl):
        self.frame_t, self.dt, self._lvl = torch.tensor(float(frame)), 1.0, lvl

    def level(self, name):
        return self._lvl


def test_commit_freezes_the_circuit_after_t_commit():
    chem = torch.tensor([[2.0, 0.0, 0.0, 3.0, 0.0, 0.0]], dtype=torch.float64)
    m = CellReactBalaskasCommit(dict(g_col=0, chan=3, rate=0.1, t_commit=20.0))
    before = m.forward(_Clock(150, Lvl({"chem": chem})))["cell"]            # t = 15
    after = m.forward(_Clock(250, Lvl({"chem": chem})))["cell"]             # t = 25
    assert torch.any(before[:, 3:] != 0) and torch.all(after == 0)


def test_committed_p3_cell_keeps_nkx22_where_the_adapting_one_loses_it():
    """One p3 cell under the Fig. S7A high profile: committed at t = 20 it is still Nkx2.2 at t = 60."""
    m = CellReactBalaskasCommit(dict(g_col=3, t_peak=12.5, t_commit=20.0))
    peak = 0.16 * 12.5 ** 2 * np.exp(-2.0)
    P, O, N = (torch.tensor([v], dtype=torch.float64) for v in (3.0, 0.0, 0.0))
    t = 0.0
    for _ in range(60000):
        if t < m.t_commit:
            G = torch.tensor([peak * m.profile(t)], dtype=torch.float64)
            dP, dO, dN = m.rates(P, O, N, G)
            P, O, N = P + 1e-3 * dP, O + 1e-3 * dO, N + 1e-3 * dN
        t += 1e-3
    assert float(N) > 3.0 and float(P) < 0.1


# ---- identity cases (INSTRUCTION.md: every variant agrees with its base where it must) ----

def test_identity_source_decay_with_nothing_to_do_is_no_reaction():
    """production 0 and decay 0: the delta is exactly zero -- the spec is chemistry-free, as without the operator."""
    x = torch.linspace(0.0, 5.0, 20, dtype=torch.float64)
    cen = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], 1)
    chem = torch.rand(20, 1, dtype=torch.float64)
    m = CellReactSourceDecay(dict(production=0.0, decay=0.0))
    assert torch.all(m.forward(H(Lvl({"chem": chem, "centroid": cen})))["cell"] == 0)


def test_identity_balaskas_rest_state_without_signal_is_stationary():
    """G = 0 at the no-signal state (P, O, N) = (alpha / k1, 0, 0) = (3, 0, 0): nothing moves."""
    chem = torch.tensor([[3.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    out = CellReactBalaskas(dict(g_col=3)).forward(H(Lvl({"chem": chem})))["cell"]
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-12)


def test_identity_adapt_at_its_peak_is_balaskas():
    """f(t_peak) = 1: at t = t_peak balaskas_adapt's delta equals balaskas's, column for column."""
    chem = torch.rand(8, 6, dtype=torch.float64) * 3
    base = CellReactBalaskas(dict(g_col=0, chan=3, rate=0.1)).forward(_Clock(125, Lvl({"chem": chem})))["cell"]
    adapt = CellReactBalaskasAdapt(dict(g_col=0, chan=3, rate=0.1, t_peak=12.5)).forward(_Clock(125, Lvl({"chem": chem})))["cell"]
    assert torch.allclose(base, adapt, atol=1e-12)


def test_identity_commit_before_t_commit_is_adapt():
    chem = torch.rand(8, 6, dtype=torch.float64) * 3
    a = CellReactBalaskasAdapt(dict(g_col=0, chan=3, rate=0.1)).forward(_Clock(150, Lvl({"chem": chem})))["cell"]
    c = CellReactBalaskasCommit(dict(g_col=0, chan=3, rate=0.1, t_commit=20.0)).forward(_Clock(150, Lvl({"chem": chem})))["cell"]
    assert torch.equal(a, c)


def test_identity_schedule_empty_is_balaskas():
    chem = torch.rand(8, 6, dtype=torch.float64) * 3
    a = CellReactBalaskas(dict(g_col=0, chan=3, rate=0.1)).forward(_Clock(150, Lvl({"chem": chem})))["cell"]
    b = CellReactBalaskasSchedule(dict(g_col=0, chan=3, rate=0.1)).forward(_Clock(150, Lvl({"chem": chem})))["cell"]
    assert torch.equal(a, b)


def test_schedule_halves_the_input_after_its_time():
    """Planted: after t = 8.6 the input is half -- the delta equals balaskas's at half the morphogen."""
    chem = torch.rand(8, 6, dtype=torch.float64) * 3
    half = chem.clone(); half[:, 0] *= 0.5
    m = CellReactBalaskasSchedule(dict(g_col=0, chan=3, rate=0.1, schedule=[[8.6, 0.5]]))
    base = CellReactBalaskas(dict(g_col=0, chan=3, rate=0.1))
    before = m.forward(_Clock(80, Lvl({"chem": chem})))["cell"]
    after = m.forward(_Clock(100, Lvl({"chem": chem})))["cell"]
    assert torch.allclose(before, base.forward(_Clock(80, Lvl({"chem": chem})))["cell"])
    assert torch.allclose(after[:, 3:], base.forward(_Clock(100, Lvl({"chem": half})))["cell"][:, 3:])
