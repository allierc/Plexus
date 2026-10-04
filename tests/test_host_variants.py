"""The host-side variant of exp 15, rig 4, against its default: an IDENTITY case where it must agree
with the default and a PLANTED case where it must differ (experiments/INSTRUCTION.md, "Code: reuse the
operator, add a variant ...").

    cell_chem_react[clock_turnover]   (default here: cell_chem_react[source_decay], the source everywhere)

    PYTHONPATH=src python -m pytest tests/test_host_variants.py -q
"""
import math
import os
import tempfile

import numpy as np
import torch
import yaml

import plexus.operators  # noqa: F401
import plexus.schema as S
from plexus.engine import build, seed as engine_seed
from plexus.models.registry import get_operator

torch.set_num_threads(1)


def host(n=16, A0=1.0, phase=0.0, signal=None):
    d = {"general": {"name": "t", "seed": 0, "n_frames": 1, "dt": 0.02, "dim": 2, "boundary": "free",
                     "world": [10.0, 10.0]},
         "sets": {"host": {"n": n, "state": {
             "pos": {"width": 2, "integration": "none", "boundary": "free"},
             "centroid": {"width": 2, "integration": "none", "boundary": "free"},
             "chem": {"width": 1, "integration": "first_order", "boundary": "free"},
             "phase": {"width": 1, "integration": "first_order", "boundary": "free"},
             "signal": {"width": 3, "integration": "none", "boundary": "free"}}}},
         "fields": {}, "seed": [], "operators": [], "schedule": []}
    f = os.path.join(tempfile.mkdtemp(), "s.yaml")
    yaml.safe_dump(d, open(f, "w"))
    sim = S.load(f)
    H = build(sim, device="cpu")
    engine_seed(H, sim, device="cpu")
    lvl = H.level("host")
    st = lvl.state.clone()
    sc = lvl.state_schema
    st[:, slice(*sc["chem"])] = A0
    st[:, slice(*sc["phase"])] = torch.as_tensor(phase, dtype=st.dtype).reshape(-1, 1) if not np.isscalar(phase) else phase
    st[:, slice(*sc["centroid"])] = torch.rand(n, 2)
    if signal is not None:
        st[:, slice(*sc["signal"])] = torch.as_tensor(signal, dtype=st.dtype)
    lvl.state = st
    H.zero_delta()
    return H, lvl


def op(name, variant=None, model=None, **p):
    return get_operator(name, variant, model)({"_at": "host", **p}, "cpu")


def test_clock_turnover_identity_no_signal_no_gate_is_source_decay_everywhere():
    """g = 0 and floor 1: dA/dt = k_in - k_out A, which is `source_decay` with every cell a source."""
    H, lvl = host(A0=torch.linspace(0.1, 3.0, 16).reshape(-1, 1), phase=torch.linspace(0, 6, 16))
    ct = op("cell_chem_react", None, "clock_turnover", k_in=0.7, k_out=0.3, rate=2.0)(H, None)["host"]
    sd = op("cell_chem_react", None, "source_decay", production=0.7, decay=0.3, rate=2.0,
            source={"axis": 0, "side": "low", "frac": 2.0})(H, None)["host"]
    assert torch.allclose(ct, sd, atol=1e-6)


def test_clock_turnover_planted_signal_and_clock_set_the_deacetylation():
    """Strain 0 at fraction m = 0.3 (weights [1, 0, 0]), K = 0.1 -> s = 0.75; phase pi/2 with floor
    0.2 -> gamma = 0.2 + 0.8 (1 + cos(pi/2)) / 2 = 0.6; A = 2: dA/dt = 1 - 0.5 (1 + 2 x 0.75) 0.6 x 2."""
    H, lvl = host(A0=2.0, phase=math.pi / 2, signal=[0.3, 0.6, 0.1])
    d = op("cell_chem_react", None, "clock_turnover", k_in=1.0, k_out=0.5, g=2.0, K=0.1, weights=[1, 0, 0],
           gate={"block": "phase", "floor": 0.2})(H, None)["host"]
    want = 1.0 - 0.5 * (1 + 2 * 0.75) * 0.6 * 2.0
    assert torch.allclose(d, torch.full_like(d, want), atol=1e-5)
    dflt = op("cell_chem_react", None, "source_decay", production=1.0, decay=0.5,
              source={"axis": 0, "side": "low", "frac": 2.0})(H, None)["host"]
    assert not torch.allclose(d, dflt)


def test_clock_turnover_planted_signal_lowers_the_mean_and_raises_the_daily_swing():
    """Integrated over days with the clock turning: a host with the signal has a LOWER mean A and a
    LARGER relative daily amplitude than a germ-free one (Kuang 2019 Fig. 1C-D's direction)."""
    day, n = 10.0, 200
    res = {}
    for m in (0.0, 0.3):
        H, lvl = host(n=1, A0=1.0, signal=[m, 0.0, 0.0])
        o = op("cell_chem_react", None, "clock_turnover", k_in=1.0, k_out=0.864, g=2.094, K=0.05,
               weights=[1, 0, 0], gate={"block": "phase", "floor": 0.2076})
        dt, tr = day / n, []
        sc = lvl.state_schema
        for k in range(30 * n):
            st = lvl.state.clone()
            st[:, slice(*sc["phase"])] = 2 * math.pi * k / n
            lvl.state = st
            dA = o(H, None)["host"]
            st = lvl.state.clone(); st[:, slice(*sc["chem"])] += dt * dA; lvl.state = st
            if k >= 29 * n:
                tr.append(float(lvl.get("chem")[0, 0]))
        tr = np.asarray(tr)
        res[m] = (tr.mean(), (tr.max() - tr.min()) / (2 * tr.mean()))
    (mg, ag), (mc, ac) = res[0.0], res[0.3]
    print("germ-free mean %.3f amp %.3f | signal mean %.3f amp %.3f" % (mg, ag, mc, ac))
    assert mg / mc > 2.0 and ag < ac


def test_clock_turnover_gate_signal_identity_same_gate_is_the_one_gate_model():
    """`gate_signal` equal to `gate` reproduces the one-gate equation exactly."""
    H, lvl = host(A0=torch.linspace(0.5, 2.5, 16).reshape(-1, 1), phase=torch.linspace(0, 6, 16),
                  signal=[0.2, 0.3, 0.5])
    kw = dict(k_in=1.0, k_out=0.6, g=3.0, K=0.1, weights=[1, 0, 0], gate={"block": "phase", "floor": 0.3})
    one = op("cell_chem_react", None, "clock_turnover", **kw)(H, None)["host"]
    two = op("cell_chem_react", None, "clock_turnover", gate_signal={"block": "phase", "floor": 0.3}, **kw)(H, None)["host"]
    assert torch.allclose(one, two, atol=1e-7)


def test_clock_turnover_gate_signal_planted_the_induced_term_keeps_its_own_clock():
    """Basal floor 0.5, induced floor 0 at phase pi (cos = -1): gamma_0 = 0.5, gamma_1 = 0 -- the microbes
    add nothing at that hour: dA/dt = 1 - 0.6 x 0.5 x A, whatever the signal."""
    H, lvl = host(A0=2.0, phase=math.pi, signal=[0.9, 0.0, 0.1])
    d = op("cell_chem_react", None, "clock_turnover", k_in=1.0, k_out=0.6, g=5.0, K=0.1, weights=[1, 0, 0],
           gate={"block": "phase", "floor": 0.5}, gate_signal={"block": "phase", "floor": 0.0})(H, None)["host"]
    assert torch.allclose(d, torch.full_like(d, 1.0 - 0.6 * 0.5 * 2.0), atol=1e-5)
