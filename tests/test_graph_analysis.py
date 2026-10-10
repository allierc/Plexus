"""The graph phase end to end on a planted law (plexus.tasks.graph_analysis; exp17, 2026-10-10): a neuron-graph law of
900 elements on a synthetic recording, its fitted values written by hand. Identity: with every edge weight 0 the
silenced rollouts equal the nominal one, the spectrum is the leak's, a pulse stays in its region. Planted: a chain of
strong middle edges from one region into another carries a pulse across, pruning above its weight silences the law,
the relay identity (W = 0 with and without the inputs clamped) holds on the free neurons, every test writes its
section of results/<run>_graph.json and the card is drawn."""
import json
import os
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import plexus.operators  # noqa: E402,F401
import plexus.paths as PP  # noqa: E402
from plexus import trainer as T  # noqa: E402
from plexus.models.registry import get_contract  # noqa: E402
from plexus.tasks import graph_analysis as G  # noqa: E402
from plexus.tasks import trace_recording as TR  # noqa: E402

N, F, TF = 900, 4, 240
OFF = [0, 80, 160, 240]
CLS = get_contract("state_diffuse").implementations["neuron_graph"]


def _recording(pos, rng):
    """A shared slow sine, per-element AR(1) noise, the first 120 elements driven by two varying stimulus columns; two
    more columns are block markers (they change at most twice)."""
    t = np.arange(TF)
    stim = np.zeros((TF, F), np.float32)
    stim[:, 0] = (np.sin(2 * np.pi * t / 40) > 0)
    stim[:, 1] = (t // 10) % 2
    stim[:, 2] = (t >= 80) & (t < 160)
    stim[:, 3] = t >= 160
    X = np.zeros((TF, N), np.float32)
    ar = np.zeros(N)
    for k in range(TF):
        ar = 0.8 * ar + 0.01 * rng.normal(size=N)
        X[k] = 0.1 + 0.02 * np.sin(2 * np.pi * k / 60) + ar
    X[:, :120] += 0.05 * stim[:, :1] + 0.03 * stim[:, 1:2]
    cond = np.zeros(TF, np.int8)
    cond[80:160], cond[160:] = 1, 2
    return {"dff": X, "stimulus": stim, "condition": cond, "names": ["a", "b", "c"], "offsets": np.array(OFF),
            "pos_um": pos.astype(np.float32), "t_s": t * 0.914}


@pytest.fixture
def planted(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 1, (N, 3)) * [300.0, 400.0, 120.0]
    pos_f = str(tmp_path / "pos.npz")
    np.savez(pos_f, pos_um=pos.astype(np.float32), offsets=np.array(OFF), names=np.array(["a", "b", "c"]))
    rec = _recording(pos, rng)
    monkeypatch.setattr(TR, "load", lambda name="x": rec)
    PP.set_data_root(str(tmp_path))
    reg = np.stack([pos[:, 0] < 100, (pos[:, 0] >= 100) & (pos[:, 0] < 200), pos[:, 0] >= 200], 1)
    reg_f = str(tmp_path / "regions.npz")
    np.savez(reg_f, regions=reg, names=np.array(["front", "middle", "back"]), inside=np.ones(N, bool), atlas_um=pos)
    opp = {"op": "state_diffuse", "model": "neuron_graph", "integrator": "exponential", "at": "neuron", "block": "dff",
           "positions": "xyz", "positions_file": pos_f, "short_k": 6, "mid_um": 150.0, "long_um": 0.0, "substeps": 1,
           "inputs": 1, "forcing": "stimulus.u", "forcing_dim": F, "seed": 0, "checkpoint": False}
    st_ = lambda w, **kw: {"width": w, "integration": "none", "boundary": "free", "record": False, **kw}   # noqa: E731
    model = {"general": {"name": "g_model", "seed": 0, "n_frames": 1, "dt": 1.0, "boundary": "wall", "dim": 3,
                         "world": [1.0, 1.0, 1.0], "save_data": False},
             "sets": {"brain": {"n": 1},
                      "neuron": {"parent": "brain", "per_parent": N, "grow_reserve": 0,
                                 "state": {"dff": {"width": 1, "role": "coordinate", "integration": "first_order", "boundary": "free"},
                                           "xyz": st_(3), "tau": st_(1), "rest": st_(1), "input": st_(F)}},
                      "stimulus": {"parent": "brain", "per_parent": F, "grow_reserve": 0, "state": {"u": st_(1)}}},
             "fields": {},
             "seed": [{"op": "seed_state_from_file", "at": "neuron", "file": pos_f, "blocks": {"xyz": "pos_um"}},
                      {"op": "seed_state_random", "at": "neuron", "block": "tau", "lo": -2.97, "hi": -2.97},
                      {"op": "seed_state_random", "at": "neuron", "block": "rest", "lo": 0.0, "hi": 0.0},
                      {"op": "seed_state_random", "at": "neuron", "block": "input", "lo": 0.0, "hi": 0.0},
                      {"op": "seed_state_random", "at": "stimulus", "block": "u", "lo": 0.0, "hi": 0.0}],
             "operators": [opp], "schedule": ["state_diffuse"], "plotting": {"renderer": "none", "max_frames": 0}}
    mp = tmp_path / "g_model.yaml"
    mp.write_text(yaml.safe_dump(model))
    tdir = tmp_path / "training" / "planted"
    tdir.mkdir(parents=True)
    tr = {"name": "g_run", "model": str(mp),
          "learnable": [{"param": "W_short", "op": "state_diffuse", "lr": 1e-3}, {"param": "W_mid", "op": "state_diffuse", "lr": 1e-3},
                        {"block": "tau", "of": "neuron", "lr": 1e-3}, {"block": "rest", "of": "neuron", "lr": 1e-3},
                        {"block": "input", "of": "neuron", "lr": 1e-3}],
          "task": {"reference": {"trace_recording": "planted", "split": "all", "test_stride": 8},
                   "drive": {"set": "stimulus", "block": "u", "window": [0, 0]}, "observe": {"set": "neuron", "block": "dff"},
                   "loss": [{"term": "trace_mse", "reduction": "norm2"}], "warmup": 2,
                   "graph": {"regions": reg_f, "steady_skip_min": 0.1, "bootstrap": {"blocks": 12, "resamples": 200, "seed": 0},
                             "prune": {"ladder": 3, "span": [0.5, 2.0]}, "prune_b": {"quantiles": [0.5]},
                             "spectrum": {"k": 6, "k_imag": 4}, "pulse": {"level_z": 2.0, "duration_s": 3.0, "settle_s": 10.0, "window_s": 20.0},
                             "movies": False, "every_frame": True}},
          "training": {"optimizer": "adam", "lr": 1e-3, "schedule": "none", "clip": 1.0, "batch": 1, "seed": 0,
                       "save_every": 500, "select": "last", "stages": [{"horizon": 1, "iters": 0}]}}
    tp = tdir / "g_run.yaml"
    tp.write_text(yaml.safe_dump(tr))
    spec = T.load(str(tp))
    root = str(tmp_path / "root")
    out = T.out_dir(spec, root)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    op = CLS({"_at": "neuron", **{k: v for k, v in opp.items() if k not in ("op", "model", "at")}})
    B = torch.zeros(N, F)
    gB = torch.Generator().manual_seed(3)
    B[:120, 0], B[:120, 1] = 0.5, 0.3
    B[:120] *= 0.2 + torch.rand(120, 1, generator=gB)                           # input weights of different sizes

    def write(W_mid=None):
        g = torch.Generator().manual_seed(7)
        fit = {"neuron.tau": -2.97 + 0.4 * torch.randn(N, 1, generator=g), "neuron.rest": torch.zeros(N, 1), "neuron.input": B.clone(),
               "state_diffuse.W_short": torch.zeros(op._E["short"][0].numel()),
               "state_diffuse.W_mid": torch.zeros(op._E["mid"][0].numel()) if W_mid is None else W_mid.clone()}
        torch.save({"fitted": fit, "it": 0, "select": "last", "model": str(mp), "task": {}, "learnable": []},
                   os.path.join(out, "models", "best.pt"))
    return {"spec": spec, "root": root, "out": out, "rec": rec, "op": op, "pos": pos, "write": write}


def test_identity_with_every_weight_zero(planted):
    P = planted
    P["write"]()
    doc = G.run(P["spec"], device="cpu", root=P["root"], tests=["constants", "silence", "spectrum", "memorise", "controls"])
    assert doc["recording"]["n_input"] == 120                                   # the varying columns' elements
    s = doc["silence"]["rollouts"]
    for k in ("W0_all", "no_stimulus"):
        assert k in s
    for m in ("brain_mean_r", "per_neuron_r"):                                  # W = 0: silencing changes nothing
        assert abs(s["nominal"]["steady"][m] - s["W0_all"]["steady"][m]) < 1e-6
    t = next(t_ for t_ in doc["silence"]["tests"]["tests"] if t_["b"] == "W0_all")
    assert t["local_r"]["difference"] == 0 and t["local_r"]["p"] == 1.0
    assert s["no_stimulus"]["steady"]["input"]["per_neuron_r"] != s["nominal"]["steady"]["input"]["per_neuron_r"]   # the drive acts
    sp = doc["spectrum"]
    assert sp["M_spectral_radius"] < 1e-9 and abs(sp["abscissa_per_s"] - sp["leak_abscissa_per_s"]) < 1e-6
    assert sp["n_slower_than_leak"] == 0
    assert "memorise" in doc["refused"] and "controls" in doc["refused"] and not doc["errors"]
    assert os.path.exists(os.path.join(P["out"], "results", "g_run_graph.json")) and os.path.exists(doc["card"])


def test_a_planted_chain_carries_a_pulse_and_prunes_away(planted):
    P, op, pos = planted, planted["op"], planted["pos"]
    snd, rcv = (t.numpy() for t in op._E["mid"])
    e1 = next(e for e in range(len(snd)) if pos[snd[e], 0] < 100 and pos[rcv[e], 0] >= 200)   # front -> back
    e2 = next((e for e in range(len(snd)) if snd[e] == rcv[e1] and pos[rcv[e], 0] >= 200), None)
    g = torch.Generator().manual_seed(11)
    W = torch.zeros(len(snd))
    bg = torch.randperm(len(snd), generator=g)[:60]
    W[bg] = 0.01 * torch.rand(60, generator=g)                                  # a weak background the ladder can cut
    W[e1] = 0.8
    if e2 is not None:
        W[e2] = 0.8
    P["write"](W)
    doc = G.run(P["spec"], device="cpu", root=P["root"],
                tests=["silence", "blocks", "clamp", "prune_w", "prune_b", "spectrum", "impulse", "leadlag"])
    assert not doc["errors"], doc["errors"]
    s = doc["silence"]["rollouts"]
    assert "W0_W_mid" in s and "W0_W_short" in s                              # one set at a time, and all
    assert abs(s["W0_W_short"]["steady"]["per_neuron_r"] - s["nominal"]["steady"]["per_neuron_r"]) < 1e-6   # nothing there
    assert any("gap" in r for r in doc["blocks"]["per_block"])
    cl = doc["clamp"]
    assert cl["n_clamped"] == 120 and cl["identity_max_abs"] < 1e-5           # with W = 0 the free neurons cannot see the inputs
    pw = doc["prune_w"]["per_set"]
    assert pw["short"].get("skipped") and "ladder" in pw["mid"]              # no nonzero short weight; the mid ladder ran
    top = max(pw["mid"]["ladder"], key=lambda r: r["threshold"])
    assert top["threshold"] > 0.8 or top["kept_share"] < 1                     # the ladder reaches the planted weight
    pb = doc["prune_b"]
    assert pb["n_input"] == 120 and pb["ladder"][0]["n_cut"] > 0
    sp = doc["spectrum"]
    assert sp["nnz"] >= 1 and sp["M_spectral_radius"] < 0.05                 # a near feed-forward chain: a small radius
    imp = doc["impulse"]
    assert "front" in imp["pulse"] and imp["pulse"]["front"]["outside_share"] > 0           # the pulse crosses the edge
    assert imp["pulse"]["front"]["W0"]["outside_share"] == 0                                # not without the graph
    assert imp["linear"]["front"]["outside_share"] > 0
    assert doc["leadlag"]["n_both"] > 0 and "spearman_model_vs_recording" in doc["leadlag"]
    j = json.load(open(os.path.join(P["out"], "results", "g_run_graph.json")))
    assert set(j["tests_run"]) == {"silence", "blocks", "clamp", "prune_w", "prune_b", "spectrum", "impulse", "leadlag"}
