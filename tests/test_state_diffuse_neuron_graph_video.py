"""state_diffuse[neuron_graph_video / neuron_graph_phase_video] (exp17 batch 26): the stimulus video as input,
e_{n,i} = [CNN_theta(V_n)]_i = m_i (b_i . h_theta(V_n)), added to the neuron graph's input term."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import plexus.operators  # noqa: E402,F401
from plexus.models.registry import get_contract  # noqa: E402

IMPL = get_contract("state_diffuse").implementations
BASE, VID = IMPL["neuron_graph"], IMPL["neuron_graph_video"]
PBASE, PVID = IMPL["neuron_graph_phase"], IMPL["neuron_graph_phase_video"]
N, SUB, PX = 1500, 8, 90


def files(tmp, T=4):
    g = np.random.default_rng(0)
    P = (g.uniform(0, 1, (N, 3)) * [300.0, 400.0, 120.0]).astype(np.float32)
    f = os.path.join(tmp, "pos.npz")
    np.savez(f, pos_um=P, offsets=np.array([0, 10, 20, 30]))
    ft = os.path.join(tmp, "types.npz")
    np.savez(ft, types=g.integers(0, T, N))
    fm = os.path.join(tmp, "vmask.npz")
    m = np.zeros(N, np.float32)
    m[:400] = 1.0
    np.savez(fm, mask=m)
    return f, ft, fm


def params(f, ft=None, fm=None, **kw):
    p = {"_at": "neuron", "block": "dff", "positions": "xyz", "positions_file": f, "inputs": 1, "substeps": 4,
         "short_k": 6, "mid_um": 32.0, "long_um": 128.0, "forcing": "stimulus.u", "forcing_dim": 22, "w_init_sd": 0.3}
    if ft:
        p["types_file"] = ft
    if fm:
        p.update(video_mask=fm, video_sub=SUB, video_px=PX, video_k=16)
    p.update(kw)
    return p


def nbs():
    g = torch.Generator().manual_seed(1)
    return {"tau": torch.full((N, 1), -1.0), "rest": 0.1 * torch.randn(N, 1, generator=g),
            "input": 0.1 * torch.randn(N, 22, generator=g)}


def inputs(clip):
    g = torch.Generator().manual_seed(3)
    x = torch.randn(N, 1, generator=torch.Generator().manual_seed(2))
    feats = torch.randn(22, 1, generator=g)
    return x, feats, torch.cat([feats, clip.reshape(-1, 1)], 0)


def clip(seed=4):
    return torch.rand(SUB * PX * PX, generator=torch.Generator().manual_seed(seed))


def run(o, u, x, frame=0):
    o.frame, o.n_frames_ref = frame, 30
    return o.step(x, None, None, u, nb=nbs())


def test_untrained_video_law_is_the_neuron_graph(tmp_path):
    """video_out starts at 0: the video law equals the law without the video, bit for bit."""
    f, _, fm = files(str(tmp_path))
    x, feats, u = inputs(clip())
    assert torch.equal(run(BASE(params(f)), feats, x), run(VID(params(f, fm=fm)), u, x))


def test_untrained_phase_video_law_is_the_phase_law(tmp_path):
    f, ft, fm = files(str(tmp_path))
    x, feats, u = inputs(clip())
    assert torch.equal(run(PBASE(params(f, ft)), feats, x), run(PVID(params(f, ft, fm=fm)), u, x))


def test_a_blank_clip_gives_no_input_and_the_mask_holds(tmp_path):
    """No bias anywhere: a blank clip (the dark block, `drive: off`) gives h = 0, e = 0; outside the mask e = 0."""
    f, _, fm = files(str(tmp_path))
    o = VID(params(f, fm=fm))
    o.video_out = torch.randn(N, 16, generator=torch.Generator().manual_seed(5))
    assert torch.equal(o.video_features(torch.zeros(SUB * PX * PX)), torch.zeros(16))
    nb = nbs()
    _, feats, u = inputs(clip())
    _, _, u0 = inputs(torch.zeros(SUB * PX * PX))
    e = o._input_drive(nb, u) - o._input_drive(nb, u0)
    assert e[:400].abs().max() > 0 and torch.equal(e[400:], torch.zeros(N - 400, 1))
    assert torch.allclose(o._input_drive(nb, u0), BASE._input_drive(o, nb, feats))


def test_gradients_reach_the_encoder_and_the_output_weights(tmp_path):
    f, _, fm = files(str(tmp_path))
    o = VID(params(f, fm=fm))
    o.video_theta = o.video_theta.clone().requires_grad_(True)
    o.video_out = (0.01 * torch.randn(N, 16, generator=torch.Generator().manual_seed(6))).requires_grad_(True)
    x, _, u = inputs(clip())
    run(o, u, x).pow(2).sum().backward()
    assert o.video_theta.grad.abs().sum() > 0
    assert o.video_out.grad[:400].abs().sum() > 0 and torch.equal(o.video_out.grad[400:], torch.zeros(N - 400, 16))


def test_a_wrong_forcing_width_is_refused(tmp_path):
    f, _, fm = files(str(tmp_path))
    o = VID(params(f, fm=fm))
    x, feats, _ = inputs(clip())
    try:
        run(o, feats, x)
    except ValueError as e:
        assert "video values" in str(e)
    else:
        raise AssertionError("a forcing without the clip must be refused")
