"""exp17: ZAPBench's own split in the trace trainer (`task.reference.split: zapbench`) -- the frames a law trains on
never reach the frames it is scored on."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from plexus.tasks import trace_recording as TR  # noqa: E402

OFF = np.array([0, 300, 700, 1000, 1500, 1800])       # 5 conditions; condition 3 is the held-out one
T = int(OFF[-1])
COND = np.repeat(np.arange(len(OFF) - 1), np.diff(OFF))


def test_labels_follow_zapbench_fractions():
    lab = TR.zapbench_split(OFF, T)
    for c in range(len(OFF) - 1):
        seg = lab[OFF[c]:OFF[c + 1]]
        assert seg[0] == -1 and seg[-1] == -1                       # one pad frame at each end
        n = len(seg) - 2
        if c == TR.ZB_HOLDOUT:
            assert (seg[1:-1] == 3).all()
        else:
            assert (seg == 2).sum() == int(n * 0.2) and (seg == 1).sum() == int(n * 0.1)
            assert (seg == 0).sum() == n - int(n * 0.2) - int(n * 0.1)
            first = {k: int(np.argmax(seg == k)) for k in (0, 1, 2)}
            assert first[0] < first[1] < first[2]                   # train, then val, then test, in time


def test_no_training_window_touches_a_scored_frame():
    lab = TR.zapbench_split(OFF, T)
    for n_in in (1, 6):
        o = TR.split_origins(lab, COND, "train", n_in, 30)
        assert len(o) > 0
        for x in o:
            assert (lab[x - n_in + 1:x + 31] == 0).all()


def test_test_windows_forecast_test_frames_only_and_holdout_starts_late():
    lab = TR.zapbench_split(OFF, T)
    o = TR.split_origins(lab, COND, "test", 6, 32)
    assert len(o) > 0 and all((lab[x + 1:x + 33] == 2).all() for x in o)
    assert all(len(set(COND[x - 5:x + 33])) == 1 for x in o)       # one condition per window
    h = TR.split_origins(lab, COND, "holdout", 4, 32)
    assert len(h) and h.min() == OFF[TR.ZB_HOLDOUT] + TR.ZB_PAD + TR.ZB_MAXCTX - 1


def test_lookup_is_learned_on_the_fit_frames_only():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(T, 7, generator=g)
    kid = np.arange(T) % 5
    fit = TR.zapbench_split(OFF, T) == 0
    a = TR.stimulus_lookup_fit(X, kid, fit)
    X2 = X.clone()
    X2[~torch.as_tensor(fit)] += 100.0                              # change every frame that is not a fit frame
    b = TR.stimulus_lookup_fit(X2, kid, fit)
    assert torch.allclose(a, b)
