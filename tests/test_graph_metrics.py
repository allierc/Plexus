"""The free rollout's network metrics in plexus.tasks.trace_recording (the graph phase, 2026-10-10) against the deck's
originals, number for number: brain_mean_metrics = exp17_slides.bm_metrics, per_neuron_r = exp17_slides.local_r
(flat0 True and False), block_scores = exp17_w0_blocks.scores, the paired bootstrap's estimates = the whole-sample
values and a law against itself gives difference 0 and p = 1."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from plexus.tasks import trace_recording as TR  # noqa: E402


def planted(n_frames=240, n_in=6, n_out=10, seed=0):
    """A recording and a learned rollout: perfect on the input neurons, flat on half the others, NaN (silenced) on one,
    and noisy elsewhere."""
    rng = np.random.default_rng(seed)
    n = n_in + n_out
    X = rng.normal(size=(n_frames, n)) + 0.5 * rng.normal(size=(n_frames, 1))
    P = X + 0.3 * rng.normal(size=(n_frames, n))
    P[:, :n_in] = X[:, :n_in]
    P[:, n_in:n_in + 4] = 0.3                                                  # flat learned traces
    P[:, n_in + 4] = np.nan                                                    # a silenced neuron
    return X, P


def test_brain_mean_and_per_neuron_r_match_the_deck(tmp_path):
    import exp17_slides as S
    X, P = planted()
    npz = str(tmp_path / "planted_movie.npz")
    np.savez(npz, frames=np.arange(len(X)), pred=P, mean_obs_all=X.mean(1), mean_pred_all=np.nanmean(P, 1))
    S._REC["planted_rec"] = {"dff": X}
    bm = TR.brain_mean_metrics(X.mean(1), np.nanmean(P, 1))
    bm_deck = S.bm_metrics(npz)
    for k in ("r", "r2", "rmse"):
        assert abs(bm[k] - bm_deck[k]) < 1e-12, k
    for flat0 in (True, False):
        mine = TR.per_neuron_r(P, X, neuron_set="recording" if flat0 else "finite", flat="zero" if flat0 else "drop")
        deck = S.local_r(npz, "planted_rec", flat0=flat0)
        assert abs(mine["mean"] - deck["mean"]) < 1e-12 and abs(mine["sd"] - deck["sd"]) < 1e-12 and mine["n"] == deck["n"]
    r = TR.per_neuron_r(P, X)["r"]
    assert (r[:6] > 0.9).all() and np.all(r[6:10] == 0) and r[10] == 0 and np.isfinite(r[11:]).all()   # flat and NaN scored 0


def test_block_scores_match_w0_blocks():
    import exp17_w0_blocks as W
    X, P = planted()
    P[:, 10] = X[:, 10].mean()                                   # the every-frame traces freeze a silenced neuron, no NaN
    mine = TR.block_scores(X, P)
    bm, lr, sd, n = W.scores(X, P, "cpu")
    assert abs(mine["brain_mean_r"] - bm) < 1e-6 and abs(mine["per_neuron_r"] - lr) < 1e-6 and mine["neurons"] == n


def test_steady_blocks_and_sums_reproduce_the_whole_sample_values():
    X, P = planted(n_frames=480)
    blk, keep_f = TR.steady_blocks(len(X), n_blocks=24, skip=1)
    assert blk.max() == 23 and keep_f.sum() == 460 and not keep_f[:20].any()
    S = TR.brain_mean_block_sums(X.mean(1), np.nanmean(P, 1), blk)
    assert abs(TR.r_from_sums(S.sum(0)) - TR.brain_mean_metrics(X.mean(1), np.nanmean(P, 1))["r"]) < 1e-12
    keep = (TR.remove_brain_mean(X).std(0) > 1e-9) & np.isfinite(P).all(0)
    per, sc = TR.block_sums(P, X, blk, keep)
    r_sum = TR.local_r_from_sums(per.sum(0), sc.sum(0), per_neuron=True).numpy()
    r_dir = TR.per_neuron_r(P, X, neuron_set="finite", flat="zero")["r"][keep]
    assert np.allclose(r_sum, r_dir, atol=1e-9)


def test_paired_bootstrap_of_a_law_against_itself_and_against_a_worse_one():
    X, P = planted(n_frames=480)
    Q = P + 0.8 * np.random.default_rng(1).normal(size=P.shape)             # a worse law
    blk, _ = TR.steady_blocks(len(X), 24, 0)
    keep = (TR.remove_brain_mean(X).std(0) > 1e-9) & np.isfinite(P).all(0) & np.isfinite(Q).all(0)
    bm = {"a": TR.brain_mean_block_sums(X.mean(1), np.nanmean(P, 1), blk),
          "a2": TR.brain_mean_block_sums(X.mean(1), np.nanmean(P, 1), blk),
          "b": TR.brain_mean_block_sums(X.mean(1), np.nanmean(Q, 1), blk)}
    loc = {}
    for k, Y in (("a", P), ("a2", P), ("b", Q)):
        per, sc = TR.block_sums(Y, X, blk, keep)
        loc[k] = (per, sc, TR.local_r_from_sums(per.sum(0), sc.sum(0))[1])
    doc = TR.paired_block_bootstrap(bm, loc, [("a", "a2"), ("a", "b")], resamples=500, seed=0)
    same, worse = doc["tests"]
    assert same["local_r"]["difference"] == 0 and same["local_r"]["p"] == 1.0 and same["brain_mean_r"]["p"] == 1.0
    assert worse["local_r"]["difference"] > 0 and worse["local_r"]["p"] < 0.05 and worse["local_r"]["stars"] != "n.s."
    assert abs(doc["laws"]["a"]["local_r"]["estimate"] - TR.per_neuron_r(P, X, "finite", "zero")["r"][keep].mean()) < 1e-9
    assert doc["blocks"] == 24 and doc["resamples"] == 500
