"""exp17: the DESTRIPED ZAPBench traces (Vijay Kumar's zap-inr run, 2026-09-30) as a recording in exp17's format.

Source: <run>/activity_amplitudes.zarr (zarr v3)
    log_a           [N, T]  log amplitude per neuron and frame (amplitude = exp(log_a), photo-electrons)
    mu_anatomy_t0   [N, 3]  positions, um, in the anatomy frame at t = 0
    bg              [T]     one background scalar per frame (not part of any neuron's amplitude; kept for the record)
    t               [T]     time

Output: graphs_data/zebrafish/<name>_recording.npz with exp17's keys -- `dff` [T, N], `pos_um` [N, 3], and the ZAPBench
release's own `stimulus`, `condition`, `offsets`, `names`, `t_s` (the same session: T must be 7,879 frames).

THE SCALE (main's docstring): not ZAPBench's dF/F -- the amplitudes sit on a floor -- but one global linear map onto
ZAPBench's dF/F distribution. `running_baseline` (ZAPBench's percentile baseline) is kept for reference.

Local only (the cluster rule): run in the devcontainer once the zarr is readable from it.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
RADIUS, PCT, CLIP = 400, 8.0, (-0.25, 1.5)


def read_zarr(path):
    """The four arrays of the zarr v3 store (tensorstore: the installed zarr-python 2 reads v2 only)."""
    import tensorstore as ts
    return {k: ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": os.path.join(path, k)}}).result().read().result()
            for k in ("log_a", "mu_anatomy_t0", "bg", "t")}


def running_baseline(A, stride, device, chunk=8192):
    """[T, N] the 8th percentile over +-RADIUS frames, at anchors every `stride` frames, interpolated in between."""
    T, N = A.shape
    anchors = np.unique(np.r_[np.arange(0, T, stride), T - 1])
    F0 = np.empty((len(anchors), N), np.float32)
    for c0 in range(0, N, chunk):
        a = torch.as_tensor(A[:, c0:c0 + chunk], device=device)
        for k, t in enumerate(anchors):
            w = a[max(0, t - RADIUS):min(T, t + RADIUS + 1)]
            F0[k, c0:c0 + a.shape[1]] = torch.quantile(w, PCT / 100, dim=0).cpu().numpy()
    # linear interpolation between anchors, all neurons at once: frame t sits between anchors k and k+1
    k = np.clip(np.searchsorted(anchors, np.arange(T), side="right") - 1, 0, len(anchors) - 2)
    w = ((np.arange(T) - anchors[k]) / (anchors[k + 1] - anchors[k])).astype(np.float32)[:, None]
    return (1 - w) * F0[k] + w * F0[k + 1]


FAILED_LOG_A = 8.0       # a neuron that ever reaches log_a > 8 (amplitude > ~3,000 against a median ~8) is a failed fit


def main():
    """THE MAP (Cedric, 2026-10-01): the amplitudes sit on a 0.1 floor for a fifth of the entries (activity above the
    background, not fluorescence with a resting level), so ZAPBench's dF/F (a percentile baseline) would divide by the
    floor. Instead: a = exp(log_a); the failed fits clamped at the 99.9th percentile of the other neurons' amplitudes;
    then ONE GLOBAL LINEAR MAP x = alpha a + beta giving the overall mean and standard deviation of ZAPBench's dF/F,
    clipped to ZAPBench's range -- every neuron's dynamics kept exactly, only the units changed."""
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr")
    ap.add_argument("--name", default="zapbench_destripe")
    a = ap.parse_args()
    from plexus.paths import graphs_data_path
    src = read_zarr(a.zarr)
    la = src["log_a"]
    N, T = la.shape
    ref = np.load(graphs_data_path("zebrafish", "zapbench_recording.npz"))
    if T != ref["dff"].shape[0]:
        raise ValueError(f"{T} frames, the ZAPBench session has {ref['dff'].shape[0]}: not the same recording")
    failed = la.max(1) > FAILED_LOG_A
    A = np.exp(la.astype(np.float32)).T.copy()                 # [T, N] amplitudes
    rng = np.random.default_rng(0)
    ok_cols = np.where(~failed)[0]
    sample = A[:, rng.choice(ok_cols, min(5000, len(ok_cols)), replace=False)]
    cap = float(np.percentile(sample, 99.9))
    np.minimum(A, cap, out=A)
    Z = ref["dff"]
    zs = Z[:, rng.choice(Z.shape[1], 5000, replace=False)].astype(np.float64)
    As = A[:, rng.choice(N, 5000, replace=False)].astype(np.float64)
    alpha = float(zs.std() / As.std())
    beta = float(zs.mean() - alpha * As.mean())
    x = np.clip(alpha * A + beta, *CLIP).astype(np.float32)
    del A
    out = graphs_data_path("zebrafish", f"{a.name}_recording.npz")
    np.savez(out, dff=x, pos_um=src["mu_anatomy_t0"].astype(np.float32), stimulus=ref["stimulus"],
             condition=ref["condition"], offsets=ref["offsets"], names=ref["names"], t_s=ref["t_s"],
             bg=src["bg"].astype(np.float32), t_src=src["t"], failed_fit=failed)
    qs = [1, 8, 25, 50, 75, 92, 99, 99.9]
    xs = x[:, rng.choice(N, 5000, replace=False)]
    prov = {"source": os.path.abspath(a.zarr), "neurons": int(N), "frames": int(T),
            "map": "x = alpha * min(exp(log_a), cap) + beta, clipped to ZAPBench's [-0.25, 1.5]",
            "failed_fits": {"rule": f"max log_a > {FAILED_LOG_A}", "neurons": int(failed.sum()),
                            "clamped_at_amplitude": cap, "cap_rule": "99.9th percentile of the other neurons' amplitudes"},
            "alpha": alpha, "beta": beta,
            "matched": "the overall mean and standard deviation of ZAPBench's dF/F (zapbench_recording.npz, all entries)",
            "quantiles": {"pct": qs, "zapbench": np.percentile(zs, qs).round(4).tolist(),
                          "destripe": np.percentile(xs, qs).round(4).tolist()},
            "mean_sd": {"zapbench": [float(zs.mean()), float(zs.std())], "destripe": [float(xs.mean()), float(xs.std())]},
            "clipped_fraction": float(((xs <= CLIP[0]) | (xs >= CLIP[1])).mean()),
            "stimulus_condition_offsets": "from zapbench_recording.npz (the same session, 7,879 frames)",
            "log_a_sha256": hashlib.sha256(np.ascontiguousarray(la).tobytes()).hexdigest()}
    json.dump(prov, open(out.replace(".npz", ".json"), "w"), indent=1)
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
