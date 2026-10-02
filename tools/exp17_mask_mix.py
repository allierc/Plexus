"""exp17: the batch-15 input mask, HALF VISUAL, HALF EPHYS (Cedric, 2026-10-02: "the mask for batch 15 should be built
on the 10 % neurons that best match the stimulus and the ephys, not ephys only").

Local analysis. Two rankings of every destriped neuron, each its best band coherence (magnitude-squared coherence,
Welch 256-frame Hann segments, half overlap, averaged over the VISUAL stimulus's 6 strongest frequencies):
  visual  over the 13 visual features that change within their condition (stim_coherence_zapbench_destripe_varying.npz)
  ephys   over the 5 ephys features (swim power L / R, turn power L / R, bout fraction) of
          zapbench_destripe_ephys_recording.npz, computed here on the same band
The mask takes neurons alternately from the top of each ranking (a neuron already taken is skipped) until it holds
`--frac` of the neurons (10 %; 20 % for batch 15, Cedric): half the slots go to the stimulus-locked, half to the swim-locked, whatever their coherence scale.

Writes graphs_data/zebrafish/input_mask_destripe_mix_coh10.npz (mask, coh_visual, coh_ephys, best_visual, best_ephys,
picked_by: 0 visual / 1 ephys / -1 not selected).
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
N_VISUAL, FRAC = 22, 0.10


def main(device="cuda:0", chunk=4000, frac=FRAC):
    from plexus.paths import graphs_data_path
    from exp17_stim_coherence import segments, NSEG, FS
    vis = np.load(os.path.join(EXP, "data", "stim_coherence_zapbench_destripe_varying.npz"))
    cv, bv, band_hz = vis["coherence"], vis["best_feature"], vis["band_hz"]
    z = np.load(graphs_data_path("zebrafish", "zapbench_destripe_ephys_recording.npz"))
    X, U = z["dff"], z["stimulus"][:, N_VISUAL:].astype(np.float32)               # the 5 ephys columns
    f = np.fft.rfftfreq(NSEG, d=1 / FS)
    band = np.array([int(np.argmin(np.abs(f - b))) for b in band_hz])          # the visual stimulus's band
    Uf = torch.fft.rfft(segments(torch.as_tensor(U, device=device)), dim=1)
    Suu = (Uf.abs() ** 2).mean(0)
    T, N = X.shape
    ce, be = np.zeros(N, np.float32), np.zeros(N, np.int16)
    for c0 in range(0, N, chunk):
        c1 = min(N, c0 + chunk)
        Xf = torch.fft.rfft(segments(torch.as_tensor(np.asarray(X[:, c0:c1], np.float32), device=device)), dim=1)
        Sxx = (Xf.abs() ** 2).mean(0)
        Sxu = torch.einsum("sfn,sfk->fnk", Xf, Uf.conj()) / Xf.shape[0]
        C = (Sxu.abs() ** 2) / (Sxx[:, :, None] * Suu[:, None, :]).clamp(min=1e-20)
        v, k = C[band].mean(0).max(1)
        ce[c0:c1], be[c0:c1] = v.cpu().numpy(), k.cpu().numpy()
    n_sel = int(round(frac * N))
    rv, re_ = np.argsort(-cv), np.argsort(-ce)
    picked = -np.ones(N, np.int8)
    iv = ie = 0
    while (picked >= 0).sum() < n_sel:                                      # alternate, skipping neurons taken
        for src, r, which in ((0, rv, "v"), (1, re_, "e")):
            if (picked >= 0).sum() >= n_sel:
                break
            i = iv if which == "v" else ie
            while picked[r[i]] >= 0:
                i += 1
            picked[r[i]] = src
            if which == "v":
                iv = i + 1
            else:
                ie = i + 1
    mask = (picked >= 0).astype(np.float32)
    out = graphs_data_path("zebrafish", f"input_mask_destripe_mix_coh{round(100 * frac)}.npz")
    np.savez(out, mask=mask, coh_visual=cv, coh_ephys=ce, best_visual=bv, best_ephys=(N_VISUAL + be).astype(np.int16),
             picked_by=picked, band_hz=band_hz)
    old_v = np.load(graphs_data_path("zebrafish", "input_mask_destripe_coh10v.npz"))["mask"] > 0
    old_e = np.load(graphs_data_path("zebrafish", "input_mask_destripe_ephys_coh10.npz"))["mask"] > 0
    s = mask > 0
    print(f"[mask mix] {int(s.sum()):,} of {N:,} neurons: {int((picked == 0).sum()):,} by visual coherence (>= "
          f"{cv[picked == 0].min():.3f}), {int((picked == 1).sum()):,} by ephys coherence (>= {ce[picked == 1].min():.3f}); "
          f"neurons in both top lists {int(((cv >= cv[picked == 0].min()) & (ce >= ce[picked == 1].min())).sum()):,}; "
          f"overlap with the visual mask {int((s & old_v).sum()):,}, with the ephys-best mask {int((s & old_e).sum()):,}")
    print(f"[mask mix] wrote {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=FRAC, help="the fraction of neurons the mask keeps (half each)")
    main(frac=ap.parse_args().frac)
