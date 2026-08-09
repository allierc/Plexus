"""refute5_fit.py -- ROUND 5 REFUTATION, part 2: the same fit with a SPATIALLY REALIZABLE F noise.

round5_fit.py:110 draws the measured-F error as `randn(F0.shape)` on a [Np,2,2] tensor -- an
INDEPENDENT error at every one of the 10 000 particles.  With 100 particles per cell that averages
down by sqrt(100) = 10 inside every cell, and the per-cell moduli are exactly what is being
estimated.  refute5_spatial.py measured the recording's own F noise field (second time difference
over the quiet stretch, healthy specimen):

    pooled lag-1 spatial autocorrelation   0.255      (white control: 3e-6)
    kxk block-variance ratio               1.45-1.58  (white control: 1.00)
    masked nodes / cell                    17499/472 = 37.1
    effective independent samples per cell 36/1.58    = 22.8

So a cell of the recording carries ~23 independent F measurements, not 100.  This script keeps
EVERYTHING else in round 5 identical and only replaces the noise draw:

  --noise indep   round 5's model (control; must reproduce round5_norm_s<seed> exactly)
  --noise grid    white on a NODE GRID of `--nodes` per unit side, nearest-neighbour to particles
  --noise gridsm  the same, smoothed to the measured lag-1 autocorrelation (the realizable model)

With --nodes 61 a sim cell (side 0.1) covers 37 nodes, matching the recording node-for-node.
The Monte-Carlo re-noising of the EIV correction uses the SAME model (the analyst is assumed to
know the noise model -- the generous assumption).

usage:
  PYTHONPATH=/workspace/Plexus/src python refute5_fit.py --device cuda:1 --noise gridsm \
      --tag round5_norm_gridsm_s90210_sF0.0039
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
DISC = "/workspace/Plexus/discovery_cardio_mpm"
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("/workspace/Plexus/src", ALG, DISC, HERE):
    sys.path.insert(0, _p)

from recover import theta_scale                                  # noqa: E402
import crash_test as CT                                          # noqa: E402
from finject import record_substeps, lerp, assemble_inj          # noqa: E402
from refute_round3 import advance                                # noqa: E402
from round5_fit import SIGMA_F, SIGMA_X, SNAP                    # noqa: E402

LAG1 = 0.2546          # refute5_spatial.json: pooled lag-1 spatial autocorrelation of the noise


def _kernel_a(target=LAG1):
    """(a,1,a) separable kernel whose lag-1 autocorrelation is `target`: 2a/(1+2a^2) = r."""
    # solve 2 target a^2 - 2 a + target = 0, take the small root
    t = target
    return (2 - np.sqrt(4 - 8 * t * t)) / (4 * t)


class NoiseF:
    """Draws a [Np,2,2] error field with the recording's spatial structure, unit per-component std."""

    def __init__(self, mode, x0, nodes, device, dtype):
        self.mode, self.shape = mode, (x0.shape[0], 2, 2)
        self.device, self.dtype = device, dtype
        self.nodes = nodes
        if mode != "indep":
            ij = (x0.clamp(0, 1 - 1e-9) * nodes).long()
            self.idx = (ij[:, 1] * nodes + ij[:, 0]).clamp(0, nodes * nodes - 1)
            a = float(_kernel_a())
            k = torch.tensor([a, 1.0, a], device=device, dtype=dtype)
            k = k / k.norm()                                  # unit-energy -> preserves variance
            self.k = k
        self.diag = {}

    def _smooth(self, g):
        # separable (a,1,a) with unit energy, applied along both node axes, periodic edges
        k = self.k
        for dim in (0, 1):
            s = torch.zeros_like(g)
            for j, w in enumerate((-1, 0, 1)):
                s = s + k[j] * torch.roll(g, shifts=w, dims=dim)
            g = s
        return g

    def __call__(self, gen):
        if self.mode == "indep":
            return torch.randn(self.shape, generator=gen, device=self.device, dtype=self.dtype)
        n = self.nodes
        g = torch.randn((n, n, 4), generator=gen, device=self.device, dtype=self.dtype)
        if self.mode == "gridsm":
            g = self._smooth(g)
        flat = g.reshape(n * n, 4)[self.idx]
        return flat.reshape(-1, 2, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--noise", default="gridsm", choices=("indep", "grid", "gridsm"))
    ap.add_argument("--nodes", type=int, default=61)
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--sigma-F", type=float, default=SIGMA_F)
    ap.add_argument("--sigma-x", type=float, default=SIGMA_X)
    ap.add_argument("--seed", type=int, default=90210)
    a = ap.parse_args()
    # NOTE: the prefix is refute5_, never round5_norm_, because round5_solve.py globs
    # round5_norm_*.npz and must not pick these up (no round-5 artefact is overwritten).
    tag = a.tag or f"refute5_norm_{a.noise}{a.nodes}_s{a.seed}_sF{a.sigma_F:g}"

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.t0, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "sigma_F": a.sigma_F, "sigma_x": a.sigma_x, "K": a.K, "T": a.T,
         "seed": a.seed, "tag": tag, "noise_mode": a.noise, "noise_nodes": a.nodes}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        sy, _ = CT.plant_and_warm(args, log)
        C, n = sy.C, sy.n_sub_per_frame
        s = theta_scale(C, sy.device)
        NF = NoiseF(a.noise, sy.x0, a.nodes, sy.device, sy.dtype)

        # ---- CONTROL: the per-cell averaging the noise model actually delivers ----------------- #
        gchk = torch.Generator(device=sy.device).manual_seed(7)
        cell_std, node_lag1 = [], []
        for _ in range(24):
            e = NF(gchk).reshape(-1, 4)
            m = torch.zeros(C + 1, 4, device=sy.device, dtype=sy.dtype)
            m.index_add_(0, sy.cid, e)
            cnt = torch.zeros(C + 1, device=sy.device, dtype=sy.dtype)
            cnt.index_add_(0, sy.cid, torch.ones_like(e[:, 0]))
            cm = m[1:] / cnt[1:, None]
            cell_std.append(float(cm.std()))
            node_lag1.append(float(e.std()))
        R["noise_control"] = {
            "per_component_unit_std": float(np.mean(node_lag1)),
            "per_cell_mean_std": float(np.mean(cell_std)),
            "effective_independent_per_cell": float(1.0 / np.mean(cell_std) ** 2),
            "note": "unit per-component input; a cell mean of k independent samples has std "
                    "1/sqrt(k), so 1/std^2 IS the effective sample count (round 5's indep model "
                    "gives 100; the recording gives ~23)"}
        log(f"[noise {a.noise}{a.nodes}] per-cell effective independent samples "
            f"{R['noise_control']['effective_independent_per_cell']:.1f} "
            f"(round-5 indep = 100, recording = 22.8)")

        frames = []
        cur = a.t0
        for k in range(a.T):
            if k > 0:
                sy.restore()
                advance(sy, cur, cur + 1)
                sy._snapshot(cur + 1)
                cur += 1
            Fs, Cs, Xs = record_substeps(sy, n)
            frames.append({"tick": cur, "x0": sy.x0.clone(), "F0": sy.F0.clone(),
                           "F1": Fs[-1].clone(), "x_next": Xs[-1].clone(),
                           "snap": {kk: getattr(sy, kk).clone() for kk in SNAP}})
        log(f"[frames] {a.T} from tick {a.t0} (ticks {frames[0]['tick']}..{frames[-1]['tick']})")

        gm = torch.Generator(device=sy.device).manual_seed(a.seed)
        gk = torch.Generator(device=sy.device).manual_seed(31337 + a.seed)

        eb = [(a.sigma_F / 2.0) * NF(gm) for _ in range(a.T + 1)]
        xs = [f["x_next"] + a.sigma_x * torch.randn(f["x_next"].shape, generator=gm,
                                                    device=sy.device, dtype=sy.dtype)
              for f in frames]

        out = {}
        R["frames"] = []
        for k, f in enumerate(frames):
            for kk in SNAP:
                setattr(sy, kk, f["snap"][kk].clone())
            F0h, F1h = f["F0"] + eb[k], f["F1"] + eb[k + 1]
            A, y0, _ = assemble_inj(sy, n, lerp(F0h, F1h, n), None)
            Az = A * s[None, :]
            b = (xs[k] - f["x0"]).reshape(-1) - y0
            Gk, rk = Az.T @ Az, Az.T @ b
            del A, Az
            torch.cuda.empty_cache()
            Gs, rs = torch.zeros_like(Gk), torch.zeros_like(rk)
            for _ in range(a.K):
                e0 = (a.sigma_F / 2.0) * NF(gk)
                e1 = (a.sigma_F / 2.0) * NF(gk)
                Aj, y0j, _ = assemble_inj(sy, n, lerp(F0h + e0, F1h + e1, n), None)
                Azj = Aj * s[None, :]
                Gs += Azj.T @ Azj
                rs += Azj.T @ ((xs[k] - f["x0"]).reshape(-1) - y0j)
                del Aj, Azj
                torch.cuda.empty_cache()
            if a.K > 0:
                Gs, rs = Gs / a.K, rs / a.K
            out[f"G{k}"] = Gk.cpu().numpy()
            out[f"r{k}"] = rk.cpu().numpy()
            out[f"Gm{k}"] = Gs.cpu().numpy()
            out[f"rm{k}"] = rs.cpu().numpy()
            R["frames"].append({"tick": f["tick"], "y_obs_norm": float((xs[k] - f["x0"]).norm())})
            log(f"    frame {k} (tick {f['tick']}) + {a.K} re-noisings [{time.time()-t_start:.0f}s]")

        out["theta_true"] = sy.theta_true.double().cpu().numpy()
        out["s"] = s.cpu().numpy()
        np.savez(os.path.join(HERE, f"{tag}.npz"), **out)

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {tag}.npz [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()
