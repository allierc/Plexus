"""exp 11, the R series (2026-09-30: iterate on run 460 for its two defects): per run, per sampled row, the membrane
near each cell and the flat giants -- the two things the human named -- plus the budding readout.

    python tools/exp11_rcard.py exp11_mc_W3_b1block_s1 exp11_mc_R1_heal_s1 ...

Columns: `dens` the median number of live membrane nodes within 1 unit (10 um) of a cell's basal centroid per unit
area (seeded ~15); `thin` cells under a quarter of that seeded median; `bare` cells with no node within 1 unit;
`giant` cells over 3x the median basal area; `h_gi` their median thickness (volume / mean cap area; others ~0.8);
`excess` the basal area over a sphere's of the same volume, minus 1; `hollow` the share of cells over 25 um from
any interior cell; `inv` cells with volume <= 0; `sentenced` cells whose target volume is under 0.1 (a dive under
way), and `stuck` the giants among them -- dives the side rule cannot complete (Finding 190)."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_measures.common import open_run          # noqa: E402
from exp_measures import exp11 as M               # noqa: E402
from plexus.operators.vertex_ops import apicobasal_geometry_3d   # noqa: E402

ROWS = (60, 100, 140, 160, 200, 240)


def row_card(T, t, dens0=15.0):
    z = T.z
    P = np.asarray(T.pos(t), float); nF = T.nF(t)
    es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
    S = np.asarray(T.vertex_block("sep", t), float)[:len(P)]
    g = apicobasal_geometry_3d(torch.tensor(P, dtype=torch.float32), torch.tensor(S, dtype=torch.float32),
                               torch.tensor(es), torch.tensor(et), torch.tensor(ef), nF)
    v, Aa, Ab = (x.numpy() for x in (g[0], g[2], g[3]))
    h = v / np.maximum(0.5 * (Aa + Ab), 1e-6)
    m = ef < nF
    cb = np.stack([np.bincount(ef[m], weights=(P - S)[es[m], k], minlength=nF) for k in range(3)], 1) \
        / np.maximum(np.bincount(ef[m], minlength=nF), 1)[:, None]
    Bm = np.asarray(z["bm_node__pos"][t], float)[np.asarray(z["bm_node__occ"][t]).reshape(-1) > 0.5]
    cnt = np.array([len(x) for x in cKDTree(Bm).query_ball_point(cb, 1.0)])
    dens = cnt / np.pi
    big = Ab > 3 * np.median(Ab)
    Q = np.asarray(z["ipt__pos"][t], float)[np.asarray(z["ipt__occ"][t]).reshape(-1) > 0.5]
    dint = cKDTree(Q).query(np.asarray(z["cell__centroid"][t], float)[:nF])[0]
    V0 = np.asarray(z["cell__V0f"][t], float).reshape(-1)[:nF]
    sent = V0 < 0.1
    return dict(cells=nF, sent=int(sent.sum()), sent_gi=int((sent & big).sum()), dens=float(np.median(dens)), thin=int((dens < 0.25 * dens0).sum()), bare=int((cnt == 0).sum()),
                giant=int(big.sum()), h_gi=float(np.median(h[big])) if big.any() else float("nan"),
                hollow=float((dint > 2.5).mean()), inv=int((v <= 0).sum()))


def main():
    for r in sys.argv[1:]:
        T = open_run("tissue/" + r)
        L = M.lobes(T, every=20)
        ex = dict(zip(L["row"], L["excess"]))
        print(r)
        for t in ROWS:
            if t >= T.n_rows():
                continue
            c = row_card(T, t)
            print(f"  row {t:3d}: cells {c['cells']:4d}  dens {c['dens']:5.1f}  thin {c['thin']:3d}  bare {c['bare']:3d}  "
                  f"giant {c['giant']:3d} (h {c['h_gi']:.2f}, stuck {c['sent_gi']:3d})  sentenced {c['sent']:3d}  excess {ex.get(t, float('nan')):.2f}  hollow {c['hollow']:.3f}  inv {c['inv']}")


if __name__ == "__main__":
    main()
