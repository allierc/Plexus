"""Smoke test for embryo_metrics.phase1_from_arrays — checks the `escape`/`r_cell_max` observables
added 2026-07-02. Run: python _tmp_metric_test.py  (no GPU / trajectory needed).

Two synthetic blastulas share a membrane ring at radius 0.3 about (0.5, 0.5):
  - CONFINED: cells tightly inside the core -> escape == 0.
  - LEAKY:    a few cells pushed past 0.9*Rd into the membrane -> escape > 0.
"""
import numpy as np
from embryo_metrics import phase1_from_arrays

T, N, M = 20, 44, 500
th = np.linspace(-np.pi, np.pi, M)
mp = np.zeros((T, M, 2)); r = 0.3
mp[:, :, 0] = 0.5 + r * np.cos(th); mp[:, :, 1] = 0.5 + r * np.sin(th)
occ = np.ones((T, N)); nt = np.arange(N) % 2
rng = np.random.default_rng(0)

confined = 0.5 + 0.08 * rng.standard_normal((T, N, 2))
m0 = phase1_from_arrays(confined, occ, nt, mp, r0=0.02)
assert m0["escape"] == 0.0, m0["escape"]

leaky = confined.copy()
leaky[-1, :6] = [0.5 + 0.29, 0.5]            # 6 cells parked in the membrane band (r=0.29 > 0.9*0.3)
m1 = phase1_from_arrays(leaky, occ, nt, mp, r0=0.02)
assert m1["escape"] > 0.0 and m1["r_cell_max"] > 0.9, (m1["escape"], m1["r_cell_max"])

print("OK  confined escape =", m0["escape"], " leaky escape =", m1["escape"],
      " r_cell_max =", m1["r_cell_max"])
