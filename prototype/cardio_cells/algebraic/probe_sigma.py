"""probe_sigma.py -- can the cached `store_stress: true` buffer supply the E-sensitivity
of the stress without recomputing it?

Claim under test (option (ii) of the column-cost question). With no active stress present,
mpm_scatter's `stress` variable at line 112-113 is
    tau_p = 2 mu_p (F-R) F^T + la_p J (J-1) I ,  mu_p = E_p/2.4, la_p = E_p*0.2/0.72
so tau_p = E_p * K_p and therefore  d tau_p / d E_c = 1[p in c] * tau_p / E_p ,
recoverable from the cached Cauchy buffer as  p.sigma * |J| / E_p  (mpm_scatter:128).

Checks:
  1. p.sigma * |J|  ==  tau recomputed from F, mu, la          (the buffer is what we think)
  2. tau(1.7 E) == 1.7 * tau(E) at fixed F                      (exact homogeneity, the column)
Usage: PYTHONPATH=/workspace/Plexus/src python probe_sigma.py --device cuda:1
"""
from __future__ import annotations

import argparse
import json

import torch

from plexus import schema
from plexus import engine as E
from plexus.paths import config_path
from plexus.models.entities import _lame


def tau_of(F, mu, la):
    a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
    J = a * d - b * c
    cs, sn = (F[:, 0, 0] + F[:, 1, 1]), (F[:, 1, 0] - F[:, 0, 1])
    r = torch.sqrt(cs * cs + sn * sn) + 1e-9
    cs, sn = cs / r, sn / r
    R = torch.stack([torch.stack([cs, -sn], -1), torch.stack([sn, cs], -1)], -2)
    eye = torch.eye(2, device=F.device, dtype=F.dtype).expand(F.shape[0], 2, 2)
    return 2 * mu[:, None, None] * ((F - R) @ F.transpose(-2, -1)) \
        + eye * (la * J * (J - 1))[:, None, None], J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--frame", type=int, default=15)
    a = ap.parse_args()
    sim = schema.load(config_path("material", "material_cardio_cells.yaml"))
    sim.n_frames = a.frame
    H, _ = E.run(sim, out_path=None, device=a.device, progress=False)
    p = H.level("mpm_particle")
    tau, J = tau_of(p.F, p.mu, p.la)
    cached = p.sigma * J.abs().clamp_min(1e-9)[:, None, None]
    rel1 = float((cached - tau).norm() / tau.norm())
    mu2, la2 = _lame(p.youngs * 1.7)
    tau2, _ = tau_of(p.F, mu2, la2)
    rel2 = float((tau2 - 1.7 * tau).norm() / (1.7 * tau).norm())
    # the E-sensitivity column, straight off the cache
    dtau_dE = p.sigma * J.abs().clamp_min(1e-9)[:, None, None] / p.youngs[:, None, None]
    rel3 = float((dtau_dE * p.youngs[:, None, None] - tau).norm() / tau.norm())
    print(json.dumps({
        "sigma_buffer_present": bool(getattr(p, "sigma", None) is not None),
        "rel_err_cached_sigma_times_J_vs_recomputed_tau": rel1,
        "rel_err_homogeneity_tau(1.7E)_vs_1.7tau(E)": rel2,
        "rel_err_dtau_dE_times_E_vs_tau": rel3,
        "J_min": float(J.min()), "J_max": float(J.max()),
        "tau_absmax": float(tau.abs().max()),
    }, indent=2))


if __name__ == "__main__":
    main()
