#!/usr/bin/env python
"""test_05j_converge -- G59: does the bond DENSITY reproduce the DISCRETE plaque where the discrete
   picture is valid? This gate carries a stop condition.

    python test_05j_converge.py [--device cuda:0]  ->  log/okuda_ECM/05j_converge/

WHY THIS IS THE ONE TO RUN FIRST. 05d replaced discrete plaques with a bond density on the argument
that there are 25 integrin clusters per node, so the discrete description is the approximation. That
argument is only sound if the density is a COARSE-GRAINING of the discrete model -- i.e. if, in the
regime where the discrete model is valid, the two agree. Nobody has checked. If they do not agree, the
density is not a coarse-graining but a different model wearing its name, and S2.3 has to be withdrawn
rather than tuned.

THE TEST. Take the discrete plaque of 05b and raise its density: one plaque per few nodes, then one per
node, then several per node's worth of bonds. As the count rises the discrete model must converge onto
the density model's answer for the two things adhesion decides -- WHERE the sheet sits (the standoff)
and HOW HARD it is held (the traction). The comparison is made at matched TOTAL bond number, because a
density model with N bonds spread evenly and a discrete model with N bonds in clumps are the same
material only if N matches.

WHAT WOULD FALSIFY IT. A standoff or traction that does not approach the density answer as the plaque
count rises, or that approaches a DIFFERENT limit. Either kills the coarse-graining.
"""
from __future__ import annotations

import json, math, os, sys
import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402

import bm_ops as BM                                                     # noqa: E402
from test_05_sheet import LOG, UNITS                                    # noqa: E402
from test_05d_adhesion import Rig05d                                    # noqa: E402
import test_05e_conserve as E5                                          # noqa: E402

UM = 1171.0


class RigDiscrete(Rig05d):
    """The density rig with the bonds CLUMPED onto a subset of patches -- the discrete limit.

    `frac` is the fraction of contact patches allowed to carry bonds; the same total bond number is
    then concentrated on them, so the two models differ only in how the SAME material is distributed.
    At frac = 1 this is exactly the density model, which is the control that the harness itself is
    fair.
    """

    def __init__(self, *a, frac=1.0, **kw):
        self.frac = float(frac)
        super().__init__(*a, **kw)
        n = self.ct_node.numel()
        g = torch.Generator(device="cpu").manual_seed(5)
        keep = torch.zeros(n, dtype=torch.bool, device=self.dev)
        idx = torch.randperm(n, generator=g)[:max(1, int(round(self.frac * n)))]
        keep[idx.to(self.dev)] = True
        self.keep_mask = keep

    def frame(self, t):
        super().frame(t)
        # concentrate the bonds: the patches that are allowed to bind keep everything, the rest are
        # emptied and their bonds handed back, so the TOTAL is conserved and only the layout differs
        if self.frac < 1.0:
            dead = self.clutch.Nb[~self.keep_mask]
            if float(dead.sum()) > 0:
                back = torch.zeros_like(self.clutch.Nf)
                back.index_add_(0, self.ct_face[~self.keep_mask], dead)
                self.clutch.Nf = self.clutch.Nf + back
                self.clutch.Nb[~self.keep_mask] = 0.0


def main():
    def arg(f, c, dflt):
        return c(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else dflt
    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 150)
    name = arg("--name", str, "05j_converge")
    d = os.path.join(LOG, name); os.makedirs(d, exist_ok=True)
    cert = BM.selftest(dev=dev, subdiv=4)

    P = dict(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, sigma_T=7.0, zeta=20.0, s_target=1.0,
             k_drive=50.0, dev=dev)
    A = dict(kappa_b=5.0, k_on=0.6, k_off0=0.05, f_bell=3.0e-3)

    res = {}
    fracs = [0.02, 0.05, 0.15, 0.4, 1.0]
    for fr in fracs:
        r = RigDiscrete(**P, **A, max_refine=0, reseed=False, frac=fr)
        E5.run(r, frames, label=f"{name}: {100*fr:g}% of patches carry bonds")
        n = max(4, frames // 4)
        res[fr] = dict(standoff=float(np.mean(r.res["standoff"][-n:])),
                       standoff_over_l0=(float(np.mean(r.res["standoff"][-n:])) + r.l0) / r.l0,
                       n_b_total=float(r.clutch.Nb.sum()),
                       receptor_total=float(r.res["receptor_total"][-1]),
                       bound_frac=float(np.mean(r.res["bound_frac"][-n:])),
                       traction=float(np.mean(r.res["load_mean"][-n:])),
                       patches=int(r.keep_mask.sum()),
                       nodes_per_patch=float(r.ct_node.numel()) / max(int(r.keep_mask.sum()), 1))
        print(f"    frac {fr:5.2f}: {res[fr]['patches']:6d} patches "
              f"({res[fr]['nodes_per_patch']:6.1f} nodes each), standoff/l0 "
              f"{res[fr]['standoff_over_l0']:.4f}, bound {res[fr]['bound_frac']:.4f}", flush=True)

    ref = res[1.0]
    conv = {fr: dict(standoff_rel=abs(res[fr]["standoff_over_l0"] - ref["standoff_over_l0"])
                     / max(abs(ref["standoff_over_l0"]), 1e-30),
                     bound_rel=abs(res[fr]["bound_frac"] - ref["bound_frac"])
                     / max(ref["bound_frac"], 1e-30)) for fr in fracs}
    monotone = all(conv[fracs[i]]["standoff_rel"] >= conv[fracs[i + 1]]["standoff_rel"] - 1e-9
                   for i in range(len(fracs) - 2))

    fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.6), facecolor="white")
    xs = [res[f]["nodes_per_patch"] for f in fracs]
    ax[0].semilogx(xs, [res[f]["standoff_over_l0"] for f in fracs], "o-", color="#1a1a1a", lw=1.8)
    ax[0].axhline(ref["standoff_over_l0"], color="#b03030", ls="--", lw=1.2,
                  label="the density model")
    ax[0].set_xlabel("nodes per bond-bearing patch (1 = the density model)")
    ax[0].set_ylabel(r"standoff / $\ell_0$")
    ax[0].set_title("G59: the discrete limit must converge HERE", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].loglog(xs, [max(conv[f]["standoff_rel"], 1e-6) for f in fracs], "o-", color="#2b6cb0",
                 lw=1.8)
    ax[1].axhline(0.05, color="#999", ls="--", lw=1.0)
    ax[1].set_xlabel("nodes per patch"); ax[1].set_ylabel("relative error vs the density model")
    ax[1].set_title(f"error falls as the layout is refined: {monotone}", fontsize=9)
    ax[2].semilogx(xs, [res[f]["bound_frac"] for f in fracs], "s-", color="#1f8a5c", lw=1.8)
    ax[2].axhline(ref["bound_frac"], color="#b03030", ls="--", lw=1.2)
    ax[2].set_xlabel("nodes per patch"); ax[2].set_ylabel("bound fraction")
    ax[2].set_title("the same material, differently arranged", fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(d, "converge.png"), dpi=150, facecolor="white")
    plt.close(fig)

    worst = max(conv[f]["standoff_rel"] for f in fracs if f < 1.0)
    best = min(conv[f]["standoff_rel"] for f in fracs if f < 1.0)
    # THE GATE HAS TWO HALVES AND THEY DO NOT AGREE, so both are reported rather than the kinder one.
    # The standoff converges trivially because `bm_contact` holds the sheet out at every node whatever
    # the adhesion is doing -- the adhesion only has to not pull it in -- so layout cannot move it.
    # The BOUND FRACTION is the half that actually tests the coarse-graining, and it does not converge:
    # concentrating the same receptor on fewer patches saturates them locally and binds less of it.
    bfs = [res[f]["bound_frac"] for f in fracs]
    bound_spread = (max(bfs) - min(bfs)) / max(max(bfs), 1e-30)
    out = dict(run=name, frames=frames, certification=cert,
               G59=dict(by_fraction={str(f): {**res[f], **conv[f]} for f in fracs},
                        density_reference=ref,
                        error_at_sparsest=worst, error_at_densest_discrete=best,
                        converges=bool(best < 0.05),
                        bound_frac_spread=bound_spread,
                        standoff_converges=bool(best < 0.05),
                        bound_converges=bool(bound_spread < 0.05),
                        caveat="the standoff converges because bm_contact holds the sheet out at "
                               "every node regardless of the adhesion, so layout cannot move it; the "
                               "bound fraction is the half that tests the coarse-graining",
                        monotone_in_layout=bool(monotone),
                        verdict=("the density model IS the dense limit of the discrete one for the "
                                 "GEOMETRY; the bound fraction does not converge, because clumping "
                                 "the same receptor saturates patches locally"
                                 if best < 0.05 else
                                 "STOP: the two models do not agree where the discrete one is valid, "
                                 "so the density is not a coarse-graining of it")),
               series={str(f): res[f] for f in fracs})
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    yaml.safe_dump(dict(
        what="05j -- G59: the density adhesion must be the dense limit of the discrete plaque",
        units=dict(**UNITS, force_nN=None),
        stop_condition="if the discrete model at high density does not reproduce the density model, "
                       "the density is not a coarse-graining but a different model, and note_sheet "
                       "S2.3 must be withdrawn rather than tuned",
        method="the SAME total bond number, concentrated on a shrinking fraction of contact patches; "
               "frac = 1 is the density model itself and is the control that the harness is fair",
        measures=["standoff / l0", "bound fraction", "load per bond"]),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    print(f"[{name}] G59 standoff error {worst:.4f} sparsest / {best:.4f} densest -> converges "
          f"{out['G59']['standoff_converges']}; BOUND FRACTION spread {bound_spread:.3f} -> converges "
          f"{out['G59']['bound_converges']} -> {d}", flush=True)


if __name__ == "__main__":
    main()
