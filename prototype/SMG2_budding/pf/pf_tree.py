"""pf_tree -- the mechanism search space for the phase-field SMG loop, as a tree over BIOLOGICAL
HYPOTHESES for HOW/WHERE clefts subdivide the dense connected bud. Every branch shares the SAME
validated phase-field substrate (pf_sim: dense tissue phi, volume-controlled growth, self-limiting
clefts); a branch fixes the cleft MECHANISM and the parameter ranges. This is the phenomenology-from-
literature / operators-from-repo replacement for the old sparse-agent mechanism_tree.

Hypotheses (organs_genesis review + Tissue active matter; repos Chaste/reaction-diffusion/CompuCell3D):
  focal_ecm            -- fibronectin deposited at concave necks, positive feedback -> penetrating cleft
                          (Yamada lab: Harunaga 2011/2014, Wang 2021). cleft_mode=curvature.
  turing_prepattern    -- a reaction-diffusion (Gray-Scott) morphogen sets cleft spacing = lobule
                          wavelength (Menshykau & Iber). cleft_mode=turing.
  differential_adhesion-- cell-cell vs cell-matrix adhesion (surface tension) sets lobule SIZE with only
                          weak active clefting (Steinberg / Wang-Yamada). cleft_mode=curvature, high kappa.
  confined_growth      -- proliferation under basement-membrane confinement buckles the bud into clefts
                          (Varner / Nelson / Goodwin). cleft_mode=curvature, high growth_frac.

  BRANCHES                    -> {hypothesis, cleft_mode, param ranges}
  sample_params(branch, rng)  -> params dict for pf_sim
  build_params(branch, p)     -> pf_sim params (adds cleft_mode)
  encode(branch, p)           -> (feature vector, names) for the surrogate
"""
from __future__ import annotations
import numpy as np

BRANCHES = {
    "focal_ecm": dict(
        hypothesis="focal fibronectin/ECM at concave necks, positive feedback -> penetrating cleft (Yamada)",
        cleft_mode="curvature",
        params={"kappa": (0.9, 1.8), "growth_frac": (1.15, 1.7), "s": (0.7, 1.8), "lam": (0.8, 1.5),
                "kappa_gate": (0.030, 0.060), "thick_gate": (0.45, 0.65)}),
    "turing_prepattern": dict(
        hypothesis="reaction-diffusion Turing morphogen sets cleft spacing = lobule wavelength (Menshykau-Iber)",
        cleft_mode="turing",
        params={"kappa": (0.9, 1.6), "growth_frac": (1.15, 1.7), "feed": (0.028, 0.045),
                "kill": (0.058, 0.066), "Dv": (0.06, 0.10), "s": (0.9, 1.5), "lam": (0.9, 1.4)}),
    "differential_adhesion": dict(
        hypothesis="cell-cell vs cell-matrix adhesion (surface tension) sets lobule size; weak active cleft (Steinberg)",
        cleft_mode="curvature",
        params={"kappa": (1.2, 2.4), "growth_frac": (1.2, 1.8), "s": (0.3, 0.9), "lam": (0.6, 1.1),
                "kappa_gate": (0.040, 0.070), "thick_gate": (0.50, 0.70)}),
    "confined_growth": dict(
        hypothesis="proliferation under basement-membrane confinement buckles the bud into clefts (Varner-Nelson)",
        cleft_mode="curvature",
        params={"kappa": (0.9, 1.6), "growth_frac": (1.5, 2.2), "beta": (3.0, 7.0), "s": (0.8, 1.6),
                "lam": (0.9, 1.5), "thick_gate": (0.45, 0.60)}),
}

# union of all tunable params (for the surrogate encoding), with global normalization ranges
PARAM_RANGE = {
    "kappa": (0.9, 2.4), "growth_frac": (1.15, 2.2), "s": (0.3, 1.8), "lam": (0.6, 1.5),
    "kappa_gate": (0.030, 0.070), "thick_gate": (0.45, 0.70), "beta": (3.0, 7.0),
    "feed": (0.028, 0.045), "kill": (0.058, 0.066), "Dv": (0.06, 0.10),
}
PARAM_FEATS = list(PARAM_RANGE)
MODES = ["curvature", "turing"]


def sample_params(branch, rng):
    out = {}
    for k, (lo, hi) in BRANCHES[branch]["params"].items():
        out[k] = float(rng.uniform(lo, hi))
    return out


def build_params(branch, params):
    """pf_sim params for this spec (branch fixes the cleft mode)."""
    return dict(cleft_mode=BRANCHES[branch]["cleft_mode"], **params)


def encode(branch, params):
    """Fixed feature vector: branch one-hot + cleft-mode one-hot + normalized params (union)."""
    feat, names = [], []
    for b in BRANCHES:
        feat.append(1.0 if b == branch else 0.0); names.append(f"branch_{b}")
    mode = BRANCHES[branch]["cleft_mode"]
    for m in MODES:
        feat.append(1.0 if m == mode else 0.0); names.append(f"mode_{m}")
    for k in PARAM_FEATS:
        lo, hi = PARAM_RANGE[k]; v = params.get(k, lo)
        feat.append(float((v - lo) / (hi - lo + 1e-9))); names.append(k)
    return np.array(feat, np.float32), names


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"{'branch':22} {'mode':10} hypothesis")
    for b in BRANCHES:
        print(f"{b:22} {BRANCHES[b]['cleft_mode']:10} {BRANCHES[b]['hypothesis']}")
        vec, names = encode(b, sample_params(b, rng))
    print(f"\nfeature-vector length = {len(vec)}  ({len(BRANCHES)} branches + {len(MODES)} modes + "
          f"{len(PARAM_FEATS)} params)")
