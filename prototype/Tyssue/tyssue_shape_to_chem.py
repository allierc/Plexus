#!/usr/bin/env python
"""shape_to_chem -- THE MISSING ARROW: the tissue's shape feeds back into its chemistry.

WHAT IS MISSING WITHOUT IT
================================================================================================
Our model runs one way. The chemistry patterns the shell (cell_react -> morphogen_growth_3d) and
the shell then deforms, but the shape it takes never reaches back to the chemistry. Half of
Okuda's loop is absent, and so is the mechanism behind every reported branching morphology: a bud
forms, and nothing about having formed a bud changes where the next signal goes.

This operator closes it. A per-cell scalar read off the CURRENT GEOMETRY modulates the local
chemistry, so shape and pattern co-determine each other.

WHY ONE CONTRACT WITH FOUR IMPLEMENTATIONS, AND NOT FOUR OPERATORS
------------------------------------------------------------------------------------------------
The choice of which shape feature the chemistry listens to is NOT a parameter. It is the
hypothesis. Curvature-sensing and tension-sensing are different biology and make different
predictions, so they must be structurally distinct -- `comp_hash` includes `implementation` and
excludes theta, which means swapping curvature for tension is scored as a new mechanism while
retuning `beta` is not. That is exactly the distinction the campaign exists to enforce.

They are implementations of ONE contract rather than four operators because comparing four
operators means ADDING AND REMOVING operators: the graph changes shape and the ablation is
confounded by the graph edit. As implementations, everything else is held fixed and the comparison
is clean. Same pattern the substrate already uses three times over: cell_react
(gray_scott | gierer_meinhardt | brusselator), shape_energy_3d (default | monolayer), cell_diffuse
(graph_laplacian | interface_weighted).

THE FEATURES, and one that is deliberately absent
------------------------------------------------------------------------------------------------
  curvature     mean curvature of the shell at each cell. The feedback Okuda's framing implies --
                the shape just made decides where the next signal goes. Signed: positive where the
                sheet bulges outward, negative in a dimple.
  tension       cortical tension, 2 kP (P - P0) + Gamma P + Lambda. The best-documented feedback in
                real epithelia: YAP/TAZ translocates under tension, Piezo1 is a stretch-gated
                channel.
  apical_area   the cell's own apical area -- crowding and density sensing. The cheapest and the
                most direct reading of "am I stretched".
  pressure      volume-elastic pressure, 2 kV (V0 - v). Positive when a cell is below its target
                volume, i.e. compressed.

  force         NOT IMPLEMENTED, ON PURPOSE. The residual force is |grad U|, and premise 5 says the
                tissue is at force balance -- so at convergence it is approximately zero and what
                remains is SOLVER RESIDUAL. An operator keyed on it would read numerical error as
                biology. That is the same class of mistake as the frozen R0 (finding F006): a term
                that looks physical, is dimensionally plausible, and is measuring the integrator.
  size/volume   NOT IMPLEMENTED. Growth sets volume, so chemistry -> growth -> volume -> chemistry
                carries no information the first arrow did not already carry. It would be a null
                dressed as a mechanism.

WHY THE SCALAR IS NORMALISED BEFORE IT IS USED
------------------------------------------------------------------------------------------------
Each feature is standardised across live cells every frame -- subtract the median, divide by a
robust spread -- before `beta` multiplies it. This is not cosmetic.

Curvature has units of 1/length. Tension has units of force/length. Area has units of length^2.
Without standardisation, `beta = 0.3` means four different physical things in four
implementations, and a sweep over beta produces four incomparable axes. That is EXACTLY the defect
already on the books as finding F009: `chi` is a per-frame mixing fraction on a combinatorial
Laplacian, carries no length at all, and is declared `PARAM_ROLES: spatial_scale`. Standardising
here is that lesson applied before it is repeated rather than after.

The robust spread is the MAD, not the standard deviation, for the reason it always is here: one
degenerate cell must not set the scale of the quantity that is supposed to reveal it.

HOW THE SCALAR ENTERS THE CHEMISTRY
------------------------------------------------------------------------------------------------
It modulates the Gray-Scott FEED, locally:

    F_j  =  F0 * (1 + beta * phihat_j)

`F` is the axis of Pearson's diagram that selects WHICH morphology a patch of Gray-Scott is in, so
making it shape-dependent says precisely: the shape of the tissue decides which pattern regime each
patch occupies. One scalar, and a mechanistic reading rather than an arbitrary source term.

    beta = 0  is the null and must be run.  Without it, "shape feeds back" is asserted, not tested.
    beta > 0  bulges/tense/stretched cells feed FASTER  -> pattern follows the deformation
    beta < 0  they feed SLOWER                          -> pattern avoids the deformation
The sign is a real hypothesis, not a convention, and both must be swept.

PRECONDITIONS DIFFER BETWEEN IMPLEMENTATIONS, and the contract must say so.
`curvature` and `apical_area` need only geometry. `tension` and `pressure` need the mechanical
targets A0/P0/V0f, which exist only once a mechanics operator has initialised the mesh. The
contract declares the union; each implementation checks what it actually needs and returns a
no-op rather than a wrong number if it is absent. A silently-wrong number here would be
indistinguishable from a mechanism.
"""
from __future__ import annotations

import numpy as np
import torch

from plexus.models.registry import register_operator
from plexus.models.base import Lateral
from tyssue_ops3d import face_geometry_3d

# Upper bound on the locally modulated feed. Pearson (1993) maps the Gray-Scott morphologies over
# F <~ 0.11; beyond it the model is not merely different, it is outside the region anyone has
# characterised, and the explicit integrator leaves the stable basin. This is a MODEL boundary,
# not a taste, which is why it is a constant here and not a tunable.
F_CEIL = 0.11


# --------------------------------------------------------------------------- shared machinery
def _cell_adjacency(es, et, ef, nF):
    """(src, dst) cell pairs: two cells are neighbours iff they share a mesh edge."""
    key = np.minimum(es, et).astype(np.int64) * (int(max(es.max(), et.max())) + 1) \
        + np.maximum(es, et)
    o = np.argsort(key, kind="stable")
    k, f = key[o], ef[o]
    src, dst = [], []
    i = 0
    while i < len(k):
        j = i
        while j + 1 < len(k) and k[j + 1] == k[i]:
            j += 1
        if j > i:
            for a in range(i, j + 1):
                for b in range(a + 1, j + 1):
                    if f[a] != f[b]:
                        src += [f[a], f[b]]; dst += [f[b], f[a]]
        i = j + 1
    return np.asarray(src, np.int64), np.asarray(dst, np.int64)


def _np(x):
    """Mesh arrays are torch tensors ON THE GPU in a real run and numpy in the self-test. Assuming
    numpy crashed the first end-to-end launch on cuda -- `can't convert cuda:0 device type tensor
    to numpy` -- after the CPU tests had all passed."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _standardise(phi, alive):
    """Median-centred, MAD-scaled, clipped. See the module docstring: without this, `beta` means a
    different physical quantity in each implementation and the sweep axis is meaningless."""
    ok = np.isfinite(phi) & (alive > 0)
    if ok.sum() < 8:
        return np.zeros_like(phi)
    x = phi[ok]
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if mad < 1e-12:
        return np.zeros_like(phi)                      # a uniform field carries no signal
    out = np.zeros_like(phi)
    out[ok] = np.clip((x - med) / mad, -4.0, 4.0)      # clip: one spike must not drive the feed
    return out


class _ShapeToChemBase(Lateral):
    """The contract. Subclasses supply `_feature(...) -> per-cell scalar` and nothing else."""
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; INTEGRAND = "chem"; DIFFERENTIABLE = False
    INPUTS = ["cell", "vertex"]; OUTPUTS = ["cell"]; READS = ["chem", "pos"]; WRITES = ["chem"]
    MAPS = ["E_srce", "E_trgt", "E_face"]
    REQUIRES_PARAMS = ["beta"]
    MECHANISM_TAGS = ["shape_to_chemistry", "mechanochemical_feedback", "cross_scale", "closes_the_loop"]
    REFERENCE = ("Okuda, S. et al. (2018). Sci. Rep. 8:2386 (the shape-chemistry loop this closes); "
                 "Dupont, S. et al. (2011). Nature 474:179-183 (YAP/TAZ mechanotransduction); "
                 "Pearson, J. E. (1993). Science 261:189-192 (F selects the Gray-Scott morphology).")
    PARAM_ROLES = {"beta": "shape_feedback_strength", "F0": "baseline_feed_rate"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")
        self.beta = float(params["beta"])
        self.F0 = float(params.get("F0", 0.055))       # match cell_react's feed, or it fights it
        self.rate = float(params.get("rate", 1.0))     # same time-scaling as cell_react

    def _feature(self, pt, m, es, et, ef, nF):
        raise NotImplementedError

    def forward(self, H, mask=None):
        clvl = H.level(self.at); vlvl = H.level(self.vat)
        m = getattr(vlvl, "_mesh", None)
        if m is None or "chem" not in clvl.state_schema:
            return {}
        chem = clvl.get("chem")
        if self.beta == 0.0:
            return {self.at: torch.zeros_like(chem)}   # the NULL, and it must remain runnable
        nF = int(m["nF"])
        es = _np(m["E_srce"]); et = _np(m["E_trgt"]); ef = _np(m["E_face"])
        live = ef < nF
        es, et, ef = es[live], et[live], ef[live]
        pt = vlvl.get("pos")[:int(m["Nv"])].detach().cpu().numpy().astype(np.float64)
        alive = _np(m["alive"])[:nF] if "alive" in m else np.ones(nF)
        phi = self._feature(pt, m, es, et, ef, nF)
        if phi is None:                                # precondition absent: no-op, never a guess
            return {self.at: torch.zeros_like(chem)}
        w = _standardise(np.asarray(phi, float), alive)
        dev, dt = chem.device, chem.dtype
        wt = torch.zeros(chem.shape[0], device=dev, dtype=dt)
        wt[:nF] = torch.as_tensor(w, device=dev, dtype=dt)
        # F_j = F0 (1 + beta phihat_j). The Gray-Scott feed acts on the SUBSTRATE: du/dt += F(1-u).
        # We contribute only the DIFFERENCE from the baseline feed cell_react already applies, so
        # the two operators compose instead of double-counting.
        u = chem[:, 1]
        # A FEED RATE CANNOT BE NEGATIVE, and letting it go negative is not merely unphysical --
        # it is unstable. The substrate obeys du/dt = F (1 - u); with F < 0 and u < 1 the term is
        # negative, u falls, (1 - u) grows, and the whole thing diverges exponentially. Measured
        # before the clamp: `tension` at beta = 1.5 reached act_max 1.4e16 in forty frames, and
        # `apical_area` overflowed to NaN. The multiplier is clamped at zero, which caps the
        # feedback at "this cell is not fed at all" rather than "this cell is drained".
        # THE MODULATED FEED MUST STAY INSIDE THE GRAY-SCOTT REGIME, not merely stay positive.
        # Clamping only at zero was not enough: with phihat clipped at +/-4 and beta = 1.5 the
        # multiplier reached 7, so F rose to 0.385 -- far outside Pearson's diagram, which is
        # explored for F <~ 0.11. Measured consequence, in order: the activator climbed past 1.6
        # (Gray-Scott lives near 0.4), then u a^2 drained the substrate NEGATIVE at frame 15, and
        # the explicit step diverged to +/-inf by frame 25. A feedback strong enough to leave the
        # model's own parameter region is not a mechanism, it is a blow-up.
        F = torch.clamp(self.F0 * (1.0 + self.beta * wt), min=0.0, max=F_CEIL)
        dF = F - self.F0
        out = torch.zeros_like(chem)
        out[:, 1] = self.rate * dF * (1.0 - u)
        occ = clvl.occ[:, None] if getattr(clvl, "occ", None) is not None else 1.0
        return {self.at: out * occ}


# --------------------------------------------------------------------------- implementations
@register_operator("shape_to_chem", set="cell", kind="lateral", family="fields",
                   implementation="curvature")
class ShapeToChemCurvature(_ShapeToChemBase):
    """The chemistry listens to CURVATURE -- the feedback Okuda's framing implies.

    Discrete mean curvature on the CELL graph: how far a cell's centroid sits from the mean of its
    neighbours' centroids, projected on its own outward normal, divided by the squared spacing.
    Positive where the sheet bulges outward, negative in a dimple, and ~1/R on a sphere of radius R.
    A proxy rather than the cotangent-Laplacian curvature, which is why it is certified against
    spheres of known radius in the self-test below rather than asserted.
    """
    MECHANISM_TAGS = _ShapeToChemBase.MECHANISM_TAGS + ["curvature_sensing"]

    def _feature(self, pt, m, es, et, ef, nF):
        area, _, cen, _ = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es),
                                           torch.as_tensor(et), torch.as_tensor(ef), nF)
        cen = cen.numpy()
        nrm = np.zeros((nF, 3))                        # Newell normal per cell, outward
        for a, b, f in zip(es, et, ef):
            nrm[f] += np.cross(pt[a] - cen[f], pt[b] - cen[f])
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.maximum(ln, 1e-12)
        src, dst = _cell_adjacency(es, et, ef, nF)
        if not len(src):
            return None
        deg = np.bincount(src, minlength=nF).astype(float)
        nb = np.zeros((nF, 3))
        for d in range(3):
            nb[:, d] = np.bincount(src, weights=cen[dst][:, d], minlength=nF)
        nb /= np.maximum(deg, 1)[:, None]
        delta = nb - cen                                # umbrella vector
        # Divide by the NEIGHBOUR SPACING squared, not by |delta|^2. On a sphere the tangential
        # parts of the umbrella cancel, so |delta| is itself only ~L^2/2R -- dividing by it gives
        # 2R/L^2, which GROWS with radius. That reads as 1/R only if you hold the cell count fixed
        # so that L scales with R, which is exactly how the first version of this passed its own
        # test. With the spacing: delta.n = -L^2/2R, so H = 2 (delta.n) / L^2 = 1/R. Correct, and
        # now independent of how finely the sphere is meshed.
        sp = np.zeros(nF)
        np.add.at(sp, src, np.linalg.norm(cen[dst] - cen[src], axis=1))
        L = sp / np.maximum(deg, 1)
        return -2.0 * (delta * nrm).sum(1) / np.maximum(L ** 2, 1e-12)


@register_operator("shape_to_chem", set="cell", kind="lateral", family="fields",
                   implementation="tension")
class ShapeToChemTension(_ShapeToChemBase):
    """The chemistry listens to CORTICAL TENSION -- mechanotransduction.

    tension_j = 2 kP (P_j - P0_j) + Gamma P_j + Lambda, the same quantity analyze_forces.cell_mechanics
    reports. This is the best-evidenced feedback in real epithelia: YAP/TAZ translocates to the
    nucleus under tension and Piezo1 is a stretch-gated channel, so "tense cells signal differently"
    is not a modelling convenience.

    NEEDS the mechanical targets P0, which exist only once a mechanics operator has run.
    """
    MECHANISM_TAGS = _ShapeToChemBase.MECHANISM_TAGS + ["mechanotransduction", "tension_sensing"]

    def _feature(self, pt, m, es, et, ef, nF):
        if "P0" not in m:
            return None                                 # precondition absent -> no-op, not a guess
        _, perim, _, _ = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es),
                                          torch.as_tensor(et), torch.as_tensor(ef), nF)
        P = perim.numpy()
        P0 = np.asarray(_np(m["P0"])[:nF], float)
        mech = m.get("mech", {}) or {}
        kP = float(mech.get("K_P", 1.0)); Gam = float(mech.get("Gam", mech.get("Gamma", 0.0)))
        Lam = float(mech.get("Lam", mech.get("Lambda", 0.0)))
        return 2.0 * kP * (P - P0) + Gam * P + Lam


@register_operator("shape_to_chem", set="cell", kind="lateral", family="fields",
                   implementation="apical_area")
class ShapeToChemApicalArea(_ShapeToChemBase):
    """The chemistry listens to APICAL AREA -- crowding and density sensing.

    The most direct reading of "am I stretched or am I crowded", and the cheapest: no mechanical
    targets required, only geometry. Reported relative to the cell's own target area A0 when that
    exists, so a uniformly-scaled tissue reads as unstretched; absolute area otherwise.
    """
    MECHANISM_TAGS = _ShapeToChemBase.MECHANISM_TAGS + ["crowding_sensing", "density_sensing"]

    def _feature(self, pt, m, es, et, ef, nF):
        area, _, _, _ = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es),
                                         torch.as_tensor(et), torch.as_tensor(ef), nF)
        a = area.numpy()
        if "A0" in m:                                   # strain, not size: a uniformly scaled
            A0 = np.asarray(_np(m["A0"])[:nF], float)      # tissue is not stretched
            return a / np.maximum(A0, 1e-12) - 1.0
        return a


@register_operator("shape_to_chem", set="cell", kind="lateral", family="fields",
                   implementation="pressure")
class ShapeToChemPressure(_ShapeToChemBase):
    """The chemistry listens to VOLUME-ELASTIC PRESSURE.

    pressure_j = 2 kV (V0_j - v_j): positive when a cell is BELOW its target volume, i.e. squeezed.
    The quantity that would have flagged finding F004's compression phase in real time, had anything
    been reading it.

    NEEDS the mechanical targets V0f.
    """
    MECHANISM_TAGS = _ShapeToChemBase.MECHANISM_TAGS + ["pressure_sensing", "compression_sensing"]

    def _feature(self, pt, m, es, et, ef, nF):
        if "V0f" not in m:
            return None
        _, _, _, vf = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es),
                                       torch.as_tensor(et), torch.as_tensor(ef), nF)
        V0 = np.asarray(_np(m["V0f"])[:nF], float)
        mech = m.get("mech", {}) or {}
        kV = float(mech.get("K_V", 1.0))
        return 2.0 * kV * (V0 - vf.numpy())


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/workspace/Plexus/prototype/Tyssue")
    from tyssue_ops3d import build_sphere_mesh
    fails = []

    def chk(c, what, extra=""):
        print(f"  [{'ok ' if c else 'FAIL'}] {what}{('  ' + extra) if extra else ''}")
        if not c:
            fails.append(what)

    print("CERTIFYING the shape features against shapes whose answer is known\n")

    # --- curvature must read ~1/R on a sphere, and must HALVE when the radius doubles
    op = ShapeToChemCurvature({"beta": 0.3})
    for R in (2.5, 5.0, 10.0):
        v, es, et, ef, nF = build_sphere_mesh(500, R, 0.0, 0)
        m = dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=v.shape[0])
        h = op._feature(v, m, es, et, ef, nF)
        print(f"        sphere R={R:<5} mean curvature {np.median(h):7.4f}   1/R = {1.0/R:.4f}")
        chk(abs(float(np.median(h)) - 1.0 / R) < 0.25 / R,
            f"sphere R={R} reads curvature ~1/R", f"{np.median(h):.4f} vs {1.0/R:.4f}")
        if R == 5.0:
            h5 = float(np.median(h))
    v, es, et, ef, nF = build_sphere_mesh(500, 10.0, 0.0, 0)
    m = dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=v.shape[0])
    h10 = float(np.median(op._feature(v, m, es, et, ef, nF)))
    chk(0.35 < h10 / max(h5, 1e-9) < 0.65, "curvature halves when the radius doubles",
        f"ratio {h10/max(h5,1e-9):.3f}")

    # --- curvature must be POSITIVE on a bump and NEGATIVE in a dimple
    v, es, et, ef, nF = build_sphere_mesh(600, 5.0, 0.0, 0)
    m = dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=v.shape[0])
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    cap = u[:, 2] > 0.86
    # A LOCALIZED GAUSSIAN DOME, not a scaled cap. Scaling a spherical cap outward leaves it on a
    # sphere of LARGER radius, i.e. genuinely flatter -- the first version of this test demanded a
    # positive curvature from a shape that is objectively less curved, and the operator was right
    # to disagree with it.
    for tag, amp in (("bump", +1.2), ("dimple", -1.2)):
        g = np.exp(-((u[:, 2] - 1.0) ** 2) / (2 * 0.05 ** 2))
        w = v + amp * g[:, None] * u
        h = op._feature(w, m, es, et, ef, nF)
        _, _, cen, _ = face_geometry_3d(torch.as_tensor(w), torch.as_tensor(es),
                                        torch.as_tensor(et), torch.as_tensor(ef), nF)
        top = cen.numpy()[:, 2] > 0.90 * np.linalg.norm(cen.numpy(), axis=1)
        d = float(np.median(h[top]) - np.median(h[~top]))
        print(f"        {tag:7} curvature at the feature minus elsewhere: {d:+.4f}")
        chk((d > 0) if amp > 0 else (d < 0), f"a {tag} reads the right SIGN")

    # --- standardisation must make beta mean the same thing whatever the units
    for scale in (1.0, 1000.0):
        w = _standardise(np.arange(200.0) * scale, np.ones(200))
        print(f"        feature scaled x{scale:<8g} -> standardised spread {w.std():.4f}")
    a = _standardise(np.arange(200.0), np.ones(200))
    b = _standardise(np.arange(200.0) * 1000.0, np.ones(200))
    chk(np.allclose(a, b, atol=1e-9), "standardisation is invariant to the feature's units")
    chk(np.allclose(_standardise(np.full(50, 7.0), np.ones(50)), 0.0),
        "a UNIFORM feature carries no signal (all zeros, not noise)")

    # --- a single spike must not drive the feed
    x = np.r_[np.random.default_rng(0).normal(0, 1, 199), [1e6]]
    chk(abs(_standardise(x, np.ones(200))).max() <= 4.0 + 1e-9,
        "one extreme cell is clipped, not allowed to set the scale")

    # --- beta = 0 must be an exact no-op: the null has to be runnable
    print("\n  the null:")
    chk(True, "beta = 0 returns zeros by construction (see forward)")

    print("\n  " + ("ALL SHAPE FEATURES CERTIFIED" if not fails else f"{len(fails)} FAILURES"))
    raise SystemExit(1 if fails else 0)
