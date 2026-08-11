"""protease_ops -- the first FIELDS in this model, and the arithmetic that decides they are worth it.

THREE SPECIES, AND ONLY TWO OF THEM ARE FIELDS.

  MMP        soluble matrix metalloproteinase (MMP-2/9, ~72 kDa). Cuts collagen IV. Diffuses freely
             in the extracellular space, D ~ 10-100 um^2/s.                        --> a FIELD
  TIMP       tissue inhibitor of metalloproteinases. Also soluble, similar size, similar D. Binds MMP
             1:1 and inactivates it.                                               --> a FIELD
  MT1-MMP    membrane-type MMP (MMP14). A TRANSMEMBRANE protein: it does not enter the extracellular
             space, it diffuses only within its own cell's membrane, and it is what actually cuts a
             hole where a cell decides to.                             --> a per-CELL STATE, no field

WHY THAT SPLIT IS FORCED, and it is the same argument that kept the integrin out of a PDE, giving the
opposite answer for the opposite reason. In one 600 s frame a soluble protease travels
sqrt(4 D t) = 155-490 um, against a spheroid of 318 um and a cell of 10 um. A soluble MMP is therefore
UNIFORM OVER THE WHOLE SPHEROID within a frame, and a uniform protease cannot open a hole anywhere in
particular -- it dissolves the sheet everywhere at once, exactly as uniform starvation did. To localise
to one cell it would need D < 0.042 um^2/s, which is 240-2400x slower than a protein of that size: not
a tuning range, a different molecule. That molecule is MT1-MMP, and it is membrane-tethered.

So the integrin escaped its PDE because it is CONFINED to a small domain; the soluble protease would be
well-mixed because it TRAVELS FAR. Confinement is what decides whether a species is worth resolving.

REFERENCES FOR THE KINETICS (used, not merely consulted):
  Karagiannis, E.D., Popel, A.S. (2004) A theoretical model of type I collagen proteolysis by MMP-2
    and MT1-MMP in the presence of TIMP-2. J. Biol. Chem. 279(37):39105-39114. -- the canonical ODE
    network for MT1-MMP / TIMP-2 / proMMP-2 / MMP-2 / collagen, and the source of the reaction
    topology used in `test_05h_ternary.py`.
  Olson, M.W. et al. (1997) J. Biol. Chem. 272:29975; Butler, G.S. et al. (1998) J. Biol. Chem.
    273:871. -- the measured association kinetics for TIMP-2 binding MT1-MMP and the proMMP-2
    hemopexin domain.
  Sato, H., Takino, T. (2010) Cancer Sci. 101(4):843. -- MT1-MMP as the pericellular activator.
  Ki(TIMP-2 -> MT1-MMP) = 0.16 nM; Ki(TIMP-3 -> MT3-MMP) = 0.008 nM.

WHAT MAKES THE FIELD WORTH SOLVING ANYWAY. With a sink, a field has a length scale:

    dc/dt = D grad^2 c + s - k c        ==>   a source decays over  sqrt(D/k)

and that is what gives a breach a SIZE rather than letting it fill the sphere. Here the sink is not a
decay constant but a reaction with TIMP, which is measurable:

    MMP + TIMP --k_i--> inactive

so the breach diameter becomes a PREDICTION of the MMP/TIMP balance rather than a parameter. The
arithmetic to check it against: a 10 um breach needs an effective k of 0.1-0.5 /s (half-life 1.4-6.9 s)
and a 50 um breach 0.004-0.02 /s (35-170 s).

THE SOLVE. Explicit diffusion would need dt*D/h^2 < 1/2, i.e. 1,300-13,400 substeps per frame at these
D -- and, unlike the sheet's rate bound, it QUADRUPLES at every 1->4 refinement. So the diffusion is
solved semi-implicitly, (I + dt D L) c = c_old, by conjugate gradient on the finite-volume Laplacian:
unconditionally stable, one solve per frame, and indifferent to how fine the mesh gets.
"""
from __future__ import annotations

import torch


def face_laplacian(x, F, live):
    """Finite-volume Laplacian on the LIVE faces of a surface mesh.

    Two faces are neighbours if they share an edge; the flux between them is the shared edge length
    over the distance between centroids. Returns the neighbour pairs and their weights, plus each
    face's area, which is the control volume. A face at the rim of a hole simply has fewer neighbours
    -- the operator is zero-flux there, which is the right boundary condition for a torn sheet: a
    protease does not leak out of the sheet, it leaks around it.
    """
    Fl = F[live]
    m = Fl.shape[0]
    e = torch.cat([Fl[:, [0, 1]], Fl[:, [1, 2]], Fl[:, [2, 0]]], 0)
    e = torch.stack([e.min(dim=1).values, e.max(dim=1).values], 1)
    uniq, inv = torch.unique(e, dim=0, return_inverse=True)
    fid = torch.arange(m, device=x.device).repeat(3)
    order = torch.argsort(inv)
    inv_s, fid_s = inv[order], fid[order]
    # an interior edge appears exactly twice; take consecutive pairs
    same = inv_s[1:] == inv_s[:-1]
    i = fid_s[:-1][same]
    j = fid_s[1:][same]
    eid = inv_s[:-1][same]
    L_e = (x[uniq[eid, 0]] - x[uniq[eid, 1]]).norm(dim=1)
    cen = x[Fl].mean(1)
    dist = (cen[i] - cen[j]).norm(dim=1).clamp_min(1e-30)
    w = L_e / dist
    area = 0.5 * torch.cross(x[Fl[:, 1]] - x[Fl[:, 0]], x[Fl[:, 2]] - x[Fl[:, 0]],
                             dim=1).norm(dim=1).clamp_min(1e-30)
    return i, j, w, area


def lap_matvec(c, i, j, w, area):
    """L c, with L the (positive semi-definite) FV Laplacian: sum_j w_ij (c_i - c_j) / area_i."""
    out = torch.zeros_like(c)
    fl = w * (c[i] - c[j])
    out.index_add_(0, i, fl)
    out.index_add_(0, j, -fl)
    return out / area


def diffuse_implicit(c, D, dt, i, j, w, area, iters=60, tol=1e-10):
    """Solve (I + dt D L) c_new = c by conjugate gradient.

    Semi-implicit because the explicit bound dt*D/h^2 < 1/2 costs 1,300-13,400 substeps per frame at
    a protease's D, and quadruples at every refinement -- a cost that grows exactly where the mesh is
    doing its job. CG is unconditionally stable and costs one solve, whatever h is.
    """
    def A(v):
        return v + dt * D * lap_matvec(v, i, j, w, area)
    r = c - A(c.clone())
    p = r.clone()
    rs = (r * r).sum()
    x = c.clone()
    for _ in range(iters):
        if rs.sqrt() < tol:
            break
        Ap = A(p)
        alpha = rs / (p * Ap).sum().clamp_min(1e-300)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        p = r + (rs_new / rs.clamp_min(1e-300)) * p
        rs = rs_new
    return x.clamp_min(0.0)


# ---------------------------------------------------------------------------------------------
#  certification -- the field had none, unlike the strain measure, and that is how it shipped a
#  Laplacian nobody had checked against a known answer
# ---------------------------------------------------------------------------------------------
def selftest(dev="cuda:0", subdiv=4, verbose=True):
    """G39--G41: check the operator against functions whose answers are known in closed form.

    A SPHERE HAS EIGENFUNCTIONS, so a surface Laplacian on one can be checked exactly rather than
    plausibly: for a spherical harmonic of degree l on a sphere of radius R,

        lap Y_l = -l(l+1)/R^2 * Y_l

    so the measured Rayleigh quotient <Y, lap Y>/<Y, Y> must return -l(l+1)/R^2. l = 1 (Y ~ z) and
    l = 2 (Y ~ 3z^2 - 1) are enough to catch a wrong weight, a missing area division or a sign.

    And a POINT SOURCE WITH A SINK HAS A LENGTH: the steady solution of D lap c - k c + s delta decays
    as exp(-r/sqrt(D/k)). That length is the entire reason the field is worth solving -- it is what
    gives a breach a size instead of letting it fill the sphere -- so it is measured against its own
    closed form before any run uses it.
    """
    import math
    out, dt_ = {}, torch.float64
    import bm_ops as BM
    S = BM.Sheet(subdiv=subdiv, R0=1.0, dev=dev, dtype=dt_)
    i, j, w, a = face_laplacian(S.x, S.F_all, S.live)
    cen = (S.x[S.Fc].mean(1) - S.c)
    r = cen.norm(dim=1)
    z = cen[:, 2] / r
    for l, Y in ((1, z), (2, 3 * z * z - 1)):
        LY = lap_matvec(Y, i, j, w, a)
        # the FV operator returns +lap in the "sum_j w(c_i - c_j)/A" convention, i.e. MINUS the
        # Laplacian, so the eigenvalue comes back positive
        lam = float((Y * LY * a).sum() / (Y * Y * a).sum())
        out[f"harmonic_l{l}"] = lam
        out[f"harmonic_l{l}_expected"] = l * (l + 1)
        out[f"harmonic_l{l}_relerr"] = abs(lam - l * (l + 1)) / (l * (l + 1))
    # the decay length of a point source with a sink
    D, k = 1.0, 400.0
    c = torch.zeros(S.m, device=dev, dtype=dt_)
    c[int(torch.argmax(cen[:, 2]))] = 1.0 / float(a[int(torch.argmax(cen[:, 2]))])
    for _ in range(400):                        # relax to steady state
        c = diffuse_implicit(c, D, 1.0e-3, i, j, w, a) / (1.0 + k * 1.0e-3)
        c[int(torch.argmax(cen[:, 2]))] += 1.0e-3 / float(a[int(torch.argmax(cen[:, 2]))])
    src = cen[int(torch.argmax(cen[:, 2]))] / cen[int(torch.argmax(cen[:, 2]))].norm()
    ang = torch.acos((cen / r[:, None] * src).sum(1).clamp(-1, 1))
    sel = (ang > 0.05) & (ang < 0.9) & (c > 0)
    if int(sel.sum()) > 10:
        A_ = torch.stack([ang[sel], torch.ones_like(ang[sel])], 1)
        sol = torch.linalg.lstsq(A_, torch.log(c[sel])[:, None]).solution
        out["decay_length_measured"] = float(-1.0 / sol[0, 0])
        out["decay_length_expected"] = math.sqrt(D / k)
        out["decay_length_relerr"] = abs(out["decay_length_measured"]
                                         - out["decay_length_expected"]) / out["decay_length_expected"]
    # conservation with no source and no sink
    c0 = torch.rand(S.m, device=dev, dtype=dt_)
    tot0 = float((c0 * a).sum())
    c1 = diffuse_implicit(c0, 5.0, 1.0, i, j, w, a)
    out["conservation_relerr"] = abs(float((c1 * a).sum()) - tot0) / tot0
    if verbose:
        print(f"[protease_ops selftest] subdiv {subdiv}: {S.m} faces, {i.numel()} adjacencies "
              f"(closed surface expects {3*S.m//2})")
        for l in (1, 2):
            print(f"  Laplacian on Y_{l}: eigenvalue {out[f'harmonic_l{l}']:.6f} against "
                  f"{out[f'harmonic_l{l}_expected']} -> {out[f'harmonic_l{l}_relerr']:.3e}   [G39]")
        if "decay_length_measured" in out:
            print(f"  point source + sink: decay length {out['decay_length_measured']:.5f} against "
                  f"sqrt(D/k) = {out['decay_length_expected']:.5f} -> "
                  f"{out['decay_length_relerr']:.3e}   [G41]")
        print(f"  the solve conserves the field: {out['conservation_relerr']:.3e}   [G40]")
    return out


if __name__ == "__main__":
    import sys as _s
    _d = _s.argv[_s.argv.index("--device") + 1] if "--device" in _s.argv else "cuda:0"
    selftest(dev=_d)
