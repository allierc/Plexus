#!/usr/bin/env python
"""Direct physics test of the monolayer-thickness operator (no Hierarchy needed): validates the geometry
(V1), the emergent-bending signature (V2: apical>basal area on a curved sheet), and a localized buckle
(V3), plus that it registered as the `monolayer` implementation of the shape_energy_3d contract."""
import numpy as np, torch
from tyssue_ops3d import build_sphere_mesh, face_geometry_3d
import tyssue_monolayer as ML
from tyssue_monolayer import monolayer_geometry_3d, _monolayer_energy_core
from plexus.models.registry import get_contract

torch.set_default_dtype(torch.float64)
R, H0, N = 5.0, 0.4, 200
verts, es, et, ef, nF = build_sphere_mesh(N, R, jitter=0.15, seed=0)
pos = torch.as_tensor(verts); es_t, et_t, ef_t = (torch.as_tensor(x) for x in (es, et, ef))
h_cell = torch.full((nF,), H0)

print("=== registration ===")
c = get_contract("shape_energy_3d")
print(f" shape_energy_3d implementations: {sorted(c.implementations)}  (kind={c.kind}, family={c.family})")
assert "monolayer" in c.implementations, "monolayer impl not registered!"

print("\n=== V1: geometry sanity (sphere R=%.1f, h0=%.2f, %d cells) ===" % (R, H0, nF))
v_f, s_f, A_ap, A_ba = monolayer_geometry_3d(pos, es_t, et_t, ef_t, nF, h_cell)
area_mid, _, _, _ = face_geometry_3d(pos, es_t, et_t, ef_t, nF)
A_tot = float(area_mid.sum())
print(f" all v_f > 0 : {bool((v_f > 0).all())}   min v_f={float(v_f.min()):.4f}")
print(f" sum(v_f)={float(v_f.sum()):.2f}   A_tot*h0={A_tot*H0:.2f}   ratio={float(v_f.sum())/(A_tot*H0):.3f} (expect ~1)")
print(f" v_f ~ A_mid*h0 ? corr={np.corrcoef(v_f.numpy(), area_mid.numpy()*H0)[0,1]:.4f}")
print(f" all s_f > 0 : {bool((s_f > 0).all())}   s_f/(2A_mid) median={float((s_f/(2*area_mid)).median()):.3f} (>1: +lateral)")

print("\n=== V2: emergent-bending signature (apical outer > basal inner on a curved shell) ===")
ratio = (A_ap / A_ba.clamp(min=1e-9)).median().item()
theo = ((R + H0/2) / (R - H0/2))**2
print(f" median A_apical/A_basal = {ratio:.4f}   theory ((R+h/2)/(R-h/2))^2 = {theo:.4f}")
print(f" -> curvature makes apical!=basal => surface tension penalises bending (EMERGENT). match={abs(ratio-theo)<0.02}")

def relax(pos0, V_eq, k_v=4.0, kappa_s=0.2, iters=250, eta=0.06, cap_frac=0.10):
    x = pos0.clone()
    eocc = torch.ones(es_t.shape[0]); vocc = torch.ones(x.shape[0])
    R0t = torch.as_tensor(float(np.linalg.norm(verts, axis=1).mean()))
    cap = cap_frac * (x[et_t] - x[es_t]).norm(dim=-1).mean()
    for _ in range(iters):
        xg = x.detach().requires_grad_(True)
        E = _monolayer_energy_core(xg, es_t, et_t, ef_t, nF, h_cell, V_eq, torch.ones(nF),
                                   k_v, kappa_s, 0.0, 0.02, R0t, eocc, vocc)
        g = torch.nan_to_num(torch.autograd.grad(E, xg)[0])
        step = -eta * g
        step = step * torch.clamp(cap / (step.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
        x = (x + step).detach()
    return x

print("\n=== V3a: rest stability (V_eq = rest prism volume; should stay ~spherical) ===")
V_rest = v_f.detach()
x_rest = relax(pos, V_rest)
r0 = pos.norm(dim=1); r1 = x_rest.norm(dim=1)
print(f" radius rest: mean {float(r0.mean()):.3f}->{float(r1.mean()):.3f}  std {float(r0.std()):.3f}->{float(r1.std()):.3f}  (stable if small change)")

print("\n=== V3b: localized buckle (raise V_eq 1.8x in the z>0.6R polar cap) ===")
cap_cells = []                                            # cells whose centroid is in the polar cap
_, _, cen, _ = face_geometry_3d(pos, es_t, et_t, ef_t, nF)
capmask = cen[:, 2] > 0.6 * R
V_grow = V_rest.clone(); V_grow[capmask] = V_rest[capmask] * 1.8
x_buck = relax(pos, V_grow, iters=400)
# radius of the cap vertices (vertices touching a cap cell) before/after
capv = torch.zeros(pos.shape[0], dtype=torch.bool)
capv[es_t[capmask[ef_t]]] = True
rb0 = pos[capv].norm(dim=1).max().item(); rb1 = x_buck[capv].norm(dim=1).max().item()
ro0 = pos[~capv].norm(dim=1).max().item(); ro1 = x_buck[~capv].norm(dim=1).max().item()
print(f" cap  max radius: {rb0:.3f} -> {rb1:.3f}   (delta {rb1-rb0:+.3f})")
print(f" rest max radius: {ro0:.3f} -> {ro1:.3f}   (delta {ro1-ro0:+.3f})")
print(f" -> cap bulged OUT more than the rest: {(rb1-rb0) > (ro1-ro0) + 0.05}")
print("\nDONE")
