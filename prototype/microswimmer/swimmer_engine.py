"""Microswimmer engine: organism (a sphere) + surface_node (its beating membrane),
a flow field (analytic Stokes) and a chemical field (advection-diffusion).

Generic over the spec -- it contains no scenario-specific branching. Builds the two
sets and two fields from the validated spec, binds the chemical field to the flow
field + organism, then iterates the schedule (the same x_{t+1}=Phi(x_t) loop and
the same `aggregate` / `<field>.diffuse` builtins as engine2.py).

Reference: Liu, Costello & Kanso, Nat Commun 16, 4154 (2025),
doi:10.1038/s41467-025-59413-x  (papers/Flow-Physics-drives-functional-design-of-microswimmers/).
"""

from __future__ import annotations

import os
import math
import sys
import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # prototype/ root

from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator

import swimmer_fields            # noqa: F401  registers flow + chemical fields
import ops_swim                  # noqa: F401  registers the four operators
import squirmer
from scenario_schema import load  # reuse the SAME validator (the contract gatekeeper)


def build(sc, device="cpu"):
    H = Hierarchy()
    H.config = sc
    width = float(getattr(sc, "world", 1.0))
    H.world_width = width

    # --- organism set (top-level sphere) ---
    os_ = sc.sets["organism"]
    n = int(os_["n"])
    R = float(os_.get("radius", 0.14))
    types = os_["types"]
    type_names = list(types.keys())                      # e.g. ['sessile'] or ['motile']
    n_mode = int(os_.get("n_mode", 40))
    state = torch.zeros(n, 4, device=device)
    # placement: a single cell centred; >1 tiled along x
    if n == 1:
        state[0, :2] = torch.tensor([width * 0.5, 0.5], device=device)
    else:
        for i in range(n):
            state[i, :2] = torch.tensor([width * (i + 1) / (n + 1), 0.5], device=device)
    org = Level("organism", level=1, state=state)
    org.type_names = type_names
    node_type = torch.zeros(n, dtype=torch.long, device=device)
    feeding = torch.zeros(n, device=device)
    mouth_off = torch.zeros(n, device=device)
    modes = torch.zeros(n, n_mode, device=device)
    for tid, (tname, t) in enumerate(types.items()):
        # one organism per spec here; assign all of this type's cells the design
        sel = slice(0, n)                                # single-type specs (n small)
        node_type[sel] = tid
        feeding[sel] = float(t.get("feeding_area", 0.5))
        mouth_off[sel] = float(t.get("mouth_offset", 0.0))
        B = squirmer.design_modes(float(t.get("feeding_area", 0.5)), tname, n_mode)
        modes[sel] = torch.tensor(B, dtype=torch.float32, device=device)
    org.register_buffer("node_type", node_type)
    org.register_buffer("axis", torch.full((n,), float(os_.get("axis", 0.0)), device=device))
    org.register_buffer("radius", torch.full((n,), R, device=device))
    org.register_buffer("feeding", feeding)
    org.register_buffer("mouth_off", mouth_off)
    org.register_buffer("modes", modes)
    H.add_level(org)

    # --- surface_node set (membrane: nodes around the sphere boundary) ---
    ss = sc.sets["surface_node"]
    ppc = int(ss["per_parent"])
    Ns = n * ppc
    parent = torch.arange(n, device=device).repeat_interleave(ppc)
    alpha = (torch.arange(ppc, device=device).float() / ppc) * 2 * math.pi   # around the circle
    alpha = alpha.repeat(n)
    sn = Level("surface_node", level=0, state=torch.zeros(Ns, 2, device=device),
               parent=parent, parent_name="organism", role="membrane")
    sn.register_buffer("theta", alpha)                   # angle from the swim axis
    sn.register_buffer("slip", torch.zeros(Ns, 2, device=device))
    # mouth colouring (a TYPE within surface_node, membership set geometrically by the cap)
    mu1 = 1.0 - 2.0 * feeding[parent]                    # cap edge: cos(theta) > mu1 -> mouth
    is_mouth = torch.cos(alpha - mouth_off[parent]) > mu1
    snt = torch.where(is_mouth, torch.zeros(Ns, dtype=torch.long, device=device),
                      torch.ones(Ns, dtype=torch.long, device=device))
    sn.register_buffer("node_type", snt)
    sn.type_names = ["mouth", "cilia"]
    H.add_level(sn)

    # --- tracer set (optional: food parcels carried by the fluid) ---
    if "tracer" in sc.sets:
        ts = sc.sets["tracer"]
        Nt = int(ts["n"])
        gt = torch.Generator(device=device).manual_seed(sc.seed + 7)
        pos = torch.rand(Nt, 2, generator=gt, device=device)
        pos[:, 0] = pos[:, 0] * width
        # keep initial parcels OUT of the sphere (flow is zero inside -> they'd sit stuck)
        d0 = pos - org.state[0, :2]
        inside = d0.norm(dim=1) < R * 1.2
        ang = torch.atan2(d0[:, 1], d0[:, 0])
        pos[inside] = org.state[0, :2] + R * 1.3 * torch.stack(
            [torch.cos(ang[inside]), torch.sin(ang[inside])], 1)
        tr = Level("tracer", level=0, state=torch.cat([pos, torch.zeros(Nt, 2, device=device)], 1))
        tr.type_names = ["food"]
        tr.register_buffer("node_type", torch.zeros(Nt, dtype=torch.long, device=device))
        H.add_level(tr)
        H.captured = 0

    # --- fields ---
    from swimmer_fields import FlowField, ChemField
    flow = None
    chem = None
    for fname, f in sc.fields.items():
        kind = f.get("kind", fname)
        res = int(f.get("res", 160))
        if kind == "flow":
            flow = FlowField(fname, f.get("couples_to", "organism"), res=res, width=width, device=device)
            H.add_field(flow)
        else:
            chem = ChemField(fname, f.get("couples_to", "surface_node"), res=res, width=width,
                             diffusion=float(f.get("diffusion", 0.08)),
                             peclet=float(f.get("peclet", 0.0)),
                             c_inf=float(f.get("c_inf", 1.0)),
                             substeps=int(f.get("substeps", 40)),
                             symmetric=bool(f.get("symmetric", True)), device=device)
            chem.radius = R
            H.add_field(chem)
    if chem is not None:
        chem.flow = flow
        chem.organism = org
    H.uptake = 0.0
    H.rng = torch.Generator(device=device).manual_seed(sc.seed)
    return H


def _mask(H, sel):
    lvl = H.level(sel.set)
    if sel.attr is None:
        return torch.ones(lvl.n, dtype=torch.bool, device=lvl.state.device)
    if sel.attr == "type":
        return lvl.node_type == lvl.type_names.index(sel.val)
    raise ValueError(f"unknown selector attribute {sel.attr!r}")


def run(sc, device="cpu"):
    H = build(sc, device)
    inst = [(o.op, get_operator(o.op)({**o.params, "to": o.to, "from": o.frm}, device), o.on)
            for o in sc.operators]
    org, sn = H.level("organism"), H.level("surface_node")
    chem_name = next((n for n, f in sc.fields.items() if f.get("kind", n) != "flow"), None)
    re = sc.record_every
    n_rec = sc.n_frames // re + 1
    org_pos = np.zeros((n_rec, org.n, 2), np.float32)
    sn_pos = np.zeros((n_rec, sn.n, 2), np.float32)
    chem_hist = np.zeros((n_rec, H.fields[chem_name].nx, H.fields[chem_name].ny), np.float32) if chem_name else None
    uptake_t = np.zeros(n_rec, np.float32)
    has_tr = "tracer" in H.levels
    tr = H.level("tracer") if has_tr else None
    tr_pos = np.zeros((n_rec, tr.n, 2), np.float32) if has_tr else None
    tr_occ = np.zeros((n_rec, tr.n), np.float32) if has_tr else None
    captured_t = np.zeros(n_rec, np.int32)
    dt = sc.dt

    rec = 0
    for frame in range(sc.n_frames + 1):
        for step in sc.schedule:
            for tok in (step if isinstance(step, list) else [step]):
                if tok == "aggregate":
                    continue                                     # organism is primary here
                elif tok.endswith(".diffuse"):
                    H.fields[tok[: -len(".diffuse")]].step()
                else:
                    for nm, ob, sel in inst:
                        if nm != tok:
                            continue
                        out = ob(H, _mask(H, sel))
                        for lname, d in out.items():
                            lvl = H.level(lname)
                            lvl.state[:, :2] = lvl.state[:, :2] + dt * d   # first-order integrate
        if frame % re == 0:
            org_pos[rec] = org.state[:, :2].cpu().numpy()
            sn_pos[rec] = sn.state[:, :2].cpu().numpy()
            if chem_name:
                chem_hist[rec] = H.fields[chem_name].grid.cpu().numpy()
            uptake_t[rec] = getattr(H, "uptake", 0.0)
            if has_tr:
                tr_pos[rec] = tr.state[:, :2].cpu().numpy()
                tr_occ[rec] = tr.occ.cpu().numpy()
            captured_t[rec] = getattr(H, "captured", 0)
            rec += 1

    flow_name = next((n for n, f in sc.fields.items() if f.get("kind", n) == "flow"), None)
    flow_grid = H.fields[flow_name].grid.cpu().numpy() if flow_name else None
    return dict(
        name=sc.name, org_pos=org_pos[:rec], sn_pos=sn_pos[:rec],
        sn_type=sn.node_type.cpu().numpy(), org_type=org.node_type.cpu().numpy(),
        type_names=org.type_names, sn_type_names=sn.type_names,
        chem=chem_hist[:rec] if chem_name else None, flow=flow_grid,
        radius=float(org.radius[0]), world=H.world_width,
        uptake=float(getattr(H, "uptake", 0.0)), uptake_t=uptake_t[:rec],
        slip=sn.slip.cpu().numpy(), axis=float(org.axis[0]),
        tr_pos=tr_pos[:rec] if has_tr else None, tr_occ=tr_occ[:rec] if has_tr else None,
        captured=int(getattr(H, "captured", 0)), captured_t=captured_t[:rec],
    )
