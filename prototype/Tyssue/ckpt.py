#!/usr/bin/env python
"""Mesh CHECKPOINT: genuinely initialise a run from a previous run's final state (e.g. smoke_hom's
homogenised vesicle) instead of a fresh sphere. save_state() dumps the vertex positions + the mesh dict
(topology + per-cell targets) + the cell-level state (chem/cen/area) to an npz; the load_mesh_3d operator
restores all of it at frame 0 (drop-in replacement for seed_mesh_3d). The tubing run then re-seeds fresh
RD spots on top (cell_rd_seed runs after), so we get: smoke_hom's uniform cells + big spots -> tubes."""
from __future__ import annotations
import numpy as np, torch
from plexus.models.registry import register_operator
from plexus.models.base import Structural

# scalar mesh-dict keys (ints / floats) + tensor keys we persist (skip hist/mech/twin/verts0 -- transient/rebuilt)
_INT = ("nF", "Nv"); _FLT = ("V0", "v_ref", "R0")
_ITEN = ("E_srce", "E_trgt", "E_face")                      # integer index tensors
_FTEN = ("A0", "P0", "alive", "divjit", "V0f", "Vbirth", "age")   # float per-cell tensors


def save_state(H, path):
    vlvl = H.level("vertex"); clvl = H.level("cell"); m = vlvl._mesh
    d = {}
    px0, px1 = vlvl.state_schema["pos"]
    d["vpos"] = vlvl.state[:, px0:px1].detach().cpu().numpy()
    d["vocc"] = (vlvl.occ.detach().cpu().numpy() if getattr(vlvl, "occ", None) is not None else np.ones(vlvl.state.shape[0]))
    d["cstate"] = clvl.state.detach().cpu().numpy()
    d["cocc"] = (clvl.occ.detach().cpu().numpy() if getattr(clvl, "occ", None) is not None else np.ones(clvl.state.shape[0]))
    for k in _INT + _FLT + _ITEN + _FTEN:
        if k in m:
            v = m[k]; d["m_" + k] = (v.detach().cpu().numpy() if torch.is_tensor(v) else np.array(v))
    np.savez(path, **d)
    print(f"[ckpt] saved {path}: {int(m['nF'])} cells, {int(m['Nv'])} verts", flush=True)


@register_operator("load_mesh_3d", set="vertex", kind="structural", family="mechanics")
class LoadMesh3D(Structural):
    """Frame-0: restore a saved mesh+cell state (drop-in for seed_mesh_3d). buffer must be >= saved size;
    reservoir sizes (Nv_max/nF_max/Ebuf) are recomputed to the NEW (larger) buffer so growth has headroom."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["initial_condition", "checkpoint", "half_edge_mesh"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self.cat = params.get("cell_set", "cell")
        self.path = params["ckpt"]

    def forward(self, H, mask=None):
        d = np.load(self.path, allow_pickle=True)
        vlvl = H.level(self.at); dev = vlvl.state.device; dt = vlvl.state.dtype; Nbuf = vlvl.state.shape[0]
        px0, px1 = vlvl.state_schema["pos"]
        vpos = torch.as_tensor(d["vpos"], dtype=dt, device=dev)
        st = vlvl.state.clone(); n = min(vpos.shape[0], Nbuf); st[:n, px0:px1] = vpos[:n]; vlvl.state = st
        if getattr(vlvl, "occ", None) is not None:
            occ = torch.zeros(Nbuf, device=dev); vo = torch.as_tensor(d["vocc"], device=dev)
            k = min(len(vo), Nbuf); occ[:k] = vo[:k]; vlvl.occ = occ
        m = {}
        for k in _INT:
            if "m_" + k in d: m[k] = int(d["m_" + k])
        for k in _FLT:
            if "m_" + k in d: m[k] = float(d["m_" + k])
        for k in _ITEN:
            if "m_" + k in d: m[k] = torch.as_tensor(d["m_" + k], device=dev)
        for k in _FTEN:
            if "m_" + k in d: m[k] = torch.as_tensor(d["m_" + k], dtype=dt, device=dev)
        m["Nv_max"] = Nbuf; m["nF_max"] = Nbuf // 2 + 64; m["Ebuf"] = 4 * Nbuf   # headroom in the NEW buffer
        m["verts0"] = d["vpos"][:m["Nv"]]
        vlvl._mesh = m
        clvl = H.level(self.cat); cdev = clvl.state.device; Cbuf = clvl.state.shape[0]
        cst = torch.as_tensor(d["cstate"], dtype=clvl.state.dtype, device=cdev)
        cs = clvl.state.clone(); n = min(cst.shape[0], Cbuf); cs[:n] = cst[:n]; clvl.state = cs
        if getattr(clvl, "occ", None) is not None:
            co = torch.zeros(Cbuf, device=cdev); cv = torch.as_tensor(d["cocc"], device=cdev)
            k = min(len(cv), Cbuf); co[:k] = cv[:k]; clvl.occ = co
        print(f"[ckpt] loaded {self.path}: {m['nF']} cells into buffer {Cbuf}", flush=True)
        return {}
