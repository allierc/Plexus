#!/usr/bin/env python
"""Mesh CHECKPOINT: genuinely initialise a run from a previous run's final state (e.g. smoke_hom's
homogenised vesicle) instead of a fresh sphere. save_state() dumps the vertex positions + the mesh dict
(topology + per-cell targets) + the cell-level state (chem/cen/area) to an npz; the load_mesh_3d operator
restores all of it at frame 0 (drop-in replacement for mesh_seed). The tubing run then re-seeds fresh
RD spots on top (cell_chem_seed runs after), so we get: smoke_hom's uniform cells + big spots -> tubes."""
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


