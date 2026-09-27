#!/usr/bin/env python
"""Cut a channel's deposited atomic model into alpha-carbon bead clouds, one per domain, in the
channel's own frame -- the anatomy of experiments/exp04_membrane_channels.

    PYTHONPATH=src python tools/channel_anatomy.py <model.cif> --name mscs \
        --parts "paddle:1-91,pore:92-113,cage:114-300" --membrane 27-113 [--flip] [--chains A,B,...]

WHAT IT WRITES, under <graphs_data>/shapes/<name>/:
    points.npz     one array per part, the alpha carbons in METRES, the channel axis along z
                   through (0, 0), the membrane's mid-plane at z = 0 -- the frame `cloud_seed`
                   reads (`cloud: <name>/<part>`, `scale` world per metre, `origin` where (0,0,0)
                   lands in the box)
    blocks.npz     per part: `<part>_chain` (0..n-1, numbered IN ORDER AROUND THE AXIS so that
                   chain k touches k-1 and k+1), `<part>_resid`, for `seed_state_from_file`
    anatomy.json   counts, radii, heights, the pore profile, the transform -- what the generator
                   and the ruler quote
    PROVENANCE.md  the file, the entry, the selection, the frame

THE FRAME, DERIVED. The axis is the normal of the plane through the chains' centroids (a C_n
ring's centroids lie on a circle about the axis); the centre is their mean. `--flip` turns it so
the cytoplasmic side is -z (read it off the printed per-part heights and the paper's figure).
The membrane's mid-plane is the mean height of the alpha carbons in `--membrane` (a residue range
spanning the transmembrane helices, from the paper), so z = 0 is the bilayer's centre.

THE PORE, AS HOLE WOULD SEE IT FROM THE ALPHA CARBONS: at every height the free radius is the
smallest distance of an alpha carbon from the axis, minus `--reach` (the van der Waals reach of a
residue beyond its alpha carbon, 0.5 nm default); the constriction is the minimum over the
heights the membrane spans. It is printed and stored so the ruler has the deposited value to be
wrong against.

Reuses the mmCIF tokeniser and symmetry-operator reader of tools/bfm_anatomy_from_structure.py.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))
from bfm_anatomy_from_structure import _cif_tokens, _operators    # noqa: E402


NOT_RESIDUES = {"HOH", "WAT", "DOD", "CA", "K", "NA", "CL", "MG", "ZN", "TL", "RB", "CS", "SO4", "PO4", "ACT", "GOL", "EDO"}


def read_atoms(path, assembly=True, hetatm=False):
    """Every ATOM record of model 1 as a dict of arrays: xyz (A), atom, resname, resid (label
    seq), chain (auth asym, else label), entity, element. With `assembly`, expanded by the entry's own
    assembly operators when there is more than the identity. With `hetatm`, HETATM records too --
    gramicidin's D-leucines and D-valines (DLE, DVA) and its formyl-valine are HETATM in 1MAG, so an
    ATOM-only read keeps 8 of its 15 residues -- except waters, ions and small ligands."""
    if path.lower().endswith((".pdb", ".ent", ".pdb.gz")):
        return _read_pdb(path, hetatm)
    cols, rows, in_loop, pend = [], [], False, []
    op_cols, op_rows, in_op, op_pend = [], [], False, []
    gen_expr = None
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("_atom_site."):
                cols.append(s.split(".", 1)[1].split()[0]); in_loop = True; continue
            if in_loop:
                if s.startswith("ATOM") or s.startswith("HETATM"):
                    pend = _cif_tokens(s)
                    if len(pend) >= len(cols):
                        rows.append(pend[:len(cols)])
                    continue
                if rows and (s.startswith("#") or s.startswith("loop_") or s.startswith("_")):
                    in_loop = False
                continue
            if s.startswith("_pdbx_struct_assembly_gen.oper_expression"):
                parts = s.split(None, 1)
                gen_expr = parts[1].strip("'\"") if len(parts) > 1 else gen_expr
                continue
            if s.startswith("_pdbx_struct_oper_list."):
                op_cols.append(s.split(".", 1)[1].split()[0]); in_op = True; continue
            if in_op:
                if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                    in_op = False
                    continue
                op_pend += _cif_tokens(s)
                while len(op_pend) >= len(op_cols):
                    op_rows.append(op_pend[:len(op_cols)]); op_pend = op_pend[len(op_cols):]
    if not rows:
        raise SystemExit(f"{path}: no _atom_site rows")
    ix = {c: i for i, c in enumerate(cols)}

    def col(*names):
        for n in names:
            if n in ix:
                return ix[n]
        return None
    cg, cx, cy, cz = col("group_PDB"), col("Cartn_x"), col("Cartn_y"), col("Cartn_z")
    ca, cr, cs = col("label_atom_id", "auth_atom_id"), col("label_comp_id", "auth_comp_id"), col("auth_seq_id", "label_seq_id")
    cc, ce, cm = col("auth_asym_id", "label_asym_id"), col("label_entity_id"), col("pdbx_PDB_model_num")
    ct = col("type_symbol")
    cb = col("B_iso_or_equiv")
    keep = [r for r in rows if (r[cg] == "ATOM" or (hetatm and r[cg] == "HETATM" and r[cr] not in NOT_RESIDUES))
            and (cm is None or r[cm] == rows[0][cm]) and r[cs] not in (".", "?")]
    out = {"xyz": np.array([[float(r[cx]), float(r[cy]), float(r[cz])] for r in keep]),
           "atom": np.array([r[ca] for r in keep]), "resname": np.array([r[cr] for r in keep]),
           "resid": np.array([int(r[cs]) for r in keep]), "chain": np.array([r[cc] for r in keep]),
           "entity": np.array([r[ce] if ce is not None else "1" for r in keep]),
           "element": np.array([r[ct] if ct is not None else r[ca][:1] for r in keep]),
           "B": np.array([float(r[cb]) if cb is not None and r[cb] not in (".", "?") else 30.0 for r in keep])}
    ops = _operators(op_cols, op_rows, gen_expr) if assembly else []
    if len(ops) > 1:
        out = {k: (np.concatenate([v @ R.T + t for R, t in ops]) if k == "xyz" else
                   np.concatenate([np.char.add(v, f"@{i}") for i in range(len(ops))]) if k == "chain" else
                   np.concatenate([v] * len(ops))) for k, v in out.items()}
        print(f"  assembly: {len(ops)} operators applied ({gen_expr})")
    return out


ION_NAMES = {"K", "NA", "RB", "TL", "CS"}


def read_atoms_ions(path, keepc, assembly):
    """The entry's monovalent cations (HETATM K, NA, RB, TL, CS), A, expanded like the chains when `assembly`."""
    if path.lower().endswith((".pdb", ".ent", ".pdb.gz")):
        X = []
        for line in open(path, errors="replace"):
            if line.startswith("ENDMDL"):
                break
            if line[:6].strip() == "HETATM" and line[17:20].strip() in ION_NAMES:
                X.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        return np.array(X) if X else None
    cols, rows, in_loop = [], [], False
    op_cols, op_rows, in_op, op_pend, gen_expr = [], [], False, [], None
    with (gzip.open if path.endswith(".gz") else open)(path, "rt", errors="replace") as f:
        for line in f:
            st = line.strip()
            if st.startswith("_atom_site."):
                cols.append(st.split(".", 1)[1].split()[0]); in_loop = True; continue
            if in_loop:
                if st.startswith("HETATM"):
                    t = _cif_tokens(st)
                    if len(t) >= len(cols):
                        rows.append(t[:len(cols)])
                    continue
                if st.startswith("ATOM"):
                    continue
                if rows and (st.startswith("#") or st.startswith("loop_") or st.startswith("_")):
                    in_loop = False
                continue
            if st.startswith("_pdbx_struct_assembly_gen.oper_expression"):
                parts = st.split(None, 1)
                gen_expr = parts[1].strip("'\"") if len(parts) > 1 else gen_expr
                continue
            if st.startswith("_pdbx_struct_oper_list."):
                op_cols.append(st.split(".", 1)[1].split()[0]); in_op = True; continue
            if in_op:
                if not st or st.startswith("#") or st.startswith("loop_") or st.startswith("_"):
                    in_op = False
                    continue
                op_pend += _cif_tokens(st)
                while len(op_pend) >= len(op_cols):
                    op_rows.append(op_pend[:len(op_cols)]); op_pend = op_pend[len(op_cols):]
    if not rows:
        return None
    ix = {c: i for i, c in enumerate(cols)}
    cr = ix.get("label_comp_id", ix.get("auth_comp_id"))
    X = np.array([[float(r[ix["Cartn_x"]]), float(r[ix["Cartn_y"]]), float(r[ix["Cartn_z"]])] for r in rows if r[cr] in ION_NAMES])
    if not len(X):
        return None
    ops = _operators(op_cols, op_rows, gen_expr) if assembly else []
    if len(ops) > 1:
        X = np.concatenate([X @ R.T + t for R, t in ops])
        X = np.unique(np.round(X, 2), axis=0)                         # ions on the symmetry axis coincide
    return X


def _read_pdb(path, hetatm=False):
    """A PDB-format model (Wang et al. 2014's open MscL, their source code 2): model 1, the SEGMENT id
    as the chain when there is one -- that model labels all five subunits chain P and tells them apart
    only by segid P1-P5."""
    op_ = gzip.open if path.endswith(".gz") else open
    rows = []
    with op_(path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            rec = line[:6].strip()
            if rec == "ATOM" or (hetatm and rec == "HETATM" and line[17:20].strip() not in NOT_RESIDUES):
                el = line[76:78].strip() or line[12:16].strip()[:1]
                try:
                    bf = float(line[60:66])
                except ValueError:
                    bf = 30.0
                rows.append(([float(line[30:38]), float(line[38:46]), float(line[46:54])], line[12:16].strip(),
                             line[17:20].strip(), int(line[22:26]), line[72:76].strip() or line[21].strip() or "A", el, bf))
    return {"xyz": np.array([r[0] for r in rows]), "atom": np.array([r[1] for r in rows]),
            "resname": np.array([r[2] for r in rows]), "resid": np.array([r[3] for r in rows]),
            "chain": np.array([r[4] for r in rows]), "entity": np.array(["1"] * len(rows)),
            "element": np.array([r[5] for r in rows]), "B": np.array([r[6] for r in rows])}


def parse_parts(s):
    parts = {}
    for tok in s.split(","):
        name, rng = tok.split(":")
        lo, hi = rng.split("-")
        parts.setdefault(name.strip(), []).append((int(lo), int(hi)))
    return parts


def in_ranges(resid, ranges):
    m = np.zeros(resid.shape, bool)
    for lo, hi in ranges:
        m |= (resid >= lo) & (resid <= hi)
    return m


WW_OCT = {"ALA": 0.50, "ARG": 1.81, "ASN": 0.85, "ASP": 3.64, "CYS": -0.02, "GLN": 0.77, "GLU": 3.63,
          "GLY": 1.15, "HIS": 2.33, "ILE": -1.12, "LEU": -1.25, "LYS": 2.80, "MET": -0.67, "PHE": -1.71,
          "PRO": 0.14, "SER": 0.46, "THR": 0.25, "TRP": -2.09, "TYR": -0.71, "VAL": -0.46}
D_AMINO = {"DAL": "ALA", "DLE": "LEU", "DVA": "VAL", "DIL": "ILE", "DPN": "PHE", "DTR": "TRP", "DTY": "TYR",
           "DSN": "SER", "DTH": "THR", "DGL": "GLU", "DAS": "ASP", "DLY": "LYS", "DAR": "ARG", "DHI": "HIS",
           "DGN": "GLN", "DSG": "ASN", "DCY": "CYS", "DPR": "PRO", "MED": "MET", "FVA": "VAL"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "CYS"}
VDW_NM = {"C": 0.170, "N": 0.155, "O": 0.152, "S": 0.180, "P": 0.180, "SE": 0.190}     # Bondi (1964)


def opm_fit(path, A, sel, to_frame):
    """Where OPM puts this protein's membrane, in this frame: OPM's file has the membrane normal on z and
    the mid-plane at z = 0 (half-thickness in its REMARK). Its chains are relabelled and may include
    antibodies, so residues are matched by NUMBER AND NAME, averaged over the chains that agree with this
    entry's sequence; z_here = s z_opm + c is fitted with s = +-1, and c is the mid-plane here."""
    half, rows, dum = None, [], {"N": [], "O": []}
    for line in open(path, errors="replace"):
        if "1/2 of bilayer thickness" in line and half is None:
            half = float(line.split(":")[1])
        if line[17:20].strip() == "DUM":
            dum.setdefault(line[12:16].strip(), []).append(float(line[46:54]))
        elif line[:6].strip() in ("ATOM", "HETATM") and line[12:16].strip() == "CA":
            rows.append((line[21], int(line[22:26]), line[17:20].strip(), float(line[46:54])))
    # THE MEMBRANE IS WHERE ITS DUMMY ATOMS ARE, not always at z = 0: OPM draws the two boundary planes as
    # DUM atoms (N one side, O the other), and for MscS (6PWN, 2VV5) they sit at -58.6 and -96.0 A -- a
    # mid-plane 77 A off the origin the first reading assumed.
    if dum.get("N") and dum.get("O"):
        z_mid = 0.5 * (float(np.median(dum["N"])) + float(np.median(dum["O"])))
        half = 0.5 * abs(float(np.median(dum["O"])) - float(np.median(dum["N"])))
    else:
        z_mid = 0.0
    ref = {}
    for rn, r in zip(A["resname"][sel], A["resid"][sel]):
        ref.setdefault(int(r), str(rn))
    zh = {}
    Z = to_frame(A["xyz"][sel])[:, 2]
    for r, z in zip(A["resid"][sel], Z):
        zh.setdefault(int(r), []).append(float(z))
    chains = {}
    for c, r, rn, z in rows:
        chains.setdefault(c, []).append((r, rn, z))
    zo = {}
    for c, lst in chains.items():
        both = [(r, rn, z) for r, rn, z in lst if r in ref]
        if len(both) < 10 or sum(rn == ref[r] for r, rn, _ in both) < 0.8 * len(both):
            continue
        for r, rn, z in both:
            zo.setdefault(r, []).append(z)
    common = sorted(set(zo) & set(zh))
    if len(common) < 10 or half is None:
        return None
    zo_ = np.array([np.mean(zo[r]) for r in common]); zh_ = np.array([np.mean(zh[r]) for r in common])
    best = None
    for sg in (1.0, -1.0):
        c = float(np.mean(zh_ - sg * zo_))
        rms = float(np.sqrt(np.mean((zh_ - sg * zo_ - c) ** 2)))
        if best is None or rms < best["rms_A"]:
            best = {"z_A": c + sg * z_mid, "sign": sg, "rms_A": rms, "half_A": half, "n": len(common)}
    return best


def hole_profiles(A, keepc, to_frame, zmem, order, cindex, mode, ww, dz=0.1, zspan=4.0):
    """THE ION PATHS, as HOLE finds them (Smart et al. 1996, J Mol Graph 14:354), from EVERY heavy atom
    of the kept chains: at each height the largest sphere that fits between the atoms' van der Waals
    surfaces, its centre searched within 0.3 nm of the path's axis. Per height: [z nm, radius nm,
    mean Wimley-White value of the residues lining it (kcal/mol, < 0 hydrophobic), enclosed 0/1] --
    ENCLOSED when the atoms within radius + 1 nm cover 6 of 8 sectors round the centre: a height with no
    wall around the path is membrane or bath, not pore. One path on the axis (`axis`), or one inside
    each chain (`chains`, a porin's barrels: its axis the point of the chain's slab-footprint farthest
    from the chain's atoms)."""
    m = np.isin(A["chain"], list(keepc)) & (A["element"] != "H") & (A["element"] != "D")
    X = to_frame(A["xyz"][m]) / 10.0
    X[:, 2] -= zmem / 10.0
    vdw = np.array([VDW_NM.get(str(e).upper(), 0.17) for e in A["element"][m]])
    # THE WALL'S CHEMISTRY BY ITS ATOMS, not its residues: a K+ filter and gramicidin's lumen are lined by
    # backbone carbonyl OXYGENS whatever the side chains are (the first pass, by residue, called both
    # hydrophobic). Polar = N or O; the column is the polar fraction of the lining atoms.
    wres = np.array([1.0 if str(e).upper() in ("N", "O") else 0.0 for e in A["element"][m]])
    ch = A["chain"][m]
    if mode == "axis":
        axes = [np.zeros(2)]
    else:
        axes = []
        for c in order:
            Y = X[(ch == c) & (np.abs(X[:, 2]) < 1.0)]
            lo, hi = Y[:, :2].min(0), Y[:, :2].max(0)
            best, bxy = -1.0, None
            for gx in np.arange(lo[0], hi[0], 0.1):
                for gy in np.arange(lo[1], hi[1], 0.1):
                    dxy = Y[:, :2] - [gx, gy]
                    rr = np.hypot(dxy[:, 0], dxy[:, 1])
                    d = rr.min()
                    # INSIDE the barrel: the chain's own atoms within d + 1.5 nm surround the point
                    ring = rr < d + 1.5
                    ang = np.arctan2(dxy[ring, 1], dxy[ring, 0])
                    if len(set(((ang + np.pi) / (2 * np.pi) * 8).astype(int) % 8)) < 7:
                        continue
                    if d > best:
                        best, bxy = d, np.array([gx, gy])
            axes.append(bxy if bxy is not None else Y[:, :2].mean(0))
    out = []
    offs = [np.array([ox, oy]) for ox in np.arange(-0.3, 0.301, 0.1) for oy in np.arange(-0.3, 0.301, 0.1)
            if ox * ox + oy * oy <= 0.0901]
    for ax in axes:
        prof = []
        for z in np.arange(-zspan, zspan + 1e-9, dz):
            near = np.abs(X[:, 2] - z) < 0.8
            if not near.any():
                prof.append([round(float(z), 2), 0.0, 0.0, 0, round(float(ax[0]), 3), round(float(ax[1]), 3)]); continue
            Xn, vn, wn = X[near], vdw[near], wres[near]
            best, bc = -1.0, ax
            for o in offs:
                c3 = np.array([ax[0] + o[0], ax[1] + o[1], z])
                R = float((np.linalg.norm(Xn - c3, axis=1) - vn).min())
                if R > best:
                    best, bc = R, c3
            # ON THE AXIS unless leaving it widens the path by more than 0.02 nm: the 0.1 nm search grid made
            # gramicidin's centre zig-zag 0.1-0.2 nm between heights, which a 0.03 nm single file cannot follow
            c0 = np.array([ax[0], ax[1], z])
            R0 = float((np.linalg.norm(Xn - c0, axis=1) - vn).min())
            if R0 >= best - 0.02:
                best, bc = R0, c0
            dxy = Xn[:, :2] - bc[:2]
            rr = np.hypot(dxy[:, 0], dxy[:, 1])
            # the wall is looked for out to R + max(1 nm, R): an OPEN MscS or MscL pore 1.2-1.7 nm wide has its helices
            # further apart than 1 nm past its radius, and read as "not enclosed" -- membrane -- at those heights
            ring = (np.abs(Xn[:, 2] - z) < 0.3) & (rr < best + max(1.0, best))
            ang = np.arctan2(dxy[ring, 1], dxy[ring, 0])
            enclosed = len(set(((ang + np.pi) / (2 * np.pi) * 8).astype(int) % 8)) >= 6
            lining = (np.linalg.norm(Xn - bc, axis=1) - vn) < best + 0.3
            prof.append([round(float(z), 2), round(max(best, 0.0), 3), round(float(wn[lining].mean()), 2) if lining.any() else 0.0,
                         int(enclosed), round(float(bc[0]), 3), round(float(bc[1]), 3)])
        # A CENTRE LINE AN ION CAN FOLLOW: the per-height centres smoothed along z (a running mean over +-0.25 nm) and
        # the radius re-measured about the smoothed line -- jumps of 0.1-0.2 nm between neighbouring heights
        # (gramicidin's helix ends) are walls for a 0.03 nm single file
        P_ = np.array(prof, float)
        if len(P_) > 5:
            w_ = max(1, int(round(0.25 / dz)))
            ker_ = np.ones(2 * w_ + 1) / (2 * w_ + 1)
            for col in (4, 5):
                pad_ = np.concatenate([np.full(w_, P_[0, col]), P_[:, col], np.full(w_, P_[-1, col])])
                P_[:, col] = np.convolve(pad_, ker_, mode="valid")
            for i_, row in enumerate(P_):
                z_ = row[0]
                near = np.abs(X[:, 2] - z_) < 0.8
                if not near.any() or row[1] <= 0:
                    continue
                c3 = np.array([row[4], row[5], z_])
                P_[i_, 1] = max(float((np.linalg.norm(X[near] - c3, axis=1) - vdw[near]).min()), 0.0)
            prof = [[round(float(r[0]), 2), round(float(r[1]), 3), float(r[2]), int(r[3]), round(float(r[4]), 3),
                     round(float(r[5]), 3)] for r in P_]
        out.append({"axis_xy_nm": [round(float(ax[0]), 3), round(float(ax[1]), 3)], "profile": prof})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--name", required=True)
    ap.add_argument("--parts", required=True, help="name:lo-hi[,name:lo-hi...] (author residue numbers)")
    ap.add_argument("--membrane", required=True, help="lo-hi: the residues whose mean height is the bilayer centre")
    ap.add_argument("--chains", default=None, help="comma list of chains to keep (default: every chain of the largest entity)")
    ap.add_argument("--flip", action="store_true", help="turn the frame so the other side is +z")
    ap.add_argument("--reach", type=float, default=0.5, help="nm a residue reaches beyond its alpha carbon")
    ap.add_argument("--extra-atoms", default=None, help="part:ATOM:lo-hi, e.g. filter_O:O:374-378 (backbone O of the filter)")
    # SYMMETRY EXPANSION IS OPT-IN. A cryo-EM channel is deposited whole; a crystal entry's assembly
    # operators may describe the lattice instead -- 2OAR's two operators turned its pentamer into ten
    # chains (2026-09-25). Expand only an entry deposited as one asymmetric protomer.
    ap.add_argument("--assembly", action="store_true", help="apply the entry's assembly operators")
    ap.add_argument("--membrane-offset", type=float, default=0.0,
                    help="A added to the --membrane residues' mean height: for a protein that sits ON the membrane "
                         "(a toxin's rim at the headgroups), the bilayer's centre lies below the residues that touch it")
    ap.add_argument("--hetatm", action="store_true", help="read HETATM residues too (D-amino acids, modified residues)")
    ap.add_argument("--axis", default="auto", choices=["auto", "ring", "pca", "c2"],
                    help="the channel axis: ring = normal of the chains' centroid plane (3+ chains), pca = the "
                         "longest direction of the alpha carbons, c2 = the rotation axis taking chain 1 onto chain 2 (a dimer)")
    ap.add_argument("--pores", default="axis", choices=["axis", "chains"],
                    help="where the ion paths are: one on the axis, or one inside each chain (a porin's barrels)")
    ap.add_argument("--core", type=float, default=1.5, help="nm, half the hydrocarbon core the belt is fitted with")
    ap.add_argument("--up", default=None, help="lo-hi[,lo-hi]: residues that must end up at +z (the outside); "
                                                "the frame is flipped if they do not (overrides --flip)")
    ap.add_argument("--ligands", action="store_true",
                    help="count the entry's non-protein molecules (lipids, alkanes; not water or ions) as wall of the "
                         "ion path -- where the paper names them as the block: TRAAK's cavity decane (4WFF), MscS's "
                         "acyl chains above the L105 cuff (6PWN)")
    ap.add_argument("--move-chain", default=None,
                    help="CHAIN:dx,dy,dz (nm, in the channel's frame): displace one chain after the membrane is placed "
                         "-- gramicidin's dissociated monomers, the closed state of a dimer channel")
    ap.add_argument("--opm", default=None, help="an OPM-oriented PDB (Lomize et al. 2006): its membrane mid-plane and "
                                                "thickness, matched to this entry residue by residue; used as the "
                                                "mid-plane when --membrane opm")
    a = ap.parse_args()

    A = read_atoms(a.model, assembly=a.assembly, hetatm=a.hetatm or a.ligands)
    polymer = np.isin(A["resname"], list(WW_OCT) + list(D_AMINO) + ["MSE", "HSD", "HSE", "HIE", "HID"])
    ca = (A["atom"] == "CA") & polymer
    if a.chains:
        keepc = set(a.chains.split(","))
    else:
        ents, cnt = np.unique(A["entity"][ca], return_counts=True)
        big = ents[np.argmax(cnt)]
        keepc = set(np.unique(A["chain"][ca & (A["entity"] == big)]))
    sel = ca & np.isin(A["chain"], list(keepc))
    chains = sorted(set(A["chain"][sel]))
    print(f"{a.model}: {int(sel.sum()):,} alpha carbons on {len(chains)} chains {[str(c) for c in chains][:12]}; residues "
          f"{A['resid'][sel].min()}-{A['resid'][sel].max()}")

    # ---- the frame: axis = normal of the chains' centroid plane, centre = their mean ----------
    cents = np.array([A["xyz"][sel & (A["chain"] == c)].mean(0) for c in chains])
    centre = cents.mean(0)
    how = a.axis if a.axis != "auto" else ("ring" if len(chains) >= 3 else "pca")
    if how == "ring":
        _, _, vt = np.linalg.svd(cents - centre)
        axis = vt[-1]
    elif how == "c2":
        # THE DIMER'S TWO-FOLD AXIS: superpose chain 1 on chain 2 over their shared residues (Kabsch);
        # the rotation's axis is the channel's. A two-chain K2P channel (TRAAK) has no centroid ring,
        # and its longest direction is its extracellular cap, not the membrane normal.
        c1, c2 = chains[0], chains[1]
        r1 = A["resid"][sel & (A["chain"] == c1)]; r2 = A["resid"][sel & (A["chain"] == c2)]
        common = np.intersect1d(r1, r2)
        P1 = np.array([A["xyz"][sel & (A["chain"] == c1) & (A["resid"] == r)][0] for r in common])
        P2 = np.array([A["xyz"][sel & (A["chain"] == c2) & (A["resid"] == r)][0] for r in common])
        H_ = (P1 - P1.mean(0)).T @ (P2 - P2.mean(0))
        U_, _, Vt_ = np.linalg.svd(H_)
        d_ = np.sign(np.linalg.det(Vt_.T @ U_.T))
        Rk = Vt_.T @ np.diag([1.0, 1.0, d_]) @ U_.T
        w_, v_ = np.linalg.eig(Rk)
        axis = np.real(v_[:, int(np.argmin(np.abs(w_ - 1.0)))])
        axis /= np.linalg.norm(axis)
        centre = 0.5 * (P1.mean(0) + P2.mean(0))
    else:
        _, _, vt = np.linalg.svd(A["xyz"][sel] - A["xyz"][sel].mean(0))
        axis = vt[0]
    print(f"  axis by {how}")
    if a.up:
        up = parse_parts(",".join(f"u:{tok}" for tok in a.up.split(",")))["u"]
        mu_ = sel & in_ranges(A["resid"], up)
        if mu_.any() and float((A["xyz"][mu_] - centre).mean(0) @ axis) < 0:
            axis = -axis
        print(f"  oriented so residues {a.up} are at +z (the outside)")
    elif a.flip:
        axis = -axis
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ex = ref - ref.dot(axis) * axis; ex /= np.linalg.norm(ex)
    ey = np.cross(axis, ex)
    Rm = np.stack([ex, ey, axis])                                   # rows: the new x, y, z

    def to_frame(x):
        return (x - centre) @ Rm.T

    # THE HYDROPHOBIC BELT, from the protein's own chemistry: the height z_c that puts the most
    # hydrophobic residues (Wimley-White octanol scale, kcal/mol) inside a core of +-`core` nm and the
    # most polar outside -- OPM's idea (Lomize et al. 2006), crudely. Always computed and stored; it is
    # the mid-plane when `--membrane belt`, and the audit's check on a paper's residue range otherwise.
    ww = {**WW_OCT, **{d: WW_OCT[l] for d, l in D_AMINO.items() if l in WW_OCT}}
    zca = to_frame(A["xyz"][sel])[:, 2] / 10.0
    wca = np.array([ww.get(str(n), 0.0) for n in A["resname"][sel]])
    zc_grid = np.arange(zca.min(), zca.max(), 0.05)
    E_belt = np.array([wca[np.abs(zca - zc) < a.core].sum() for zc in zc_grid])
    z_belt = float(zc_grid[int(np.argmin(E_belt))]) * 10.0                          # A
    opm = opm_fit(a.opm, A, sel, to_frame) if a.opm else None
    if opm:
        print(f"  OPM ({os.path.basename(a.opm)}): mid-plane at {opm['z_A'] / 10:.2f} nm in this frame, hydrophobic "
              f"half-thickness {opm['half_A'] / 10:.2f} nm, {opm['n']} residues matched, rms {opm['rms_A'] / 10:.2f} nm, "
              f"normal {'along' if opm['sign'] > 0 else 'against'} this axis")
    if a.membrane == "opm":
        if not opm:
            raise SystemExit("--membrane opm needs --opm <file> with residues matching this entry")
        zmem = opm["z_A"] + a.membrane_offset
    elif a.membrane == "belt":
        zmem = z_belt + a.membrane_offset
    else:
        mem = parse_parts(",".join(f"m:{tok}" for tok in a.membrane.split(",")))["m"]     # one or several lo-hi ranges
        zmem = float(to_frame(A["xyz"][sel & in_ranges(A["resid"], mem)])[:, 2].mean()) + a.membrane_offset
    print(f"  hydrophobic belt centre at {z_belt / 10:.2f} nm; mid-plane used {zmem / 10:.2f} nm "
          f"(belt - used = {(z_belt - zmem) / 10:+.2f} nm)")
    if a.move_chain:
        cm_, v_ = a.move_chain.split(":")
        d_frame = np.array([float(v) for v in v_.split(",")]) * 10.0            # nm -> A, frame axes
        A["xyz"][A["chain"] == cm_] += Rm.T @ d_frame
        print(f"  chain {cm_} moved by {v_} nm in the channel's frame")
    print(f"  axis {np.round(axis, 4).tolist()} through {np.round(centre, 2).tolist()} A; membrane mid-plane "
          f"at {zmem:.2f} A along it (residues {a.membrane})")

    # chains numbered in order AROUND the axis
    ang = [float(np.arctan2(*to_frame(c[None])[0, [1, 0]])) for c in cents]
    order = [chains[i] for i in np.argsort(ang)]
    cindex = {c: k for k, c in enumerate(order)}

    parts = parse_parts(a.parts)
    pts, blocks, info = {}, {}, {"model": os.path.basename(a.model), "chains": order, "parts": {}}
    for name, ranges in parts.items():
        m = sel & in_ranges(A["resid"], ranges)
        X = to_frame(A["xyz"][m])
        X[:, 2] -= zmem
        pts[name] = (X * 1e-10).astype(np.float32)                 # angstrom -> metre
        blocks[f"{name}_chain"] = np.array([cindex[c] for c in A["chain"][m]], np.float32)
        blocks[f"{name}_resid"] = A["resid"][m].astype(np.float32)
        # 1 where the residue's side chain is hydrophobic (the moving ion path's wall chemistry, per bead)
        blocks[f"{name}_hyd"] = np.array([1.0 if D_AMINO.get(str(rn), str(rn)) in HYDROPHOBIC else 0.0
                                          for rn in A["resname"][m]], np.float32)
        # the side chain's transfer free energy water -> octanol (Wimley-White whole residue minus glycine), kT
        blocks[f"{name}_dg"] = np.array([(WW_OCT.get(D_AMINO.get(str(rn), str(rn)), 1.15) - 1.15) / 0.593
                                         for rn in A["resname"][m]], np.float32)
        rho = np.hypot(X[:, 0], X[:, 1]) / 10.0
        info["parts"][name] = {"n": int(m.sum()), "residues": ranges, "r_nm": [round(float(rho.min()), 2), round(float(rho.max()), 2)],
                               "z_nm": [round(float(X[:, 2].min()) / 10, 2), round(float(X[:, 2].max()) / 10, 2)]}
        print(f"  part {name:10s} {int(m.sum()):5d} CA  r {rho.min():5.2f}-{rho.max():5.2f} nm  "
              f"z {X[:, 2].min() / 10:6.2f} to {X[:, 2].max() / 10:6.2f} nm")
    # THE WHOLE PROTEIN AS ONE CLOUD, its parts kept as a `domain` index (in --parts order): one
    # set whose chains are held apart by the chain block and whose domains are only a colour.
    names = list(parts)
    pts["all"] = np.concatenate([pts[k] for k in names])
    blocks["all_chain"] = np.concatenate([blocks[f"{k}_chain"] for k in names])
    blocks["all_resid"] = np.concatenate([blocks[f"{k}_resid"] for k in names])
    blocks["all_domain"] = np.concatenate([np.full(len(pts[k]), i, np.float32) for i, k in enumerate(names)])
    blocks["all_hyd"] = np.concatenate([blocks[f"{k}_hyd"] for k in names])
    blocks["all_dg"] = np.concatenate([blocks[f"{k}_dg"] for k in names])
    info["domains"] = names
    info["n_all"] = int(len(pts["all"]))
    # ONE CLOUD PER CHAIN, `c0` ... `c<n-1>` in ring order: the renderer draws one skin per SET, so
    # a pore can only be seen to part if each chain is its own set -- and a set per chain is also
    # the per-protomer colour of a cryo-EM figure. Domains ride along as a block.
    for k in range(len(order)):
        m = blocks["all_chain"] == k
        pts[f"c{k}"] = pts["all"][m]
        blocks[f"c{k}_domain"] = blocks["all_domain"][m]
        blocks[f"c{k}_resid"] = blocks["all_resid"][m]
        blocks[f"c{k}_hyd"] = blocks["all_hyd"][m]
        blocks[f"c{k}_dg"] = blocks["all_dg"][m]
    info["per_chain"] = [int((blocks["all_chain"] == k).sum()) for k in range(len(order))]
    if a.extra_atoms:
        for tok in a.extra_atoms.split(","):
            name, atom, rng = tok.split(":")
            lo, hi = rng.split("-")
            m = np.isin(A["chain"], list(keepc)) & (A["atom"] == atom) & in_ranges(A["resid"], [(int(lo), int(hi))])
            X = to_frame(A["xyz"][m]); X[:, 2] -= zmem
            pts[name] = (X * 1e-10).astype(np.float32)
            blocks[f"{name}_chain"] = np.array([cindex[c] for c in A["chain"][m]], np.float32)
            blocks[f"{name}_resid"] = A["resid"][m].astype(np.float32)
            info["parts"][name] = {"n": int(m.sum()), "atom": atom, "residues": [[int(lo), int(hi)]],
                                   "z_nm": [round(float(X[:, 2].min()) / 10, 2), round(float(X[:, 2].max()) / 10, 2)]}
            print(f"  extra {name:10s} {int(m.sum()):5d} {atom}   z {X[:, 2].min() / 10:6.2f} to {X[:, 2].max() / 10:6.2f} nm")

    # ---- the pore, from the alpha carbons: free radius at each height -------------------------
    allX = pts["all"] * 1e9                                                   # nm
    rho = np.hypot(allX[:, 0], allX[:, 1]) - a.reach
    zs = np.arange(np.floor(allX[:, 2].min()), np.ceil(allX[:, 2].max()) + 0.01, 0.25)
    prof = []
    for z in zs:
        near = np.abs(allX[:, 2] - z) < 0.3
        prof.append(float(rho[near].min()) if near.any() else float("nan"))
    prof = np.array(prof)
    tm = (zs > -2.5) & (zs < 2.5) & np.isfinite(prof)
    k = int(np.nanargmin(np.where(tm, prof, np.nan)))
    info["pore_profile_nm"] = [[round(float(z), 2), round(float(r), 3)] for z, r in zip(zs, prof) if np.isfinite(r)]
    info["constriction_nm"] = {"radius": round(float(prof[k]), 3), "diameter": round(2 * float(prof[k]), 3), "z": round(float(zs[k]), 2),
                               "reach_nm": a.reach}
    info["frame"] = {"axis": axis.tolist(), "centre_A": centre.tolist(), "membrane_z_A": zmem, "flip": a.flip,
                     "belt_z_A": z_belt, "membrane_by": a.membrane}
    if opm:
        info["opm"] = {"file": os.path.basename(a.opm), "midplane_minus_used_nm": round((opm["z_A"] - zmem) / 10, 2),
                       "half_thickness_nm": round(opm["half_A"] / 10, 2), "matched": opm["n"],
                       "rms_nm": round(opm["rms_A"] / 10, 2)}
    info["belt_minus_used_nm"] = round((z_belt - zmem) / 10, 2)
    if a.ligands:
        lig = ~polymer
        print(f"  ligands counted as wall: {sorted(set(A['resname'][lig].tolist()))} ({int(lig.sum())} atoms)")
        keepc = set(keepc) | set(A["chain"][lig].tolist())
    info["pores"] = hole_profiles(A, keepc, to_frame, zmem, order, cindex, a.pores, ww)
    # THE CHARGES THE ION PATH SEES, from the atoms: a formal charge on each ionisable side chain's tip
    # (Asp CG and Glu CD -1, Lys NZ and Arg CZ +1; His neutral, pH 7), and the partial charges of the
    # backbone carbonyls that FACE a path inside the membrane (O -0.5, C +0.5; a K+ filter's and
    # gramicidin's lining) -- the chemistry selectivity comes out of, instead of being written in.
    qx, qq, qk, qb = [], [], [], []
    mm = np.isin(A["chain"], list(keepc))
    Xall = to_frame(A["xyz"][mm]) / 10.0
    Xall[:, 2] -= zmem / 10.0
    # one charge per ionisable RESIDUE: at its tip atom, or -- where the entry left the side chain unbuilt
    # (4NPQ's lysines and arginines, a modelled state with backbone only) -- at the farthest side-chain atom
    # it has, else its alpha carbon. Counting tip atoms alone gave GLIC closed 168 charges and open 325.
    res_ = {}
    for x_, rn, at, c_, r_, b_ in zip(Xall, A["resname"][mm], A["atom"][mm], A["chain"][mm], A["resid"][mm], A["B"][mm]):
        if str(rn) in ("ASP", "GLU", "LYS", "ARG"):
            res_.setdefault((str(c_), int(r_), str(rn)), {})[str(at)] = (x_, float(b_))
    for (c_, r_, rn), atoms in res_.items():
        tip = {"ASP": "CG", "GLU": "CD", "LYS": "NZ", "ARG": "CZ"}[rn]
        if tip in atoms:
            x_, b_ = atoms[tip]
        else:
            side = {k: v for k, v in atoms.items() if k not in ("N", "CA", "C", "O")}
            if side and "CA" in atoms:
                x_, b_ = max(side.values(), key=lambda v: float(np.linalg.norm(v[0] - atoms["CA"][0])))
            else:
                x_, b_ = atoms.get("CA", next(iter(atoms.values())))
        qx.append(x_); qq.append(-1.0 if rn in ("ASP", "GLU") else 1.0); qk.append(0.0); qb.append(b_)
    for po in info["pores"]:
        pprof = np.array([[r[0], r[1], r[4], r[5]] for r in po["profile"]])
        for x_, at, b_ in zip(Xall, A["atom"][mm], A["B"][mm]):
            if at not in ("O", "C") or abs(x_[2]) > 2.0:
                continue
            k_ = int(np.argmin(np.abs(pprof[:, 0] - x_[2])))
            if np.hypot(x_[0] - pprof[k_, 2], x_[1] - pprof[k_, 3]) < pprof[k_, 1] + 0.35:
                qx.append(x_); qq.append(-0.5 if at == "O" else 0.5); qk.append(1.0); qb.append(float(b_))
    # THE CRYSTAL'S OWN IONS ON THE PATH: K+ (or Na+, Rb+, Tl+ -- a K+ filter's surrogates) that the entry
    # resolved within 0.5 nm of an ion path inside |z| < 2.5 nm -- a filter's sites as the structure shows them
    # occupied. The rig starts its cations there (`xtal_ions`) instead of waiting for the bath to find them.
    try:
        B_ = read_atoms_ions(a.model, keepc, a.assembly)
    except Exception:                                                  # noqa: BLE001
        B_ = None
    if B_ is not None and len(B_):
        Xi = to_frame(B_) / 10.0                                        # A -> nm, in the frame
        Xi[:, 2] -= zmem / 10.0
        keep_i = []
        for po in info["pores"]:
            pprof = np.array([[r[0], r[4], r[5]] for r in po["profile"]])
            for k_, x_ in enumerate(Xi):
                if abs(x_[2]) > 2.5:
                    continue
                j_ = int(np.argmin(np.abs(pprof[:, 0] - x_[2])))
                if np.hypot(x_[0] - pprof[j_, 1], x_[1] - pprof[j_, 2]) < 0.5:
                    keep_i.append(k_)
        keep_i = sorted(set(keep_i))
        # ONE ION PER SITE, AND NOT EVERY SITE: symmetry copies of an ion on the axis land within a fraction of an
        # angstrom of each other (merged within 0.1 nm); and a filter's four sites are the average of two
        # alternating fillings (S1+S3, S2+S4, ~half each), so the ions are taken top-down at least 0.6 nm apart --
        # KcsA: the outer site, S1, S3 and the cavity ion, not all eight positions the density shows
        chosen = []
        for k_ in sorted(keep_i, key=lambda k: -Xi[k, 2]):
            if all(np.linalg.norm(Xi[k_] - Xi[c_]) > 0.1 and abs(Xi[k_, 2] - Xi[c_, 2]) >= 0.6 for c_ in chosen):
                chosen.append(k_)
        keep_i = chosen
        if keep_i:
            pts["xtal_ions"] = (Xi[keep_i] * 1e-9).astype(np.float32)
            info["xtal_ions"] = {"n": len(keep_i), "z_nm": [round(float(Xi[k_, 2]), 2) for k_ in keep_i]}
            print(f"  crystal ions on the path: {len(keep_i)} at z {info['xtal_ions']['z_nm']} nm")
    if qx:
        pts["qsite"] = (np.array(qx) * 1e-9).astype(np.float32)
        blocks["qsite_q"] = np.array(qq, np.float32)
        blocks["qsite_backbone"] = np.array(qk, np.float32)
        blocks["qsite_B"] = np.array(qb, np.float32)                  # the atom's B-factor, A^2
        info["charges"] = {"tips": int(sum(1 for k in qk if k == 0)), "net_tips": float(sum(q for q, k in zip(qq, qk) if k == 0)),
                           "lumen_carbonyl_atoms": int(sum(1 for k in qk if k == 1))}
        print(f"  charges: {info['charges']['tips']} side-chain tips (net {info['charges']['net_tips']:+.0f} e), "
              f"{info['charges']['lumen_carbonyl_atoms']} carbonyl atoms facing the path in the membrane")
    for k_, po in enumerate(info["pores"]):
        core = [r for r in po["profile"] if abs(r[0]) <= 1.5]
        rmin = min(core, key=lambda r: r[1]) if core else None
        print(f"  ION PATH {k_} (all atoms, HOLE-like) at xy {po['axis_xy_nm']} nm: narrowest radius in the core "
              f"{rmin[1]:.3f} nm at z {rmin[0]:.2f} (lining {rmin[2]:.0%} polar atoms, "
              f"{'enclosed' if rmin[3] else 'NOT ENCLOSED'})" if rmin else "  no core")
    print(f"  PORE (alpha carbons, reach {a.reach} nm): narrowest free radius {prof[k]:.3f} nm (diameter "
          f"{2 * prof[k]:.3f} nm) at z {zs[k]:.2f} nm inside the membrane's +-2.5 nm")
    print("  profile z(nm):r(nm) " + " ".join(f"{z:.1f}:{r:.2f}" for z, r in zip(zs, prof) if np.isfinite(r) and abs(z) <= 6))

    from plexus.paths import graphs_data_path
    folder = os.path.join(graphs_data_path(), "shapes", a.name)
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, "points.npz"), **pts)
    np.savez_compressed(os.path.join(folder, "blocks.npz"), **blocks)
    json.dump(info, open(os.path.join(folder, "anatomy.json"), "w"), indent=1)
    with open(os.path.join(folder, "PROVENANCE.md"), "w") as f:
        f.write(f"# {a.name}\n\nFrom `{os.path.abspath(a.model)}` by `tools/channel_anatomy.py`:\n"
                f"parts `{a.parts}`, membrane mid-plane = mean height of residues {a.membrane}, "
                f"chains {order} numbered around the axis{', frame flipped' if a.flip else ''}.\n"
                f"points.npz: alpha carbons in metres, axis z, bilayer centre z = 0. blocks.npz: chain and residue per bead.\n")
    print(f"  wrote {folder}/points.npz, blocks.npz, anatomy.json")


if __name__ == "__main__":
    main()
