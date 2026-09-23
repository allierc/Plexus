#!/usr/bin/env python
"""Measure a flagellar-motor structure -- a cryo-EM map (.map/.mrc, gzipped or not) or an atomic
model (.cif/.pdb) -- into the ring dimensions bfm_motor_spec.py needs, and cut it into shape-library
parts that MPM can fill.

    python tools/bfm_anatomy_from_structure.py builder/exp_02_bacterium/data/emd_16723.map.gz
    python tools/bfm_anatomy_from_structure.py builder/exp_02_bacterium/data/emd_16723.map.gz \
        --level 0.05 --slabs "c_ring:-35:-5,ms_ring:-5:8,rod:8:40,hook:40:80" --parts bfm_cjejuni
    python tools/bfm_anatomy_from_structure.py builder/exp_02_bacterium/data/9HMF.cif [--parts ...]
    python tools/bfm_anatomy_from_structure.py --selftest

WHAT IT DOES. The motor is a stack of rings about one axis. For a MAP the axis is the direction of
least extent of the thresholded density (a ring stack is wide two ways and long one way is NOT
true for a whole motor, so for a map the axis is taken as the direction of GREATEST extent of the
density above `--level`, which for a motor with its hook is the rod's direction; `--axis x|y|z`
overrides it). The SILHOUETTE is then printed: at every height along the axis, in nanometres, the
radial extent of the density -- the r_in / r_out / z0 / z1 table that `PARTS` in
tools/bfm_motor_spec.py takes per part -- and `--slabs name:z0:z1,...` cuts the density into named
bands along the axis, each written as a closed isosurface into the shape library
(`<graphs_data>/shapes/<name>/parts.npz` + `parts.json` + `PROVENANCE.md`), the format
`plexus.shapes` reads, so a spec can say `shape: mesh:<name>/<part>` and fill the real shape with
material points instead of a cylinder.

For an ATOMIC MODEL each protein entity is a band [r_in, r_out] x [z0, z1] with its copy number
(chains) about the axis of least extent of all atoms (a ring), or `--axis`.

NO EXTERNAL PARSER. gemmi, Biopython and mrcfile are not in this environment; mmCIF's `_atom_site`
loop and the 1,024-byte MRC2014 header are read directly, and `--selftest` writes a synthetic C34
ring in both formats and measures it back.

Data this was written for, all to be downloaded by the human (instruction.md section 1):
  EMD-16723 / EMD-16724   C. jejuni whole motor in situ, 9.4 A, and its periplasmic scaffold
                          (Drobnic et al. 2025, Nat Microbiol 10:1723); PDB 9HMF, the scaffold's C-alpha model
  PDB 8T8O / 8T8P         Salmonella C-ring / MS-ring (Singh et al. 2024);  PDB 7CGO / EMD-30359 (Tan et al. 2021)
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


# ================================================================== readers
def read_mmcif_atoms(path, assembly=True):
    """(xyz [N,3] Angstrom, chain [N], entity [N], {entity: description}) from the `_atom_site` loop.

    With `assembly`, the deposited coordinates are EXPANDED by the entry's own symmetry operators
    (`_pdbx_struct_oper_list`, the ones `_pdbx_struct_assembly_gen` names): a ring deposited as one
    asymmetric unit with 17 rotations becomes the 17-fold ring, each copy's chains suffixed `@k`.
    A loop row in mmCIF may wrap over several lines, so rows are gathered by token count."""
    cols, rows, in_loop = [], [], False
    ent_cols, ent_rows, in_ent, desc = [], [], False, {}
    op_cols, op_rows, in_op, pend = [], [], False, []
    gen_expr = None
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("_atom_site."):
                cols.append(s.split(".", 1)[1].split()[0]); in_loop = True; continue
            if in_loop:
                if s.startswith("ATOM") or s.startswith("HETATM"):
                    rows.append(s.split()); continue
                if rows:
                    in_loop = False
                continue
            if s.startswith("_pdbx_struct_assembly_gen.oper_expression"):
                gen_expr = s.split(None, 1)[1].strip("'\"") if len(s.split(None, 1)) > 1 else gen_expr
                continue
            if s.startswith("_pdbx_struct_oper_list."):
                op_cols.append(s.split(".", 1)[1].split()[0]); in_op = True; continue
            if in_op:
                if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                    in_op = False
                    continue
                pend += _cif_tokens(s)
                while len(pend) >= len(op_cols):
                    op_rows.append(pend[:len(op_cols)]); pend = pend[len(op_cols):]
                continue
            if s.startswith("_entity.") and not s.startswith("_entity_"):
                ent_cols.append(s.split(".", 1)[1].split()[0]); in_ent = True; continue
            if in_ent:
                if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                    in_ent = False
                    continue
                ent_rows.append(_cif_tokens(s))
    if not rows:
        raise SystemExit(f"{path}: no _atom_site loop found")
    ix = {c: i for i, c in enumerate(cols)}
    def col(name, alt=None):
        k = name if name in ix else alt
        return None if k is None or k not in ix else ix[k]
    cx, cy, cz = col("Cartn_x"), col("Cartn_y"), col("Cartn_z")
    cch, cen = col("label_asym_id", "auth_asym_id"), col("label_entity_id")
    keep = [r for r in rows if len(r) > max(cx, cy, cz)]
    xyz = np.array([[float(r[cx]), float(r[cy]), float(r[cz])] for r in keep])
    chain = np.array([r[cch] if cch is not None else "A" for r in keep])
    ent = np.array([r[cen] if cen is not None else "1" for r in keep])
    if ent_cols and ent_rows and all(len(r) == len(ent_cols) for r in ent_rows):
        ei = {c: i for i, c in enumerate(ent_cols)}
        if "id" in ei and "pdbx_description" in ei:
            for r in ent_rows:
                desc[r[ei["id"]]] = r[ei["pdbx_description"]].strip("'\"")
    ops = _operators(op_cols, op_rows, gen_expr) if assembly else []
    if len(ops) > 1:
        X = [xyz @ R.T + t for R, t in ops]
        xyz = np.concatenate(X)
        chain = np.concatenate([np.char.add(chain, f"@{k}") for k in range(len(ops))])
        ent = np.concatenate([ent] * len(ops))
        print(f"  assembly: {len(ops)} symmetry operators applied ({gen_expr}); "
              f"axis of the rotations {np.round(_rotation_axis(ops), 3).tolist()}")
    return xyz, chain, ent, desc


def _cif_tokens(line):
    """Split a CIF data line into tokens, keeping quoted strings whole."""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c.isspace():
            i += 1; continue
        if c in "'\"":
            j = line.find(c, i + 1)
            j = n if j < 0 else j
            out.append(line[i + 1:j]); i = j + 1; continue
        j = i
        while j < n and not line[j].isspace():
            j += 1
        out.append(line[i:j]); i = j
    return out


def _operators(cols, rows, expr):
    """[(R [3,3], t [3])] for the operator ids `expr` names ('1,2,3' or '(1-17)'); identity if none."""
    if not cols or not rows:
        return []
    ix = {c: i for i, c in enumerate(cols)}
    want = None
    if expr:
        want = set()
        for tok in expr.replace("(", "").replace(")", "").split(","):
            tok = tok.strip()
            if "-" in tok:
                lo, hi = tok.split("-"); want.update(str(k) for k in range(int(lo), int(hi) + 1))
            elif tok:
                want.add(tok)
    ops = []
    for r in rows:
        if want is not None and r[ix["id"]] not in want:
            continue
        R = np.array([[float(r[ix[f"matrix[{i}][{j}]"]]) for j in (1, 2, 3)] for i in (1, 2, 3)])
        t = np.array([float(r[ix[f"vector[{i}]"]]) for i in (1, 2, 3)])
        ops.append((R, t))
    return ops


def _rotation_axis(ops):
    """The common axis of a set of rotations: the eigenvector of eigenvalue 1 of the first non-identity R."""
    for R, _ in ops:
        if np.abs(R - np.eye(3)).max() > 1e-6:
            w, v = np.linalg.eig(R)
            a = np.real(v[:, np.argmin(np.abs(w - 1.0))])
            return a / np.linalg.norm(a) * (1 if a[2] >= 0 else -1)
    return np.array([0.0, 0.0, 1.0])


def read_pdb_atoms(path):
    xyz, chain, ent = [], [], []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                chain.append(line[21].strip() or "A"); ent.append(line[17:20].strip())
    return np.array(xyz), np.array(chain), np.array(ent), {}


def read_mrc(path):
    """(density as data[z, y, x] float32, voxel (x, y, z) Angstrom, origin (x, y, z) Angstrom).

    MRC2014: NX NY NZ MODE at 0, NXSTART at 16, MX MY MZ at 28, CELLA at 40, MAPC MAPR MAPS at 64,
    NSYMBT at 92, ORIGIN at 196. The axis order is taken from MAPC/MAPR/MAPS, so a map stored with
    a permuted fast axis still comes back as data[z, y, x]. `.gz` is read through gzip."""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        h = f.read(1024)
        i4 = lambda o: int(np.frombuffer(h[o:o + 4], np.int32)[0])
        f4 = lambda o: float(np.frombuffer(h[o:o + 4], np.float32)[0])
        nx, ny, nz, mode = i4(0), i4(4), i4(8), i4(12)
        nstart = np.array([i4(16), i4(20), i4(24)])
        mx, my, mz = i4(28), i4(32), i4(36)
        cella = np.array([f4(40), f4(44), f4(48)])
        mapc, mapr, maps = i4(64), i4(68), i4(72)
        nsymbt = i4(92)
        origin = np.array([f4(196), f4(200), f4(204)])
        f.read(nsymbt)
        dt = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16, 12: np.float16}[mode]
        data = np.frombuffer(f.read(), dtype=dt)[: nx * ny * nz].reshape(nz, ny, nx).astype(np.float32)
    axes = [maps, mapr, mapc]                          # (slow, medium, fast) as xyz axis numbers 1..3
    perm = [axes.index(k) for k in (3, 2, 1)]          # which file axis holds z, y, x
    data = np.transpose(data, perm)
    vox = cella / np.array([mx, my, mz], float)       # Angstrom per voxel along x, y, z
    if not origin.any():
        origin = nstart * vox                          # older maps carry the origin as a start index
    return data, vox, origin


# ================================================================== geometry
def principal_axis(pts, longest):
    """The centre and a unit axis of a point cloud: its direction of GREATEST extent (a rod, a whole
    motor with its hook) or of LEAST extent (a single ring). Pointing +z-ish."""
    c = pts.mean(0)
    w, v = np.linalg.eigh(np.cov((pts - c).T))
    a = v[:, -1] if longest else v[:, 0]
    if a[2] < 0:
        a = -a
    return c, a / np.linalg.norm(a)


def frame(xyz, centre, axis):
    """Points in the axis's own frame: (x', y') across it, z along it. [N, 3]."""
    rel = xyz - centre
    e1 = np.cross(axis, [1.0, 0.0, 0.0])
    e1 = e1 if np.linalg.norm(e1) > 1e-6 else np.cross(axis, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(axis, e1)
    return np.stack([rel @ e1, rel @ e2, rel @ axis], 1)


def bands(xyz, chain, ent, desc, centre, axis, scale=0.1):
    """Per entity: radial and axial extent in nm (1st-99th percentile), copies, atoms."""
    P = frame(xyz, centre, axis) * scale
    r, z = np.linalg.norm(P[:, :2], axis=1), P[:, 2]
    out = []
    for e in sorted(set(ent), key=lambda k: (len(k), k)):
        m = ent == e
        out.append(dict(entity=e, name=desc.get(e, e), copies=int(len(set(chain[m]))), atoms=int(m.sum()),
                        r_mean=float(r[m].mean()),
                        r_in=float(np.percentile(r[m], 1)), r_out=float(np.percentile(r[m], 99)),
                        z0=float(np.percentile(z[m], 1)), z1=float(np.percentile(z[m], 99))))
    return out


def map_points(data, vox, origin, level):
    """The voxel centres above `level`, in nm, and their densities."""
    zi, yi, xi = np.nonzero(data > level)
    pts = (np.stack([xi * vox[0], yi * vox[1], zi * vox[2]], 1) + origin[None, :]) * 0.1
    return pts, data[zi, yi, xi]


def silhouette(P, nm_step=1.0):
    """At each height z (nm) the radial extent of the points (5th-99.5th percentile) and their count."""
    r, z = np.linalg.norm(P[:, :2], axis=1), P[:, 2]
    edges = np.arange(z.min(), z.max() + nm_step, nm_step)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z >= lo) & (z < hi)
        if m.sum() < 20:
            continue
        out.append((0.5 * (lo + hi), float(np.percentile(r[m], 5)), float(np.percentile(r[m], 99.5)), int(m.sum())))
    return out


# ================================================================== parts (the shape library)
def isosurface_from_points(P_nm, spacing=0.5, blur=1.0):
    """A closed surface around a point cloud: blurred occupancy on a grid, contoured at half its
    median occupied value. Returns (V [n,3] in METRES, the library's unit, F [m,3])."""
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes
    lo = P_nm.min(0) - 3 * spacing
    dim = np.ceil((P_nm.max(0) - lo + 3 * spacing) / spacing).astype(int) + 1
    D = np.zeros(tuple(dim), np.float32)
    ijk = np.clip(((P_nm - lo) / spacing).astype(int), 0, dim - 1)
    np.add.at(D, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1.0)
    D = gaussian_filter(D, blur)
    V, F, _, _ = marching_cubes(D, level=0.5 * float(np.median(D[D > 0])))
    return ((V * spacing + lo) * 1e-9).astype(np.float64), F.astype(np.int64)



def stator_positions(xyz, chain, ent, centre, axis, entity, scale=0.1):
    """One point per symmetry copy of `entity` (the stator's MotB, say): the centroid of that
    copy's chains, as (radius nm, angle deg, z nm) about the axis. Copies are the `@k` suffix
    the assembly expansion adds; an unexpanded model gives one row."""
    P = frame(xyz, centre, axis) * scale
    m = ent == entity
    keys = sorted({c.split("@")[1] if "@" in c else "0" for c in chain[m]}, key=int)
    out = []
    for k in keys:
        mk = m & np.array([("@" + k) == ("@" + c.split("@")[1] if "@" in c else "@0") for c in chain])
        q = P[mk].mean(0)
        out.append((float(np.hypot(q[0], q[1])), float(np.degrees(np.arctan2(q[1], q[0]))), float(q[2])))
    return out


def record_model(xyz, chain, ent, desc, centre, axis, rows, stators, title, why, png=None):
    """A record MEASUREMENT of an atomic model: side and top projections coloured by entity, the
    per-copy stator centroids marked, and the band table beside them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from builder_figure import record, style
    P = frame(xyz, centre, axis) * 0.1
    cols = ["#c0392b", "#2f6fb5", "#4a7c59", "#e08214", "#7b5aa6", "#17becf", "#8c564b", "#7f7f7f"]
    ents = [r["entity"] for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.15]))
    for e, col in zip(ents, cols):
        m = ent == e
        lab = desc.get(e, e)[:34]
        axs[0].scatter(P[m, 0], P[m, 2], s=0.6, color=col, alpha=0.35, lw=0, rasterized=True)
        axs[1].scatter(P[m, 0], P[m, 1], s=0.6, color=col, alpha=0.35, lw=0, rasterized=True)
    if stators:
        r_s = np.array([q[0] for q in stators]); th = np.radians([q[1] for q in stators]); z_s = np.array([q[2] for q in stators])
        axs[1].plot(r_s * np.cos(th), r_s * np.sin(th), "o", mfc="none", mec="black", ms=9, mew=1.2)
        axs[0].plot(r_s * np.cos(th), z_s, "o", mfc="none", mec="black", ms=7, mew=1.0)
    axs[0].set_xlabel("x across the axis (nm)"); axs[0].set_ylabel("z along the axis (nm; + = away from the inner membrane)")
    axs[0].set_aspect("equal"); axs[0].axhline(0, color="#bbbbbb", lw=0.6)
    axs[1].set_xlabel("x (nm)"); axs[1].set_ylabel("y (nm)"); axs[1].set_aspect("equal")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=col, ms=6, label=desc.get(e, e)[:34]) for e, col in zip(ents, cols)]
    if stators:
        handles.append(Line2D([], [], marker="o", ls="", mfc="none", mec="black", ms=7,
                              label=f"{len(stators)} stator centroids, r {r_s.mean():.1f} nm"))
    axs[1].legend(handles=handles, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=2)
    axs[0].text(0, 1.03, "A  side view, every atom, coloured by protein", transform=axs[0].transAxes, fontsize=10)
    axs[1].text(0, 1.03, "B  top view, looking down the axis", transform=axs[1].transAxes, fontsize=10)
    axs[2].axis("off")
    lines = [f"{'protein':34s}{'copies':>7s}{'r mean':>8s}{'r 1-99%':>12s}{'z 1-99%':>12s}"]
    for r in rows:
        lines.append(f"{r['name'][:34]:34s}{r['copies']:7d}{r['r_mean']:8.1f}{r['r_in']:6.1f}-{r['r_out']:<5.1f}{r['z0']:6.1f}-{r['z1']:<5.1f}")
    axs[2].text(0, 0.98, "C  bands about the axis (nm)\n\n" + "\n".join(lines), transform=axs[2].transAxes,
                fontsize=8.2, family="monospace", va="top")
    fig.tight_layout()
    if png:
        fig.savefig(png, dpi=150, facecolor="white", bbox_inches="tight"); print(f"[figure] -> {png}", flush=True)
    else:
        record(fig, why, name=title)


def short_name(desc, e):
    """A part key from an entity description: the protein's name where the paper uses one."""
    d = desc.lower()
    for k, v in (("flil", "FliL"), ("pfla", "PflA"), ("tpr", "PflB"), ("pdz", "PflCD"),
                 ("lipoprotein", "Lipo"), ("motb", "MotB"), ("mota", "MotA"), ("flig", "FliG"),
                 ("flif", "FliF"), ("flim", "FliM"), ("flin", "FliN")):
        if k in d:
            return v
    return f"entity_{e}"


def write_cloud(name, xyz, ent, desc, centre, axis, provenance):
    """The atoms themselves, per entity, into `<graphs_data>/shapes/<name>/points.npz` in metres,
    in the axis frame (axis = z, centre = origin), for `cloud_seed`."""
    from plexus.paths import graphs_data_path
    folder = os.path.join(graphs_data_path(), "shapes", name)
    os.makedirs(folder, exist_ok=True)
    P = frame(xyz, centre, axis) * 1e-10
    arrays = {}
    for e in sorted(set(ent), key=lambda k: (len(k), k)):
        arrays[short_name(desc.get(e, e), e)] = P[ent == e].astype(np.float32)
    np.savez_compressed(os.path.join(folder, "points.npz"), **arrays)
    with open(os.path.join(folder, "PROVENANCE.md"), "a") as f:
        f.write(provenance)
    print(f"[cloud] {folder}/points.npz: " + ", ".join(f"{k} {len(v):,}" for k, v in arrays.items()))
    return folder

def write_parts(name, parts, provenance):
    from plexus.paths import graphs_data_path
    folder = os.path.join(graphs_data_path(), "shapes", name)
    os.makedirs(folder, exist_ok=True)
    arrays, meta = {}, {}
    for pname, (V, F) in parts.items():
        arrays[f"{pname}__V"], arrays[f"{pname}__F"] = V, F
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        vol = abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)
        meta[pname] = {"volume": vol, "centroid": V.mean(0).tolist(), "extent": (V.max(0) - V.min(0)).tolist(),
                       "n_vertices": int(len(V)), "n_faces": int(len(F))}
    np.savez_compressed(os.path.join(folder, "parts.npz"), **arrays)
    json.dump(meta, open(os.path.join(folder, "parts.json"), "w"), indent=1)
    with open(os.path.join(folder, "PROVENANCE.md"), "w") as f:
        f.write(provenance)
    print(f"[parts] {folder}: " + ", ".join(f"{k} ({m['n_faces']:,} faces, {m['volume'] * 1e27:.0f} nm^3)"
                                             for k, m in meta.items()))
    return folder


def parse_slabs(s):
    """`name:z0:z1[:r0:r1],...` (nm along the axis, optional radial band) -> [(name, z0, z1, r0, r1)].

    THE RADIAL BAND IS WHAT SEPARATES A STATOR FROM A DISK: at one height the C. jejuni motor holds
    the C-ring, the stator units around it and the proximal disk outside those, so an axial slab
    alone would fuse all three into one part."""
    out = []
    for tok in (s or "").split(","):
        if not tok.strip():
            continue
        f = tok.split(":")
        n, z0, z1 = f[0].strip(), float(f[1]), float(f[2])
        r0, r1 = (float(f[3]), float(f[4])) if len(f) >= 5 else (0.0, float("inf"))
        out.append((n, z0, z1, r0, r1))
    return out


# ================================================================== self-test
def _selftest(tmp="/tmp/bfm_anatomy_selftest"):
    """A synthetic C34 annulus (r 15-22.5 nm, 20 nm tall) in each format, read back and measured."""
    os.makedirs(tmp, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 34
    lines = ["data_test", "#", "loop_", "_entity.id", "_entity.pdbx_description", "1 'FliM-like post'", "#", "loop_",
             "_atom_site.group_PDB", "_atom_site.id", "_atom_site.label_asym_id", "_atom_site.label_entity_id",
             "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z"]
    k = 0
    for i in range(n):
        th = 2 * np.pi * i / n
        for _ in range(60):
            rr = rng.uniform(150.0, 225.0); zz = rng.uniform(-220.0, -20.0); dth = rng.uniform(-0.05, 0.05)
            k += 1
            lines.append(f"ATOM {k} {chr(65 + i % 26)}{i // 26} 1 {rr * np.cos(th + dth):.3f} {rr * np.sin(th + dth):.3f} {zz:.3f}")
    cif = os.path.join(tmp, "ring.cif"); open(cif, "w").write("\n".join(lines) + "\n#\n")
    xyz, ch, en, de = read_mmcif_atoms(cif)
    c, a = principal_axis(xyz, longest=False)
    b = bands(xyz, ch, en, de, c, a)[0]
    print(f"  cif: {b['name']} x{b['copies']}: r {b['r_in']:.1f}-{b['r_out']:.1f} nm, height {b['z1'] - b['z0']:.1f} nm "
          f"(expected r 15-22.5, height 20)")
    vox, N = 2.0, 260
    g = (np.arange(N) - N / 2) * vox
    Z, Y, X = np.meshgrid(g, g, g, indexing="ij")           # data[z, y, x]
    R = np.sqrt(X ** 2 + Y ** 2)
    D = ((R > 150) & (R < 225) & (Z > -220) & (Z < -20)).astype(np.float32)
    hdr = np.zeros(256, np.int32)
    hdr[0:3] = [N, N, N]; hdr[3] = 2; hdr[7:10] = [N, N, N]
    hdr[10:13] = np.array([N * vox] * 3, np.float32).view(np.int32)
    hdr[13:16] = np.array([90.0] * 3, np.float32).view(np.int32)
    hdr[16:19] = [1, 2, 3]
    hdr[49:52] = np.array([-N / 2 * vox] * 3, np.float32).view(np.int32)
    hdr[52] = int.from_bytes(b"MAP ", "little")
    mrc = os.path.join(tmp, "ring.mrc.gz")
    with gzip.open(mrc, "wb") as f:
        f.write(hdr.tobytes()); f.write(D.tobytes())
    data, v, o = read_mrc(mrc)
    pts, _ = map_points(data, v, o, 0.5)
    c2, a2 = principal_axis(pts, longest=False)
    P = frame(pts, c2, a2)
    sil = silhouette(P)
    rin, rout = np.median([s[1] for s in sil]), np.median([s[2] for s in sil])
    print(f"  map (gz): voxel {v[0]:.1f} A, {len(pts):,} voxels above level; height {sil[-1][0] - sil[0][0] + 1:.0f} nm, "
          f"r {rin:.1f}-{rout:.1f} nm (expected 20; 15-22.5)")
    V, F = isosurface_from_points(frame(xyz, c, a) * 0.1)
    print(f"  surface: {len(F):,} faces from the atoms; ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--level", type=float, default=None, help="map threshold (density units); default mean + 3 sd")
    ap.add_argument("--axis", default=None, choices=[None, "x", "y", "z", "longest", "shortest"],
                    help="the motor axis; default: longest extent for a map, shortest for a model")
    ap.add_argument("--slabs", default="", help="map only: 'name:z0:z1[:r0:r1],...' bands (nm along and across the axis) to cut into parts")
    ap.add_argument("--parts", default=None, help="write shape-library parts under this name")
    ap.add_argument("--no-assembly", action="store_true", help="atomic model: keep the deposited asymmetric unit, do not apply its symmetry operators")
    ap.add_argument("--stator-entity", default=None, help="atomic model: the entity id whose symmetry copies are the stator units (their centroids are printed and marked)")
    ap.add_argument("--record", default=None, help="write a record MEASUREMENT figure into $PLEXUS_BUILDER under this name")
    ap.add_argument("--why", default=None, help="the why text of that record step")
    ap.add_argument("--png", default=None, help="with --record: redraw an existing step's figure to this path instead of taking a new step")
    ap.add_argument("--cloud", default=None, help="atomic model: write the atoms per entity as shapes/<name>/points.npz for cloud_seed")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return
    if not a.files:
        ap.error("give a .cif/.pdb/.map/.mrc file (gz ok), or --selftest")

    def axis_of(pts, default_longest):
        if a.axis in ("x", "y", "z"):
            ax = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]["xyz".index(a.axis)])
            return pts.mean(0), ax
        return principal_axis(pts, longest=(a.axis == "longest") if a.axis else default_longest)

    parts, prov = {}, ["# Provenance", ""]
    for path in a.files:
        base = os.path.basename(path)
        ext = base.replace(".gz", "").rsplit(".", 1)[-1].lower()
        print(f"\n== {path}")
        if ext in ("cif", "mmcif", "pdb", "ent"):
            xyz, ch, en, de = read_pdb_atoms(path) if ext in ("pdb", "ent") else read_mmcif_atoms(path, assembly=not a.no_assembly)
            c, ax = axis_of(xyz, default_longest=False)
            print(f"  {len(xyz):,} atoms, {len(set(ch))} chains; axis {np.round(ax, 3).tolist()}, centre {np.round(c * 0.1, 2).tolist()} nm")
            print(f"  {'entity':8s} {'name':40s} {'copies':>6s} {'r_mean':>6s} {'r_in':>6s} {'r_out':>6s} {'z0':>7s} {'z1':>7s}  (nm, about the axis)")
            for b in bands(xyz, ch, en, de, c, ax):
                print(f"  {b['entity']:8s} {b['name'][:40]:40s} {b['copies']:6d} {b['r_mean']:6.1f} {b['r_in']:6.1f} {b['r_out']:6.1f} {b['z0']:7.1f} {b['z1']:7.1f}")
                if a.parts:
                    key = (b["name"].replace(" ", "_")[:24] or f"entity_{b['entity']}")
                    parts[key] = isosurface_from_points(frame(xyz[en == b["entity"]], c, ax) * 0.1)
            if a.stator_entity:
                st = stator_positions(xyz, ch, en, c, ax, a.stator_entity)
                rs = np.array([q[0] for q in st]); zs = np.array([q[2] for q in st])
                print(f"  stators = entity {a.stator_entity}: {len(st)} copies, centroid radius {rs.mean():.2f} +- {rs.std():.2f} nm, "
                      f"z {zs.mean():.2f} nm; angles {np.round(sorted(q[1] for q in st), 1).tolist()} deg")
            else:
                st = []
            if a.record:
                record_model(xyz, ch, en, de, c, ax, bands(xyz, ch, en, de, c, ax), st, a.record,
                             a.why or f"MEASUREMENT of {base}: the atomic model about its symmetry axis.", png=a.png)
            if a.cloud:
                write_cloud(a.cloud, xyz, en, de, c, ax,
                            f"# points.npz\n- {base}: atomic model, {len(xyz):,} atoms after the entry's symmetry expansion; "
                            f"one array per entity, metres, axis z, centre at the origin; written by tools/bfm_anatomy_from_structure.py --cloud\n")
            prov.append(f"- {base}: atomic model, {len(xyz):,} atoms; parts are density isosurfaces of each entity's atoms on a 0.5 nm grid")
        elif ext in ("map", "mrc", "ccp4"):
            data, vox, org = read_mrc(path)
            lvl = a.level if a.level is not None else float(data.mean() + 3.0 * data.std())
            print(f"  grid {data.shape[::-1]} (x, y, z), voxel {vox[0]:.3f} A, origin {np.round(org, 1).tolist()} A; "
                  f"density {data.min():.4g}..{data.max():.4g}, level {lvl:.4g}")
            pts, dens = map_points(data, vox, org, lvl)
            if pts.shape[0] < 100:
                raise SystemExit(f"only {pts.shape[0]} voxels above level {lvl}; lower --level")
            c, ax = axis_of(pts, default_longest=True)
            P = frame(pts, c, ax)
            print(f"  {len(pts):,} voxels above level; axis {np.round(ax, 3).tolist()}, centre {np.round(c, 1).tolist()} nm")
            print(f"  silhouette along the axis (nm):\n  {'z':>7s} {'r_in':>6s} {'r_out':>6s} {'voxels':>7s}")
            for zz, ri, ro, nn in silhouette(P):
                print(f"  {zz:7.1f} {ri:6.1f} {ro:6.1f} {nn:7d}")
            slabs = parse_slabs(a.slabs)
            if a.parts:
                if slabs:
                    rr = np.linalg.norm(P[:, :2], axis=1)
                    for nm, z0, z1, r0, r1 in slabs:
                        m = (P[:, 2] >= z0) & (P[:, 2] < z1) & (rr >= r0) & (rr < r1)
                        if m.sum() < 50:
                            print(f"  slab {nm}: only {int(m.sum())} voxels in {z0}..{z1} nm; skipped"); continue
                        parts[nm] = isosurface_from_points(P[m], spacing=max(0.5, vox[0] * 0.1))
                        print(f"  slab {nm}: {int(m.sum()):,} voxels, z {z0}..{z1} nm -> {len(parts[nm][1]):,} faces")
                else:
                    parts[base.split(".")[0]] = isosurface_from_points(P, spacing=max(0.5, vox[0] * 0.1))
            prov.append(f"- {base}: cryo-EM map, level {lvl:.4g}, axis {np.round(ax, 3).tolist()}"
                        + (f", slabs {a.slabs}" if slabs else "") + "; parts are isosurfaces of the density above the level")
        else:
            print(f"  (unknown extension {ext}; give .cif/.pdb or .map/.mrc, gz ok)")
    if a.parts and parts:
        write_parts(a.parts, parts, "\n".join(prov) + "\n")


if __name__ == "__main__":
    main()
