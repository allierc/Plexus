"""The anatomy audit of experiment 4: does the membrane sit where the protein says it should, and does
the channel's pore cross it?

    PYTHONPATH=src python tools/channel_audit.py exp04_v5b exp04_v8c ... [--fig out.png]

WHY (the human, 2026-09-26, on a side view of record step 0019): "the grey part will always block the
flow?" -- alpha-hemolysin's stems stood BELOW the membrane slab for a whole round, so no ion could ever
cross, and no gating model could have changed that. A channel conducts only if its lumen crosses the
whole hydrocarbon core. That is geometry, readable from frame 0, and it is read here by code for every
channel before it runs again.

WHAT IT READS, per run, at the first and the last recorded frame:

    membrane        the lipid beads' mid-plane and faces (5th/95th percentile of z, from lipids farther
                    than the protein's reach from the axis -- the bulk bilayer), the Born barrier's
                    core (`slab_barrier` z0 +- half_thickness) and the continuum solve's insulating
                    slab (`electrolyte_conduction.slab`), all in nm from the lipid mid-plane
    hydrophobic     the Wimley-White octanol scale on each bead's residue (the deposited file's residue
      belt          names, beads matched to residues chain by chain in file order): the height z_c that
                    puts the most hydrophobic residues inside a 3.0 nm core and the most polar outside
                    -- the membrane's place by the protein's own chemistry (OPM's idea, crudely). Its
                    offset from the lipid mid-plane is the mismatch.
    pore wall       per 0.25 nm of height, the radius r_wall of the nearest protein bead to the axis and
                    whether the beads within r_wall + 1 nm ENCLOSE the axis (6 of 8 angular sectors):
                    a height inside the core with no enclosing wall is lipid territory -- the pore does
                    not cross the membrane there. r_free = r_wall - 0.5 nm (a residue's reach beyond its
                    alpha carbon, as `pore_probe`).
    lumen lipids    lipid beads inside the wall at their height (they plug the pore; the continuum solve
                    treats them as insulators)

Verdict: SPANS (an enclosing wall at every height of the core) or DOES NOT SPAN (the heights where it is
missing), the narrowest r_free inside the core, lumen lipids, and the belt offset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

REACH_NM = 0.5            # a residue's reach beyond its alpha carbon (pore_probe's)
CORE_NM = 3.0             # the hydrocarbon core the belt is fitted with
BIN_NM = 0.25
OUT = os.path.join(ROOT, "experiments", "specs", "exp04", "anatomy_audit.jsonl")


def _run_dir(run):
    from plexus.paths import graphs_data_path
    return graphs_data_path("channel", run)


def _load(run):
    d = _run_dir(run)
    sp = yaml.safe_load(open(os.path.join(d, "spec.yaml")))
    z = np.load(os.path.join(d, "trajectory.npz"))
    L_nm = float(sp["general"]["units"]["length_um"]) * 1000.0          # nm per world unit
    return sp, z, L_nm


def _protein_sets(sp):
    ions = {"K", "Cl", "Na"}
    return [s for s in sp["sets"] if "_" in s and s.split("_")[-1].isdigit() and s not in ions]


def _frames(z, name):
    a = z[f"{name}__pos"]
    return a if a.ndim == 3 else a[None]


def _residues(sp, prot):
    """Residue names per bead, from the deposited file named in the shape folder's anatomy.json,
    matched chain by chain in file order over the kept residue ranges; None when the counts differ."""
    try:
        from plexus.paths import graphs_data_path
        seeds = sp.get("seed") or []
        cloud = next((o.get("cloud") for o in seeds if o.get("at") == prot[0]), None)
        if not cloud:
            return None
        folder = os.path.join(graphs_data_path(), "shapes", cloud.split("/")[0])
        info = json.load(open(os.path.join(folder, "anatomy.json")))
        from channel_anatomy import read_atoms
        cif = os.path.join(ROOT, "experiments", "exp04_membrane_channels", "data", info["model"])
        a = read_atoms(cif, assembly=False)
        ranges = [tuple(r) for p in info["parts"].values() if "atom" not in p for r in p["residues"]]
        m = (a["atom"] == "CA")
        keep = np.zeros(len(a["atom"]), bool)
        for lo, hi in ranges:
            keep |= (a["resid"] >= lo) & (a["resid"] <= hi)
        m &= keep
        out = []
        for ch, n in zip(info["chains"], info["per_chain"]):
            names = a["resname"][m & (a["chain"] == ch)]
            if len(names) != n:
                return None
            out.append(names)
        return out
    except Exception:                                                   # noqa: BLE001
        return None


def audit(run):
    sp, z, L = _load(run)
    prot = _protein_sets(sp)
    if not prot:
        return {"run": run, "error": "no protein sets"}
    lip = _frames(z, "lipid") * L
    P = [_frames(z, s) * L for s in prot]
    nF = min(lip.shape[0], min(p.shape[0] for p in P))
    ops = sp.get("operators") or []
    born = next((o for o in ops if o.get("op") == "slab_barrier"), None)
    cond = next((o for o in ops if o.get("op") == "electrolyte_conduction"), None)
    res = _residues(sp, prot)
    WW = __import__("channel_spec").WW_OCT
    out = {"run": run, "channel_sets": f"{prot[0].rsplit('_', 1)[0]} x{len(prot)}"}
    for tag, f in (("first", 0), ("last", nF - 1)):
        X = np.concatenate([p[f] for p in P])
        ax = X[:, :2].mean(0)
        r_prot = np.linalg.norm(X[:, :2] - ax, axis=1)
        R_reach = float(np.percentile(r_prot, 98)) + 1.0
        Lf = lip[f]
        r_lip = np.linalg.norm(Lf[:, :2] - ax, axis=1)
        bulk = Lf[r_lip > R_reach]
        if len(bulk) < 20:
            bulk = Lf
        zm = float(np.median(bulk[:, 2]))
        lo, hi = float(np.percentile(bulk[:, 2], 5)) - zm, float(np.percentile(bulk[:, 2], 95)) - zm
        rec = {"membrane_faces_nm": [round(lo, 2), round(hi, 2)]}
        if born:
            z0 = float(born["z0"]) * L - zm
            h = float(born["half_thickness"]) * L
            rec["born_core_nm"] = [round(z0 - h, 2), round(z0 + h, 2)]
        if cond and cond.get("slab"):
            rec["conduction_slab_nm"] = [round(float(cond["slab"][0]) * L - zm, 2),
                                         round(float(cond["slab"][1]) * L - zm, 2)]
        zp = X[:, 2] - zm
        rec["protein_z_nm"] = [round(float(zp.min()), 2), round(float(zp.max()), 2)]
        # the core the pore must cross: the Born core when there are explicit ions, else the lipid faces
        c_lo, c_hi = rec.get("born_core_nm", [lo, hi])
        edges = np.arange(c_lo, c_hi + 1e-9, BIN_NM)
        ang = np.arctan2(X[:, 1] - ax[1], X[:, 0] - ax[0])
        missing, rfree = [], []
        for zb in edges[:-1]:
            sel = (zp >= zb - 0.3) & (zp < zb + BIN_NM + 0.3)
            if sel.sum() < 3:
                missing.append(round(float(zb), 2)); continue
            rw = float(r_prot[sel].min())
            ring = sel & (r_prot < rw + 1.0)
            sectors = len(set(((ang[ring] + np.pi) / (2 * np.pi) * 8).astype(int) % 8))
            if sectors < 6:
                missing.append(round(float(zb), 2)); continue
            rfree.append(rw - REACH_NM)
        rec["spans"] = not missing
        rec["core_heights_without_a_wall_nm"] = missing[:12] + (["..."] if len(missing) > 12 else [])
        rec["narrowest_free_radius_in_core_nm"] = round(float(min(rfree)), 3) if rfree else None
        # lipids inside the wall at their own height, within the core
        zl = Lf[:, 2] - zm
        inside = 0
        for zb in edges[:-1]:
            sel = (zp >= zb - 0.3) & (zp < zb + BIN_NM + 0.3)
            if sel.sum() < 3:
                continue
            rw = float(r_prot[sel].min())
            inside += int(((zl >= zb) & (zl < zb + BIN_NM) & (r_lip < rw)).sum())
        rec["lumen_lipids"] = inside
        if res is not None and tag == "first":
            w = np.concatenate([[WW.get(str(n), 0.0) for n in names] for names in res])
            if len(w) == len(zp):
                best, zc_best = None, None
                for zc in np.arange(zp.min(), zp.max(), 0.1):
                    inn = np.abs(zp - zc) < CORE_NM / 2
                    E = float(w[inn].sum())                                     # low = hydrophobic inside
                    if best is None or E < best:
                        best, zc_best = E, float(zc)
                rec["hydrophobic_belt_centre_nm"] = round(zc_best, 2)
        out[tag] = rec
    return out


def figure(runs, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(runs)
    fig, axs = plt.subplots(n, 2, figsize=(9.0, 3.2 * n), squeeze=False)
    for i, run in enumerate(runs):
        sp, z, L = _load(run)
        prot = _protein_sets(sp)
        lip = _frames(z, "lipid") * L
        P = [_frames(z, s) * L for s in prot]
        nF = min(lip.shape[0], min(p.shape[0] for p in P))
        ions = [s for s in ("K", "Cl") if f"{s}__pos" in z.files]
        for j, (tag, f) in enumerate((("first", 0), ("last", nF - 1))):
            a = axs[i, j]
            X = np.concatenate([p[f] for p in P])
            ax0 = X[:, :2].mean(0)
            zm = results[i].get(tag, {})
            Lf = lip[f]
            zmid = float(np.median(Lf[:, 2]))
            cut = lambda Y: np.abs(Y[:, 1] - ax0[1]) < 1.0                # noqa: E731  a 2 nm thick cut through the axis
            a.scatter(Lf[cut(Lf), 0] - ax0[0], Lf[cut(Lf), 2] - zmid, s=6, c="0.7", lw=0)
            cols = plt.cm.tab10(np.linspace(0, 1, 10))
            for k, p in enumerate(P):
                Y = p[f]
                a.scatter(Y[cut(Y), 0] - ax0[0], Y[cut(Y), 2] - zmid, s=5, color=cols[k % 10], lw=0)
            for s, c in zip(ions, ("#9b6bff", "#3cb371")):
                Y = _frames(z, s)[f] * L
                a.scatter(Y[cut(Y), 0] - ax0[0], Y[cut(Y), 2] - zmid, s=4, color=c, lw=0)
            for key, ls in (("membrane_faces_nm", "-"), ("born_core_nm", "--")):
                if key in zm:
                    for h in zm[key]:
                        a.axhline(h, color="k", ls=ls, lw=0.7)
            if "hydrophobic_belt_centre_nm" in results[i].get("first", {}):
                a.axhline(results[i]["first"]["hydrophobic_belt_centre_nm"], color="tab:orange", lw=1.0)
            a.set_xlim(-8, 8); a.set_ylim(-8, 8); a.set_aspect("equal")
            a.set_xlabel("x from the pore axis (nm)"); a.set_ylabel("z from the lipid mid-plane (nm)")
            v = "SPANS" if zm.get("spans") else "DOES NOT SPAN"
            a.set_title(f"{run}, {tag} frame: {v}; lumen lipids {zm.get('lumen_lipids')}", fontsize=8, loc="left")
            for sp_ in ("top", "right"):
                a.spines[sp_].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"[audit] figure -> {os.path.relpath(path, ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    results = []
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for run in a.runs:
        r = audit(run)
        results.append(r)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(json.dumps(r))
    if a.fig:
        figure(a.runs, results, a.fig)


if __name__ == "__main__":
    main()
