"""A crowd of channels in one membrane: exp04's OmpF ion-flow rig (step 0070, v11t) carried by N copies.

    PYTHONPATH=src python tools/channel_crowd_spec.py [--n 100] [--spacing_nm 20] [--frames 10] [--name crowd100_ompf]

Writes config/channel/<name>.yaml. Every ion parameter, both pore tables (the K+ and the Cl- paths) and the
protein, lipid and charge-site clouds are v11t's, read from its spec as run; what changes is only how many:

- `channel` holds N entities placed by random sequential adsorption in a square of side sqrt(N) x spacing
  (so the MEAN spacing is `spacing_nm`), no two centres nearer than 0.75 x spacing, each turned by its own
  random angle about the membrane normal (`seed_state` on `orient`);
- the three protomers and the charge sites are child sets of `channel` (one copy per channel, turned by
  its `orient`), and each protomer displaces the lipid under it (`cloud_seed ... displace`);
- the lipid is one bilayer-thick slab of random points across the whole box, at 10 per nm^2;
- the ions are four sets, K and Cl on each side of the membrane, at v11t's own number per unit area and in
  its own z bands, so a colour tells which side an ion started on;
- `slab_barrier` carries v11t's pore tables on every channel (`parents: channel`), and the drive is
  Goldman's constant field across the core (`field_force ... constant_field`) instead of the conduction
  solve, which has no grid at this size.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "experiments", "exp04_membrane_channels", "spec", "0070.yaml")
SHAPE = "exp04r_ompf_v11t"
NM = 1.0 / 14.0                                     # world per nm (length_um 0.014)


def rsa(n, side, dmin, margin, seed):
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(200000):
        p = rng.uniform(margin, side - margin, 2)
        if all(np.hypot(*(p - q)) >= dmin for q in pts):
            pts.append(p)
            if len(pts) == n:
                return np.array(pts)
    raise RuntimeError(f"placed {len(pts)} of {n} channels; lower the density")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--spacing_nm", type=float, default=20.0)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--movie_frames", type=int, default=100)
    ap.add_argument("--name", default="crowd100_ompf")
    a = ap.parse_args()

    S = yaml.safe_load(open(BASE))
    ops = {}
    for o in S["operators"]:
        ops.setdefault(o["op"], []).append(o)
    z0 = float(ops["slab_barrier"][0]["z0"])
    h = float(ops["slab_barrier"][0]["half_thickness"])
    wpm = float(S["seed"][0]["scale"])
    ox, oy = (float(v) for v in S["seed"][0]["origin"][:2])

    side = math.sqrt(a.n) * a.spacing_nm * NM
    C = rsa(a.n, side, 0.75 * a.spacing_nm * NM, 0.5, seed=0)
    d = np.hypot(*(C[:, None] - C[None]).transpose(2, 0, 1)) + np.eye(a.n) * 1e9
    print(f"{a.n} channels in {side / NM:.0f} nm x {side / NM:.0f} nm: nearest neighbour "
          f"{d.min(1).mean() / NM:.1f} nm mean, {d.min() / NM:.1f} nm min")

    pts = np.load(os.path.join(ROOT, "graphs_data", "shapes", SHAPE, "points.npz"))
    area = side * side
    bands = {}
    for sp in ("K", "Cl"):
        q = pts[sp] * wpm
        lo = q[:, 2] < 0
        bands[(sp, "bottom")] = (int(round(lo.sum() * area)), float(q[lo, 2].min() + z0), float(q[lo, 2].max() + z0))
        bands[(sp, "top")] = (int(round((~lo).sum() * area)), float(q[~lo, 2].min() + z0), float(q[~lo, 2].max() + z0))
    lipid_half = float(np.abs(pts["lipid"][:, 2]).max() * wpm)
    n_lipid = int(round(10.0 * (side / NM) ** 2))

    ion_sets = [f"{sp}_{side_}" for sp in ("K", "Cl") for side_ in ("top", "bottom")]
    sp_of = {s: s.split("_")[0] for s in ion_sets}
    pos = {"pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"}}
    mob = {"pos": {"width": 3, "role": "coordinate", "integration": "first_order", "boundary": "world"}}
    prot = ["OmpF_1", "OmpF_2", "OmpF_3"]

    sets = {"cell": {"n": 1, "start": [[side / 2, side / 2, z0]], "state": S["sets"]["cell"]["state"]},
            "channel": {"n": a.n, "start": [[round(float(x), 6), round(float(y), 6), z0] for x, y in C],
                        "state": {**pos, "orient": {"width": 1, "integration": "none", "unit": "1"}}},
            "lipid": {"n": n_lipid, "start": [0.0, 0.0, z0 - lipid_half, side, side, z0 + lipid_half], "state": pos}}
    for k, p_ in enumerate(prot):
        sets[p_] = {"parent": "channel", "per_parent": int(pts[f"c{k}"].shape[0]), "state": pos}
    sets["qsite"] = {"parent": "channel", "per_parent": int(pts["qsite"].shape[0]),
                     "state": {**pos, "q": {"width": 1, "integration": "none", "unit": "1"}}}
    for s in ion_sets:
        n_, zlo, zhi = bands[(sp_of[s], s.split("_")[1])]
        sets[s] = {"n": n_, "start": [0.0, 0.0, round(zlo, 6), side, side, round(zhi, 6)], "state": mob}

    seeds = [{"op": "seed_state", "at": "channel", "block": "orient", "pattern": "random", "lo": 0.0, "hi": 360.0,
              "seed": 1}]
    seeds += [{"op": "cloud_seed", "at": p_, "cloud": f"{SHAPE}/c{k}", "scale": wpm, "rotate": "orient",
               "displace": "lipid", "reach": round(0.5 * NM, 6)} for k, p_ in enumerate(prot)]
    seeds += [{"op": "cloud_seed", "at": "qsite", "cloud": f"{SHAPE}/qsite", "scale": wpm, "rotate": "orient"},
              {"op": "seed_state_from_file", "at": "qsite", "file": f"shapes/{SHAPE}/blocks.npz",
               "blocks": {"q": "qsite_q"}}]

    def per_ion(v):
        return {s: v[sp_of[s]] for s in ion_sets} if isinstance(v, dict) else v

    def ion_map(v, extra=None):
        out = {s: v[sp_of[s]] for s in ion_sets}
        out.update(extra or {})
        return out

    pp = ops["pair_potential"]
    ii, il, ip, iq = pp[0], pp[1], pp[2], pp[3]
    new_ops = [dict(ops["membrane_potential"][0])]
    new_ops.append({**ii, "at": ion_sets[0], "sets": ion_sets, "charge": per_ion(ii["charge"]),
                    "sigma": per_ion(ii["sigma"]), "mobility": per_ion(ii["mobility"])})
    new_ops.append({**il, "at": ion_sets[0], "sets": ion_sets, "mobility": per_ion(il["mobility"])})
    new_ops.append({**ip, "at": ion_sets[0], "sets": ion_sets, "with_sets": prot, "mobility": per_ion(ip["mobility"]),
                    "sigma": ion_map(ip["sigma"], {p_: ip["sigma"]["OmpF_1"] for p_ in prot})})
    new_ops.append({**iq, "at": ion_sets[0], "sets": ion_sets, "charge": per_ion(iq["charge"]),
                    "sigma": ion_map(iq["sigma"], {"qsite": iq["sigma"]["qsite"]}), "mobility": per_ion(iq["mobility"])})
    ff = {o["at"]: o for o in ops["field_force"]}
    sb = {o["at"]: o for o in ops["slab_barrier"]}
    br = {o["at"]: o for o in ops["brownian"]}
    for k, s in enumerate(ion_sets):
        sp = sp_of[s]
        f_ = {kk: vv for kk, vv in ff[sp].items() if kk not in ("field", "channel", "volt_per_sim")}
        new_ops.append({**f_, "at": s, "constant_field": {"cell": "cell", "slab": [round(z0 - h, 6), round(z0 + h, 6)]}})
        local = [{"profile": [[r[0], r[1], r[2], r[3], round(r[4] - ox, 7), round(r[5] - oy, 7)] for r in po["profile"]]}
                 for po in sb[sp]["pores"]]
        new_ops.append({**sb[sp], "at": s, "pores": local, "parents": "channel", "rotate": "orient"})
        new_ops.append({**br[sp], "at": s, "seed": 3 + k})
    cc = ops["compartment_count"][0]
    new_ops.append({"op": "compartment_count", "at": "cell", "z_m": cc["z_m"], "cell": "cell", "sets": ion_sets,
                    "charges": [1.0 if sp_of[s] == "K" else -1.0 for s in ion_sets],
                    "block_charge": "q_in", "block_current": "I_in", "window": cc["window"]})

    P = dict(S["plotting"])
    colors = {**{p_: P["colors"][p_] for p_ in prot}, "lipid": P["colors"]["lipid"],
              "K_top": [0.7, 0.53, 1.0], "K_bottom": [1.0, 0.62, 0.85],
              "Cl_top": [0.35, 0.85, 0.45], "Cl_bottom": [0.85, 0.9, 0.35]}
    P.update({"compartment_sets": prot + ["lipid"] + ion_sets, "hide_sets": ["cell", "channel", "qsite"],
              "surface": {**{k: v for k, v in P["surface"].items() if k in prot + ["lipid"]},
                          **{s: {"render": "dots", "point_size": 2.0, "every": 3} for s in ion_sets}},
              "opacity": {**{p_: 1.0 for p_ in prot}, "lipid": 0.45, **{s: 1.0 for s in ion_sets}},
              "colors": colors, "max_frames": a.movie_frames, "duration_s": 10.0, "stills": 8, "keep_stills": True,
              "curve": [c for c in P["curve"] if "q_in" in c["quantity"]]})
    P["curve"][0].update({"ymin": -15.0 * a.n, "ymax": 3.0 * a.n})
    P["zoom"] = 1.25
    P.pop("descriptions", None)

    g = dict(S["general"])
    g.update({"name": a.name, "n_frames": a.frames, "world": [round(side, 6), round(side, 6), 1.0],
              "record_cap": min(a.frames + 1, 101)})
    g.pop("field_record_cap", None)
    spec = {"general": g, "sets": sets, "fields": {}, "seed": seeds, "operators": new_ops,
            "schedule": [o["op"] for o in new_ops], "plotting": P}
    out = os.path.join(ROOT, "config", "channel", f"{a.name}.yaml")
    yaml.safe_dump(spec, open(out, "w"), sort_keys=False, default_flow_style=None, width=120)
    n_ions = sum(sets[s]["n"] for s in ion_sets)
    print(f"-> {os.path.relpath(out, ROOT)}: {n_ions:,} ions, {n_lipid:,} lipid points, "
          f"{a.n * sum(sets[p_]['per_parent'] for p_ in prot):,} protein beads, {a.frames} frames")


if __name__ == "__main__":
    main()
