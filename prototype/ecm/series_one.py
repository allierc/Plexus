"""69-75: the spring-graph membrane at work, and the configurations that change its dynamics.

CHOSEN FROM WHAT THE SWEEPS RULED OUT. `k_bond` moves nothing across three orders (the 4x4 grid), and
the secretion threshold at 0.006 is already mapped, so neither is a useful axis for showing different
behaviour. What the sweeps pointed AT is remodelling: strain sits at growth_rate x tau regardless of
stiffness or supply, so `tau` is the parameter that decides whether the sheet ever approaches its break
threshold. These seven vary the things that actually move it.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "prototype", "Tyssue"),
          os.path.join(_ROOT, "discovery_okuda")):
    sys.path.insert(0, p)
import json

import aniso
import combine as C
import membrane_ops
import run_ecm as R
import tissue as TIS

GATE = os.path.join(_ROOT, "log", "okuda_ECM", "49_aniso_i0_fibres", "load.npz")

# k = 5000: inside the graph-mode ceiling of 8,220 measured at k = 200..40,000.
BASE = dict(membrane_particles=45000, membrane_cutoff=0.008, membrane_break=0.35,
            membrane_bond_k=5.0e3, membrane_adhesion=1.0e4, membrane_tau=60.0,
            membrane_jitter=0.35, membrane_reserve=12.5, membrane_secrete_rate=0.012,
            membrane_impl="graph", membrane_drag=40.0)

SERIES = {
    "69_graph_reference": dict(),
    # tau = 0 removes `basement_membrane_remodel` entirely: rest lengths are frozen, so stretch
    # ACCUMULATES instead of being forgotten. This is the one configuration in which the 0.35 break
    # threshold is reachable at all -- every run so far sat at 0.08 because remodelling absorbed it.
    "70_no_remodelling": dict(membrane_tau=0.0),
    "71_fast_remodelling": dict(membrane_tau=10.0),
    "72_starved": dict(membrane_secrete_rate=0.002),
    # the measured null: without an anchor the sheet SLIDES over the epithelium and never stretches,
    # mean bond strain 0.0000 at every stiffness. Kept in the series because it is the control that
    # says the strain in the others is real.
    "73_no_adhesion": dict(membrane_adhesion=0.0),
    "74_brittle": dict(membrane_break=0.08),
    "75_on_ovoid": dict(_gated=True),

    # --- 76 onward: THE FIXED SEED ------------------------------------------------------------------
    # Everything above 75 was deleted. The sheet those runs started from was not packed: the relaxation
    # ran over the whole 45,000-particle reservoir and the sheet then kept every 13.5th of them, and a
    # SUBSET of a blue-noise set is not blue noise -- thinning randomises it back to Poisson. Measured on
    # the seeded sheet, nearest-neighbour distance was 0.505 of a hexagonal packing with cv 0.411,
    # against 0.461 / 0.535 for points thrown at random. So it began 8% better than random, which is why
    # holes were visible in the first frames and grew from there. Relaxing the laid-down count directly
    # gives 0.885 with cv 0.046.
    #
    # 76 is the new nominal: same configuration as 69, on a sheet that starts packed.
    "76_reference_fixed_seed": dict(),

    # --- 77: THE BIOLOGICAL VERSION -----------------------------------------------------------------
    # Two rules replaced, both because the versions they replace are not things a cell could do.
    #
    # DEPOSITION. Cells secrete basement membrane basally, into the patch of surface directly beneath
    # themselves -- laminin polymerises where integrin and dystroglycan nucleate it, collagen IV
    # assembles onto that. So deposition is local to each cell and UNIFORM PER UNIT BASAL AREA. Placing
    # new material beside an existing node is a random walk and clumps; placing it in the largest gap
    # packs better but is worse biology, because it needs a global view of where the holes are and a cell
    # has none. Uniform fills a hole at the same rate it adds anywhere else, which is the honest
    # mechanism, and leaves the evening-out to the network.
    #
    # ANCHORS. `u0` was frozen at seeding, which says a patch of membrane remembers where a cell was
    # four hundred frames ago. Focal adhesions turn over in seconds to minutes, and the cells themselves
    # divide and swap neighbours, so the tissue a patch is attached to MOVES. Frozen anchors are also
    # what pins the holes: material that relaxes into a gap is dragged back out by its own tether.
    "77_biological": dict(membrane_deposit="uniform", membrane_tau_adh=40.0),
    # the two controls that say which of the two changes did the work
    "78_uniform_only": dict(membrane_deposit="uniform", membrane_tau_adh=0.0),
    "79_turnover_only": dict(membrane_deposit="parent", membrane_tau_adh=40.0),

}


def main():
    name = sys.argv[1]
    dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 402
    over = dict(SERIES[name])
    gated = over.pop("_gated", False)
    myo = over.pop("_myo", None)          # junction-level knobs: they belong to PASS 1, not the spec
    # RECORDED BEFORE THE POP CONSUMES IT. `_gated` is the only change 75 makes, so popping it left an
    # empty dict and the run archived itself as `{"reference": true}` -- identical to 69's label, for a
    # run on a tissue of aspect 1.332 against 69's 1.021.
    label = dict(over)
    if myo:
        label.update(myo)
    if gated:
        label["tissue"] = "gated ovoid (aspect 1.33)"
    for t in (membrane_ops.BOND_TRACE, membrane_ops.MEMBRANE_STRAIN,
              membrane_ops.SECRETE_TRACE, membrane_ops.BOND_SNAPSHOTS, membrane_ops.HOOP_TRACE):
        t.clear()
    tk = dict(frames=401, device=dev, buffer_x=4, myosin=1.0)
    if myo:
        tk.update(myo)
        # `:g` on a str raises; `myo_keyed_on` is a string ("tension"/"strain_rate"), so the tag has to
        # format by type. The cache key is built from this, so getting it wrong either crashes or -- worse
        # -- collides two different configurations onto one cached tissue.
        tk["tag_extra"] = "_" + "_".join(
            f"{k}{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k}{v}"
            for k, v in sorted(myo.items()))
    if gated:
        tk.update(gate_npz=GATE, gate_p_half="auto", gate_hill=6.0, gate_floor=0.08,
                  gate_smooth_frames=25, gate_smooth_phi=360.0, tag_extra="_gated_myo")
    npz = TIS.load_or_build(**tk)
    cfg = dict(aniso.BASE)
    cfg.update(BASE)
    cfg.update(over)
    cfg["membrane"] = npz
    spec, info = C.build(name, npz, **cfg)
    spec["general"]["n_frames"] = frames
    d = os.path.join(R.LOG, name)
    os.makedirs(d, exist_ok=True)
    info["varied"] = label or {"reference": True}
    json.dump(info, open(os.path.join(d, "pass1.json"), "w"), indent=1)
    R.run(name, spec, device=dev, movie=True, render_kw={"movie_frames": 150, "fps": 15})

    bt = np.asarray(membrane_ops.BOND_TRACE, float)
    z = np.load(os.path.join(d, "traj.npz"))
    al = np.asarray(z["malive"]) if "malive" in z.files else None
    ms = np.asarray(membrane_ops.MEMBRANE_STRAIN[-1], float)
    if al is not None:
        ms = ms[al]
    ms = np.nan_to_num(ms, nan=-1, posinf=-1, neginf=-1)
    P = np.asarray(z["mpos"])[-1][al] if al is not None else np.asarray(z["mpos"])[-1]
    u = P - P.mean(0)
    u /= np.linalg.norm(u, axis=1)[:, None]
    th = np.arccos(np.clip(u[:, 2], -1, 1))
    ph = np.arctan2(u[:, 1], u[:, 0])
    bi = (np.clip((th / np.pi * 16).astype(int), 0, 15) * 32
          + np.clip(((ph + np.pi) / (2 * np.pi) * 32).astype(int), 0, 31))
    # FROM THE TISSUE FILE, not the module global: the global is empty whenever pass 1 came from cache.
    _tz = np.load(npz)
    t1 = np.asarray(_tz["t1_trace"], float) if "t1_trace" in _tz.files else np.zeros((1, 4))
    if t1.size == 0:
        t1 = np.zeros((1, 4))
    info["result"] = dict(bonds_start=int(bt[0, 0]), bonds_end=int(bt[-1, 0]),
                          lcc_end=float(bt[-1, 3]) if bt.shape[1] > 3 else None,
                          mean_degree_z=float(bt[-1, 4]) if bt.shape[1] > 4 else None,
                          t1_total=int(t1[-1, 2]) if len(t1) else 0,
                          t1_per_cell_per_frame=float(t1[:, 1].sum() / max(t1[:, 3].mean(), 1)
                                                      / max(len(t1), 1)),
                          n_alive=int(al.sum()) if al is not None else None,
                          strain_end=float(ms.mean()),
                          strain_p99=float(np.percentile(ms, 99)),
                          coverage=len(np.unique(bi)) / (16 * 32))
    json.dump(info, open(os.path.join(d, "pass1.json"), "w"), indent=1)
    print("SERIES " + json.dumps({"name": name, **info["result"]}), flush=True)


if __name__ == "__main__":
    main()
