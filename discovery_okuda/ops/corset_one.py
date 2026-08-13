"""The corset, run as two stages, because one-way coupling admits no other route.

Stage A runs the membrane with anisotropic crosslinks and records its hoop tension by latitude.
Stage B rebuilds the epithelium with growth gated on that recording, and measures the shape.

WHY TWO STAGES. The sheet cannot push the tissue: the epithelium is a replay, and the matrix and the
membrane both read it without writing to it. So the only way a membrane property can reach the tissue is
the way the ovoid already does it -- record a map in one pass, rebuild the other pass gated on it.
Runs 96-99 skipped stage B, recorded the tension and had nowhere to send it, and came back identical.

THE PREDICTION, and it has the opposite sign to every shape result so far. A polar-dense MATRIX resists
at the poles and gives an oblate tissue, aspect r_eq/r_ax = 1.43. A CORSET resists at the equator, so it
should give a prolate one, aspect < 1. Same gate, same pipeline, opposite outcome -- so if a corset map
produces an oblate tissue, the gate is not reading what we believe it reads.
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "discovery_okuda", "ops"),
          os.path.join(_ROOT, "discovery_okuda")):
    sys.path.insert(0, p)
import aniso
import combine as C
import membrane_ops
import run_ecm as R
import tissue as TIS

CONFIGS = {
    "102_corset_off":      dict(aniso=1.0),
    "103_corset_x3":       dict(aniso=3.0),
    "104_corset_x10":      dict(aniso=10.0),
    "105_corset_reversed": dict(aniso=0.1),
    # the competing explanation: elongation generated INSIDE the epithelium by tension-keyed myosin,
    # with an isotropic membrane. If this elongates as much as 104, the corset is not necessary.
    "106_polarised_myosin": dict(aniso=1.0, myo=dict(myo_beta=2.0, myo_keyed_on="tension")),
    "107_corset_and_myosin": dict(aniso=10.0, myo=dict(myo_beta=2.0, myo_keyed_on="tension")),
}


def main():
    name = sys.argv[1]
    dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 402
    cfg = dict(CONFIGS[name])
    an = cfg.pop("aniso")
    myo = cfg.pop("myo", None)

    # ---- stage A: the sheet, recording its hoop tension --------------------------------------------
    for t in (membrane_ops.BOND_TRACE, membrane_ops.MEMBRANE_STRAIN, membrane_ops.SECRETE_TRACE,
              membrane_ops.BOND_SNAPSHOTS, membrane_ops.HOOP_TRACE):
        t.clear()
    tk = dict(frames=401, device=dev, buffer_x=4, myosin=1.0)
    if myo:
        tk.update(myo)
        tk["tag_extra"] = "_" + "_".join(f"{k}{v}" for k, v in sorted(myo.items()))
    npz = TIS.load_or_build(**tk)
    base = dict(aniso.BASE)
    base.update(membrane=npz, membrane_particles=45000, membrane_cutoff=0.008, membrane_break=0.35,
                membrane_bond_k=5.0e3, membrane_adhesion=1.0e4, membrane_tau=60.0,
                membrane_jitter=0.35, membrane_reserve=12.5, membrane_secrete_rate=0.012,
                membrane_impl="graph", membrane_aniso=an, membrane_record_hoop=True)
    spec, info = C.build(name, npz, **base)
    spec["general"]["n_frames"] = frames
    if frames < 100:
        name = "_smoke_" + name          # same guard as series_one: never share a folder with a real run
        print(f"[corset] {frames} frames -> writing to {name}", flush=True)
    # WIPED BEFORE WRITING. A relaunch that leaves the previous attempt's files behind gives a folder
    # whose contents came from two different runs, and no way to tell which is which -- 79 held a
    # 24-frame movie from a smoke test beside nothing else, and 77 held one beside a 402-frame
    # trajectory. Either the folder is this run's output or it is empty.
    d = os.path.join(R.LOG, name)
    if os.path.isdir(d):
        import shutil
        shutil.rmtree(d, ignore_errors=True)
        print(f"[corset] cleared {d} before writing", flush=True)
    os.makedirs(d, exist_ok=True)
    R.run(name, spec, device=dev, movie=False, render_kw={"strip_only": True})

    hoop = os.path.join(d, "hoop.npz")
    if not os.path.exists(hoop):
        raise RuntimeError(f"{name}: stage A produced no hoop.npz -- with nothing to gate on, stage B "
                           f"would silently reproduce the ungated tissue and look like a null result.")

    # ---- stage B: the epithelium, gated on that ----------------------------------------------------
    tk2 = dict(tk)
    tk2.update(gate_npz=hoop, gate_p_half="auto", gate_hill=6.0, gate_floor=0.08,
               gate_smooth_frames=25, gate_smooth_phi=360.0,
               tag_extra=(tk.get("tag_extra", "") + f"_corset{an:g}"))
    npz2 = TIS.load_or_build(**tk2)
    z = np.load(npz2)
    eq, ax = float(np.asarray(z["r_eq"])[-1]), float(np.asarray(z["r_ax"])[-1])
    hp = np.asarray(np.load(hoop)["pmap"], float)
    pole = float(hp[-1, :4, :].mean()); equ = float(hp[-1, 14:18, :].mean())
    out = dict(name=name, aniso=an, myosin=bool(myo), r_eq=eq, r_ax=ax, aspect=eq / max(ax, 1e-9),
               cells=int(np.asarray(z["n_cells"])[-1]),
               hoop_pole=pole, hoop_equator=equ, hoop_ratio=equ / max(pole, 1e-12))
    info["corset"] = out
    json.dump(info, open(os.path.join(d, "pass1.json"), "w"), indent=1)
    print("CORSET " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
