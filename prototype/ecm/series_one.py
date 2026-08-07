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
    # x10 and x20 the reference RATE. A prediction first, so the runs can refute it: these should look
    # like 69. The rate sweep put the supply/demand knee at 0.0064 per frame and the reference is
    # already 0.012, twice above it -- past the knee the particle count is set by AREAL DEMAND
    # (want = n0 (R/R0)^2), not by supply, so pouring in ten times the material faster cannot make a
    # thicker sheet, only reach the same one sooner. If 76 and 77 differ from 69 by more than the run
    # to run scatter, the demand-limited picture is wrong.
    "76_secrete_x10": dict(membrane_secrete_rate=0.12),
    "77_secrete_x20": dict(membrane_secrete_rate=0.24),
    # --- the junction twin: the same treatment for the OTHER new level -------------------------------
    # These vary myosin, not the membrane, so the panel to read is the junction network (bottom right).
    # `activity` is the blebbistatin knob and is degenerate with Lambda by construction; `beta` is the
    # tension feedback and is NOT -- it moves junction-length disorder at constant tissue size, which no
    # uniform tension scale can do. `myo_new` is what a junction born from a division starts with.
    "78_myo_reference": dict(_myo=dict(myosin=1.0, myo_beta=1.0, myo_tau=20.0)),
    "79_blebbistatin": dict(_myo=dict(myosin=0.3, myo_beta=1.0, myo_tau=20.0)),
    "80_hypercontractile": dict(_myo=dict(myosin=2.0, myo_beta=1.0, myo_tau=20.0)),
    "81_no_feedback": dict(_myo=dict(myosin=1.0, myo_beta=0.0, myo_tau=20.0)),
    "82_strong_feedback": dict(_myo=dict(myosin=1.0, myo_beta=4.0, myo_tau=20.0)),
    "83_weak_newborn": dict(_myo=dict(myosin=1.0, myo_beta=1.0, myo_tau=20.0, myo_new=0.3)),

    # --- v2: THE SAME SEVEN, RE-VERIFIED OVERDAMPED --------------------------------------------------
    # Identical configurations to 69-75, under NEW NAMES on purpose. Re-running into the original folders
    # is what destroyed their results once already: a relaunch was killed mid-flight and overwrote
    # pass1.json before writing its own, so the reference the comparison needs was gone (recovered only
    # because the cluster .out files still held the SERIES lines). A regression check that can destroy
    # its own baseline is not a check.
    #
    # These are not a pure regression test either: `membrane_inertial=False` is now the default, so the
    # sheet is integrated overdamped rather than given a mass. Bond counts, node counts and coverage
    # should track the originals closely -- that is the "nothing broke" half. Strain and any sinking
    # should MOVE -- that is the fix doing something.
    "69v2_graph_reference": dict(),
    "70v2_no_remodelling": dict(membrane_tau=0.0),
    "71v2_fast_remodelling": dict(membrane_tau=10.0),
    "72v2_starved": dict(membrane_secrete_rate=0.002),
    "73v2_no_adhesion": dict(membrane_adhesion=0.0),
    "74v2_brittle": dict(membrane_break=0.08),
    "75v2_on_ovoid": dict(_gated=True),

    # --- 84-95: WHAT MAKES THE HOLES ----------------------------------------------------------------
    # An external reviewer: the holes in the reference sheet are too big -- real microperforations are
    # ~1 um across, against a 5-10 um cell footprint, so a hole should be about an eighth of a cell and
    # ours are about two and a half. Measured, the sheet is NOT at its resolution limit: the largest gaps
    # are 6.1 node spacings where a well-packed sheet gives ~1.5. And the seeded sheet is fine (clearance
    # 1.10 spacings, better than uniform random at 1.53) -- it degrades to 3.08 during the run. The holes
    # are made by the deposition rule, which places a node beside a random parent rather than into the
    # gap, and a random walk clumps. A relaxation pass on the new material takes 4.4 back to 1.75.
    #
    # So two things could be setting hole size and they are confounded: PACKING (relaxation) and
    # RESOLUTION (node count). This grid separates them, at 45k/135k/270k nodes x relaxation off/on.
    #
    # And the same grid again WITHOUT ADHESION, because the reviewer's second point is a different
    # mechanism that currently looks identical: unanchored, the sheet does not clump, it SLIDES OFF as a
    # whole (coverage 1.00 -> 0.895), and no amount of relaxation or resolution addresses that. If the
    # no-adhesion holes shrink with relaxation and nodes they were packing; if they do not, the sheet is
    # drifting and the missing piece is an attachment to the stroma.
    "84_holes_45k_r0": dict(membrane_particles=45000, membrane_reserve=13.5, membrane_relax_new=0),
    "85_holes_45k_r8": dict(membrane_particles=45000, membrane_reserve=13.5, membrane_relax_new=8),
    "86_holes_135k_r0": dict(membrane_particles=135000, membrane_reserve=40.5, membrane_relax_new=0),
    "87_holes_135k_r8": dict(membrane_particles=135000, membrane_reserve=40.5, membrane_relax_new=8),
    "88_holes_270k_r0": dict(membrane_particles=270000, membrane_reserve=81.0, membrane_relax_new=0),
    "89_holes_270k_r8": dict(membrane_particles=270000, membrane_reserve=81.0, membrane_relax_new=8),
    "90_holes_45k_r0_noadh": dict(membrane_particles=45000, membrane_reserve=13.5, membrane_relax_new=0, membrane_adhesion=0.0),
    "91_holes_45k_r8_noadh": dict(membrane_particles=45000, membrane_reserve=13.5, membrane_relax_new=8, membrane_adhesion=0.0),
    "92_holes_135k_r0_noadh": dict(membrane_particles=135000, membrane_reserve=40.5, membrane_relax_new=0, membrane_adhesion=0.0),
    "93_holes_135k_r8_noadh": dict(membrane_particles=135000, membrane_reserve=40.5, membrane_relax_new=8, membrane_adhesion=0.0),
    "94_holes_270k_r0_noadh": dict(membrane_particles=270000, membrane_reserve=81.0, membrane_relax_new=0, membrane_adhesion=0.0),
    "95_holes_270k_r8_noadh": dict(membrane_particles=270000, membrane_reserve=81.0, membrane_relax_new=8, membrane_adhesion=0.0),

    # --- 84-91: WHICH VARIABLE THE MYOSIN FEEDBACK IS KEYED TO ---------------------------------------
    # The audit's charge: keyed to LENGTH the feedback homogenises junction lengths, where the measured
    # biology (Bertet 2004, Fernandez-Gonzalez 2009) has myosin recruited by TENSION and enriched on
    # DISASSEMBLING junctions -- a positive feedback that GENERATES T1s. And in the fast-myosin limit the
    # length law is analytically a harmonic edge spring, so "beta lowers junction-length CV" is its
    # definition, not evidence. The discriminating observable is the T1 RATE, which the two laws move in
    # OPPOSITE directions: length-keyed should suppress T1s, tension-keyed should produce them.
    "84_key_length_b0": dict(_myo=dict(myosin=1.0, myo_beta=0.0)),
    "85_key_length_b2": dict(_myo=dict(myosin=1.0, myo_beta=2.0)),
    "86_key_tension_b2": dict(_myo=dict(myosin=1.0, myo_beta=2.0, myo_keyed_on="tension")),
    "87_key_tension_b4": dict(_myo=dict(myosin=1.0, myo_beta=4.0, myo_keyed_on="tension")),
    "88_key_tension_stab": dict(_myo=dict(myosin=1.0, myo_beta=2.0, myo_keyed_on="tension",
                                          myo_destabilising=0)),
    "89_key_strainrate_b2": dict(_myo=dict(myosin=1.0, myo_beta=2.0, myo_keyed_on="strain_rate")),
    "90_key_strainrate_b4": dict(_myo=dict(myosin=1.0, myo_beta=4.0, myo_keyed_on="strain_rate")),
    # the control the CV argument never had: a uniform tension scale matched to 87's mean myosin, so any
    # difference in T1 rate cannot be attributed to overall contractility
    "91_lambda_matched": dict(_myo=dict(myosin=1.16, myo_beta=0.0)),

    # --- 92-100: THE MEMBRANE WITHOUT INERTIA -------------------------------------------------------
    # At Re ~ 1e-10 the equation of motion is gamma*x_dot = F, not m*x_ddot = F. Everything reported
    # about the sheet's dynamics -- oscillation about a moving anchor, critical damping, tracking lag,
    # sinking, and the stability ceiling at k ~ 8e3 -- follows from a mass that should not be there.
    # 92 vs 96 is the direct comparison; 93-95 ask whether the ceiling exists at all overdamped.
    "92_od_k5e3": dict(membrane_bond_k=5.0e3),
    "93_od_k5e4": dict(membrane_bond_k=5.0e4),
    "94_od_k5e5": dict(membrane_bond_k=5.0e5),
    "95_od_k5e6": dict(membrane_bond_k=5.0e6),
    "96_inertial_k5e3": dict(membrane_bond_k=5.0e3, membrane_inertial=True),
    "97_od_no_adhesion": dict(membrane_bond_k=5.0e3, membrane_adhesion=0.0),
    "98_od_no_remodel": dict(membrane_bond_k=5.0e3, membrane_tau=0.0),
    "99_od_brittle": dict(membrane_bond_k=5.0e3, membrane_break=0.08),
    "100_od_on_ovoid": dict(membrane_bond_k=5.0e3, _gated=True),
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
              membrane_ops.SECRETE_TRACE, membrane_ops.BOND_SNAPSHOTS):
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
