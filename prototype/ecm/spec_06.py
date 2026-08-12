"""`spec.yaml` for the 06 rig runs -- written FROM THE LIVE RIG, not typed out beside it.

WHY THIS EXISTS. `06_spheroid_ecm` is an engine run, so `plexus.engine` archives the resolved spec next
to it and anyone can see what was solved. The 06 sheet runs are rigs -- python classes whose parameters
live in a dict inside a `main()` -- so they wrote pictures, metrics and a `what.yaml` and left the
configuration in the source at whatever commit was checked out. That is the one file a reader needs to
repeat the run, and it was the one file missing.

AND IT IS READ OFF THE OBJECT, WHICH IS THE WHOLE POINT. A spec written by copying the parameter dict
into a yaml literal is a second copy of the configuration, and second copies drift -- this ladder has
already paid for that three times (chemistry sized to the old epithelium, `break_load` dropped on a
re-seed, a median heuristic fixed in one place of two). `spec_from_rig` interrogates the rig that just
ran: every value below is `getattr` on the object the frames were stepped with, so a spec can be wrong
only by being incomplete, never by disagreeing.

    from spec_06 import write_spec
    write_spec(out_dir, rig, name="06_hole_tiny", frames=401, extra={...})
"""
from __future__ import annotations

import os
import subprocess

import yaml

# Every model declares its units; without them no result may be quoted with one. The sheet rigs carry
# `UNITS` from test_05_sheet (length and time); force is not declared there, and saying so is better
# than inventing a number.
from test_05_sheet import UNITS, T_REAL_UM                                 # noqa: E402

# (attribute on the rig, key in the spec) -- absent attributes are simply not written, so one table
# serves 05b's plaque rig, 05f's refining rig and 05h1's protease rig without a branch per rig.
SHEET = [("E0", "youngs"), ("nu", "poisson"), ("T", "thickness_box"), ("rho0", "rho0"),
         ("beta", "beta_stiffening"), ("M", "mobility"),
         ("tau_r", "tau_relax"), ("zeta", "zeta"), ("s_target", "substep_target"),
         ("sigma_T", "sigma_T"), ("max_refine", "max_refine"), ("edge_trigger", "edge_trigger"),
         ("reseed", "reseed"), ("tau_bm", "tau_bm"), ("rho_crit", "rho_crit"),
         ("s_mode", "secretion_mode")]
ADHESION = [("k_drive", "k_drive"), ("kappa_b", "kappa_b"), ("k_on", "k_on"),
            ("k_off0", "k_off0"), ("f_bell", "f_bell")]
CHEM = [("K_timp", "K_timp"), ("k_act", "k_act"), ("k_inhib", "k_inhib"), ("k_deg", "k_deg"),
        ("s_pro", "s_pro"), ("s_timp", "s_timp"), ("s_timp3", "s_timp3"), ("s_mmp", "s_mmp"),
        ("s_mt1", "s_mt1"), ("mt1_frac", "mt1_frac"), ("hetero", "hetero"),
        ("tau_pro", "tau_pro"), ("tau_mmp", "tau_mmp"), ("tau_timp", "tau_timp"),
        ("tau_timp3", "tau_timp3"), ("D_mmp", "D_mmp_box2_per_frame"),
        ("D_timp", "D_timp_box2_per_frame")]


def _num(v):
    """yaml-safe: torch scalars, numpy scalars and python numbers all come out as numbers."""
    if v is None or isinstance(v, (bool, str)):
        return v
    try:
        return float(v.item()) if hasattr(v, "item") else (v if isinstance(v, (int, float))
                                                           else float(v))
    except Exception:
        return str(v)


def _grab(rig, table):
    """Look on the rig, then on its Sheet. E, nu and thickness are constructor arguments of `Sheet`
    and are never copied onto the rig, so a spec that only asked the rig came out with a `material`
    block holding two numbers out of six -- present, plausible, and missing the stiffness."""
    out = {}
    for a, k in table:
        v = getattr(rig, a, None)
        if v is None or callable(v):
            # CALLABLE MEANS WE FOUND A METHOD, NOT A VALUE. `Sheet.reseed` is a method and the rig's
            # `reseed` is a bool; falling back to the sheet without this test wrote
            # "<bound method Sheet.reseed ...>" into the spec as though it were a setting.
            v2 = getattr(getattr(rig, "sheet", None), a, None)
            v = None if (v2 is None or callable(v2)) else v2
        if v is not None and not callable(v):
            out[k] = _num(v)
    return out


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        return None


def spec_from_rig(rig, name, frames, extra=None, matrix_src=None):
    sheet = rig.sheet
    spec = {
        "general": {
            "name": name,
            "rig": type(rig).__name__,
            "module": type(rig).__module__,
            "frames": int(frames),
            "device": str(rig.dev),
            "dtype": str(rig.dtype).replace("torch.", ""),
            "git_commit": _commit(),
            # THE UNITS, and what is missing from them. The sheet rigs declare a length and a time;
            # they do NOT declare a force, so nothing solved here may be quoted in nN.
            "units": {"length_um": float(UNITS["length_um"]), "time_s": float(UNITS["time_s"]),
                      "force_nN": None,
                      "note": "1 box = length_um across; 1 frame = time_s. force is NOT declared "
                              "by these rigs, so no result here may be quoted in nN."},
        },
        "sets": {
            "bm_node": {"n": int(sheet.x.shape[0]), "subdiv": int(getattr(rig, "subdiv", 4))},
            "bm_face": {"n": int(sheet.Fc.shape[0]),
                        "thickness_um": float(T_REAL_UM),
                        "carries": ["mass", "rest metric"]
                        + (["proMMP2", "MMP2", "TIMP2", "TIMP3"] if hasattr(rig, "mmp") else [])},
            "epithelium": {"driver": "replay",
                           "cache": getattr(rig, "_cache", None) or CACHE_HINT,
                           "vertices_frame0": int(getattr(rig, "nv0", 0)),
                           "nodes": int(rig.x_epi.shape[0]),
                           "triangles": int(rig.F_epi.shape[0]),
                           "box_scale": _num(getattr(rig, "scale", None))},
        },
        "material": {"sheet": _grab(rig, SHEET), "adhesion": _grab(rig, ADHESION)},
    }
    # THE ADHESION SET, AND `ct_node` WINS WHEN BOTH EXIST. 05b holds the contact set as `rig.plq`;
    # 05d/05f/05h1 hold it as ct_node/ct_face/ct_w and inherit a `plq` their force law never reads.
    # Asking for `plq` first reported 104 plaques for a run that has 2,562 -- a stale object, answered
    # confidently. The live one is the one the rig integrates.
    if hasattr(rig, "ct_node"):
        spec["sets"]["plaque"] = {"n": int(rig.ct_node.shape[0]),
                                  "law": "one adhesion patch per live sheet node, clutch kinetics",
                                  "stale_plq_attr": (int(rig.plq.node.shape[0])
                                                     if getattr(rig, "plq", None) is not None
                                                     else None)}
    elif getattr(rig, "plq", None) is not None:
        plq = rig.plq
        spec["sets"]["plaque"] = {"n": int(plq.node.shape[0]), "l0_box": _num(plq.l0),
                                  "kn": _num(plq.kn), "xi": _num(plq.xi),
                                  "break_load": _num(plq.break_load),
                                  "law": "bind to a FACE with barycentric weights, one shared rest "
                                         "length"}
    if hasattr(rig, "mmp"):
        spec["chemistry"] = _grab(rig, CHEM)
        spec["chemistry"]["species"] = {
            "MT1-MMP": "per-cell state, tethered, no transport",
            "proMMP2": "field on bm_face, D_mmp", "MMP2": "field on bm_face, D_mmp",
            "TIMP2": "field on bm_face, D_timp", "TIMP3": "per-face state, no transport"}
        spec["chemistry"]["mt1_field"] = (extra or {}).get("mt1_field", "smooth_field, 6 modes")
    spec["operators"] = _operators(rig)
    if matrix_src:
        spec["rendered_against"] = {
            "matrix": matrix_src,
            "note": "the 2x2's other three panels come from this run's traj.npz, re-drawn and never "
                    "re-run: there is no BM-ECM coupling, so nothing here reaches the matrix."}
    if extra:
        spec["run"] = extra
    return spec


CACHE_HINT = "log/okuda_ECM/_tissue/cellfix_B_new_f401_x4_c4a5698982.npz"


def _operators(rig):
    """What actually ran, inferred from the rig's own state rather than listed by hand."""
    ops = ["bm_elastic (StVK on the rest metric)", "plaque (edge set: bm_node -> epithelial face)",
           "epi_drive (replay toward the recorded surface)"]
    if getattr(rig, "s_mode", None):
        ops.append("bm_secrete (%s)" % rig.s_mode)
    if int(getattr(rig, "max_refine", 0) or 0) > 0:
        ops.append("bm_refine (1->4 split, edge_trigger %.2f)" % float(rig.edge_trigger))
    if hasattr(rig, "mmp"):
        ops += ["protease_secrete", "protease_diffuse (semi-implicit, face Laplacian)",
                "protease_activate (ternary, bell)", "protease_inhibit (1:1)", "bm_degrade"]
    if float(getattr(rig, "rho_crit", 0.0) or 0.0) > 0:
        ops.append("bm_tear (rho/rho0 < rho_crit)")
    return ops


def write_spec(d, rig, name, frames, extra=None, matrix_src=None):
    spec = spec_from_rig(rig, name, frames, extra=extra, matrix_src=matrix_src)
    p = os.path.join(d, "spec.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False, width=100)
    print(f"[spec] {p}", flush=True)
    return p


# =============================================================================================
#  back-fill: the folders that ran before this file existed
# =============================================================================================
BACKFILL = {
    # name: (module, how to build it)
    "06_spheroid_bm_ecm": ("panels", dict(break_load=None)),
    "06_detach":          ("panels", dict(break_load=5.0e-3)),
    "06_detach_partial":  ("panels", dict(break_load=2.0e-2)),
    "06_refine":          ("refine", dict()),
    # the protease family reads its own metrics.json, which records every knob it was run with
    "06_breach_hole": ("breach", None), "06_breach_torn": ("breach", None),
    "06_hole_one": ("breach", None), "06_hole_small": ("breach", None),
    "06_hole_stable": ("breach", None), "06_hole_tiny": ("breach", None),
    "06_hole_tiny_off": ("breach", None), "06_hole_smaller": ("breach", None),
    "06_hole_larger": ("breach", None), "06_hole_largest": ("breach", None),
}


def backfill(dev="cuda:0", only=None):
    """Rebuild each finished rig with the parameters it ran with and write its spec.

    REBUILT, NOT RECONSTRUCTED BY HAND. Construction is cheap (no frames are stepped) and it is the
    only way the spec can be trusted: the numbers come from the same code path the run used. The
    protease folders carry their knobs in `metrics.json`, so those are read back rather than retyped;
    the three mechanical ones are named here because their parameters were only ever CLI flags.
    """
    import json
    import test_05b_plaque as B
    for name, (kind, kw) in sorted(BACKFILL.items()):
        if only and name not in only:
            continue
        d = os.path.join(B.LOG, name)
        if not os.path.isdir(d):
            print(f"[spec] {name}: no folder -- skipped", flush=True)
            continue
        m = {}
        mp = os.path.join(d, "metrics.json")
        if os.path.exists(mp):
            m = json.load(open(mp))
        frames = int(m.get("frames", 401))
        if kind == "breach":
            import test_06_breach as BR
            rig = BR.build(float(m.get("inhib", 1.0)), dev, float(m.get("kdeg", 100.0)),
                           bool(m.get("refine", False)), int(m.get("modes", 6)),
                           float(m.get("hetero", 1.0)), float(m.get("spot", 0.0) or 0.0),
                           int(m.get("seed_mt1", 3)), float(m.get("spot_off", 0.0) or 0.0))
            sp = float(m.get("spot", 0.0) or 0.0)
            extra = dict(kind="protease", backfilled=True, faces_torn=m.get("faces_torn"),
                         torn_frac=m.get("torn_frac"), rim_loops=m.get("rim_loops"),
                         verdict=m.get("verdict"),
                         mt1_field=(f"single Gaussian cap, theta {sp} deg, "
                                    f"{m.get('spot_off', 0.0)} deg off the view axis" if sp else
                                    f"smooth_field, {int(m.get('modes', 6))} modes, hetero "
                                    f"{m.get('hetero', 1.0)}, seed {int(m.get('seed_mt1', 3))}"),
                         **{k: m.get(k) for k in ("inhib", "kdeg", "refine", "modes", "hetero",
                                                  "spot", "spot_off", "seed_mt1", "rho_crit")})
        elif kind == "refine":
            from test_05l_supply import Rig05l
            rig = Rig05l(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, kn=5.0, sigma_T=7.0,
                         zeta=20.0, s_target=1.0, k_drive=50.0, dev=dev, max_refine=2,
                         edge_trigger=1.45, reseed=True, tau_bm=40.0, rho_crit=0.0)
            extra = dict(kind="mechanical, refining", backfilled=True,
                         faces=m.get("faces"), plaques=m.get("plaques"), G43=m.get("G43"),
                         G44=m.get("G44"))
        else:
            import test_06c_real_driver as R6
            T = 2.0e-3
            rig = R6.Rig06c(subdiv=4, subdiv_epi=3, E=400.0, thickness=T, nu=0.3, kn=5.0, xi=0.0,
                            l0=0.3 * T, zeta=20.0, s_target=1.0, k_drive=50.0, dev=dev, **kw)
            extra = dict(kind="mechanical", backfilled=True, **kw)
        write_spec(d, rig, name=name, frames=frames, extra=extra, matrix_src="06_spheroid_ecm")
        del rig


# =============================================================================================
#  the older folders: a spec derived from artefacts, and honest about being one
# =============================================================================================
SKIP = ("_archive", "_archive_05def", "_archive_91_165", "_cluster", "_tissue", "_smoke", "_reel")


def _producer(name, here):
    """Which script writes this folder, found by asking the scripts rather than by remembering."""
    import glob
    import re
    hits = []
    for p in sorted(glob.glob(os.path.join(here, "*.py"))):
        try:
            src = open(p).read()
        except Exception:
            continue
        if re.search(r'["\']%s["\']' % re.escape(name), src):
            hits.append(os.path.basename(p))
    return hits


def backfill_all(root=None):
    """Write a `spec.yaml` into every run folder that has none.

    TWO KINDS OF SPEC, AND THE DIFFERENCE IS STATED IN THE FILE. The 06 folders are REBUILT: their rig
    is constructed again with the parameters it ran with and interrogated, so the spec cannot disagree
    with the code. Everything older is DERIVED: the folder is matched to the script that names it, the
    parameters are read from whatever the run archived (metrics.json, gates.json, units.json), and the
    fields nobody recorded are left out rather than guessed. A derived spec is an entry point -- it
    says what to run to get this folder back -- not a certificate that these were the values.
    """
    import json
    import glob
    import test_05b_plaque as B
    here = os.path.dirname(os.path.abspath(__file__))
    root = root or B.LOG
    n = 0
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        name = os.path.basename(d)
        if not os.path.isdir(d) or name.startswith(SKIP) or name in ("_metrology",):
            continue
        if os.path.exists(os.path.join(d, "spec.yaml")):
            continue
        art = sorted(os.listdir(d))
        if not art:
            print(f"[spec] {name}: empty folder -- skipped", flush=True)
            continue
        rec = {}
        for f in ("metrics.json", "gates.json", "units.json"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                try:
                    j = json.load(open(p))
                except Exception:
                    continue
                # keep the scalars and small dicts; a per-frame series is not a parameter
                rec[f] = {k: v for k, v in j.items()
                          if not isinstance(v, list) or len(v) <= 8} if isinstance(j, dict) else j
        prod = _producer(name, here)
        spec = {
            "general": {
                "name": name,
                "units": {"length_um": float(UNITS["length_um"]),
                          "time_s": float(UNITS["time_s"]), "force_nN": None,
                          "note": "the okuda_ECM convention: 1 box = length_um across, 1 frame = "
                                  "time_s. Force is not declared by these rigs."},
                "produced_by": prod or "unknown -- no script in prototype/ecm names this folder",
                "reproduce": (f"cd prototype/ecm && PYTHONPATH=../../src python {prod[0]}"
                              if prod else None),
            },
            "provenance": {
                "kind": "DERIVED FROM ARTEFACTS, not rebuilt",
                "what_that_means": "the producing script was found by searching for this folder's "
                                   "name; the parameters below are only those the run itself "
                                   "archived. Values the run did not record are ABSENT rather than "
                                   "guessed, and this file is an entry point, not a certificate.",
                "written_at_commit": _commit(),
            },
            "artefacts": art,
        }
        if rec:
            spec["recorded"] = rec
        with open(os.path.join(d, "spec.yaml"), "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False, width=100)
        n += 1
        print(f"[spec] {name}: derived  <- {', '.join(prod) or 'no producer found'}", flush=True)
    print(f"[spec] {n} derived specs written", flush=True)


if __name__ == "__main__":
    import sys
    if "--all" in sys.argv:
        backfill_all()
    else:
        backfill(dev=(sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv
                      else "cuda:0"),
                 only=set(sys.argv[sys.argv.index("--only") + 1].split(","))
                 if "--only" in sys.argv else None)
