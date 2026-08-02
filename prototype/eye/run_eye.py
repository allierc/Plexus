#!/usr/bin/env python
"""run_eye -- run the zebrafish oculomotor spec, score it, and render the movie.

    python run_eye.py --preset probe --label calib --particles 45000
    python run_eye.py --preset atlas --label final --particles 110000

Every run is archived to `archive/tNN_<label>/` (NN auto-increments):

    spec.yaml    the Plexus2 spec that produced it -- the deliverable
    movie.mp4    the six-panel movie
    strip.png    five key frames, for a glance
    curves.npz   the captured traces
    diag.json    the metrics, so "convincing" is a test and not an impression

`trial()` is importable: `sweep_eye.py` calls it once per configuration.
"""
from __future__ import annotations

import os
import sys
import json
import glob
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch

import plexus.operators            # noqa: F401  stock operator library
import eye_ops                     # noqa: F401  eye operators (prototype-local)
import muscle_ops                  # noqa: F401  muscle-as-tissue operators (prototype-local)
import eye_anatomy as EA
import eye_spec as ES
import render_eye
from plexus.schema import load as load_spec
from plexus.engine import run as engine_run

ARCHIVE = os.path.join(HERE, "archive")


# --------------------------------------------------------------------------- #
#  capture: one pass of the sim, snapshotting what the trajectory has no room for
# --------------------------------------------------------------------------- #
def _scalars(p, sel):
    """(Green-Lagrange strain, von Mises stress) for a subset of a particle set -- the two
    continuum fields the generic trajectory has no room for."""
    F = p.F[sel]
    Ft = F.transpose(-2, -1)
    I3 = torch.eye(3, device=F.device)
    E = 0.5 * (Ft @ F - I3.expand_as(F))                      # rotation-free strain
    strain = E.reshape(F.shape[0], -1).norm(dim=1)
    U, S, Vh = torch.linalg.svd(F)                            # polar rotation, as mpm_scatter
    U = U.clone(); Vh = Vh.clone()
    U[torch.det(U) < 0, :, -1] *= -1
    Vh[torch.det(Vh) < 0, -1, :] *= -1
    R = U @ Vh
    J = torch.linalg.det(F)
    sig = 2 * p.mu[sel][:, None, None] * ((F - R) @ Ft) \
        + I3 * (p.la[sel] * J * (J - 1))[:, None, None]
    sig = 0.5 * (sig + sig.transpose(-2, -1))
    dev = sig - I3 * (sig.diagonal(dim1=-2, dim2=-1).sum(-1) / 3)[:, None, None]
    vm = torch.sqrt(1.5 * dev.reshape(F.shape[0], -1).pow(2).sum(1))
    return strain, vm


def capture_run(sim, device, stride=3, n_shell=26000, n_cut=16000, n_mus=14000,
                n_grid_pts=9000, seed=0):
    keys = ("frame", "shell", "cut_pos", "cut_strain", "cut_vm",
            "mus_pos", "mus_strain", "mus_vm", "act", "tension", "length",
            "ins", "pull", "axis", "gaze", "target", "centre", "gpos", "gvel")
    rec = {k: [] for k in keys}
    idx = {}
    rng = np.random.default_rng(seed)
    prog = next(o.params["program"] for o in sim.operators if o.op == "oculomotor_drive")
    cmd = eye_ops.OculomotorDrive({"program": prog}, "cpu")     # evaluates the COMMAND per frame

    def _pick(mask_t, k):
        ii = torch.nonzero(mask_t, as_tuple=False).flatten().cpu().numpy()
        if ii.size > k:
            ii = np.sort(rng.choice(ii, k, replace=False))
        return torch.as_tensor(ii, dtype=torch.long)

    def hook(H, frame):
        if frame % stride and frame != sim.n_frames:
            return
        p = H.levels["mpm_particle"]
        q = H.levels["muscle_particle"]
        if not idx:
            if not hasattr(p, "rest_rn") or not hasattr(q, "s"):
                return
            dev = p.state.device
            idx["shell"] = _pick(p.rest_rn > 0.94, n_shell).to(dev)
            idx["cut"] = _pick(p.rest[:, 0] < 0.012 * EA.A_EQ, n_cut).to(dev)
            idx["mus"] = _pick(torch.ones(q.n, dtype=torch.bool, device=dev), n_mus).to(dev)
            idx["tissue"] = p.tissue[idx["shell"]].cpu().numpy()
            idx["mus_parent"] = q.parent[idx["mus"]].cpu().numpy()
            idx["mus_s"] = q.s[idx["mus"]].cpu().numpy()
        s, cu, mu_i = idx["shell"], idx["cut"], idx["mus"]
        X, Y = p.get("pos"), q.get("pos")
        c_strain, c_vm = _scalars(p, cu)
        m_strain, m_vm = _scalars(q, mu_i)

        m, eye = H.levels["muscle"], H.levels["eye"]
        g = H.fields["mpm_grid"]
        gv = g.v.reshape(*g.shape, 3)
        gmag = gv.norm(dim=-1)
        act_cells = torch.nonzero(gmag > 1e-5, as_tuple=False)
        if act_cells.shape[0] > n_grid_pts:
            keep = torch.randperm(act_cells.shape[0], device=act_cells.device)[:n_grid_pts]
            act_cells = act_cells[keep]
        gp = (act_cells.float() + 0.5) * g.dx
        gm = gmag[act_cells[:, 0], act_cells[:, 1], act_cells[:, 2]]

        f32 = lambda t: t.detach().cpu().numpy().astype(np.float32)
        rec["frame"].append(frame)
        rec["shell"].append(f32(X[s]))
        rec["cut_pos"].append(f32(X[cu]))
        rec["cut_strain"].append(f32(c_strain))
        rec["cut_vm"].append(f32(c_vm))
        rec["mus_pos"].append(f32(Y[mu_i]))
        rec["mus_strain"].append(f32(m_strain))
        rec["mus_vm"].append(f32(m_vm))
        rec["act"].append(f32(m.get("act")[:, 0]))
        rec["tension"].append(f32(m.get("tension")[:, 0]))
        rec["length"].append(f32(m.get("length")[:, 0]))
        rec["ins"].append(f32(m.ins_pos))
        rec["pull"].append(f32(m.pull))
        rec["axis"].append(f32(m.axis))
        rec["gaze"].append(f32(eye.get("gaze")[0]))
        rec["centre"].append(f32(eye.get("pos")[0]))
        rec["target"].append(np.asarray(cmd.target(frame), np.float32))
        rec["gpos"].append(f32(gp))
        rec["gvel"].append(f32(gm))

    H, _ = engine_run(sim, out_path=None, device=device, on_frame=hook, progress=False)
    out = {k: (np.asarray(v) if k not in ("gpos", "gvel") else v) for k, v in rec.items()}
    out["tissue"] = idx["tissue"]
    out["mus_parent"] = idx["mus_parent"]
    out["mus_s"] = idx["mus_s"]
    out["rest_length"] = H.levels["muscle"].rest_length.detach().cpu().numpy()
    out["origins"] = EA.origins_world().astype(np.float32)
    return H, out


# --------------------------------------------------------------------------- #
#  metrics: "convincing" as a test
# --------------------------------------------------------------------------- #
# what the anatomy says SHOULD be recruited for each kind of command; the drive never
# tabulates this -- it projects onto axes computed from the geometry -- so agreement is
# a real check that the plant is wired right.
EXPECTED = {
    "abduction": {"LR"}, "adduction": {"MR"},
    "elevation": {"SR", "IO"}, "depression": {"IR", "SO"},
    "intorsion": {"SO", "SR"}, "extorsion": {"IO", "IR"},
}


def classify(cmd):
    h, v, t = cmd
    if abs(t) > max(abs(h), abs(v)):
        return "intorsion" if t > 0 else "extorsion"
    if abs(v) > abs(h):
        return "elevation" if v > 0 else "depression"
    if abs(h) > 1e-6:
        return "abduction" if h > 0 else "adduction"
    return "primary"


def diagnose(cap, sim):
    g, t, fr, act = cap["gaze"], cap["target"], cap["frame"], cap["act"]
    prog = np.asarray(next(o.params["program"] for o in sim.operators
                           if o.op == "oculomotor_drive"), float)
    holds, ok_recruit, n_recruit = [], 0, 0
    for i in range(len(prog)):
        f0 = prog[i, 0]
        f1 = prog[i + 1, 0] if i + 1 < len(prog) else sim.n_frames
        if f1 - f0 < 30:
            continue
        sel = (fr >= f0 + 0.6 * (f1 - f0)) & (fr <= f1)
        if sel.sum() < 2:
            continue
        cmd = prog[i, 1:4]
        got = g[sel].mean(0)
        a = act[sel].mean(0)
        kind = classify(cmd)
        top = [EA.MUSCLE_KEYS[j] for j in np.argsort(-a)[:2]]
        good = None
        if kind in EXPECTED:
            n_recruit += 1
            good = bool(set(top) & EXPECTED[kind]) and top[0] in EXPECTED[kind]
            ok_recruit += int(good)
        holds.append({
            "frames": [int(f0), int(f1)], "kind": kind,
            "command_hvt": [round(float(x), 2) for x in cmd],
            "achieved_hvt": [round(float(x), 2) for x in got],
            "error_deg": round(float(np.linalg.norm(got - cmd)), 2),
            "recruited": top, "recruit_ok": good,
            "activation": {EA.MUSCLE_KEYS[j]: round(float(a[j]), 3) for j in range(EA.N_MUSCLE)},
        })
    tm = render_eye.tracking_metrics(cap)
    c = cap["centre"]
    drift = np.linalg.norm(c - c[0], axis=1)
    errs = [h["error_deg"] for h in holds]
    L, L0 = cap["length"], cap["rest_length"][None, :]
    shorten = 100.0 * (1.0 - L / L0)                       # % shortening per muscle over time
    return {
        "n_frames": int(sim.n_frames),
        "range_hvt_deg": [round(float(np.ptp(g[:, k])), 2) for k in range(3)],
        "mean_settle_error_deg": round(float(np.mean(errs)), 2) if errs else None,
        "tracking_rms_deg": [round(float(x), 2) for x in tm["rms_deg"]],
        "tracking_settled_rms_deg": [round(float(x), 2) for x in tm["settled_rms_deg"]],
        "max_settle_error_deg": round(float(np.max(errs)), 2) if errs else None,
        "recruitment_correct": f"{ok_recruit}/{n_recruit}" if n_recruit else "n/a",
        "centroid_drift_max_frac_radius": round(float(drift.max() / EA.A_EQ), 4),
        "strain_p99": round(float(np.percentile(cap["cut_strain"], 99)), 4),
        "strain_max": round(float(cap["cut_strain"].max()), 4),
        "vonmises_p99": round(float(np.percentile(cap["cut_vm"], 99)), 3),
        "activation_range": [round(float(cap["act"].min()), 3), round(float(cap["act"].max()), 3)],
        "max_shortening_pct": {EA.MUSCLE_KEYS[j]: round(float(shorten[:, j].max()), 2)
                               for j in range(EA.N_MUSCLE)},
        "peak_shortening_pct": round(float(shorten.max()), 2),
        "muscle_strain_p99": round(float(np.percentile(cap["mus_strain"], 99)), 4),
        "holds": holds,
    }


def verdict(d):
    """The acceptance test. A run is convincing when the eye reaches its commands, the
    right muscles do it, the globe stays in its socket, and it deforms a LITTLE."""
    checks = {
        "reaches_commands": (d["max_settle_error_deg"] is not None
                             and d["max_settle_error_deg"] < 6.0),
        "torsion_demonstrated": d["range_hvt_deg"][2] > 6.0,
        "wide_gaze_range": d["range_hvt_deg"][0] > 35.0 and d["range_hvt_deg"][1] > 20.0,
        "correct_recruitment": (d["recruitment_correct"] != "n/a"
                                and d["recruitment_correct"].split("/")[0]
                                == d["recruitment_correct"].split("/")[1]),
        "stays_in_socket": d["centroid_drift_max_frac_radius"] < 0.06,
        "deformable_not_floppy": 0.004 < d["strain_p99"] < 0.12,
        "muscles_contract": 4.0 < d["peak_shortening_pct"] < 38.0,   # below 38%: not buckled
    }
    return checks, all(checks.values())


# --------------------------------------------------------------------------- #
def next_archive_dir(label):
    os.makedirs(ARCHIVE, exist_ok=True)
    used = [int(os.path.basename(d)[1:3])
            for d in glob.glob(os.path.join(ARCHIVE, "t[0-9][0-9]_*"))
            if os.path.basename(d)[1:3].isdigit()]
    n = (max(used) + 1) if used else 1
    d = os.path.join(ARCHIVE, f"t{n:02d}_{label}")
    os.makedirs(d, exist_ok=True)
    return d


def trial(label, device="cuda:0", stride=3, movie=True, note="", **kw):
    """One archived trial: build the spec -> run -> score -> render. Returns (dir, diag)."""
    preset = kw.pop("preset", "atlas")
    spec = ES.build_spec(name=f"eye_{preset}_{label}", preset=preset, **kw)
    limit = ES.cfl_limit(spec)
    sub = kw.get("substep_dt") or min(1.2e-4, limit * 0.95)
    if sub > limit:
        sub = limit * 0.95
    spec["schedule"][-1]["substep_dt"] = float(f"{sub:.3e}")

    outdir = next_archive_dir(label)
    spec_path = ES.write_spec(spec, os.path.join(outdir, "spec.yaml"))
    sim = load_spec(spec_path)
    print(f"[trial] {os.path.basename(outdir)}  preset={preset}  N={sim.sets['mpm_particle']['per_parent']} "
          f"frames={sim.n_frames}  substep_dt={sub:.2e} ({round(sim.dt / sub)}/frame)", flush=True)

    _, cap = capture_run(sim, device, stride=stride)
    d = diagnose(cap, sim)
    checks, passed = verdict(d)
    d["checks"], d["passed"], d["note"] = checks, passed, note
    d["config"] = {k: v for k, v in kw.items()}
    d["config"]["preset"] = preset
    d["substep_dt"] = sub
    with open(os.path.join(outdir, "diag.json"), "w") as f:
        json.dump(d, f, indent=2)
    np.savez_compressed(os.path.join(outdir, "curves.npz"),
                        **{k: v for k, v in cap.items()
                           if k in ("frame", "act", "tension", "length", "rest_length",
                                    "gaze", "target", "centre", "ins", "pull", "axis")})

    print(f"  range h/v/t = {d['range_hvt_deg']}   err mean/max = "
          f"{d['mean_settle_error_deg']}/{d['max_settle_error_deg']} deg   "
          f"recruit {d['recruitment_correct']}   drift {d['centroid_drift_max_frac_radius']}   "
          f"strain_p99 {d['strain_p99']}   shorten% {d['max_shortening_pct']}\n"
          f"  tracking rms h/v/t = {d['tracking_rms_deg']}   settled = "
          f"{d['tracking_settled_rms_deg']} deg", flush=True)
    for h in d["holds"]:
        flag = "" if h["recruit_ok"] is None else (" ok" if h["recruit_ok"] else " MISS")
        print(f"    {str(h['frames']):<12} {h['kind']:<11} {h['command_hvt']} -> "
              f"{h['achieved_hvt']}  err {h['error_deg']:>5}  {h['recruited']}{flag}", flush=True)
    print(f"  checks: " + "  ".join(f"{k}={'Y' if v else 'N'}" for k, v in checks.items())
          + f"   => {'PASS' if passed else 'fail'}", flush=True)

    if movie:
        render_eye.render(cap, float(sim.dt), os.path.join(outdir, "movie.mp4"),
                          os.path.join(outdir, "strip.png"))
    print(f"[trial] -> {outdir}\n", flush=True)
    return outdir, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="atlas", choices=list(ES.PRESETS))
    ap.add_argument("--label", default="run")
    ap.add_argument("--particles", type=int, default=45000)
    ap.add_argument("--n_grid", type=int, default=128)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--mparticles", type=int, default=2600)
    ap.add_argument("--contract", type=float, default=26.0)
    ap.add_argument("--drag", type=float, default=5.0)
    ap.add_argument("--muscle_drag", type=float, default=6.0)
    ap.add_argument("--k_bone", type=float, default=9000.0)
    ap.add_argument("--k_sleeve", type=float, default=2600.0)
    ap.add_argument("--muscle_youngs", type=float, default=110.0)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--oblique_strength", type=float, default=0.70)
    ap.add_argument("--mus_width", type=float, default=0.044)
    ap.add_argument("--mus_thickness", type=float, default=0.030)
    ap.add_argument("--mus_frac", type=float, default=0.95)
    ap.add_argument("--mus_gap", type=float, default=0.042)
    ap.add_argument("--kp", type=float, default=0.10)
    ap.add_argument("--ki", type=float, default=0.0)
    ap.add_argument("--kd", type=float, default=0.010)
    ap.add_argument("--gain", type=float, default=1.2)
    ap.add_argument("--tonic", type=float, default=0.20)
    ap.add_argument("--tau", type=float, default=0.020)
    ap.add_argument("--k_socket", type=float, default=5000.0)
    ap.add_argument("--k_fat", type=float, default=4000.0)
    ap.add_argument("--dt", type=float, default=0.003)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()
    trial(a.label, device=a.device, stride=a.stride, movie=not a.no_movie,
          preset=a.preset, n_particles=a.particles, n_muscle_particles=a.mparticles,
          n_grid=a.n_grid, n_frames=a.frames, dt=a.dt, contract=a.contract, drag=a.drag,
          muscle_drag=a.muscle_drag, muscle_youngs=a.muscle_youngs, k_bone=a.k_bone,
          stretch_activation=a.beta, oblique_strength=a.oblique_strength, k_sleeve=a.k_sleeve,
          mus_width=a.mus_width, mus_thickness=a.mus_thickness,
          mus_frac=a.mus_frac, mus_gap=a.mus_gap,
          kp=a.kp, ki=a.ki, kd=a.kd, gain=a.gain, tonic=a.tonic, tau=a.tau,
          k_socket=a.k_socket, k_fat=a.k_fat)


if __name__ == "__main__":
    main()
