"""The trained six-law circuit of exp18 as ONE analog netlist, swept by a sin-cos potentiometer, checked in ngspice.

    PYTHONPATH=src:tools /workspace/.conda_envs/neural-graph-linux/bin/python tools/exp18_spice_network.py [run]

THE CIRCUIT (GNN_Transformer.tex Eq. exp_model), time scale x1000 (1 s of the task = 1 ms of the circuit):

    neuron i      a capacitor C (100 nF) with a leak resistor R_i = (tau_i / 1000) / C       -> -v_i / tau_i
    tanh          a stage t_i = tanh(v_i) (a BJT differential pair gives I tanh(V / 2 V_T))
    synapses      two FIXED transconductance networks, g^R_ij = W_ij cos(phi_t(j)t(i)), g^I_ij = W_ij sin(phi_...),
                  each summing onto a 1-ohm node:  S^R_i = sum_j g^R_ij t_j,  S^I_i = sum_j g^I_ij t_j
    the knob      ONE voltage V_knob = alpha (1 V = 1 rad) into a sin-cos stage: V_c = cos(alpha), V_s = sin(alpha)
    modulation    per neuron, two multipliers: G (V_c S^R_i + V_s S^I_i) = G sum_j W_ij tanh(v_j) cos(phi - alpha)
    input, bias   Gamma B_i u(t) and b_i, injected with the modulation as one current C * 1000 * (...)
    read-out      y = sum_i M_i t_i + b_out, transconductances onto a 1-ohm node

The six laws are six positions of the one knob (the trained angles alpha_k). THE CHECK, per law, on that law's first
held-out trial: (1) the parameters, extracted from the checkpoint, rebuilt in numpy with the trainer's own Euler step
(dt = 1/60 s) must reproduce the PyTorch output -- the extraction is right; (2) ngspice must reproduce the SAME
equations solved in continuous time (numpy, 64 substeps per frame) -- the netlist is right; (3) each against the
target law -- does the circuit work, and does the trained discrete-time model survive in continuous time.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
GD = "/groups/saalfeld/home/allierc/GraphData/log/training/neural/"
OUT = os.path.join(ROOT, "experiments", "exp18_phase_modulation", "spice")
NGSPICE = "/workspace/.conda_envs/neural-graph-linux/bin/ngspice"
K = 1000.0
C_NEURON = 100e-9


def extract(run: str, device="cpu"):
    """Every trained number of the circuit, read from the checkpoint through the rulers' own restore."""
    import torch
    from exp_measures.common import TrainingRun
    from exp_measures.exp18 import _restore, _phase_op
    T = TrainingRun(GD + run)
    TR, spec, sim, learn, ck, U, Y, cond = _restore(T, device)
    names = T.results[f"{run}_test"]["cell_names"]
    trials = [int(np.flatnonzero(cond == k)[0]) for k in range(len(names))]
    with torch.no_grad():
        H, _ = TR.rollout(sim, learn, U[trials[0]], spec["task"], device, grad=False)
        _, Yp = TR.rollout(sim, learn, U[trials], spec["task"], device, grad=False)
    op = _phase_op(H)
    ops = dict(zip(H.operator_names, H.operators))
    lvl, syn, aff, mot = (H.level(n) for n in ("neuron", "synapse", "afferent", "motor"))
    w = syn.get("w").reshape(-1)
    if op.dale:
        w = w.abs() * op._dale_sign(lvl, syn).reshape(-1).to(w.dtype)
    upd = ops["neuron_update"]
    tau = lvl.get(upd.tau_block)[:, 0].clamp(min=float(upd.tau_min))
    prj = ops["project"]
    P = {"N": lvl.n, "pre": syn.pre.detach().numpy(), "post": syn.post.detach().numpy(), "W": w.detach().numpy().astype(float),
         "type": lvl.node_type.detach().numpy(), "phi": op.phi.detach().numpy().astype(float), "alpha": op.alpha.detach().numpy().astype(float),
         "G": lvl.type_params[lvl.node_type][:, 2].detach().numpy().astype(float), "tau": tau.detach().numpy().astype(float),
         "b": lvl.get("bias")[:, 0].detach().numpy().astype(float),
         "aff_pre": aff.pre.detach().numpy(), "aff_post": aff.post.detach().numpy(), "B": aff.get("w").reshape(-1).detach().numpy().astype(float),
         "Gamma": float(prj.params.get("gain", 1.0)),
         "mot_pre": mot.pre.detach().numpy(), "M": mot.get("w").reshape(-1).detach().numpy().astype(float),
         "b_out": float(H.level("output").get("bias").reshape(-1)[0]),
         "dt": float(sim.dt), "names": names, "U": U[trials].detach().numpy().astype(float),
         "Y": Y[trials, :, 0].detach().numpy().astype(float), "torch": Yp[:, :, 0].detach().numpy().astype(float)}
    for k in ("aff_pre", "mot_pre"):
        assert P[k].max() < (8 if k == "aff_pre" else P["N"])
    return P


def rhs(P, v, u, alpha, gate=1.0):
    """dv/dt of every neuron (GNN_Transformer.tex Eq. exp_model), u the input lines [C], alpha the knob (rad);
    `gate` = 1 when the context is present (|z| = 1), 0 before it arrives (z = 0: every factor is 0)."""
    t = np.tanh(v)
    fac = gate * np.cos(P["phi"][P["type"][P["pre"]], P["type"][P["post"]]] - alpha)
    syn = np.zeros(P["N"])
    np.add.at(syn, P["post"], P["W"] * t[P["pre"]] * fac)
    inp = np.zeros(P["N"])
    np.add.at(inp, P["aff_post"], P["B"] * u[P["aff_pre"]])
    return -v / P["tau"] + P["b"] + P["G"] * syn + P["Gamma"] * inp


def readout(P, v):
    return float((P["M"] * np.tanh(v[P["mot_pre"]])).sum() + P["b_out"])


def drive_sequence(P, k) -> np.ndarray:
    """The input the trained model actually integrates, frame by frame: the trainer writes frame f's drive at the
    hook of frame f and the NEXT tick integrates it, after one unrecorded tick from rest that already reads u[0].
    The first tick runs BEFORE any drive is written: stimulus 0 and context 0 (so z = 0 and the modulation is
    off). The sequence is 0, u[0], u[1], ..., u[T-2], with the context gate 0, 1, 1, ... -- found by matching
    PyTorch to 1e-5 (2026-10-03)."""
    u = P["U"][k]
    return np.concatenate([np.zeros_like(u[:1]), u[:-1]], 0)


def simulate(P, k, sub=1):
    """Explicit Euler of Eq. exp_model with `sub` substeps per frame from rest, drive held over each frame; the
    read-out at the START of each frame (y[0] = b_out, the read-out at rest), as the trainer records it."""
    alpha = P["alpha"][k]
    v = np.zeros(P["N"])
    R = [readout(P, v)]
    for f, uf in enumerate(drive_sequence(P, k)):
        for _ in range(sub):
            v = v + (P["dt"] / sub) * rhs(P, v, uf, alpha, 0.0 if f == 0 else 1.0)
        R.append(readout(P, v))
    return np.array(R)


def netlist(P, k) -> str:
    """The analog circuit, law k = knob at alpha_k, input = law k's first held-out stimulus (zero-order hold)."""
    N, dts = P["N"], P["dt"] / K
    L = [f"* exp18 trained six-law circuit, knob at alpha = {P['alpha'][k]:.6f} rad (law {P['names'][k]}), time x{K:g}",
         f"Vknob knob 0 DC {P['alpha'][k]:.9f}",
         f"Vgate gate 0 PWL(0 0 {dts * (1 - 1e-6):.9g} 0 {dts:.9g} 1)",
         "Bcos vc 0 V=V(gate)*cos(V(knob))", "Bsin vs 0 V=V(gate)*sin(V(knob))"]
    u = drive_sequence(P, k)[:, 0]
    pts = []
    for f, x in enumerate(u):                                   # zero-order hold: each frame's value held for dt
        pts += [f"{f * dts:.9g} {x:.9g}", f"{(f + 1) * dts - dts * 1e-6:.9g} {x:.9g}"]
    L.append("Vu u 0 PWL(" + " ".join(pts) + ")")
    phi, ty = P["phi"], P["type"]
    gR = P["W"] * np.cos(phi[ty[P["pre"]], ty[P["post"]]])
    gI = P["W"] * np.sin(phi[ty[P["pre"]], ty[P["post"]]])
    inj = {i: [] for i in range(N)}
    for e, (j, i) in enumerate(zip(P["pre"], P["post"])):
        L.append(f"GR{e} 0 sr{i} t{j} 0 {gR[e]:.9g}")
        L.append(f"GI{e} 0 si{i} t{j} 0 {gI[e]:.9g}")
    for e, (s, i) in enumerate(zip(P["aff_pre"], P["aff_post"])):
        if s == 0:
            inj[i].append(f"{P['Gamma'] * P['B'][e]:.9g}*V(u)")
    for i in range(N):
        R = (P["tau"][i] / K) / C_NEURON
        L += [f"C{i} v{i} 0 {C_NEURON:g} IC=0", f"R{i} v{i} 0 {R:.9g}",
              f"Bt{i} t{i} 0 V=tanh(V(v{i}))",
              f"Rsr{i} sr{i} 0 1", f"Rsi{i} si{i} 0 1"]
        drive = " + ".join([f"{P['b'][i]:.9g}",
                            f"{P['G'][i]:.9g}*(V(vc)*V(sr{i}) + V(vs)*V(si{i}))"] + inj[i])
        L.append(f"Bin{i} 0 v{i} I={C_NEURON * K:.9g}*({drive})")
    for e, i in enumerate(P["mot_pre"]):
        L.append(f"GM{e} 0 y t{i} 0 {P['M'][e]:.9g}")
    L += ["Ry y 0 1", f"Iy 0 y DC {P['b_out']:.9g}"]
    T = (len(u) + 1) * dts
    L += [".options reltol=1e-6 abstol=1e-12 vntol=1e-9",
          f".tran {dts / 8:.6g} {T:.6g} 0 {dts / 8:.6g} uic", ".end"]
    return "\n".join(L) + "\n"


def run_spice(deck: str, n_frames: int, dts: float) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "y.txt")
        ctl = deck.replace(".end\n", f".control\nrun\nwrdata {data} v(y)\nquit\n.endc\n.end\n")
        open(os.path.join(d, "net.cir"), "w").write(ctl)
        r = subprocess.run([NGSPICE, "-b", "net.cir"], cwd=d, capture_output=True, text=True)
        if not os.path.exists(data):
            raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
        a = np.loadtxt(data)
    t_frames = np.arange(n_frames) * dts                           # the state at the START of each frame
    return np.interp(t_frames, a[:, 0], a[:, 1])


def nmse(a, b):
    n = min(len(a), len(b))
    return float(((a[:n] - b[:n]) ** 2).mean() / max((b[:n] ** 2).mean(), 1e-30))


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "exp18_m5_alpha_phirandlearn_zf285"
    os.makedirs(OUT, exist_ok=True)
    P = extract(run)
    print(f"[spice] {run}: {P['N']} neurons, {len(P['W'])} synapses -> {2 * len(P['W'])} transconductances; "
          f"knob positions {np.round(P['alpha'], 3).tolist()} rad")
    # (1) the extraction and the timing: numpy Euler at the trainer's dt must BE the PyTorch model
    e1 = max(float(np.abs(simulate(P, k, 1) - P["torch"][k]).max()) for k in range(len(P["names"])))
    print(f"[spice] numpy Euler (dt = 1/60 s) vs PyTorch, max |dy| over the six laws: {e1:.2e}")
    rows = []
    traces = {}
    for k, name in enumerate(P["names"]):
        y_euler = simulate(P, k, 1)
        y_cont = simulate(P, k, 64)
        deck = netlist(P, k)
        if k == 0:
            open(os.path.join(OUT, f"network_{run}_{name}.cir"), "w").write(deck)
        y_sp = run_spice(deck, P["torch"].shape[1], P["dt"] / K)
        tgt = P["Y"][k]
        rows.append({"law": name, "alpha": round(float(P["alpha"][k]), 4),
                     "torch_vs_target": nmse(P["torch"][k], tgt), "euler_vs_torch_maxabs": float(np.abs(y_euler - P["torch"][k]).max()),
                     "spice_vs_continuous": nmse(y_sp, y_cont), "continuous_vs_target": nmse(y_cont, tgt),
                     "spice_vs_target": nmse(y_sp, tgt)})
        traces[name] = {"t": (np.arange(len(tgt)) * P["dt"]).tolist(), "target": tgt.tolist(),
                        "torch": P["torch"][k].tolist(), "spice": y_sp.tolist(), "u": P["U"][k][:, 0].tolist()}
        r = rows[-1]
        print(f"  {name:14s} alpha {r['alpha']:+.3f}  torch/target {r['torch_vs_target']:.4f}  "
              f"spice/continuous {r['spice_vs_continuous']:.1e}  continuous/target {r['continuous_vs_target']:.4f}  "
              f"spice/target {r['spice_vs_target']:.4f}", flush=True)
    json.dump({"run": run, "rows": rows, "traces": traces}, open(os.path.join(OUT, f"network_{run}.json"), "w"))


if __name__ == "__main__":
    main()
