#!/usr/bin/env python
"""tune_gaze -- gradient descent on the rotation controller, in the `inverse_slime` style.

    python tune_gaze.py archive/t13_s_a            # identify + tune from an archived run
    python tune_gaze.py archive/t13_s_a --verify   # then re-run the full MPM sim with the result

WHY NOT DIFFERENTIATE THE MPM ROLLOUT. The engine can keep the tape (`run(..., grad=True)`),
but a rollout of 45k+13k material points x 25 substeps x hundreds of frames unrolls into a
graph whose memory scales as points x steps -- exactly the cost Plexus\\,2 says makes a global
loss impractical. `inverse_slime` gets around it by refusing to roll out at all: it fits a
DIFFERENTIABLE model of one tick against the true trajectory, teacher-forced, and takes a
dense exact gradient there. The same move works here, in two stages.

STAGE 1 -- IDENTIFY THE PLANT (teacher-forced, one step, no rollout).
The globe is a rigid-ish body on an elastic suspension, so its rotation obeys, per axis,

    theta_ddot = B u  -  C theta_dot  -  K theta ,

where the input u(t) is the net commanded rotation the six activations ask for,
u = sum_m a_m(t) * axis_m(t), read straight out of the archived run: `act` and `axis` are
both recorded every captured frame. B, C, K come from one ridge least-squares fit of the
recorded (theta, theta_dot, theta_ddot, u) -- one step, exact, no simulation.

STAGE 2 -- TUNE THE CONTROLLER ON THE IDENTIFIED PLANT.
The identified plant is a handful of numbers and differentiable in closed form, so the whole
gaze program can be rolled out on it in milliseconds and Adam can descend on the controller
gains (kp, ki, kd, gain, tonic) against the tracking error, with the SAME rectified-projection
drive and first-order activation dynamics the real operator uses. The tuned gains are then put
back into the real spec and verified in the full MPM simulation -- which is the only result
that counts.

This is Loop II of the paper on a system whose ground truth we wrote ourselves: fit a
mechanistic reduced model to the observations, optimise on it, verify on the real one.
"""
from __future__ import annotations

import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch

import eye_anatomy as EA


# --------------------------------------------------------------------------- #
#  stage 1: identify
# --------------------------------------------------------------------------- #
def identify(curves, dt, ridge=1e-3):
    """Ridge least-squares fit of  theta_ddot = B u - C theta_dot - K theta,  per axis.

    `u` is the net rotation the innervation asks for: sum over muscles of activation times
    that muscle's own measured rotation axis. Both are in the archived run, so this is a
    one-step teacher-forced fit -- the trajectory is never re-simulated."""
    g = curves["gaze"].astype(np.float64)              # [T,3] degrees
    act = curves["act"].astype(np.float64)             # [T,6]
    axis = curves["axis"].astype(np.float64)           # [T,6,3]
    fr = curves["frame"].astype(np.float64)
    h = float(np.median(np.diff(fr))) * dt             # seconds between captured frames

    # the head-frame rotation the activations command: (-x elevation, +y abduction, +z intorsion)
    w = np.einsum("tm,tmk->tk", act, axis)             # [T,3] in (x, y, z)
    u = np.stack([w[:, 1], -w[:, 0], w[:, 2]], 1)      # -> (horizontal, vertical, torsion)

    gd = np.gradient(g, h, axis=0)
    gdd = np.gradient(gd, h, axis=0)
    out = []
    for k in range(3):
        X = np.stack([u[:, k], -gd[:, k], -g[:, k]], 1)
        A = X.T @ X + ridge * np.eye(3) * max(np.trace(X.T @ X), 1.0) / 3.0
        b = X.T @ gdd[:, k]
        B, C, K = np.linalg.solve(A, b)
        pred = X @ np.array([B, C, K])
        r2 = 1.0 - np.var(gdd[:, k] - pred) / max(np.var(gdd[:, k]), 1e-12)
        out.append({"B": float(B), "C": float(C), "K": float(K), "r2": float(r2)})
    return out, h


# --------------------------------------------------------------------------- #
#  stage 2: tune
# --------------------------------------------------------------------------- #
def rollout(gains, plant, axes0, program, n_frames, h, tau, dev="cpu"):
    """Differentiable rollout of the identified plant under the real control law.

    Deliberately the SAME law as `oculomotor_drive`: a PID on the gaze error, projected onto
    each muscle's rotation axis and RECTIFIED (reciprocal innervation), then first-order
    activation dynamics. Only the mechanics is replaced by the identified second-order model.
    """
    kp, ki, kd, gain, tonic = gains
    B = torch.tensor([p["B"] for p in plant], device=dev)
    C = torch.tensor([p["C"] for p in plant], device=dev)
    K = torch.tensor([p["K"] for p in plant], device=dev)
    th = torch.zeros(3, device=dev)
    thd = torch.zeros(3, device=dev)
    acc = torch.zeros(3, device=dev)
    a = torch.full((EA.N_MUSCLE,), float(tonic.detach()) if torch.is_tensor(tonic) else tonic,
                   device=dev)
    A = torch.tensor(axes0, dtype=torch.float32, device=dev)     # [6,3] head-frame axes
    prog = torch.tensor(program, dtype=torch.float32, device=dev)
    loss, traj = 0.0, []
    ptr = 0
    for t in range(n_frames):
        while ptr + 1 < prog.shape[0] and prog[ptr + 1, 0] <= t:
            ptr += 1
        tgt = prog[ptr, 1:4]
        err = tgt - th
        acc = acc * 0.996 + err
        e = kp * err + ki * acc - kd * thd
        omega = torch.stack([-e[1], e[0], e[2]])                 # head frame
        drive = A @ omega
        a_t = torch.clamp(tonic + gain * torch.clamp(drive, min=0.0), 0.0, 1.0)
        a = a + (h / tau) * (a_t - a)
        w = (a[:, None] * A).sum(0)
        u = torch.stack([w[1], -w[0], w[2]])
        thdd = B * u - C * thd - K * th
        thd = thd + h * thdd
        th = th + h * thd
        loss = loss + ((th - tgt) ** 2).sum()
        traj.append(th)
    return loss / n_frames, torch.stack(traj)


def tune(plant, axes0, program, n_frames, h, tau, init, steps=600, lr=0.05):
    raw = torch.tensor(np.log(np.asarray(init, float)), requires_grad=True)   # positive gains
    opt = torch.optim.Adam([raw], lr=lr)
    hist = []
    for i in range(steps):
        opt.zero_grad()
        g = torch.exp(raw)
        loss, _ = rollout((g[0], g[1], g[2], g[3], g[4]), plant, axes0, program, n_frames, h, tau)
        loss.backward()
        opt.step()
        with torch.no_grad():
            raw.clamp_(np.log(1e-5), np.log(30.0))
        hist.append(float(loss))
        if i % 100 == 0:
            print(f"  [tune] {i:4d}  loss {float(loss):9.3f}  "
                  f"kp {float(torch.exp(raw[0])):.4f} ki {float(torch.exp(raw[1])):.5f} "
                  f"kd {float(torch.exp(raw[2])):.4f} gain {float(torch.exp(raw[3])):.3f} "
                  f"tonic {float(torch.exp(raw[4])):.3f}", flush=True)
    return [float(x) for x in torch.exp(raw).detach()], hist


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--verify", action="store_true", help="re-run the full MPM sim with the result")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import yaml
    spec = yaml.safe_load(open(os.path.join(a.run_dir, "spec.yaml")))
    cur = dict(np.load(os.path.join(a.run_dir, "curves.npz")))
    dt = float(spec["general"]["dt"])
    n_frames = int(spec["general"]["n_frames"])
    drv = next(o for o in spec["operators"] if o["op"] == "oculomotor_drive")
    program = np.asarray(drv["program"], float)
    tau = float(drv.get("tau", 0.02))

    plant, h = identify(cur, dt)
    for k, nm in enumerate(("horizontal", "vertical", "torsion")):
        p = plant[k]
        print(f"[identify] {nm:<11} B={p['B']:9.2f}  C={p['C']:7.2f}  K={p['K']:8.2f}  "
              f"R^2={p['r2']:.3f}", flush=True)

    axes0 = cur["axis"][0]                                   # rest rotation axes, [6,3]
    init = [drv.get("kp", 0.11), max(drv.get("ki", 0.0), 1e-4), max(drv.get("kd", 0.01), 1e-4),
            drv.get("gain", 1.9), max(drv.get("tonic", 0.12), 1e-3)]
    l0, tr0 = rollout([torch.tensor(x) for x in init], plant, axes0, program, n_frames, h, tau)
    gains, hist = tune(plant, axes0, program, n_frames, h, tau, init, steps=a.steps)
    l1, tr1 = rollout([torch.tensor(x) for x in gains], plant, axes0, program, n_frames, h, tau)

    names = ["kp", "ki", "kd", "gain", "tonic"]
    print("\n[tune] surrogate tracking loss  {:.3f} -> {:.3f}  ({:.1f}x better)".format(
        float(l0), float(l1), float(l0) / max(float(l1), 1e-9)))
    for n, a0, a1 in zip(names, init, gains):
        print(f"        {n:<6} {a0:.4f}  ->  {a1:.4f}")
    out = {"plant": plant, "init": dict(zip(names, [float(x) for x in init])),
           "tuned": dict(zip(names, gains)),
           "surrogate_loss": {"before": float(l0), "after": float(l1)}}
    with open(os.path.join(a.run_dir, "tuned_gains.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[tune] -> {os.path.join(a.run_dir, 'tuned_gains.json')}")

    if a.verify:
        import run_eye
        cfg = json.load(open(os.path.join(a.run_dir, "diag.json")))["config"]
        cfg.update(kp=gains[0], ki=gains[1], kd=gains[2], gain=gains[3], tonic=gains[4])
        preset = cfg.pop("preset", "probe")
        print("\n[verify] re-running the full MPM simulation with the tuned gains", flush=True)
        run_eye.trial("tuned", device=a.device, stride=5, movie=True, preset=preset,
                      note="gains from tune_gaze.py (identified plant + Adam)", **cfg)


if __name__ == "__main__":
    main()
