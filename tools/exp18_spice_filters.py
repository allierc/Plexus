"""The six laws of exp18 as six classic analog circuits: schematic + ngspice AC check against the target law.

    PYTHONPATH=src /workspace/.conda_envs/neural-graph-linux/bin/python tools/exp18_spice_filters.py

THE "SEPARATE CIRCUIT PER TASK" BASELINE (GNN_Transformer.tex Table 5): one textbook filter per law, before the
trained network puts all six into ONE wiring diagram swept by a sin-cos potentiometer (tools/exp18_spice_network.py).

TIME SCALE x1000 (an analog-computer convention): 1 Hz of the task is 1 kHz in the circuit, 1 s is 1 ms, so the
components are practical (the 1 Hz resonator is a real RLC: 25.3 mH, 1 uF, 127 ohm). Every law is taken from
config/task/m5_six.yaml, never retyped, and each circuit's AC response is compared with the law's own H(s) at the
scaled frequency. Op-amps are ideal (a voltage-controlled voltage source of gain 1e6).

    integrate      H = 1/s                         inverting RC integrator + unity inverter      RC = 1 ms
    delay 0.1 s    H = exp(-sT)                    four first-order all-pass stages (Pade 4)     RC = T/8
    lowpass        4th-order Butterworth, 1 Hz     two unity-gain Sallen-Key stages, Q 0.541 / 1.307
    highpass       4th-order Butterworth, 1 Hz     the same with R and C exchanged
    resonator      w0^2 / (s^2 + 2 z w0 s + w0^2)  series RLC, output across C
    differentiate  s / (0.15 s + 1)                series C into R (RC high-pass) x gain 1/0.15
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import yaml

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "experiments", "exp18_phase_modulation", "spice")
NGSPICE = "/workspace/.conda_envs/neural-graph-linux/bin/ngspice"
K = 1000.0                                   # time scale: circuit seconds = task seconds / K
OPAMP = ".subckt opamp inp inn out\nE1 out 0 inp inn 1e6\n.ends\n"


def laws():
    return {t["name"]: t for t in yaml.safe_load(open(os.path.join(ROOT, "config", "task", "m5_six.yaml")))["teachers"]}


def target_H(law: dict, s: np.ndarray) -> np.ndarray:
    """The law's own transfer function at Laplace variable s (rad per TASK second)."""
    if law["law"] == "integrate":
        return 1.0 / s
    if law["law"] == "delay":
        return np.exp(-s * float(law["seconds"]))
    if law["law"] == "laplace":
        return np.polyval(law["num"], s) / np.polyval(law["den"], s)
    if law["law"] == "lti":
        from scipy import signal
        b, a = signal.butter(int(law["order"]), 2 * np.pi * float(law["cutoff_hz"]),
                             btype=law["band"], analog=True)
        return np.polyval(b, s) / np.polyval(a, s)
    raise ValueError(law)


def netlist(name: str, law: dict) -> tuple[str, str]:
    """The circuit of one law, input node `in` (AC 1 V), output node `out`; component values from the law."""
    if name == "integrate":
        R, C = 1e6, (1.0 / K) / 1e6                   # RC = 1/K s = 1 ms: 1 MOhm, 1 nF
        body = (f"R1 in n1 {R:g}\nC1 n1 o1 {C:g}\nX1 0 n1 o1 opamp\n"
                f"R2 o1 n2 10k\nR3 n2 out 10k\nX2 0 n2 out opamp\n")
        return body, f"RC integrator: R {R/1e6:g} MOhm, C {C*1e9:g} nF, then a unity inverter"
    if name == "delay":
        T = float(law["seconds"]) / K
        R, C = 10e3, (T / 8) / 10e3
        body, prev = "", "in"
        for k in range(4):                           # first-order all-pass: (1 - s RC) / (1 + s RC)
            body += (f"Ra{k} {prev} a{k} 10k\nRb{k} a{k} o{k} 10k\nRc{k} {prev} p{k} {R:g}\nCc{k} p{k} 0 {C:g}\n"
                     f"X{k} p{k} a{k} o{k} opamp\n")
            prev = f"o{k}"
        body += f"Rout {prev} out 1\n"
        return body, f"4 first-order all-pass stages, each R {R/1e3:g} kOhm, C {C*1e9:.3g} nF (RC = T/8)"
    if name in ("lowpass", "highpass"):
        w0 = 2 * np.pi * float(law["cutoff_hz"]) * K
        qs = [1 / (2 * np.cos(np.pi * (2 * k + 1) / 8)) for k in range(2)]  # 0.541, 1.307
        body, prev = "", "in"
        for k, q in enumerate(qs):
            if name == "lowpass":                    # unity Sallen-Key: R1 = R2 = R, C1 = 2Q/(w0 R), C2 = 1/(2Q w0 R)
                R = 10e3
                C1, C2 = 2 * q / (w0 * R), 1 / (2 * q * w0 * R)
                body += (f"R{k}a {prev} m{k} {R:g}\nR{k}b m{k} p{k} {R:g}\nC{k}a m{k} o{k} {C1:g}\n"
                         f"C{k}b p{k} 0 {C2:g}\nX{k} p{k} o{k} o{k} opamp\n")
            else:                                    # high-pass: C1 = C2 = C, R1 = 1/(2Q w0 C), R2 = 2Q/(w0 C)
                C = 10e-9
                R1, R2 = 1 / (2 * q * w0 * C), 2 * q / (w0 * C)
                body += (f"C{k}a {prev} m{k} {C:g}\nC{k}b m{k} p{k} {C:g}\nR{k}a m{k} o{k} {R1:g}\n"
                         f"R{k}b p{k} 0 {R2:g}\nX{k} p{k} o{k} o{k} opamp\n")
            prev = f"o{k}"
        body += f"Rout {prev} out 1\n"
        return body, f"two unity-gain Sallen-Key {name} stages, Q {qs[0]:.3f} and {qs[1]:.3f}, cut-off {law['cutoff_hz']} Hz x {K:g}"
    if name == "resonator":
        a2, a1, a0 = law["den"]
        w0, z = np.sqrt(a0 / a2) * K, a1 / (2 * np.sqrt(a0 * a2))
        C = 1e-6
        L = 1 / (w0 ** 2 * C)
        R = 2 * z * np.sqrt(L / C)
        body = f"R1 in n1 {R:g}\nL1 n1 out {L:g}\nC1 out 0 {C:g}\n"
        return body, f"series RLC: R {R:.0f} Ohm, L {L*1e3:.1f} mH, C {C*1e6:g} uF (f0 {w0/2/np.pi:.0f} Hz, damping {z:.2f})"
    if name == "differentiate":
        tau = law["den"][0] / K
        gain = law["num"][0] / law["den"][0]
        C = 1e-6
        R = tau / C
        Rf = 10e3 * (gain - 1)
        body = f"C1 in n1 {C:g}\nR1 n1 0 {R:g}\nRg m 0 10k\nRf out m {Rf:g}\nX1 n1 m out opamp\n"
        return body, f"RC high-pass: C {C*1e6:g} uF, R {R:.0f} Ohm (tau {tau*1e3:.2f} ms), then a non-inverting gain {gain:.2f}"
    raise ValueError(name)


def ac(name: str, body: str, f: np.ndarray) -> np.ndarray:
    """ngspice AC sweep at the given circuit frequencies; returns the complex V(out)/V(in)."""
    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "out.txt")
        deck = (f"* exp18 {name}\n{OPAMP}Vin in 0 DC 0 AC 1\n{body}"
                f".ac lin {len(f)} {f[0]:g} {f[-1]:g}\n.control\nrun\nwrdata {data} v(out)\nquit\n.endc\n.end\n")
        open(os.path.join(d, "deck.cir"), "w").write(deck)
        subprocess.run([NGSPICE, "-b", "deck.cir"], cwd=d, capture_output=True, text=True, check=True)
        a = np.loadtxt(data)
    return a[:, 1] + 1j * a[:, 2]


def schematic(name: str, path: str) -> None:
    """A drawn schematic of one law's circuit (schemdraw), components labelled with their values."""
    import schemdraw
    import schemdraw.elements as elm
    lw = laws()[name]
    with schemdraw.Drawing(file=path, show=False, fontsize=11) as d:
        d.config(unit=2.2)
        if name == "resonator":
            body, _ = netlist(name, lw)
            v = {l.split()[0]: float(l.split()[-1]) for l in body.strip().splitlines()}
            elm.SourceSin().up().label("u(t)")
            elm.Resistor().right().label(f"{v['R1']:.0f} $\\Omega$")
            elm.Inductor().right().label(f"{v['L1'] * 1e3:.1f} mH")
            d.push()
            elm.Capacitor().down().label(f"{v['C1'] * 1e6:g} $\\mu$F")
            elm.Line().left().tox(0)
            d.pop()
            elm.Line().right(1)
            elm.Dot(open=True).label("y(t)", loc="right")
        elif name == "differentiate":
            elm.Dot(open=True).label("u(t)", loc="left")
            elm.Capacitor().right().label("1 $\\mu$F")
            d.push()
            elm.Resistor().down().label("150 $\\Omega$")
            elm.Ground()
            d.pop()
            op = elm.Opamp(leads=True).anchor("in2").right()
            elm.Line().at(op.out).right(0.6)
            elm.Dot(open=True).label("y(t)", loc="right")
            elm.Resistor().at(op.in1).down().label("10 k$\\Omega$", loc="bottom")
            elm.Ground()
        elif name == "integrate":
            op = elm.Opamp(leads=True)
            elm.Line().at(op.in2).down(0.6)
            elm.Ground()
            elm.Resistor().at(op.in1).left().label("1 M$\\Omega$")
            elm.Dot(open=True).label("u(t)", loc="left")
            elm.Line().at(op.in1).up(1.6)
            elm.Capacitor().right().tox(op.out).label("1 nF")
            elm.Line().down().toy(op.out)
            elm.Line().at(op.out).right(0.8)
            elm.Dot(open=True).label("$-\\!\\int u$ (then $\\times -1$)", loc="right")
        else:                                       # lowpass / highpass / delay: one stage drawn, xN noted
            op = elm.Opamp(leads=True)
            first, second = ((elm.Resistor, elm.Resistor) if name == "lowpass"
                             else (elm.Capacitor, elm.Capacitor) if name == "highpass" else (elm.Resistor, elm.Capacitor))
            elm.Line().at(op.in2).left(0.4)
            second().left().label("2" if name != "delay" else "C")
            d.push()
            (elm.Capacitor if name == "lowpass" else elm.Resistor)().down().label("C2" if name == "lowpass" else "R2")
            elm.Ground()
            d.pop()
            first().left().label("1" if name != "delay" else "R")
            elm.Dot(open=True).label("u(t)", loc="left")
            elm.Line().at(op.out).right(0.8)
            elm.Dot(open=True).label({"lowpass": "y (x2 stages)", "highpass": "y (x2 stages)",
                                      "delay": "y (x4 stages)"}[name], loc="right")
            elm.Line().at(op.in1).down(0.8)
            elm.Line().right().tox(op.out)
            elm.Line().up().toy(op.out)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    L = laws()
    order = ["integrate", "delay", "lowpass", "highpass", "resonator", "differentiate"]
    f_task = np.linspace(0.05, 2.0, 160)                  # Hz in the task's units: the stimulus band
    fig, axes = plt.subplots(2, 6, figsize=(17, 5.6), gridspec_kw={"height_ratios": [1.0, 1.0]})
    report = []
    for c, name in enumerate(order):
        body, text = netlist(name, L[name])
        Hc = ac(name, body, f_task * K)
        Ht = target_H(L[name], 2j * np.pi * f_task)
        err = float(np.max(np.abs(Hc - Ht)) / np.max(np.abs(Ht)))
        report.append((name, text, err))
        open(os.path.join(OUT, f"law_{name}.cir"), "w").write(f"* exp18 law {name}: {text}\n{OPAMP}{body}")
        png = os.path.join(OUT, f"law_{name}_schematic.png")
        schematic(name, png)
        axes[0, c].imshow(plt.imread(png))
        axes[0, c].axis("off")
        axes[0, c].set_title(name, fontsize=10)
        ax = axes[1, c]
        if name == "delay":                           # an all-pass delay is flat in gain: its law is the PHASE
            ax.plot(f_task, np.degrees(np.unwrap(np.angle(Ht))), color="#2e8b4f", lw=2.4, label="target law")
            ax.plot(f_task, np.degrees(np.unwrap(np.angle(Hc))), color="black", lw=1.0, label="ngspice circuit")
            ax.set_ylabel("phase (degrees)", fontsize=8)
        else:
            ax.loglog(f_task, np.abs(Ht), color="#2e8b4f", lw=2.4, label="target law")
            ax.loglog(f_task, np.abs(Hc), color="black", lw=1.0, label="ngspice circuit")
        ax.set_xlabel("task frequency (Hz)  [circuit: x1000]", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.text(0.03, 0.04, f"max |dH| / max |H| = {err:.1e}", transform=ax.transAxes, fontsize=7)
        if c == 0:
            ax.set_ylabel("gain |H|", fontsize=8)
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "six_laws_circuits.png"), dpi=130, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "six_laws_circuits.pdf"), bbox_inches="tight")
    for name, text, err in report:
        print(f"{name:14s} {err:9.2e}   {text}")


if __name__ == "__main__":
    main()
