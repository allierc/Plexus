"""Figure 2 of GNN_Transformer.tex Part V: the trained six-law network as ONE analog circuit with ONE knob.

    PYTHONPATH=src:tools /workspace/.conda_envs/neural-graph-linux/bin/python tools/exp18_spice_figure.py

    a   one neuron cell of the netlist (tools/exp18_spice_network.py): the sin-cos potentiometer, the two fixed
        synapse networks, two multipliers, the RC membrane and the tanh stage (a BJT differential pair)
    b   ngspice output at the six knob positions (the trained angles), each on its law's first held-out trial,
        against the target law -- read from spice/network_<run>.json, written by tools/exp18_spice_network.py
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SP = os.path.join(ROOT, "experiments", "exp18_phase_modulation", "spice")
RUN = "exp18_m5_alpha_phirandlearn_zf285"
OUT = os.path.join(ROOT, "papers", "figs_gnn_transformer", "exp18_spice")


def netlist_cell(cir: str):
    """One cell's SPICE elements, read from the generated netlist: the cell that receives the stimulus and has the
    most synapses. Returns (cell index, {element name: netlist line}, its GR / GI lines, the global lines)."""
    import re
    from collections import Counter
    L = open(cir).read().splitlines()
    n_in = Counter(re.match(r"GR\d+ 0 sr(\d+) ", l).group(1) for l in L if l.startswith("GR"))
    stim = [re.match(r"Bin(\d+) ", l).group(1) for l in L if l.startswith("Bin") and "V(u)" in l]
    c = max(stim, key=lambda k: n_in.get(k, 0))
    own = {l.split()[0]: l for l in L if re.match(rf"^(C|R|Bt|Rsr|Rsi|Bin){c} ", l)}
    gr = [l for l in L if re.match(rf"^GR\d+ 0 sr{c} ", l)]
    gi = [l for l in L if re.match(rf"^GI\d+ 0 si{c} ", l)]
    glob = {l.split()[0]: l for l in L if l.split() and l.split()[0] in ("Vknob", "Vgate", "Bcos", "Bsin", "Vu")}
    return c, own, gr, gi, glob


def cell_schematic(path: str, cir: str, dark: bool = False) -> int:
    """The cell AS NETLISTED: every symbol is one SPICE element of `cir` (name, type, value), drawn from ground up
    to its net and wired by net labels, as in an LTspice / KiCad schematic. Row 1: the broadcast (Vknob, Bcos,
    Bsin), the stimulus (Vu) and the two synapse nodes (GR..., Rsr / GI..., Rsi); row 2: the membrane (Bin, C, R)
    and the tanh stage (Bt), with Bin's full expression written under it."""
    import schemdraw
    import schemdraw.elements as elm
    c, own, gr, gi, glob = netlist_cell(cir)
    v = lambda line: line.split()[-1]
    H, W = 2.4, 3.3                                       # element height, spacing of parallel elements
    net = "#7fb2ff" if dark else "#1f4e9c"
    with schemdraw.Drawing(file=path, show=False, fontsize=10) as d:
        d.config(unit=H)
        if dark:
            d.config(color="white", bgcolor="black")

        net_color = net

        def column(x, y, net, elements):
            top = y + H
            if len(elements) > 1:
                elm.Line().at((x, top)).right(W * (len(elements) - 1))
            elm.Label().at((x + W * (len(elements) - 1) / 2, top + 0.45)).label(net, fontsize=11, color=net_color)
            for k, (cls, lab) in enumerate(elements):
                elm.Ground().at((x + W * k, y))
                cls().at((x + W * k, y)).up().label(lab, loc="bottom", fontsize=8.5, ofst=0.15)

        g = lambda line: f"{line.split()[0]}  {float(v(line)):+.3f} S\nfrom {line.split()[3]}"
        column(0.0, 0, "knob", [(elm.SourceV, f"Vknob\nDC {float(v(glob['Vknob'])):.3f} V")])
        column(3.6, 0, "vc", [(elm.SourceControlledV, "Bcos\ngate$\\cdot$cos V(knob)")])
        column(7.2, 0, "vs", [(elm.SourceControlledV, "Bsin\ngate$\\cdot$sin V(knob)")])
        column(10.8, 0, "u", [(elm.SourceV, "Vu\nstimulus, PWL")])
        column(14.8, 0, f"sr{c}", [(elm.SourceControlledI, g(l)) for l in gr[:2]] + [(elm.Resistor, f"Rsr{c}\n1 $\\Omega$")])
        elm.Label().at((14.8 + W, -0.9)).label(f"... {len(gr)} GR in all", fontsize=8.5)
        column(26.0, 0, f"si{c}", [(elm.SourceControlledI, g(l)) for l in gi[:2]] + [(elm.Resistor, f"Rsi{c}\n1 $\\Omega$")])
        elm.Label().at((26.0 + W, -0.9)).label(f"... {len(gi)} GI in all", fontsize=8.5)
        y2 = -6.0
        column(0.0, y2, f"v{c}", [(elm.SourceControlledI, f"Bin{c}"),
                                  (elm.Capacitor, f"C{c}\n{float(own[f'C{c}'].split()[3]) * 1e9:g} nF"),
                                  (elm.Resistor, f"R{c}\n{float(v(own[f'R{c}'])):.0f} $\\Omega$")])
        column(12.0, y2, f"t{c}", [(elm.SourceControlledV, f"Bt{c}\ntanh V(v{c})")])
        expr = own[f"Bin{c}"].split("I=", 1)[1]
        elm.Label().at((0.0, y2 - 1.3)).label(f"Bin{c}:  I = {expr}", fontsize=8.5, halign="left")
    return int(c)


DECK = os.path.join(ROOT, "experiments", "exp18_phase_modulation", "presentation", "figs")


def traces(data, path, dark=False):
    import matplotlib.pyplot as plt
    cols_fit = "white" if dark else "black"
    rows = {r["law"]: r for r in data["rows"]}
    with plt.style.context("dark_background" if dark else "default"):
        fig = plt.figure(figsize=(12.5, 6.4), facecolor="black" if dark else "white")
        gs = fig.add_gridspec(2, 3)
        for k, (name, tr) in enumerate(data["traces"].items()):
            a = fig.add_subplot(gs[k // 3, k % 3])
            t = np.array(tr["t"])
            n = min(len(t), len(tr["spice"]), len(tr["target"]))
            a.plot(t[:n], np.array(tr["target"])[:n], color="#2e8b4f", lw=2.4, label="target law")
            a.plot(t[:n], np.array(tr["spice"])[:n], color=cols_fit, lw=0.9, label="ngspice circuit")
            r = rows[name]
            a.set_title(f"{'abcdef'[k]}   knob at {r['alpha']:+.3f} rad: {name}   (error {r['spice_vs_target']:.3f})",
                        loc="left", fontsize=8.5)
            a.tick_params(labelsize=7)
            if k % 3 == 0:
                a.set_ylabel("output y", fontsize=8)
            if k >= 3:
                a.set_xlabel("time (task s; circuit ms)", fontsize=8)
            if k == 0:
                a.legend(fontsize=7, frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=150 if dark else 110, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    data = json.load(open(os.path.join(SP, f"network_{RUN}.json")))
    cir = os.path.join(SP, f"network_{RUN}_integrate.cir")
    cell = cell_schematic(os.path.join(SP, "cell_schematic.png"), cir)
    cell_schematic(OUT + "_cell.pdf", cir)                  # the same drawing, vector, for the paper
    traces(data, OUT + ".pdf")
    traces(data, OUT + ".png")
    # the deck: black (Cedric 2026-10-04)
    os.makedirs(DECK, exist_ok=True)
    cell_schematic(os.path.join(DECK, "spice_cell.png"), cir, dark=True)
    traces(data, os.path.join(DECK, "spice_traces.png"), dark=True)
    print("wrote", OUT + ".pdf", "and the deck's spice_cell.png / spice_traces.png; cell", cell)


if __name__ == "__main__":
    main()
