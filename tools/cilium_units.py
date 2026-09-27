#!/usr/bin/env python
"""What a Platynereis cilium's constants ACTUALLY are, and what this rig would need to carry them.

    python tools/cilium_units.py

WHY. The cilium rig's numbers were each tuned until the picture looked right -- drag k = 12, bending
k_bend = 5e4, a node mass inherited from an MPM particle's radius -- and the spec declares
`force NOT DECLARED (ratios only)`, so nothing ever checked them against a cilium. The beat then
looks plausible while the force handed to the water is off by whatever the mass happens to be, which
is exactly the state run 0296 is in: real flow, five thousand times too little of it.

Every physical input below is a published measurement or a textbook formula, cited inline. The
script converts them into the per-unit-node-mass constants `drag` and `rod_elastic` actually take,
and then reports the two numbers that decide whether the rig can run them at all: the sperm number,
which says whether the beat has the right SHAPE, and the explicit-integration timestep, which says
what it would cost.
"""
from __future__ import annotations

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

# ---- physical inputs, every one of them sourced ------------------------------------------------
L_um    = 20.0      # cilium length. Poon, Jekely & Wan (2025) Sci Adv 11:eadw4067, prototroch
D_um    = 0.2       # cilium diameter, same source
F_HZ    = 10.0      # beat frequency, 10 +- 2 Hz over 21 larvae, same source
B_pNum2 = 580.0     # axoneme bending rigidity. Sartori et al. (2016) eLife 5:e13258, from
                    #   Gittes' 20 pN um^2 per microtubule x (9 doublets x 3 + 2 singlets)
ETA     = 1.0e-3    # sea water dynamic viscosity, Pa s
RHO_CIL = 1100.0    # cilium density, kg/m^3 (protein-rich cytoplasm)

# ---- the rig -----------------------------------------------------------------------------------
N_NODE   = 24
LENGTH_UM_PER_WORLD = 50.0
DT_RIG   = 1.0e-3
K_RIG    = 12.0
KBEND_RIG = 5.0e4


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from builder_figure import record, style

    L = L_um * 1e-6
    a = D_um * 1e-6 / 2.0
    om = 2.0 * math.pi * F_HZ
    B = B_pNum2 * 1e-12 * 1e-12                     # pN um^2 -> N m^2
    h = L / (N_NODE - 1)                            # segment length, m
    h_w = (L_um / (N_NODE - 1)) / LENGTH_UM_PER_WORLD   # the same, in world units
    M_PER_WORLD = LENGTH_UM_PER_WORLD * 1e-6

    # Gray & Hancock (1955) J Exp Biol 32:802 -- resistive force theory for a slender body
    zper = 4.0 * math.pi * ETA / math.log(2.0 * L / a)
    zpar = 2.0 * math.pi * ETA / (math.log(2.0 * L / a) - 0.5)
    m_node = RHO_CIL * math.pi * a * a * h          # a cylinder one segment long

    # the per-unit-mass constants the two operators take, in world units and seconds
    k_perp = zper * h / m_node                      # 1/s : a = -k_perp v for transverse motion
    k_drag = k_perp / (zper / zpar)                 # `drag` takes the TANGENTIAL k and a ratio
    k_bend = B / (h ** 3 * m_node)                  # 1/s^2 : rod_elastic's 4th-difference constant

    Sp = L * (zper * om / B) ** 0.25
    Sp_rig = (L_um / LENGTH_UM_PER_WORLD) * (
        (K_RIG * 2.0 * om) / (KBEND_RIG * h_w ** 4)) ** 0.25

    dt_bend = 0.5 / math.sqrt(k_bend)               # explicit inertial limit, rod_elastic's own
    dt_od = zper * h ** 4 / B                       # OVERDAMPED limit: no mass, no sqrt(k/m)

    rows = [
        ("cilium length L", f"{L_um:g} um", "Poon 2025"),
        ("cilium diameter", f"{D_um:g} um", "Poon 2025"),
        ("beat frequency", f"{F_HZ:g} Hz", "Poon 2025"),
        ("bending rigidity B", f"{B_pNum2:g} pN um^2", "Sartori 2016"),
        ("transverse drag zeta_perp", f"{zper:.3e} Pa s", "Gray & Hancock 1955"),
        ("drag anisotropy", f"{zper / zpar:.2f}", "Gray & Hancock 1955"),
        ("node mass (one segment)", f"{m_node:.3e} kg", "rho pi a^2 h"),
    ]
    print("\n  PHYSICAL INPUTS")
    for a_, b_, c_ in rows:
        print(f"    {a_:28s} {b_:>16s}   {c_}")
    print("\n  WHAT THE OPERATORS WOULD NEED         rig today        physical        ratio")
    print(f"    drag k (1/s)                  {K_RIG:14.4g} {k_drag:15.4g} {k_drag / K_RIG:12.3g}")
    print(f"    rod_elastic k_bend (1/s^2)    {KBEND_RIG:14.4g} {k_bend:15.4g} {k_bend / KBEND_RIG:12.3g}")
    print(f"\n  SPERM NUMBER  Sp = L (zeta_perp omega / B)^(1/4)")
    print(f"    real cilium                   {Sp:14.2f}")
    print(f"    the rig as it runs today      {Sp_rig:14.2f}   ({Sp_rig / Sp:.1f}x too damped)")
    print(f"\n  TIMESTEP")
    print(f"    rig today                     {DT_RIG:14.3g} s")
    print(f"    physical k_bend, INERTIAL     {dt_bend:14.3g} s   ({DT_RIG / dt_bend:,.0f}x smaller)")
    print(f"    physical, OVERDAMPED          {dt_od:14.3g} s   ({DT_RIG / dt_od:,.0f}x smaller)")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    labs = ["rig today\n(dt 1e-3)", "physical B,\nwith inertia", "physical B,\nOVERDAMPED",
            "overdamped,\n12 nodes"]
    h12 = L / 11.0
    dt_od12 = zper * h12 ** 4 / B
    vals = [DT_RIG, dt_bend, dt_od, dt_od12]
    cols = ["#999999", "#c0392b", "#e08214", "#4a7c59"]
    ax[0].bar(range(4), vals, color=cols, width=0.6)
    ax[0].set_yscale("log"); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(labs, fontsize=8)
    ax[0].set_ylabel("largest stable timestep (s)")
    for i, v in enumerate(vals):
        ax[0].annotate(f"{v:.1e}\n{0.6 / v:,.0f} steps/0.6 s", (i, v), ha="center",
                       textcoords="offset points", xytext=(0, 4), fontsize=7.5)
    ax[0].text(0, 1.08, "carrying the real bending rigidity is a TIMESTEP problem, and dropping "
                        "inertia is what solves it", transform=ax[0].transAxes, fontsize=9.5)
    style(ax[0])

    ax[1].bar([0, 1], [Sp, Sp_rig], color=["#2f6fb5", "#c0392b"], width=0.5)
    ax[1].axhspan(1, 5, color="#dddddd", alpha=.6, zorder=0)
    ax[1].text(1.35, 3.0, "Sp 1-5: where real cilia\nand sperm live", fontsize=8, color="#666666")
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["real Platynereis\ncilium", "the rig\ntoday"],
                                                    fontsize=9)
    ax[1].set_ylabel("sperm number Sp = L (zeta_perp omega / B)^(1/4)")
    for i, v in enumerate([Sp, Sp_rig]):
        ax[1].annotate(f"{v:.2f}", (i, v), ha="center", textcoords="offset points",
                       xytext=(0, 4), fontsize=9)
    ax[1].text(0, 1.08, f"and the beat's SHAPE is wrong for a separate reason: Sp is "
                        f"{Sp_rig / Sp:.1f}x too high, so the wave over-damps",
               transform=ax[1].transAxes, fontsize=9.5)
    style(ax[1])
    fig.tight_layout()

    record(fig,
           "MEASUREMENT, not a run: the cilium rig's constants checked against a real Platynereis "
           "cilium for the first time. Every number in the rig was tuned until the picture looked "
           "right and the spec declares `force NOT DECLARED (ratios only)`, so nothing had ever "
           "compared them to an animal. Inputs, all sourced: L = 20 um, diameter 0.2 um and 10 Hz "
           "from Poon, Jekely & Wan (2025) Sci Adv 11:eadw4067 on the prototroch itself; bending "
           "rigidity B = 580 pN um^2 from Sartori et al. (2016) eLife 5:e13258; transverse drag "
           f"zeta_perp = {zper:.2e} Pa s from Gray & Hancock (1955) resistive force theory. TWO "
           "SEPARATE THINGS ARE WRONG AND THEY NEED DIFFERENT FIXES. Right panel, the beat's "
           f"SHAPE: a real cilium sits at sperm number Sp = {Sp:.2f}, comfortably inside the 1-5 "
           f"band where cilia and sperm live, while the rig runs at Sp = {Sp_rig:.2f} -- "
           f"{Sp_rig / Sp:.1f}x too damped, which is why the wave decays from 50 degrees at the "
           "base to a few at the tip instead of propagating. Sp depends only on the RATIO of drag "
           "to rigidity, so it can be fixed without knowing either absolutely. Left panel, the "
           "force MAGNITUDE, which cannot: carrying the real B and the real node mass "
           f"({m_node:.2e} kg, a 0.87 um cylinder of cytoplasm) needs k_bend = {k_bend:.2e} per "
           f"second squared, whose explicit inertial timestep is {dt_bend:.1e} s -- "
           f"{DT_RIG / dt_bend:,.0f} times smaller than the rig's. THAT IS NOT A PHYSICS PROBLEM, "
           "IT IS INERTIA THE MODEL SHOULD NOT HAVE. A cilium in water runs at Reynolds number "
           "about 1e-5, where inertia is irrelevant by several decades, and the mass is purely a "
           "numerical nuisance: dropping it (EMIT velocity rather than acceleration, v = f/zeta, "
           f"which the engine already supports) moves the limit to {dt_od:.1e} s, and to "
           f"{dt_od12:.1e} s at 12 nodes because that limit goes as h^4. That is the route to a "
           "cilium with physical constants, and it is also the standard way flagella are solved.",
           name="cilium constants vs a real Platynereis cilium: Sp 4x too damped, and inertia is the timestep")


if __name__ == "__main__":
    main()
