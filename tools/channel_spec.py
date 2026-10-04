#!/usr/bin/env python
"""Write the specs of experiment 4 (experiments/exp04_membrane_channels.md) and print what they
predict, so every row of its results table has numbers to be wrong against.

    PYTHONPATH=src python tools/channel_spec.py anatomy <channel>   # data/<cif> -> shapes/exp04_<channel>/
    PYTHONPATH=src python tools/channel_spec.py <K><s>              # e.g. 1a -> config/channel/exp04_v1a.yaml
    PYTHONPATH=src python tools/channel_spec.py --list              # every version, its parent and its change

WHY A SCRIPT WRITES THE SPEC (the reason tools/bfm_motor_spec.py gives, and it holds here): every
number in a spec is DERIVED -- the box from the protein's extent, the lipid lattice from the area
per lipid, the lipid cohesion from the bilayer's area modulus, the time step from the stiffest
spring, the tension protocol from the patch's own relaxation time, the units from kT and a
residue's Stokes drag. Written by hand those are thirty chances to disagree.

VERSIONS: each spec is its PARENT plus ONE change (`change`, dotted keys into the parameter dict
`P` below), and `why` says what that change is for -- the one-change rule of
experiments/INSTRUCTION.md, checkable here rather than asserted in prose. A round is three siblings
of one channel, each one change from the same parent.

THE UNITS: energy kT (4.11 pN nm at 298 K); length the box (L nm); time the one that makes a
residue's Stokes mobility 1 (a 0.3 nm bead in water, zeta = 6 pi eta a), so tau = zeta L^2 / kT.
Every force is overdamped, v = mobility x force, with mobility 1 for every bead -- a lipid bead
therefore moves ~700x faster than a lipid does in a bilayer, and the run's clock is the model's
relaxation clock, stated, not the channel's millisecond one.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
DATA = os.path.join(REPO, "experiments", "exp04_membrane_channels", "data")
PY = "/workspace/.conda_envs/neural-graph-linux/bin/python"

KT_J = 4.11e-21                     # J, 298 K
# Wimley-White octanol scale, whole residue, water -> octanol, kcal/mol (Wimley, Creamer & White 1996,
# Biochemistry 35:5109); the side chain's share is the residue's minus glycine's (1.15)
WW_OCT = {"ALA": 0.50, "ARG": 1.81, "ASN": 0.85, "ASP": 3.64, "CYS": -0.02, "GLN": 0.77, "GLU": 3.63,
          "GLY": 1.15, "HIS": 2.33, "ILE": -1.12, "LEU": -1.25, "LYS": 2.80, "MET": -0.67, "PHE": -1.71,
          "PRO": 0.14, "SER": 0.46, "THR": 0.25, "TRP": -2.09, "TYR": -0.71, "VAL": -0.46}
KCAL_PER_KT = 0.593
E_C = 1.602176634e-19

# ---- the anatomy of each channel: which deposited file, which residues make which domain ----------
# (author residue numbers; --membrane is the residue range whose mean height is the bilayer's centre)
ANATOMY = {
    # MscS, 6PWN (Reddy 2019, nanodisc, His-tag construct; residues 1-280): anchor + TM1 + TM2 = the
    # paddle, TM3a + TM3b = the pore, the rest the cytoplasmic cage. The bilayer's centre is the mean
    # height of residues 16-90 (the anchor's W16 through the TM1-TM2 hairpin): Reddy's Figure 2 puts
    # the 38 A bilayer around the whole upper TM region, TM3b ~15 A below its cytoplasmic face. The
    # entry has the cage at +z, hence the flip (cytoplasm at -z in every exp04 spec).
    "mscs": {"cif": "6PWN.cif", "parts": "paddle:1-91,tm3a:92-113,tm3b:114-128,cage:129-280", "membrane": "16-90",
             "name": "MscS", "lining": "tm3a", "flip": True},
    # MscL, 2OAR (M. tuberculosis, closed; residues 1-125): S1 + TM1 line the pore, then the periplasmic
    # loop, TM2 and the cytoplasmic C-terminal bundle, which this entry already puts at -z.
    "mscl": {"cif": "2OAR.cif", "parts": "s1:1-14,tm1:15-43,loop:44-68,tm2:69-99,ctd:100-125", "membrane": "15-43,69-95",
             "name": "MscL", "lining": "tm1"},
    # Kv1.2, 8VC6 (Wu et al. 2023, in K+, conducting; the transmembrane core, residues 138-421): the
    # voltage sensors S1-S4 and the pore domain S5-P-S6, and the selectivity filter TVGYG = 374-378
    # (T374 is KcsA's T75), whose backbone carbonyls -- O and C of 374-378 -- are placed as their own
    # beads from the deposited atoms. The S6 bundle crossing sits at +z in the entry, hence the flip.
    "kv12": {"cif": "8VC6.cif", "parts": "vsd:138-311,pore:312-421", "membrane": "160-420",
             "name": "Kv1.2", "lining": "pore", "flip": True,
             "extra": "filter_O:O:374-378,filter_C:C:374-378"},
    # alpha-hemolysin, 9M4P (Chatterjee et al. 2026, the mature heptameric pore in red-cell membranes;
    # residues 1-293). The stem (110-150) is LEFT OUT of the assembling protomers: Chatterjee find the
    # prestem detached and disordered from the earliest arcs. The rim's lipid-binding residues W179 and
    # R200 sit at the headgroups, so the bilayer's centre is ~20 A below their mean height. The stem
    # points to +z in the entry; flipped so the cap is outside (+z) and the stem would enter the cell.
    "ahl": {"cif": "9M4P.cif", "parts": "cap:1-109,rim:151-293", "membrane": "179-179,200-200",
            "membrane_offset": -20.0, "name": "alpha-hemolysin", "flip": True, "rim_residues": [170, 215]},
    # the ion-flow test: a membrane with a hole and no protein (build_hole)
    "hole": {"name": "hole (ion-flow test)"},
    # an integration fixture, never a result: a synthetic five-helix ring (tests only, /tmp)
    "selftest": {"cif": "/tmp/exp04_smoke/selftest.cif", "parts": "tm1:1-30,tm2:35-64", "membrane": "1-64",
                 "name": "SelfTest", "lining": "tm1"},
}

# ---- THE ION-FLOW RIG's STRUCTURES (from 2026-09-26): two states per channel, each cut into its own shape
# folder `exp04r_<key>` by channel_anatomy.py, with the all-atom ion path (HOLE-like) stored beside the
# alpha carbons. `membrane: belt` puts the mid-plane at the protein's own hydrophobic belt (Wimley-White);
# a residue range is a paper's placement. `state` is what the entry is: closed, open, prepore, pore, ...
RIG = {
    # MscS: OPM, AS THE PAPER'S NANODISC DENSITY SAYS. Reddy 2019: "the prominent cavities formed between the
    # TM1-TM2 hairpin and TM3 are fully located outside the membrane" and the N-terminal anchor is embedded
    # in the outer leaflet -- true with OPM's plane (6PWN) and false with exp04's (the mean height of
    # residues 16-90), which sat ~1.6 nm lower on the protein (the anatomy audit, 2026-09-26).
    "mscs_c": {"cif": "6PWN.cif", "parts": "paddle:1-91,tm3a:92-113,tm3b:114-128,cage:129-280", "membrane": "opm",
               "opm": "6pwn", "up": "25-35", "ligands": True, "state": "closed", "channel": "mscs"},
    "mscs_o": {"cif": "2VV5.cif", "parts": "paddle:25-91,tm3a:92-113,tm3b:114-128,cage:129-280", "membrane": "opm",
               "opm": "2vv5", "up": "25-35", "state": "open", "channel": "mscs"},
    "mscl_c": {"cif": "2OAR.cif", "parts": "s1:1-14,tm1:15-43,loop:44-68,tm2:69-99,ctd:100-125", "membrane": "opm",
               "opm": "2oar", "up": "44-68", "state": "closed", "channel": "mscl"},
    "mscl_o": {"cif": "elife-01834-code2-v1.pdb", "parts": "s1:1-14,tm1:15-43,loop:44-68,tm2:69-99,ctd:100-104",
               "membrane": "belt", "opm": "2oar", "up": "44-68", "state": "open (Wang 2014 model)", "channel": "mscl"},
    "kv12": {"cif": "8VC6.cif", "parts": "vsd:138-311,pore:312-421", "membrane": "opm", "opm": "2a79", "up": "350-380",
             "state": "open, conducting", "channel": "kv12"},
    "ahl_pre": {"cif": "9M4A.cif", "parts": "cap:1-109,stem:110-150,rim:151-293", "membrane": "179-179,200-200",
                "membrane_offset": -20.0, "opm": "7ahl", "up": "1-109", "state": "prepore IV", "channel": "ahl"},
    "ahl_pore": {"cif": "9M4P.cif", "parts": "cap:1-109,stem:110-150,rim:151-293", "membrane": "opm", "opm": "7ahl",
                 "up": "1-109", "state": "pore", "channel": "ahl"},
    "kcsa_c": {"cif": "1K4C.cif", "assembly": True, "chains": "C@0,C@1,C@2,C@3",
               "parts": "tm1:22-52,turret:53-61,phelix:62-74,filter:75-79,tm2:80-124", "membrane": "opm", "opm": "1k4c",
               "up": "53-61", "state": "closed", "channel": "kcsa"},
    "kcsa_o": {"cif": "3F5W.cif", "assembly": True, "chains": "C@0,C@1,C@2,C@3",
               "parts": "tm1:30-52,turret:53-61,phelix:62-74,filter:75-79,tm2:80-117", "membrane": "opm", "opm": "3f5w",
               "up": "53-61", "state": "open (inactivated filter)", "channel": "kcsa"},
    # KcsA's PARTIAL OPENINGS (Cuello 2010: the filter conducts at small openings and collapses at 32 A): 17 A (3F7Y)
    # and 14 A (3FB5), taken from OPM's oriented files (RCSB is blocked here; the coordinates are the entries')
    "kcsa_o17": {"cif": "opm/3f7y.pdb", "chains": "A,B,C,D", "parts": "tm1:22-52,turret:53-61,phelix:62-74,filter:75-79,tm2:80-124",
                 "membrane": "opm", "opm": "3f7y", "up": "53-61", "state": "open 17 A (filter conducting)", "channel": "kcsa"},
    "kcsa_o14": {"cif": "opm/3fb5.pdb", "chains": "A,B,C,D", "parts": "tm1:22-52,turret:53-61,phelix:62-74,filter:75-79,tm2:80-124",
                 "membrane": "opm", "opm": "3fb5", "up": "53-61", "state": "open 14 A (filter conducting)", "channel": "kcsa"},
    "gA": {"cif": "1MAG.cif", "hetatm": True, "axis": "pca", "parts": "helix:1-16", "membrane": "opm", "opm": "1mag",
           "state": "dimer, the channel", "channel": "gramicidin"},
    # the dimer DISSOCIATED: the upper monomer (chain B) moved 1.0 nm sideways in its leaflet -- gramicidin's
    # closed state is two monomers not in register (Hladky & Haydon 1972); a constructed geometry, stated
    "gA_apart": {"cif": "1MAG.cif", "hetatm": True, "axis": "pca", "parts": "helix:1-16", "membrane": "opm", "opm": "1mag",
                 "move_chain": "B:1.0,0,0", "state": "monomers apart (constructed)", "channel": "gramicidin"},
    "glic_c": {"cif": "4NPQ.cif", "chains": "A,B,C,D,E", "parts": "ecd:5-192,tmd:193-315", "membrane": "opm",
               "opm": "4npq", "up": "5-192", "state": "closed (pH 7)", "channel": "glic"},
    "glic_o": {"cif": "4HFI.cif", "parts": "ecd:5-192,tmd:193-315", "membrane": "opm", "opm": "4hfi", "up": "5-192",
               "state": "open (pH 4)", "channel": "glic"},
    "navab_c": {"cif": "5VB2.cif", "parts": "vsd:999-1117,vsd:1995-2117,pore:1118-1226,pore:2118-2226",
                "membrane": "opm", "opm": "5vb2", "up": "1170-1185,2170-2185", "state": "closed", "channel": "navab"},
    "navab_o": {"cif": "5VB8.cif", "assembly": True, "parts": "vsd:999-1117,pore:1118-1226", "membrane": "opm",
                "opm": "5vb8", "up": "1170-1185", "state": "open", "channel": "navab"},
    "traak_c": {"cif": "4WFF.cif", "chains": "A,B", "axis": "c2", "parts": "all:28-286", "membrane": "opm",
                "opm": "4wff", "up": "61-110", "ligands": True, "state": "non-conductive", "channel": "traak"},
    "traak_o": {"cif": "4WFE.cif", "chains": "A,B", "axis": "c2", "parts": "all:28-286", "membrane": "opm",
                "opm": "4wfe", "up": "61-110", "state": "conductive", "channel": "traak"},
    "ompf": {"cif": "2OMF.cif", "assembly": True, "pores": "chains", "parts": "barrel:1-340", "membrane": "opm",
             "opm": "2omf", "up": "150-170,195-210,235-250", "state": "open trimer", "channel": "ompf"},
}


def rig_anatomy(key, flip=None):
    """Cut RIG[key] into graphs_data/shapes/exp04r_<key> (channel_anatomy.py)."""
    a = RIG[key]
    cmd = [PY, os.path.join(REPO, "tools", "channel_anatomy.py"), os.path.join(DATA, a["cif"]), "--name", f"exp04r_{key}",
           "--parts", a["parts"], "--membrane", a["membrane"]]
    f_ = a.get("flip") if flip is None else flip
    cmd += ["--flip"] if f_ else []
    cmd += ["--up", a["up"]] if a.get("up") else []
    cmd += ["--opm", os.path.join(DATA, "opm", a["opm"] + ".pdb")] if a.get("opm") else []
    cmd += ["--assembly"] if a.get("assembly") else []
    cmd += ["--hetatm"] if a.get("hetatm") else []
    cmd += ["--ligands"] if a.get("ligands") else []
    cmd += ["--move-chain", a["move_chain"]] if a.get("move_chain") else []
    cmd += ["--chains", a["chains"]] if a.get("chains") else []
    cmd += ["--axis", a["axis"]] if a.get("axis") else []
    cmd += ["--pores", a["pores"]] if a.get("pores") else []
    cmd += ["--membrane-offset", str(a["membrane_offset"])] if a.get("membrane_offset") else []
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.path.join(REPO, "src")})
    return r.stdout + r.stderr


# muted per-protomer colours, the way a cryo-EM paper colours its chains
CHAIN_COLORS = [[0.36, 0.58, 0.80], [0.90, 0.62, 0.36], [0.47, 0.72, 0.52], [0.84, 0.47, 0.47],
                [0.66, 0.56, 0.79], [0.62, 0.50, 0.42], [0.86, 0.60, 0.74], [0.55, 0.70, 0.72],
                [0.80, 0.78, 0.50]]
LIPID_COLOR = [0.82, 0.82, 0.84]

# ---- the base parameters of a mechanosensitive run (MscS, MscL) -----------------------------------
MECHANO = {
    "n_grid": 128,                    # the electrolyte's grid: 128^3 over the box
    "eta_Pa_s": 1.0e-3, "bead_stokes_nm": 0.3,
    "enm_cutoff_nm": 1.0, "enm_k_kT_nm2": 169.0,   # 1 kcal/mol/A^2 (Atilgan 2001's gamma), per spring
    "go_cutoff_nm": 0.8,                            # native tertiary contacts within 8 A (inside and between chains)
    "go_eps_kT": "derived",           # tau_1/2 dA / N_break (GATING above); rounds 1-3 bracketed it at 0.5-2 kT and never opened
    "go_eps_factor": 1.0,             # a multiplier on the derived well, for a sibling that tests the derivation
    "local_max_sep": 4,               # springs only i..i+4 along a chain: helices rigid, packing breakable
    "wca_sigma_nm": 0.45,             # excluded volume between chains
    "area_per_lipid_nm2": 0.65,       # POPC (Kucerka et al. 2011); one bead per lipid per leaflet
    "leaflet_z_nm": 1.0,              # each leaflet's bead plane, +-1 nm from the bilayer centre
    "K_A_mN_m": 230.0,                # bilayer area modulus (Rawicz et al. 2000), REPORTED, not imposed (see eps_LL)
    "lipid_law": "cooke",             # the cosine-tail lipid (Cooke, Kremer & Deserno 2005); "lj" was round 1's
    "eps_LL_kT": 1.5,                 # the Cooke well depth, kT; (lj: 2.3, T* 0.43 -- a liquid near its critical point, it yielded)
    "lipid_tail_over_sigma": 1.6,     # the tail's width w_c / sigma (Cooke 2005's fluid-bilayer value)
    "lipid_sigma_over_a": 0.8909,     # 2^(-1/6): the flat well reaches the seeded spacing a (lj: 0.89 of a)
    "frame_eps_ratio": 1.0,           # the frame's beads ARE frozen lipids (x2 biased the tension read at the frame
                                      # by the first shell's extra adhesion: +9 to +36 mN/m at zero bulk stress)
    "domain_rigid": True,             # Go wells only between domains and chains (none inside a domain)
    "shape_match": True,              # each domain a rigid body (Mueller 2005), hinged by the backbone springs
    "lipid_protein_law": "lj",        # SHORT-range adhesion: a lipid touches the surface it sits on, not 20 residues
    "lipid_protein_eps_kT": 1.0,      # per lipid-residue pair
    "go_exclude": [],                 # domain names whose beads get no Go well (a lock tested by removal)
    "lipid_protein_sigma_nm": 0.75, "lipid_protein_eps_ratio": 1.0,
    "lipid_z_k_kT_nm2": 100.0,        # the hydrophobic core holds a lipid at its depth, +-0.1 nm
    "lipid_excl_nm": 0.75,            # no lipid seeded closer than this to an alpha carbon
    "annulus_nm": 5.0,                # lipid ring beyond the protein's widest transmembrane radius
    "bath_nm": 3.0,                   # water beyond the protein's extent, above and below
    "tension_max_mN_m": 15.0,         # the ramp's top: 1.5 x MscL's tau_1/2, under the ~17 mN/m the Cooke layer carried at most (round 3)
    "ramp_tau": [0.5, 6.0, 2.0, 6.0, 1.0],   # settle, up, hold, down, hold -- in units of the patch's relaxation time
    "psi_mV": -100.0,                 # the cell's potential, inside negative
    "cap_um2": 6.0,                   # the cell's membrane area (E. coli), for its capacitance
    "sigma_S_m": 2.5,                 # 200 mM KCl
    "conduct_every": 100, "conduct_iters": 150,
    "ins_protein_nm": 0.5,            # a residue's reach beyond its alpha carbon: insulator and pore probe
    "ins_lipid_nm": 0.55,             # covers the hexagonal lattice (spacing / sqrt 3 = 0.50 nm)
    "membrane_slab_nm": 4.0,          # the insulating slab outside the lipid beads' own reach
    "kT_noise": 1.0,
    "dt_safety": 0.3,
    "max_frames_cap": 400000,         # ~41 ns, ~60 min on ${CLUSTER_QUEUE_PREFIX}l4: a TM1 helix pulled by ~30 pN drifts ~0.2 nm/ns, 1.4 nm in ~8 ns
    "affine_stretch": True,           # radial_drive scales every lipid with the frame (a barostat's molecular scaling)
    "probe_nm": 0.14,                 # the ion-accessible volume: an ion's centre stays a water radius off every bead
    "n_frames_override": None,        # an integration test's few frames; never set on a record run
    "K_A_drive_mN_m": 55.0,           # the layer's modulus as MEASURED in round 3 (64.3, 60.7, 38.5 mN/m), used to size the
                                      # stretch; round 2's 100 (13 mN/m at 9.6% area strain) read the edge, not the patch
    "drive_mode": "strain",           # radial_drive: STRAIN -- the frame's stretch is prescribed and the tension the
                                      # membrane carries is read (a tension set-point lagged the soft Cooke layer:
                                      # 3 mN/m carried of 10 applied at frame 20,000); "tension" is the set-point form
    # rendering (the cryo-EM look of 0073)
    # the channel large (zoom 1.6 frames the central 62% of the box: the channel ~40% of the frame
    # height, as 0073's motor; the scale bar stays), seen from the side (elev 15, not 40 onto the
    # periplasmic loops), its two front chains not drawn and the slab cut in front of it, so the
    # lumen and the current in it are in view (round 3's judges: "a speck", "no pore to read")
    "camera": {"elev": 40.0, "azim": 30.0}, "zoom": 1.0,     # step 0007's view (the human, 2026-09-26): whole
    "hide_front_chains": 0, "lipid_near_side": False,         # protein, whole slab, no cut-away
    "protein_opacity": 1.0, "lipid_opacity": 0.5,
    "protein_spacing_nm": 0.3, "protein_blur": 1.5, "protein_iso": 0.3,
    "lipid_spacing_nm": 0.4, "lipid_blur": 2.0, "lipid_iso": 0.3,
    "slice_axis": "x", "slice_cmap": "inferno", "slice_opacity": 0.9,
    "near_side": False,
}

# ---- THE WELL OF A BREAKABLE CONTACT, DERIVED (round 3's finding: bracketing it never opened) ------
# At the tension of half-opening the closed and open states are equally likely, so the work the
# membrane does on the protein, tau_1/2 x dA, pays the free energy of the contacts that opening
# breaks: N_break x eps = tau_1/2 x dA, hence eps = tau_1/2 dA / N_break. N_break is COUNTED from the
# model's own native-contact set (the same rule `elastic_network` applies: pairs within go_cutoff
# between chains, or between domains of a chain beyond local_max_sep), less the contacts inside the
# domains the opening leaves together (`kept`). tau_1/2 and dA are the channel's measured values.
# Entropy is left out: the open state's extra freedom would lower the barrier, so this eps is an
# upper bound on the depth that still lets tau_1/2 open the channel.
GATING = {
    # MscL: tau_1/2 ~10 mN/m (Wang 2014, "high pressure (~10 mN/m) causes MscL to open"); dA 20 nm^2,
    # the in-plane expansion of the transition measured in situ (Chiang, Anishkin & Sukharev 2004,
    # Biophys. J. 86:2846) -- NOT in the papers folder, stated. The iris (Betanzos et al. 2002, cited
    # by Wang 2014) tilts and slides TM1, TM2 and S1 apart; the C-terminal bundle stays a bundle.
    "mscl": {"tension_half_mN_m": 10.0, "dA_nm2": 20.0, "kept": ["ctd"],
             "source": "tau_1/2 Wang 2014; dA Chiang et al. 2004 (not in the papers folder)",
             # Wang et al. 2014's own open model (Source code 2, backbone), for the two-basin runs
             "open_cif": "elife-01834-code2-v1.pdb"},
    # MscS: tau_1/2 the band's centre (4-7 mN/m, Reddy 2019 and the literature it cites); the open state
    # the A106V structure 2VV5 (Wang et al. 2008), which Reddy 2019 cites as the expanded state
    "mscs": {"tension_half_mN_m": 5.5, "dA_nm2": None, "kept": ["cage"], "open_cif": "2VV5.cif",
             "source": "tau_1/2 the band's centre; dA and N_break from 6PWN and 2VV5"},
    # TRAAK: activated "over a broad tension range from ~0.5-4 mN/m to ~12 mN/m" and expanding "up to 2.7 nm2
    # in the conductive state" (Brohawn et al. 2014); tau_1/2 6 mN/m, the range's middle -- stated
    "traak": {"tension_half_mN_m": 6.0, "dA_nm2": 2.7, "kept": ["all"], "open_cif": "4WFE.cif",
              "source": "Brohawn et al. 2014 (papers/nihms639323.pdf); tau_1/2 the middle of the activation range"},
    # alpha-hemolysin: the prepore IV (9M4A) -> pore (9M4P) of Chatterjee 2026, superposed on the cap; NO offset
    # (tau_1/2 0): what drives the stems down is the membrane's pull on their residues (`insertion_drive`)
    "ahl": {"tension_half_mN_m": 0.0, "dA_nm2": None, "kept": ["cap"], "open_cif": "9M4P.cif",
            "source": "Chatterjee et al. 2026: prepore IV and pore in red-cell membranes"},
}


def _outline_area(X, zwin=1.5, reach=0.5, nbin=72):
    """The in-plane area inside a protein's outline across the membrane's core: the outermost alpha
    carbon within |z| < zwin (nm, bilayer centre z = 0) in each of `nbin` sectors about the axis,
    plus a residue's reach beyond it; A = 1/2 sum r^2 dtheta."""
    m = np.abs(X[:, 2]) < zwin
    th = np.arctan2(X[m, 1], X[m, 0]); r = np.hypot(X[m, 0], X[m, 1]) + reach
    edges = np.linspace(-np.pi, np.pi, nbin + 1)
    rmax = np.array([r[(th >= edges[k]) & (th < edges[k + 1])].max() if ((th >= edges[k]) & (th < edges[k + 1])).any()
                     else np.nan for k in range(nbin)])
    rmax = np.where(np.isnan(rmax), np.nanmean(rmax), rmax)
    return float(0.5 * (rmax ** 2).sum() * (2 * np.pi / nbin))


def _read_ca(path):
    """Alpha carbons of model 1 of an mmCIF or PDB file: (xyz nm, resid, chain). A modelled state
    (Wang et al. 2014's open MscL, their Source code 2) comes as PDB; deposited ones as mmCIF."""
    if path.lower().endswith((".pdb", ".ent", ".pdb.gz")):
        import gzip
        op_ = gzip.open if path.endswith(".gz") else open
        X, R, C = [], [], []
        with op_(path, "rt", errors="replace") as f:
            for line in f:
                if line.startswith("ENDMDL"):
                    break
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    X.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    # the SEGMENT id first: a modelled oligomer (Wang 2014's MscL) labels all five subunits
                    # chain P and tells them apart only by segid P1-P5
                    R.append(int(line[22:26])); C.append(line[72:76].strip() or line[21].strip() or "A")
        return np.array(X) / 10.0, np.array(R), np.array(C)
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from channel_anatomy import read_atoms
    a = read_atoms(path, assembly=False)
    m = a["atom"] == "CA"
    return a["xyz"][m] / 10.0, a["resid"][m], a["chain"][m]


def _open_state(g, channel, pts, folder):
    """The OPEN deposited state in the model's own frame: its alpha carbons keyed by (model chain k,
    resid), superposed on the closed model over the domain the opening leaves together (`kept`),
    with the chains matched by the best of the ring's 2n relabellings (rotations, and the mirror
    order). Returns (rms of that superposition in nm, the key -> position map)."""
    Xo, ro, co = _read_ca(os.path.join(DATA, g["open_cif"]))
    info = json.load(open(os.path.join(folder, "anatomy.json")))
    bl = dict(np.load(os.path.join(folder, "blocks.npz")))
    n = len(info["chains"])
    cho = sorted(set(co))
    kd = info["domains"].index(g["kept"][0])
    model = {(k, int(r)): x for k in range(n)
             for x, r in zip(pts[f"c{k}"] * 1e9, bl[f"c{k}_resid"].ravel())}
    kept_keys = [(k, int(r)) for k in range(n)
                 for r, d in zip(bl[f"c{k}_resid"].ravel(), bl[f"c{k}_domain"].ravel()) if int(d) == kd]
    opos = {(c, int(r)): x for x, r, c in zip(Xo, ro, co)}

    def kabsch(A, B):                        # R, t minimising |A @ R.T + t - B|
        ca_, cb_ = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca_).T @ (B - cb_))
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        return R, cb_ - ca_ @ R.T
    best = None
    for sgn in (1, -1):
        for off in range(n):
            lab = {k: cho[(off + sgn * k) % n] for k in range(n)}
            ks = [key for key in kept_keys if (lab[key[0]], key[1]) in opos]
            A = np.array([opos[(lab[k], r)] for k, r in ks]); B = np.array([model[(k, r)] for k, r in ks])
            R, t = kabsch(A, B)
            rms = float(np.sqrt((((A @ R.T + t) - B) ** 2).sum(1).mean()))
            if best is None or rms < best[0]:
                best = (rms, lab, R, t)
    rms, lab, R, t = best
    inv = {v: k for k, v in lab.items()}
    return rms, {(inv[c], r): x @ R.T + t for (c, r), x in opos.items() if c in inv}


def go_eps_derived(channel, P, pts, folder):
    """eps (kT) = tau_1/2 dA / N_break, with N_break counted on the model's own contact set."""
    g = GATING[channel]
    chains = sorted([k for k in pts if k.startswith("c") and k[1:].isdigit()], key=lambda s_: int(s_[1:]))
    bl = dict(np.load(os.path.join(folder, "blocks.npz")))
    info = json.load(open(os.path.join(folder, "anatomy.json")))
    doms = info["domains"]
    X = np.concatenate([pts[c] * 1e9 for c in chains])
    ch = np.concatenate([np.full(len(pts[c]), i) for i, c in enumerate(chains)])
    idx = np.concatenate([np.arange(len(pts[c])) for c in chains])
    dm = np.concatenate([bl[f"{c}_domain"].astype(int).ravel() for c in chains])
    d = np.linalg.norm(X[:, None] - X[None], axis=-1)
    iu = np.triu(np.ones(d.shape, bool), 1)
    near = iu & (d < P["go_cutoff_nm"])
    other = ch[:, None] != ch[None]
    intra = (~other) & (np.abs(idx[:, None] - idx[None]) > P["local_max_sep"])
    if P["domain_rigid"]:
        intra &= dm[:, None] != dm[None]
    go = near & (other | intra)
    for ex in P["go_exclude"]:
        k = doms.index(ex)
        go &= ~((dm[:, None] == k) | (dm[None] == k))
    kept = np.zeros_like(go)
    for kd in g["kept"]:
        k = doms.index(kd)
        kept |= (dm[:, None] == k) & (dm[None] == k)
    n_all, n_break = int(go.sum()), int((go & ~kept).sum())
    extra = {}
    if P.get("go_eps_between_chains_kT") == "derived":
        # ONLY THE INTERFACES BETWEEN SUBUNITS BREAK: each subunit keeps its own packing (its wells stay
        # at go_eps_kT), and so do the kept domains' inter-chain contacts
        n_break = int((go & other & ~kept).sum())
        extra = {"counted": "contacts between chains, less those inside the kept domains"}
    if g.get("open_cif") and os.path.isfile(os.path.join(DATA, g["open_cif"])):
        # MEASURED ON THE OPEN STRUCTURE: a contact is broken when opening stretches it by more than
        # 0.3 nm (the 12-6 well keeps < 1/5 of its depth there); dA from the two outlines.
        rms, opos = _open_state(g, channel, pts, folder)
        keys = [(int(k), int(r)) for k, r in zip(ch, np.concatenate([bl[f"{c}_resid"].ravel() for c in chains]))]
        I, J = np.where(go)
        ev = [(i, j) for i, j in zip(I, J) if keys[i] in opos and keys[j] in opos]
        brk = [(i, j) for i, j in ev if np.linalg.norm(opos[keys[i]] - opos[keys[j]]) > d[i, j] + 0.3]
        n_break = len(brk)
        A_c = _outline_area(X)
        A_o = _outline_area(np.array(list(opos.values())))
        g = {**g, "dA_nm2": round(A_o - A_c, 2)}
        extra = {"open_cif": g["open_cif"], "superposition_rms_nm": round(rms, 3), "contacts_evaluable": len(ev),
                 "area_closed_nm2": round(A_c, 1), "area_open_nm2": round(A_o, 1)}
        print(f"  THE OPEN STATE, {g['open_cif']}: superposed on the model's {g['kept'][0]} at {rms:.3f} nm rms; "
              f"{n_break} of the {len(ev)} contacts it can place stretch by > 0.3 nm ({n_all} in all); "
              f"in-plane area {A_c:.1f} -> {A_o:.1f} nm^2")
    work_kT = g["tension_half_mN_m"] * g["dA_nm2"] / 4.11              # mN/m x nm^2 = pN nm; kT = 4.11 pN nm
    eps = work_kT / max(n_break, 1)
    note = {"go_contacts": n_all, "go_contacts_broken": n_break, "kept_domains": g["kept"],
            "tension_half_mN_m": g["tension_half_mN_m"], "dA_nm2": g["dA_nm2"],
            "work_kT": round(work_kT, 2), "go_eps_kT": round(eps, 4), "source": g["source"], **extra}
    print(f"  THE WELL, DERIVED: tau_1/2 {g['tension_half_mN_m']} mN/m x dA {g['dA_nm2']} nm^2 = {work_kT:.1f} kT, "
          f"shared by {n_break} of the model's {n_all} native contacts (all but those inside {g['kept']}) "
          f"-> {eps:.4f} kT per contact  [{g['source']}]")
    return eps, note


VERSIONS = {}          # label -> {"channel", "parent", "change", "why"}
# VERSIONS THAT HAVE RUN ARE FROZEN: their config/channel/exp04_v<label>.yaml is the spec the record
# links to, and the base above has moved since (round 1 ran on the Lennard-Jones layer, without
# domain rigidity or the force cap). Regenerating one would rewrite history under the table.
RAN = {"H1", "H2", "H3", "1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c", "4a", "4b", "4c", "5a", "5b", "5c", "6a", "6b", "6c", "7a", "7b", "7c", "8a", "8b", "8c"}


def version(label, channel, parent=None, change=None, why=""):
    VERSIONS[label] = {"channel": channel, "parent": parent, "change": change or {}, "why": why}


def flow_version(label, shape, parent=None, change=None, why=""):
    """A version built by `build_flow` (the ion-flow rig) on the deposited state RIG[shape]."""
    VERSIONS[label] = {"channel": RIG[shape]["channel"], "parent": parent, "change": {"shape": shape, **(change or {})},
                       "why": why, "builder": "flow"}


# ---- the integration fixture (tests/, /tmp; never a row of the experiment) ------------------
# ---- H: the ion-flow test on a bare hole (no protein): the element every channel's flow is counted with --
version("H1", "hole", None, {}, "a 1 nm hole in a sealed membrane, K+ and Cl- half on each side, -100 mV clamped")
version("H2", "hole", "H1", {"hole_nm": 0.0}, "H1 with the hole shut (the control: nothing may cross)")
version("H3", "hole", "H1", {"psi_mV": 100.0}, "H1 at +100 mV (the flow must reverse)")
version("H4", "hole", "H1", {}, "H1 with the ions' 8 pair potentials merged into 2 -- ions with ions under "
        "Coulomb + WCA with per-set charge, size and mobility, ions with the lipids -- the same forces")
version("H5", "hole", "H4", {"hole_nm": 0.0}, "H4 with the hole shut (the control, merged form)")
version("H6", "hole", "H4", {"psi_mV": 100.0}, "H4 at +100 mV (the flow must reverse, merged form)")
version("Ht", "hole", "H1", {"n_frames_override": 20000}, "hole crash check (/tmp, never a row)")
# ---- BATCH 10: ALL TEN CHANNELS IN THE ION-FLOW RIG, TWO JOBS EACH (the human, 2026-09-26) -------------------
# Each channel's two deposited states held static in a sealed membrane placed by OPM, K+ and Cl- half on each
# side, -100 mV clamped, the ions counted: does the closed state stop the flow and the open state pass it?
# A channel with one state (Kv1.2, gramicidin A, OmpF) is run at -100 and +100 mV (rectification, selectivity).
for _lab, _shape, _chg, _why in (
        ("10a", "mscs_c", {}, "MscS closed (6PWN, nanodisc) in the ion-flow rig"),
        ("10b", "mscs_o", {}, "MscS open (2VV5) in the ion-flow rig"),
        ("10c", "mscl_c", {}, "MscL closed (2OAR) in the ion-flow rig"),
        ("10d", "mscl_o", {}, "MscL open (Wang 2014's model) in the ion-flow rig"),
        ("10e", "kv12", {}, "Kv1.2 conducting (8VC6) in the ion-flow rig at -100 mV"),
        ("10f", "kv12", {"psi_mV": 100.0}, "Kv1.2 conducting (8VC6) in the ion-flow rig at +100 mV"),
        ("10g", "ahl_pre", {}, "alpha-hemolysin prepore IV (9M4A) in the ion-flow rig: stems not across the membrane"),
        ("10h", "ahl_pore", {}, "alpha-hemolysin pore (9M4P) in the ion-flow rig"),
        ("10i", "kcsa_c", {}, "KcsA closed (1K4C) in the ion-flow rig"),
        ("10j", "kcsa_o", {}, "KcsA open (3F5W) in the ion-flow rig"),
        ("10k", "gA", {}, "gramicidin A dimer (1MAG) in the ion-flow rig at -100 mV"),
        ("10l", "gA", {"psi_mV": 100.0}, "gramicidin A dimer (1MAG) in the ion-flow rig at +100 mV"),
        ("10m", "glic_c", {}, "GLIC closed (4NPQ, pH 7) in the ion-flow rig"),
        ("10n", "glic_o", {}, "GLIC open (4HFI, pH 4) in the ion-flow rig"),
        ("10o", "navab_c", {}, "NavAb closed (5VB2) in the ion-flow rig"),
        ("10p", "navab_o", {}, "NavAb open (5VB8) in the ion-flow rig"),
        ("10q", "traak_c", {}, "TRAAK non-conductive (4WFF) in the ion-flow rig"),
        ("10r", "traak_o", {}, "TRAAK conductive (4WFE) in the ion-flow rig"),
        ("10s", "ompf", {}, "OmpF trimer (2OMF) in the ion-flow rig at -100 mV"),
        ("10t", "ompf", {"psi_mV": 100.0}, "OmpF trimer (2OMF) in the ion-flow rig at +100 mV")):
    flow_version(_lab, _shape, None, _chg, _why)
# ---- BATCH 11 (2026-09-26): THE GATE MOVES for the tension-gated three; the other seven counted at 1 M --------
# MscS, MscL, TRAAK: the closed state in a live membrane (Cooke lipids, the patch's frame), the two-basin elastic
# network with the open state as its second basin (dV = tau_1/2 x dA), the ions in the rig's layer with the
# measured closed and open paths blended by the protein's own basin weight: ramped to 12 mN/m (a) against the
# same with no tension (b) -- does tension open it, and do the ions then flow?
_G11 = {"ions": True, "two_basin": True, "go_eps_kT": 1.0, "psi_mV": -100.0, "tension_max_mN_m": 12.0}
version("11a", "mscs", None, {**_G11, "rig_shape": "mscs_c", "rig_open": "mscs_o"},
        "MscS: the gate moves -- 6PWN in a live membrane, 2VV5 as the open basin, ions, tension ramped to 12 mN/m")
version("11b", "mscs", "11a", {"tension_max_mN_m": 0.0}, "11a with no tension (the control: must stay shut)")
version("11c", "mscl", None, {**_G11, "rig_shape": "mscl_c", "rig_open": "mscl_o"},
        "MscL: the gate moves -- 2OAR in a live membrane, Wang 2014's model as the open basin, ions, 12 mN/m")
version("11d", "mscl", "11c", {"tension_max_mN_m": 0.0}, "11c with no tension (the control)")
version("11q", "traak", None, {**_G11, "rig_shape": "traak_c", "rig_open": "traak_o"},
        "TRAAK: the gate moves -- 4WFF (cavity decane) in a live membrane, 4WFE as the open basin, ions, 12 mN/m")
version("11r", "traak", "11q", {"tension_max_mN_m": 0.0}, "11q with no tension (the control)")
# the seven others: the batch-10 states counted at 1 M salt, where a channel's current is ~5-7x larger
_C11 = {"conc_mM": 1000.0, "psi_mV": -200.0}
for _lab, _shape, _chg, _why in (
        ("11e", "kv12", {}, "Kv1.2 (8VC6) at 1 M KCl, -200 mV"),
        ("11f", "kv12", {"psi_mV": 200.0}, "Kv1.2 (8VC6) at 1 M KCl, +200 mV"),
        ("11g", "ahl_pre", {}, "alpha-hemolysin prepore (9M4A) at 1 M KCl, -200 mV"),
        ("11h", "ahl_pore", {}, "alpha-hemolysin pore (9M4P) at 1 M KCl, -200 mV"),
        ("11i", "kcsa_c", {}, "KcsA closed (1K4C) at 1 M KCl, -200 mV"),
        ("11j", "kcsa_o", {}, "KcsA open (3F5W) at 1 M KCl, -200 mV"),
        ("11k", "gA", {}, "gramicidin A dimer (1MAG) at 1 M KCl, -200 mV"),
        ("11l", "gA_apart", {}, "gramicidin A monomers 1 nm apart (constructed closed state) at 1 M KCl, -200 mV"),
        ("11m", "glic_c", {}, "GLIC closed (4NPQ) at 1 M KCl, -200 mV"),
        ("11n", "glic_o", {}, "GLIC open (4HFI) at 1 M KCl, -200 mV"),
        ("11o", "navab_c", {"cation": "Na"}, "NavAb closed (5VB2) at 1 M NaCl, -200 mV (Na+, its own ion)"),
        ("11p", "navab_o", {"cation": "Na"}, "NavAb open (5VB8) at 1 M NaCl, -200 mV"),
        ("11s", "ompf", {"psi_mV": -100.0}, "OmpF (2OMF) at 1 M KCl, -100 mV (Cowan 1992's condition: 0.8 nS)"),
        ("11t", "ompf", {"psi_mV": 100.0}, "OmpF (2OMF) at 1 M KCl, +100 mV")):
    flow_version(_lab, _shape, None, {**_C11, **_chg}, _why)
flow_version("Ft", "mscl_o", None, {"n_frames_override": 400}, "ion-flow rig crash check (/tmp, never a row)")
version("Gt", "mscl", "11c", {"n_frames_override": 600}, "moving-gate crash check (/tmp, never a row)")
# ---- BATCH 12 (2026-09-26): THE GATE AS ONE COORDINATE, and the rig's filters breathing ------------------------------
# MscS, MscL, TRAAK: `morph_gate` -- the protein on the straight path between its closed and open deposited states,
# moved along it by the live membrane's push against dV = tau_1/2 x dA; tension ramped to 12 mN/m (a) against none
# (b). Alpha-hemolysin: the same path from the prepore IV to the pore, driven by the stems' Wimley-White pull into
# the core (`insertion_drive`, g) against the same without it (h), no tension, no offset. The six others: the rig
# again at 1 M with the batch-12 rig -- carbonyls that breathe (B-factor tethers), the dry band 0.25-0.33 nm, the
# cations' density drawn as the field.
def mech_version(label, channel, parent, change, why):
    version(label, channel, parent, change, why)
    VERSIONS[label]["builder"] = "mechano"
_M12 = {"ions": True, "two_basin": True, "morph": True, "go_eps_kT": 1.0, "psi_mV": -100.0, "tension_max_mN_m": 12.0}
mech_version("12a", "mscs", None, {**_M12, "rig_shape": "mscs_c", "rig_open": "mscs_o"},
             "MscS on its closed -> open path (morph_gate), live membrane, ions, tension ramped to 12 mN/m")
mech_version("12b", "mscs", "12a", {"tension_max_mN_m": 0.0}, "12a with no tension (the control)")
mech_version("12c", "mscl", None, {**_M12, "rig_shape": "mscl_c", "rig_open": "mscl_o"},
             "MscL on its closed -> open path (morph_gate), live membrane, ions, 12 mN/m")
mech_version("12d", "mscl", "12c", {"tension_max_mN_m": 0.0}, "12c with no tension (the control)")
mech_version("12q", "traak", None, {**_M12, "rig_shape": "traak_c", "rig_open": "traak_o"},
             "TRAAK on its non-conductive -> conductive path (morph_gate), live membrane, ions, 12 mN/m")
mech_version("12r", "traak", "12q", {"tension_max_mN_m": 0.0}, "12q with no tension (the control)")
mech_version("12g", "ahl", None, {**_M12, "rig_shape": "ahl_pre", "rig_open": "ahl_pore", "tension_max_mN_m": 0.0,
                                  "insertion_drive": True},
             "alpha-hemolysin on its prepore -> pore path, driven by the stems' hydrophobic pull into the core; ions")
mech_version("12h", "ahl", "12g", {"insertion_drive": False}, "12g without the insertion drive (the control)")
_C12 = {"conc_mM": 1000.0, "psi_mV": -200.0}
for _lab, _shape, _chg, _why in (
        ("12e", "kv12", {}, "Kv1.2 (8VC6), batch-12 rig (breathing filter), 1 M KCl, -200 mV"),
        ("12f", "kv12", {"psi_mV": 200.0}, "Kv1.2 (8VC6), batch-12 rig, 1 M KCl, +200 mV"),
        ("12i", "kcsa_c", {}, "KcsA closed (1K4C), batch-12 rig, 1 M KCl, -200 mV"),
        ("12j", "kcsa_o", {}, "KcsA open (3F5W), batch-12 rig, 1 M KCl, -200 mV"),
        ("12k", "gA", {}, "gramicidin A dimer, batch-12 rig, 1 M KCl, -200 mV"),
        ("12l", "gA_apart", {}, "gramicidin A monomers apart, batch-12 rig, 1 M KCl, -200 mV"),
        ("12m", "glic_c", {}, "GLIC closed (4NPQ), batch-12 rig (dry band 0.25-0.33 nm), 1 M KCl, -200 mV"),
        ("12n", "glic_o", {}, "GLIC open (4HFI), batch-12 rig, 1 M KCl, -200 mV"),
        ("12o", "navab_c", {"cation": "Na"}, "NavAb closed (5VB2), batch-12 rig, 1 M NaCl, -200 mV"),
        ("12p", "navab_o", {"cation": "Na"}, "NavAb open (5VB8), batch-12 rig, 1 M NaCl, -200 mV"),
        ("12s", "ompf", {"psi_mV": -100.0}, "OmpF (2OMF), batch-12 rig, 1 M KCl, -100 mV"),
        ("12t", "ompf", {"psi_mV": 100.0}, "OmpF (2OMF), batch-12 rig, 1 M KCl, +100 mV")):
    flow_version(_lab, _shape, None, {**_C12, **_chg}, _why)
# ---- BATCH 13 (2026-09-26), the static channels: the K+ filters' carbonyls at K+-O 0.25 nm closest approach
# (q_fil_radius 0.112 nm), KcsA's CONDUCTING open state (3F7Y, 17 A), the forbidden core a hard constraint;
# GLIC and NavAb, open in batch 12, at physiological salt and reversed; OmpF at 150 mM and 1 M
_R13 = {"conc_mM": 1000.0, "psi_mV": -200.0, "q_fil_radius_nm": 0.112}
for _lab, _shape, _chg, _why in (
        ("13e", "kv12", {}, "Kv1.2, carbonyls at K+-O 0.25 nm, 1 M KCl, -200 mV"),
        ("13f", "kv12", {"psi_mV": 200.0}, "Kv1.2, carbonyls at K+-O 0.25 nm, 1 M KCl, +200 mV"),
        ("13i", "kcsa_c", {}, "KcsA closed (1K4C), carbonyls at 0.25 nm, 1 M KCl, -200 mV"),
        ("13j", "kcsa_o17", {}, "KcsA OPEN WITH A CONDUCTING FILTER (3F7Y, 17 A), carbonyls at 0.25 nm, 1 M KCl, -200 mV"),
        ("13k", "gA", {}, "gramicidin A dimer, carbonyls at 0.25 nm, 1 M KCl, -200 mV"),
        ("13l", "gA_apart", {}, "gramicidin A monomers apart, carbonyls at 0.25 nm, 1 M KCl, -200 mV"),
        ("13m", "glic_o", {"conc_mM": 150.0, "psi_mV": -100.0}, "GLIC open (4HFI) at 150 mM KCl, -100 mV"),
        ("13n", "glic_o", {"psi_mV": 200.0}, "GLIC open (4HFI) at 1 M KCl, +200 mV (the flow reversed)"),
        ("13o", "navab_o", {"cation": "Na", "conc_mM": 150.0, "psi_mV": -100.0}, "NavAb open (5VB8) at 150 mM NaCl, -100 mV"),
        ("13p", "navab_o", {"cation": "Na", "psi_mV": 200.0}, "NavAb open (5VB8) at 1 M NaCl, +200 mV (reversed)"),
        ("13s", "ompf", {"conc_mM": 150.0, "psi_mV": -100.0}, "OmpF at 150 mM KCl, -100 mV (physiological)"),
        ("13t", "ompf", {}, "OmpF at 1 M KCl, -200 mV")):
    flow_version(_lab, _shape, None, {**_R13, **_chg}, _why)
flow_version("R13", "kv12", None, {**_R13, "n_frames_override": 1500}, "batch-13 render check (/tmp, never a row)")
# ---- BATCH 14 (2026-09-26): THE RIG WITH EVERY PATH FIX -- a path's FULL step width added (a filter's single file
# had no free core: ~9 kT on the axis all along it), its centre line smoothed (gramicidin's zig-zagged 0.1-0.2 nm),
# the wall looked for out to R + max(1, R) nm (open MscS / MscL read as "not enclosed" at their widest), the
# carbonyls breathing at K+-O 0.25 nm, the filter STARTING with its crystal ions (or its O-ring sites), the core a
# hard constraint. Every channel, closed and open again, at 1 M, -200 mV (MscS, MscL, alpha-hemolysin at 150 mM).
_S14 = {"conc_mM": 1000.0, "psi_mV": -200.0, "q_fil_radius_nm": 0.112, "xtal_ions": True}
for _lab, _shape, _chg, _why in (
        ("14a", "mscs_c", {"conc_mM": 150.0, "psi_mV": -100.0}, "MscS closed, the fixed rig, 150 mM, -100 mV"),
        ("14b", "mscs_o", {"conc_mM": 150.0, "psi_mV": -100.0}, "MscS open (2VV5), the fixed rig, 150 mM, -100 mV"),
        ("14c", "mscl_c", {"conc_mM": 150.0, "psi_mV": -100.0}, "MscL closed, the fixed rig, 150 mM, -100 mV"),
        ("14d", "mscl_o", {"conc_mM": 150.0, "psi_mV": -100.0}, "MscL open (Wang model), the fixed rig, 150 mM, -100 mV"),
        ("14e", "kv12", {}, "Kv1.2, the fixed rig: a free single file, its filter seeded at S1/S3, 1 M, -200 mV"),
        ("14f", "kv12", {"psi_mV": 200.0}, "Kv1.2, the fixed rig, 1 M, +200 mV"),
        ("14g", "ahl_pre", {"conc_mM": 150.0, "psi_mV": -100.0}, "alpha-hemolysin prepore, the fixed rig, 150 mM, -100 mV"),
        ("14h", "ahl_pore", {"conc_mM": 150.0, "psi_mV": -100.0}, "alpha-hemolysin pore, the fixed rig, 150 mM, -100 mV"),
        ("14i", "kcsa_c", {}, "KcsA closed (1K4C), the fixed rig, its crystal K+ in the filter, 1 M, -200 mV"),
        ("14j", "kcsa_o17", {}, "KcsA open 17 A (3F7Y), the fixed rig, filter seeded, 1 M, -200 mV"),
        ("14k", "gA", {}, "gramicidin A dimer, the fixed rig (a followable centre line), 1 M, -200 mV"),
        ("14l", "gA_apart", {}, "gramicidin A monomers apart, the fixed rig, 1 M, -200 mV"),
        ("14m", "glic_c", {}, "GLIC closed, the fixed rig, 1 M, -200 mV"),
        ("14n", "glic_o", {}, "GLIC open, the fixed rig, 1 M, -200 mV"),
        ("14o", "navab_c", {"cation": "Na"}, "NavAb closed, the fixed rig, 1 M NaCl, -200 mV"),
        ("14p", "navab_o", {"cation": "Na"}, "NavAb open, the fixed rig, 1 M NaCl, -200 mV"),
        ("14q", "traak_c", {}, "TRAAK non-conductive (cavity decane), the fixed rig, crystal K+, 1 M, -200 mV"),
        ("14r", "traak_o", {}, "TRAAK conductive (4WFE), the fixed rig, crystal K+, 1 M, -200 mV"),
        ("14s", "ompf", {"psi_mV": -100.0}, "OmpF, the fixed rig, 1 M, -100 mV"),
        ("14t", "ompf", {"conc_mM": 150.0, "psi_mV": -100.0}, "OmpF, the fixed rig, 150 mM, -100 mV")):
    flow_version(_lab, _shape, None, {**_S14, **_chg}, _why)
flow_version("S14a", "gA", None, {"n_frames_override": 300, "lipid_blur": 1.0, "lipid_spacing_nm": 0.3}, "slab-look check (/tmp, never a row)")
flow_version("S14b", "gA", None, {"n_frames_override": 300, "lipid_blur": 0.7, "lipid_spacing_nm": 0.25}, "slab-look check (/tmp, never a row)")
flow_version("X14", "kv12", None, {**_R13, "xtal_ions": True, "n_frames_override": 1500}, "filter-seeding check (/tmp, never a row)")
flow_version("X14c", "kcsa_c", None, {**_R13, "xtal_ions": True, "n_frames_override": 1500}, "filter-seeding check (/tmp, never a row)")
# ---- BATCH 13, the moving gates: batch 12's morph gates did not feel the tension (MscS lambda 0.093 at 12 mN/m
# against 0.095 without). (a) the lipids' pull made stronger -- lipid-protein adhesion 1 -> 2 kT per pair (5c's
# 2.5 tore the membrane); (b) the membrane's own MEASURED tension doing tau dA of work on the gate, a stated
# coupling (morph_gate.tension_block). TRAAK with its paper's offset (tau_1/2 x 2.7 nm^2 = 3.9 kT, closed favoured
# at rest) and the work term, 12 mN/m (q) against none (r). Alpha-hemolysin: the insertion drive with (h) and
# without (g) a 20 kT barrier along the path; the core outside every path now a hard constraint.
_M13 = {**_M12, "q_fil_radius_nm": 0.112, "xtal_ions": True}
mech_version("13a", "mscs", None, {**_M13, "rig_shape": "mscs_c", "rig_open": "mscs_o", "lipid_protein_eps_kT": 2.0},
             "MscS on its path, the lipids' pull stronger (adhesion 2 kT per pair), 12 mN/m")
mech_version("13b", "mscs", None, {**_M13, "rig_shape": "mscs_c", "rig_open": "mscs_o", "morph_tension_work": True},
             "MscS on its path, the membrane's measured tension doing tau dA of work on the gate (stated), 12 mN/m")
mech_version("13c", "mscl", None, {**_M13, "rig_shape": "mscl_c", "rig_open": "mscl_o", "lipid_protein_eps_kT": 2.0},
             "MscL on its path, the lipids' pull stronger (adhesion 2 kT per pair), 12 mN/m")
mech_version("13d", "mscl", None, {**_M13, "rig_shape": "mscl_c", "rig_open": "mscl_o", "morph_tension_work": True},
             "MscL on its path, the membrane's measured tension doing tau dA of work on the gate (stated), 12 mN/m")
mech_version("13q", "traak", None, {**_M13, "rig_shape": "traak_c", "rig_open": "traak_o", "morph_tension_work": True,
                                    "dV_kT": 3.9},
             "TRAAK on its path with its paper's offset (3.9 kT) and the tension work term, 12 mN/m")
mech_version("13r", "traak", "13q", {"tension_max_mN_m": 0.0}, "13q with no tension (the control)")
mech_version("13g", "ahl", None, {**_M13, "rig_shape": "ahl_pre", "rig_open": "ahl_pore", "tension_max_mN_m": 0.0,
                                  "insertion_drive": True},
             "alpha-hemolysin prepore -> pore driven by the stems' hydrophobic pull; the hard core; no path barrier")
mech_version("13h", "ahl", "13g", {"morph_barrier_kT": 20.0}, "13g with a 20 kT barrier along the path")
mech_version("Mt", "mscs", "12a", {"n_frames_override": 600}, "morph-gate crash check (/tmp, never a row)")
mech_version("Ma", "ahl", "12g", {"n_frames_override": 600}, "morph insertion crash check (/tmp, never a row)")
flow_version("Fa", "mscs_c", None, {"n_frames_override": 300}, "ion-flow rig crash check, the largest box (/tmp, never a row)")
flow_version("Fs", "ompf", None, {"n_frames_override": 300}, "ion-flow rig crash check, three paths (/tmp, never a row)")
flow_version("Fd", "mscl_o", None, {}, "refactor check: must equal 10d (/tmp, never a row)")
flow_version("Fg", "ahl_pre", None, {"n_frames_override": 3000}, "barrier-slope check on the prepore (/tmp, never a row)")
flow_version("Fe", "kv12", None, {"n_frames_override": 900}, "render check: the slab without holes (/tmp, never a row)")
flow_version("Fv", "kv12", None, {"conc_mM": 1000.0, "psi_mV": -200.0, "n_frames_override": 60000},
             "breathing-filter check: Kv1.2 at 1 M, -200 mV (/tmp, never a row)")
flow_version("Fk", "gA", None, {"n_frames_override": 900}, "render check: gramicidin drawn (/tmp, never a row)")
flow_version("Fb", "mscs_o", None, {"n_frames_override": 3000}, "barrier-slope check on open MscS (/tmp, never a row)")
version("0t", "selftest", None, {"n_frames_override": 600, "conduct_every": 50, "n_grid": 64},
        "integration smoke test on a synthetic ring: every operator, a few frames, no movie")
# ---- the membrane calibration (fixture, /tmp; the patch held at its seeded area, the carried tension read)
for _k, _s in (("c95", 0.95), ("c100", 1.00), ("c105", 1.05), ("d90", 0.90), ("d93", 0.93), ("d96", 0.96)):
    version(_k, "selftest", None, {"n_frames_override": 6000, "conduct_every": 3000, "n_grid": 64,
                                   "drive_mode": "strain", "tension_max_mN_m": 0.0, "lipid_sigma_over_a": _s},
            f"membrane calibration: Cooke lipids at sigma/a {_s}, patch area fixed at the seeded 0.65 nm^2 per lipid")
# ---- round 1: MscS, three siblings of the base, bracketing the one number nobody measured ---------
version("1a", "mscs", None, {"go_eps_kT": 1.0},
        "MscS base, tertiary Go contacts 1.0 kT each (the low bracket: does the fold hold?)")
version("1b", "mscs", None, {"go_eps_kT": 1.5},
        "MscS base, tertiary Go contacts 1.5 kT each (the middle)")
version("1c", "mscs", None, {"go_eps_kT": 2.0},
        "MscS base, tertiary Go contacts 2.0 kT each (the high bracket: does it still open?)")
# ---- round 2: MscS again, on the corrected base (Cooke lipids, rigid domains, the force cap) --------
# Round 1 blew up before it could answer anything; its three changes to the base are recorded in
# its rows. The siblings bracket the one unmeasured number again: the well of a breakable contact,
# now only BETWEEN domains and chains.
version("2a", "mscs", None, {"go_eps_kT": 1.0},
        "MscS on the corrected base, interface Go wells 1.0 kT (the low bracket)")
version("2b", "mscs", None, {"go_eps_kT": 1.5},
        "MscS on the corrected base, interface Go wells 1.5 kT (the middle)")
version("2c", "mscs", None, {"go_eps_kT": 2.0},
        "MscS on the corrected base, interface Go wells 2.0 kT (the high bracket)")
# ---- round 4: Kv1.2, the filter's permittivity bracketed (how hard ions in it push on each other) ----
version("4a", "kv12", None, {"slab_eps_ratio": 2.0}, "Kv1.2, eps in the filter half of water's (ratio 2)")
version("4b", "kv12", None, {"slab_eps_ratio": 4.0}, "Kv1.2, eps in the filter a quarter of water's (ratio 4)")
version("4c", "kv12", None, {"slab_eps_ratio": 8.0}, "Kv1.2, eps in the filter an eighth of water's (ratio 8)")
version("4t", "kv12", None, {"slab_eps_ratio": 4.0, "n_frames_override": 3000},
        "Kv1.2 crash check (/tmp, never a row)")
# ---- round 8: Kv1.2, the base after round 6 (its judges): the Born barrier narrowed to 0.35 nm over the
# filter's heights (no bypass), the one electrolyte solve converged (60,000 sweeps: round 6 left 0.1-0.6
# mV/nm in the baths), the ions' cores guarded above their pull (WCA 0.2 nm/step, Coulomb 0.03), the two
# drawn chains translucent (0.45) so the filter's ions show; permittivity ratio 8 (6b's) --------------
# With the walls above the pull the filter binds less: a local 150k-frame run at ratio 8 held 0.14 ions
# (6b: 1.29, its ions sunk into the guarded O cores) and passed 251 pS; at ratio 16, 0.94 ions and 151 pS.
# So the base takes ratio 16 (eps 4.9 in the filter) and the siblings go on into the literature's 2-10.
_K8 = {"slab_eps_ratio": 16.0, "pore_profile": True, "filter_pore_nm": 0.35, "solve_iters_first": 60000,
       "wca_cap_nm": 0.2, "coulomb_cap_nm": 0.03, "chain_opacity": 0.45}
version("8b", "kv12", None, dict(_K8),
        "Kv1.2 base after round 6: pore-shaped Born barrier, converged solve, walls above the pull, translucent chains; "
        "permittivity ratio 16 (eps 4.9 in the filter)")
version("8a", "kv12", "8b", {"slab_eps_ratio": 24.0}, "8b with the filter's permittivity ratio 16 -> 24 (eps 3.3)")
version("8c", "kv12", "8b", {"slab_eps_ratio": 32.0}, "8b with the filter's permittivity ratio 16 -> 32 (eps 2.5)")
version("k8t", "kv12", "8b", {"n_frames_override": 150000}, "Kv1.2 round-8 base, 150k frames, local (/tmp, never a row)")
version("k8f", "kv12", "8b", {"filter_friction": 0.1, "n_frames_override": 150000},
        "Kv1.2 8b with single-file friction 0.1 in the filter, 150k frames, local (/tmp, never a row)")
version("k8u", "kv12", "8a", {"n_frames_override": 150000}, "Kv1.2 round-8 ratio 16, 150k frames, local (/tmp, never a row)")
# ---- round 6: Kv1.2 on the base after round 4 (sealed core, breathing filter, cut-away), from 4c --------
version("6b", "kv12", None, {"slab_eps_ratio": 8.0},
        "Kv1.2 base after round 4: Born-sealed core, sealed solve, carbonyls tethered by their B-factors, "
        "WCA cores guarded at 0.05 nm/step; permittivity ratio 8 (4c's); two front chains not drawn")
version("6a", "kv12", "6b", {"slab_eps_ratio": 4.0}, "6b with the filter's permittivity ratio 8 -> 4")
version("6c", "kv12", "6b", {"psi_mV": 300.0}, "6b at +300 mV instead of +150 (twice the drive: linearity, and twice the events)")
version("k6u", "kv12", None, {"slab_eps_ratio": 8.0, "n_frames_override": 150000},
        "Kv1.2 base after round 4, 150k frames, local (/tmp, never a row)")
version("k6t", "kv12", None, {"slab_eps_ratio": 8.0, "n_frames_override": 20000},
        "Kv1.2 crash check of the base after round 4: sealed core, breathing filter, cut-away (/tmp, never a row)")
# ---- round 5: MscL on the base after round 3 (a derived well, a barostat stretch, the ion-accessible
# current, 400,000 frames, 15 mN/m at the top) -- a base and two single changes ------------------------
# Round 3 found two walls, both needed: the wells were 4-9x too deep for ~10 mN/m to pay (3a, 3b), and
# 120,000 frames was a fifth of the time the edge's stretch takes to reach the protein (3c). The base
# removes both; its siblings ask whether the lipids grip hard enough to pass the pull, and whether a
# harder stretch is needed.
version("5b", "mscl", None, {"go_eps_kT": 1.0, "go_eps_between_chains_kT": "derived", "tension_max_mN_m": 12.0},
        "MscL base after round 3: each subunit's own packing 1 kT, the 468 contacts BETWEEN subunits derived "
        "(tau_1/2 dA / N = 0.104 kT), C-terminal bundle kept; barostat stretch to 12 mN/m; ion-accessible current; 400k frames")
version("5a", "mscl", "5b", {"go_eps_between_chains_keep_none": True},
        "5b with the C-terminal bundle's inter-chain contacts at the derived well too (is the bundle the lock?)")
version("5c", "mscl", "5b", {"lipid_protein_eps_kT": 2.5},
        "5b with the lipid-residue grip 1 -> 2.5 kT (does the membrane's pull reach the protein?)")
version("l5t", "mscl", "5b", {"n_frames_override": 4000, "conduct_every": 500},
        "MscL crash check of the round-5 base (/tmp, never a row)")
version("l5f", "mscl", "5b", {}, "MscL round-5 base, full length, run locally (/tmp, never a row)")
version("l5g", "mscl", "5b", {"go_eps_kT": 0.5}, "MscL new base at round 3a's 0.5 kT wells, full length, local (/tmp, never a row)")
version("l5h", "mscl", "5b", {"go_eps_kT": 0.25}, "MscL new base at 0.25 kT wells, full length, local (/tmp, never a row)")
version("l5i", "mscl", "5b", {"go_eps_kT": 1.0, "go_eps_between_chains_kT": "derived", "n_frames_override": 150000},
        "MscL: each subunit's own packing 1 kT, the interfaces between subunits derived, CTD bundle kept (/tmp, never a row)")
version("l7t", "mscl", "5b", {"rigid_merge": [["tm1", "tm2"]], "n_frames_override": 150000},
        "MscL 5b with each subunit's TM1+TM2 one rigid body, 150k frames, local (/tmp, never a row)")
version("l5k", "mscl", "5b", {"thinning": True, "n_frames_override": 150000},
        "MscL 5b with a bilayer that thins as it stretches (volume-conserving leaflet heights) (/tmp, never a row)")
version("l5j", "mscl", "5b", {"go_eps_kT": 1.0, "go_eps_between_chains_kT": 0.0, "n_frames_override": 150000},
        "MscL: each subunit's own packing 1 kT, NO well between subunits (lipids alone hold the ring), CTD kept (/tmp, never a row)")
version("l5c", "mscl", "5b", {"go_eps_kT": 0.5, "go_exclude": ["s1", "tm1", "tm2"]},
        "MscL calibration: no Go well touching S1/TM1/TM2 (lipid pressure alone holds the TM core), 0.5 kT elsewhere (/tmp, never a row)")
version("l5v", "mscl", "5b", {"n_frames_override": 3000, "conduct_every": 300, "camera": {"elev": 62.0, "azim": 30.0},
                              "hide_front_chains": 0, "lipid_near_side": False, "zoom": 1.5},
        "MscL render check: from above-side, whole ring, whole slab (/tmp, never a row)")
version("l5r", "mscl", "5b", {"n_frames_override": 3000, "conduct_every": 300},
        "MscL render check: zoomed, two front chains hidden, slab cut in front (/tmp, never a row)")
# ---- round 7: alpha-hemolysin, the loose prepore that must leak (v3: ASSEMBLY2) -------------------------
# v1 (cap and rim only, no stem) could never leak; whole rigid protomers interpenetrate (their stems wrap
# ~160 deg round the barrel); so seven protomers stand 0.8 nm out from the heptamer's ring, each with the
# upper half-barrel where 9M4A (prepore state IV) has it and its tips disordered, the stems flexible.
version("7b", "ahl2", None, {},
        "alpha-HL base: a loose prepore of 7 whole protomers (9M4P/9M4A), flexible stems, disordered tips, "
        "template interfaces 1 kT, fluid bilayer, conduction bounded by the stems' ring; 400k frames")
version("7a", "ahl2", "7b", {"insertion_drive": True},
        "7b with the stems' hydrophobic insertion drive (Wimley-White side chains in the core)")
version("7c", "ahl2", "7b", {"go_eps_kT": 2.0},
        "7b with the interfaces between protomers 1 -> 2 kT per template pair (a stronger zipper)")
version("7t", "ahl", None, {"go_eps_kT": 1.0, "n_frames_override": 4000}, "alpha-HL crash check (/tmp, never a row)")
# ---- round 9: alpha-hemolysin, the base after round 7 (its judges): 7a (the insertion drive) with the
# upper half-barrel rigid in each body (Chatterjee's stable stem region; flexible it melted and shut the
# lumen) and the tips hung through the lower leaflet (the drive only acts at the core's edges, where no tip
# ever went). A local 150k-frame run (a4t): the ring now GATHERS over the run (3 -> 5 -> 7 protomers, the
# bodies 3.58 -> 3.20 nm off the axis), the tips stay at their native depth (median -1.0 nm), and six
# lower-leaflet lipids enter the future lumen -- no leak yet in 15 ns.
_A9 = {"rigid_upper_stem": True, "insertion_drive": True, "tip_lift_nm": -1.4, "tip_z_min_nm": -1.8}
version("9b", "ahl2", None, dict(_A9),
        "alpha-HL base after round 7: 7a with the upper half-barrel rigid and the tips hung through the lower leaflet; 400k frames")
version("9a", "ahl2", "9b", {"stem_go_eps_kT": 2.0}, "9b with the tips' own wells (hairpin and body) 1 -> 2 kT (a stronger lower-barrel zipper)")
version("9c", "ahl2", "9b", {"n_frames": 800000}, "9b run twice as long (800k frames, ~80 ns): time for the tips")
version("a4t", "ahl2", None, {"rigid_upper_stem": True, "insertion_drive": True, "tip_lift_nm": -1.4,
                              "tip_z_min_nm": -1.8, "n_frames_override": 150000},
        "alpha-HL v4: rigid upper half-barrel, tips hung through the lower leaflet, insertion drive; 150k, local (/tmp, never a row)")
version("a2g", "ahl2", None, {"n_frames_override": 200000, "insertion_drive": True},
        "alpha-HL loose prepore with the stems' hydrophobic insertion drive, 200k frames, local (/tmp, never a row)")
version("a2f", "ahl2", None, {"n_frames_override": 200000},
        "alpha-HL loose prepore, flexible stems, 200k frames, local (/tmp, never a row)")
version("a2t", "ahl2", None, {"n_frames_override": 3000, "conduct_every": 300},
        "alpha-HL v2 crash check: whole protomers, fluid lipids, conduction (/tmp, never a row)")
# ---- round 3: MscL on the fixed base (Gauss-Seidel electrolyte), a base and two single changes -------
version("3b", "mscl", None, {"go_eps_kT": 1.0, "tension_max_mN_m": 20.0},
        "MscL base: interface Go wells 1.0 kT, stretch sized for 20 mN/m")
version("3a", "mscl", "3b", {"go_eps_kT": 0.5},
        "3b with interface Go wells 1.0 -> 0.5 kT (weaker packing)")
version("3c", "mscl", "3b", {"tension_max_mN_m": 35.0},
        "3b with the stretch sized for 35 mN/m instead of 20 (drive harder)")


def params_of_ions(label):
    """A Kv1.2 version's parameters: ION, then each ancestor's one change."""
    chain, v = [], label
    while v is not None:
        chain.append(v)
        v = VERSIONS[v]["parent"]
    P = copy.deepcopy(ION)
    for v in reversed(chain):
        P.update(VERSIONS[v]["change"])
    return P


def params_of(label):
    """The parameter dict of a version: its channel's base, then each ancestor's one change."""
    chain, v = [], label
    while v is not None:
        chain.append(v)
        v = VERSIONS[v]["parent"]
    P = copy.deepcopy(MECHANO)
    for v in reversed(chain):
        for k, val in VERSIONS[v]["change"].items():
            d = P
            parts = k.split(".")
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = val
    return P


# ================================================================================ anatomy
def anatomy(channel):
    a = ANATOMY[channel]
    cif = os.path.join(DATA, a["cif"])
    if not os.path.isfile(cif):
        raise SystemExit(f"{cif} is not there yet -- the human downloads it (RCSB is blocked here)")
    cmd = [PY, os.path.join(REPO, "tools", "channel_anatomy.py"), cif, "--name", f"exp04_{channel}",
           "--parts", a["parts"], "--membrane", a["membrane"]]
    if a.get("flip"):
        cmd.append("--flip")
    if a.get("extra"):
        cmd += ["--extra-atoms", a["extra"]]
    if a.get("membrane_offset"):
        cmd += ["--membrane-offset", str(a["membrane_offset"])]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": os.path.join(REPO, "src")})


def _shape(channel):
    from plexus.paths import graphs_data_path
    folder = os.path.join(graphs_data_path(), "shapes", f"exp04_{channel}")
    info = json.load(open(os.path.join(folder, "anatomy.json")))
    pts = dict(np.load(os.path.join(folder, "points.npz")))
    return folder, info, pts


# ================================================================================ physics helpers
def lj_lattice_K_A(sigma_over_a, eps=1.0, cut=2.5):
    """Per-bead energy of a 2D hexagonal LJ lattice of spacing a (in units of a), its minimum and
    the area modulus there: K_A = A d^2E/dA^2 per bead (energy per area, per unit eps, a = 1)."""
    pts = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            if i == 0 and j == 0:
                continue
            pts.append((i + 0.5 * j, j * math.sqrt(3) / 2))
    R = np.hypot(*np.array(pts).T)

    def E(scale):
        r = R * scale
        s = sigma_over_a
        m = r < cut * s
        return 0.5 * float((4 * eps * ((s / r[m]) ** 12 - (s / r[m]) ** 6)).sum())
    area = lambda sc: math.sqrt(3) / 2 * sc * sc                       # noqa: E731
    sc = np.linspace(0.9, 1.1, 2001)
    Es = np.array([E(x) for x in sc])
    k = int(np.argmin(Es))
    h = 1e-3
    A0 = area(sc[k])
    dA = area(sc[k] + h) - area(sc[k])
    d2 = (E(sc[k] + h) - 2 * E(sc[k]) + E(sc[k] - h)) / (dA * dA)
    return sc[k], A0 * d2


# ---- PLAIN-ENGLISH NAMES (the human, 2026-09-26: "do not over-simplify, write plain English names") -----------------
# A spec's set, field and state-block names are code -- `qsite`, `qfil`, `elec`, `cden`, `psi`, `nK_in` -- and the
# watcher's plexus pane printed them as if they were words. Every name keeps its code (operators, the ruler and every
# trajectory address them by it) and gains a `title`, the name a reader is given; `spec_summary` shows the title.
# Written into every spec this generator writes (`titled`), and back-filled into specs already run (it is a label,
# not physics: the engine ignores it).
PROTEIN_NAME = {"SelfTest": "test protein", "MscS": "MscS", "MscL": "MscL", "Kv12": "Kv1.2", "alphah": "alpha-hemolysin",
                "aHL": "alpha-hemolysin", "KcsA": "KcsA", "gramic": "gramicidin A", "GLIC": "GLIC", "NavAb": "NavAb",
                "TRAAK": "TRAAK", "OmpF": "OmpF"}
SET_TITLE = {"cell": "the cell's interior (the voltage-clamped compartment)", "lipid": "membrane lipids",
             "frame": "frozen lipids at the patch's rim (they hold the tension)",
             "qsite": "charged residues (one charge per ionisable side chain)",
             "qfil": "backbone carbonyls lining the pore (on springs)",
             "filter_O": "carbonyl oxygens of the selectivity filter", "filter_C": "carbonyl carbons of the selectivity filter",
             "K": "potassium ions (K+)", "Na": "sodium ions (Na+)", "Cl": "chloride ions (Cl-)"}
FIELD_TITLE = {"elec": "electric potential in the electrolyte", "cden": "cation density (time-averaged)",
               "kden": "potassium density (time-averaged)"}
BLOCK_TITLE = {"pos": "position", "psi": "membrane potential", "j_channel": "channel current",
               "g_channel": "channel conductance", "tension": "membrane tension", "tension_applied": "applied tension",
               "strain": "membrane strain", "pore_r": "pore radius", "domain": "protein domain", "force": "force from the ions",
               "nK_in": "K+ inside the cell", "nNa_in": "Na+ inside the cell", "nCl_in": "Cl- inside the cell",
               "q_in": "charge moved into the cell", "I_in": "current into the cell", "q": "charge",
               "w_open": "gate's open fraction", "dg": "insertion free energy (Wimley-White)",
               "n_filter": "ions in the filter", "n_crossed": "ions that crossed", "n_largest": "largest assembly",
               "n_bonds": "bonds between monomers", "ring": "ring membership", "rim": "rim bead", "body": "rigid body"}


def set_title(name):
    m = re.match(r"^(.*)_(\d+)$", str(name))
    if m and m.group(1) in PROTEIN_NAME:
        return f"{PROTEIN_NAME[m.group(1)]}, chain {int(m.group(2))}"
    return SET_TITLE.get(str(name))


def titled(spec):
    """The spec with a plain-English `title` on each set, field and state block it has a name for (in place)."""
    for n, v in (spec.get("sets") or {}).items():
        if isinstance(v, dict):
            t = set_title(n)
            if t and not v.get("title"):
                v["title"] = t
            for b, bv in (v.get("state") or {}).items():
                if isinstance(bv, dict) and b in BLOCK_TITLE and not bv.get("title"):
                    bv["title"] = BLOCK_TITLE[b]
    for n, v in (spec.get("fields") or {}).items():
        if isinstance(v, dict) and n in FIELD_TITLE and not v.get("title"):
            v["title"] = FIELD_TITLE[n]
    return spec


def _py(o):
    """Plain Python numbers all the way down: a numpy scalar in a spec is a YAML tag nobody reads."""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def hex_lattice(a_nm, r_out_nm):
    pts = []
    hgt = a_nm * math.sqrt(3) / 2
    n = int(r_out_nm / hgt) + 2
    for iy in range(-n, n + 1):
        y = iy * hgt
        x0 = 0.5 * a_nm if iy % 2 else 0.0
        for ix in range(-int(r_out_nm / a_nm) - 2, int(r_out_nm / a_nm) + 3):
            pts.append((x0 + ix * a_nm, y))
    return np.array(pts)


# ================================================================================ the spec
def _rig_copy(key, label):
    """The rig's deposited state RIG[key] copied into this version's OWN shape folder (a shared points.npz
    let one version's membrane overwrite another's -- round 7), returned as (folder, info, pts, name)."""
    import shutil
    from plexus.paths import graphs_data_path
    src = os.path.join(graphs_data_path(), "shapes", f"exp04r_{key}")
    name = f"exp04r_{key}_v{label}"
    dst = os.path.join(graphs_data_path(), "shapes", name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst, json.load(open(os.path.join(dst, "anatomy.json"))), dict(np.load(os.path.join(dst, "points.npz"))), name


def build_mechano(label, P, channel):
    # `rig_shape` -- THE RIG'S STRUCTURE (OPM-placed membrane, all-atom ion paths; 2026-09-26) instead of the
    # rounds-1-8 shape, whose MscS membrane sat 1.63 nm too low; the leaflets at OPM's faces
    h_core = None
    if P.get("rig_shape"):
        folder, info, pts, shp = _rig_copy(P["rig_shape"], label)
        h_core = float(info.get("opm", {}).get("half_thickness_nm") or 1.5)
        P["leaflet_z_nm"] = round(h_core + FLOW["head_offset_nm"], 3)
        P["membrane_slab_nm"] = round(2 * h_core, 3)
    else:
        folder, info, pts = _shape(channel)
        shp = f"exp04_{channel}"
    gnote = None
    if P["go_eps_kT"] == "derived":
        P["go_eps_kT"], gnote = go_eps_derived(channel, P, pts, folder)
        P["go_eps_kT"] = P["go_eps_kT"] * P.get("go_eps_factor", 1.0)
    if P.get("go_eps_between_chains_kT") == "derived":
        P["go_eps_between_chains_kT"], gnote = go_eps_derived(channel, P, pts, folder)
        P["go_eps_between_chains_kT"] = P["go_eps_between_chains_kT"] * P.get("go_eps_factor", 1.0)
    chains = [k for k in sorted(pts) if k.startswith("c") and k[1:].isdigit()]
    chains.sort(key=lambda s: int(s[1:]))
    nch = len(chains)
    X_all = pts["all"] * 1e9                                             # nm, bilayer centre z = 0
    rho = np.hypot(X_all[:, 0], X_all[:, 1])
    tm = np.abs(X_all[:, 2]) < 2.0
    r_tm = float(rho[tm].max())
    R_patch = r_tm + P["annulus_nm"]
    a_L = math.sqrt(2.0 * P["area_per_lipid_nm2"] / math.sqrt(3.0))       # hex spacing
    R_frame_out = R_patch + 2 * a_L * math.sqrt(3) / 2
    z_lo = float(X_all[:, 2].min()) - P["bath_nm"]
    z_hi = float(X_all[:, 2].max()) + P["bath_nm"]
    z_lo = min(z_lo, -P["membrane_slab_nm"] / 2 - P["bath_nm"])
    z_hi = max(z_hi, P["membrane_slab_nm"] / 2 + P["bath_nm"])
    L = max(2 * (R_frame_out + 1.5), z_hi - z_lo)                         # nm, a cube
    L = math.ceil(L)
    zc = -z_lo / L + 0.5 * (1.0 - (z_hi - z_lo) / L)                      # the bilayer centre, world
    world_per_m = 1.0 / (L * 1e-9)
    nm = lambda x: x / L                                                  # noqa: E731

    # ---- units ------------------------------------------------------------------------------
    zeta = 6 * math.pi * P["eta_Pa_s"] * P["bead_stokes_nm"] * 1e-9       # kg/s
    tau_s = zeta * (L * 1e-9) ** 2 / KT_J                                 # s per sim time
    force_nN = KT_J / (L * 1e-9) * 1e9
    units = {"length_um": L * 1e-3, "time_s": tau_s, "force_nN": force_nN}
    E_to_sim = lambda kT_per_nm2: kT_per_nm2 * L * L                       # noqa: E731  spring kT/nm^2 -> sim
    tension_sim = lambda mN_m: mN_m * 1e-3 * (L * 1e-9) ** 2 / KT_J        # noqa: E731

    # ---- the lipid bilayer: two leaflets of beads -----------------------------------------------
    # THE COHESION CANNOT COME FROM THE AREA MODULUS, and that is a finding, not a choice. Matching
    # a bilayer's K_A of 230 mN/m with one Lennard-Jones bead per lipid per leaflet needs a well of
    # 0.31 kT (the lattice sum below, at T = 0) -- and a 2D LJ layer that weak at room temperature is
    # a GAS: it has no zero-tension state and would evaporate out of the patch. A condensed, flowing
    # layer needs T* = kT/eps between the 2D triple point (~0.41) and critical point (~0.46; Smit &
    # Frenkel 1991, J. Chem. Phys. 94:5663), so eps = 2.3 kT, and the layer's area modulus is then
    # several times the real one: STATED, measured by the ruler (tension against area strain), and
    # harmless to the question -- the work that opens a channel is tension x area change, which does
    # not involve the modulus. The T = 0 modulus of this lattice is printed for the record.
    sigma_LL = a_L * P["lipid_sigma_over_a"]
    eps_LL = P["eps_LL_kT"]
    if P["lipid_law"] == "lj":
        _, KA_per_eps = lj_lattice_K_A(sigma_LL / a_L)
        KA_T0_mN_m = 2.0 * KA_per_eps * eps_LL * KT_J / (a_L * 1e-9) ** 2 * 1e3   # both leaflets
    else:
        # THE COOKE LAYER'S MODULUS, from its tail's curvature at the flat well's edge (6 neighbours
        # at a): U''(r_c+) = eps pi^2 / (2 w_c^2) per pair; a 2D hexagonal lattice's area modulus is
        # (sqrt 3 / 4) x 6 x U'' / 2 per bead area... an order-of-magnitude figure, measured for real
        # by the ruler (tension against area strain).
        w_c = P["lipid_tail_over_sigma"] * sigma_LL * 1e-9
        u2 = eps_LL * KT_J * math.pi ** 2 / (2.0 * w_c ** 2)
        KA_T0_mN_m = 2.0 * (math.sqrt(3.0) / 2.0) * u2 * 1e3
    lipid_pair = ({"law": "cooke", "tail": nm(P["lipid_tail_over_sigma"] * sigma_LL)} if P["lipid_law"] == "cooke"
                  else {"law": "lj"})
    lat = hex_lattice(a_L, R_frame_out + 0.1)
    r_lat = np.hypot(lat[:, 0], lat[:, 1])
    lipid, frame = [], []
    for zl in (-P["leaflet_z_nm"], P["leaflet_z_nm"]):
        near = np.abs(X_all[:, 2] - zl) < 1.2
        Pn = X_all[near]
        th_p = np.arctan2(Pn[:, 1], Pn[:, 0]); r_p = np.hypot(Pn[:, 0], Pn[:, 1])
        for (x, y), r in zip(lat, r_lat):
            if r > R_patch:
                if r <= R_frame_out:
                    frame.append((x, y, zl))
                continue
            if Pn.shape[0]:
                d = np.sqrt((Pn[:, 0] - x) ** 2 + (Pn[:, 1] - y) ** 2 + (Pn[:, 2] - zl) ** 2)
                if d.min() < P["lipid_excl_nm"]:
                    continue
                # inside the protein's ring (its lumen, or between its helices): the widest protein
                # radius in this bead's direction, +-15 degrees
                th = math.atan2(y, x)
                dth = np.abs((th_p - th + np.pi) % (2 * np.pi) - np.pi)
                sect = r_p[dth < math.radians(15)]
                if sect.size and r < sect.max():
                    continue
            lipid.append((x, y, zl))
    lipid = np.array(lipid); frame = np.array(frame)
    C3 = None
    if P.get("lipid_model") == "cooke3":
        # PHASE G B3 (2026-09-30): THE MEMBRANE WITH A CORE around the channel -- Cooke's three-bead lipids (BILAYER's
        # parameters, the build_bilayer geometry: heads at +-(0.5 sigma + 2 b) about the OPM centre, tails inward), a
        # molecule kept only if none of its beads is within `lipid_excl_nm` of an alpha carbon or inside the protein's
        # ring at that bead's height (+-15 deg sector, +-1.2 nm); the rim's molecules frozen as the frame. Beads ordered
        # heads first, then tail pairs, as `type_layout: ordered` types them; `mol` numbers the molecules.
        B_ = {**BILAYER, **(P.get("cooke3") or {})}
        s3 = B_["sigma_nm"]; b3 = B_["bond_sigma"] * s3
        a3 = math.sqrt(2.0 * B_["area_per_lipid_sigma2"] * s3 * s3 / math.sqrt(3.0))
        zh3 = 0.5 * s3 + 2 * b3
        lat3 = hex_lattice(a3, R_frame_out + 0.1)
        thp = np.arctan2(X_all[:, 1], X_all[:, 0]); rp_ = np.hypot(X_all[:, 0], X_all[:, 1])
        mols, rims = [], []
        for sgn in (1.0, -1.0):
            for (x, y) in lat3:
                r = math.hypot(x, y)
                m = [(x, y, sgn * zh3), (x, y, sgn * (zh3 - b3)), (x, y, sgn * (zh3 - 2 * b3))]
                if r > R_patch:
                    if r <= R_frame_out:
                        rims.append(m)
                    continue
                bad = False
                for (bx, by, bz) in m:
                    if np.sqrt((X_all[:, 0] - bx) ** 2 + (X_all[:, 1] - by) ** 2 + (X_all[:, 2] - bz) ** 2).min() < P["lipid_excl_nm"]:
                        bad = True; break
                    near = np.abs(X_all[:, 2] - bz) < 1.2
                    dth = np.abs((thp - math.atan2(by, bx) + np.pi) % (2 * np.pi) - np.pi)
                    sect = rp_[near & (dth < math.radians(15))]
                    if sect.size and r < sect.max():
                        bad = True; break
                if not bad:
                    mols.append(m)
        mols, rims = np.array(mols), np.array(rims)
        lipid = np.concatenate([mols[:, 0], mols[:, 1:].reshape(-1, 3)])
        frame = np.concatenate([rims[:, 0], rims[:, 1:].reshape(-1, 3)])
        mol3 = np.concatenate([np.arange(len(mols)), np.repeat(np.arange(len(mols)), 2)])
        bl_ = dict(np.load(os.path.join(folder, "blocks.npz")))
        bl_["lipid_mol"] = mol3.astype(np.float64)[:, None]
        np.savez_compressed(os.path.join(folder, "blocks.npz"), **bl_)
        C3 = {"B": B_, "s": s3, "b": b3, "nL": len(mols), "nF": len(rims)}
    np.savez_compressed(os.path.join(folder, "membrane.npz"),
                        lipid=(lipid * 1e-9).astype(np.float32), frame=(frame * 1e-9).astype(np.float32))

    # ---- THE IONS (`ions`), the same layer as the static rig's: the closed state's measured paths and the
    # open state's (`rig_open`), blended by the two basins' weight, which the elastic network publishes
    IL = None
    if P.get("ions"):
        from plexus.paths import graphs_data_path
        PF = {**FLOW, **{k_: P[k_] for k_ in P if k_ in FLOW}, "dt_safety": FLOW["dt_safety"]}
        info_o = (json.load(open(os.path.join(graphs_data_path(), "shapes", f"exp04r_{P['rig_open']}", "anatomy.json")))
                  if P.get("rig_open") else None)
        blq = dict(np.load(os.path.join(folder, "blocks.npz")))
        names_ = [f"{CHANNEL_NAME[channel].replace('.', '').replace('-', '').replace(' ', '')[:6]}_{k + 1}" for k in range(nch)]
        zeta_ = 6 * math.pi * P["eta_Pa_s"] * P["bead_stokes_nm"] * 1e-9
        tau_ = zeta_ * (L * 1e-9) ** 2 / KT_J
        IL = _ion_layer(PF, label=label, shape=shp, L=L, zc=zc, nm=nm, zeta=zeta_, tau_s=tau_,
                        force_nN=KT_J / (L * 1e-9) * 1e9, h=h_core or P["membrane_slab_nm"] / 2, X_all=X_all,
                        lipid=lipid, z_lo=z_lo, z_hi=z_hi, names=names_, info_c=info, qpts=pts.get("qsite"),
                        qblk={k_: blq[k_] for k_ in ("qsite_q", "qsite_backbone", "qsite_B") if k_ in blq}, info_o=info_o,
                        xtal=pts.get("xtal_ions"),
                        gate=("cell", "w_open") if (info_o is not None and P.get("two_basin")) else None)

    # ---- stiffness, time step, protocol -----------------------------------------------------------
    k_enm = E_to_sim(P["enm_k_kT_nm2"])
    lj_curv = (57.1 * eps_LL / nm(sigma_LL) ** 2 if P["lipid_law"] == "lj"   # U'' at the LJ minimum, sim
               else max(57.1 * eps_LL / nm(sigma_LL) ** 2, eps_LL * math.pi ** 2 / (2.0 * nm(P["lipid_tail_over_sigma"] * sigma_LL) ** 2)))
    go_curv = 72.0 * P["go_eps_kT"] / nm(0.6) ** 2
    lam = max(24 * k_enm, 6 * lj_curv, 6 * go_curv, E_to_sim(P["lipid_z_k_kT_nm2"]))
    if C3 is not None:
        k3 = E_to_sim(C3["B"]["k_bond_kT_sigma2"] / C3["s"] ** 2)
        lam = max(24 * k_enm, 6 * go_curv, 4 * k3, 6 * 57.1 / nm(C3["B"]["head_b"] * C3["s"]) ** 2,
                  6 * 57.1 * P["lipid_protein_eps_kT"] / nm(P["lipid_protein_sigma_nm"]) ** 2)
    dt = P["dt_safety"] / lam
    if IL is not None:
        dt = min(dt, IL["dt"])                                            # the ions' contacts may be the stiffer
    # THE MODEL'S OWN MODULUS SETS ITS CLOCK, not the real bilayer's: the layer is stiffer (see
    # eps_LL above), so the patch relaxes faster. Half the T = 0 lattice value stands in for the
    # thermal softening of a liquid near its triple point -- stated; the ruler measures the real one.
    KA_model = 0.5 * KA_T0_mN_m
    KA_sim = KA_model * 1e-3 * (L * 1e-9) ** 2 / KT_J                     # sim energy per world^2
    density = 2.0 / (P["area_per_lipid_nm2"] / L ** 2)                    # beads per world^2, both leaflets
    tau_patch = density * nm(R_patch) ** 2 / KA_sim                       # sim time
    fr_tau = tau_patch / dt
    seg = [max(200, int(round(x * fr_tau))) for x in P["ramp_tau"]]
    n_frames = int(sum(seg))
    scale_note = ""
    if n_frames > P["max_frames_cap"]:
        f = P["max_frames_cap"] / n_frames
        seg = [max(200, int(x * f)) for x in seg]
        n_frames = int(sum(seg))
        scale_note = f" (CAPPED at {P['max_frames_cap']} frames: the ramp is {1 / f:.1f}x faster than the quasi-static design)"
    if P.get("n_frames_override"):
        f = P["n_frames_override"] / n_frames
        seg = [max(1, int(x * f)) for x in seg]
        n_frames = int(sum(seg))
    T = tension_sim(P["tension_max_mN_m"])
    if P["drive_mode"] == "strain":
        # THE STRETCH THAT CARRIES THE TARGET TENSION in this layer: area strain = tau / K_A, with the
        # layer's own modulus (KA_T0 above); the frame's LINEAR strain is sqrt(1 + tau/K_A) - 1. The
        # tension it actually produces is measured -- the modulus here is an estimate.
        KA_drive = P.get("K_A_drive_mN_m") or KA_T0_mN_m
        T = math.sqrt(1.0 + P["tension_max_mN_m"] / max(KA_drive, 1e-6)) - 1.0
    f0 = 0
    prot = [[0, 0.0]]
    f0 += seg[0]; prot.append([f0, 0.0])
    # `tension_cycles: n` (Phase G, 2026-09-29): stretch, hold, release, hold -- n times, so one run shows the gate
    # opening AND closing more than once (the human: "to see multiple closing/opening")
    for _c in range(int(P.get("tension_cycles", 1))):
        f0 += seg[1]; prot.append([f0, T])
        f0 += seg[2]; prot.append([f0, T])
        f0 += seg[3]; prot.append([f0, 0.0])
        f0 += seg[4]; prot.append([f0, 0.0])
    n_frames = f0
    if P.get("strain_top_linear") is not None:
        # PHASE G1 (2026-09-29): THE STRETCH GIVEN AS TWO LINEAR STRAINS OF THE FRAME, from batch 12's own calibration
        # (12a/12b: tension = -4.7 + 57 x area strain, mN/m, over the holds): `rest_strain_linear` puts the patch at zero
        # tension from frame 0 (area strain 0.083 -> linear 0.040; unstretched it rests at -4.7 mN/m, compressed), and
        # `strain_top_linear` is held after one ramp to the end of the run (no release: a clamped gate's mean force
        # needs one long hold). Frames: `hold_frames` = [settle, ramp, hold].
        s0, s1 = float(P.get("rest_strain_linear", 0.0)), float(P["strain_top_linear"])
        if P["drive_mode"] == "tension":
            # G1' (2026-09-29): CONSTANT TENSION, not constant frame area -- at a fixed frame the open end's outline
            # swallowed ~40 lipids and stretched the rest of the patch (G1a-e: 3.8 -> 8.1 mN/m from lambda 0 to 1 "at
            # rest"), so the two tensions were not two tensions. The frame now moves until the membrane carries the value.
            s0, s1 = tension_sim(float(P.get("rest_tension_mN_m", 0.0))), tension_sim(float(P["hold_tension_mN_m"]))
        fs = [int(x) for x in P["hold_frames"]]
        prot = [[0, s0], [fs[0], s0], [fs[0] + fs[1], s1], [sum(fs), s1]]
        n_frames = sum(fs)
    mu_R = 1.0 / (4 * math.pi * KA_sim * 0.2 * tau_patch) * float(P.get("mobility_R_factor", 1.0))
    # ONE GUARD FOR EVERY BOND, CONTACT AND PAIR: no single force may move a bead more than 0.02 nm in
    # one step (mobility 1: dx = f dt). A numerical guard, counted in the log whenever it acts; the
    # uncapped Go wall blew round 1 up (exp04, 2026-09-25).
    f_cap = nm(0.02) / dt

    # ---- electricity ------------------------------------------------------------------------------
    volt_per_sim = (force_nN * 1e-9) * (L * 1e-9) / E_C                   # an e-psi of 1 sim energy, in volts
    psi_sim = P["psi_mV"] * 1e-3 / volt_per_sim
    C_F = 1e-2 * P["cap_um2"] * 1e-12                                     # 1 uF/cm^2 = 1e-2 F/m^2
    cap_sim = C_F / E_C * volt_per_sim                                    # charges per sim e-psi
    dx_m = L * 1e-9 / P["n_grid"]
    # Hille's estimate for the DEPOSITED pore and for the paper's open pore -- the baseline to beat
    L_pore = 3.0e-9
    r_open = {"mscs": 0.65e-9, "mscl": 1.4e-9}.get(channel, 1.0e-9)
    G_open = 1.0 / (L_pore / (P["sigma_S_m"] * math.pi * r_open ** 2) + 1.0 / (2 * P["sigma_S_m"] * r_open))
    I_open_pA = G_open * P["psi_mV"] * 1e-3 * 1e12
    j_open = abs(G_open * P["psi_mV"] * 1e-3) / (math.pi * r_open ** 2) * 1e12 / 1e18   # pA/nm^2 in the pore

    # ---- the lumen: the pore-lining domain's outer radius inside the bilayer, + 0.5 nm ---------------
    lin = (ANATOMY.get(channel) or {}).get("lining")
    if lin and lin in pts:
        Xl = pts[lin] * 1e9
        rl = np.hypot(Xl[:, 0], Xl[:, 1])[np.abs(Xl[:, 2]) < 2.0]
        lumen_nm = float(rl.max()) + 0.5 if rl.size else r_tm
    else:
        lumen_nm = r_tm

    # ---- sets ---------------------------------------------------------------------------------------
    names = [f"{CHANNEL_NAME[channel].replace('.', '').replace('-', '').replace(' ', '')[:6]}_{k + 1}" for k in range(nch)]
    sets = {"cell": {"n": 1, "start": [[0.5, 0.5, round(zc, 6)]], "state": {
        "pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"},
        "psi": {"width": 1, "integration": "first_order", "unit": "voltage"},
        "j_channel": {"width": 1, "integration": "none", "unit": "current"},
        "g_channel": {"width": 1, "integration": "none", "unit": "1"},
        "tension": {"width": 1, "integration": "none", "unit": "tension"},
        "tension_applied": {"width": 1, "integration": "none", "unit": "tension"},
        "strain": {"width": 1, "integration": "none", "unit": "1"},
        "pore_r": {"width": 1, "integration": "none", "unit": "length"},
        **({"w_open": {"width": 1, "integration": "none", "unit": "1"}, **IL["cell_blocks"]} if IL is not None else {}),
        **({"w_open": {"width": 1, "integration": "none", "unit": "1"}} if (IL is None and P.get("morph")) else {}),
        **({"f_gate": {"width": 1, "integration": "none", "unit": "energy"}} if P.get("morph_clamp") is not None else {})}}}
    bead_state = {"pos": {"width": 3, "role": "coordinate", "integration": "first_order", "boundary": "world"}}
    # `rigid_merge: [[a, b, ...], ...]` -- DOMAINS THAT MOVE AS ONE BODY, for the rigid-body projection only:
    # MscL's iris tilts each subunit's TM1-TM2 pair as a unit (Sukharev & Guy 2001; Wang 2014's helix
    # tilt), so the membrane's pull on TM2 must become a torque on TM1's gate. Round 5's three judges
    # measured TM2 peeling off TM1 (their 1 kT wells 98 -> 37% made) while TM1 stayed at the axis. The
    # `domain` block is untouched -- the Go rules and the lumen's lining (TM1) still read it; shape_match
    # reads `body`.
    merge = P.get("rigid_merge") or []
    if merge:
        bl_all = dict(np.load(os.path.join(folder, "blocks.npz")))
        for ck in chains:
            body = bl_all[f"{ck}_domain"].copy()
            for grp in merge:
                k0 = info["domains"].index(grp[0])
                for other in grp[1:]:
                    body[bl_all[f"{ck}_domain"] == info["domains"].index(other)] = k0
            bl_all[f"{ck}_body"] = body
        np.savez_compressed(os.path.join(folder, "blocks.npz"), **bl_all)
    for nmk, ck in zip(names, chains):
        sets[nmk] = {"n": int(len(pts[ck])), "start": [[0.5, 0.5, round(zc, 6)]],
                     "state": {**bead_state, "domain": {"width": 1, "integration": "none", "unit": "1"},
                               **({"body": {"width": 1, "integration": "none", "unit": "1"}} if merge else {}),
                               **({"dg": {"width": 1, "integration": "none", "unit": "energy"}} if P.get("insertion_drive") else {})}}
    sets["lipid"] = {"n": int(len(lipid)), "start": [[0.5, 0.5, round(zc, 6)]], "state": dict(bead_state)}
    sets["frame"] = {"n": int(len(frame)), "start": [[0.5, 0.5, round(zc, 6)]], "state": {
        "pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"},
        "force": {"width": 3, "integration": "none", "unit": "force"}}}
    if C3 is not None:
        sets["lipid"].update({"type_layout": "ordered", "types": {"head": {"count": C3["nL"]}, "tail": {"count": 2 * C3["nL"]}}})
        sets["lipid"]["state"]["mol"] = {"width": 1, "integration": "none", "unit": "1"}
        sets["frame"].update({"type_layout": "ordered", "types": {"head": {"count": C3["nF"]}, "tail": {"count": 2 * C3["nF"]}}})
        for nmk in names:                                             # one type per chain: the table's second index
            sets[nmk].update({"type_layout": "ordered", "types": {"residue": {"count": sets[nmk]["n"]}}})
    origin = [0.5, 0.5, round(zc, 6)]
    seeds = [{"op": "cloud_seed", "at": nmk, "cloud": f"{shp}/{ck}", "origin": origin, "scale": world_per_m}
             for nmk, ck in zip(names, chains)]
    # each chain's beads carry their domain (the anatomy's --parts, in order), for the lumen and the colour
    seeds += [{"op": "seed_state_from_file", "at": nmk, "file": f"shapes/{shp}/blocks.npz",
               "blocks": {"domain": f"{ck}_domain", **({"body": f"{ck}_body"} if merge else {}),
                          **({"dg": f"{ck}_dg"} if P.get("insertion_drive") else {})}}
              for nmk, ck in zip(names, chains)]
    lining_idx = info["domains"].index(lin) if (lin and lin in info.get("domains", [])) else 0
    # lipid and frame come from membrane.npz: cloud_seed reads points.npz, so they are added there too
    pz = dict(np.load(os.path.join(folder, "points.npz")))
    pz["lipid"] = (lipid * 1e-9).astype(np.float32)
    pz["frame"] = (frame * 1e-9).astype(np.float32)
    np.savez_compressed(os.path.join(folder, "points.npz"), **pz)
    if C3 is not None:
        seeds.append({"op": "seed_state_from_file", "at": "lipid", "file": f"shapes/{shp}/blocks.npz", "blocks": {"mol": "lipid_mol"}})
    seeds += [{"op": "cloud_seed", "at": "lipid", "cloud": f"{shp}/lipid", "origin": origin, "scale": world_per_m},
              {"op": "cloud_seed", "at": "frame", "cloud": f"{shp}/frame", "origin": origin, "scale": world_per_m}]

    # ---- THE OPEN BASIN (`two_basin`): the channel's open state, from the file GATING names, as a second
    # native geometry for elastic_network (Okazaki et al. 2006). Its positions per model bead (NaN where the
    # open model lacks a residue) go into points.npz under this version's own keys; dV = tau_1/2 x dA, the
    # open basin's free energy above the closed one at zero tension, with dA the two outlines' difference.
    basin = {}
    g_ = GATING.get(channel, {})
    if P.get("two_basin") and g_.get("open_cif") and os.path.isfile(os.path.join(DATA, g_["open_cif"])):
        rms_o, opos = _open_state(g_, channel, pts, folder)
        bl_o = dict(np.load(os.path.join(folder, "blocks.npz")))
        pz = dict(np.load(os.path.join(folder, "points.npz")))
        keys = []
        for k, ck in enumerate(chains):
            rr = bl_o[f"{ck}_resid"].ravel()
            O = np.array([opos.get((k, int(r_)), [np.nan] * 3) for r_ in rr], np.float64) * 1e-9
            pz[f"open_{label}_{ck}"] = O.astype(np.float32); keys.append(f"open_{label}_{ck}")
        np.savez_compressed(os.path.join(folder, "points.npz"), **pz)
        A_c = _outline_area(X_all)
        A_o = _outline_area(np.array(list(opos.values())))
        dA = A_o - A_c
        dV = g_["tension_half_mN_m"] * dA / 4.11                          # kT
        if P.get("dV_kT") is not None:
            # a STATED offset (TRAAK: tau_1/2 x the PAPER's 2.7 nm^2 = 3.9 kT, where our outline says -0.6)
            dV = float(P["dV_kT"])
        dA_world = dA / (L * L)
        basin = {"open_reference": shp, "open_parts": keys, "open_scale": world_per_m,
                 "basin_offset": round(dV, 3), "basin_coupling": float(P.get("basin_coupling_kT", 10.0))}
        print(f"  THE OPEN BASIN from {g_['open_cif']}: superposed at {rms_o:.3f} nm rms; in-plane area "
              f"{A_c:.1f} -> {A_o:.1f} nm^2 (dA {dA:.1f}); dV = tau_1/2 {g_['tension_half_mN_m']} mN/m x dA = {dV:.1f} kT; "
              f"coupling {basin['basin_coupling']} kT; {sum(int(np.isnan(np.load(os.path.join(folder, 'points.npz'))[k_][:, 0]).sum()) for k_ in keys)} beads missing")
    elif P.get("two_basin"):
        raise SystemExit(f"two_basin: the open state {g_.get('open_cif')!r} is not in {DATA} yet -- the human downloads it")

    if IL is not None:
        sets.update(IL["sets"])
        seeds += IL["seeds"]
        if basin:
            basin["gate_block"] = ["cell", "w_open"]
    ops = ([{"op": "shape_match", "at": names[0], "sets": names, "domain_block": "body" if merge else "domain", "alpha": 1.0}]
           if P["shape_match"] else []) + [
        {"op": "membrane_potential", "at": "cell", "capacitance": cap_sim, "pump_max": 0.0, "pump_rev": 1.0,
         "leak": 0.0, "psi0": psi_sim, "h0": [0.0, 0.0], "flux": "j_channel"},
        {"op": "elastic_network", "at": names[0], "sets": names, "cutoff": nm(P["enm_cutoff_nm"]), "k": k_enm,
         "local_max_sep": P["local_max_sep"], "go_epsilon": P["go_eps_kT"], "go_cutoff": nm(P["go_cutoff_nm"]),
         "mobility": 1.0, "f_max": f_cap, **({"domain_block": "domain"} if P["domain_rigid"] else {}),
         **({"go_exclude_domains": [info["domains"].index(d) for d in P["go_exclude"]]} if P["go_exclude"] else {}),
         **({"go_eps_between_chains": P["go_eps_between_chains_kT"],
             "go_eps_between_chains_keep": ([] if P.get("go_eps_between_chains_keep_none")
                                            else [info["domains"].index(d) for d in GATING[channel]["kept"]])}
            if P.get("go_eps_between_chains_kT") is not None else {}), **basin},
        {"op": "pair_potential", "at": names[0], "sets": names, "law": "wca", "sigma": nm(P["wca_sigma_nm"]),
         "epsilon": 1.0, "exclude_same_set": True, "mobility": 1.0, "f_max": f_cap},
        {"op": "pair_potential", "at": "lipid", **lipid_pair, "sigma": nm(sigma_LL), "epsilon": eps_LL,
         "mobility": 1.0, "f_max": f_cap},
        {"op": "pair_potential", "at": "lipid", "with_sets": names,
         **({"law": "cooke", "tail": nm(P["lipid_tail_over_sigma"] * P["lipid_protein_sigma_nm"])}
            if P["lipid_protein_law"] == "cooke" else {"law": "lj"}),
         "sigma": nm(P["lipid_protein_sigma_nm"]), "epsilon": P["lipid_protein_eps_kT"],
         "mobility": 1.0, "mobility_with": 1.0, "react": True, "f_max": f_cap},
        {"op": "pair_potential", "at": "lipid", "with": "frame", **lipid_pair, "sigma": nm(sigma_LL),
         "epsilon": eps_LL * P["frame_eps_ratio"], "mobility": 1.0, "react": False, "f_max": f_cap},
        {"op": "tether", "at": "lipid", "k": E_to_sim(P["lipid_z_k_kT_nm2"]), "axes": [2], "mobility": 1.0,
         **({"thin_with": {"set": "cell", "block": "strain", "mid": round(zc, 6)}} if P.get("thinning") else {})},
        {"op": "brownian", "at": "lipid", "kT": P["kT_noise"], "mobility": 1.0, "seed": 1 + 1000 * int(P.get("noise_replicate", 0))},
    ] + [{"op": "brownian", "at": nmk, "kT": 0.0 if P.get("protein_brownian") is False else P["kT_noise"],
          "mobility": 1.0, "seed": 10 + i}                 # kT 0, not dropped: a set with no motion operator is
         for i, nmk in enumerate(names)] + [                # not integrated at all (the G1 smoke test froze)
        {"op": "radial_drive", "at": "frame", "protocol": prot, "mode": P["drive_mode"], "mobility_R": mu_R,
         "cell": "cell", "axis": 2, "centre": origin, "max_step": nm(0.01 * a_L), "avg_frames": 2000.0,
         **({"affine_sets": ["lipid"]} if P.get("affine_stretch") else {})},
        {"op": "pore_probe", "at": names[0], "sets": names, "bead_radius": nm(P["ins_protein_nm"]),
         "z": [zc + nm(-2.5), zc + nm(2.5)], "h": nm(0.3), "slices": 21, "cell": "cell"},
        {"op": "electrolyte_conduction", "at": "cell", "field": "elec",
         "insulators": [{"set": s, "radius": nm(P["ins_protein_nm"])} for s in names]
                       + [{"set": "lipid", "radius": nm(P["ins_lipid_nm"])}],
         "sigma": P["sigma_S_m"], "volt_per_sim": volt_per_sim, "dx_m": dx_m,
         "sim_current_per_A": tau_s / E_C, "slab": [zc - nm(P["membrane_slab_nm"] / 2), zc + nm(P["membrane_slab_nm"] / 2)],
         "seal": "frame", "seal_margin": nm(0.5 * a_L), "cell": "cell",
         "lining": {"block": "domain", "value": lining_idx}, "axis_sets": names,
         "every": P["conduct_every"], "iters": P["conduct_iters"], "iters_first": 6000, "omega": 1.0,
         **({"probe_radius": nm(P["probe_nm"])} if P.get("probe_nm") else {})},
    ]
    if C3 is not None:
        # THE COOKE MEMBRANE'S OPERATORS replace the one-bead lipid's (its pair law, its rim law, its depth spring):
        # bonds inside each lipid, the head/tail table within the membrane and against the rim, and against each chain
        # a table of its own -- heads WCA at `lipid_protein_sigma_nm`, tails that WCA plus the cos^2 tail of depth
        # `lipid_protein_eps_kT` (the hydrophobic belt: a tail sticks to the protein's core, a head does not).
        B_ = C3["B"]; s3 = C3["s"]
        hb, sw, wt = nm(B_["head_b"] * s3), nm(s3), nm(B_["tail_width_sigma"] * s3)
        sp, wp = nm(P["lipid_protein_sigma_nm"]), nm(B_["tail_width_sigma"] * P["lipid_protein_sigma_nm"])
        tab = {"head": {"head": {"law": "wca", "sigma": hb}, "tail": {"law": "wca", "sigma": hb}},
               "tail": {"head": {"law": "wca", "sigma": hb},
                        "tail": {"law": "cooke", "sigma": sw, "epsilon": B_["eps_tail_kT"], "tail": wt}}}
        tabp = {"head": {"residue": {"law": "wca", "sigma": sp}},
                "tail": ({"residue": {"law": "wca", "sigma": sp}} if P.get("lipid_protein_repulsive") else
                         {"residue": {"law": "cooke", "sigma": sp, "epsilon": P["lipid_protein_eps_kT"], "tail": wp}})}
        keep = [o for o in ops if not (o["op"] in ("pair_potential", "tether") and o.get("at") == "lipid")]
        at_ = next(i for i, o in enumerate(keep) if o["op"] == "brownian" and o.get("at") == "lipid")
        new3 = [{"op": "elastic_network", "at": "lipid", "within": "mol", "cutoff": nm(2.5 * C3["b"]),
                 "k": E_to_sim(B_["k_bond_kT_sigma2"] / s3 ** 2), "go_epsilon": 0.0, "mobility": 1.0, "f_max": f_cap},
                {"op": "pair_potential", "model": "table", "at": "lipid", "pair_table": tab, "exclude": "mol",
                 "mobility": 1.0, "f_max": f_cap},
                {"op": "pair_potential", "model": "table", "at": "lipid", "with": "frame", "pair_table": tab,
                 "mobility": 1.0, "react": False, "f_max": f_cap}] + [
                {"op": "pair_potential", "model": "table", "at": "lipid", "with": nmk, "pair_table": tabp,
                 "mobility": 1.0, "mobility_with": 1.0, "react": True, "f_max": f_cap} for nmk in names]
        ops = keep[:at_] + new3 + keep[at_:]
    if P.get("insertion_drive"):
        # THE HYDROPHOBIC EFFECT ON EVERY RESIDUE (residue_slab, Wimley-White): what carries a pore-former's
        # stems across the core; along the morph path its pull is projected like the lipids' push
        ops.append({"op": "residue_slab", "at": names[0], "sets": names, "z0": round(zc, 6),
                    "half_thickness": nm(h_core or P["membrane_slab_nm"] / 2), "edge": nm(0.3), "block": "dg", "mobility": 1.0})
    if P.get("morph"):
        # THE GATE AS ONE COORDINATE (`morph`, batch 12): the protein held on the straight path between its two
        # deposited states and moved along it by what pushes on it -- the lipids above all -- against the offset
        # dV = tau_1/2 x dA (`morph_gate`); the elastic network and the rigid-domain projection it replaces could
        # not let 12 mN/m move a helix bundle through thousands of kT of springs (batch 11).
        if not basin:
            raise SystemExit("morph: needs `two_basin` (the open state) to build its path")
        ops = [o for o in ops if not (o["op"] in ("shape_match", "elastic_network") and o.get("at") != "lipid")]
        ops.insert(0, {"op": "morph_gate", "at": names[0], "sets": names, "open_reference": basin["open_reference"],
                       "open_parts": basin["open_parts"], "open_scale": basin["open_scale"], "origin": origin,
                       "basin_offset": basin["basin_offset"], "barrier": float(P.get("morph_barrier_kT", 0.0)),
                       "gate_block": ["cell", "w_open"], "mobility": 1.0,
                       **({"tension_block": ["cell", "tension"], "area_change": dA_world} if P.get("morph_tension_work") else {})})
        if P.get("morph_waypoints"):
            # PHASE G: the gate on a RECORDED path (morph_gate[waypoints]) -- the parts w<j>_c<k> of the named shape,
            # the seed's own frame (the rig's closed beads), c<k> in the protein sets' order
            wp = P["morph_waypoints"]
            ops[0].update({"model": "waypoints", "open_reference": wp["shape"], "waypoints": int(wp["n"]),
                           "open_parts": [f"c{k}" for k in range(len(names))]})
        if P.get("morph_clamp") is not None:
            # PHASE G1: the gate HELD at lambda0 on the recorded path, the lipids' push along it recorded
            # (morph_gate[clamp]; thermodynamic integration -- the free gate needs ~230 ns to cross, batch 13)
            if not P.get("morph_waypoints"):
                raise SystemExit("morph_clamp: needs `morph_waypoints` (the clamp is a waypoints model)")
            ops[0].update({"model": "clamp", "lambda0": float(P["morph_clamp"]), "force_block": ["cell", "f_gate"]})
            for k_ in ("basin_offset", "barrier", "tension_block", "area_change"):
                ops[0].pop(k_, None)
    if IL is not None:
        # THE VOLTAGE CLAMPED (as the rig's): one ion crossing a 10-20 nm patch would move it by tens of mV;
        # the conduction solve's lumen is the ion paths' (and it writes the potential the ions feel)
        for o in ops:
            if o["op"] == "membrane_potential":
                o["capacitance"] = 1e12
            if o["op"] == "electrolyte_conduction":
                o.pop("lining", None)
                o.update(IL["lumens"])
                o["path_z0"] = round(zc, 6)
        ops += IL["ops"]
    schedule = [o["op"] for o in ops]

    # ---- the picture: the cryo-EM look of 0073 ----------------------------------------------------
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from bfm_scaffold_spec import LOOKS
    surf_p = {"render": "surface", "spacing": round(nm(P["protein_spacing_nm"]), 6), "blur": P["protein_blur"],
              "iso_frac": P["protein_iso"], "smooth": 20, "specular": 0.0, "ambient": 0.32, "diffuse": 0.72}
    surf_l = {"render": "surface", "spacing": round(nm(P["lipid_spacing_nm"]), 6), "blur": P["lipid_blur"],
              "iso_frac": P["lipid_iso"], "smooth": 30, "specular": 0.0, "ambient": 0.32, "diffuse": 0.72,
              "recontour": True}
    I_axis = max(10.0, 1.5 * abs(I_open_pA))
    h_win = min(0.45, (lumen_nm + 3.0) / L)                 # the pore and 3 nm around it
    # THE CHAINS NEAREST THE CAMERA ARE NOT DRAWN (`hide_front_chains`), so the lumen, and the current
    # in it, can be seen from the side -- "two subunits removed for clarity", as a structure paper
    # opens a channel. They are simulated like the rest; only the picture leaves them out. The camera
    # looks from (cos e cos az, cos e sin az, sin e) (live_movie), so the front chains are those whose
    # centroid's azimuth about the axis is nearest az.
    hidden = []
    if P.get("hide_front_chains"):
        az = math.radians(P["camera"]["azim"])
        th = [math.atan2(float(pts[c][:, 1].mean()), float(pts[c][:, 0].mean())) for c in chains]
        order = sorted(range(nch), key=lambda k: -math.cos(th[k] - az))
        hidden = [names[k] for k in order[:int(P["hide_front_chains"])]]
    drawn = [n for n in names if n not in hidden]
    if P.get("lipid_near_side"):                            # the slab opened in front of the channel
        surf_l["near_side"] = P["lipid_near_side"]
    plotting = {
        "renderer": "vtk_points", "up_axis": 2, "box_frame": True, **LOOKS["cryo"],
        "render_3d": "compartments", "compartment_sets": drawn + ["lipid"],
        "hide_sets": ["frame", "cell"] + hidden,
        "surface": {**{n: dict(surf_p) for n in names}, "lipid": surf_l},
        "surface_min_points": 30,
        "opacity": {**{n: P["protein_opacity"] for n in names}, "lipid": P["lipid_opacity"]},
        "colors": {**{n: CHAIN_COLORS[i % len(CHAIN_COLORS)] for i, n in enumerate(names)}, "lipid": LIPID_COLOR},
        # THE CURRENT AS A DENSITY: two nested isosurfaces of |j| (pA/nm^2), 15% and 50% of the density
        # the paper's open pore would carry -- the jet through the pore and the bulbs at its mouths,
        # drawn like a map, glowing on black. (A textured plane through the translucent bilayer
        # mis-blended: render smoke test, 2026-09-25.)
        "field_iso": {"field": "elec", "channel": 0, "levels": [round(0.15 * j_open, 6), round(0.5 * j_open, 6)],
                      "colors": ["#ff7b1c", "#ffe27a"], "opacity": [0.28, 0.75], "smooth": 15},
        "camera": P["camera"], "zoom": P["zoom"],
        **({"near_side": P["near_side"]} if P["near_side"] else {}), "near_side_centre": origin,
        # THREE PANELS, the renderer's legible maximum: the current (its title carries the potential it
        # flows at, which the cell's capacitance holds within 0.1 mV over the run -- a flat fourth panel
        # would say only that), the pore, and the tension that opens it.
        "curve": [
            {"quantity": "block:cell:j_channel", "unit": "current_pA", "smooth": 20.0,
             "ymin": -I_axis if I_open_pA < 0 else -0.05 * I_axis, "ymax": 0.05 * I_axis if I_open_pA < 0 else I_axis,
             "ylabel": f"current at {P['psi_mV']:+.0f} mV"},
            {"quantity": "block:cell:pore_r", "unit": "length_nm", "factor": 2.0, "ymin": 0.0,
             "ymax": round(2 * r_open * 1e9 * 1.6, 1), "ylabel": "pore diameter"},
            {"quantity": "block:cell:tension", "unit": "tension_mN_m", "smooth": 50.0, "ymin": -5.0,
             "ymax": round(P["tension_max_mN_m"] * 1.3, 0), "ylabel": "membrane tension"}],
        "subject": names[0], "keep_stills": True, "stills": 8, "max_frames": int(P.get("movie_frames", 300)),
        "real_time": False, "duration_s": 10.0, "curve_time": {"per_frame_s": dt * tau_s, "unit": "ns"},
        # THE LIVE MOVIE IS THE MOVIE: a replay from trajectory.npz has no grid fields and would drop
        # the current slice; every panel declares ymin/ymax, so the live pass draws them.
        "replay_curves": False,
    }
    if IL is not None:
        pl = IL["plot"]
        plotting["compartment_sets"] = plotting["compartment_sets"] + pl["sets"]
        plotting["hide_sets"] = plotting["hide_sets"] + pl["hide"]
        plotting["surface"].update(pl["surface"]); plotting["opacity"].update(pl["opacity"]); plotting["colors"].update(pl["colors"])
        plotting["field_iso"] = pl["field_iso"]                   # the cations' density: where they crowd
        # the moving gate's panels: the charge moved, the GATE itself (lambda, 0 closed -> 1 open), the tension
        plotting["curve"] = [pl["curves"][0],
                             {"quantity": "block:cell:w_open", "ymin": 0.0, "ymax": 1.0,
                              "ylabel": "gate: 0 closed, 1 open"}, plotting["curve"][2]]
    if P.get("morph_clamp") is not None:
        # PHASE G panels (the human, 2026-09-29: "replace the middle curve by a tension plot"): the push along the
        # path, the TENSION in the middle, the pore
        plotting["curve"] = [{"quantity": "block:cell:f_gate", "smooth": 200.0, "ymin": -400.0, "ymax": 400.0,
                              "ylabel": "push along path (kT)"},
                             plotting["curve"][2], plotting["curve"][1]]
    for cv in plotting["curve"]:
        cv.setdefault("font_size", 21); cv.setdefault("tick_font_size", 16)

    spec = {
        "general": {"name": f"exp04_v{label}", "seed": 0, "n_frames": n_frames, "dt": dt, "boundary": "wall",
                    "dim": 3, "world": [1.0, 1.0, 1.0], "record_cap": 3001,
                    "field_record_cap": 2, "units": units},
        "sets": sets,
        "fields": {"elec": {"frame": "grid", "res": P["n_grid"], "components": 2 if IL is not None else 1},
                   **({"cden": {"frame": "grid", "res": P["n_grid"], "components": 1}} if IL is not None else {})},
        "seed": seeds,
        "operators": ops,
        "schedule": schedule,
        "plotting": plotting,
    }
    out = os.path.join(REPO, "config", "channel", f"exp04_v{label}.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(_py(titled(spec)), f, sort_keys=False, default_flow_style=None, width=110)

    pred = {
        "box_nm": L, "grid_dx_nm": round(L / P["n_grid"], 3), "chains": nch, "beads_protein": int(sum(len(pts[c]) for c in chains)),
        "lipids": int(len(lipid)), "frame_beads": int(len(frame)), "R_patch_nm": round(R_patch, 2),
        "a_lipid_nm": round(a_L, 3), "sigma_LL_nm": round(sigma_LL, 3), "eps_LL_kT": round(eps_LL, 3),
        "K_A_T0_mN_m": round(KA_T0_mN_m, 1), "K_A_real_mN_m": P["K_A_mN_m"],
        "tau_s_per_sim": tau_s, "dt_sim": dt, "dt_ps": dt * tau_s * 1e12, "n_frames": n_frames,
        "tau_patch_frames": round(fr_tau), "protocol_frames": [p[0] for p in prot],
        "sim_time_ns": n_frames * dt * tau_s * 1e9,
        "drive_top": T, "drive_mode": P["drive_mode"], "G_open_hille_nS": G_open * 1e9, "I_open_pA": I_open_pA,
        "deposited_constriction_diameter_nm": info.get("constriction_nm", {}).get("diameter"),
        "psi_sim": psi_sim, "volt_per_sim": volt_per_sim, "lumen_radius_nm": round(lumen_nm, 2),
        "go_eps_kT": round(float(P["go_eps_kT"]), 4),
        **({"go_eps_between_chains_kT": round(float(P["go_eps_between_chains_kT"]), 4)}
           if P.get("go_eps_between_chains_kT") is not None else {}),
        **({"go_derivation": gnote} if gnote else {}),
        **({"ions": IL["record"], "state": f"{P.get('rig_shape')} -> {P.get('rig_open')} under tension",
            "rig_shape": P.get("rig_shape"), "core_half_nm_opm": h_core} if IL is not None else {})}
    print(f"wrote {os.path.relpath(out, REPO)}  ({VERSIONS[label]['why']}){scale_note}")
    for k, v in pred.items():
        print(f"  {k:36s} {v}")
    json.dump(pred, open(out.replace(".yaml", ".pred.json"), "w"), indent=1, default=float)
    return spec, pred


# ================================================================================ Kv1.2: ions in a pore
ION = {
    "n_grid": 128,
    "eta_Pa_s": 1.0e-3, "bead_stokes_nm": 0.3,
    "psi_mV": 150.0,                  # depolarised: K+ flows OUT (bottom = cytoplasm -> top)
    "conc_mM": 150.0,                 # symmetric KCl; K+ explicit, the anions a Debye screen
    "D_K_m2_s": 1.96e-9,              # K+ in water (Hille)
    "l_B_nm": 0.70,                   # Bjerrum length in water at 298 K
    "slab_eps_ratio": 4.0,            # eps_water / eps in the filter and cavity (both ions in the slab)
    "q_O": -0.5, "q_C": 0.5,          # a backbone carbonyl's partial charges (a dipole: C+ O-)
    "sigma_KO_nm": 0.23, "sigma_KC_nm": 0.28, "sigma_KCA_nm": 0.30, "sigma_KL_nm": 0.45, "sigma_KK_nm": 0.26,
    "annulus_nm": 2.0, "bath_nm": 4.0, "area_per_lipid_nm2": 0.65, "leaflet_z_nm": 1.0, "lipid_excl_nm": 0.75,
    "membrane_slab_nm": 4.0, "ins_protein_nm": 0.5, "ins_lipid_nm": 0.55, "core_radius_nm": 0.2,
    "sigma_S_m": 1.9,                 # 150 mM KCl
    "n_frames": 1000000, "dt_safety": 0.3,
    "density_every": 20, "density_window": 20000.0,
    "camera": {"elev": 40.0, "azim": 30.0}, "zoom": 1.0,     # step 0007's view (the human, 2026-09-26)
    "hide_front_chains": 0,
    "current_window": 200000.0,       # the current's running mean, frames (~28 ns): one ion is ~6 pA, not a 56 pA spike
    # ---- after round 4 (its judges): a membrane that insulates, a filter that breathes, a cut-away --------
    "born": True,                     # the hydrocarbon core's Born barrier to ions, outside the pore (slab_barrier)
    "ion_radius_nm": 0.138, "eps_core": 2.0, "eps_water": 78.5,   # K+ (Pauling); the core; water -> W ~99 kT
    "core_half_nm": 1.4, "core_edge_nm": 0.3,                     # POPC's hydrocarbon core ~2.8 nm thick
    "pore_radius_nm": 1.0, "pore_edge_nm": 0.2,                   # inside it the protein's beads decide
    "pore_profile": False,            # the barrier's pore narrowed over the filter's heights (below)
    "filter_pore_nm": 0.35,           # ... to this radius (the filter's O rings stand 0.2-0.3 nm off the axis)
    "solve_iters_first": 8000,        # Gauss-Seidel sweeps of the (one) electrolyte solve
    "chain_opacity": 1.0,             # the drawn chains' opacity (translucent shows the ions in the filter)
    "filter_friction": None,          # K+ mobility in the filter as a fraction of bulk (BD models of K channels: 0.1)
    "lumen_radius_nm": 1.0,           # the solve: the slab conducts only within 1 nm of the axis (it leaked 71.6%)
    "flex_filter": True,              # the carbonyls tethered to their deposited places by their own B-factors
    "B_O_A2": 15.2, "B_C_A2": 11.7,   # 8VC6's mean B over residues 374-378 (O: 9.7-23.5, C: 7.1-20.0)
    "coulomb_cap_nm": 0.1,            # the one guard left on an ion's forces: Coulomb only; the WCA cores are walls
    "show_radius_nm": 1.5, "show_half_height_nm": 3.5,             # ions drawn only near the pore
    "n_frames_override": None,
}


def build_ions(label, P, channel):
    """Kv1.2: the deposited conducting pore, static; potassium ions as the one moving set."""
    folder, info, pts = _shape(channel)
    chains = sorted([k for k in pts if k.startswith("c") and k[1:].isdigit()], key=lambda s: int(s[1:]))
    nch = len(chains)
    X_all = pts["all"] * 1e9
    rho = np.hypot(X_all[:, 0], X_all[:, 1])
    r_tm = float(rho[np.abs(X_all[:, 2]) < 2.0].max())
    R_patch = r_tm + P["annulus_nm"]
    a_L = math.sqrt(2.0 * P["area_per_lipid_nm2"] / math.sqrt(3.0))
    z_lo = min(float(X_all[:, 2].min()), -P["membrane_slab_nm"] / 2) - P["bath_nm"]
    z_hi = max(float(X_all[:, 2].max()), P["membrane_slab_nm"] / 2) + P["bath_nm"]
    L = math.ceil(max(2 * (R_patch + 1.0), z_hi - z_lo))
    zc = -z_lo / L + 0.5 * (1.0 - (z_hi - z_lo) / L)
    world_per_m = 1.0 / (L * 1e-9)
    nm = lambda x: x / L                                                  # noqa: E731

    zeta = 6 * math.pi * P["eta_Pa_s"] * P["bead_stokes_nm"] * 1e-9
    tau_s = zeta * (L * 1e-9) ** 2 / KT_J
    force_nN = KT_J / (L * 1e-9) * 1e9
    units = {"length_um": L * 1e-3, "time_s": tau_s, "force_nN": force_nN}
    mu_K = P["D_K_m2_s"] / (KT_J / zeta)                                  # the ion's mobility / a residue's
    volt_per_sim = (force_nN * 1e-9) * (L * 1e-9) / E_C
    psi_sim = P["psi_mV"] * 1e-3 / volt_per_sim
    debye_nm = 0.304 / math.sqrt(P["conc_mM"] / 1000.0)
    k_c = P["l_B_nm"] / L                                                 # kT x l_B, sim energy x world

    # ---- the membrane: static lipid beads (insulator, the ions' wall, the picture) ----------------
    lat = hex_lattice(a_L, R_patch + 0.1)
    lipid = []
    for zl in (-P["leaflet_z_nm"], P["leaflet_z_nm"]):
        near = np.abs(X_all[:, 2] - zl) < 1.2
        Pn = X_all[near]
        th_p = np.arctan2(Pn[:, 1], Pn[:, 0]); r_p = np.hypot(Pn[:, 0], Pn[:, 1])
        for (x, y) in lat:
            r = math.hypot(x, y)
            if r > R_patch:
                continue
            d = np.sqrt((Pn[:, 0] - x) ** 2 + (Pn[:, 1] - y) ** 2 + (Pn[:, 2] - zl) ** 2)
            if d.min() < P["lipid_excl_nm"]:
                continue
            dth = np.abs((th_p - math.atan2(y, x) + np.pi) % (2 * np.pi) - np.pi)
            sect = r_p[dth < math.radians(15)]
            if sect.size and r < sect.max():
                continue
            lipid.append((x, y, zl))
    lipid = np.array(lipid)

    # ---- the filter's sites, from its carbonyl oxygens: 5 rings, 4 sites between them ----------------
    O = pts["filter_O"] * 1e9
    zO = np.sort(np.unique(np.round(O[:, 2], 1)))
    rings = []
    for zz in sorted(O[:, 2]):
        if not rings or abs(zz - rings[-1][-1]) > 0.15:
            rings.append([zz])
        else:
            rings[-1].append(zz)
    zr = [float(np.mean(r)) for r in rings]
    sites = [0.5 * (a + b) for a, b in zip(zr[:-1], zr[1:])]              # S4 (lowest) ... S1 (highest)
    filter_z = [min(zr) - 0.1, max(zr) + 0.1]

    # ---- the ions: two in the filter where Wu 2023 sees density (IS1, IS3), the rest in the bath --------
    rng = np.random.default_rng(0)
    half = L / 2
    vol = (L * L) * ((z_hi - z_lo) - P["membrane_slab_nm"])
    n_ion = int(round(P["conc_mM"] * 1e-3 * 602.2 * 1e-3 * vol))       # mM x 0.6022 per nm^3 / 1000
    ions = []
    if len(sites) >= 3:
        ions += [(0.0, 0.0, sites[-1]), (0.0, 0.0, sites[-3])]            # the highest (S1) and the third (S3)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.concatenate([X_all, lipid]))
    while len(ions) < n_ion:
        p_ = np.array([rng.uniform(-half + 0.3, half - 0.3), rng.uniform(-half + 0.3, half - 0.3),
                       rng.uniform(z_lo + 0.3, z_hi - 0.3)])
        if abs(p_[2]) < P["membrane_slab_nm"] / 2 + 0.3:
            continue
        if tree.query(p_)[0] < 0.6:
            continue
        ions.append(tuple(p_))
    ions = np.array(ions)
    pz = dict(np.load(os.path.join(folder, "points.npz")))
    pz["lipid"] = (lipid * 1e-9).astype(np.float32)
    pz["K"] = (ions * 1e-9).astype(np.float32)
    np.savez_compressed(os.path.join(folder, "points.npz"), **pz)

    # ---- time step from the stiffest pair the ion meets --------------------------------------------
    wca_curv = 57.1 / nm(P["sigma_KO_nm"]) ** 2
    dt = P["dt_safety"] / (mu_K * wca_curv)
    f_cap = nm(0.02) / (mu_K * dt)
    c_cap = nm(P.get("coulomb_cap_nm", 0.02)) / (mu_K * dt)
    # THE PORE'S OWN RADIUS AT EACH HEIGHT, for the Born barrier: the nearest alpha carbon of the pore
    # domain (S5-P-S6) to the axis in a 0.6 nm slice, less an ion-residue contact (0.45 nm), never under
    # 0.2 nm. A 1 nm cylinder let ions slip between the filter's helices (round 6: 175-380 pS).
    pore_prof = []
    if P.get("pore_profile"):
        # the FILTER'S heights (its carbonyl rings, +-0.1 nm) are narrowed to `filter_pore_nm`; the cavity
        # below and the vestibule above keep `pore_radius_nm`. Round 6: 13 of 51 ions (6a) and 15 of 168
        # (6c) went round the filter through a bead-free annulus 0.45-0.91 nm off the axis behind its
        # strands, where the real protein has side chains.
        O_ = pts["filter_O"] * 1e9
        zf0, zf1 = float(O_[:, 2].min()) - 0.1, float(O_[:, 2].max()) + 0.1
        for z_ in np.arange(-P["core_half_nm"] - 0.4, P["core_half_nm"] + 0.41, 0.05):
            rp_ = P["filter_pore_nm"] if zf0 <= z_ <= zf1 else P["pore_radius_nm"]
            pore_prof.append((round(float(z_), 3), rp_))
    # the Born energy of K+ from water into the core (Parsegian 1969): l_B(vacuum) / (2 a) x (1/eps_m - 1/eps_w)
    W_born = 56.0 / (2.0 * P["ion_radius_nm"]) * (1.0 / P["eps_core"] - 1.0 / P["eps_water"])
    # the carbonyls' tethers from their B-factors: <u^2> per axis = B / (8 pi^2), k = kT / <u^2>
    k_O = 8 * math.pi ** 2 / (P["B_O_A2"] / 100.0); k_C = 8 * math.pi ** 2 / (P["B_C_A2"] / 100.0)   # kT/nm^2
    n_frames = int(P["n_frames_override"] or P["n_frames"])

    names = [f"Kv12_{k + 1}" for k in range(nch)]
    static = {"pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"}}
    sets = {"cell": {"n": 1, "start": [[0.5, 0.5, round(zc, 6)]], "state": {
        "pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"},
        "psi": {"width": 1, "integration": "first_order", "unit": "voltage"},
        "j_channel": {"width": 1, "integration": "none", "unit": "current"},
        "n_filter": {"width": 1, "integration": "none", "unit": "count"},
        "n_crossed": {"width": 1, "integration": "none", "unit": "count"}}}}
    for nmk, ck in zip(names, chains):
        sets[nmk] = {"n": int(len(pts[ck])), "start": [[0.5, 0.5, round(zc, 6)]], "state": dict(static)}
    for s in ("filter_O", "filter_C"):
        sets[s] = {"n": int(len(pts[s])), "start": [[0.5, 0.5, round(zc, 6)]],
                   "state": ({"pos": {"width": 3, "role": "coordinate", "integration": "first_order", "boundary": "world"}}
                             if P.get("flex_filter") else dict(static))}
    sets["lipid"] = {"n": int(len(lipid)), "start": [[0.5, 0.5, round(zc, 6)]], "state": dict(static)}
    sets["K"] = {"n": int(len(ions)), "start": [[0.5, 0.5, round(zc, 6)]], "state": {
        "pos": {"width": 3, "role": "coordinate", "integration": "first_order", "boundary": "world"}}}
    origin = [0.5, 0.5, round(zc, 6)]
    seeds = [{"op": "cloud_seed", "at": s, "cloud": f"exp04_{channel}/{c}", "origin": origin, "scale": world_per_m}
             for s, c in list(zip(names, chains)) + [("filter_O", "filter_O"), ("filter_C", "filter_C"),
                                                     ("lipid", "lipid"), ("K", "K")]]
    slab = {"z": [zc + nm(-P["membrane_slab_nm"] / 2), zc + nm(P["membrane_slab_nm"] / 2)], "eps_ratio": P["slab_eps_ratio"]}
    coul = {"law": "coulomb", "k_c": k_c, "debye": nm(debye_nm), "cutoff": nm(2.5), "slab": slab, "f_max": c_cap}
    flex = bool(P.get("flex_filter"))
    # THE WCA CORES ARE WALLS: round 4's cap (0.02 nm per step, ~72 kT/nm) clamped the K-O core below the
    # carbonyls' own pull and made the oxygens penetrable (every transit passed 0.07-0.13 nm from an O
    # centre, contact 0.258). With NO cap an ion seeded 0.25 nm from a ring was fired 8 nm in one record
    # (local k6u, 2026-09-26). The guard stays, at 0.05 nm per step (~180 kT/nm), above the carbonyls'
    # pull (~55 kT/nm per O at 0.2 nm), so the wall holds and a single step cannot overshoot it.
    wca_cap = {"f_max": nm(P.get("wca_cap_nm", 0.05)) / (mu_K * dt)} if flex else {"f_max": f_cap}
    # (round 6's judges: with the Coulomb capped at 0.1 nm per step and the core at 0.05, the 8x pull of a
    # carbonyl beat its own wall -- ions sat 0.17 nm from an O centre, one 0.008 nm; the wall's guard must
    # exceed the pull's, `wca_cap_nm` > `coulomb_cap_nm`)
    react_O = {"react": True, "mobility_with": 1.0} if flex else {"react": False}
    ops = [
        {"op": "membrane_potential", "at": "cell", "capacitance": 1e12, "pump_max": 0.0, "pump_rev": 1.0,
         "leak": 0.0, "psi0": psi_sim, "h0": [0.0, 0.0], "flux": "j_channel"},
        {"op": "electrolyte_conduction", "at": "cell", "field": "elec",
         "insulators": [{"set": s, "radius": nm(P["ins_protein_nm"])} for s in names]
                       + [{"set": "lipid", "radius": nm(P["ins_lipid_nm"])}],
         "sigma": P["sigma_S_m"], "volt_per_sim": volt_per_sim, "dx_m": L * 1e-9 / P["n_grid"],
         "sim_current_per_A": tau_s / E_C, "slab": slab["z"], "cell": "cell", "axis_sets": names,
         "lining": {"block": "domain", "value": 1} if False else None,
         "lumen_core": {"radius": nm(P["core_radius_nm"]), "z": [zc + nm(-2.5), zc + nm(2.5)]},
         **({"lumen_radius": nm(P["lumen_radius_nm"])} if P.get("lumen_radius_nm") else {}),
         "every": 200000, "iters": 300, "iters_first": int(P.get("solve_iters_first", 8000)), "omega": 1.0, "cell_writes": False},
        {"op": "pair_potential", "at": "K", **coul, "charge": 1.0, "mobility": mu_K},
        {"op": "pair_potential", "at": "K", "with": "filter_O", **coul, "charge": 1.0, "charge_with": P["q_O"],
         "mobility": mu_K, **react_O},
        {"op": "pair_potential", "at": "K", "with": "filter_C", **coul, "charge": 1.0, "charge_with": P["q_C"],
         "mobility": mu_K, **react_O},
        {"op": "pair_potential", "at": "K", "law": "wca", "sigma": nm(P["sigma_KK_nm"]), "epsilon": 1.0,
         "mobility": mu_K, **wca_cap},
        {"op": "pair_potential", "at": "K", "with": "filter_O", "law": "wca", "sigma": nm(P["sigma_KO_nm"]),
         "epsilon": 1.0, "mobility": mu_K, **react_O, **wca_cap},
        {"op": "pair_potential", "at": "K", "with_sets": names, "law": "wca", "sigma": nm(P["sigma_KCA_nm"]),
         "epsilon": 1.0, "mobility": mu_K, "react": False, **wca_cap},
        {"op": "pair_potential", "at": "K", "with": "lipid", "law": "wca", "sigma": nm(P["sigma_KL_nm"]),
         "epsilon": 1.0, "mobility": mu_K, "react": False, **wca_cap},
        {"op": "field_force", "at": "K", "field": "elec", "channel": 0, "charge": 1.0, "mobility": mu_K,
         "volt_per_sim": volt_per_sim},
    ] + ([{"op": "brownian", "at": "K", "kT": 1.0, "mobility": mu_K, "seed": 3}] if not P.get("filter_friction") else []) + ([
        # THE FILTER BREATHES: each carbonyl O and C held to its deposited place by a spring of its own
        # B-factor's stiffness, kicked by the bath like any residue, pushed by the ions it holds
        {"op": "tether", "at": "filter_O", "k": k_O * L * L, "axes": [0, 1, 2], "mobility": 1.0},
        {"op": "tether", "at": "filter_C", "k": k_C * L * L, "axes": [0, 1, 2], "mobility": 1.0},
        {"op": "brownian", "at": "filter_O", "kT": 1.0, "mobility": 1.0, "seed": 4},
        {"op": "brownian", "at": "filter_C", "kT": 1.0, "mobility": 1.0, "seed": 5},
    ] if flex else []) + ([
        {"op": "slab_barrier", "at": "K", "height": W_born, "z0": round(zc, 6), "half_thickness": nm(P["core_half_nm"]),
         "edge": nm(P["core_edge_nm"]), "pore_radius": nm(P["pore_radius_nm"]), "pore_edge": nm(P["pore_edge_nm"]),
         "axis_xy": [0.5, 0.5], "mobility": mu_K,
         **({"pore_profile": [[round(zc + nm(z_), 6), round(nm(r_), 6)] for z_, r_ in pore_prof]} if pore_prof else {})},
    ] if P.get("born") else []) + [
        {"op": "flux_counter", "at": "K", "z_m": round(zc, 6), "radius": nm(1.0), "window": P.get("current_window", 20000.0), "charge": 1.0,
         "cell": "cell", "block": "j_channel", "wrap": True, "sign": -1.0,
         "filter_z": [zc + nm(filter_z[0]), zc + nm(filter_z[1])], "occupancy_block": "n_filter", "axis_xy": [0.5, 0.5]},
        {"op": "density_field", "at": "K", "field": "kden", "s": nm(0.12), "every": P["density_every"],
         "window": P["density_window"], "within": [0.5 - nm(2.0), 0.5 + nm(2.0), 0.5 - nm(2.0), 0.5 + nm(2.0),
                                                   zc + nm(-3.0), zc + nm(3.0)]},
    ]
    if P.get("filter_friction"):
        # SINGLE-FILE FRICTION IN THE FILTER (friction_zone, which replaces K's brownian and runs first):
        # the mobility of K+ inside the filter's cylinder cut to `filter_friction` of the bulk's
        O_ = pts["filter_O"] * 1e9
        ops.insert(0, {"op": "friction_zone", "at": "K", "factor": float(P["filter_friction"]),
                       "radius": nm(P["filter_pore_nm"]), "z": [zc + nm(float(O_[:, 2].min()) - 0.1), zc + nm(float(O_[:, 2].max()) + 0.1)],
                       "edge": nm(0.1), "axis_xy": [0.5, 0.5], "kT": 1.0, "mobility": mu_K, "seed": 3})
    for o in ops:
        if o.get("lining") is None:
            o.pop("lining", None)
        o.pop("cell_writes", None)
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from bfm_scaffold_spec import LOOKS
    surf_p = {"render": "surface", "spacing": round(nm(0.3), 6), "blur": 1.5, "iso_frac": 0.3, "smooth": 20,
              "specular": 0.0, "ambient": 0.32, "diffuse": 0.72}
    bulk_density = P["conc_mM"] * 1e-3 * 602.2 * 1e-3 * L ** 3              # ions per world^3
    # TWO SUBUNITS REMOVED FOR CLARITY, as a potassium-channel figure shows the filter: the two chains
    # nearest the camera are simulated but not drawn, the other two keep their closed skins, and the
    # membrane slab is cut in front. Round 4's `near_side: true` KEPT the near half and hid the pore; a
    # planar cut (`far`) left the chains as hollow, saw-edged shells (crash check, 2026-09-26).
    az = math.radians(P["camera"]["azim"])
    th = [math.atan2(float(pts[c][:, 1].mean()), float(pts[c][:, 0].mean())) for c in chains]
    order = sorted(range(nch), key=lambda k: -math.cos(th[k] - az))
    hidden = [names[k] for k in order[:int(P.get("hide_front_chains", 0))]]
    plotting = {
        "renderer": "vtk_points", "up_axis": 2, "box_frame": True, **LOOKS["cryo"],
        "render_3d": "compartments", "compartment_sets": [n for n in names if n not in hidden] + ["lipid"],
        "hide_sets": ["cell", "filter_O", "filter_C"] + hidden,
        "surface": {**{n: dict(surf_p) for n in names},
                    "lipid": {"render": "surface", "spacing": round(nm(0.4), 6), "blur": 2.0, "iso_frac": 0.3, "smooth": 30,
                              "specular": 0.0, "ambient": 0.32, "diffuse": 0.72,
                              **({"near_side": "far"} if P.get("lipid_near_side") else {})}},
        "surface_min_points": 30,
        "near_side_centre": [0.5, 0.5, round(zc, 6)],
        "opacity": {**{n: float(P.get("chain_opacity", 1.0)) for n in names}, "lipid": 0.45},
        "colors": {**{n: CHAIN_COLORS[i % len(CHAIN_COLORS)] for i, n in enumerate(names)}, "lipid": LIPID_COLOR},
        "spheres": {"set": "K", "radius": round(nm(0.2), 6), "color": "#b388ff", "specular": 0.0, "ambient": 0.35, "diffuse": 0.75,
                    "within": {"centre": [0.5, 0.5, round(zc, 6)], "radius": nm(P["show_radius_nm"]),
                               "half_height": nm(P["show_half_height_nm"])}},
        # the K+ density at 3x and 6x the bath's: round 4's 20x and 80x sat above the field's own peak (8.25x)
        "field_iso": {"field": "kden", "channel": 0, "levels": [round(3 * bulk_density, 3), round(6 * bulk_density, 3)],
                      "colors": ["#9a6bff", "#e3d4ff"], "opacity": [0.25, 0.7], "smooth": 10},
        "camera": P["camera"], "zoom": P["zoom"],
        "curve": [
            {"quantity": "block:cell:j_channel", "unit": "current_pA", "ymin": -2.0, "ymax": 12.0,
             "ylabel": f"current at {P['psi_mV']:+.0f} mV"},
            {"quantity": "block:cell:n_filter", "ymin": 0.0, "ymax": 4.0, "ylabel": "K+ in the filter"},
            {"quantity": "block:cell:n_crossed", "ymin": 0.0, "ymax": 10.0, "ylabel": "K+ across"}],
        "subject": names[0], "keep_stills": True, "stills": 8, "max_frames": 300,
        "real_time": False, "duration_s": 10.0, "replay_curves": False, "curve_time": {"per_frame_s": dt * tau_s, "unit": "ns"},
    }
    for cv in plotting["curve"]:
        cv.setdefault("font_size", 21); cv.setdefault("tick_font_size", 16)
    spec = {
        "general": {"name": f"exp04_v{label}", "seed": 0, "n_frames": n_frames, "dt": dt, "boundary": "wall",
                    "dim": 3, "world": [1.0, 1.0, 1.0], "record_cap": 3001,
                    "field_record_cap": 2, "units": units},
        "sets": sets,
        "fields": {"elec": {"frame": "grid", "res": P["n_grid"], "components": 2},
                   "kden": {"frame": "grid", "res": P["n_grid"], "components": 1}},
        "seed": seeds, "operators": ops, "schedule": [o["op"] for o in ops], "plotting": plotting,
    }
    out = os.path.join(REPO, "config", "channel", f"exp04_v{label}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(_py(titled(spec)), f, sort_keys=False, default_flow_style=None, width=110)
    G_expect = 20e-12
    pred = {"box_nm": L, "grid_dx_nm": round(L / P["n_grid"], 3), "chains": nch, "ions": int(len(ions)),
            "lipids": int(len(lipid)), "filter_rings_z_nm": [round(z, 2) for z in zr], "sites_z_nm": [round(s, 2) for s in sites],
            "mu_K": round(mu_K, 3), "debye_nm": round(debye_nm, 3), "dt_ps": dt * tau_s * 1e12, "n_frames": n_frames,
            "sim_time_ns": n_frames * dt * tau_s * 1e9,
            "events_expected_at_20pS": G_expect * abs(P["psi_mV"]) * 1e-3 / E_C * n_frames * dt * tau_s,
            "born_barrier_kT": round(W_born, 1), "k_O_kT_nm2": round(k_O, 1), "k_C_kT_nm2": round(k_C, 1)}
    print(f"wrote {os.path.relpath(out, REPO)}  ({VERSIONS[label]['why']})")
    for k, v in pred.items():
        print(f"  {k:36s} {v}")
    json.dump(_py(pred), open(out.replace(".yaml", ".pred.json"), "w"), indent=1)
    return spec, pred


# ================================================================================ alpha-hemolysin: assembly
ASSEMBLY = {
    "eta_Pa_s": 1.0e-3, "bead_stokes_nm": 0.3,
    "n_protomers": 8,                 # one more than a heptamer: which ring closes is the question
    "offset_nm": 1.0,                 # seeded this far outside the heptamer's own ring radius
    "turn_deg": 10.0,                 # and turned about their own vertical axis by up to this much
    "jitter_deg": 3.0,                # and misplaced around the ring by up to this much
    "go_eps_kT": 1.0, "go_cutoff_nm": 0.8,
    "wca_sigma_nm": 0.45,
    "rim_k_kT_nm2": 50.0,             # the rim held at the headgroups' height (its lipid-binding residues)
    "annulus_nm": 3.0, "bath_nm": 3.0, "area_per_lipid_nm2": 0.65, "leaflet_z_nm": 1.0,
    "n_frames": 120000, "dt_safety": 0.3, "seed": 0,
    "camera": {"elev": 55.0, "azim": 30.0}, "zoom": 1.0,
    "n_frames_override": None,
}


def build_assembly(label, P, channel):
    """Eight protomers of one deposited chain, placed loose around a ring, held only by the ring's
    own interface (a template of residue pairs) -- and asked which ring they close."""
    folder, info, pts = _shape(channel)
    ref = pts["c0"] * 1e9                                                  # nm, the reference protomer
    rb = dict(np.load(os.path.join(folder, "blocks.npz")))
    resid = rb["c0_resid"]
    lo_r, hi_r = ANATOMY[channel]["rim_residues"]
    rim = ((resid >= lo_r) & (resid <= hi_r)).astype(np.float32)
    N = int(P["n_protomers"])
    rng = np.random.default_rng(int(P["seed"]))
    c0 = ref.mean(0)
    th0 = math.atan2(c0[1], c0[0]); r0 = math.hypot(c0[0], c0[1])

    def rotz(X, a, about=(0.0, 0.0)):
        ca, sa = math.cos(a), math.sin(a)
        Y = X.copy()
        x, y = X[:, 0] - about[0], X[:, 1] - about[1]
        Y[:, 0] = about[0] + ca * x - sa * y
        Y[:, 1] = about[1] + sa * x + ca * y
        return Y
    prot = []
    for k in range(N):
        a = 2 * math.pi * k / N + math.radians(rng.uniform(-P["jitter_deg"], P["jitter_deg"]))
        Y = rotz(ref, a)                                                  # to its slot, facing the axis
        u = np.array([math.cos(th0 + a), math.sin(th0 + a), 0.0])
        Y = Y + P["offset_nm"] * u                                        # out from the ring
        cy = Y.mean(0)
        Y = rotz(Y, math.radians(rng.uniform(-P["turn_deg"], P["turn_deg"])), about=(cy[0], cy[1]))
        prot.append(Y)
    allp = np.concatenate(prot)
    r_max = float(np.hypot(allp[:, 0], allp[:, 1]).max())
    R_patch = r_max + P["annulus_nm"]
    a_L = math.sqrt(2.0 * P["area_per_lipid_nm2"] / math.sqrt(3.0))
    z_lo = -P["leaflet_z_nm"] - 1.0 - P["bath_nm"]
    z_hi = float(allp[:, 2].max()) + P["bath_nm"]
    L = math.ceil(max(2 * (R_patch + 1.0), z_hi - z_lo))
    zc = -z_lo / L + 0.5 * (1.0 - (z_hi - z_lo) / L)
    world_per_m = 1.0 / (L * 1e-9)
    nm = lambda x: x / L                                                  # noqa: E731
    lat = hex_lattice(a_L, R_patch + 0.1)
    lipid = np.array([(x, y, zl) for zl in (-P["leaflet_z_nm"], P["leaflet_z_nm"]) for (x, y) in lat
                      if math.hypot(x, y) <= R_patch])
    pz = dict(np.load(os.path.join(folder, "points.npz")))
    for k, Y in enumerate(prot):
        pz[f"m{k}"] = (Y * 1e-9).astype(np.float32)
    pz["lipid_flat"] = (lipid * 1e-9).astype(np.float32)
    np.savez_compressed(os.path.join(folder, "points.npz"), **pz)
    np.savez_compressed(os.path.join(folder, "assembly_blocks.npz"), **{f"m{k}_rim": rim for k in range(N)})

    zeta = 6 * math.pi * P["eta_Pa_s"] * P["bead_stokes_nm"] * 1e-9
    tau_s = zeta * (L * 1e-9) ** 2 / KT_J
    force_nN = KT_J / (L * 1e-9) * 1e9
    units = {"length_um": L * 1e-3, "time_s": tau_s, "force_nN": force_nN}
    go_curv = 72.0 * P["go_eps_kT"] / nm(0.5) ** 2
    wca_curv = 57.1 / nm(P["wca_sigma_nm"]) ** 2
    dt = P["dt_safety"] / (6 * max(go_curv, wca_curv))
    f_cap = nm(0.02) / dt
    n_frames = int(P["n_frames_override"] or P["n_frames"])

    names = [f"aHL_{k + 1}" for k in range(N)]
    bead = {"pos": {"width": 3, "role": "coordinate", "integration": "first_order", "boundary": "world"},
            "rim": {"width": 1, "integration": "none", "unit": "1"}}
    sets = {"cell": {"n": 1, "start": [[0.5, 0.5, round(zc, 6)]], "state": {
        "pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"},
        "n_largest": {"width": 1, "integration": "none", "unit": "count"},
        "n_bonds": {"width": 1, "integration": "none", "unit": "count"},
        "ring": {"width": 1, "integration": "none", "unit": "1"}}}}
    for n_ in names:
        sets[n_] = {"n": int(len(ref)), "start": [[0.5, 0.5, round(zc, 6)]], "state": dict(bead)}
    sets["lipid"] = {"n": int(len(lipid)), "start": [[0.5, 0.5, round(zc, 6)]],
                     "state": {"pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"}}}
    origin = [0.5, 0.5, round(zc, 6)]
    seeds = [{"op": "cloud_seed", "at": n_, "cloud": f"exp04_{channel}/m{k}", "origin": origin, "scale": world_per_m}
             for k, n_ in enumerate(names)]
    seeds += [{"op": "seed_state_from_file", "at": n_, "file": f"shapes/exp04_{channel}/assembly_blocks.npz",
               "blocks": {"rim": f"m{k}_rim"}} for k, n_ in enumerate(names)]
    seeds += [{"op": "cloud_seed", "at": "lipid", "cloud": f"exp04_{channel}/lipid_flat", "origin": origin, "scale": world_per_m}]
    ops = [{"op": "shape_match", "at": names[0], "sets": names, "alpha": 1.0},
           {"op": "elastic_network", "at": names[0], "sets": names, "cutoff": nm(0.4), "k": 1.0,
            "go_epsilon": P["go_eps_kT"], "go_cutoff": nm(P["go_cutoff_nm"]), "go_mode": "template",
            "reference": f"exp04_{channel}", "reference_parts": ["c0", "c1"], "reference_scale": world_per_m,
            "mobility": 1.0, "f_max": f_cap},
           {"op": "pair_potential", "at": names[0], "sets": names, "law": "wca", "sigma": nm(P["wca_sigma_nm"]),
            "epsilon": 1.0, "exclude_same_set": True, "mobility": 1.0, "f_max": f_cap}]
    for i_, n_ in enumerate(names):
        ops.append({"op": "tether", "at": n_, "k": P["rim_k_kT_nm2"] * L * L, "axes": [2], "block": "rim", "mobility": 1.0})
        ops.append({"op": "brownian", "at": n_, "kT": 1.0, "mobility": 1.0, "seed": 20 + i_})
    ops.append({"op": "assembly_probe", "at": names[0], "sets": names, "contact": nm(0.7), "min_contacts": 15,
                "every": 200, "cell": "cell"})
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from bfm_scaffold_spec import LOOKS
    surf_p = {"render": "surface", "spacing": round(nm(0.3), 6), "blur": 1.5, "iso_frac": 0.3, "smooth": 20,
              "specular": 0.0, "ambient": 0.32, "diffuse": 0.72}
    plotting = {
        "renderer": "vtk_points", "up_axis": 2, "box_frame": True, **LOOKS["cryo"],
        "render_3d": "compartments", "compartment_sets": names + ["lipid"], "hide_sets": ["cell"],
        "surface": {**{n_: dict(surf_p) for n_ in names},
                    "lipid": {"render": "surface", "spacing": round(nm(0.4), 6), "blur": 2.0, "iso_frac": 0.3, "smooth": 30,
                              "specular": 0.0, "ambient": 0.32, "diffuse": 0.72}},
        "surface_min_points": 30,
        "opacity": {**{n_: 1.0 for n_ in names}, "lipid": 0.5},
        "colors": {**{n_: CHAIN_COLORS[i_ % len(CHAIN_COLORS)] for i_, n_ in enumerate(names)}, "lipid": LIPID_COLOR},
        "camera": P["camera"], "zoom": P["zoom"],
        "curve": [
            {"quantity": "block:cell:n_largest", "ymin": 0.0, "ymax": float(N + 1), "ylabel": "largest assembly"},
            {"quantity": "block:cell:n_bonds", "ymin": 0.0, "ymax": float(N + 2), "ylabel": "protomer contacts"},
            {"quantity": "block:cell:ring", "ymin": -0.1, "ymax": 1.1, "ylabel": "ring closed"}],
        "subject": names[0], "keep_stills": True, "stills": 8, "max_frames": 300,
        "real_time": False, "duration_s": 10.0, "replay_curves": False, "curve_time": {"per_frame_s": dt * tau_s, "unit": "ns"},
    }
    for cv in plotting["curve"]:
        cv.setdefault("font_size", 21); cv.setdefault("tick_font_size", 16)
    spec = {
        "general": {"name": f"exp04_v{label}", "seed": 0, "n_frames": n_frames, "dt": dt, "boundary": "wall",
                    "dim": 3, "world": [1.0, 1.0, 1.0], "record_cap": 3001, "units": units},
        "sets": sets, "fields": {}, "seed": seeds, "operators": ops,
        "schedule": [o["op"] for o in ops], "plotting": plotting,
    }
    out = os.path.join(REPO, "config", "channel", f"exp04_v{label}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(_py(titled(spec)), f, sort_keys=False, default_flow_style=None, width=110)
    d01 = np.linalg.norm(pts["c0"][:, None, :] * 1e9 - pts["c1"][None, :, :] * 1e9, axis=-1)
    pred = {"box_nm": L, "protomers": N, "beads_each": int(len(ref)), "rim_beads": int(rim.sum()),
            "interface_pairs_within_cutoff": int((d01 < P["go_cutoff_nm"]).sum()),
            "heptamer_ring_radius_nm": round(r0, 2), "seeded_radius_nm": round(r0 + P["offset_nm"], 2),
            "dt_ps": dt * tau_s * 1e12, "n_frames": n_frames, "sim_time_ns": n_frames * dt * tau_s * 1e9,
            "lipids": int(len(lipid))}
    print(f"wrote {os.path.relpath(out, REPO)}  ({VERSIONS[label]['why']})")
    for k, v in pred.items():
        print(f"  {k:36s} {v}")
    json.dump(_py(pred), open(out.replace(".yaml", ".pred.json"), "w"), indent=1)
    return spec, pred


# ---- alpha-hemolysin v2: the ring that LEAKS -------------------------------------------------------
# v1 left the stem out, so its ring could only assemble: no barrel, no lumen, no current. v2 keeps
# each protomer whole -- cap, stem and rim from the mature pore 9M4P, one rigid body -- in a FLUID
# bilayer of Cooke lipids, and lets conduction find out when there is a hole: the lumen is bounded by
# the stems' own ring (their median distance from the axis, re-read at every solve), so current can
# pass only once the stems stand close enough to seal it from the lipids, and only if the lipids that
# stood where the lumen will be have left it. STATED: the stems start inserted, as hairpins across the
# membrane. Chatterjee et al. 2026 show them detached and disordered in the earliest arcs and
# inserted only after the prepore forms; a model in which they insert is the next step, not this one.
ASSEMBLY2 = {
    "eta_Pa_s": 1.0e-3, "bead_stokes_nm": 0.3,
    "n_protomers": 7, "offset_nm": 0.8, "turn_deg": 0.0, "jitter_deg": 0.0, "seed": 0,
    "go_eps_kT": 1.0, "go_cutoff_nm": 0.8, "wca_sigma_nm": 0.45,
    "rim_k_kT_nm2": 50.0,
    "annulus_nm": 3.0, "bath_nm": 3.0, "area_per_lipid_nm2": 0.65, "leaflet_z_nm": 1.0, "lipid_excl_nm": 0.75,
    "eps_LL_kT": 1.5, "lipid_tail_over_sigma": 1.6, "lipid_sigma_over_a": 0.8909,
    "lipid_protein_eps_kT": 1.0, "lipid_protein_sigma_nm": 0.75, "lipid_z_k_kT_nm2": 100.0,
    "psi_mV": -40.0,                  # a red cell's resting potential (inside negative)
    "cap_um2": 140.0,                 # a red cell's membrane area, for its capacitance
    "sigma_S_m": 1.9,                 # 150 mM KCl
    "n_grid": 128, "conduct_every": 100, "conduct_iters": 150, "probe_nm": 0.14,
    "ins_protein_nm": 0.5, "ins_lipid_nm": 0.55, "membrane_slab_nm": 4.0,
    "n_frames": 400000, "dt_safety": 0.3,
    "camera": {"elev": 40.0, "azim": 30.0}, "zoom": 1.0,     # step 0007's view (the human, 2026-09-26)
    "hide_front": 0,
    "flexible_stem": True,            # v3: the stem a chain (seeded per `stem_start`)
    "stem_start": "prepore",          # prepore: 9M4A's upper half-barrel, tips disordered; detached: all a coil
    "tip_residues": [123, 135],       # disordered in 9M4A (prepore state IV)
    "tip_z_min_nm": -0.3,             # the tips' coil stays above the lower leaflet's beads (at -1 nm)
    "tip_lift_nm": 0.2,               # the coil's middle lifted (+) or hung (-) from the anchors' line
    "rigid_upper_stem": False,        # v4: the upper half-barrel rigid with the body, only the tips a chain
    "stem_z_min_nm": 2.4,             # the coil's floor: above the upper leaflet's head groups
    "stem_go_eps_kT": 1.0,            # the stem's own hairpin and its contacts with its body
    "insertion_drive": False,         # residue_slab on the stems: the Wimley-White side chains in the core
    "core_half_nm": 1.4, "core_edge_nm": 0.3,
    "n_frames_override": None,
}


def detached_stem(X, resid, stem_mask, z_min, seed=0, bond=0.38, n_iter=400, lift=0.9, out=0.4):
    """The prestem DETACHED AND DISORDERED (Chatterjee et al. 2026): the stem's beads re-placed as a
    random coil that joins its two anchors on the body (the residues either side of it), kept above
    the membrane (z >= z_min, nm) and off the body. A Brownian bridge between the anchors, then
    relaxed by projection: bond lengths to `bond`, beads lifted to z_min, pushed off body beads
    (>= 0.45 nm) and off each other beyond their neighbours (>= 0.45 nm). Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    idx = np.where(stem_mask)[0]
    a = X[idx[0] - 1]; b = X[idx[-1] + 1]
    n = len(idx) + 1                                                      # steps anchor -> anchor
    steps = rng.normal(0.0, bond / math.sqrt(3.0), (n, 3))
    walk = np.cumsum(steps, 0)
    t = np.arange(1, n + 1)[:, None] / n
    path = a + walk - t * walk[-1] + t * (b - a)                          # the bridge ends on b
    Y = path[:-1].copy()
    # lift the loop's middle up, under its own cap, and a little OUT from the ring's axis: eight coils
    # hung toward the axis met in the middle (their closest beads 0.02 nm apart)
    mid = 0.5 * (a + b)
    outward = np.array([mid[0], mid[1], 0.0]); outward /= max(np.linalg.norm(outward), 1e-9)
    w = np.sin(np.pi * np.arange(1, n) / n)[:, None]
    Y = Y + w * (out * outward + np.array([0.0, 0.0, lift]))
    body = X[~stem_mask]
    for _ in range(n_iter):
        chain = np.vstack([a, Y, b])
        for k in range(1, len(chain)):                                    # bonds
            d = chain[k] - chain[k - 1]; L_ = np.linalg.norm(d)
            corr = (L_ - bond) * d / max(L_, 1e-9)
            if k == 1:
                chain[k] -= corr
            elif k == len(chain) - 1:
                chain[k - 1] += corr
            else:
                chain[k] -= 0.5 * corr; chain[k - 1] += 0.5 * corr
        Y = chain[1:-1]
        Y[:, 2] = np.maximum(Y[:, 2], z_min)                               # above the membrane
        D = np.linalg.norm(Y[:, None] - body[None], axis=-1)              # off the body
        k_, j_ = np.where(D < 0.45)
        for k, j in zip(k_, j_):
            d = Y[k] - body[j]; Y[k] = body[j] + d / max(np.linalg.norm(d), 1e-9) * 0.45
        D = np.linalg.norm(Y[:, None] - Y[None], axis=-1)                 # off each other
        k_, j_ = np.where((D < 0.45) & (np.abs(np.arange(len(Y))[:, None] - np.arange(len(Y))[None]) > 2))
        for k, j in zip(k_, j_):
            if k < j:
                d = Y[k] - Y[j]; m_ = 0.5 * (Y[k] + Y[j]); u = d / max(np.linalg.norm(d), 1e-9)
                Y[k] = m_ + 0.225 * u; Y[j] = m_ - 0.225 * u
    out = X.copy(); out[idx] = Y
    return out


def relax_coils(prot, coil, z_min, bond=0.38, d_min=0.45, n_iter=600):
    """The seeded coils made to AVOID EACH OTHER once the protomers are placed: each coil (the beads
    `coil` of every protomer, a contiguous run whose neighbours either side are its anchors) is
    relaxed by projection against every other bead -- its own body, the other protomers -- keeping
    its bond lengths, its anchors and its floor z >= z_min. Returns the new protomers and the
    closest approach left (nm)."""
    idx = np.where(coil)[0]
    prot = [P_.copy() for P_ in prot]
    N = len(prot)
    for _ in range(n_iter):
        for k in range(N):
            Y = prot[k]
            chain = Y[idx[0] - 1: idx[-1] + 2].copy()
            for j in range(1, len(chain)):                                  # bonds, anchors fixed
                d = chain[j] - chain[j - 1]; L_ = np.linalg.norm(d)
                corr = (L_ - bond) * d / max(L_, 1e-9)
                if j == 1:
                    chain[j] -= corr
                elif j == len(chain) - 1:
                    chain[j - 1] += corr
                else:
                    chain[j] -= 0.5 * corr; chain[j - 1] += 0.5 * corr
            C = chain[1:-1]
            C[:, 2] = np.maximum(C[:, 2], z_min)
            others = np.concatenate([prot[m] for m in range(N) if m != k] + [Y[~coil]])
            D = np.linalg.norm(C[:, None] - others[None], axis=-1)
            a_, b_ = np.where(D < d_min)
            for a, b in zip(a_, b_):
                d = C[a] - others[b]; C[a] = others[b] + d / max(np.linalg.norm(d), 1e-9) * d_min
            Y[idx] = C
    left = min(float(np.linalg.norm(prot[i][coil][:, None] - prot[j][None], axis=-1).min())
               for i in range(N) for j in range(N) if i != j)
    return prot, left


def build_assembly2(label, P, channel):
    """Eight whole protomers (cap, stem, rim) loose in a fluid membrane; which ring closes, and
    whether it leaks, is theirs to find."""
    from plexus.paths import graphs_data_path
    folder = os.path.join(graphs_data_path(), "shapes", "exp04_ahl2")
    info = json.load(open(os.path.join(folder, "anatomy.json")))
    pts = dict(np.load(os.path.join(folder, "points.npz")))
    bl = dict(np.load(os.path.join(folder, "blocks.npz")))
    ref = pts["c0"] * 1e9                                                  # nm, one protomer of the pore
    resid = bl["c0_resid"].ravel(); dom = bl["c0_domain"].ravel().astype(np.float32)
    stem_k = info["domains"].index("stem")
    if P.get("flexible_stem"):
        # v3: the body (cap + rim) one rigid domain (0), the stem a chain (1), seeded detached
        stem = dom == stem_k
        if P.get("rigid_upper_stem"):
            # v4 (round 7's judges): the upper half-barrel (110-122, 136-150) is part of the rigid body, as
            # 9M4A holds it ("a stable stem region except for the lower part of the TM segment", Chatterjee
            # 2026); only the tips are a chain. Flexible at 1 kT it melted in 1,339 frames and shut the lumen.
            lo_t, hi_t = P["tip_residues"]
            stem = stem & (resid >= lo_t) & (resid <= hi_t)
        dom = np.where(stem, 1.0, 0.0).astype(np.float32); stem_k = 1
        ref_pore = ref
        ref = None
    lo_r, hi_r = ANATOMY["ahl"]["rim_residues"]
    rim = ((resid >= lo_r) & (resid <= hi_r)).astype(np.float32)
    N = int(P["n_protomers"])
    rng = np.random.default_rng(int(P["seed"]))
    flex_ = bool(P.get("flexible_stem"))
    base = ref_pore if flex_ else ref
    c0 = base.mean(0)
    th0 = math.atan2(c0[1], c0[0]); r0 = math.hypot(c0[0], c0[1])

    def rotz(X, a, about=(0.0, 0.0)):
        ca, sa = math.cos(a), math.sin(a)
        Y = X.copy(); x, y = X[:, 0] - about[0], X[:, 1] - about[1]
        Y[:, 0] = about[0] + ca * x - sa * y; Y[:, 1] = about[1] + sa * x + ca * y
        return Y
    prot = []
    for k in range(N):
        a = 2 * math.pi * k / N + math.radians(rng.uniform(-P["jitter_deg"], P["jitter_deg"]))
        # each protomer's own detached coil (its own seed), in the protomer's frame before placing it.
        # `stem_start: prepore` -- the PREPORE STATE IV of Chatterjee et al. (9M4A): the upper half of
        # the barrel already in place (its residues 110-122 and 136-150 lie within 0.05 nm of the
        # pore's), only the tips (123-135) disordered: a short coil at the half-barrel's foot, above the
        # lower leaflet. `detached`: the whole stem a coil above the membrane (the arcs).
        if flex_ and P.get("stem_start") == "prepore":
            lo_t, hi_t = P["tip_residues"]
            tip = (dom == 1) & (resid >= lo_t) & (resid <= hi_t)
            own = detached_stem(ref_pore, resid, tip, z_min=P["tip_z_min_nm"], seed=int(P["seed"]) * 100 + k,
                                lift=P.get("tip_lift_nm", 0.2), out=0.0)
        elif flex_:
            own = detached_stem(ref_pore, resid, dom == 1, z_min=P["stem_z_min_nm"], seed=int(P["seed"]) * 100 + k)
        else:
            own = ref
        Y = rotz(own, a)
        u = np.array([math.cos(th0 + a), math.sin(th0 + a), 0.0])
        Y = Y + P["offset_nm"] * u
        cy = Y.mean(0)
        prot.append(rotz(Y, math.radians(rng.uniform(-P["turn_deg"], P["turn_deg"])), about=(cy[0], cy[1])))
    if flex_:
        mv = ((dom == 1) & (resid >= P["tip_residues"][0]) & (resid <= P["tip_residues"][1])
              if P.get("stem_start") == "prepore" else (dom == 1))
        prot, left = relax_coils(prot, mv, z_min=P["tip_z_min_nm"] if P.get("stem_start") == "prepore"
                                 else P["stem_z_min_nm"])
        print(f"  coils relaxed against every other bead: closest approach {left:.2f} nm")
    allp = np.concatenate(prot)
    if flex_:
        clash = min(float(np.linalg.norm(prot[i][dom == 1][:, None] - prot[j][None], axis=-1).min())
                    for i in range(N) for j in range(N) if i != j)
        print(f"  stems: closest stem bead to another protomer {clash:.2f} nm")
        if clash < 0.30:                                                  # the force cap resolves 0.3-0.45 nm in a few steps
            raise SystemExit(f"stems clash ({clash:.2f} nm < 0.30): move the coils or the ring")
    ref = prot[0]
    r_max = float(np.hypot(allp[:, 0], allp[:, 1]).max())
    R_patch = r_max + P["annulus_nm"]
    a_L = math.sqrt(2.0 * P["area_per_lipid_nm2"] / math.sqrt(3.0))
    R_frame_out = R_patch + 2 * a_L * math.sqrt(3) / 2
    z_lo = min(float(allp[:, 2].min()), -P["membrane_slab_nm"] / 2) - P["bath_nm"]
    z_hi = float(allp[:, 2].max()) + P["bath_nm"]
    L = math.ceil(max(2 * (R_frame_out + 1.0), z_hi - z_lo))
    zc = -z_lo / L + 0.5 * (1.0 - (z_hi - z_lo) / L)
    world_per_m = 1.0 / (L * 1e-9)
    nm = lambda x: x / L                                                  # noqa: E731

    # ---- the fluid bilayer (Cooke lipids, as the mechanosensitive runs) and its frozen edge -----------
    sigma_LL = a_L * P["lipid_sigma_over_a"]
    lat = hex_lattice(a_L, R_frame_out + 0.1)
    r_lat = np.hypot(lat[:, 0], lat[:, 1])
    lipid, frame = [], []
    # the half-barrel's inner radius in the upper leaflet: the upper stems' median distance from the axis
    r_in_lumen = None
    if flex_ and P.get("stem_start") == "prepore":
        up = np.concatenate([Y_[(dom == 1) & (np.abs(Y_[:, 2] - P["leaflet_z_nm"]) < 0.8)] for Y_ in prot])
        if up.size:
            r_in_lumen = float(np.median(np.hypot(up[:, 0], up[:, 1])))
    for zl in (-P["leaflet_z_nm"], P["leaflet_z_nm"]):
        near = allp[np.abs(allp[:, 2] - zl) < 1.2]
        for (x, y), r in zip(lat, r_lat):
            if r > R_patch:
                if r <= R_frame_out:
                    frame.append((x, y, zl))
                continue
            if near.shape[0] and np.sqrt(((near - np.array([x, y, zl])) ** 2).sum(1)).min() < P["lipid_excl_nm"]:
                continue
            # THE PREPORE'S LUMEN IS WATER: in state IV (9M4A) the upper half of the barrel already spans the
            # upper leaflet, and nothing lies inside it -- so no upper-leaflet lipid is seeded within the
            # half-barrel's ring. The lower leaflet is seeded whole: the tips must clear it themselves.
            if (P.get("stem_start") == "prepore" and zl > 0 and r_in_lumen is not None
                    and math.hypot(x, y) < r_in_lumen):
                continue
            lipid.append((x, y, zl))                                      # INSIDE the ring too: they must leave
    lipid = np.array(lipid); frame = np.array(frame)
    for k, Y in enumerate(prot):
        pts[f"m{k}"] = (Y * 1e-9).astype(np.float32)
    pts["lipid"] = (lipid * 1e-9).astype(np.float32); pts["frame"] = (frame * 1e-9).astype(np.float32)
    np.savez_compressed(os.path.join(folder, "points.npz"), **pts)
    dg = np.zeros(len(resid), np.float32)
    if P.get("insertion_drive"):
        sys.path.insert(0, os.path.join(REPO, "tools"))
        from channel_anatomy import read_atoms
        a_ = read_atoms(os.path.join(DATA, ANATOMY["ahl"]["cif"]), assembly=False)
        m_ = (a_["atom"] == "CA") & (a_["chain"] == info["chains"][0])
        rn = {int(r): str(n) for r, n in zip(a_["resid"][m_], a_["resname"][m_])}
        for i_, r_ in enumerate(resid):
            if dom[i_] == stem_k and int(r_) in rn:
                dg[i_] = (WW_OCT.get(rn[int(r_)], 1.15) - 1.15) / KCAL_PER_KT
        print(f"  insertion drive on the stems: sum of the side chains' transfer energies {dg.sum():.1f} kT per protomer "
              f"(most favourable {dg.min():.1f}, least {dg.max():.1f})")
    # ONE FILE PER VERSION: siblings are generated one after another into the same shape folder and read it
    # when their jobs START, so a shared file carried the LAST sibling's blocks into all three (round 7:
    # 7a's insertion drive would have run with 7c's zeros)
    blocks_name = f"assembly_blocks_{label}.npz"
    np.savez_compressed(os.path.join(folder, blocks_name),
                        **{f"m{k}_rim": rim for k in range(N)}, **{f"m{k}_domain": dom for k in range(N)},
                        **{f"m{k}_dg": dg for k in range(N)})
    n_in = int((np.hypot(lipid[:, 0], lipid[:, 1]) < r0).sum())

    # ---- units, stiffness, time step --------------------------------------------------------------
    zeta = 6 * math.pi * P["eta_Pa_s"] * P["bead_stokes_nm"] * 1e-9
    tau_s = zeta * (L * 1e-9) ** 2 / KT_J
    force_nN = KT_J / (L * 1e-9) * 1e9
    units = {"length_um": L * 1e-3, "time_s": tau_s, "force_nN": force_nN}
    go_curv = 72.0 * P["go_eps_kT"] / nm(0.5) ** 2
    wca_curv = 57.1 / nm(P["wca_sigma_nm"]) ** 2
    lj_curv = max(57.1 * P["eps_LL_kT"] / nm(sigma_LL) ** 2,
                  P["eps_LL_kT"] * math.pi ** 2 / (2.0 * nm(P["lipid_tail_over_sigma"] * sigma_LL) ** 2))
    k_stem = 169.0 * L * L if P.get("flexible_stem") else 0.0             # 1 kcal/mol/A^2 backbone springs, sim
    dt = P["dt_safety"] / (6 * max(go_curv, wca_curv, lj_curv, P["lipid_z_k_kT_nm2"] * L * L / 6, 4 * k_stem))
    f_cap = nm(0.02) / dt
    n_frames = int(P["n_frames_override"] or P["n_frames"])

    # ---- electricity (as the mechanosensitive runs) ------------------------------------------------
    volt_per_sim = (force_nN * 1e-9) * (L * 1e-9) / E_C
    psi_sim = P["psi_mV"] * 1e-3 / volt_per_sim
    cap_sim = 1e-2 * P["cap_um2"] * 1e-12 / E_C * volt_per_sim
    dx_m = L * 1e-9 / P["n_grid"]
    r_open = 0.70e-9                                                      # 9M4P's narrowest free radius
    G_open = 1.0 / (5.0e-9 / (P["sigma_S_m"] * math.pi * r_open ** 2) + 1.0 / (2 * P["sigma_S_m"] * r_open))
    I_open_pA = G_open * P["psi_mV"] * 1e-3 * 1e12
    j_open = abs(G_open * P["psi_mV"] * 1e-3) / (math.pi * r_open ** 2) * 1e12 / 1e18

    names = [f"aHL_{k + 1}" for k in range(N)]
    bead = {"pos": {"width": 3, "role": "coordinate", "integration": "first_order", "boundary": "world"},
            "rim": {"width": 1, "integration": "none", "unit": "1"},
            "domain": {"width": 1, "integration": "none", "unit": "1"},
            **({"dg": {"width": 1, "integration": "none", "unit": "energy"}} if P.get("insertion_drive") else {})}
    sets = {"cell": {"n": 1, "start": [[0.5, 0.5, round(zc, 6)]], "state": {
        "pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"},
        "psi": {"width": 1, "integration": "first_order", "unit": "voltage"},
        "j_channel": {"width": 1, "integration": "none", "unit": "current"},
        "g_channel": {"width": 1, "integration": "none", "unit": "1"},
        "pore_r": {"width": 1, "integration": "none", "unit": "length"},
        "n_largest": {"width": 1, "integration": "none", "unit": "count"},
        "n_bonds": {"width": 1, "integration": "none", "unit": "count"},
        "ring": {"width": 1, "integration": "none", "unit": "1"}}}}
    for n_ in names:
        sets[n_] = {"n": int(len(ref)), "start": [[0.5, 0.5, round(zc, 6)]], "state": dict(bead)}
    bead_l = {"pos": {"width": 3, "role": "coordinate", "integration": "first_order", "boundary": "world"}}
    sets["lipid"] = {"n": int(len(lipid)), "start": [[0.5, 0.5, round(zc, 6)]], "state": dict(bead_l)}
    sets["frame"] = {"n": int(len(frame)), "start": [[0.5, 0.5, round(zc, 6)]], "state": {
        "pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"}}}
    origin = [0.5, 0.5, round(zc, 6)]
    seeds = [{"op": "cloud_seed", "at": n_, "cloud": f"exp04_ahl2/m{k}", "origin": origin, "scale": world_per_m}
             for k, n_ in enumerate(names)]
    seeds += [{"op": "seed_state_from_file", "at": n_, "file": f"shapes/exp04_ahl2/{blocks_name}",
               "blocks": {"rim": f"m{k}_rim", "domain": f"m{k}_domain",
                          **({"dg": f"m{k}_dg"} if P.get("insertion_drive") else {})}} for k, n_ in enumerate(names)]
    seeds += [{"op": "cloud_seed", "at": "lipid", "cloud": "exp04_ahl2/lipid", "origin": origin, "scale": world_per_m},
              {"op": "cloud_seed", "at": "frame", "cloud": "exp04_ahl2/frame", "origin": origin, "scale": world_per_m}]
    lipid_pair = {"law": "cooke", "tail": nm(P["lipid_tail_over_sigma"] * sigma_LL)}
    flex = bool(P.get("flexible_stem"))
    ops = [{"op": "shape_match", "at": names[0], "sets": names, "alpha": 1.0,
            **({"domain_block": "domain", "flexible": [1]} if flex else {})},
           {"op": "membrane_potential", "at": "cell", "capacitance": cap_sim, "pump_max": 0.0, "pump_rev": 1.0,
            "leak": 0.0, "psi0": psi_sim, "h0": [0.0, 0.0], "flux": "j_channel"}]
    if flex:
        # THE STEM KNOWS ITS HAIRPIN: backbone springs to i+2 and Go wells for its tertiary pairs and its
        # contacts with its own body, all with rest lengths from the pore (9M4P), not from the coil it
        # starts as; and it cannot pass through its own body (a 0.3 nm core, under the 0.38 nm bond).
        ops += [{"op": "elastic_network", "at": names[0], "sets": names, "cutoff": nm(1.0), "k": 169.0 * L * L,
                 "local_max_sep": 2, "domain_block": "domain", "flexible_domains": [1],
                 "go_epsilon": P["stem_go_eps_kT"], "go_cutoff": nm(P["go_cutoff_nm"]),
                 "rest_from_reference": True, "reference": "exp04_ahl2", "reference_parts": ["c0"],
                 "reference_scale": world_per_m, "mobility": 1.0, "f_max": f_cap}]
        ops += [{"op": "pair_potential", "at": n_, "law": "wca", "sigma": nm(0.30), "epsilon": 1.0,
                 "exclude": "domain", "mobility": 1.0, "f_max": f_cap} for n_ in names]
    ops += [
           {"op": "elastic_network", "at": names[0], "sets": names, "cutoff": nm(0.4), "k": 0.0 if flex else 1.0,
            "go_epsilon": P["go_eps_kT"], "go_cutoff": nm(P["go_cutoff_nm"]), "go_mode": "template",
            "reference": "exp04_ahl2", "reference_parts": ["c0", "c1"], "reference_scale": world_per_m,
            "mobility": 1.0, "f_max": f_cap},
           {"op": "pair_potential", "at": names[0], "sets": names, "law": "wca", "sigma": nm(P["wca_sigma_nm"]),
            "epsilon": 1.0, "exclude_same_set": True, "mobility": 1.0, "f_max": f_cap},
           {"op": "pair_potential", "at": "lipid", **lipid_pair, "sigma": nm(sigma_LL), "epsilon": P["eps_LL_kT"],
            "mobility": 1.0, "f_max": f_cap},
           {"op": "pair_potential", "at": "lipid", "with_sets": names, "law": "lj",
            "sigma": nm(P["lipid_protein_sigma_nm"]), "epsilon": P["lipid_protein_eps_kT"],
            "mobility": 1.0, "mobility_with": 1.0, "react": True, "f_max": f_cap},
           {"op": "pair_potential", "at": "lipid", "with": "frame", **lipid_pair, "sigma": nm(sigma_LL),
            "epsilon": P["eps_LL_kT"], "mobility": 1.0, "react": False, "f_max": f_cap},
           {"op": "tether", "at": "lipid", "k": P["lipid_z_k_kT_nm2"] * L * L, "axes": [2], "mobility": 1.0},
           {"op": "brownian", "at": "lipid", "kT": 1.0, "mobility": 1.0, "seed": 1}]
    for i_, n_ in enumerate(names):
        ops.append({"op": "tether", "at": n_, "k": P["rim_k_kT_nm2"] * L * L, "axes": [2], "block": "rim", "mobility": 1.0})
        ops.append({"op": "brownian", "at": n_, "kT": 1.0, "mobility": 1.0, "seed": 20 + i_})
    if P.get("insertion_drive"):
        ops.append({"op": "residue_slab", "at": names[0], "sets": names, "z0": round(zc, 6),
                    "half_thickness": nm(P["core_half_nm"]), "edge": nm(P["core_edge_nm"]), "block": "dg", "mobility": 1.0})
    ops += [{"op": "assembly_probe", "at": names[0], "sets": names, "contact": nm(0.7), "min_contacts": 15,
             "every": 200, "cell": "cell"},
            {"op": "pore_probe", "at": names[0], "sets": names, "bead_radius": nm(P["ins_protein_nm"]),
             "z": [zc + nm(-2.5), zc + nm(3.5)], "h": nm(0.3), "slices": 25, "cell": "cell"},
            {"op": "electrolyte_conduction", "at": "cell", "field": "elec",
             "insulators": [{"set": n_, "radius": nm(P["ins_protein_nm"])} for n_ in names]
                           + [{"set": "lipid", "radius": nm(P["ins_lipid_nm"])}],
             "sigma": P["sigma_S_m"], "volt_per_sim": volt_per_sim, "dx_m": dx_m,
             "sim_current_per_A": tau_s / E_C, "slab": [zc - nm(P["membrane_slab_nm"] / 2), zc + nm(P["membrane_slab_nm"] / 2)],
             "seal": "frame", "seal_margin": nm(0.5 * a_L), "cell": "cell",
             "lining": {"block": "domain", "value": stem_k}, "axis_sets": names,
             "every": P["conduct_every"], "iters": P["conduct_iters"], "iters_first": 6000, "omega": 1.0,
             "probe_radius": nm(P["probe_nm"])}]
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from bfm_scaffold_spec import LOOKS
    # a protomer whose stem is a chain is re-contoured every frame: a skin built once and carried by its
    # beads creased into scribbles as the stem moved (crash check, 2026-09-26)
    surf_p = {"render": "surface", "spacing": round(nm(0.3), 6), "blur": 1.5, "iso_frac": 0.3, "smooth": 20,
              "specular": 0.0, "ambient": 0.32, "diffuse": 0.72, **({"recontour": True} if flex else {})}
    surf_l = {"render": "surface", "spacing": round(nm(0.4), 6), "blur": 2.0, "iso_frac": 0.3, "smooth": 30,
              "specular": 0.0, "ambient": 0.32, "diffuse": 0.72, "recontour": True,
              **({"near_side": "far"} if P.get("lipid_near_side") else {})}
    I_axis = max(10.0, 1.5 * abs(I_open_pA))
    # TWO PROTOMERS REMOVED FOR CLARITY, the two nearest the camera: the barrel inside the ring, and the
    # current in it, are then in view (as the mechanosensitive and potassium specs open theirs)
    az = math.radians(P["camera"]["azim"])
    th = [math.atan2(float(Y_[:, 1].mean()), float(Y_[:, 0].mean())) for Y_ in prot]
    order = sorted(range(N), key=lambda k: -math.cos(th[k] - az))
    hidden = [names[k] for k in order[:int(P.get("hide_front", 0))]]
    plotting = {
        "renderer": "vtk_points", "up_axis": 2, "box_frame": True, **LOOKS["cryo"],
        "render_3d": "compartments", "compartment_sets": [n_ for n_ in names if n_ not in hidden] + ["lipid"],
        "hide_sets": ["cell", "frame"] + hidden, "scale_bar_lift": 0.1,
        "surface": {**{n_: dict(surf_p) for n_ in names}, "lipid": surf_l},
        "surface_min_points": 30,
        "opacity": {**{n_: 1.0 for n_ in names}, "lipid": 0.5},
        "colors": {**{n_: CHAIN_COLORS[i_ % len(CHAIN_COLORS)] for i_, n_ in enumerate(names)}, "lipid": LIPID_COLOR},
        "field_iso": {"field": "elec", "channel": 0, "levels": [round(0.15 * j_open, 6), round(0.5 * j_open, 6)],
                      "colors": ["#ff7b1c", "#ffe27a"], "opacity": [0.28, 0.75], "smooth": 15},
        "camera": P["camera"], "zoom": P["zoom"], "near_side_centre": origin,
        "curve": [
            {"quantity": "block:cell:n_largest", "ymin": 0.0, "ymax": float(N + 1), "ylabel": "protomers in the largest assembly"},
            {"quantity": "block:cell:j_channel", "unit": "current_pA", "smooth": 20.0,
             "ymin": -I_axis if I_open_pA < 0 else -0.05 * I_axis, "ymax": 0.05 * I_axis if I_open_pA < 0 else I_axis,
             "ylabel": f"current at {P['psi_mV']:+.0f} mV"},
            {"quantity": "block:cell:pore_r", "unit": "length_nm", "factor": 2.0, "ymin": 0.0, "ymax": 3.0,
             "ylabel": "narrowest lumen diameter"}],
        "subject": names[0], "keep_stills": True, "stills": 8, "max_frames": 300,
        "real_time": False, "duration_s": 10.0, "replay_curves": False, "curve_time": {"per_frame_s": dt * tau_s, "unit": "ns"},
    }
    for cv in plotting["curve"]:
        cv.setdefault("font_size", 21); cv.setdefault("tick_font_size", 16)
    spec = {
        "general": {"name": f"exp04_v{label}", "seed": 0, "n_frames": n_frames, "dt": dt, "boundary": "wall",
                    "dim": 3, "world": [1.0, 1.0, 1.0], "record_cap": 3001,
                    "field_record_cap": 2, "units": units},
        "sets": sets, "fields": {"elec": {"frame": "grid", "res": P["n_grid"], "components": 1}},
        "seed": seeds, "operators": ops, "schedule": [o["op"] for o in ops], "plotting": plotting,
    }
    out = os.path.join(REPO, "config", "channel", f"exp04_v{label}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(_py(titled(spec)), f, sort_keys=False, default_flow_style=None, width=110)
    pred = {"box_nm": L, "protomers": N, "beads_each": int(len(ref)), "stem_beads": int((dom == stem_k).sum()),
            "lipids": int(len(lipid)), "lipids_inside_the_ring": n_in, "frame_beads": int(len(frame)),
            "heptamer_ring_radius_nm": round(r0, 2), "seeded_radius_nm": round(r0 + P["offset_nm"], 2),
            "dt_ps": dt * tau_s * 1e12, "n_frames": n_frames, "sim_time_ns": n_frames * dt * tau_s * 1e9,
            "G_open_hille_nS": G_open * 1e9, "I_open_pA": I_open_pA}
    print(f"wrote {os.path.relpath(out, REPO)}  ({VERSIONS[label]['why']})")
    for k, v in pred.items():
        print(f"  {k:36s} {v}")
    json.dump(_py(pred), open(out.replace(".yaml", ".pred.json"), "w"), indent=1)
    return spec, pred


# ================================================================================ the ion-flow test
# A MEMBRANE WITH A HOLE, NO PROTEIN: the element every channel's flow is measured with, tested alone.
# A slab of static lipid beads and the hydrocarbon core's Born barrier seal the box except a cylindrical
# hole of radius `hole_nm`; K+ and Cl- start with half of each species on each side; the voltage is
# clamped (the electrolyte's face potentials, the solve sealed outside the hole); compartment_count
# reads N_in of each species and the current carried into the cell. The same geometry's continuum
# conductance (the solver's own, and Hille's formula) is what the counted current must reproduce.
HOLE = {
    "box_nm": 10.0, "hole_nm": 1.0, "core_half_nm": 1.4, "core_edge_nm": 0.3, "pore_edge_nm": 0.2,
    "psi_mV": -100.0,                  # inside negative: K+ drawn in, Cl- pushed out
    "conc_mM": 150.0, "D_K_m2_s": 1.96e-9, "D_Cl_m2_s": 2.03e-9,     # Hille
    "r_K_nm": 0.138, "r_Cl_nm": 0.181,                                  # Pauling radii (Born energies, contacts)
    "l_B_nm": 0.70, "coulomb_cut_nm": 2.0, "eps_core": 2.0, "eps_water": 78.5,
    "area_per_lipid_nm2": 0.65, "leaflet_z_nm": 1.0, "lipid_clear_nm": 0.45, "sigma_ion_lipid_nm": 0.45,
    "ins_lipid_nm": 0.55, "n_grid": 96, "eta_Pa_s": 1.0e-3, "bead_stokes_nm": 0.3,
    "n_frames": 100000, "dt_safety": 0.3, "seed": 0, "window_frames": 20000.0,
    "camera": {"elev": 40.0, "azim": 30.0}, "zoom": 1.0,     # step 0007's view
    "n_frames_override": None,
}


def build_hole(label, P):
    L = float(P["box_nm"])
    zc = 0.5
    world_per_m = 1.0 / (L * 1e-9)
    nm = lambda x: x / L                                                  # noqa: E731
    zeta = 6 * math.pi * P["eta_Pa_s"] * P["bead_stokes_nm"] * 1e-9
    tau_s = zeta * (L * 1e-9) ** 2 / KT_J
    force_nN = KT_J / (L * 1e-9) * 1e9
    units = {"length_um": L * 1e-3, "time_s": tau_s, "force_nN": force_nN}
    mu = {"K": P["D_K_m2_s"] / (KT_J / zeta), "Cl": P["D_Cl_m2_s"] / (KT_J / zeta)}
    q = {"K": 1.0, "Cl": -1.0}
    rad = {"K": P["r_K_nm"], "Cl": P["r_Cl_nm"]}
    volt_per_sim = (force_nN * 1e-9) * (L * 1e-9) / E_C
    psi_sim = P["psi_mV"] * 1e-3 / volt_per_sim
    k_c = P["l_B_nm"] / L
    Rh = float(P["hole_nm"]); h = float(P["core_half_nm"])
    rng = np.random.default_rng(int(P["seed"]))
    # ---- the membrane: two leaflets of static lipid beads to the box's walls, less the hole ----------
    a_L = math.sqrt(2.0 * P["area_per_lipid_nm2"] / math.sqrt(3.0))
    lat = hex_lattice(a_L, 0.75 * L)
    lipid = np.array([(x, y, zl) for zl in (-P["leaflet_z_nm"], P["leaflet_z_nm"]) for (x, y) in lat
                      if abs(x) < 0.5 * L - 0.1 and abs(y) < 0.5 * L - 0.1
                      and math.hypot(x, y) > Rh + P["lipid_clear_nm"]])
    # ---- the ions: N/2 of each species in each bath, 150 mM ------------------------------------------
    zb = h + 0.5                                                          # the baths start this far off the mid-plane
    v_side = L * L * (0.5 * L - zb - 0.2)
    n_side = int(round(P["conc_mM"] * 1e-3 * 0.6022 * v_side))            # per species per side
    ions = {"K": [], "Cl": []}
    placed = []
    for side in (-1.0, 1.0):
        for sp_ in ("K", "Cl"):
            k_ = 0
            while k_ < n_side:
                p_ = np.array([rng.uniform(-0.5 * L + 0.2, 0.5 * L - 0.2), rng.uniform(-0.5 * L + 0.2, 0.5 * L - 0.2),
                               side * rng.uniform(zb, 0.5 * L - 0.2)])
                if placed and np.min(np.linalg.norm(np.array(placed) - p_, axis=1)) < 0.4:
                    continue
                placed.append(p_); ions[sp_].append(p_); k_ += 1
    # ONE SHAPE FOLDER PER VERSION: versions built one after another into a shared points.npz overwrote
    # each other's clouds (the pytest's three runs, 2026-09-26)
    shape = f"exp04_hole_v{label}"
    folder = os.path.join(__import__("plexus.paths", fromlist=["graphs_data_path"]).graphs_data_path(), "shapes", shape)
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, "points.npz"),
                        **{"lipid": (lipid * 1e-9).astype(np.float32),
                           "K": (np.array(ions["K"]) * 1e-9).astype(np.float32),
                           "Cl": (np.array(ions["Cl"]) * 1e-9).astype(np.float32)})
    # ---- time step from the stiffest contact an ion meets ---------------------------------------------
    sig = {("K", "K"): 2 * rad["K"] / 1.1225, ("Cl", "Cl"): 2 * rad["Cl"] / 1.1225,
           ("K", "Cl"): (rad["K"] + rad["Cl"]) / 1.1225}
    curv = 57.1 / nm(min(sig.values())) ** 2
    dt = P["dt_safety"] / (max(mu.values()) * curv)
    n_frames = int(P["n_frames_override"] or P["n_frames"])
    # ---- the conductance the counted current must reproduce -------------------------------------------
    c_m3 = P["conc_mM"] * 1e-3 * 6.02214e23 * 1e3
    sigma = E_C ** 2 / KT_J * c_m3 * (P["D_K_m2_s"] + P["D_Cl_m2_s"])      # the ideal ions' own, S/m
    r_m, L_m = Rh * 1e-9, 2 * h * 1e-9
    G_hille = 1.0 / (L_m / (sigma * math.pi * r_m ** 2) + 1.0 / (2 * sigma * r_m)) if Rh > 0 else 0.0
    T_run = n_frames * dt * tau_s
    n_expected = G_hille * abs(P["psi_mV"]) * 1e-3 * T_run / E_C
    W = {sp_: 56.0 / (2.0 * rad[sp_]) * (1.0 / P["eps_core"] - 1.0 / P["eps_water"]) for sp_ in rad}

    sets = {"cell": {"n": 1, "start": [[0.5, 0.5, zc]], "state": {
        "pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"},
        "psi": {"width": 1, "integration": "first_order", "unit": "voltage"},
        "j_channel": {"width": 1, "integration": "none", "unit": "current"},
        "g_channel": {"width": 1, "integration": "none", "unit": "1"},
        "nK_in": {"width": 1, "integration": "none", "unit": "count"},
        "nCl_in": {"width": 1, "integration": "none", "unit": "count"},
        "q_in": {"width": 1, "integration": "none", "unit": "count"},
        "I_in": {"width": 1, "integration": "none", "unit": "current"}}}}
    sets["lipid"] = {"n": int(len(lipid)), "start": [[0.5, 0.5, zc]], "state": {
        "pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"}}}
    for sp_ in ("K", "Cl"):
        sets[sp_] = {"n": int(len(ions[sp_])), "start": [[0.5, 0.5, zc]], "state": {
            "pos": {"width": 3, "role": "coordinate", "integration": "first_order", "boundary": "world"}}}
    origin = [0.5, 0.5, zc]
    seeds = [{"op": "cloud_seed", "at": s_, "cloud": f"{shape}/{s_}", "origin": origin, "scale": world_per_m}
             for s_ in ("lipid", "K", "Cl")]
    ops = [
        {"op": "membrane_potential", "at": "cell", "capacitance": 1e12, "pump_max": 0.0, "pump_rev": 1.0,
         "leak": 0.0, "psi0": psi_sim, "h0": [0.0, 0.0], "flux": "j_channel"},
        {"op": "electrolyte_conduction", "at": "cell", "field": "elec",
         "insulators": [{"set": "lipid", "radius": nm(P["ins_lipid_nm"])}],
         "sigma": sigma, "volt_per_sim": volt_per_sim, "dx_m": L * 1e-9 / P["n_grid"],
         "sim_current_per_A": tau_s / E_C, "slab": [zc - nm(h), zc + nm(h)], "cell": "cell",
         "lumen_radius": nm(Rh), "axis_sets": ["lipid"],
         "every": 10 ** 9, "iters": 300, "iters_first": 30000, "omega": 1.0},
        # THE IONS' PAIR FORCES IN TWO INSTANCES (the human, 2026-09-26: "why pair potential x15?"):
        # ions with ions -- Coulomb and the WCA cores summed pair by pair, each ion's charge, size and
        # mobility per set -- and ions with the lipids. H1-H3 wrote the same forces as EIGHT instances
        # (tests/test_channel_ops.py::test_merged_pair_potential_equals_the_split_one). The guards cap
        # the STEP a pair force makes in one frame: 0.05 nm for Coulomb, 0.2 nm for a core.
        {"op": "pair_potential", "at": "K", "sets": ["K", "Cl"], "law": ["coulomb", "wca"], "k_c": k_c,
         "debye": 0.0, "cutoff": nm(P["coulomb_cut_nm"]), "charge": {s_: q[s_] for s_ in ("K", "Cl")},
         "sigma": {s_: nm(2 * rad[s_] / 1.1225) for s_ in ("K", "Cl")}, "epsilon": 1.0,
         "mobility": dict(mu), "step_max": {"coulomb": nm(0.05), "wca": nm(0.2)}},
        {"op": "pair_potential", "at": "K", "sets": ["K", "Cl"], "with": "lipid", "law": "wca",
         "sigma": nm(P["sigma_ion_lipid_nm"]), "epsilon": 1.0, "mobility": dict(mu), "react": False,
         "step_max": nm(0.2)},
    ]
    for sp_ in ("K", "Cl"):
        ops += [{"op": "field_force", "at": sp_, "field": "elec", "channel": 0, "charge": q[sp_], "mobility": mu[sp_],
                 "volt_per_sim": volt_per_sim},
                {"op": "slab_barrier", "at": sp_, "height": W[sp_], "z0": zc, "half_thickness": nm(h),
                 "edge": nm(P["core_edge_nm"]), "pore_radius": nm(Rh), "pore_edge": nm(P["pore_edge_nm"]),
                 "axis_xy": [0.5, 0.5], "mobility": mu[sp_]},
                {"op": "brownian", "at": sp_, "kT": 1.0, "mobility": mu[sp_], "seed": 3 if sp_ == "K" else 4}]
    ops.append({"op": "compartment_count", "at": "cell", "z_m": zc, "cell": "cell", "sets": ["K", "Cl"],
                "charges": [1.0, -1.0], "blocks_in": ["nK_in", "nCl_in"], "block_charge": "q_in",
                "block_current": "I_in", "window": P["window_frames"]})
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from bfm_scaffold_spec import LOOKS
    plotting = {
        "renderer": "vtk_points", "up_axis": 2, "box_frame": True, **LOOKS["cryo"],
        "render_3d": "compartments", "compartment_sets": ["lipid", "K", "Cl"], "hide_sets": ["cell"],
        "surface": {"lipid": {"render": "surface", "spacing": round(nm(0.4), 6), "blur": 2.0, "iso_frac": 0.3, "smooth": 30,
                              "specular": 0.0, "ambient": 0.32, "diffuse": 0.72},
                    # THE IONS AS SMALL DOTS (the human's ask): K+ violet, Cl- green
                    "K": {"render": "dots", "point_size": 4.0}, "Cl": {"render": "dots", "point_size": 4.0}},
        "surface_min_points": 30,
        "opacity": {"lipid": 0.45, "K": 1.0, "Cl": 1.0},
        "colors": {"lipid": LIPID_COLOR, "K": [0.70, 0.53, 1.00], "Cl": [0.35, 0.85, 0.45]},
        "camera": P["camera"], "zoom": P["zoom"],
        "curve": [
            {"quantity": "block:cell:nK_in", "ymin": 0.0, "ymax": float(2 * n_side + 2), "ylabel": "K+ inside"},
            {"quantity": "block:cell:nCl_in", "ymin": 0.0, "ymax": float(2 * n_side + 2), "ylabel": "Cl- inside"},
            {"quantity": "block:cell:q_in", "ymin": -2.0, "ymax": max(10.0, 2.0 * n_expected + 5.0),
             "ylabel": f"charge moved in at {P['psi_mV']:+.0f} mV"}],
        "subject": "lipid", "keep_stills": True, "stills": 8, "max_frames": 300,
        "real_time": False, "duration_s": 10.0, "replay_curves": False,
        "curve_time": {"per_frame_s": dt * tau_s, "unit": "ns"},
    }
    for cv in plotting["curve"]:
        cv.setdefault("font_size", 21); cv.setdefault("tick_font_size", 16)
    spec = {
        "general": {"name": f"exp04_v{label}", "seed": 0, "n_frames": n_frames, "dt": dt, "boundary": "wall",
                    "dim": 3, "world": [1.0, 1.0, 1.0], "record_cap": 3001, "field_record_cap": 2, "units": units},
        "sets": sets, "fields": {"elec": {"frame": "grid", "res": P["n_grid"], "components": 2}},
        "seed": seeds, "operators": ops, "schedule": [o["op"] for o in ops], "plotting": plotting,
    }
    out = os.path.join(REPO, "config", "channel", f"exp04_v{label}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(_py(titled(spec)), f, sort_keys=False, default_flow_style=None, width=110)
    pred = {"box_nm": L, "hole_radius_nm": Rh, "core_thickness_nm": 2 * h, "ions_per_species_per_side": n_side,
            "lipids": int(len(lipid)), "sigma_ideal_S_m": round(sigma, 3), "G_hille_nS": round(G_hille * 1e9, 4),
            "psi_mV": P["psi_mV"], "dt_ps": dt * tau_s * 1e12, "n_frames": n_frames, "sim_time_ns": T_run * 1e9,
            "charges_expected_across": round(n_expected, 1), "born_K_kT": round(W["K"], 1), "born_Cl_kT": round(W["Cl"], 1)}
    print(f"wrote {os.path.relpath(out, REPO)}  ({VERSIONS[label]['why']})")
    for k, v in pred.items():
        print(f"  {k:36s} {v}")
    json.dump(_py(pred), open(out.replace(".yaml", ".pred.json"), "w"), indent=1)
    return spec, pred

# ================================================================================ THE ION-FLOW RIG (2026-09-26)
# ONE BUILDER FOR EVERY CHANNEL: a deposited state (RIG[shape], cut by channel_anatomy.py) held static in a
# sealed membrane, K+ and Cl- half on each side, the voltage clamped, the ions counted. Where an ion may go
# is the structure's own: the Born barrier of the membrane's hydrocarbon core (its thickness OPM's for this
# protein) except inside the ion paths measured HOLE-like on all atoms, and a DRY PLUG wherever a path is
# hydrophobic and narrower than liquid water can fill (Beckstein & Sansom 2003). Charged side chains and the
# carbonyls lining a path carry their charges. Nothing in it says whether the state conducts.
CHANNEL_NAME = {"mscs": "MscS", "mscl": "MscL", "kv12": "Kv1.2", "ahl": "alpha-hemolysin", "ahl2": "alpha-hemolysin",
                "kcsa": "KcsA", "gramicidin": "gramicidin A", "glic": "GLIC", "navab": "NavAb", "traak": "TRAAK",
                "ompf": "OmpF", "hole": "hole (ion-flow test)"}
FLOW = {
    "shape": None, "cation": "K",
    "psi_mV": -100.0,                  # inside negative: cations drawn in
    "conc_mM": 150.0, "D_K_m2_s": 1.96e-9, "D_Cl_m2_s": 2.03e-9, "r_K_nm": 0.138, "r_Cl_nm": 0.181,
    "l_B_nm": 0.70, "coulomb_cut_nm": 2.0, "eps_core": 2.0, "eps_water": 78.5,
    "area_per_lipid_nm2": 0.65, "head_offset_nm": 0.2, "lipid_excl_nm": 0.6, "sigma_ion_lipid_nm": 0.45,
    "ion_ca_contact_nm": 0.21,         # an ion's closest approach to an alpha carbon, beyond its own radius
    "q_site_radius_nm": 0.14,          # a charged tip's contact radius (an O or N)
    # A LUMEN CARBONYL'S CONTACT RADIUS, for batch 13: K+-O 0.25 nm at closest approach (0.138 + 0.112), not 0.28.
    # Passing a filter's O ring (its radius ~0.226 nm: sites 0.28 nm from 8 O, rings 0.33 nm apart) then costs
    # ~3 kT per ring -- the few kT simulations of filter transits report -- where 0.28 nm cost ~25 kT and not one
    # K+ crossed Kv1.2, KcsA or TRAAK in batches 10-12. Stated: 0.27-0.29 nm is the crystal SITE distance.
    "q_fil_radius_nm": 0.14,
    "ins_lipid_nm": 0.55, "ins_protein_nm": 0.30, "probe_nm": 0.133,
    "annulus_nm": 2.5, "bath_nm": 2.5, "core_edge_nm": 0.3, "pore_edge_nm": 0.1,
    # THE DRY GATE'S RADII, A FITTED CONSTANT (stated): a hydrophobic segment narrower than r_dry holds vapour,
    # wider than r_wet liquid. Beckstein & Sansom's 0.45-0.60 nm is for a pore of pure methane spheres; on the
    # deposited states here it dried the Kv1.2 and open-KcsA cavities (0.42-0.51 nm), which crystal structures
    # show full of water and K+. 0.30-0.40 nm separates every CLOSED gate of the set (0.02-0.32 nm) from the
    # open Kv1.2, KcsA, MscS and MscL paths (>= 0.42 nm) -- chosen on those labels, so it cannot be a finding
    # for them; GLIC open (4HFI, 0.33 nm at I9') and NavAb open (5VB8, 0.34 nm) fall inside it. A wall is
    # HYDROPHOBIC below 20% polar lining atoms (N, O), also fitted: every closed gate of the set is 0-20%, while
    # the K+ filters' mouths and NavAb's E177 ring, where ions do pass, are 20-31%.
    # BATCH 12: 0.25-0.33 nm. The 0.30-0.40 band closed the OPEN GLIC (0.33 nm at I9') and NavAb (0.34) as well
    # (batch 10; the judge); every closed state of the set is <= 0.21 nm once its paper's ligands are wall, every
    # open >= 0.33 -- the band sits between, chosen on the labels, so the static contrast is not a finding.
    "r_dry_nm": 0.25, "r_wet_nm": 0.33, "hydrophobic_below": 0.20, "single_file_nm": 0.03, "dry_smooth_nm": 0.2,
    "filter_eps_ratio": None, "dehydrated_below_nm": 0.2, "dehydrated_edge_nm": 0.2,
    "flex_carbonyl": True,
    "xtal_ions": False,                # batch 14: start a filter with its crystal ions (or its O-ring sites)
    "charges": True,
    "n_grid": 96, "eta_Pa_s": 1.0e-3, "bead_stokes_nm": 0.3,
    "n_frames": 300000, "dt_safety": 0.3, "seed": 0, "window_frames": 20000.0,
    "camera": {"elev": 40.0, "azim": 30.0}, "zoom": 1.0,
    "n_frames_override": None,
}


def params_of_flow(label):
    Pf = copy.deepcopy(FLOW)
    chain_, v_ = [], label
    while v_ is not None:
        chain_.append(v_); v_ = VERSIONS[v_]["parent"]
    for v_ in reversed(chain_):
        Pf.update(VERSIONS[v_]["change"])
    return Pf


def _ion_paths(info, r_ion, P):
    """Per path, the barrier's table [z, rp, R, dry, cx, cy] in nm (frame), from the anatomy's HOLE profile."""
    out = []
    for po in info["pores"]:
        pr = np.array(po["profile"], float)                     # z, R, polar fraction, enclosed, cx, cy
        z, R, pol, enc = pr[:, 0], pr[:, 1], pr[:, 2], pr[:, 3] > 0.5
        rp = np.where(enc, R - r_ion, 0.0)
        polar = pol >= P["hydrophobic_below"]
        rp = np.where(enc & polar & (rp < P["single_file_nm"]), P["single_file_nm"], np.maximum(rp, 0.0))
        # the barrier's step ends at rp and is FREE only inside rp - edge: the FULL edge is added, so the free radius
        # is R - r_ion (or the single file's). Half an edge left a filter's single-file path (0.03 + 0.05 < 0.1)
        # with no free core at all: an ion ON the axis paid ~9 kT along the whole filter (batches 12-13: not one
        # K+ through Kv1.2 or KcsA)
        rp = np.where(rp > 0, rp + P["pore_edge_nm"], 0.0)
        t = np.clip((R - P["r_dry_nm"]) / (P["r_wet_nm"] - P["r_dry_nm"]), 0.0, 1.0)
        dry = np.where(enc & ~polar, 0.5 * (1.0 + np.cos(np.pi * t)), 0.0)
        # a plug's edges as ramps, not steps: its z-gradient is the force that stops an ion
        w = max(int(round(P["dry_smooth_nm"] / max(float(z[1] - z[0]), 1e-6))), 1)
        ker = np.concatenate([np.arange(1, w + 1), np.arange(w, 0, -1)]).astype(float); ker /= ker.sum()
        pad = np.concatenate([np.zeros(w), dry, np.zeros(w)])            # no wrap-around from one end to the other
        dmax = np.max(np.stack([pad[w + k: w + k + len(dry)] for k in range(-w, w + 1)]), axis=0)
        dry_s = np.convolve(dmax, ker, mode="same")
        out.append(np.stack([z, rp, R, np.clip(dry_s, 0.0, 1.0), pr[:, 4], pr[:, 5]], 1))
    return out


def _ion_layer(P, *, label, shape, L, zc, nm, zeta, tau_s, force_nN, h, X_all, lipid, z_lo, z_hi, names,
               info_c, qpts=None, qblk=None, info_o=None, gate=None, folder_pts=None, xtal=None):
    """THE IONS OF A CHANNEL SPEC -- shared by the static rig (`build_flow`) and the moving gate (`build_mechano`
    with `ions`): the cation (K+, or Na+ for NavAb) and Cl- at `conc_mM` in both baths, their pair forces, the
    field's force, the Born barrier with the measured ion paths (the closed state's, and the open state's
    blended by `gate` when given), the charged residues (`qsite`), their brownian bath and the counter.
    Returns a dict: sets, seeds, ops, lumens (for the caller's conduction solve), dt, and the record."""
    C = P["cation"]
    D_C, r_C = {"K": (P["D_K_m2_s"], P["r_K_nm"]), "Na": (1.33e-9, 0.102)}[C]
    mu = {C: D_C / (KT_J / zeta), "Cl": P["D_Cl_m2_s"] / (KT_J / zeta)}
    q = {C: 1.0, "Cl": -1.0}
    rad = {C: r_C, "Cl": P["r_Cl_nm"]}
    k_c = P["l_B_nm"] / L
    rng = np.random.default_rng(int(P["seed"]))
    from scipy.spatial import cKDTree
    tree_p = cKDTree(X_all)
    zb = h + P["head_offset_nm"] + 0.4
    tree_l = cKDTree(lipid)
    placed, ions = [], {C: [], "Cl": []}
    for side in (-1.0, 1.0):
        zmax = (z_hi if side > 0 else -z_lo) - 0.2
        v_side = (L - 0.4) ** 2 * max(zmax - zb, 0.1)
        n_side = int(round(P["conc_mM"] * 1e-3 * 0.6022 * v_side))
        for sp_ in (C, "Cl"):
            k_, tries = 0, 0
            while k_ < n_side and tries < 400000:
                tries += 1
                p_ = np.array([rng.uniform(-0.5 * L + 0.2, 0.5 * L - 0.2), rng.uniform(-0.5 * L + 0.2, 0.5 * L - 0.2),
                               side * rng.uniform(zb, zmax)])
                if tree_p.query(p_)[0] < 0.5 or tree_l.query(p_)[0] < 0.4:
                    continue
                if placed and np.min(np.linalg.norm(np.array(placed[-4000:]) - p_, axis=1)) < 0.4:
                    continue
                placed.append(p_); ions[sp_].append(p_); k_ += 1
    # THE FILTER STARTS OCCUPIED (`xtal_ions`, the human: "a pore needs its ions"): the cations the crystal resolved on
    # the path (channel_anatomy: one per site, alternate sites), or -- where the entry models none -- the sites a
    # tetrameric filter's carbonyl-O rings define, midway between consecutive rings, every other one. Batch 12's
    # Kv1.2 pore held 0.07 K+ on average against the 2-3 a K+ filter always holds. As many Cl- are added to the
    # outer bath, so the box stays neutral.
    seeded = []
    if P.get("xtal_ions"):
        if xtal is not None and len(xtal):
            seeded = [np.asarray(x_, float) * 1e9 for x_ in xtal]
        elif qpts is not None and "qsite_backbone" in (qblk if isinstance(qblk, dict) else {}):
            O_ = np.asarray(qpts, float)[(qblk["qsite_backbone"] > 0.5) & (qblk["qsite_q"] < 0)] * 1e9
            O_ = O_[(np.hypot(O_[:, 0], O_[:, 1]) < 0.35) & (np.abs(O_[:, 2]) < h + 0.5)]
            rings = []
            for zz in sorted(O_[:, 2]):
                if rings and abs(zz - np.mean(rings[-1])) < 0.1:
                    rings[-1].append(zz)
                else:
                    rings.append([zz])
            zr = [float(np.mean(r_)) for r_ in rings if len(r_) >= 4]
            sites = [0.5 * (u_ + v_) for u_, v_ in zip(zr[:-1], zr[1:]) if abs(v_ - u_) < 0.45]
            seeded = [np.array([0.0, 0.0, z_]) for z_ in sorted(sites, reverse=True)[::2]]
    for p_ in seeded:
        ions[C].append(p_)
    for _ in seeded:                                                        # the neutralising Cl-, outside
        while True:
            p_ = np.array([rng.uniform(-0.5 * L + 0.2, 0.5 * L - 0.2), rng.uniform(-0.5 * L + 0.2, 0.5 * L - 0.2),
                           rng.uniform(zb, z_hi - 0.2)])
            if tree_p.query(p_)[0] >= 0.5 and tree_l.query(p_)[0] >= 0.4:
                ions["Cl"].append(p_); break
    n_ions = {s_: len(v) for s_, v in ions.items()}
    world_per_m = 1.0 / (L * 1e-9)
    from plexus.paths import graphs_data_path
    sfolder = os.path.join(graphs_data_path(), "shapes", shape)
    os.makedirs(sfolder, exist_ok=True)
    fp = os.path.join(sfolder, "points.npz")
    save = dict(np.load(fp)) if os.path.isfile(fp) else {}
    save.update(folder_pts or {})
    save[C] = (np.array(ions[C]) * 1e-9).astype(np.float32)
    save["Cl"] = (np.array(ions["Cl"]) * 1e-9).astype(np.float32)
    has_q = bool(P["charges"]) and qpts is not None
    qblk = qblk if isinstance(qblk, dict) else {"qsite_q": qblk}
    # THE FILTER BREATHES (`flex_carbonyl`): the backbone carbonyls that face a path in the membrane are their own
    # set, each held to its deposited place by a spring of its atoms' B-factor stiffness (<u^2> per axis =
    # B / 8 pi^2), kicked by the bath, pushed by the ions it holds. Static, a K+ filter's O rings stand 0.19-0.23 nm
    # from the axis -- inside a K+-O contact of 0.28 nm -- and no ion crossed Kv1.2, KcsA or TRAAK in batch 10.
    flex = has_q and bool(P.get("flex_carbonyl")) and "qsite_backbone" in qblk
    if has_q:
        bb = (qblk["qsite_backbone"] > 0.5) if flex else np.zeros(len(qpts), bool)
        save["qsite"] = qpts[~bb]
        bp = os.path.join(sfolder, "blocks.npz")
        bsave = dict(np.load(bp)) if os.path.isfile(bp) else {}         # keep the chains' own blocks
        bsave["qsite_q"] = qblk["qsite_q"][~bb]
        if flex:
            save["qfil"] = qpts[bb]
            bsave["qfil_q"] = qblk["qsite_q"][bb]
            B_fil = float(np.median(qblk["qsite_B"][bb])) if "qsite_B" in qblk and bb.any() else 20.0
            if not B_fil > 1.0:
                # AN NMR ENTRY HAS NO B-FACTORS (1MAG: all 0): 20 A^2, a crystal's typical backbone carbonyl, stated
                B_fil = 20.0
        np.savez_compressed(bp, **bsave)
    flex = flex and bool(bb.any())
    # gramicidin has no ionisable side chain: every charge site is a lumen carbonyl, so with the carbonyls breathing
    # the static set is EMPTY -- and a set of none cannot be seeded (batch 12's 12k/12l died on it)
    has_tips = has_q and bool((~bb).any())
    np.savez_compressed(fp, **save)
    sig_ion = {s_: 2 * rad[s_] / 1.1225 for s_ in rad}
    sig_ref = 2 * 0.138 / 1.1225                                          # the protein bead's own sigma (nm)
    curv = 57.1 / nm(min(min(sig_ion.values()), (0.138 + P["ion_ca_contact_nm"]) / 1.1225)) ** 2
    dt = P["dt_safety"] / (max(mu.values()) * curv)
    W = {sp_: 56.0 / (2.0 * rad[sp_]) * (1.0 / P["eps_core"] - 1.0 / P["eps_water"]) for sp_ in rad}

    def world_paths(info, r_ion):
        out = []
        for tab in _ion_paths(info, r_ion, P):
            rows = [[round(zc + nm(z_), 7), round(nm(rp_), 7), round(nm(R_), 7), round(float(d_), 4),
                     round(0.5 + nm(cx_), 7), round(0.5 + nm(cy_), 7)] for z_, rp_, R_, d_, cx_, cy_ in tab]
            out.append({"profile": rows})
        return out
    record = {}
    for tag, info in (("closed", info_c), ("open", info_o)):
        if info is None:
            continue
        tabs = _ion_paths(info, rad[C], P)
        record[f"dry_plug_on_path_{tag}"] = [bool(np.any(t[:, 3] > 0.5)) for t in tabs]
        record[f"narrowest_R_in_core_nm_{tag}"] = [round(float(t[np.abs(t[:, 0]) <= h][:, 2].min()), 3) for t in tabs]
        # PASSABLE END TO END: every height of the core has a free radius for the ion's centre (rp > 0) and no dry
        # plug stands anywhere on the path -- the narrowest WALL distance alone read gramicidin's separated monomers
        # (11l) as open, when the core's lower half has no path at all (the batch-11 judge)
        record[f"path_free_radius_min_nm_{tag}"] = [round(float(t[np.abs(t[:, 0]) <= h][:, 1].min()), 3) for t in tabs]
        record[f"path_passable_{tag}"] = [bool(t[np.abs(t[:, 0]) <= h][:, 1].min() > 0
                                               and not np.any(t[np.abs(t[:, 0]) <= h + 2.0][:, 3] > 0.5)) for t in tabs]
    sets = {}
    if has_tips:
        sets["qsite"] = {"n": int((~bb).sum()), "start": [[0.5, 0.5, round(zc, 6)]], "state": {
            "pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"},
            "q": {"width": 1, "integration": "none", "unit": "1"}}}
    if flex:
        sets["qfil"] = {"n": int(bb.sum()), "start": [[0.5, 0.5, round(zc, 6)]], "state": {
            "pos": {"width": 3, "role": "coordinate", "integration": "first_order", "boundary": "world"},
            "q": {"width": 1, "integration": "none", "unit": "1"}}}
    for sp_ in (C, "Cl"):
        sets[sp_] = {"n": n_ions[sp_], "start": [[0.5, 0.5, round(zc, 6)]], "state": {
            "pos": {"width": 3, "role": "coordinate", "integration": "first_order", "boundary": "world"}}}
    origin = [0.5, 0.5, round(zc, 6)]
    seeds = [{"op": "cloud_seed", "at": s_, "cloud": f"{shape}/{s_}", "origin": origin, "scale": world_per_m}
             for s_ in [C, "Cl"] + (["qsite"] if has_tips else []) + (["qfil"] if flex else [])]
    if has_tips:
        seeds.append({"op": "seed_state_from_file", "at": "qsite", "file": f"shapes/{shape}/blocks.npz",
                      "blocks": {"q": "qsite_q"}})
    if flex:
        seeds.append({"op": "seed_state_from_file", "at": "qfil", "file": f"shapes/{shape}/blocks.npz",
                      "blocks": {"q": "qfil_q"}})
    ions_ = [C, "Cl"]
    # THE DEHYDRATED STRETCH HAS ITS OWN PERMITTIVITY (`filter_eps_ratio`): where the measured lumen is narrower than
    # `dehydrated_below_nm` an ion has lost its water shell, and two charges there -- two ions in a filter, an ion and
    # the carbonyls that hold it -- interact at the protein's permittivity, not water's: the pair law's `slab`, the
    # Coulomb force multiplied by eps_water / eps_there. Without it the rig's filters were bulk water with weak
    # carbonyls (2-4 kT a site): no ion stayed in Kv1.2's filter in 47 ns, the crystal's own ions left, and a filter
    # that holds nothing has nothing to knock on (batches 13-14). Rounds 4-8's Kv1.2 builder carried this lever over
    # the whole slab; here it covers only the narrow stretch the structure measures, so a wide water-filled pore
    # (OmpF, GLIC, NavAb) keeps water's. The ratio is not measured here: stated, and swept.
    slab_d = None
    if P.get("filter_eps_ratio"):
        spans = []
        for info_ in (info_c, info_o):
            if info_ is None:
                continue
            for t in _ion_paths(info_, rad[C], P):
                m_ = (np.abs(t[:, 0]) <= h + 0.5) & (t[:, 2] < P["dehydrated_below_nm"])
                # the LONGEST contiguous narrow stretch, not the span of all of them: closed KcsA's filter and its
                # shut bundle crossing would otherwise enclose the water-filled cavity between them
                k_ = 0
                while k_ < len(m_):
                    if m_[k_]:
                        j_ = k_
                        while j_ + 1 < len(m_) and m_[j_ + 1]:
                            j_ += 1
                        spans.append((float(t[k_, 0]), float(t[j_, 0])))
                        k_ = j_
                    k_ += 1
        if spans:
            z0_, z1_ = max(spans, key=lambda ab: ab[1] - ab[0])
            # smooth edges (`dehydrated_edge_nm`): a switch paid its jump with no force -- a pump (the pair law's `slab`)
            slab_d = {"z": [round(zc + nm(z0_), 6), round(zc + nm(z1_), 6)], "eps_ratio": float(P["filter_eps_ratio"]),
                      "edge": nm(P["dehydrated_edge_nm"])}
            record["dehydrated_window_nm"] = [round(z0_, 3), round(z1_, 3)]
        record["filter_eps_ratio"] = float(P["filter_eps_ratio"])
    dslab = {"slab": slab_d} if slab_d else {}
    # the ion's own Born cost of the region, 56 / (2 a) (1 / eps_region - 1 / eps_water) kT, carried once per ion (on
    # the ion-ion law): without it the region's stronger carbonyls are a trap -- at ratio 16 Kv1.2's sites fell to
    # -24 to -42 kT with nothing to pay
    eps_f = P["eps_water"] / float(P["filter_eps_ratio"]) if slab_d else None
    born_f = {sp_: round(56.0 / (2.0 * rad[sp_]) * (1.0 / eps_f - 1.0 / P["eps_water"]), 3) for sp_ in rad} if slab_d else {}
    if slab_d:
        record["filter_born_kT"] = born_f
    dslab_ii = {"slab": {**slab_d, "born": born_f}} if slab_d else {}
    ops = [
        {"op": "pair_potential", "at": C, "sets": ions_, "law": ["coulomb", "wca"], "k_c": k_c,
         "debye": 0.0, "cutoff": nm(P["coulomb_cut_nm"]), "charge": dict(q),
         "sigma": {s_: nm(sig_ion[s_]) for s_ in ions_}, "epsilon": 1.0,
         "mobility": dict(mu), "step_max": {"coulomb": nm(0.05), "wca": nm(0.1)}, **dslab_ii},
        {"op": "pair_potential", "at": C, "sets": ions_, "with": "lipid", "law": "wca",
         "sigma": nm(P["sigma_ion_lipid_nm"]), "epsilon": 1.0, "mobility": dict(mu), "react": False,
         "step_max": nm(0.1)},
        {"op": "pair_potential", "at": C, "sets": ions_, "with_sets": names, "law": "wca",
         "sigma": {**{s_: nm(2 * (rad[s_] + P["ion_ca_contact_nm"]) / 1.1225 - sig_ref) for s_ in ions_},
                   **{n_: nm(sig_ref) for n_ in names}},
         "epsilon": 1.0, "mobility": dict(mu), "react": False, "step_max": nm(0.1)},
    ]
    if has_tips:
        ops.append({"op": "pair_potential", "at": C, "sets": ions_, "with": "qsite", "law": ["coulomb", "wca"],
                    "k_c": k_c, "debye": 0.0, "cutoff": nm(P["coulomb_cut_nm"]), "charge": dict(q),
                    "charge_with": "q",
                    "sigma": {**{s_: nm(sig_ion[s_]) for s_ in ions_}, "qsite": nm(2 * P["q_site_radius_nm"] / 1.1225)},
                    "epsilon": 1.0, "mobility": dict(mu), "react": False,
                    "step_max": {"coulomb": nm(0.05), "wca": nm(0.1)}, **dslab})
    if flex:
        k_fil = 8 * math.pi ** 2 * 100.0 / B_fil                           # kT/nm^2, from <u^2> = B / 8 pi^2 per axis
        ops += [{"op": "pair_potential", "at": C, "sets": ions_, "with": "qfil", "law": ["coulomb", "wca"],
                 "k_c": k_c, "debye": 0.0, "cutoff": nm(P["coulomb_cut_nm"]), "charge": dict(q), "charge_with": "q",
                 "sigma": {**{s_: nm(sig_ion[s_]) for s_ in ions_}, "qfil": nm(2 * P["q_fil_radius_nm"] / 1.1225)},
                 "epsilon": 1.0, "mobility": dict(mu), "react": True, "mobility_with": 1.0,
                 "step_max": {"coulomb": nm(0.05), "wca": nm(0.1)}, **dslab},
                {"op": "tether", "at": "qfil", "k": k_fil * L * L, "axes": [0, 1, 2], "mobility": 1.0},
                {"op": "brownian", "at": "qfil", "kT": 1.0, "mobility": 1.0, "seed": 6}]
    # THE IONS AS A FIELD (criterion F): the cation's density, averaged over `window` frames, drawn at 3x and 6x
    # the bath's -- where the ions crowd: the pore, its mouths, a filter's sites
    ops.append({"op": "density_field", "at": C, "field": "cden", "s": nm(0.15), "every": 20, "window": 20000.0,
                "within": [0.0, 1.0, 0.0, 1.0, zc - nm(h + 3.0), zc + nm(h + 3.0)]})
    for sp_ in ions_:
        sb = {"op": "slab_barrier", "at": sp_, "height": W[sp_], "z0": round(zc, 6), "half_thickness": nm(h),
              "edge": nm(P["core_edge_nm"]), "pore_edge": nm(P["pore_edge_nm"]), "mobility": mu[sp_],
              "pores": world_paths(info_c, rad[sp_]), "plug_margin": nm(0.2), "step_max": nm(0.1), "hard": True}
        if info_o is not None and gate:
            sb.update({"pores_open": world_paths(info_o, rad[sp_]), "gate": list(gate)})
        ops += [{"op": "field_force", "at": sp_, "field": "elec", "channel": 0, "charge": q[sp_], "mobility": mu[sp_],
                 "volt_per_sim": (force_nN * 1e-9) * (L * 1e-9) / E_C}, sb,
                {"op": "brownian", "at": sp_, "kT": 1.0, "mobility": mu[sp_], "seed": 3 if sp_ == C else 4}]
    # INSIDE MEANS BELOW THE CORE'S CYTOPLASMIC FACE, not below the mid-plane: batch 10's alpha-hemolysin
    # prepore (10g) counted 2 K+ and 1 Cl- "inside" -- both against the field -- that sat in its stems' lumen a
    # few tenths of a nm below the mid-plane, never through the membrane. An ion counts once it is through.
    ops.append({"op": "compartment_count", "at": "cell", "z_m": round(zc - nm(h), 6), "cell": "cell", "sets": ions_,
                "charges": [1.0, -1.0], "blocks_in": [f"n{C}_in", "nCl_in"], "block_charge": "q_in",
                "block_current": "I_in", "window": P["window_frames"]})
    lumens = {"lumens": world_paths(info_c, 0.0)}
    if info_o is not None and gate:
        lumens.update({"lumens_open": world_paths(info_o, 0.0), "gate": list(gate)})
    cell_blocks = {f"n{C}_in": {"width": 1, "integration": "none", "unit": "count"},
                   "nCl_in": {"width": 1, "integration": "none", "unit": "count"},
                   "q_in": {"width": 1, "integration": "none", "unit": "count"},
                   "I_in": {"width": 1, "integration": "none", "unit": "current"}}
    n_in0 = sum(1 for p_ in ions[C] if p_[2] < 0)
    col_C = {"K": [0.70, 0.53, 1.00], "Na": [1.00, 0.62, 0.25]}[C]
    bulk_density = P["conc_mM"] * 1e-3 * 602.2 * 1e-3 * L ** 3              # ions per world^3
    plot = {"sets": [C, "Cl"], "hide": (["qsite"] if has_tips else []) + (["qfil"] if flex else []),
            "field_iso": {"field": "cden", "channel": 0, "levels": [round(3 * bulk_density, 3), round(6 * bulk_density, 3)],
                          "colors": ["#9a6bff", "#e3d4ff"] if C == "K" else ["#ff9a3c", "#ffe0b0"],
                          "opacity": [0.25, 0.7], "smooth": 10},
            "surface": {C: {"render": "dots", "point_size": 4.0}, "Cl": {"render": "dots", "point_size": 4.0}},
            "opacity": {C: 1.0, "Cl": 1.0}, "colors": {C: col_C, "Cl": [0.35, 0.85, 0.45]},
            # THE PANELS (the batch-11 judge: "no current in pA, no state panel, counts without units"): the charge
            # moved into the cell, the CURRENT it makes in pA (compartment_count's windowed rate), the cations inside
            "curves": [{"quantity": "block:cell:q_in", "ymin": -10.0, "ymax": 30.0,
                        "ylabel": f"charge moved in (e) at {P['psi_mV']:+.0f} mV"},
                       {"quantity": "block:cell:I_in", "unit": "current_pA", "smooth": 20.0,
                        "ymin": -5.0 if P["psi_mV"] < 0 else -40.0, "ymax": 40.0 if P["psi_mV"] < 0 else 5.0,
                        "ylabel": "current into the cell"},
                       {"quantity": f"block:cell:n{C}_in", "ymin": 0.0, "ymax": float(2 * n_in0 + 4), "ylabel": f"{C}+ inside (ions)"}]}
    record["seeded_in_the_filter"] = [round(float(p_[2]), 2) for p_ in seeded]
    record.update({"cation": C, "ions": n_ions, "charges": (info_c.get("charges") if has_q else None),
                   "born_cation_kT": round(W[C], 1), "born_Cl_kT": round(W["Cl"], 1), "ion_paths": len(info_c["pores"]),
                   "dt_ion": dt})
    record["flexible_carbonyls"] = (int(bb.sum()), round(B_fil, 1)) if flex else None
    return {"sets": sets, "seeds": seeds, "ops": ops, "lumens": lumens, "dt": dt, "cell_blocks": cell_blocks,
            "plot": plot, "record": record, "mu": mu}


def build_flow(label, P):
    from plexus.paths import graphs_data_path
    key = P["shape"]
    rig = RIG[key]
    ch = rig["channel"]
    folder = os.path.join(graphs_data_path(), "shapes", f"exp04r_{key}")
    info = json.load(open(os.path.join(folder, "anatomy.json")))
    pts = dict(np.load(os.path.join(folder, "points.npz")))
    blk = dict(np.load(os.path.join(folder, "blocks.npz")))
    chains = sorted([k for k in pts if k.startswith("c") and k[1:].isdigit()], key=lambda k: int(k[1:]))
    X_all = np.concatenate([pts[c] for c in chains]) * 1e9
    h = float(info.get("opm", {}).get("half_thickness_nm") or 1.5)             # the hydrocarbon core, OPM's
    rho = np.hypot(X_all[:, 0], X_all[:, 1])
    r_tm = float(rho[np.abs(X_all[:, 2]) < h + 1.0].max())
    R_patch = r_tm + P["annulus_nm"]
    z_lo = min(float(X_all[:, 2].min()), -h - 1.0) - P["bath_nm"]
    z_hi = max(float(X_all[:, 2].max()), h + 1.0) + P["bath_nm"]
    L = math.ceil(max(2 * (R_patch + 0.5), z_hi - z_lo))
    zc = -z_lo / L + 0.5 * (1.0 - (z_hi - z_lo) / L)
    world_per_m = 1.0 / (L * 1e-9)
    nm = lambda x: x / L                                                  # noqa: E731
    zeta = 6 * math.pi * P["eta_Pa_s"] * P["bead_stokes_nm"] * 1e-9
    tau_s = zeta * (L * 1e-9) ** 2 / KT_J
    force_nN = KT_J / (L * 1e-9) * 1e9
    units = {"length_um": L * 1e-3, "time_s": tau_s, "force_nN": force_nN}
    volt_per_sim = (force_nN * 1e-9) * (L * 1e-9) / E_C
    psi_sim = P["psi_mV"] * 1e-3 / volt_per_sim
    from scipy.spatial import cKDTree
    tree_p = cKDTree(X_all)

    # ---- the membrane: head beads just outside OPM's hydrocarbon faces, cleared from the protein ------
    a_L = math.sqrt(2.0 * P["area_per_lipid_nm2"] / math.sqrt(3.0))
    lat = hex_lattice(a_L, 0.72 * L)
    # CLEARED BY THE PROTEIN'S OWN OUTLINE AT EACH LAYER (the 2D convex hull of its beads within 1.2 nm of the
    # layer, and 0.6 nm round every bead): the angular-sector test this replaces cut wedge-shaped holes in the
    # slab round irregular proteins (Kv1.2's sensors, NavAb, TRAAK -- batch 10's contact sheet). A third layer
    # at the mid-plane closes the drawn slab into one: ions never reach it (the core's barrier), and the
    # conduction solve's core is insulating already.
    from scipy.spatial import Delaunay
    lipid = []
    for zl in (-(h + P["head_offset_nm"]), 0.0, h + P["head_offset_nm"]):
        near = np.abs(X_all[:, 2] - zl) < 1.2
        hull = Delaunay(X_all[near][:, :2]) if near.sum() >= 4 else None
        for (x, y) in lat:
            if abs(x) > 0.5 * L - 0.15 or abs(y) > 0.5 * L - 0.15:
                continue
            if tree_p.query([x, y, zl])[0] < P["lipid_excl_nm"]:
                continue
            if hull is not None and hull.find_simplex([[x, y]])[0] >= 0:
                continue
            lipid.append((x, y, zl))
    lipid = np.array(lipid)
    shape = f"exp04r_{key}_v{label}"                                      # one folder per version
    names = [f"{CHANNEL_NAME[ch].replace(' ', '').replace('-', '').replace('.', '')[:6]}_{k + 1}" for k in range(len(chains))]
    folder_pts = {**{c: pts[c] for c in chains}, "lipid": (lipid * 1e-9).astype(np.float32)}
    IL = _ion_layer(P, label=label, shape=shape, L=L, zc=zc, nm=nm, zeta=zeta, tau_s=tau_s, force_nN=force_nN, h=h,
                    X_all=X_all, lipid=lipid, z_lo=z_lo, z_hi=z_hi, names=names, info_c=info,
                    qpts=pts.get("qsite"), qblk={k_: blk[k_] for k_ in ("qsite_q", "qsite_backbone", "qsite_B") if k_ in blk},
                    folder_pts=folder_pts, xtal=pts.get("xtal_ions"))
    dt = IL["dt"]
    n_frames = int(P["n_frames_override"] or P["n_frames"])
    T_run = n_frames * dt * tau_s
    c_m3 = P["conc_mM"] * 1e-3 * 6.02214e23 * 1e3
    D_C = {"K": P["D_K_m2_s"], "Na": 1.33e-9}[P["cation"]]
    sigma = E_C ** 2 / KT_J * c_m3 * (D_C + P["D_Cl_m2_s"])

    static = {"pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"}}
    sets = {"cell": {"n": 1, "start": [[0.5, 0.5, round(zc, 6)]], "state": {
        "pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"},
        "psi": {"width": 1, "integration": "first_order", "unit": "voltage"},
        "j_channel": {"width": 1, "integration": "none", "unit": "current"},
        "g_channel": {"width": 1, "integration": "none", "unit": "1"}, **IL["cell_blocks"]}}}
    for nmk, c in zip(names, chains):
        sets[nmk] = {"n": int(len(pts[c])), "start": [[0.5, 0.5, round(zc, 6)]], "state": dict(static)}
    sets["lipid"] = {"n": int(len(lipid)), "start": [[0.5, 0.5, round(zc, 6)]], "state": dict(static)}
    sets.update(IL["sets"])
    origin = [0.5, 0.5, round(zc, 6)]
    seeds = [{"op": "cloud_seed", "at": s_, "cloud": f"{shape}/{c}", "origin": origin, "scale": world_per_m}
             for s_, c in list(zip(names, chains)) + [("lipid", "lipid")]] + IL["seeds"]
    ops = [
        {"op": "membrane_potential", "at": "cell", "capacitance": 1e12, "pump_max": 0.0, "pump_rev": 1.0,
         "leak": 0.0, "psi0": psi_sim, "h0": [0.0, 0.0], "flux": "j_channel"},
        {"op": "electrolyte_conduction", "at": "cell", "field": "elec",
         "insulators": [{"set": s_, "radius": nm(P["ins_protein_nm"])} for s_ in names]
                       + [{"set": "lipid", "radius": nm(P["ins_lipid_nm"])}],
         "sigma": sigma, "volt_per_sim": volt_per_sim, "dx_m": L * 1e-9 / P["n_grid"],
         "sim_current_per_A": tau_s / E_C, "slab": [zc - nm(h), zc + nm(h)], "cell": "cell",
         "probe_radius": nm(P["probe_nm"]), **IL["lumens"],
         "every": 10 ** 9, "iters": 300, "iters_first": 30000, "omega": 1.0},
    ] + IL["ops"]
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from bfm_scaffold_spec import LOOKS
    surf_p = {"render": "surface", "spacing": round(nm(0.3), 6), "blur": 1.5, "iso_frac": 0.3, "smooth": 20,
              "specular": 0.0, "ambient": 0.32, "diffuse": 0.72}
    pl = IL["plot"]
    plotting = {
        "renderer": "vtk_points", "up_axis": 2, "box_frame": True, **LOOKS["cryo"],
        "render_3d": "compartments", "compartment_sets": names + ["lipid"] + pl["sets"],
        "hide_sets": ["cell"] + pl["hide"],
        "surface": {**{n_: dict(surf_p) for n_ in names},
                    # the slab's blur: 2.0 over 0.4 nm voxels closed the drawn surface over a 0.2 nm pore (the human, step
                    # 0061: "its grey part is a barrier?" -- it was not, it only looked it); `lipid_blur` / `lipid_spacing_nm`
                    "lipid": {"render": "surface", "spacing": round(nm(P.get("lipid_spacing_nm", 0.4)), 6),
                              "blur": float(P.get("lipid_blur", 2.0)), "iso_frac": 0.3, "smooth": 30,
                              "specular": 0.0, "ambient": 0.32, "diffuse": 0.72}, **pl["surface"]},
        "surface_min_points": 8,                                          # gramicidin's chains have 16 beads
        "opacity": {**{n_: 1.0 for n_ in names}, "lipid": 0.45, **pl["opacity"]},
        "colors": {**{n_: CHAIN_COLORS[i % len(CHAIN_COLORS)] for i, n_ in enumerate(names)}, "lipid": LIPID_COLOR,
                   **pl["colors"]},
        "camera": P["camera"], "zoom": P["zoom"], "curve": pl["curves"], "field_iso": pl["field_iso"],
        "subject": names[0], "keep_stills": True, "stills": 8, "max_frames": 300,
        "real_time": False, "duration_s": 10.0, "replay_curves": False,
        "curve_time": {"per_frame_s": dt * tau_s, "unit": "ns"},
    }
    for cv in plotting["curve"]:
        cv.setdefault("font_size", 21); cv.setdefault("tick_font_size", 16)
    spec = {
        "general": {"name": f"exp04_v{label}", "seed": 0, "n_frames": n_frames, "dt": dt, "boundary": "wall",
                    "dim": 3, "world": [1.0, 1.0, 1.0], "record_cap": 3001, "field_record_cap": 2, "units": units},
        "sets": sets, "fields": {"elec": {"frame": "grid", "res": P["n_grid"], "components": 2},
                                 "cden": {"frame": "grid", "res": P["n_grid"], "components": 1}},
        "seed": seeds, "operators": ops, "schedule": [o["op"] for o in ops], "plotting": plotting,
    }
    out = os.path.join(REPO, "config", "channel", f"exp04_v{label}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(_py(titled(spec)), f, sort_keys=False, default_flow_style=None, width=110)
    rec = IL["record"]
    pred = {"channel": ch, "shape": key, "state": rig["state"], "box_nm": L, "core_half_nm_opm": h,
            "ion_paths": rec["ion_paths"], "narrowest_R_in_core_nm": rec["narrowest_R_in_core_nm_closed"],
            "path_passable": rec["path_passable_closed"], "path_free_radius_min_nm": rec["path_free_radius_min_nm_closed"],
            "dry_plug_on_path": rec["dry_plug_on_path_closed"], "charges": rec["charges"], "ions": rec["ions"],
            "lipids": int(len(lipid)), "cation": rec["cation"], "psi_mV": P["psi_mV"], "conc_mM": P["conc_mM"],
            "dt_ps": dt * tau_s * 1e12, "n_frames": n_frames, "sim_time_ns": T_run * 1e9,
            "born_cation_kT": rec["born_cation_kT"], "born_Cl_kT": rec["born_Cl_kT"]}
    print(f"wrote {os.path.relpath(out, REPO)}  ({VERSIONS[label]['why']})")
    for k, v in pred.items():
        print(f"  {k:28s} {v}")
    json.dump(_py(pred), open(out.replace(".yaml", ".pred.json"), "w"), indent=1)
    return spec, pred


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__); return
    if a[0] == "--list":
        for k, v in VERSIONS.items():
            print(f"{k:5s} {v['channel']:6s} parent {str(v['parent']):5s} change {v['change']}  -- {v['why']}")
        return
    if a[0] == "anatomy":
        anatomy(a[1]); return
    label = a[0]
    if label not in VERSIONS:
        raise SystemExit(f"no version {label!r}; have {', '.join(VERSIONS)}")
    if label in RAN:
        raise SystemExit(f"version {label} has run; its config/channel/exp04_v{label}.yaml is frozen (see RAN)")
    ch = VERSIONS[label]["channel"]
    if VERSIONS[label].get("builder") == "flow":
        build_flow(label, params_of_flow(label)); return
    if VERSIONS[label].get("builder") == "mechano":
        build_mechano(label, params_of(label), ch); return
    P = params_of(label)
    if ch in ("mscs", "mscl", "traak", "selftest"):
        build_mechano(label, P, ch)
    elif ch == "kv12":
        build_ions(label, params_of_ions(label), ch)
    elif ch == "ahl":
        Pa = copy.deepcopy(ASSEMBLY)
        chain_, v_ = [], label
        while v_ is not None:
            chain_.append(v_); v_ = VERSIONS[v_]["parent"]
        for v_ in reversed(chain_):
            Pa.update(VERSIONS[v_]["change"])
        build_assembly(label, Pa, ch)
    elif VERSIONS[label].get("builder") == "flow":
        build_flow(label, params_of_flow(label))
    elif ch == "hole":
        Ph = copy.deepcopy(HOLE)
        chain_, v_ = [], label
        while v_ is not None:
            chain_.append(v_); v_ = VERSIONS[v_]["parent"]
        for v_ in reversed(chain_):
            Ph.update(VERSIONS[v_]["change"])
        build_hole(label, Ph)
    elif ch == "ahl2":
        Pa = copy.deepcopy(ASSEMBLY2)
        chain_, v_ = [], label
        while v_ is not None:
            chain_.append(v_); v_ = VERSIONS[v_]["parent"]
        for v_ in reversed(chain_):
            Pa.update(VERSIONS[v_]["change"])
        build_assembly2(label, Pa, ch)
    else:
        raise SystemExit(f"channel {ch}: generator not written yet")


if __name__ == "__main__":
    main()
# ---- the dehydrated stretch's permittivity: fixtures for tools/channel_axis_energy.py (/tmp, never a row) ------------
flow_version("E15a", "kv12", None, {**_S14, "filter_eps_ratio": 8.0, "n_frames_override": 300}, "filter eps check (/tmp, never a row)")
flow_version("E15b", "kv12", None, {**_S14, "filter_eps_ratio": 16.0, "n_frames_override": 300}, "filter eps check (/tmp, never a row)")
flow_version("E15k", "kcsa_o17", None, {**_S14, "filter_eps_ratio": 8.0, "n_frames_override": 300}, "filter eps check (/tmp, never a row)")
flow_version("E15t", "traak_o", None, {**_S14, "filter_eps_ratio": 8.0, "n_frames_override": 300}, "filter eps check (/tmp, never a row)")
flow_version("E15g", "gA", None, {**_S14, "filter_eps_ratio": 8.0, "n_frames_override": 300}, "filter eps check (/tmp, never a row)")
flow_version("E15n", "navab_o", None, {**_S14, "cation": "Na", "filter_eps_ratio": 8.0, "n_frames_override": 300}, "filter eps check (/tmp, never a row)")
flow_version("E15o", "ompf", None, {**_S14, "filter_eps_ratio": 8.0, "n_frames_override": 300}, "filter eps check (/tmp, never a row)")
# ---- BATCH 15 (2026-09-26), THE K+ FILTERS: THE DEHYDRATED STRETCH AT THE PROTEIN'S PERMITTIVITY -------------------
# Batches 10-14 passed not one K+ through Kv1.2, KcsA, TRAAK or gramicidin, and no K+ ever stood in Kv1.2's filter in
# 47 ns (13e, 14e: the crystal's own ions left). tools/channel_axis_energy.py read why: the rig treated the filter as
# bulk water -- the carbonyls held a K+ by 2-4 kT a site, less than the confinement costs, so the filter stayed empty
# and had nothing to knock on. Here the stretch whose measured lumen is narrower than 0.2 nm (the filter; all of
# gramicidin; nothing in NavAb, OmpF, GLIC) has its own permittivity for every pair of charges in it, smooth-edged
# (conservative), and each ion there pays its Born cost: ratio 8 (eps 10: Kv1.2's sites -10 to -22 kT net of a
# 16 kT cost, two neighbours repelling ~16 kT) against ratio 16 (eps 5). The ratio is not measured: stated, swept.
# The hard constraint returns an ion to its last allowed place (the flip bug of batches 13-14 fixed).
for _lab, _shape, _r, _why in (
        ("15e", "kv12", 8.0, "Kv1.2, the filter at eps 10 (ratio 8) with the ion's Born cost, 1 M, -200 mV"),
        ("15f", "kv12", 16.0, "Kv1.2, the filter at eps 5 (ratio 16) with the ion's Born cost, 1 M, -200 mV"),
        ("15i", "kcsa_o17", 8.0, "KcsA open with a conducting filter (3F7Y), the filter at eps 10, 1 M, -200 mV"),
        ("15j", "kcsa_o17", 16.0, "KcsA open with a conducting filter (3F7Y), the filter at eps 5, 1 M, -200 mV"),
        ("15k", "gA", 8.0, "gramicidin A dimer, its whole channel at eps 10, 1 M, -200 mV"),
        ("15l", "gA", 16.0, "gramicidin A dimer, its whole channel at eps 5, 1 M, -200 mV"),
        ("15q", "traak_o", 8.0, "TRAAK conductive (4WFE), the filter at eps 10, 1 M, -200 mV"),
        ("15r", "traak_o", 16.0, "TRAAK conductive (4WFE), the filter at eps 5, 1 M, -200 mV")):
    flow_version(_lab, _shape, None, {**_S14, "filter_eps_ratio": _r}, _why)
# ---- FROZEN (2026-09-26, the experiment stopped after batch 15): every label that ran is a record, never re-launched --
RAN |= {"H4", "H5", "H6"} | {f"{b}{c}" for b in (10, 11, 12, 13, 14) for c in "abcdefghijklmnopqrst"} \
    | {"15e", "15f", "15i", "15j", "15k", "15l", "15q", "15r"}
# ---- MscS ONLY, FROM 2026-09-29 (the human): the morph trainer and the rig, tested together (tools/mscs_morph.py) ----
# Records, never launched through channel_round: each is a path of the 13,671 atoms both deposited states share.
for _lab, _why in (("M1a", "MscS closed -> open by the morph trainer: a learned rest-shape rate field fitted to the open outline (grid mass)"),
                   ("M1b", "MscS closed (6PWN) -> open (2VV5), the straight path through every atom (the path morph_gate uses)")):
    VERSIONS[_lab] = {"channel": "mscs", "parent": None, "change": {}, "why": _why, "builder": "morph"}
RAN |= {"M1a", "M1b"}
VERSIONS["M2a"] = {"channel": "mscs", "parent": "M1a", "change": {}, "builder": "morph",
                   "why": "M1a through the trainer's new objective: log_mse + volume (0.5) + shrink prior 1e-4 on the rate field"}
RAN |= {"M2a"}
VERSIONS["M3a"] = {"channel": "mscs", "parent": "M2a", "change": {}, "builder": "morph",
                   "why": "M2a + the per-atom term point_mse (weight 100): each atom trained to its own open position"}
RAN |= {"M3a"}
VERSIONS["M3b"] = {"channel": "mscs", "parent": "M3a", "change": {}, "builder": "morph",
                   "why": "M3a on a finer control lattice (K 12 -> 24, nodes ~0.6 nm apart) so neighbouring regions can stretch differently"}
RAN |= {"M3b"}
VERSIONS["M3c"] = {"channel": "mscs", "parent": "M3b", "change": {}, "builder": "morph",
                   "why": "M3b + excluded volume: pair_potential[mpm] WCA between atoms (contact 0.25 nm, skipping same and adjacent residues)"}
RAN |= {"M3c"}
for _lab, _why in (("M4a", "M3c with the repulsion inside the MPM substep (recomputed every substep)"),
                   ("M4b", "M4a with the repulsion 5x stronger (eps 2e-7) and the scatter's acceleration cap 200 -> 2000"),
                   ("M4c", "M4a on a 10x stiffer material (youngs 90 -> 900): the grid-scale resistance to compression"),
                   ("M4d", "M3b (no repulsion) with a spatial smooth prior (1e-4) on the rate lattice"),
                   ("M4e", "M4a with a finer last grid (64 -> 96, 0.19 nm cells): does resolution let the repulsion act")):
    VERSIONS[_lab] = {"channel": "mscs", "parent": "M3c", "change": {}, "builder": "morph", "why": _why}
RAN |= {"M4a", "M4b", "M4c", "M4d", "M4e"}
VERSIONS["M5a"] = {"channel": "mscs", "parent": "M4e", "change": {}, "builder": "morph",
                   "why": "M4e done properly: grid 96 in the last stage with the substep halved (0.00034 -> 0.00017), repulsion inside the substep"}
VERSIONS["M5b"] = {"channel": "mscs", "parent": "M3b", "change": {}, "builder": "morph",
                   "why": "the hybrid: M3b's MPM path, each frame's atoms relaxed at the atom scale (bonds at closed lengths, WCA 0.25 nm, tether to the MPM frame)"}
RAN |= {"M5a", "M5b"}
VERSIONS["M5c"] = {"channel": "mscs", "parent": "M5b", "change": {}, "builder": "morph",
                   "why": "M5b relaxed harder: 1000 steps a frame, tether to the MPM frame 1 -> 0.1, WCA 1 -> 5"}
RAN |= {"M5c"}
# ---- PHASE G (2026-09-29): MscS opens BECAUSE the membrane is stretched -- the gate on the recorded hybrid path M5c
# (morph_gate[waypoints]), NO stated offset (dV 0) and NO stated tension-work term: what the lipids and the ions push
# along the path is the whole energy. Three stretch-release cycles to 12 mN/m, 1000 movie frames (the human).
_G3 = {**_M13, "rig_shape": "mscs_c", "rig_open": "mscs_o", "dV_kT": 0.0,
       "morph_waypoints": {"shape": "exp04r_mscs_path", "n": 21}, "tension_cycles": 3, "movie_frames": 1000}
mech_version("G3a", "mscs", None, dict(_G3), "MscS on the recorded path M5c, no offset, 3 stretch-release cycles to 12 mN/m")
mech_version("G3b", "mscs", "G3a", {"tension_max_mN_m": 0.0}, "G3a with no tension (the control: must stay shut)")
mech_version("G3c", "mscs", None, {**_G3, "lipid_protein_eps_kT": 2.0}, "G3a with the lipid-protein adhesion 2 kT")
mech_version("G3d", "mscs", "G3c", {"tension_max_mN_m": 0.0}, "G3c with no tension (the control)")
mech_version("G3t", "mscs", "G3a", {"n_frames_override": 600, "tension_cycles": 1}, "G3a smoke test (/tmp, never a row)")
# ---- PHASE G1 (2026-09-29, revision 1): THE LIPIDS' WORK ON THE GATE, MEASURED BY CLAMPING IT -----------------------
# Batch 13's free gate could not cross in a run (a net 10 kT moves lambda 0.02 in the hold; ~230 ns to cross), so the
# test is thermodynamic: lambda HELD at 0, 0.25, 0.5, 0.75, 1 on the M5c path (morph_gate[clamp]) and the push along the
# path averaged over a long hold, at zero tension (a-e: the patch pre-stretched to its zero-tension size, linear
# strain 0.040 from batch 12's calibration tension = -4.7 + 57 x area strain) and at 8 mN/m (f-j: linear 0.106, area
# 0.223). No ions (they do not see the tension; the conduction solve runs once), the protein's Brownian kicks off
# (it is held on the path; they would only add noise to the force). tools/gate_pmf.py integrates G(lambda) per tension.
_G1 = {**_G3, "ions": False, "tension_cycles": 1, "movie_frames": 300, "protein_brownian": False,
       "conduct_every": 50000, "rest_strain_linear": 0.040, "hold_frames": [20000, 60000, 220000]}
for _i, _lam in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
    mech_version(f"G1{'abcde'[_i]}", "mscs", None, {**_G1, "morph_clamp": _lam, "strain_top_linear": 0.040},
                 f"MscS clamped at lambda {_lam} on the M5c path, the patch at zero tension: the push along the path")
    mech_version(f"G1{'fghij'[_i]}", "mscs", None, {**_G1, "morph_clamp": _lam, "strain_top_linear": 0.106},
                 f"MscS clamped at lambda {_lam} on the M5c path, the patch stretched to 8 mN/m: the push along the path")
mech_version("G1s", "mscs", "G1j", {"hold_frames": [200, 200, 600]}, "G1j smoke test (/tmp, never a row)")
# ---- G1' = T0a-i / T8a-i (2026-09-29): G1 at CONSTANT TENSION (the frame's set-point mode, 5x faster: its response ~8k frames against the
# 2,000-frame running mean it reads), nine lambdas 0.125 apart (G1's push jumped between neighbours), 0 and 8 mN/m.
_G1p = {**_G1, "drive_mode": "tension", "mobility_R_factor": 5.0, "rest_tension_mN_m": 0.0, "strain_top_linear": 0.0}
for _i in range(9):
    _lam = _i / 8.0
    mech_version(f"T0{'abcdefghi'[_i]}", "mscs", None, {**_G1p, "morph_clamp": _lam, "hold_tension_mN_m": 0.0},
                 f"MscS clamped at lambda {_lam} on the M5c path, the membrane HELD at 0 mN/m (set-point): the push along the path")
    mech_version(f"T8{'abcdefghi'[_i]}", "mscs", None, {**_G1p, "morph_clamp": _lam, "hold_tension_mN_m": 8.0},
                 f"MscS clamped at lambda {_lam} on the M5c path, the membrane HELD at 8 mN/m (set-point): the push along the path")
mech_version("T8t", "mscs", "T8i", {"hold_frames": [5000, 5000, 20000]}, "T8i tension-mode check (/tmp, never a row)")
VERSIONS["G2a"] = {"channel": "mscs", "parent": None, "change": {}, "builder": "analysis",
                   "why": "G2 on G1a-j (tools/gate_pmf.py): G(lambda) at the two frame strains, the lipids' work W, dA_eff, tau_1/2"}
RAN |= {f"G1{c}" for c in "abcdefghij"} | {"G2a"}
VERSIONS["G2b"] = {"channel": "mscs", "parent": "G2a", "change": {}, "builder": "analysis",
                   "why": "G2 on T0a-i / T8a-i (tools/gate_pmf.py): G(lambda) at 0 and 8 mN/m HELD, the protein's own share, the lipids' work W(lambda)"}
RAN |= {f"T{t_}{c}" for t_ in "08" for c in "abcdefghi"} | {"G2b"}
# ---- G2c's batch (2026-09-29, night): IS THE STRETCH'S WORK LINEAR IN TENSION (the force-from-lipids signature, W = tau
# dA_eff), and does W survive a second noise history? T4a-i: held at 4 mN/m; T9a-i: 8 mN/m again, the lipids' noise
# seeded differently (noise_replicate 1). Same nine lambdas as T0/T8.
for _i in range(9):
    _lam = _i / 8.0
    mech_version(f"T4{'abcdefghi'[_i]}", "mscs", None, {**_G1p, "morph_clamp": _lam, "hold_tension_mN_m": 4.0},
                 f"MscS clamped at lambda {_lam} on the M5c path, the membrane HELD at 4 mN/m: the push along the path")
    mech_version(f"T9{'abcdefghi'[_i]}", "mscs", None, {**_G1p, "morph_clamp": _lam, "hold_tension_mN_m": 8.0, "noise_replicate": 1},
                 f"T8{'abcdefghi'[_i]} again (lambda {_lam}, 8 mN/m) with the lipids' noise seeded differently: the replicate")
# ---- G1'' (2026-09-29, night): THE CLAMP ON THE STRAIGHT PATH (6PWN -> 2VV5, as a one-waypoint path) -- at the rig's
# C-alpha level it is sterically clean (inter-chain WCA 24-30 kT all along, closest pair 0.435 nm), where M5c climbs to
# 263 kT of the protein's own clashes; and it ends ON the open structure (path gate). S0a-i: 0 mN/m; S8a-i: 8 mN/m.
_S = {**_G1p, "morph_waypoints": {"shape": "exp04r_mscs_straight", "n": 2}}
for _i in range(9):
    _lam = _i / 8.0
    mech_version(f"S0{'abcdefghi'[_i]}", "mscs", None, {**_S, "morph_clamp": _lam, "hold_tension_mN_m": 0.0},
                 f"MscS clamped at lambda {_lam} on the STRAIGHT path, the membrane HELD at 0 mN/m: the push along the path")
    mech_version(f"S8{'abcdefghi'[_i]}", "mscs", None, {**_S, "morph_clamp": _lam, "hold_tension_mN_m": 8.0},
                 f"MscS clamped at lambda {_lam} on the STRAIGHT path, the membrane HELD at 8 mN/m: the push along the path")
mech_version("S8t", "mscs", "S8i", {"hold_frames": [200, 200, 600]}, "S8i smoke test (/tmp, never a row)")
VERSIONS["G2c"] = {"channel": "mscs", "parent": "G2b", "change": {}, "builder": "analysis",
                   "why": "G2 on T0 against T8, T9 (8 mN/m, another noise) and T4 (4 mN/m): is W linear in tension; the joint dA_eff and tau_1/2"}
RAN |= {f"T{t_}{c}" for t_ in "49" for c in "abcdefghi"} | {"G2c"}
VERSIONS["G2d"] = {"channel": "mscs", "parent": "G2b", "change": {}, "builder": "analysis",
                   "why": "G2 on the STRAIGHT path, S0a-i against S8a-i (tools/gate_pmf.py): W, dA_eff, tau_1/2, the protein's own share"}
RAN |= {f"S{t_}{c}" for t_ in "08" for c in "abcdefghi"} | {"G2d"}


# ================================================================================ B: a bilayer WITH A CORE (Phase G)
# 2026-09-30: the rig's lipids are one bead per leaflet at the head planes, so tension grips the protein only there
# (G2d: the straight path, which widens in the core and not at the upper head plane, received W = -1.4 +- 3.3 kT). The
# membrane below is Cooke, Kremer & Deserno's (2005, PRE 72:011506) three-bead lipid: a head and two tail beads,
# heads WCA at b = 0.95 sigma against every bead, tails WCA at sigma plus the cos^2 tail (depth eps, width 1.6 sigma)
# among themselves. Bonds and the head-to-tail-end stiffness are harmonic springs inside each lipid (elastic_network
# `within: mol`, Cooke's FENE + bending replaced by springs at the seeded lengths -- stated), the non-bonded table
# skips pairs inside a lipid (`exclude: mol`). No depth spring: the membrane can thin. The rim: frozen lipids of the
# same kind, read by the pipette (radial_drive) as the rig's frame.
BILAYER = {
    "sigma_nm": 0.8,                  # Cooke's sigma: the core (4 tail beads) ~3.2-3.4 nm, MscS closed 3.38 (OPM 6PWN)
    "area_per_lipid_sigma2": 1.2,     # the seeded area per lipid per leaflet, sigma^2 (Cooke 2005's fluid bilayers: ~1.2)
    "eps_tail_kT": 0.91,              # kT / eps = 1.1: inside Cooke's fluid window at w_c = 1.6 sigma
    "tail_width_sigma": 1.6,
    "head_b": 0.95,                   # the head's WCA size, in sigma
    "bond_sigma": 1.0,                # seeded bond length, sigma (springs rest there)
    "k_bond_kT_sigma2": 30.0,         # Cooke's FENE stiffness scale; one k for bonds and the head-tail-end spring
    "R_patch_nm": 9.0, "frame_rows": 2, "bath_nm": 3.0,
    "eta_Pa_s": 1.0e-3, "bead_stokes_nm": 0.3, "kT_noise": 1.0, "dt_safety": 0.3,
    "tension_mN_m": 0.0,              # the pipette's set-point over the hold
    "hold_frames": [20000, 40000, 240000],
    "mobility_R_factor": 5.0,
    "movie_frames": 300,
}


def build_bilayer(label, P):
    from plexus.paths import graphs_data_path
    s = P["sigma_nm"]
    b = P["bond_sigma"] * s
    A = P["area_per_lipid_sigma2"] * s * s
    a_hex = math.sqrt(2.0 * A / math.sqrt(3.0))
    R, Rf = P["R_patch_nm"], P["R_patch_nm"] + P["frame_rows"] * a_hex * math.sqrt(3) / 2
    z_head = 0.5 * s + 2 * b                                   # tail ends 0.5 sigma off the mid-plane
    lat = hex_lattice(a_hex, Rf + 0.1)
    r_lat = np.hypot(lat[:, 0], lat[:, 1])
    mols, rims = [], []
    for sgn in (1.0, -1.0):
        for (x, y), r in zip(lat, r_lat):
            m = [(x, y, sgn * z_head), (x, y, sgn * (z_head - b)), (x, y, sgn * (z_head - 2 * b))]
            (mols if r <= R else rims if r <= Rf else []).append(m)
    mols, rims = np.array(mols), np.array(rims)                   # [n, 3 beads, 3]
    L = math.ceil(max(2 * (Rf + 1.5), 2 * z_head + 2 * P["bath_nm"]))
    zc = 0.5
    world_per_m = 1.0 / (L * 1e-9)
    nm = lambda x: x / L                                              # noqa: E731
    zeta = 6 * math.pi * P["eta_Pa_s"] * P["bead_stokes_nm"] * 1e-9
    tau_s = zeta * (L * 1e-9) ** 2 / KT_J
    force_nN = KT_J / (L * 1e-9) * 1e9
    tension_sim = lambda mN_m: mN_m * 1e-3 * (L * 1e-9) ** 2 / KT_J     # noqa: E731
    # beads ordered HEADS FIRST, then the tail pairs (`type_layout: ordered` gives the first n the first type)
    def order(M):
        return np.concatenate([M[:, 0], M[:, 1:].reshape(-1, 3)]), np.concatenate(
            [np.arange(len(M)), np.repeat(np.arange(len(M)), 2)])
    Xl, mol = order(mols); Xf, _ = order(rims)
    shp = f"exp04b_{label}"
    folder = os.path.join(graphs_data_path(), "shapes", shp)
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, "points.npz"), lipid=(Xl * 1e-9).astype(np.float64),
                        frame=(Xf * 1e-9).astype(np.float64))
    np.savez_compressed(os.path.join(folder, "blocks.npz"), mol=mol.astype(np.float64)[:, None])
    k_sim = P["k_bond_kT_sigma2"] / (s * s) * L * L
    lam = max(4.0 * k_sim, 6 * 57.1 / nm(P["head_b"] * s) ** 2)
    dt = P["dt_safety"] / lam
    f_cap = nm(0.02) / dt
    sw = nm(s); hb = nm(P["head_b"] * s); wt = nm(P["tail_width_sigma"] * s)
    table = {"head": {"head": {"law": "wca", "sigma": hb}, "tail": {"law": "wca", "sigma": hb}},
             "tail": {"head": {"law": "wca", "sigma": hb},
                      "tail": {"law": "cooke", "sigma": sw, "epsilon": P["eps_tail_kT"], "tail": wt}}}
    fs = [int(x) for x in P["hold_frames"]]
    T = tension_sim(P["tension_mN_m"])
    prot = [[0, 0.0], [fs[0], 0.0], [fs[0] + fs[1], T], [sum(fs), T]]
    KA_guess = 200.0 * 1e-3 * (L * 1e-9) ** 2 / KT_J                 # sim; only the pipette's response time
    tau_patch = 2.0 / nm(A) * nm(R) ** 2 / KA_guess
    mu_R = 1.0 / (4 * math.pi * KA_guess * 0.2 * tau_patch) * P["mobility_R_factor"]
    origin = [0.5, 0.5, zc]
    nL, nF = len(mols), len(rims)
    sets = {"cell": {"n": 1, "start": [[0.5, 0.5, zc]], "state": {
                "pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"},
                "tension": {"width": 1, "integration": "none", "unit": "tension"},
                "tension_applied": {"width": 1, "integration": "none", "unit": "tension"},
                "strain": {"width": 1, "integration": "none", "unit": "1"}}},
            "lipid": {"n": int(3 * nL), "start": [origin], "type_layout": "ordered",
                      "types": {"head": {"count": int(nL)}, "tail": {"count": int(2 * nL)}},
                      "state": {"pos": {"width": 3, "role": "coordinate", "integration": "first_order", "boundary": "world"},
                                "mol": {"width": 1, "integration": "none", "unit": "1"}}},
            "frame": {"n": int(3 * nF), "start": [origin], "type_layout": "ordered",
                      "types": {"head": {"count": int(nF)}, "tail": {"count": int(2 * nF)}},
                      "state": {"pos": {"width": 3, "role": "coordinate", "integration": "none", "boundary": "world"},
                                "force": {"width": 3, "integration": "none", "unit": "force"}}}}
    seeds = [{"op": "cloud_seed", "at": "lipid", "cloud": f"{shp}/lipid", "origin": origin, "scale": world_per_m},
             {"op": "cloud_seed", "at": "frame", "cloud": f"{shp}/frame", "origin": origin, "scale": world_per_m},
             {"op": "seed_state_from_file", "at": "lipid", "file": f"shapes/{shp}/blocks.npz", "blocks": {"mol": "mol"}}]
    ops = [
        {"op": "elastic_network", "at": "lipid", "within": "mol", "cutoff": nm(2.5 * b), "k": k_sim, "go_epsilon": 0.0,
         "mobility": 1.0, "f_max": f_cap},
        {"op": "pair_potential", "model": "table", "at": "lipid", "pair_table": table, "exclude": "mol", "mobility": 1.0,
         "f_max": f_cap},
        {"op": "pair_potential", "model": "table", "at": "lipid", "with": "frame", "pair_table": table, "mobility": 1.0,
         "react": False, "f_max": f_cap},
        {"op": "brownian", "at": "lipid", "kT": P["kT_noise"], "mobility": 1.0, "seed": 1},
        {"op": "radial_drive", "at": "frame", "protocol": prot, "mode": "tension", "mobility_R": mu_R, "cell": "cell",
         "axis": 2, "centre": origin, "max_step": nm(0.01 * a_hex), "avg_frames": 2000.0, "affine_sets": ["lipid"]},
    ]
    from bfm_scaffold_spec import LOOKS
    plotting = {"renderer": "vtk_points", "up_axis": 2, "box_frame": True, **LOOKS["cryo"],
                "render_3d": "compartments", "compartment_sets": ["lipid"], "hide_sets": ["frame", "cell"],
                "surface": {"lipid": {"render": "surface", "spacing": round(nm(0.4), 6), "blur": 2.0, "iso_frac": 0.3,
                                      "smooth": 30, "specular": 0.0, "ambient": 0.32, "diffuse": 0.72, "recontour": True}},
                "opacity": {"lipid": 0.6}, "colors": {"lipid": LIPID_COLOR},
                "camera": {"elev": 20.0, "azim": 30.0}, "zoom": 1.0,
                "curve": [{"quantity": "block:cell:tension", "unit": "tension_mN_m", "smooth": 50.0, "ymin": -5.0,
                           "ymax": 15.0, "ylabel": "membrane tension", "font_size": 21, "tick_font_size": 16},
                          {"quantity": "block:cell:strain", "smooth": 50.0, "ymin": -0.12, "ymax": 0.2, "ylabel": "patch strain",
                           "font_size": 21, "tick_font_size": 16}],
                "subject": "lipid", "keep_stills": True, "stills": 8, "max_frames": int(P["movie_frames"]),
                "real_time": False, "duration_s": 10.0, "curve_time": {"per_frame_s": dt * tau_s, "unit": "ns"},
                "replay_curves": False}
    spec = {"general": {"name": f"exp04_v{label}", "seed": 0, "n_frames": int(sum(fs)), "dt": dt, "boundary": "wall",
                        "dim": 3, "world": [1.0, 1.0, 1.0], "record_cap": 3001, "field_record_cap": 2,
                        "units": {"length_um": L * 1e-3, "time_s": tau_s, "force_nN": force_nN}},
            "sets": sets, "fields": {}, "seed": seeds, "operators": ops, "schedule": [o["op"] for o in ops],
            "plotting": plotting}
    out = os.path.join(REPO, "config", "channel", f"exp04_v{label}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(_py(titled(spec)), f, sort_keys=False, default_flow_style=None, width=110)
    pred = {"box_nm": L, "R_patch_nm": R, "lipids": nL, "rim_lipids": nF, "beads": 3 * nL, "sigma_nm": s, "area_per_lipid_nm2": A,
            "z_head_nm": z_head, "core_seeded_nm": 2 * (z_head - b) + 0.0, "dt_ps": dt * tau_s * 1e12,
            "sim_time_ns": sum(fs) * dt * tau_s * 1e9, "tension_mN_m": P["tension_mN_m"], "eps_tail_kT": P["eps_tail_kT"]}
    json.dump(pred, open(out.replace(".yaml", ".pred.json"), "w"), indent=1, default=float)
    print(f"wrote {os.path.relpath(out, REPO)} ({VERSIONS[label]['why']})")
    for k_, v_ in pred.items():
        print(f"  {k_:24s} {v_}")
    return spec, pred


def bil_version(label, change, why):
    VERSIONS[label] = {"channel": "bilayer", "parent": None, "change": dict(change), "builder": "bilayer", "why": why}


def params_of_bilayer(label):
    return {**BILAYER, **VERSIONS[label]["change"]}


bil_version("B2t", {"hold_frames": [2000, 2000, 6000], "tension_mN_m": 8.0}, "Cooke bilayer smoke test (/tmp, never a row)")
# B2 (2026-09-30): the protein-free Cooke patch -- held at 0, 4, 8, 12 mN/m (the pipette's set-point): its rest area
# per lipid, K_A (Rawicz 2000: 230-265 mN/m), thickness and thinning under tension, and whether it stays a fluid.
# Two tail depths: kT/eps 1.1 (a-d) and 1.0 (e-h), both inside Cooke's fluid window at w_c = 1.6 sigma.
for _i, _tau in enumerate((0.0, 4.0, 8.0, 12.0)):
    bil_version(f"B2{'abcd'[_i]}", {"tension_mN_m": _tau}, f"Cooke bilayer (tail depth 0.91 kT), no protein, held at {_tau:g} mN/m")
    bil_version(f"B2{'efgh'[_i]}", {"tension_mN_m": _tau, "eps_tail_kT": 1.0}, f"Cooke bilayer (tail depth 1.0 kT), no protein, held at {_tau:g} mN/m")
RAN |= set()
# B3 smoke: MscS clamped on the straight path in the Cooke membrane
_B3 = {**_S, "lipid_model": "cooke3", "lipid_protein_eps_kT": 1.0}
mech_version("B3t", "mscs", None, {**_B3, "morph_clamp": 1.0, "hold_tension_mN_m": 8.0, "hold_frames": [300, 300, 1400]},
             "MscS in the Cooke membrane, clamped open on the straight path, 8 mN/m: smoke test (/tmp, never a row)")
VERSIONS["B2r"] = {"channel": "bilayer", "parent": None, "change": {}, "builder": "analysis",
                   "why": "B2's ruler (tools/bilayer_ruler.py): rest area per lipid, K_A, thickness and thinning, lateral diffusion of the Cooke patch"}
RAN |= {f"B2{c}" for c in "abcdefgh"} | {"B2r"}
# ---- B3 (2026-09-30, 01:20): MscS IN THE MEMBRANE WITH A CORE, clamped on the STRAIGHT path, 0 and 8 mN/m -- does the
# stretch's work W appear once the tails can pull on the protein across the core (G2d: -1.4 +- 3.3 kT with one-bead
# lipids)? The membrane is B2's tail depth 0.91 kT, seeded at its measured rest area (0.671 nm^2 = 1.05 sigma^2).
_B3c = {**_S, "lipid_model": "cooke3", "lipid_protein_eps_kT": 1.0, "cooke3": {"area_per_lipid_sigma2": 1.05}}
for _i in range(9):
    _lam = _i / 8.0
    mech_version(f"C0{'abcdefghi'[_i]}", "mscs", None, {**_B3c, "morph_clamp": _lam, "hold_tension_mN_m": 0.0},
                 f"MscS in the Cooke membrane (a core), clamped at lambda {_lam} on the straight path, HELD at 0 mN/m")
    mech_version(f"C8{'abcdefghi'[_i]}", "mscs", None, {**_B3c, "morph_clamp": _lam, "hold_tension_mN_m": 8.0},
                 f"MscS in the Cooke membrane (a core), clamped at lambda {_lam} on the straight path, HELD at 8 mN/m")
VERSIONS["G2e"] = {"channel": "mscs", "parent": "G2d", "change": {}, "builder": "analysis",
                   "why": "G2 on the straight path IN THE COOKE MEMBRANE, C0a-i against C8a-i (tools/gate_pmf.py): does W appear once the core can pull"}
RAN |= {f"C{t_}{c}" for t_ in "08" for c in "abcdefghi"} | {"G2e"}
# ---- B3' (2026-09-30, ~03:00): B3 was an artifact -- the tails walked into the C-alpha model's gaps (C0a: 28 lipid beads in
# the pore, 101 among the helices, against 0 and 9 with the one-bead lipid) and their 1 kT pull, summed over ~500
# touching beads, made pushes of thousands of kT. Here a residue has its side chains' size (lipid beads kept >= 1.0 nm
# from an alpha carbon: WCA sigma 0.89 nm, seeding exclusion 1.0 nm) and NO stickiness (heads and tails both WCA): the
# clean force-from-lipid test, tension transmitted through excluded volume alone.
_B3r = {**_B3c, "lipid_protein_repulsive": True, "lipid_protein_sigma_nm": 0.89, "lipid_excl_nm": 1.0}
for _i in range(9):
    _lam = _i / 8.0
    mech_version(f"D0{'abcdefghi'[_i]}", "mscs", None, {**_B3r, "morph_clamp": _lam, "hold_tension_mN_m": 0.0},
                 f"MscS in the Cooke membrane, residues sized and not sticky, clamped at lambda {_lam} (straight path), 0 mN/m")
    mech_version(f"D8{'abcdefghi'[_i]}", "mscs", None, {**_B3r, "morph_clamp": _lam, "hold_tension_mN_m": 8.0},
                 f"MscS in the Cooke membrane, residues sized and not sticky, clamped at lambda {_lam} (straight path), 8 mN/m")
mech_version("D0t", "mscs", "D0a", {"hold_frames": [2000, 2000, 16000]}, "D0a intrusion check (/tmp, never a row)")
VERSIONS["G2f"] = {"channel": "mscs", "parent": "G2e", "change": {}, "builder": "analysis",
                   "why": "G2 on the straight path in the Cooke membrane with sized, non-sticky residues, D0a-i against D8a-i (tools/gate_pmf.py)"}
RAN |= {f"D{t_}{c}" for t_ in "08" for c in "abcdefghi"} | {"G2f"}
# ---- B3'' (2026-09-30, 04:00): G2f's membrane (sized, non-sticky residues) at 4 and 12 mN/m -- is W linear in tension,
# and does the fully emergent landscape (no stated energy: +40.8 kT at rest, -28.0 kT of stretch work at 7.95 mN/m)
# really tip open near its predicted 11.6 mN/m? B2d held 12 mN/m without tearing.
for _i in range(9):
    _lam = _i / 8.0
    mech_version(f"D4{'abcdefghi'[_i]}", "mscs", None, {**_B3r, "morph_clamp": _lam, "hold_tension_mN_m": 4.0},
                 f"MscS in the Cooke membrane, residues sized and not sticky, clamped at lambda {_lam} (straight path), 4 mN/m")
    mech_version(f"D12{'abcdefghi'[_i]}", "mscs", None, {**_B3r, "morph_clamp": _lam, "hold_tension_mN_m": 12.0},
                 f"MscS in the Cooke membrane, residues sized and not sticky, clamped at lambda {_lam} (straight path), 12 mN/m")
VERSIONS["G2g"] = {"channel": "mscs", "parent": "G2f", "change": {}, "builder": "analysis",
                   "why": "G2 on D0 against D4, D8, D12 (tools/gate_pmf.py --more): W against tension, and where the emergent landscape tips open"}
RAN |= {"G2g"}
RAN |= {f"D{t_}{c}" for t_ in ("4", "12") for c in "abcdefghi"}
# ---- wetting checks (2026-09-30, 05:05): D's non-sticky residues DEWET -- 5-8 lipid beads within 1.3 nm of a TM C-alpha at
# rest, 0 at 12 mN/m: the Cooke tails leave a void around a purely repulsive protein. Sized residues (1.0 nm) WITH a tail
# attraction (the hydrophobic belt): does it wet without intruding?
for _e, _l in ((0.5, "W5t"), (1.0, "W1t")):
    mech_version(_l, "mscs", None, {**_B3c, "lipid_protein_sigma_nm": 0.89, "lipid_excl_nm": 1.0, "lipid_protein_eps_kT": _e,
                                    "morph_clamp": 0.0, "hold_tension_mN_m": 0.0, "hold_frames": [2000, 2000, 26000]},
                 f"sized residues, tail attraction {_e} kT: wetting and intrusion check (/tmp, never a row)")
# ---- E (2026-09-30, 05:20): sized residues (1.0 nm) AND tails attracted to them at 0.5 kT: the membrane wets the protein
# (first shell ~250 beads at rest, W5t) without entering the pore; clamped on the straight path, 0 and 8 mN/m.
_E = {**_B3c, "lipid_protein_sigma_nm": 0.89, "lipid_excl_nm": 1.0, "lipid_protein_eps_kT": 0.5}
for _i in range(9):
    _lam = _i / 8.0
    mech_version(f"E0{'abcdefghi'[_i]}", "mscs", None, {**_E, "morph_clamp": _lam, "hold_tension_mN_m": 0.0},
                 f"MscS in the Cooke membrane, residues sized and wetted (tails 0.5 kT), clamped at lambda {_lam} (straight path), 0 mN/m")
    mech_version(f"E8{'abcdefghi'[_i]}", "mscs", None, {**_E, "morph_clamp": _lam, "hold_tension_mN_m": 8.0},
                 f"MscS in the Cooke membrane, residues sized and wetted (tails 0.5 kT), clamped at lambda {_lam} (straight path), 8 mN/m")
RAN |= {f"E{t_}{c}" for t_ in "08" for c in "abcdefghi"}
VERSIONS["G2h"] = {"channel": "mscs", "parent": "G2g", "change": {}, "builder": "analysis",
                   "why": "G2 on the straight path in the Cooke membrane with sized residues the tails WET (0.5 kT), E0a-i against E8a-i (tools/gate_pmf.py)"}
RAN |= {"G2h"}
# ---- F (2026-09-30, 06:28): the wetting scan between D (repulsive: W 28 kT, dewets) and E (tails 0.5 kT: wets, W 4 +- 12 kT,
# lipids in the pockets and the pore, rest cost 461 kT): tails at 0.15 (Fa..) and 0.3 kT (Fb..), 5 lambdas, 0 and 8 mN/m.
# labels F<tension><letter>: letters p-t are tails 0.15 kT at lambda 0, 0.25, 0.5, 0.75, 1; u-y the same at 0.3 kT
for _e, _letters in ((0.15, "pqrst"), (0.3, "uvwxy")):
    for _lam, _c in zip((0.0, 0.25, 0.5, 0.75, 1.0), _letters):
        for _t in (0, 8):
            mech_version(f"F{_t}{_c}", "mscs", None,
                         {**_E, "lipid_protein_eps_kT": _e, "morph_clamp": _lam, "hold_tension_mN_m": float(_t)},
                         f"MscS in the Cooke membrane, sized residues, tails {_e} kT, clamped at lambda {_lam} (straight path), {_t} mN/m")
RAN |= {f"F{t_}{c}" for t_ in "08" for c in "pqrstuvwxy"}
VERSIONS["G2i"] = {"channel": "mscs", "parent": "G2h", "change": {}, "builder": "analysis",
                   "why": "G2 on the wetting scan, tails 0.15 kT (F0p-t against F8p-t; tools/gate_pmf.py)"}
VERSIONS["G2j"] = {"channel": "mscs", "parent": "G2h", "change": {}, "builder": "analysis",
                   "why": "G2 on the wetting scan, tails 0.3 kT (F0u-y against F8u-y; tools/gate_pmf.py)"}
RAN |= {"G2i", "G2j"}
# ---- H (2026-09-30, 07:33): the wetting window CONFIRMED or not -- tails 0.15 kT (G2i: W 21.9 +- 6.3 kT, dA_eff 11.3 +- 3.2
# nm^2 on 5 lambdas), now 9 lambdas and a second noise history (noise_replicate 1), 0 and 8 mN/m.
_H = {**_E, "lipid_protein_eps_kT": 0.15, "noise_replicate": 1}
for _i in range(9):
    _lam = _i / 8.0
    mech_version(f"H0{'abcdefghi'[_i]}", "mscs", None, {**_H, "morph_clamp": _lam, "hold_tension_mN_m": 0.0},
                 f"MscS in the Cooke membrane, sized residues, tails 0.15 kT, clamped at lambda {_lam} (straight path), 0 mN/m, replicate noise")
    mech_version(f"H8{'abcdefghi'[_i]}", "mscs", None, {**_H, "morph_clamp": _lam, "hold_tension_mN_m": 8.0},
                 f"MscS in the Cooke membrane, sized residues, tails 0.15 kT, clamped at lambda {_lam} (straight path), 8 mN/m, replicate noise")
RAN |= {f"H{t_}{c}" for t_ in "08" for c in "abcdefghi"}
VERSIONS["G2k"] = {"channel": "mscs", "parent": "G2i", "change": {}, "builder": "analysis",
                   "why": "G2 on H0a-i against H8a-i (tails 0.15 kT, nine lambdas, another noise history): the wetting window confirmed or not"}
RAN |= {"G2k"}
