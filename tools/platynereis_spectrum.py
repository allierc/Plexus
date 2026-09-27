#!/usr/bin/env python
"""Why this connectome latches instead of beating, and what would make it beat.

    python tools/platynereis_spectrum.py

R5's stroke is periodic, but its rhythm comes from `phase_clock` -- a clock declared in the spec.
The circuit only sets the stroke's AMPLITUDE. R7 asks for the rhythm to come from the NEURONS,
and before tuning anything it is worth asking whether this network can produce one at all.

IT CANNOT, AND THE REASON IS A THEOREM RATHER THAN A SETTING.

Every one of the 4,664 measured weights is positive -- they are synapse counts, and a count is
non-negative. For a linearised rate network `dv/dt = -a v + g W v`, the mode that goes unstable
first as `g` rises is the one with the largest real part, and it oscillates only if that
eigenvalue is complex. Perron-Frobenius says a non-negative matrix has its spectral radius as a
REAL, non-negative eigenvalue: the leading mode of an all-excitatory network is always real. So
raising the gain makes this network LATCH -- every cell saturating together against the tanh --
and no amount of tuning turns that into a beat.

Measured here: the leading eigenvalue is real at +0.900 (the radius the region was scaled to),
and the leading COMPLEX pair sits at 0.294 +- 0.072i, needing a gain of 3.40 to cross the leak.
By then the real mode is already unstable by a factor of 3 and owns the dynamics.

WHAT WOULD MAKE IT BEAT is inhibition, and the animal has it. Verasztó et al. (2017), eLife
6:e26000, describe the ciliomotor circuit this dataset contains: the single cholinergic MC cell
drives ciliary ARREST, the serotonergic Ser-h1 drives BEATING, and the two alternate. All five of
those cell types are in the compendium -- `MC`, `Ser-h1`, `Ser-tr1`, `Loop`, `INpreSer` -- but
the adjacency matrix carries only counts, so the signs have to be put back from the biology.
This file writes that signed connectome and reports what it does to the spectrum.

THE SIGNS ARE AN ASSUMPTION AND ARE LABELLED AS ONE. The only sign the paper pins down for this
dataset is MC's; everything else is a stated rule, not a measurement, and the report prints which
is which.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

REGION = "platynereis_larva_4117"

# WHAT IS KNOWN, AND FROM WHERE. Verasztó, C. et al. (2017). eLife 6:e26000, "Ciliomotor
# circuitry underlying whole-body coordination of ciliary activity in the Platynereis larva":
# the MC cell is cholinergic and its activity ARRESTS the prototroch; Ser-h1 is serotonergic and
# its activity drives BEATING. In a rate model the arrest driver is the inhibitory one.
KNOWN_INHIBITORY = {"MC"}
KNOWN_EXCITATORY = {"Ser-h1", "Ser-tr1"}


def load():
    from plexus.paths import graphs_data_path
    R = os.path.join(graphs_data_path(), "neural_regions", REGION)
    z = np.load(os.path.join(R, "neurons.npz"), allow_pickle=True)
    c = np.load(os.path.join(R, "connectome.npz"), allow_pickle=True)
    return R, z, c


def dense(ei, w, n):
    """W[post, pre] -- the matrix the membrane law multiplies a state vector by."""
    W = np.zeros((n, n), np.float64)
    W[ei[1], ei[0]] = w
    return W


def spectrum(W, label):
    ev = np.linalg.eigvals(W)
    r, im = np.real(ev), np.imag(ev)
    lead = int(np.argmax(r))
    cplx = np.abs(im) > 1e-9
    out = {"label": label, "radius": float(np.abs(ev).max()), "max_re": float(r.max()),
           "lead_is_complex": bool(cplx[lead]), "n_complex": int(cplx.sum())}
    if cplx.any():
        j = int(np.argmax(np.where(cplx, r, -np.inf)))
        out["lead_complex_re"] = float(r[j])
        out["lead_complex_im"] = float(abs(im[j]))
        out["gain_to_oscillate"] = float(1.0 / r[j]) if r[j] > 0 else float("inf")
        out["hz_at_onset"] = float(abs(im[j]) / (2 * np.pi))
    out["gain_to_latch"] = float(1.0 / r.max()) if r.max() > 0 else float("inf")
    return out


def report(s):
    print(f"  {s['label']}")
    print(f"    spectral radius            {s['radius']:.4f}")
    print(f"    largest real part          {s['max_re']:+.4f}   "
          f"({'COMPLEX' if s['lead_is_complex'] else 'REAL'} leading mode)")
    print(f"    gain that destabilises it  {s['gain_to_latch']:.3f}")
    if "lead_complex_re" in s:
        print(f"    leading complex pair       {s['lead_complex_re']:+.4f} "
              f"+- {s['lead_complex_im']:.4f}i  ->  {s['hz_at_onset']:.4f} Hz")
        print(f"    gain it needs to cross     {s['gain_to_oscillate']:.3f}")
        ratio = s["gain_to_oscillate"] / max(s["gain_to_latch"], 1e-12)
        print(f"    it is reached {ratio:.2f}x LATER than the latch"
              if ratio > 1 else
              f"    it is reached {1 / ratio:.2f}x BEFORE the latch  <- this network can beat")
    print()


def sign_edges(z, ei, frac_inhibitory: float, seed: int = 0):
    """A sign per EDGE, taken from its presynaptic cell. Dale's rule: a cell is E or I, not both.

    The rule, stated so it can be disagreed with:
      * `MC` is inhibitory -- the one sign the paper pins down for this dataset;
      * `Ser-h1` and `Ser-tr1` are excitatory, likewise;
      * of the remaining INTERNEURONS, a fraction `frac_inhibitory` is made inhibitory, chosen
        deterministically from a seeded permutation so the same fraction always picks the same
        cells and two runs of one number are one experiment;
      * every sensory neuron, motoneuron and non-neuronal cell stays excitatory.

    Interneurons and not the whole population, because that is where inhibition lives in an
    annelid nerve cord, and a sensory cell made inhibitory would be a stranger claim than the one
    being tested.
    """
    ct = np.array([str(t) for t in z["celltype"]])
    cn, ci = list(z["cell_class_names"]), np.asarray(z["cell_class_id"])
    inh = np.isin(ct, list(KNOWN_INHIBITORY))
    pool = np.where((ci == cn.index("Interneuron")) & ~inh & ~np.isin(ct, list(KNOWN_EXCITATORY)))[0]
    k = int(round(frac_inhibitory * len(pool)))
    if k:
        rng = np.random.default_rng(seed)
        inh[rng.permutation(pool)[:k]] = True
    return inh, int(inh.sum()), len(pool)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", default="0.0,0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--write", type=float, default=None,
                    help="write connectome_dale.npz at this inhibitory fraction")
    a = ap.parse_args()
    R, z, c = load()
    ei = np.asarray(c["edge_index"])
    w = np.asarray(c["weights"] if "weights" in c.files else c["contacts"], np.float64)
    n = int(z["xyz"].shape[0])

    print(f"{REGION}: {n:,} cells, {ei.shape[1]:,} edges, "
          f"{int((w < 0).sum())} of them negative\n")
    report(spectrum(dense(ei, w, n), "as measured -- every weight a synapse count, none negative"))

    print("  with Dale signs put back, by inhibitory fraction of the interneurons:\n")
    for f in [float(x) for x in a.fracs.split(",")]:
        inh, n_inh, n_pool = sign_edges(z, ei, f)
        ws = np.where(inh[ei[0]], -w, w)
        s = spectrum(dense(ei, ws, n), f"frac {f:.2f}  ({n_inh} inhibitory cells of {n_pool} "
                                       f"interneurons + the MC cell)")
        report(s)

    if a.write is not None:
        inh, n_inh, n_pool = sign_edges(z, ei, a.write)
        ws = np.where(inh[ei[0]], -w, w)
        out = {k: np.asarray(c[k]) for k in c.files}
        out["weights"] = ws.astype(np.float32)
        out["dale_inhibitory"] = inh
        out["meta_source"] = np.asarray(
            str(c["meta_source"]) + f"  [Dale signs added {a.write:.2f} of interneurons "
            f"inhibitory plus the MC cell, per Veraszto et al. 2017 eLife 6:e26000; "
            f"tools/platynereis_spectrum.py]")
        p = os.path.join(R, "connectome_dale.npz")
        np.savez(p, **out)
        print(f"  wrote {p}\n    {n_inh} inhibitory cells, {int((ws < 0).sum()):,} of "
              f"{len(ws):,} edges negative")
